//--------------------------------------------------------------------------------------
// File: graph.cpp
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "accelerator.h"
#include "graph.h"
#include "ptaskutils.h"
#include "dxtask.h"
#include "cutask.h"
#include "cltask.h"
#include "hosttask.h"
#include "windows.h"
#include "outputport.h"
#include "InputPort.h"
#include "Scheduler.h"
#include "PTaskRuntime.h"
#include "MemorySpace.h"
#include <assert.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include "shrperft.h"
#include "XMLWriter.h"
#include "XMLReader.h"
#include "Recorder.h"
#include <thrust/system_error.h>
#include <iostream>
#include "nvtxmacros.h"
#include "GraphProfiler.h"
#include "InitializerChannel.h"
#include "graphInputChannel.h"
#include "graphOutputChannel.h"
#include "internalChannel.h"
#include "multichannel.h"
#include "instrumenter.h"
#include "cuda.h"
#include "ThreadPool.h"
#include "Partitioner.h"
#include "ScopedPoolManager.h"
#include "cuaccelerator.h"
#include "signalprofiler.h"
using namespace std;
using namespace PTask::Runtime;

#ifdef PROFILE_GRAPHS
#define graph_profile_initializea(a)             { m_pGraphProfiler = new GraphProfiler(a); }
#define graph_profile_destroya(a)                { delete (a)->m_pGraphProfiler; (a)->m_pGraphProfiler=NULL; }
#define graph_profile_reporta(a)                 ((a)->m_pGraphProfiler)->Report(std::cerr)
#define graph_profile_initialize()               graph_profile_initializea(this)
#define graph_profile_destroy()                  graph_profile_destroya(this)
#define graph_profile_report()                   graph_profile_reporta(this)
#define record_task_thread_alive(x)              (x)->OnTaskThreadAlive()
#define record_task_thread_exit(x)               (x)->OnTaskThreadExit()
#define record_task_thread_block_run(x)          (x)->OnTaskThreadBlockRunningGraph()
#define record_task_thread_wake_run(x)           (x)->OnTaskThreadWakeRunningGraph()
#define record_task_thread_block_tasks(x)        (x)->OnTaskThreadBlockTasksAvailable()
#define record_task_thread_wake_tasks(x)         (x)->OnTaskThreadWakeTasksAvailable()
#define record_task_thread_dequeue_attempt(x)    (x)->OnTaskThreadDequeueAttempt()
#define record_task_thread_dequeue_complete(x,y) (x)->OnTaskThreadDequeueComplete(y)
#define record_inflight_dispatch_attempt(x)      (x)->OnTaskThreadDispatchAttempt()
#define record_dispatch_attempt_complete(x,y)    (x)->OnTaskThreadDispatchComplete(y)
#else 
#define graph_profile_initializea(a)             
#define graph_profile_destroya(a)                
#define graph_profile_reporta(x)                 
#define graph_profile_initialize()               
#define graph_profile_destroy()                  
#define graph_profile_report()                   
#define record_task_thread_alive(x)
#define record_task_thread_exit(x)
#define record_task_thread_block_run(x)          
#define record_task_thread_wake_run(x)           
#define record_task_thread_block_tasks(x)        
#define record_task_thread_wake_tasks(x)         
#define record_task_thread_dequeue_attempt(x)
#define record_task_thread_dequeue_complete(x,y) 
#define record_inflight_dispatch_attempt(x)      
#define record_dispatch_attempt_complete(x,y)   
#endif
#ifdef CUDA_SUPPORT
#include <cuda_runtime.h>
// #define CUDAFREE()
#define CUDAFREE()                      { int dev; cudaGetDevice(&dev); cudaFree(0); }
#define PRIME_GRAPH_RUNNER_THREAD()		CUDAFREE()
#else
#define PRIME_GRAPH_RUNNER_THREAD()
#endif

static const bool g_bUseDLLPartitioner = false;
#define HintsAny FALSE
#define HintsNo FALSE
#define HintsYes TRUE
#define ExplicitAny FALSE
#define ExplicitNo FALSE
#define ExplicitYes TRUE
#define DontCare 1
#define NonMandatory 2
#define Mandatory 3
typedef BOOL (*lpfeedbackfn)(const char * szMessage, ...);

#define RequirePartitionModeSettings(hints, hmodemand, expl, explmodemand) {                            \
    lpfeedbackfn lpfnErr = PTask::Runtime::HandleError;                                                 \
    lpfeedbackfn lpfnWarn = PTask::Runtime::MandatoryInform;                                            \
    if(((hmodemand) != DontCare) && m_bHasSchedulerPartitionHints != (hints)) {                         \
        lpfeedbackfn feedbackfn = (((hmodemand)==Mandatory) ? lpfnErr : lpfnWarn);                      \
        (*feedbackfn)((const char *)"XXX: %s::%s Partition hints %s for partition mode=%s\n",           \
                      __FILE__,                                                                         \
                      __FUNCTION__,                                                                     \
                      (m_bHasSchedulerPartitionHints?"present":"not present"),                          \
                      GraphPartitioningModeString(m_ePartitioningMode));                                \
       if(hmodemand) return; }                                                                          \
    if(((explmodemand)!=DontCare) && m_bExplicitPartition != (expl)) {                                  \
        lpfeedbackfn feedbackfn = (((explmodemand)==Mandatory) ? lpfnErr : lpfnWarn);                   \
        (*feedbackfn)((const char *)"XXX: %s::%s Explicit partition %s for partition mode=%s\n",        \
                      __FILE__,                                                                         \
                      __FUNCTION__,                                                                     \
                      (m_bExplicitPartition?"present":"not present"),                                   \
                      GraphPartitioningModeString(m_ePartitioningMode));                                \
       if(explmodemand) return; }}

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Compare tasks by effprio. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    struct compare_tasks_by_static_prio {     
        bool operator() (const Task * lhs, const Task * rhs) { 
            return ((Task*)lhs)->GetPriority() > 
                ((Task*)rhs)->GetPriority(); 
        } 
    };

    /// <summary>   The accelerator assignment counter. </summary>
    UINT     Graph::m_uiAccAssignmentCounter = 0;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates the graph. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   The new graph. </returns>
    ///-------------------------------------------------------------------------------------------------

    Graph * 
    Graph::CreateGraph(
        VOID
        )
    {
        return new Graph(PTask::Runtime::GetRuntimeTerminateEvent());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a graph. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <param name="lpszName"> [in,out] If non-null, the name. </param>
    ///
    /// <returns>   The new graph. </returns>
    ///-------------------------------------------------------------------------------------------------

    Graph * 
    Graph::CreateGraph(
        char * lpszName
        )
    {
        return new Graph(PTask::Runtime::GetRuntimeTerminateEvent(), lpszName);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    Graph::Graph(
        VOID
        )  : 
        Lockable("Graph")
    {
        Initialize(Runtime::GetRuntimeTerminateEvent(), NULL);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    Graph::Graph(
        HANDLE hRuntimeTerminateEvent
        )  : 
        Lockable("Graph")
    {
        Initialize(hRuntimeTerminateEvent, NULL);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="lpszName"> [in] non-null, the name. </param>
    ///-------------------------------------------------------------------------------------------------

    Graph::Graph(
        HANDLE hRuntimeTerminateEvent,
        char * lpszName
        ) : 
         Lockable(lpszName)
    {
        Initialize(hRuntimeTerminateEvent, lpszName);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets managed object. </summary>
    ///
    /// <remarks>   Crossbac, 7/10/2013. </remarks>
    ///
    /// <param name="parameter1">   The first parameter. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::SetManagedObject(
        VOID
        )
    {
        m_bManagedObject = TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes this object. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="lpszName"> [in] non-null, the name. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::Initialize(
        HANDLE hRuntimeTerminateEvent,
        char * lpszName
        )
    {
        m_ePartitioningMode = (GRAPHPARTITIONINGMODE)PTask::Runtime::GetDefaultGraphPartitioningMode();
        m_bHasSchedulerPartitionHints = FALSE;
        m_bUserManagedStrictAssignment = FALSE;
        m_bExplicitPartition = FALSE;
        m_pExplicitPartition = NULL;
        m_uiExplicitPartitionElems = 0;
        m_pScopedPoolManager = NULL;
        m_pGraphProfiler = NULL;
        m_bManagedObject = FALSE;
        m_bForceTeardown = FALSE;
        m_bDestroyComplete = FALSE;
        m_bForcedTeardownComplete = FALSE;
        m_bFinalized = FALSE;
        m_bEverRan = FALSE;
        m_bMutable = FALSE;
        m_eState = PTGS_INITIALIZING;
        m_bGraphHasCycles = FALSE;
        m_bLargeGraph = FALSE;
        m_uiOutstandingExecuteThreads = 0;
        m_uiThreadsAwaitingRunnableGraph = 0;
        m_luiTriggerControlSignals = DBCTLC_NONE;
        if(lpszName != NULL) { 
            size_t nNameLength = strlen(lpszName);
            m_lpszGraphName = new char[nNameLength+1];
            strcpy_s(m_lpszGraphName, nNameLength+1, lpszName);
        } else {
            size_t nNameLength = strlen("PTask.Graph.XXXXXXXXXXXXXXXX");
            m_lpszGraphName = new char[nNameLength+1];
            sprintf_s(m_lpszGraphName, nNameLength+1, "PTaskGraph.%d", ptaskutils::nextuid());
        }
        char lpszRunningEventName[MAX_GRAPH_EVENT_NAME];
        char lpszStopEventName[MAX_GRAPH_EVENT_NAME];
        char lpszProbeEventName[MAX_GRAPH_EVENT_NAME];
        char lpszTeardownEventName[MAX_GRAPH_EVENT_NAME];
        char lpszTeardownCompleteEventName[MAX_GRAPH_EVENT_NAME];
        char lpszGraphQuiescentEventName[MAX_GRAPH_EVENT_NAME];
        sprintf_s(lpszRunningEventName, MAX_GRAPH_EVENT_NAME, "%s.RunningEvent_%d", m_lpszGraphName, ptaskutils::nextuid());
        sprintf_s(lpszStopEventName, MAX_GRAPH_EVENT_NAME, "%s.StopEvent_%d", m_lpszGraphName, ptaskutils::nextuid());
        sprintf_s(lpszProbeEventName, MAX_GRAPH_EVENT_NAME, "%s.ProbeGraphEvent_%d", m_lpszGraphName, ptaskutils::nextuid());
        sprintf_s(lpszTeardownEventName, MAX_GRAPH_EVENT_NAME, "%s.TeardownEvent_%d", m_lpszGraphName, ptaskutils::nextuid());
        sprintf_s(lpszTeardownCompleteEventName, MAX_GRAPH_EVENT_NAME, "%s.TeardownCompleteEvent_%d", m_lpszGraphName, ptaskutils::nextuid());
        sprintf_s(lpszGraphQuiescentEventName, MAX_GRAPH_EVENT_NAME, "%s.QuiescentEvent_%d", m_lpszGraphName, ptaskutils::nextuid());
        m_hRuntimeTerminateEvent = hRuntimeTerminateEvent;
        m_hGraphRunningEvent = CreateEventA(NULL, TRUE, FALSE, lpszRunningEventName);
        m_hGraphStopEvent = CreateEventA(NULL, TRUE, FALSE, lpszStopEventName);
        m_hProbeGraphEvent = CreateEventA(NULL, TRUE, FALSE, lpszProbeEventName);
        m_hGraphTeardownEvent = CreateEventA(NULL, TRUE, FALSE, lpszTeardownEventName);
        m_hGraphTeardownComplete = CreateEventA(NULL, TRUE, FALSE, lpszTeardownCompleteEventName);
        m_hGraphQuiescentEvent = CreateEventA(NULL, TRUE, FALSE, lpszGraphQuiescentEventName);
        m_hMonitorProc = NULL;
        m_phReadyQueue = NULL;
        m_uiReadyQSurplus = 0;
		m_bMustPrimeGraphRunnerThreads = FALSE;
		m_uiNascentGraphRunnerThreads = 0;
        InitializeCriticalSection(&m_csReadyQ);
        InitializeCriticalSection(&m_csWaitingThreads);
        m_bCrossRuntimeSharingChecksRequired = FALSE;
        UINT uiCtrValue = InterlockedIncrement(&m_uiAccAssignmentCounter);
        m_bStrictAcceleratorAssignment = PTask::Runtime::GetGraphAssignmentPolicy() == GMP_ROUND_ROBIN;
        m_uiAffinitizedAccelerator = m_bStrictAcceleratorAssignment ? uiCtrValue : 0xFFFFFFFF;
        m_pStrictAffinityAccelerator = NULL;
        graph_profile_initialize();
        m_pPartitioner = NULL;
        Scheduler::NotifyGraphCreate(this);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deletes the ready queue signal (event objects--what we actually create here
    ///             depends on a the threading mode and other runtime settings). </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::DeleteReadyQueueSignals(
        UINT nTaskThreads
        )
    {
        if(m_phReadyQueue) {
            if(GetThreadPoolSignalPerThread()) {
                for(UINT ui=0; ui<nTaskThreads; ui++) {
                    if(m_phReadyQueue[ui] != NULL && m_phReadyQueue[ui] != INVALID_HANDLE_VALUE)
                        CloseHandle(m_phReadyQueue[ui]);
                }
            } else {
                if(m_phReadyQueue[0] != NULL && m_phReadyQueue[0] != INVALID_HANDLE_VALUE)
                    CloseHandle(m_phReadyQueue[0]);
            }
            delete [] m_phReadyQueue;
            m_phReadyQueue = NULL;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::DestroyGraph(
        Graph * pGraph,
        BOOL bGCSweep
        )
    {
        pGraph->Lock();
        if(!pGraph->m_bManagedObject || bGCSweep) {
            pGraph->Unlock();
            delete pGraph;
            return TRUE;
        } else {
            pGraph->Destroy();
        }
        pGraph->Unlock();
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   this is the work associated with a dtor, encapsulated so that we
    ///             can leave the object alive if the runtime exits before a GC deletes the
    ///             graph.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::Destroy(
        VOID
        )
    {
        if(m_bDestroyComplete || m_bForcedTeardownComplete)
            return;

        recordGraphDestroyStart();

        // if the graph is still alive in some respect, either because it has never had its stop method
        // called or its teardown method we need to force those method calls to ensure that the
        // scheduler does not continue to work with data structures related to this graph. 
        
        Lock();
        if(!m_bForceTeardown) {

            // normal graph destructor path.
            if(__IsRunning()) {

                if(!m_bManagedObject) {

                    // when graphs are wrapped by managed objects, we can't enforce life-cycle
                    // relationships among various PTask components because GC sweeps that delete
                    // wrapped objects may occur after the runtime has exited. So only complain 
                    // about this if this graph is not being managed by a GC. 
                    Runtime::Warning("XXXX: Graph::~Graph() while in the running state. Forcing stop...\n");
                }
                __Stop();     
            }
            if(!__IsTorndown() || __IsTearingDown()) {
                
                if(!m_bManagedObject) {

                    // when graphs are wrapped by managed objects, we can't enforce life-cycle
                    // relationships among various PTask components because GC sweeps that delete
                    // wrapped objects may occur after the runtime has exited. So only complain 
                    // about this if this graph is not being managed by a GC. 
                    PTask::Runtime::Warning("XXXX: Graph::~Graph() called on a live graph. Forcing teardown...\n");
                }
                Teardown(); 
            }

            // it is possible the user started a teardown for this
            // graph and the desructor was called before it completed. 
            // wait on the teardown signal just to be sure. 
            Unlock();
            WaitForSingleObject(m_hGraphTeardownComplete, INFINITE);
            Scheduler::NotifyGraphDestroy(this);
            graph_profile_report();
            Lock();
        }

        // record the number of task threads so we can clean up 
        // per thread-pool-member data structures later. We need to
        // take this value now because the call to teardown will empty the list
        UINT nTaskThreads = static_cast<UINT>(m_lpvhTaskThreads.size());

        // delete members whose values have no dependence
        // on the structure of the actual graph...
        
        if(m_lpszGraphName) {
            delete [] m_lpszGraphName;
            m_lpszGraphName = NULL;
        }
        if(m_hGraphRunningEvent) {
            CloseHandle(m_hGraphRunningEvent);
            m_hGraphRunningEvent = INVALID_HANDLE_VALUE;
        }
        if(m_hGraphStopEvent) {
            CloseHandle(m_hGraphStopEvent);
            m_hGraphStopEvent = INVALID_HANDLE_VALUE;
        }
        DeleteReadyQueueSignals(nTaskThreads);
        if(m_hProbeGraphEvent) {
            CloseHandle(m_hProbeGraphEvent);
            m_hProbeGraphEvent = INVALID_HANDLE_VALUE;
        }
        CloseHandle(m_hGraphTeardownComplete);
        CloseHandle(m_hGraphQuiescentEvent);
        m_hGraphQuiescentEvent = INVALID_HANDLE_VALUE;
        m_hGraphTeardownComplete = INVALID_HANDLE_VALUE;
        if(m_pExplicitPartition) delete [] m_pExplicitPartition;
        Unlock();
        DeleteCriticalSection(&m_csReadyQ);
        DeleteCriticalSection(&m_csWaitingThreads);
        graph_profile_destroy();
        if (m_pPartitioner) delete m_pPartitioner;
        if(m_pScopedPoolManager != NULL) {
            delete m_pScopedPoolManager;
            m_pScopedPoolManager = NULL;
        }
        recordGraphDestroyLatency();
        m_bDestroyComplete = TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Graph::~Graph() {
        Destroy();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this graph is alive. A graph is 'alive' until it has 
    ///             reached any of the teardown states. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   true if alive, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::IsAlive(
        VOID
        )
    {
        assert(LockIsHeld());
        return __IsAlive();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this graph is running. A graph is only running if it is
    ///             in the RUNNING state.</summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   true if running, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::IsRunning(
        VOID
        )
    {
        // assert(LockIsHeld());
        return __IsRunning();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this graph is runnable but not running. A graph is only running if it is
    ///             in the RUNNABLE state.</summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   true if running, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::IsStopped(
        VOID
        )
    {
        assert(LockIsHeld());
        return __IsStopped();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this graph is finalized. This graph is finalized if it
    ///             has called its "finalize" method and is in the runnble, running,
    ///             quiescing, or teardown related states.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   true if finalized, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::IsFinalized(
        VOID
        )
    {
        assert(LockIsHeld());
        return __IsFinalized();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this graph is torn down already. This graph is finalized if it
    ///             has called its Teardown method. Lock required for this method call.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   true if torn down, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::IsTorndown(
        VOID
        )
    {
        assert(LockIsHeld());
        return __IsTorndown();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this graph is still in a teardown operation. Lock required for this
    ///             method call.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   true if torn down, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::IsTearingDown(
        VOID
        )
    {
        assert(LockIsHeld());
        return __IsTearingDown();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this graph is alive. A graph is 'alive' until it has 
    ///             reached any of the teardown states. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   true if alive, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::__IsAlive(
        VOID
        )
    {
        return m_eState != PTGS_TEARINGDOWN &&
               m_eState != PTGS_TEARDOWNCOMPLETE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this graph is running. A graph is only running if it is
    ///             in the RUNNING state.</summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   true if running, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::__IsRunning(
        VOID
        )
    {
        return m_eState == PTGS_RUNNING;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this graph is runnable but not running. A graph is only running if it is
    ///             in the RUNNABLE state.</summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   true if running, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::__IsStopped(
        VOID
        )
    {
        return m_eState == PTGS_RUNNABLE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this graph is finalized. This graph is finalized if it
    ///             has called its "finalize" method and is in the runnble, running,
    ///             quiescing, or teardown related states. Private version, no lock required.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   true if finalized, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::__IsFinalized(
        VOID
        )
    {
        BOOL bFinalized =  m_eState == PTGS_RUNNABLE ||
                           m_eState == PTGS_RUNNING ||
                           m_eState == PTGS_QUIESCING ||
                           m_eState == PTGS_TEARINGDOWN ||
                           m_eState == PTGS_TEARDOWNCOMPLETE;
        if(!bFinalized)
            return m_bFinalized;
        assert(m_bFinalized);
        return bFinalized;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this graph is torn down already. This graph is finalized if it
    ///             has called its Teardown method. No lock required for this method call.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   true if torn down, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::__IsTorndown(
        VOID
        )
    {
        return m_eState == PTGS_TEARDOWNCOMPLETE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this graph is still in a teardown operation. no lock required for this
    ///             method call.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   true if torn down, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::__IsTearingDown(
        VOID
        )
    {
        return m_eState == PTGS_TEARINGDOWN;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a task. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pKernel">              [in,out] If non-null, the kernel. </param>
    /// <param name="uiInputPortCount">     Number of input ports. </param>
    /// <param name="pvInputPorts">         [in,out] If non-null, the pv input ports. </param>
    /// <param name="uiOutputPortCount">    Number of output ports. </param>
    /// <param name="pvOutputPorts">        [in,out] If non-null, the pv output ports. </param>
    /// <param name="lpszTaskName">         [in,out] If non-null, name of the task. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------
    #pragma warning(disable:4996)
    Task*
    Graph::AddTask(
        CompiledKernel *	pKernel,
        UINT				uiInputPortCount,
        Port **				pvInputPorts,
        UINT				uiOutputPortCount,
        Port **				pvOutputPorts,
        char *				lpszTaskName
        )
    {
        Task * pTask = NULL;
        if(NULL == (pTask = CreatePlatformSpecificTask(pKernel, lpszTaskName))) {
            return NULL;
        }

        Lock();
        assert(m_eState == PTGS_INITIALIZING);
        int nTrueInputPorts = 0;
        int nConstInputPorts = 0;
        pTask->SetGraph(this);
        for(UINT i=0; i<uiInputPortCount; i++) {

            // unfortunately, while our API design doesn't distinguish
            // between const and input ports for port numbers, the underlying
            // accelerator runtime does. (I.e. there can be a const input at index 0
            // *and* a true input at index 0. So we need to renumber the ports.
            Port * pPort = pvInputPorts[i];
            m_pPortMap[pPort] = pTask;
            if(pPort->GetPortType() == INPUT_PORT)
                pTask->BindPort(pvInputPorts[i], (PORTINDEX) nTrueInputPorts++);
            else if(pPort->GetPortType() == STICKY_PORT)
                pTask->BindPort(pvInputPorts[i], (PORTINDEX) nConstInputPorts++);
            else if(pPort->GetPortType() == META_PORT) 
                pTask->BindPort(pvInputPorts[i], (PORTINDEX) i);
            else if(pPort->GetPortType() == INITIALIZER_PORT) 
                pTask->BindPort(pvInputPorts[i], (PORTINDEX) nTrueInputPorts++);

            // Store the original index of each port, to facilitate reconstruction of
            // the input array for graph serialization/deserialization.
            pPort->SetOriginalIndex(i);
        }
        for(UINT i=0; i<uiOutputPortCount; i++) {
            pTask->BindPort(pvOutputPorts[i], (PORTINDEX) i);
            m_pPortMap[pvOutputPorts[i]] = pTask;

            // Store the original index of each port, to facilitate reconstruction of
            // the input array for graph serialization/deserialization.
            pvOutputPorts[i]->SetOriginalIndex(i);
        }
        string strTask(lpszTaskName);
        while(m_vTasks.find(strTask) != m_vTasks.end()) {
            char szUniquifier[8];
            strTask += _itoa(::GetTickCount() % 10, szUniquifier, 10); 
        }
        m_vTasks[strTask] = pTask;
        pTask->ResolveInOutPortBindings();
        pTask->ResolveMetaPortBindings();
        pTask->ResolveDependentPortRequirements();
        pTask->AddRef();
        m_vBackendFrameworks.insert(pTask->GetAcceleratorClass());
        Unlock();
        return pTask;
        #pragma warning(default:4996)
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a task by name </summary>
    ///
    /// <remarks>   jcurrey, 5/8/2013. </remarks>
    ///
    /// <param name="lpszTaskName">  [in] Name of the task. </param>
    ///
    /// <returns>   null if it fails, else the task. </returns>
    ///-------------------------------------------------------------------------------------------------

    Task *
    Graph::GetTask(char * lpszTaskName)
    {
        Task * pTask = nullptr;
        Lock();
        map<std::string, Task*>::iterator ti = m_vTasks.find(lpszTaskName);
        if(ti != m_vTasks.end()) 
            pTask = ti->second;
        Unlock();
        return pTask;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a super task. </summary>
    ///
    /// <param name="pKernel">              [in,out] If non-null, the kernel. </param>
    /// <param name="uiInputPortCount">     Number of input ports. </param>
    /// <param name="pvInputPorts">         [in,out] If non-null, the pv input ports. </param>
    /// <param name="uiOutputPortCount">    Number of output ports. </param>
    /// <param name="pvOutputPorts">        [in,out] If non-null, the pv output ports. </param>
    /// <param name="lpszTaskName">         [in,out] If non-null, name of the task. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------
    #pragma warning(disable:4996)
    Task*
    Graph::AddTask(
        UINT                uiKernelCount,
        CompiledKernel **	ppKernels,
        UINT				uiInputPortCount,
        Port **				pvInputPorts,
        UINT				uiOutputPortCount,
        Port **				pvOutputPorts,
        char *				lpszTaskName
        )
    {
        Task * pTask = NULL;
        if(NULL == (pTask = CreateSuperTask(ppKernels, uiKernelCount, lpszTaskName))) {
            return NULL;
        }

        Lock();
        assert(m_eState == PTGS_INITIALIZING);
        int nTrueInputPorts = 0;
        int nConstInputPorts = 0;
        for(UINT i=0; i<uiInputPortCount; i++) {
            // unfortunately, while our API design doesn't distinguish
            // between const and input ports for port numbers, the underlying
            // accelerator runtime does. (I.e. there can be a const input at index 0
            // *and* a true input at index 0. So we need to renumber the ports.
            Port * pPort = pvInputPorts[i];
            m_pPortMap[pPort] = pTask;
            if(pPort->GetPortType() == INPUT_PORT)
                pTask->BindPort(pvInputPorts[i], (PORTINDEX) nTrueInputPorts++);
            else if(pPort->GetPortType() == STICKY_PORT)
                pTask->BindPort(pvInputPorts[i], (PORTINDEX) nConstInputPorts++);
            else if(pPort->GetPortType() == META_PORT) 
                pTask->BindPort(pvInputPorts[i], (PORTINDEX) i);
            else if(pPort->GetPortType() == INITIALIZER_PORT) {
                pTask->BindPort(pvInputPorts[i], (PORTINDEX) i);
                nTrueInputPorts++;
            }
        }
        for(UINT i=0; i<uiOutputPortCount; i++) {
            pTask->BindPort(pvOutputPorts[i], (PORTINDEX) i);
            m_pPortMap[pvOutputPorts[i]] = pTask;
        }
        string strTask(lpszTaskName);
        while(m_vTasks.find(strTask) != m_vTasks.end()) {
            char szUniquifier[8];
            strTask += _itoa(::GetTickCount() % 10, szUniquifier, 10); 
        }
        m_vTasks[strTask] = pTask;
        pTask->ResolveInOutPortBindings();
        pTask->ResolveMetaPortBindings();
        pTask->ResolveDependentPortRequirements();
        pTask->AddRef();
        pTask->SetGraph(this);
        m_vBackendFrameworks.insert(pTask->GetAcceleratorClass());
        Unlock();
        return pTask;
        #pragma warning(default:4996)
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets backend frameworks. </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the backend frameworks. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::set<ACCELERATOR_CLASS> * 
    Graph::GetBackendFrameworks(
        VOID
        )
    {
        return &m_vBackendFrameworks;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets affinity for the entire graph to a single accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="uiAcceleratorId">  Identifier for the accelerator. </param>
    /// <param name="affinityType">     Type of the affinity. </param>
    ///
    /// <returns>   PTRESULT--use PTSUCCESS()/PTFAILED() macros </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT 
    Graph::SetAffinity(
        UINT uiAcceleratorId, 
        AFFINITYTYPE affinityType
        )
    {
        PTRESULT pr = PTASK_OK;
        Lock();
        m_uiAffinitizedAccelerator = uiAcceleratorId;
        m_bStrictAcceleratorAssignment = TRUE;
        m_bUserManagedStrictAssignment = TRUE;
        m_pStrictAffinityAccelerator = Scheduler::GetAcceleratorById(m_uiAffinitizedAccelerator);
        map<std::string, Task*>::iterator mi;
        for(mi=m_vTasks.begin(); mi!=m_vTasks.end(); mi++) {
            Task * pTask = mi->second;
            if(!PTSUCCESS(pr = Runtime::SetTaskAffinity(pTask, uiAcceleratorId, affinityType)))
                break;
        }
        Unlock();
        return pr;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets strict affinitized accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 3/27/2014. </remarks>
    ///
    /// <returns>   null if it fails, else the strict affinitized accelerator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Graph::GetStrictAffinitizedAccelerator(
        VOID
        )
    {
        if(m_pStrictAffinityAccelerator != NULL) {
            assert(m_bStrictAcceleratorAssignment);
            assert(m_bUserManagedStrictAssignment);
        }
        return m_pStrictAffinityAccelerator;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a super task. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pKernel">      [in,out] If non-null, the kernel. </param>
    /// <param name="lpszTaskName"> [in,out] If non-null, name of the task. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Task * 
    Graph::CreateSuperTask(
        CompiledKernel ** ppKernel,
        UINT uiKernelCount,
        char * lpszTaskName
        )
    {
        // TODO: fly in prototype code from branch on SVC-CROSSBAC-1 
        // that implements super task class!
        UNREFERENCED_PARAMETER(ppKernel);
        UNREFERENCED_PARAMETER(uiKernelCount);
        UNREFERENCED_PARAMETER(lpszTaskName);
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a platform specific task. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pKernel">      [in,out] If non-null, the kernel. </param>
    /// <param name="lpszTaskName"> [in,out] If non-null, name of the task. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Task * 
    Graph::CreatePlatformSpecificTask(
        CompiledKernel * pKernel,
        char * lpszTaskName
        )
    {
        assert(pKernel != NULL);
        const char * szFile = pKernel->GetSourceFile();
        ACCELERATOR_CLASS accTargetClass = ptaskutils::SelectAcceleratorClass(szFile);
        return CreatePlatformSpecificTask(pKernel, accTargetClass, lpszTaskName);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a platform specific task. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pKernel">      [in,out] If non-null, the kernel. </param>
    /// <param name="accClass">     The acc class. </param>
    /// <param name="lpszTaskName"> [in,out] If non-null, name of the task. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Task * 
    Graph::CreatePlatformSpecificTask(
        CompiledKernel * pKernel,
        ACCELERATOR_CLASS accClass,
        char * lpszTaskName
        )
    {
        switch(accClass) {
        case ACCELERATOR_CLASS_DIRECT_X: 
    #ifdef CUDA_SUPPORT
        case ACCELERATOR_CLASS_CUDA:
    #endif
    #ifdef OPENCL_SUPPORT
        case ACCELERATOR_CLASS_OPEN_CL:
    #endif
        case ACCELERATOR_CLASS_HOST:
            break; // these are supported
        case ACCELERATOR_CLASS_REFERENCE:
        case ACCELERATOR_CLASS_UNKNOWN:
        default:
            PTask::Runtime::MandatoryInform("%s::%s(%s, %s): attempt to create PTask for unsupported platform type\n",
                                            __FILE__,
                                            __FUNCTION__,
                                            lpszTaskName,
                                            AccClassString(accClass));
            return NULL;
        }

        Task * pTask = NULL;
        set<Accelerator*> vCapableAccelerators;
        Scheduler::FindEnabledCapableAccelerators(accClass, vCapableAccelerators);
        switch(accClass) {
        case ACCELERATOR_CLASS_DIRECT_X: 
            pTask = new DXTask(m_hRuntimeTerminateEvent, m_hGraphTeardownEvent, m_hGraphStopEvent, m_hGraphRunningEvent, pKernel); 
            break;
        case ACCELERATOR_CLASS_CUDA: 
#ifdef CUDA_SUPPORT
            pTask = new CUTask(m_hRuntimeTerminateEvent, m_hGraphTeardownEvent, m_hGraphStopEvent, m_hGraphRunningEvent, pKernel); 
#endif
            break;
        case ACCELERATOR_CLASS_OPEN_CL: 
#ifdef OPENCL_SUPPORT
            pTask = new CLTask(m_hRuntimeTerminateEvent, m_hGraphTeardownEvent, m_hGraphStopEvent, m_hGraphRunningEvent, pKernel); 
#endif
            break;
        case ACCELERATOR_CLASS_HOST: 
            pTask = new HostTask(m_hRuntimeTerminateEvent, m_hGraphTeardownEvent, m_hGraphStopEvent, m_hGraphRunningEvent, pKernel); 
            break;
        default:
            assert(false);
            return NULL;
        }

        if(lpszTaskName) pTask->SetTaskName(lpszTaskName);
        if(SUCCEEDED(pTask->Create(vCapableAccelerators, pKernel)))
            return pTask;
        delete pTask;
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Attempt to preallocate blocks on output channels
    ///             
    ///             Allocation of data-blocks and platform-specific buffers can be a signficant
    ///             latency expense at dispatch time. We can actually preallocate output datablocks
    ///             and create device-side buffers at graph construction time. For each node in the
    ///             graph, allocate data blocks on any output ports, and create device-specific
    ///             buffers for all accelerators capable of executing the node.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 2/15/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::AllocateBlockPools(
        VOID
        ) 
    {
        if(!PTask::Runtime::GetBlockPoolsEnabled()) {
            PTask::Runtime::Inform("%s::%s: PTask block pools disabled--skipping block pool allocation for: %s\n",
                                   __FILE__,
                                   __FUNCTION__,
                                   m_lpszGraphName);
            return;
        }

        if(PTask::Runtime::IsVerbose()) {
            std::map<UINT, UINT>::iterator mi;
            std::map<UINT, UINT> vAllocationPercentages;
            MemorySpace::GetAllocationPercentages(vAllocationPercentages);
            for(mi=vAllocationPercentages.begin(); mi!=vAllocationPercentages.end(); mi++) {
                if(mi->second > 50) {
                    PTask::Runtime::Inform("%s::%s: pre-blockpool-alloc: MEMSPACE %d -> %d%% full.\n",
                                           __FILE__,
                                           __FUNCTION__,
                                           mi->first,
                                           mi->second);
                }
            }
        }

        if(m_pScopedPoolManager != NULL) {
            MARKRANGEENTER(L"AllocateScopedBlockPools");
            if(!m_pScopedPoolManager->AllocatePools()) {
                PTask::Runtime::HandleError("Failed to allocate scoped block pools!\n");
            }
            MARKRANGEEXIT();
        }


        MARKRANGEENTER(L"AllocateBlockPools");
        Lock();

        // make sure we're in a state for which this is legal.
        // complain loudly if it is not legal, but try to 
        // gracefully handle it if this API is being abused.
        assert(m_eState == PTGS_INITIALIZING);
        if(m_eState != PTGS_INITIALIZING) {
            MARKRANGEEXIT();
            Unlock();
            return;
        }

        std::vector<Accelerator*> oDXPool;
        std::vector<Accelerator*> oCLPool;
        std::vector<Accelerator*> oPoolAcc;
        std::vector<Accelerator*> oPoolPL;
        std::set<Accelerator*> oLockSet;
        std::map<ACCELERATOR_CLASS, std::set<Accelerator*>>* ppAccMap;
        std::map<BlockPoolOwner*, std::vector<Accelerator*>*> oPoolOwners;
        ppAccMap = Scheduler::EnumerateBlockPoolAccelerators();
        std::map<ACCELERATOR_CLASS, std::set<Accelerator*>>& pAccMap = *ppAccMap;
        std::set<Accelerator*>::iterator aci;

        if(GetStrictAffinitizedAccelerator() != NULL) {

            // if the graph has a scoped affinity to a particular accelerator,
            // it suffices to find that accelerator and be done with it. 
            // otherwise, we need to go over all available accelerator objects.
            
            Accelerator * pStrictAcc = GetStrictAffinitizedAccelerator();
			oPoolAcc.push_back(pStrictAcc);
			oPoolPL.push_back(pStrictAcc);
			oLockSet.insert(pStrictAcc);

        } else {

            // this graph has no strict affinity. this means that creating a scoped
            // block pool must consider all the available accelerator objects to 
            // which tasks may be dynamically bound. typically, this means query for
            // all the accelerators of the given classes for which we support pooling.

            for(aci=pAccMap[ACCELERATOR_CLASS_CUDA].begin(); aci!=pAccMap[ACCELERATOR_CLASS_CUDA].end(); aci++) {
                Accelerator * pAccelerator = *aci;
                oPoolAcc.push_back(pAccelerator);
                oPoolPL.push_back(pAccelerator);
                oLockSet.insert(pAccelerator);
            }
        }

        Accelerator * pHostAcc = *(pAccMap[ACCELERATOR_CLASS_HOST].begin());
        oPoolPL.push_back(pHostAcc);
        oLockSet.insert(pHostAcc);
        oDXPool.assign(pAccMap[ACCELERATOR_CLASS_DIRECT_X].begin(), pAccMap[ACCELERATOR_CLASS_DIRECT_X].end());
        oCLPool.assign(pAccMap[ACCELERATOR_CLASS_OPEN_CL].begin(), pAccMap[ACCELERATOR_CLASS_OPEN_CL].end());
        std::map<ACCELERATOR_CLASS, std::vector<Accelerator*>*> pNonAsyncPools;
        pNonAsyncPools[ACCELERATOR_CLASS_DIRECT_X] = &oDXPool;
        pNonAsyncPools[ACCELERATOR_CLASS_OPEN_CL] = &oCLPool;

        map<std::string, Task*>::iterator ti;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {

            std::map<UINT, Port*>::iterator pi;
            Task * pTask = ti->second;
            assert(pTask != NULL);

            // construct a list of all accelerators that may be bound to dispatch resources associated with
            // this task. This includes all accelerators of the tasks accelerator class, as well as all
            // accelerators matching all classes on ports with dependent accelerator bindings. 

            std::map<UINT, Port*>* oPorts = pTask->GetOutputPortMap();
            for(pi=oPorts->begin(); pi!=oPorts->end(); pi++) {

                OutputPort * pOPort = reinterpret_cast<OutputPort*>(pi->second);
                // if(pOPort->HasDownstreamHostConsumer())
                //     DebugBreak();

                if(pOPort->IsBlockPoolCandidate()) {

                    ACCELERATOR_CLASS eTaskClass = pOPort->GetTask()->GetAcceleratorClass();
                    ACCELERATOR_CLASS ePortClass = pOPort->GetDependentAcceleratorClass(0);
                    if(pOPort->HasDependentAcceleratorBinding() || eTaskClass != ACCELERATOR_CLASS_HOST) {

                        if(eTaskClass == ACCELERATOR_CLASS_CUDA || ePortClass == ACCELERATOR_CLASS_CUDA) {
                            if(pOPort->HasDownstreamHostConsumer()) {                        
                                pOPort->SetRequestsPageLocked(TRUE);
                                oPoolOwners[pOPort] = &oPoolPL;
                            } else if(pOPort->GetTemplate()->HasInitialValue()) {
                                oPoolOwners[pOPort] = &oPoolPL;
                            } else {
                                oPoolOwners[pOPort] = &oPoolAcc;
                            }
                        } else {
                            vector<Accelerator*>* pAccList = pNonAsyncPools[eTaskClass];
                            oPoolOwners[pOPort] = pAccList;
                            vector<Accelerator*>::iterator pvi;
                            for(pvi=pAccList->begin(); pvi!=pAccList->end(); pvi++) {
                                oLockSet.insert(*pvi);
                            }
                        }
                    }
                }
            }            
        }

        std::map<std::string, Channel*>::iterator ci;
        for(ci=m_vChannels.begin(); ci!=m_vChannels.end(); ci++) {
            Channel * pChannel = ci->second;
            if(pChannel->GetType() == CT_INITIALIZER) {
                InitializerChannel * pIChannel = reinterpret_cast<InitializerChannel*>(pChannel);
                if(pIChannel->IsBlockPoolCandidate()) {
                    Port * pIPort = pIChannel->GetBoundPort(CE_DST);
                    Task * pTask = pIPort->GetTask();
                    ACCELERATOR_CLASS eTaskClass = pTask->GetAcceleratorClass();
                    ACCELERATOR_CLASS ePortClass = pIPort->GetDependentAcceleratorClass(0);
                    BOOL bDependentBinding = pIPort->HasDependentAcceleratorBinding();
                    if(bDependentBinding || eTaskClass != ACCELERATOR_CLASS_HOST) {                    
                        if(eTaskClass == ACCELERATOR_CLASS_CUDA || ePortClass == ACCELERATOR_CLASS_CUDA) {
                            if(pIChannel->IsPagelockedBlockPoolCandidate())
                                oPoolOwners[pIChannel] = &oPoolPL;
                            else
                                oPoolOwners[pIChannel] = &oPoolAcc;
                        } else {
                            vector<Accelerator*>* pAccList = pNonAsyncPools[eTaskClass];
                            oPoolOwners[pIChannel] = pAccList;
                            vector<Accelerator*>::iterator pvi;
                            for(pvi=pAccList->begin(); pvi!=pAccList->end(); pvi++) {
                                oLockSet.insert(*pvi);
                            }
                        }
                    }
                }

            } else if(pChannel->GetType() == CT_GRAPH_INPUT) {

                GraphInputChannel * pIChannel = reinterpret_cast<GraphInputChannel*>(pChannel);
                if(pIChannel->HasBlockPool()) {
                    Port * pBoundPort = pIChannel->GetBoundPort(CE_DST);
                    if(pBoundPort->HasDependentAcceleratorBinding()) {
                        oPoolOwners[pIChannel] = &oPoolPL;
                    } else {
                        ACCELERATOR_CLASS eTaskClass = pBoundPort->GetTask()->GetAcceleratorClass();
						BOOL bHasInitialValue = pIChannel->GetTemplate()->HasInitialValue();
                        if(eTaskClass == ACCELERATOR_CLASS_CUDA) {//  || eTaskClass == ACCELERATOR_CLASS_DIRECT_X) {
							if(!bHasInitialValue) {
								oPoolOwners[pIChannel] = &oPoolPL;
							} else {
								oPoolOwners[pIChannel] = &oPoolPL;
							}
                        } else {
                            vector<Accelerator*>* pAccList = pNonAsyncPools[eTaskClass];
                            oPoolOwners[pIChannel] = pAccList;
                            vector<Accelerator*>::iterator pvi;
                            for(pvi=pAccList->begin(); pvi!=pAccList->end(); pvi++) {
                                oLockSet.insert(*pvi);
                            }
                        }
                    }
                }
            }
        }

        std::set<Accelerator*>::iterator lsi;
        std::map<BlockPoolOwner*, std::vector<Accelerator*>*>::iterator pi;
        for(lsi=oLockSet.begin(); lsi!=oLockSet.end(); lsi++) 
            (*lsi)->Lock();
        for(pi=oPoolOwners.begin(); pi!=oPoolOwners.end(); pi++) {
            PTask::Runtime::Inform("Allocating Block pool for %s!\n", pi->first->GetPoolOwnerName());         
            pi->first->AllocateBlockPoolAsync(pi->second);            
            m_vBlockPoolOwners.push_back(pi->first);
            BlockPoolOwner::RegisterActivePoolOwner(this, pi->first);
        }
        for(pi=oPoolOwners.begin(); pi!=oPoolOwners.end(); pi++) {
            pi->first->FinalizeBlockPoolAsync();
        }
        for(lsi=oLockSet.begin(); lsi!=oLockSet.end(); lsi++) 
            (*lsi)->Unlock();

        Unlock();

        if(PTask::Runtime::IsVerbose()) {
            std::map<UINT, UINT>::iterator mi;
            std::map<UINT, UINT> vAllocationPercentages;
            MemorySpace::GetAllocationPercentages(vAllocationPercentages);
            for(mi=vAllocationPercentages.begin(); mi!=vAllocationPercentages.end(); mi++) {
                if(mi->second > 50) {
                    PTask::Runtime::Inform("%s::%s: POST-blockpool-alloc: MEMSPACE %d -> %d%% full.\n",
                                           __FILE__,
                                           __FUNCTION__,
                                           mi->first,
                                           mi->second);
                }
            }
        }

        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the block pools. </summary>
    ///
    /// <remarks>   crossbac, 6/18/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::ReleaseBlockPools(
        VOID
        )
    {
        Lock();
        if(m_pScopedPoolManager) 
            m_pScopedPoolManager->DestroyPools();
        std::vector<BlockPoolOwner*>::iterator vi;
        for(vi=m_vBlockPoolOwners.begin(); vi!=m_vBlockPoolOwners.end(); vi++) {
            BlockPoolOwner * pOwner = *vi;
            PTask::Runtime::Inform("BlockPool:%s [LWM,HWM,AV,OWN]=%d, %d, %d, %d\n", 
                                   pOwner->GetPoolOwnerName(),
                                   pOwner->GetLowWaterMark(),
                                   pOwner->GetHighWaterMark(),
                                   pOwner->GetAvailableBlockCount(),
                                   pOwner->GetOwnedBlockCount());
            pOwner->DestroyBlockPool();
            BlockPoolOwner::RetirePoolOwner(pOwner);
        }
        m_vBlockPoolOwners.clear();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Propagate channel consistency properties. If a channel has the
    ///             "wants most recent view" property set, and descriptor ports have been
    ///             bound to that channel or its destination port, then those associated
    ///             deferred channels also need to have the "want most recent view" property
    ///             to avoid having the descriptor blocks get out of sync with the blocks on the
    ///             channel being described. Since these channels are created automatically, 
    ///             the programmer should never really have to know they exist at all, and therefore
    ///             should not be held responsible for setting that property. Consequently, we do this
    ///             before the graph is put in the run state. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/14/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::PropagateChannelConsistencyProperties(
        VOID
        ) {
        Lock();

        // make sure we're in a state for which this is legal.
        // complain loudly if it is not legal, but try to 
        // gracefully handle it if this API is being abused.
        assert(m_eState == PTGS_INITIALIZING);
        if(m_eState != PTGS_INITIALIZING) {
            MARKRANGEEXIT();
            Unlock();
            return;
        }

        map<string, Channel*>::iterator mi;
        for(mi=m_vChannels.begin(); mi!=m_vChannels.end(); mi++) {
            Channel * pGraphChannel = mi->second;
            if(pGraphChannel->GetWantMostRecentView()) {
                Port * pDstPort = pGraphChannel->GetBoundPort(CE_DST);
                if(pDstPort != NULL) {
                    std::vector<DEFERREDCHANNELDESC*>* pDeferredChannels = pDstPort->GetDeferredChannels();
                    if(pDeferredChannels != NULL && pDeferredChannels->size() > 0) {
                        for(std::vector<DEFERREDCHANNELDESC*>::iterator si=pDeferredChannels->begin();
                            si != pDeferredChannels->end(); 
                            si++) {
                            DEFERREDCHANNELDESC * pDesc = *si;
                            Channel * pChannel = pDesc->pChannel;                    
                            if(!pChannel->GetWantMostRecentView()) {
                                pChannel->SetWantMostRecentView(TRUE);
                            }
                        }
                    }
                }                
            }
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind trigger. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <param name="pPort">            [in,out] If non-null, the port. </param>
    /// <param name="pChannel">         [in,out] If non-null, the channel. </param>
    /// <param name="luiTriggerSignal"> (optional) the trigger signal. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::BindTrigger(
        __in Port *        pPort, 
        __in Channel *     pChannel, 
        __in CONTROLSIGNAL luiTriggerSignal
        )
    {
        assert(LockIsHeld());
        assert(pPort != NULL);
        assert(pChannel != NULL);
        assert(HASSIGNAL(luiTriggerSignal));
        assert(ptaskutils::SignalCount(luiTriggerSignal == 1));
        m_luiTriggerControlSignals |= luiTriggerSignal;
        std::map<Port*, std::map<Channel*, CONTROLSIGNAL>*>::iterator mi;
        mi = m_pTriggeredChannelMap.find(pPort);
        std::map<Channel*, CONTROLSIGNAL> * pPortEntry;
        if(mi == m_pTriggeredChannelMap.end()) {
            pPortEntry = new std::map<Channel*, CONTROLSIGNAL>();
            m_pTriggeredChannelMap[pPort] = pPortEntry;
        } else {
            pPortEntry = mi->second;
        }
        assert(pPortEntry != NULL);
        assert(pPortEntry->find(pChannel) == pPortEntry->end());
        (*pPortEntry)[pChannel] = luiTriggerSignal;

        std::map<Port*, std::map<CONTROLSIGNAL, std::set<Channel*>*>*>::iterator ci;
        std::map<CONTROLSIGNAL, std::set<Channel*>*> * pControlEntry;
        ci = m_pTriggerSignalMap.find(pPort);
        if(ci == m_pTriggerSignalMap.end()) {
            pControlEntry = new std::map<CONTROLSIGNAL, std::set<Channel*>*>();
            std::set<Channel*>* pChannelSet = new std::set<Channel*>();
            pChannelSet->insert(pChannel);
            (*pControlEntry)[luiTriggerSignal] = pChannelSet;
            m_pTriggerSignalMap[pPort] = pControlEntry;
        } else {
            pControlEntry = ci->second;
            std::map<CONTROLSIGNAL, std::set<Channel*>*>::iterator ti;
            ti=pControlEntry->find(luiTriggerSignal);
            if(ti==pControlEntry->end()) {
                std::set<Channel*>* pChannelSet = new std::set<Channel*>();
                pChannelSet->insert(pChannel);
                (*pControlEntry)[luiTriggerSignal] = pChannelSet;
            } else {
                std::set<Channel*>* pChannelSet = ti->second;
                pChannelSet->insert(pChannel);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind trigger. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <param name="pChannel">         [in,out] If non-null, the channel. </param>
    /// <param name="pTask">            [in,out] (optional) the trigger signal. </param>
    /// <param name="luiTriggerSignal"> The lui trigger signal. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::BindTrigger(
        __in Channel *     pChannel, 
        __in Task *        pTask,
        __in CONTROLSIGNAL luiTriggerSignal
        )
    {
        assert(LockIsHeld());
        assert(pChannel != NULL);
        assert(HASSIGNAL(luiTriggerSignal));
        assert(ptaskutils::SignalCount(luiTriggerSignal == 1));
        m_luiTriggerControlSignals |= luiTriggerSignal;
        std::map<Channel*, std::map<CONTROLSIGNAL, std::set<Task*>*>*>::iterator mi;
        mi = m_pChannelTriggers.find(pChannel);
        std::map<CONTROLSIGNAL, std::set<Task*>*>* pChannelEntry;
        if(mi == m_pChannelTriggers.end()) {
            pChannelEntry = new std::map<CONTROLSIGNAL, std::set<Task*>*>();
            m_pChannelTriggers[pChannel] = pChannelEntry;
        } else {
            pChannelEntry = mi->second;
        }
        assert(pChannelEntry != NULL);
        std::set<Task*>* pSignalEntry = NULL;
        std::map<CONTROLSIGNAL, std::set<Task*>*>::iterator si;
        si = pChannelEntry->find(luiTriggerSignal);
        if(si == pChannelEntry->end()) {
            pSignalEntry = new std::set<Task*>();
            (*pChannelEntry)[luiTriggerSignal] = pSignalEntry;
        } else {
            pSignalEntry = si->second;
        }
        assert(pSignalEntry != NULL);
        pSignalEntry->insert(pTask);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an input channel. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pPort">            [in,out] If non-null, the port. </param>
    /// <param name="lpszChannelName">  (optional) [in] If non-null, name of the channel. </param>
    /// <param name="bSwitchChannel">   (optional) if true, make the channel switchable. </param>
    /// <param name="pTriggerPort">     [in,out] If non-null, the trigger port. </param>
    /// <param name="luiTriggerSignal"> The trigger signal. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    GraphInputChannel * 
    Graph::AddInputChannel(
        __in Port *        pPort,
        __in char *        lpszChannelName,
        __in BOOL          bSwitchChannel,
        __in Port *        pTriggerPort,
        __in CONTROLSIGNAL luiTriggerSignal
        ) 
    {
        Lock();

        // make sure we're in a state for which this is legal.
        // complain loudly if it is not legal, but try to 
        // gracefully handle it if this API is being abused.
        assert(m_eState == PTGS_INITIALIZING);
        if(m_eState != PTGS_INITIALIZING) {
            Unlock();
            return NULL;
        }

        GraphInputChannel * pChannel = NULL;
        if(lpszChannelName != NULL && m_vChannels.find(std::string(lpszChannelName)) != m_vChannels.end()) {
            assert(false && "channel names should be unique per graph!");
            PTask::Runtime::HandleError("%s::%s(%s): duplicate channel name!\n", 
                                        __FILE__,
                                        __FUNCTION__,
                                        lpszChannelName);
            Unlock();
            return NULL;
        } else {
            BOOL bChannelPool = pPort->HasUpstreamChannelPool();
            pChannel = new GraphInputChannel(this,
                                             pPort->GetTemplate(), 
                                             m_hRuntimeTerminateEvent, 
                                             m_hGraphTeardownEvent, 
                                             m_hGraphStopEvent, 
                                             lpszChannelName, 
                                             bChannelPool);
            if(bSwitchChannel) {
                pPort->BindControlChannel(pChannel);
            } else {
                pPort->BindChannel(pChannel);
            }
            pChannel->BindPort(pPort, CE_DST);
            m_vChannels[pChannel->GetName()] = pChannel;
            m_pChannelDstMap[pChannel] = pPort;
            if(pTriggerPort != NULL) {
                CTLSIGITERATOR iter(luiTriggerSignal);
                for(CONTROLSIGNAL luiSig=iter.begin();
                    luiSig != iter.end(); 
                    luiSig = ++iter) {
                    BindTrigger(pTriggerPort, pChannel, luiSig);
                }
            }
            pChannel->AddRef();
        }
        Unlock();
        return pChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an initializer channel. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pPort">            [in,out] If non-null, the port. </param>
    /// <param name="lpszChannelName">  (optional) [in] If non-null, name of the channel. </param>
    /// <param name="bSwitchChannel">   (optional) if true, make the channel switchable. </param>
    /// <param name="pTriggerPort">     [in,out] If non-null, the trigger port. </param>
    /// <param name="luiTriggerSignal"> The trigger signal. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    InitializerChannel * 
    Graph::AddInitializerChannel(
        __in Port *        pPort,
        __in char *        lpszChannelName,
        __in BOOL          bSwitchChannel,
        __in Port *        pTriggerPort,
        __in CONTROLSIGNAL luiTriggerSignal
        ) 
    {
        Lock();
        InitializerChannel * pChannel = NULL;
        if(lpszChannelName != NULL && m_vChannels.find(std::string(lpszChannelName)) != m_vChannels.end()) {
            assert(false && "channel names should be unique per graph!");
            PTask::Runtime::HandleError("%s::%s(%s): duplicate channel name!\n", 
                                        __FILE__,
                                        __FUNCTION__,
                                        lpszChannelName);
            Unlock();
            return NULL;
        } else {

            pChannel = new InitializerChannel(this,
                                              pPort->GetTemplate(), 
                                              m_hRuntimeTerminateEvent,
                                              m_hGraphTeardownEvent,
                                              m_hGraphStopEvent,
                                              lpszChannelName,
                                              TRUE);

            if(bSwitchChannel) {
                pPort->BindControlChannel(pChannel);
            } else {
                pPort->BindChannel(pChannel);
            }
            pChannel->BindPort(pPort, CE_DST);
            m_vChannels[pChannel->GetName()] = pChannel;
            m_pChannelDstMap[pChannel] = pPort;
            if(pTriggerPort != NULL) {
                CTLSIGITERATOR iter(luiTriggerSignal);
                for(CONTROLSIGNAL luiSig=iter.begin();
                    luiSig != iter.end(); 
                    luiSig = ++iter) {
                    BindTrigger(pTriggerPort, pChannel, luiSig);
                }
            }
            pChannel->AddRef();
        }
        Unlock();
        return pChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an output channel to to the graph. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pPort">            [in] non-null, the port. </param>
    /// <param name="lpszChannelName">  (optional) [in] If non-null, name of the channel. </param>
    /// <param name="pTriggerTask">     [in,out] (optional) the trigger channel. </param>
    /// <param name="luiTriggerSignal"> (optional) the trigger signal. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    GraphOutputChannel * 
    Graph::AddOutputChannel(
        __in Port *        pPort, 
        __in char *        lpszChannelName,
        __in Task *        pTriggerTask,
        __in CONTROLSIGNAL luiTriggerSignal
        )
    {
        Lock();
        GraphOutputChannel * pChannel = NULL;
        if(lpszChannelName != NULL && m_vChannels.find(lpszChannelName) != m_vChannels.end()) {
            assert(false && "unique channel names required per graph!");
            PTask::Runtime::HandleError("%s::%s(%s) duplicate channel name!\n", 
                                        __FILE__,
                                        __FUNCTION__,
                                        lpszChannelName);
            Unlock();
            return NULL;
        } else {

            pChannel = new GraphOutputChannel(this,
                                              pPort->GetTemplate(), 
                                              m_hRuntimeTerminateEvent,
                                              m_hGraphTeardownEvent,
                                              m_hGraphStopEvent, 
                                              lpszChannelName,
                                              TRUE);

            pPort->BindChannel(pChannel);
            pChannel->BindPort(pPort, CE_SRC);
            BOOL bTriggerChannel = pTriggerTask != NULL;
            if(bTriggerChannel) {
                pChannel->SetTriggerChannel(this, TRUE);
                CTLSIGITERATOR iter(luiTriggerSignal);
                for(CONTROLSIGNAL luiSig=iter.begin();
                    luiSig != iter.end(); 
                    luiSig = ++iter) {
                    BindTrigger(pChannel, pTriggerTask, luiSig);
                }
            }
            m_vChannels[pChannel->GetName()] = pChannel;
            m_pChannelSrcMap[pChannel] = pPort;
            pChannel->AddRef();
        }
        Unlock();
        return pChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an internal channel. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pSrcPort">         [in,out] If non-null, source port. </param>
    /// <param name="pDstPort">         [in,out] If non-null, destination port. </param>
    /// <param name="lpszChannelName">  (optional) [in] If non-null, name of the channel. </param>
    /// <param name="bSwitchChannel">   (optional) if true, make the channel switchable. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    InternalChannel * 
    Graph::AddInternalChannel(
        __in Port * pSrcPort, 
        __in Port * pDstPort,
        __in char * lpszChannelName,
        __in BOOL   bSwitchChannel
        ) 
    {
        Lock();
        InternalChannel * pChannel = NULL; 
        if(lpszChannelName != NULL && m_vChannels.find(lpszChannelName) != m_vChannels.end()) {
            assert(false && "unique channel names required per graph!");
            PTask::Runtime::HandleError("%s::%s(%s) duplicate channel name!\n", 
                                        __FILE__,
                                        __FUNCTION__,
                                        lpszChannelName);
            Unlock();
            return NULL;
        } else {

            pChannel = new InternalChannel(this,
                                           pSrcPort->GetTemplate(), 
                                           m_hRuntimeTerminateEvent, 
                                           m_hGraphTeardownEvent,
                                           m_hGraphStopEvent, 
                                           lpszChannelName,
                                           TRUE);

            pSrcPort->BindChannel(pChannel);
            pChannel->BindPort(pSrcPort, CE_SRC);
            if(bSwitchChannel) {
                pDstPort->BindControlChannel(pChannel);
            } else {
                pDstPort->BindChannel(pChannel);
            }
            pChannel->BindPort(pDstPort, CE_DST);
            m_vChannels[pChannel->GetName()] = pChannel;
            m_pChannelSrcMap[pChannel] = pSrcPort;
            m_pChannelDstMap[pChannel] = pDstPort;
            pChannel->AddRef();
            Task * pSrcTask = pSrcPort->GetTask();
            Task * pDstTask = pDstPort->GetTask();
            if(pSrcTask != NULL && pSrcTask == pDstTask) {
                m_bGraphHasCycles = TRUE;
            }
        }
        Unlock();
        return pChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an input channel. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="ppDst">            [in,out] If non-null, the port. </param>
    /// <param name="nDestPorts">       Destination ports. </param>
    /// <param name="lpszChannelName">  (optional) [in] If non-null, name of the channel. </param>
    /// <param name="bSwitchChannel">   (optional) if true, make the channel switchable. </param>
    /// <param name="pTriggerPort">     [in,out] If non-null, the trigger port. </param>
    /// <param name="luiTriggerSignal"> The lui trigger signal. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    MultiChannel * 
    Graph::AddMultiChannel(
        __in Port **       ppDst,
        __in int           nDestPorts,
        __in char *        lpszChannelName,
        __in BOOL          bSwitchChannel,
        __in Port *        pTriggerPort,
        __in CONTROLSIGNAL luiTriggerSignal
        ) 
    {
        Lock();
        MultiChannel * pChannel = NULL;
        if(lpszChannelName != NULL && m_vChannels.find(std::string(lpszChannelName)) != m_vChannels.end()) {
            assert(false && "unique channel names required per graph!");
            PTask::Runtime::HandleError("%s::%s(%s) duplicate channel name!\n", 
                                        __FILE__,
                                        __FUNCTION__,
                                        lpszChannelName);
            Unlock();
            return NULL;
        } else {

            pChannel = new MultiChannel(this,
                                        ppDst[0]->GetTemplate(), 
                                        m_hRuntimeTerminateEvent, 
                                        m_hGraphTeardownEvent,
                                        m_hGraphStopEvent, 
                                        lpszChannelName,
                                        TRUE);

            for(int i=0; i<nDestPorts; i++) {
                Port * pPort = ppDst[i];
                GraphInputChannel * pIChannel = new GraphInputChannel(this,
                                                                      pPort->GetTemplate(), 
                                                                      m_hRuntimeTerminateEvent, 
                                                                      m_hGraphTeardownEvent,
                                                                      m_hGraphStopEvent, 
                                                                      lpszChannelName,
                                                                      FALSE);

                if(bSwitchChannel) {
                    pPort->BindControlChannel(pIChannel);
                } else {
                    pPort->BindChannel(pIChannel);
                }
                pIChannel->BindPort(pPort, CE_DST);
                if(pTriggerPort != NULL) 
                    BindTrigger(pTriggerPort, pIChannel, luiTriggerSignal);
                pIChannel->AddRef();
                m_vChannels[pIChannel->GetName()] = pChannel;
                m_pChannelDstMap[pIChannel] = pPort;
                pChannel->CoalesceChannel(pIChannel);
            }
            m_vChannels[pChannel->GetName()] = pChannel;
            m_pChannelDstMap[pChannel] = NULL;
            pChannel->AddRef();
        }
        Unlock();
        return pChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the global trigger operation. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    /// <param name="luiCode">  The code. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::ExecuteTrigger(
        __in Port *         pPort, 
        __in CONTROLSIGNAL  luiCode
        )
    {
        Lock();
        assert(ptaskutils::SignalCount(luiCode) == 1);
        std::map<Port*, std::map<CONTROLSIGNAL, std::set<Channel*>*>*>::iterator ii;
        ii=m_pTriggerSignalMap.find(pPort);
        assert(ii!=m_pTriggerSignalMap.end());
        if(ii!=m_pTriggerSignalMap.end()) {
            std::map<CONTROLSIGNAL, std::set<Channel*>*>::iterator si;
            std::map<CONTROLSIGNAL, std::set<Channel*>*>* pTMap = ii->second;
            si=pTMap->find(luiCode);
            // assert(si!=pTMap->end()); <-- this is ok. blocks can have control we don't care about!
            if(si!=pTMap->end()) {
                std::set<Channel*>* pSet = si->second;
                for(std::set<Channel*>::iterator ci=pSet->begin(); ci!=pSet->end(); ci++) {
                    Channel * pChannel = *ci;
                    pChannel->PushInitializer();
                }
            }
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the global trigger operation. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <param name="pChannel"> [in,out] If non-null, the port. </param>
    /// <param name="luiCode">  The code. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::ExecuteTrigger(
        __in Channel * pChannel, 
        __in CONTROLSIGNAL luiCode
        )
    {
        Lock();
        assert(ptaskutils::SignalCount(luiCode) == 1);
        assert(HASSIGNAL(luiCode));
        if(HASSIGNAL(luiCode)) {
            std::map<Channel*, std::map<CONTROLSIGNAL, std::set<Task*>*>*>::iterator li;
            li = m_pChannelTriggers.find(pChannel);
            assert(li != m_pChannelTriggers.end());
            if(li != m_pChannelTriggers.end()) {
                std::map<CONTROLSIGNAL, std::set<Task*>*>* pSignalMap = li->second;
                std::map<CONTROLSIGNAL, std::set<Task*>*>::iterator si;
                si=pSignalMap->find(luiCode);
                // assert(si!=pTMap->end()); <-- this is ok. blocks can have control we don't care about!
                if(si!=pSignalMap->end()) {
                    std::set<Task*>* pSet = si->second;
                    for(std::set<Task*>::iterator ci=pSet->begin(); ci!=pSet->end(); ci++) {
                        Task * pTask = *ci;
                        pTask->ReleaseStickyBlocks();
                    }
                }
            }
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the global trigger operation. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <param name="pChannel"> [in,out] If non-null, the port. </param>
    /// <param name="luiCode">  The code. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::ExecuteTriggers(
        __in Channel * pChannel, 
        __in CONTROLSIGNAL luiCode
        )
    {
        assert(HASSIGNAL(luiCode));
        if(HASSIGNAL(luiCode)) {
            CTLSIGITERATOR iter(luiCode);
            CONTROLSIGNAL luiCurSignal = iter.begin();
            while(luiCurSignal != iter.end()) {
                //PTask::Runtime::Inform("%s::%s: Executing channel trigger for channel %s, code=%d\n", 
                //                       __FILE__,
                //                       __FUNCTION__,
                //                       pChannel->GetName(), 
                //                       (UINT)luiCurSignal);
                ExecuteTrigger(pChannel, luiCurSignal);
                luiCurSignal = ++iter;
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the global trigger operation. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    /// <param name="luiCod e">  The code. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::ExecuteTriggers(
        __in Port *        pPort, 
        __in CONTROLSIGNAL luiCode
        )
    {
        assert(HASSIGNAL(luiCode));
        if(HASSIGNAL(luiCode)) {
            CTLSIGITERATOR iter(luiCode);
            CONTROLSIGNAL luiCurSignal = iter.begin();
            while(luiCurSignal != iter.end()) {
                ExecuteTrigger(pPort, luiCurSignal);
                luiCurSignal = ++iter;
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind descriptor port. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pDescribedPort">   [in,out] If non-null, the described port. </param>
    /// <param name="pDescriberPort">   [in,out] If non-null, the describer port. </param>
    /// <param name="func">             (optional) the func. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::BindDescriptorPort(
        Port * pDescribedPort, 
        Port * pDescriberPort,
        DESCRIPTORFUNC func
        )
    {
        static int descriptorPortChannelCount = 0;

        Lock();
        assert(pDescribedPort != NULL);
        assert(pDescriberPort != NULL);
        PORTTYPE pType = pDescriberPort->GetPortType();
        if(pType == META_PORT || pType == INPUT_PORT || pType == STICKY_PORT) {
            const size_t szBufSize = 256;
            char szChannelName[szBufSize];
            sprintf_s(szChannelName, szBufSize, "BindDescriptorPort_Channel_%d", descriptorPortChannelCount++);
            GraphInputChannel * pChannel = AddInputChannel(pDescriberPort, szChannelName);
            pDescriberPort->BindDescriptorPort(pDescribedPort, func);
            pDescribedPort->BindDeferredChannel(pChannel, func);
        } else if(pType == OUTPUT_PORT) {
            // when an output port is a descriptor port, that means that
            // it provides the meta-data or template data facet for the
            // block leaving the described port on output. This complicates
            // the process of allocating output blocks, since we need to know
            // all the allocation sizes before we can actually allocate a block. 
            // -------------------
            // the binding semantics are backwards from the input port version.
            pDescribedPort->BindDescriptorPort(pDescriberPort, func);
        } else {
            assert(FALSE);
        }
        RECORDACTION(BindDescriptorPort, pDescribedPort, pDescriberPort, func);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind control port. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pController">      [in,out] If non-null, the controller. </param>
    /// <param name="pGatedPort">       [in,out] If non-null, the gated port. </param>
    /// <param name="bInitiallyOpen">   (optional) the initially open. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::BindControlPort(
        Port * pController,
        Port * pGatedPort,
        BOOL bInitiallyOpen
        )
    {
        Lock();
        // currently, only output ports can be gated.
        assert(pGatedPort->GetPortType() == OUTPUT_PORT);
        assert(pController->GetPortType() != INITIALIZER_PORT);
        pController->AddGatedPort(pGatedPort);
        ((OutputPort*)pGatedPort)->SetControlPort(pController, bInitiallyOpen);
        RECORDACTION(BindControlPort, pController, pGatedPort, bInitiallyOpen);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Make the graph mutable. Can fail if PTask::Runtime::*GraphMutabilityMode is
    ///             turned off, which it is by default. A graph is 'mutable' if previously configured
    ///             structures can be changed. (In an unmutable graph, structures are 'write-once').
    ///             For example, in a mutable graph, port bindings can be changed, which in an
    ///             unmutable graph, once a port is bound to an object such as a channel, it cannot
    ///             be unbound and bound to some other channel later. Currently, respect for the
    ///             mutability of substructures in a graph is somewhat spotty, as it has only been
    ///             tested to deal with mutable control propagation bindings.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/8/2014. </remarks>
    ///
    /// <param name="bMutable"> (Optional) true to make the graph mutable. </param>
    ///
    /// <returns>   true if the mode change succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Graph::SetMutable(
        __in BOOL bMutable
        )
    {
        Lock();        
        BOOL bModeChange = (bMutable == m_bMutable);
        BOOL bModeChangeOK = bModeChange && (!bMutable || PTask::Runtime::GetGraphMutabilityMode());
        BOOL bSuccess = !bModeChange || bModeChangeOK;
        if(bSuccess) m_bMutable = bMutable;        
        if(!bSuccess) {
            PTask::Runtime::Inform("%s::%s(%s) failed PTask::Runtime::*GraphMutabilityMode() not enabled!\n",
                                   __FILE__,
                                   __FUNCTION__,
                                   (bMutable?"TRUE":"FALSE"));
        }
        Unlock();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this graph is mutable. A graph is 'mutable' if previously configured
    ///             structures can be changed. (In an unmutable graph, structures are 'write-once').
    ///             For example, in a mutable graph, port bindings can be changed, which in an
    ///             unmutable graph, once a port is bound to an object such as a channel, it cannot
    ///             be unbound and bound to some other channel later. Currently, respect for the
    ///             mutability of substructures in a graph is somewhat spotty, as it has only been
    ///             tested to deal with mutable control propagation bindings.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/8/2014. </remarks>
    ///
    /// <returns>   true if mutable, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Graph::IsMutable(
        VOID
        )
    {
        return m_bMutable;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind control propagation port. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pInputPort">   [in,out] If non-null, the input port. </param>
    /// <param name="pOutputPort">  [in,out] If non-null, the output port. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::BindControlPropagationPort(
        Port * pInputPort,
        Port * pOutputPort
        )
    {
        Lock();

        assert(pOutputPort->GetPortType() == OUTPUT_PORT);
        assert(pInputPort->GetPortType() != INITIALIZER_PORT);
        Port * pCurrentSource = pOutputPort->GetControlPropagationSource();
        BOOL bConflictingBindingExists = (pCurrentSource != NULL && pCurrentSource != pInputPort);
        BOOL bBindingSuccess = !bConflictingBindingExists || m_bMutable;

        if(bConflictingBindingExists) {

            // if there is a conflicting binding already for the control propagation source of the given
            // output port, we handle it according to whether this graph is mutable or not. (Graphs are
            // 'write-once', or immutable by default). If the graph is mutable, then its ok to over-write
            // an existing binding. Otherwise, we ignore the call and complain. 
            
            PTask::Runtime::MandatoryInform("WARNING: %s::%s(%d): %s control propagation binding on %s:\n"
                                            "   %s -> %s %s invalidate current binding:\n"
                                            "   %s -> %s\n",
                                            __FILE__,
                                            __FUNCTION__,
                                            __LINE__,
                                            (bBindingSuccess?"overwriting":"skipping"),
                                            pInputPort->GetTask(),
                                            pInputPort->GetVariableBinding(),
                                            pOutputPort->GetVariableBinding(),
                                            (bBindingSuccess?"will":"would"),
                                            pCurrentSource->GetVariableBinding(),
                                            pOutputPort->GetVariableBinding());
        } 

        if(bBindingSuccess) {

            pInputPort->AddControlPropagationPort(pOutputPort);
            pOutputPort->SetControlPropagationSource(pInputPort);
            RECORDACTION2P(BindControlPropagationPort, pInputPort, pOutputPort);
        }

        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind iteration ports. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pInputPort">   [in,out] If non-null, the input port. </param>
    /// <param name="pOutputPort">  [in,out] If non-null, the output port. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::BindIterationScope(
        Port * pMetaPort, 
        Port * pScopedPort
        )
    {
        Lock();
        assert(pMetaPort->GetPortType() == META_PORT);
        assert(pScopedPort->GetPortType() == INPUT_PORT);
        pMetaPort->BindIterationTarget(pScopedPort);
        pScopedPort->SetIterationSource(pMetaPort);
        RECORDACTION2P(BindIterationScope, pMetaPort, pScopedPort);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind control propagation port. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pInputPort">           [in,out] If non-null, the input port. </param>
    /// <param name="pControlledChannel">   [in,out] If non-null, the channel. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::BindControlPropagationChannel(
        Port * pInputPort, 
        Channel * pControlledChannel
        )
    {
        Lock();
        assert(pInputPort->GetPortType() != INITIALIZER_PORT);
        pInputPort->AddControlPropagationChannel(pControlledChannel);
        pControlledChannel->SetControlPropagationSource(pInputPort);
        RECORDACTION2P(BindControlPropagationChannel,pInputPort, pControlledChannel);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pops the ready q. </summary>
    ///
    /// <remarks>   Crossbac, 1/29/2013. </remarks>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Task * 
    Graph::PopReadyQ(
        VOID
        )
    {
        // get a task on which to attempt a dispatch.
        // If we are in single-thread mode, the readyq 
        // event must be reset. In multi-thread mode, the
        // events are created per-thread, and are automatically
        // reset
        assert(m_bUseReadyQ);
        if(!m_bUseReadyQ) return NULL;
        Task * pExecuteTask = NULL;
        EnterCriticalSection(&m_csReadyQ);
        BOOL bTeardown = (m_bForceTeardown || m_eState == PTGS_TEARDOWNCOMPLETE || m_eState == PTGS_TEARINGDOWN);
        if(m_vReadyQ.size() && !bTeardown) {
            if(PTask::Runtime::g_bSortThreadPoolQueues) {
                std::sort(m_vReadyQ.begin(), m_vReadyQ.end(), compare_tasks_by_static_prio());
            }
            pExecuteTask = m_vReadyQ.front();
            m_vReadyQ.pop_front();
            m_vReadySet.erase(pExecuteTask);
            if(!m_bSingleThreaded) {
                EnterCriticalSection(&m_csWaitingThreads);
                assert(m_vOutstandingTasks.find(pExecuteTask)==m_vOutstandingTasks.end());
                m_vOutstandingTasks.insert(pExecuteTask);
                LeaveCriticalSection(&m_csWaitingThreads);
            }
            if(m_vReadyQ.size() == 0)
                SignalTaskQueueEmpty();                
            if(m_vReadyQ.size() > 0) 
                SignalTasksAvailable();
        }
        LeaveCriticalSection(&m_csReadyQ);
        return pExecuteTask;
    }

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Called by graph runner threads before entering their run loop. A graph 
	/// 			may be considered truly 'running' when all it's threads are up. </summary>
	///
	/// <remarks>	crossbac, 8/12/2013. </remarks>
	///-------------------------------------------------------------------------------------------------

	void 
	Graph::SignalGraphRunnerThreadAlive(
		VOID
		)
	{
		InterlockedDecrement(&m_uiNascentGraphRunnerThreads);
	}

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Wait until all graph runner threads are alive. </summary>
	///
	/// <remarks>	crossbac, 8/12/2013. </remarks>
	///-------------------------------------------------------------------------------------------------

	void 
	Graph::WaitGraphRunnerThreadsAlive(
		VOID
		)
	{
		while(m_uiNascentGraphRunnerThreads>0) {
			DWORD dwSleepInterval = m_bMustPrimeGraphRunnerThreads ? 100 : 1;
			Sleep(dwSleepInterval);
		}
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Get any backend-specific per-thread initialization off the critical path. Called
    ///             by graph runner threads before they signal that they are alive.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 8/12/2013. </remarks>
    ///
    /// <param name="dwThreadId">           Identifier for the thread. </param>
    /// <param name="uiThreadRoleIndex">    Handle of the thread. </param>
    /// <param name="uiThreadRoleMax">      The thread role maximum. </param>
    /// <param name="bPooledThread">        true if the thread is a globally pooled thread. </param>
    ///-------------------------------------------------------------------------------------------------

	void 
	Graph::PrimeGraphRunnerThread(
		__in DWORD dwThreadId,
        __in UINT uiThreadRoleIndex,
        __in UINT uiThreadRoleMax,
        __in BOOL bPooledThread
		)
	{
        if(uiThreadRoleIndex >= uiThreadRoleMax) {
            PTask::Runtime::MandatoryInform("%s::%s(%d): priming graph runner thread %d with thread pool size %d. Are there multiple live graphs?\n",
                                            __FILE__,
                                            __FUNCTION__,
                                            __LINE__,
                                            uiThreadRoleIndex,
                                            uiThreadRoleMax);
        }
        Accelerator::InitializeTLSContextManagement(PTTR_GRAPHRUNNER,
                                                    uiThreadRoleIndex,
                                                    uiThreadRoleMax,
                                                    bPooledThread);
		if(m_bMustPrimeGraphRunnerThreads) {
            std::set<Accelerator*> vAccs;
            std::set<Accelerator*>::iterator si;
            Scheduler::EnumerateEnabledAccelerators(ACCELERATOR_CLASS_CUDA, vAccs);
            if(!bPooledThread || PTask::Runtime::GetApplicationThreadsManagePrimaryContext()) {

                // We started this thread. Make the CUDA context do its thing.
                // Note that if it is a *pooled* thread, we expect that
                // the thread priming was already done when the thread
                // was created by the pool.
                
                for(si=vAccs.begin(); si!=vAccs.end(); si++) {
                    CUAccelerator * pAccelerator = (CUAccelerator*)(*si);
                    pAccelerator->Lock();
                    pAccelerator->MakeDeviceContextCurrent();
			        PRIME_GRAPH_RUNNER_THREAD();
                    pAccelerator->ReleaseCurrentDeviceContext();
                    pAccelerator->Unlock();
                }
            }

			std::map<std::string, Task*>::iterator vi;
			for(vi=m_vTasks.begin(); vi!=m_vTasks.end(); vi++) {
                for(si=vAccs.begin(); si!=vAccs.end(); si++) {

                    // call any task-specific warmup for tasks that
                    // *might* be executed on this thread. This primarily
                    // boils down to configuring slab-allocators for 
                    // temporary buffers that tasks may need, but lack 
                    // a way to get them through PTask. 
                    
                    CUAccelerator * pAccelerator = (CUAccelerator*)(*si);
                    pAccelerator->Lock();
                    pAccelerator->MakeDeviceContextCurrent();
				    vi->second->OnWorkerThreadEntry(dwThreadId); 
                    pAccelerator->ReleaseCurrentDeviceContext();
                    pAccelerator->Unlock();
                }
			}
		}
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Graph runner thread proc. This version of the graph runner proc is designed
    ///             for use in contexts where the mapping from threads to tasks is 1:1. In this
    ///             case it suffices to wait for the graph to either enter the running state
    ///             or for the graph/runtime to start exiting. As long as we are not in a teardown
    ///             condition, just call the task's Execute method. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="lpvDescriptor">    pointer to a descriptor for the graph/task pair. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD WINAPI
    Graph::GraphRunnerProc(
        __in LPVOID lpvDescriptor
        ) 
    {
        PTASKERTHREADDESC pDesc = (PTASKERTHREADDESC) lpvDescriptor;
        Task * pTask = pDesc->pTask;
        Graph * pGraph = pDesc->pGraph;
        HANDLE vWaitHandles[] = { pDesc->hGraphRunningEvent,
                                  pDesc->hRuntimeTerminateEvent,
                                  pDesc->hGraphTeardownEvent };
        DWORD nHandles = sizeof(vWaitHandles) / sizeof(HANDLE);
        BOOL bTerminate = FALSE;
        BOOL bRunning = FALSE;
        BOOL bWaitingRunnable = TRUE;

        UINT uiThreadRoleIndex = pDesc->nThreadIdx;
        UINT uiThreadRoleMax = pDesc->nGraphRunnerThreadCount;
        BOOL bGraphRunnerThreadPooled = pDesc->bIsPooledThread;
		pGraph->PrimeGraphRunnerThread(GetCurrentThreadId(), uiThreadRoleIndex, uiThreadRoleMax, bGraphRunnerThreadPooled);
		pGraph->SignalGraphRunnerThreadAlive();

        try {
            
            while(!bTerminate && IsGraphAlive(pDesc->pGraphState)) {

                // wait for the graph to enter the run state. Until we see a signal that
                // allows us to attempt to schedule tasks, we are in the await (quiescent)
                // state, so we really only want to increment the blocked worker count when 
                // we have seen a transition from running to non-running.
                bWaitingRunnable = TRUE;
                InterlockedIncrement(&pGraph->m_uiThreadsAwaitingRunnableGraph);
                DWORD dwWait = WaitForMultipleObjects(nHandles, vWaitHandles, FALSE, INFINITE);
                switch(dwWait) {
                case WAIT_OBJECT_0 + 0: bRunning = TRUE; break;
                case WAIT_OBJECT_0 + 1: bRunning = FALSE; bTerminate = TRUE; break; // terminate--the run time is shutting down
                case WAIT_OBJECT_0 + 2: bRunning = FALSE; bTerminate = TRUE; break; // teardown--the graph is being torn down
                }
                InterlockedDecrement(&pGraph->m_uiThreadsAwaitingRunnableGraph);
                if(bRunning) {
                    // only decrement the waiting thread count if we exit in a runnable
                    // state--any other exit from the wait condition should force us
                    // to exit this thread.
                    bWaitingRunnable = FALSE;
                    assert(!bTerminate);
                }

                if(bRunning && IsGraphRunning(pDesc->pGraphState)) {
                    InterlockedIncrement(&pGraph->m_uiOutstandingExecuteThreads);
                    pTask->Execute(pDesc->pGraphState);
                    InterlockedDecrement(&pGraph->m_uiOutstandingExecuteThreads);
                }
            }
            delete pDesc;
#pragma warning(disable:4996)
        } catch(thrust::system_error &e) {
            PTask::Runtime::HandleError("%s: %s: Thrust exception: %s\n", 
                                        __FUNCTION__, 
                                        pTask->GetTaskName(),
                                        e.what());
#pragma warning(default:4996)
		} catch(...) {
            PTask::Runtime::HandleError("%s: %s: exception: %s\n", 
                                        __FUNCTION__, 
                                        pTask->GetTaskName());
        }
        Accelerator::DeinitializeTLSContextManagement();
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Graph monitor proc. Wakes up periodically, and if the runtime
    ///             is in the diagnostics mode, checks forward progress. If no dispatch
    ///             events have occurred since the given threshold, and there are user
    ///             threads blocked on Pull calls to externally exposed ports, we
    ///             hypothsize that the graph is hung for some reason and dump the
    ///             global state of the graph to the console. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="p">    The graph. </param>
    ///
    /// <returns>   dont care </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD WINAPI
    Graph::GraphMonitorProc(LPVOID p) {
        Graph * pGraph = reinterpret_cast<Graph*>(p);
        HANDLE vStartWaitHandles[] = { pGraph->m_hGraphRunningEvent,
                                       pGraph->m_hGraphStopEvent,
                                       pGraph->m_hRuntimeTerminateEvent,
                                       pGraph->m_hGraphTeardownEvent,
                                       pGraph->m_hProbeGraphEvent };
        HANDLE vWaitHandles[] = { pGraph->m_hGraphStopEvent,
                                  pGraph->m_hRuntimeTerminateEvent,
                                  pGraph->m_hGraphTeardownEvent,
                                  pGraph->m_hProbeGraphEvent };
        DWORD nStartHandles = sizeof(vStartWaitHandles) / sizeof(HANDLE);
        DWORD nHandles = sizeof(vWaitHandles) / sizeof(HANDLE);
        DWORD dwEnterRunstateTicks = 0;
        BOOL bRunning = FALSE;
        BOOL bTerminate = FALSE;

        while(!bTerminate && IsGraphAlive(&pGraph->m_eState)) {

            // if we are not running yet, block until the graph enters the run state.
            // otherwise, block until we get a probe, a time out, or  stop. 
            // Naturally, the usual caveats WRT watching for graph and runtime level
            // teardown apply in both cases.
            
            if(!bRunning) {

                DWORD dwWait = WaitForMultipleObjects(nStartHandles, vStartWaitHandles, FALSE, DEFAULT_MONITOR_TIMEOUT);
                switch(dwWait) {
                case WAIT_OBJECT_0:     bRunning = TRUE; dwEnterRunstateTicks = ::GetTickCount(); break; 
                case WAIT_OBJECT_0 + 1: bRunning = FALSE; dwEnterRunstateTicks = 0; break; 
                case WAIT_OBJECT_0 + 2: bTerminate = TRUE; break; 
                case WAIT_OBJECT_0 + 3: bTerminate = TRUE; break; 
                case WAIT_OBJECT_0 + 4: // probe graph
                    PTask::Runtime::Warning("graph not running: PTask graph probe event ignored");
                    break;
                case WAIT_TIMEOUT:  
                    continue;
                }

            } else {
                    
                DWORD dwWait = WaitForMultipleObjects(nHandles, vWaitHandles, FALSE, DEFAULT_MONITOR_TIMEOUT);
                switch(dwWait) {
                case WAIT_OBJECT_0 + 0: bRunning = FALSE; dwEnterRunstateTicks = 0; break; 
                case WAIT_OBJECT_0 + 1: bTerminate = TRUE; break; 
                case WAIT_OBJECT_0 + 2: bTerminate = TRUE; break; 
                case WAIT_OBJECT_0 + 3: // probe graph
                    if(!bRunning) {
                        PTask::Runtime::Warning("graph not running: PTask graph probe event ignored");
                    } else {
                        pGraph->ReportGraphState(std::cout, "C:\\temp\\graphprobe.dot", TRUE);
                    }
                    break;
                case WAIT_TIMEOUT:  
                    // timeout--if the graph has never run, just keep waiting.
                    // if we have set the running flag, *and* the graph is still
                    // alive/running, then check to see when the last dispatch
                    // occurred. If it was longer than the watchdog threshold
                    // then dump the state of the graph.
                    if(PTask::Runtime::GetUseGraphMonitorWatchdog()) {
                        bRunning = pGraph->IsRunning();
                        if(bRunning) {
                            DWORD dwCurrentTicks = ::GetTickCount();
                            DWORD dwLastDispatch = Scheduler::GetLastDispatchTimestamp();
                            DWORD dwLastEvent = dwLastDispatch ? dwLastDispatch : dwEnterRunstateTicks;
                            DWORD dwDelta = (dwCurrentTicks - dwLastEvent);
                            if(dwDelta > PTask::Runtime::GetDispatchWatchdogThreshold()) {
                                std::cout 
                                    << "Graph Monitor Proc: last dispatch was " 
                                    << dwDelta << " ticks ago. Hung? Dumping graph state..."
                                    << std::endl;
                                pGraph->ReportGraphState(std::cout, "C:\\temp\\hunggraph.dot", TRUE);
                            }
                        }
                    }
                    break;
                }
            }
        }

        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Add the given task thread to the pool of threads that
    ///             are known to be awaiting new potentially ready tasks to enter
    ///             the ready queue. </summary>
    ///
    /// <remarks>   Crossbac, 3/17/2013. </remarks>
    ///
    /// <param name="uiThreadPoolIdx">  Zero-based index of the thread pool. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Graph::EnqueueWaitingTaskThread(
        UINT uiThreadPoolIdx
        )
    {
        BOOL bMustWaitForWork = TRUE;
        assert(!m_bSingleThreaded);
        assert(m_bUseReadyQ);
        if(PTask::Runtime::GetThreadPoolSignalPerThread()) {
            EnterCriticalSection(&m_csWaitingThreads);
            if(m_eState == PTGS_RUNNING && !m_bForceTeardown) {

                assert(m_vWaitingThreads.find(uiThreadPoolIdx) == m_vWaitingThreads.end());
                assert(m_vOutstandingNotifications.find(uiThreadPoolIdx) == m_vOutstandingNotifications.end());

                if(m_uiReadyQSurplus > 0) {
                    // if there is a surplus of tasks available, and we are entering
                    // the waiter list, then decrement the surplus and just take the
                    // work for ourselves. In fact, don't even put ourselves on the wait list.
                    --m_uiReadyQSurplus;
                    bMustWaitForWork = FALSE;
                } else {
                    // no surplus was recorded, so while there may be entries in the
                    // ready q, they could be assigned to other threads. Add ourselves
                    // to the waiter list, and enter the wait state.
                    bMustWaitForWork = TRUE;
                    m_vWaitingThreads.insert(uiThreadPoolIdx);
                }
            }
            LeaveCriticalSection(&m_csWaitingThreads);
        } else {
            // no fine-grain signalling, every thread waits
            // on the same event: therefore we must always wait.
            bMustWaitForWork = TRUE;
        }
        return bMustWaitForWork;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Remove the given task thread from the pool of threads that
    ///             are known to be awaiting new potentially ready tasks to enter
    ///             the ready queue, and which have been signalled to attempt to
    ///             dequeue something. When the oustanding signals list is empty
    ///             but the runq is non-empty, we need to signal another thread. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/17/2013. </remarks>
    ///
    /// <param name="uiThreadPoolIdx">  Zero-based index of the thread pool. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::AcknowledgeReadyQSignal(
        UINT uiThreadPoolIdx
        )
    {
        assert(!m_bSingleThreaded);
        assert(m_bUseReadyQ);
        if(!PTask::Runtime::GetThreadPoolSignalPerThread())
            return;
        EnterCriticalSection(&m_csWaitingThreads);
        assert(m_vWaitingThreads.find(uiThreadPoolIdx) == m_vWaitingThreads.end() || m_eState == PTGS_QUIESCING);
        assert(m_vOutstandingNotifications.find(uiThreadPoolIdx) != m_vOutstandingNotifications.end() || m_eState == PTGS_QUIESCING);
        m_vOutstandingNotifications.erase(uiThreadPoolIdx);
        LeaveCriticalSection(&m_csWaitingThreads);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notifies a dispatch attempt complete. </summary>
    ///
    /// <remarks>   Crossbac, 3/17/2013. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::NotifyDispatchAttemptComplete(
        Task* pTask
        )
    {
        assert(!m_bSingleThreaded);
        assert(m_bUseReadyQ);
        EnterCriticalSection(&m_csReadyQ);
        EnterCriticalSection(&m_csWaitingThreads);
        m_vOutstandingTasks.erase(pTask);
        std::set<Task*>::iterator si = m_vDeferredTasks.find(pTask);
        if(si != m_vDeferredTasks.end()) {
            if(m_vReadySet.find(pTask) == m_vReadySet.end()) {
                m_vDeferredTasks.erase(si);
                m_vReadySet.insert(pTask);
                m_vReadyQ.push_back(pTask);
                assert(m_vReadySet.size() > 0);
                assert(m_vReadyQ.size() > 0);
            } else {
                // another one is in the queue again already
                // so we can't move this to the ready queue
                // again yet--leave it deferred.
                // which means no action needed.
            }
            if(m_vReadyQ.size()) {
                assert(m_vReadySet.size() == m_vReadyQ.size());
                SignalTasksAvailable();            
            }
        }
        LeaveCriticalSection(&m_csWaitingThreads);
        LeaveCriticalSection(&m_csReadyQ);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Graph runner proc multi-threaded, with queueing to share threads from a dispatch
    ///             thread pool.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="lpvDescriptor">    The p. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD WINAPI
    Graph::GraphRunnerProcMTQ(
        __in LPVOID lpvDescriptor
        ) 
    {
        NAMETHREAD(L"GraphRunnerProcMTQ");
        DWORD dwWait = 0;
        PTASKERTHREADDESC_ST pTaskDescs = (PTASKERTHREADDESC_ST) lpvDescriptor;
        Graph * pGraph = pTaskDescs->pGraph;

        HANDLE vStartWaitHandles[] = { pTaskDescs->hGraphRunningEvent,
                                       pTaskDescs->hGraphTeardownEvent,
                                       pTaskDescs->hRuntimeTerminateEvent };
        HANDLE vWaitHandles[] = { pTaskDescs->hGraphTeardownEvent,
                                  pTaskDescs->hRuntimeTerminateEvent,
                                  pTaskDescs->hGraphStopEvent,
                                  pTaskDescs->hReadyQ };

        DWORD dwStartHandles = sizeof(vStartWaitHandles) / sizeof(HANDLE);
        DWORD dwHandles = sizeof(vWaitHandles) / sizeof(HANDLE);
        UINT uiThreadPoolIdx = pTaskDescs->nThreadIdx;
        UINT uiThreadRoleMax = pTaskDescs->nGraphRunnerThreadCount;
        map<HANDLE, Task*> portStatusMap;
        BOOL bTerminate = FALSE;
        BOOL bRunning = FALSE;
        BOOL bAwaitingRunnableGraph = FALSE;

        BOOL bGraphRunnerThreadPooled = pTaskDescs->bIsPooledThread;
		pGraph->PrimeGraphRunnerThread(GetCurrentThreadId(), uiThreadPoolIdx, uiThreadRoleMax, bGraphRunnerThreadPooled);
		pGraph->SignalGraphRunnerThreadAlive();
        record_task_thread_alive(pGraph);

        try {
            while(!bTerminate && IsGraphAlive(pTaskDescs->pGraphState)) {

                record_task_thread_block_run(pGraph);
                bAwaitingRunnableGraph = TRUE;
                InterlockedIncrement(&pGraph->m_uiThreadsAwaitingRunnableGraph);
                dwWait = WaitForMultipleObjects(dwStartHandles, vStartWaitHandles, FALSE, INFINITE); 
                record_task_thread_wake_run(pGraph);
                switch(dwWait) {
                case WAIT_TIMEOUT: continue;
                case WAIT_OBJECT_0 + 0: bRunning = TRUE; break;
                case WAIT_OBJECT_0 + 1: bRunning = FALSE; bTerminate = TRUE; break;
                case WAIT_OBJECT_0 + 2: bRunning = FALSE; bTerminate = TRUE; break;
                }
                InterlockedDecrement(&pGraph->m_uiThreadsAwaitingRunnableGraph);               
                if(bRunning) {
                    assert(!bTerminate);
                    bAwaitingRunnableGraph = FALSE;
                } else {
                    continue;
                }

                Task * pExecuteTask = NULL;    // next task to work on.
                BOOL bMustWaitForWork = FALSE; // do we have to wait on a semaphore?
                BOOL bOKToDequeue = FALSE;     // having optionally waited, is there actually work?

                // the call to enqueue ourselves as a waiter will return TRUE if we need to wait on an event to
                // be assigned work. The wait may complete in a state that suggests we elide any attempt to pop
                // something from the ready queue (e.g. termination/error). If EnqueueWaitingTaskThread returns
                // false, that means we needn't wait--consequently, it is also OK to attempt to pop something
                // from the ready queue.
                
                bMustWaitForWork = pGraph->EnqueueWaitingTaskThread(uiThreadPoolIdx);
                bOKToDequeue = !bMustWaitForWork;
                if(bMustWaitForWork) {
                    record_task_thread_block_tasks(pGraph);
                    MARKRANGEENTER(L"Wait-Tasks");
                    dwWait = WaitForMultipleObjects(dwHandles, 
                                                    vWaitHandles,
                                                    FALSE,
                                                    DEFAULT_RUNNING_TIMEOUT);
                    MARKRANGEEXIT();
                    record_task_thread_wake_tasks(pGraph);
                    switch(dwWait) {
                    case WAIT_TIMEOUT: continue; 
                    case WAIT_FAILED:  PTask::Runtime::HandleError("%s: wait fail\n", __FUNCTION__); continue; 
                    case WAIT_ABANDONED: PTask::Runtime::HandleError("%s: wait abandoned\n", __FUNCTION__); continue;
                    case WAIT_OBJECT_0 + 0: bOKToDequeue = FALSE; bRunning = FALSE; bTerminate = TRUE; continue; 
                    case WAIT_OBJECT_0 + 1: bOKToDequeue = FALSE; bRunning = FALSE; bTerminate = TRUE; continue; 
                    case WAIT_OBJECT_0 + 2: bOKToDequeue = FALSE; bRunning = FALSE; continue; 
                    case WAIT_OBJECT_0 + 3: 
                        // acknowledge the signal and attempt to do some work. 
                        pGraph->AcknowledgeReadyQSignal(uiThreadPoolIdx);
                        bOKToDequeue = TRUE;
                        break;
                    }
                }

                if(!bTerminate && 
                    bRunning && 
                   IsGraphRunning(pTaskDescs->pGraphState) && 
                   bOKToDequeue) {

                    record_task_thread_dequeue_attempt(pGraph);
                    pExecuteTask = pGraph->PopReadyQ();
                    record_task_thread_dequeue_complete(pGraph, pExecuteTask);
                    if(pExecuteTask != NULL) {
                        record_inflight_dispatch_attempt(pGraph);
                        MARKRANGEENTER(L"dispatch-attempt");
                        InterlockedIncrement(&pGraph->m_uiOutstandingExecuteThreads);
                        BOOL bExecuted = pExecuteTask->AttemptExecute(pTaskDescs->pGraphState);
                        pGraph->NotifyDispatchAttemptComplete(pExecuteTask);
                        InterlockedDecrement(&pGraph->m_uiOutstandingExecuteThreads);
                        MARKRANGEEXIT();
                        record_dispatch_attempt_complete(pGraph, bExecuted);
                        UNREFERENCED_PARAMETER(bExecuted);
                    }
                }

            }
            delete [] pTaskDescs->ppTasks;
            delete pTaskDescs;
        } catch(exception e) {
            PTask::Runtime::HandleError("%s: Exception caught %s\n", 
                __FUNCTION__, e.what());    
        }
        record_task_thread_exit(pGraph);
        Accelerator::DeinitializeTLSContextManagement();
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Graph runner proc single-threaded. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="p">    The p. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD WINAPI
    Graph::GraphRunnerProcSTQ(
        LPVOID p
        ) 
    {
        NAMETHREAD(L"GraphRunnerProcSTQ");
        PTASKERTHREADDESC_ST pTaskDescs = (PTASKERTHREADDESC_ST) p;
        Graph * pGraph = pTaskDescs->pGraph;        
        HANDLE vStartWaitHandles[] = { pTaskDescs->hGraphRunningEvent,
                                       pTaskDescs->hGraphTeardownEvent,
                                       pTaskDescs->hRuntimeTerminateEvent };
        HANDLE vWaitHandles[] = { pTaskDescs->hReadyQ,
                                  pTaskDescs->hGraphTeardownEvent,
                                  pTaskDescs->hRuntimeTerminateEvent,
                                  pTaskDescs->hGraphStopEvent };
        DWORD dwStartHandles = sizeof(vStartWaitHandles) / sizeof(HANDLE);
        DWORD dwHandles = sizeof(vWaitHandles) / sizeof(HANDLE);

        BOOL bTerminate = FALSE;
        BOOL bRunning = FALSE; 
        BOOL bAwaitingRunnableGraph = FALSE;
        BOOL bGraphRunnerThreadPooled = pTaskDescs->bIsPooledThread;
		pGraph->PrimeGraphRunnerThread(GetCurrentThreadId(), 0, 1, bGraphRunnerThreadPooled);
		pGraph->SignalGraphRunnerThreadAlive();
        record_task_thread_alive(pGraph);

        try {

            while(!bTerminate && IsGraphAlive(pTaskDescs->pGraphState)) {

                record_task_thread_block_run(pGraph);
                bAwaitingRunnableGraph = TRUE;
                InterlockedIncrement(&pGraph->m_uiThreadsAwaitingRunnableGraph);
                DWORD dwWait = WaitForMultipleObjects(dwStartHandles, vStartWaitHandles, FALSE, INFINITE);
                record_task_thread_wake_run(pGraph);
                switch(dwWait) {
                case WAIT_OBJECT_0 + 0: bRunning = TRUE; break;
                case WAIT_OBJECT_0 + 1: bRunning = FALSE; bTerminate = TRUE; break; 
                case WAIT_OBJECT_0 + 2: bRunning = FALSE; bTerminate = TRUE; break; 
                }
                InterlockedDecrement(&pGraph->m_uiThreadsAwaitingRunnableGraph);
                if(bRunning) {
                    bAwaitingRunnableGraph = FALSE;
                    assert(!bTerminate);
                } else {
                    continue;
                }

                Task * pExecuteTask = NULL;
                record_task_thread_block_tasks(pGraph);
                dwWait = WaitForMultipleObjects(dwHandles, vWaitHandles, FALSE, INFINITE);
                record_task_thread_wake_tasks(pGraph);
                switch(dwWait) {
                case WAIT_OBJECT_0 + 1: bRunning = FALSE; bTerminate = TRUE; continue;
                case WAIT_OBJECT_0 + 2: bRunning = FALSE; bTerminate = TRUE; continue;
                case WAIT_OBJECT_0 + 3: bRunning = FALSE; bTerminate = FALSE; continue;
                case WAIT_OBJECT_0 + 0:
                    record_task_thread_dequeue_attempt(pGraph);
                    if(IsGraphRunning(&pGraph->m_eState)) {
                        pExecuteTask = pGraph->PopReadyQ();
                        record_task_thread_dequeue_complete(pGraph, pExecuteTask);
                        if(pExecuteTask != NULL) {
                            record_inflight_dispatch_attempt(pGraph);
                            InterlockedIncrement(&pGraph->m_uiOutstandingExecuteThreads);
                            BOOL bSuccess = pExecuteTask->AttemptExecute(pTaskDescs->pGraphState);
                            InterlockedDecrement(&pGraph->m_uiOutstandingExecuteThreads);
                            record_dispatch_attempt_complete(pGraph, bSuccess);
                            UNREFERENCED_PARAMETER(bSuccess);
                        }
                    }
                    break;
                }
            }
        } catch(...) {
            PTask::Runtime::HandleError("%s: Exception caught\n", __FUNCTION__);
        }
        delete [] pTaskDescs->ppTasks;
        delete pTaskDescs;
        record_task_thread_exit(pGraph);    
        Accelerator::DeinitializeTLSContextManagement();
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the dominant accelerator class of the graph. 
    ///             A class is dominant if all the non-host tasks in the graph have
    ///             that class, and all dependent ports in the graph also have that
    ///             class. This is used to help the partitioner figure out if it
    ///             can indeed attempt a static partition of the graph.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/18/2014. </remarks>
    ///
    /// <returns>   The dominant accelerator class. </returns>
    ///-------------------------------------------------------------------------------------------------

    ACCELERATOR_CLASS 
    Graph::GetDominantAcceleratorClass(
        VOID
        )
    {
        ACCELERATOR_CLASS eDominantClass = ACCELERATOR_CLASS_UNKNOWN;
        assert(m_eState == PTGS_INITIALIZING && !m_bFinalized);
        std::map<std::string, Task* >::iterator vi;
        for(vi=m_vTasks.begin(); vi!=m_vTasks.end(); vi++) {

            Task * pTask = vi->second;
            ACCELERATOR_CLASS eTaskClass = pTask->GetAcceleratorClass();
            if(eTaskClass == ACCELERATOR_CLASS_HOST) {

                // host task. if the task has no dependent bindings, it
                // does not affect the "dominant" class. If it does, 
                // then the dependent class should be treated as the
                // task class. 
               
                if(!pTask->HasDependentAcceleratorBindings())
                    continue;

                UINT uiAccIndex = 0;
                int nRequiredCount = 0;
                eTaskClass = pTask->GetDependentAcceleratorClass(uiAccIndex, nRequiredCount);
                if(eTaskClass == ACCELERATOR_CLASS_UNKNOWN)
                    continue;

            } 

            assert(eTaskClass != ACCELERATOR_CLASS_UNKNOWN &&
                   eTaskClass != ACCELERATOR_CLASS_HOST);

            // non-host task class. either set the dominant class,
            // or verify that this task matches the dominant class.
            // if it does not match, it means the graph has multiple
            // accelerator classes, and therefore has no dominant 
            // class, so we can return early. 
               
            if(eDominantClass == ACCELERATOR_CLASS_UNKNOWN)
                eDominantClass = eTaskClass;
            else if(eDominantClass != eTaskClass) 
                return ACCELERATOR_CLASS_UNKNOWN;  // heterogeneous graph
        }
        return eDominantClass;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Finalize the structure of the graph before running it. 
    ///             This is an opportunity to complete any bindings that
    ///             could not be completed before all graph objects were 
    ///             present, sort any port traversal orders, preallocate block pools,
    ///             etc. 
    /// 		    </summary>
    ///
    /// <remarks>   Crossbac </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::Finalize(
        VOID
        ) 
    {
        Lock();

        if(m_eState == PTGS_INITIALIZING && !m_bFinalized) {

            Partition();                                            
            AllocateBlockPools();
            PropagateChannelConsistencyProperties();

            m_bMustPrimeGraphRunnerThreads |= PTask::Runtime::GetApplicationThreadsManagePrimaryContext();

            std::map<std::string, Task* >::iterator vi;
            for(vi=m_vTasks.begin(); vi!=m_vTasks.end(); vi++) {

                // call the on-graph-complete method of
                // every task to give each node a chance
                // to finalize data structures.
                Task * pTask = vi->second;
                pTask->OnGraphComplete();
				
				if(!m_bMustPrimeGraphRunnerThreads) {

					// deal with pathological thrust behavior. thrust uses the CUDA runtime API, which does some
					// high-latency per-thread initialization we want off the critical path. If a task is host task
					// with dependent bindings but does not want platform-specific objects with each dispatch (this
					// is the hallmark of thrust, which cannot use streams if we pass them in), infer that the
					// graph contains thrust tasks and make sure each graph-runner-proc makes a call that
					// initializes cudaRT before letting the graph::run method return. Sadly, this doesn't get the
					// overhead off the critical path for distributed dandelion, but it works for any single-
					// machine system using ptask. 

					m_bMustPrimeGraphRunnerThreads |= 
						((pTask->GetAcceleratorClass() == ACCELERATOR_CLASS_HOST) &&
						  pTask->HasDependentAcceleratorBindings() &&
						  !pTask->DependentBindingsRequirePSObjects());
				}
            }
            m_bFinalized = TRUE;
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Set the partitioning mode to use for this graph.
    ///     Should pass only one of the following values:
    ///       GRAPHPARTITIONINGMODE_NONE = 0:
    ///         The runtime will not partition graphs across multiple available accelerators.
    ///       GRAPHPARTITIONINGMODE_HINTED = 1:
    ///         The runtime will partition graphs across multiple available accelerators,
    ///         according to hints given explicitly by the application via PTask::SetSchedulerPartitionHint().
    ///       GRAPHPARTITIONINGMODE_HEURISTIC = 2:
    ///         The runtime will partition graphs across multiple available accelerators,
    ///         available accelerators, using a set of experimental heuristics.
    ///       AUTOPARTITIONMODE_OPTIMAL = 2:
    ///         The runtime will attempt to auto-partition graphs across multiple
    ///         available accelerators, using a graph cut algorithm that finds the min-cut.
    ///
    ///     The value cannot be changed after graph finalization (calling Run() on the graph).
    ///
    ///     The default is the value specified by PTask::Runtime::GetDefaultGraphPartitioningMode()
    ///     at the time the graph is created.
    /// </summary>
    ///
    /// <remarks>   jcurrey, 1/27/2014. </remarks>
    ///
    /// <param name="mode"> The graph partitioning mode. </param>
    ///
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::SetPartitioningMode(
        int mode
        )
    {
        Lock();
        assert(m_eState == PTGS_INITIALIZING);
        if(m_eState != PTGS_INITIALIZING) {
            PTask::Runtime::MandatoryInform("Graph::SetGraphPartitioningMode() called while graph not initializing. Ignored!");
        } else {
            m_ePartitioningMode = (GRAPHPARTITIONINGMODE)mode;
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Get the partitioning mode to use for this graph.
    ///     Will return one of the following values:
    ///       GRAPHPARTITIONINGMODE_NONE = 0:
    ///         The runtime will not partition graphs across multiple available accelerators.
    ///       GRAPHPARTITIONINGMODE_HINTED = 1:
    ///         The runtime will partition graphs across multiple available accelerators,
    ///         according to hints given explicitly by the application via PTask::SetSchedulerPartitionHint().
    ///       GRAPHPARTITIONINGMODE_HEURISTIC = 2:
    ///         The runtime will partition graphs across multiple available accelerators,
    ///         available accelerators, using a set of experimental heuristics.
    ///       AUTOPARTITIONMODE_OPTIMAL = 2:
    ///         The runtime will attempt to auto-partition graphs across multiple
    ///         available accelerators, using a graph cut algorithm that finds the min-cut.
    ///
    ///     The default is the value specified by PTask::Runtime::GetDefaultGraphPartitioningMode()
    ///     at the time the graph is created.
    /// </summary>
    ///
    /// <remarks>   jcurrey, 1/27/2014. </remarks>
    ///
    /// <returns>   The graph partitioning mode. </returns>
    ///-------------------------------------------------------------------------------------------------

    int
    Graph::GetPartitioningMode()
    {
        return m_ePartitioningMode;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Configure the optimal partitioner for this graph.
    ///
    ///     Currently the only configurable properties are the directory in which files related to the
    ///     execution of the partitioner will be written, and the prefix of the file names. This is useful 
    ///     when running a number of configurations for which you want to save and possibly inspect the 
    ///     partitioner input and output.
    ///
    ///     If this method is not called, the directory defaults to "C:\temp" and the prefix to 
    ///     "ptask_optimal_partition", resulting in the following files being created:
    ///       C:\temp\ptask_optimal_partitioner.partition_input.txt
    ///       C:\temp\ptask_optimal_partitioner.partition_output.txt
    ///
    ///     The partitioning mode must be set to AUTOPARTITIONINGMODE_OPTIMAL before calling this method.
    /// </summary>
    ///
    /// <remarks>   jcurrey, 1/31/2014. </remarks>
    ///
    /// <param name="workingDir"> The directory in which the model and solution files should be written. </param>
    /// <param name="fileNamePrefix"> The prefix of the model and solution file names. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::ConfigureOptimalPartitioner(
        int numPartitions,
        const char * workingDir,
        const char * fileNamePrefix
        )
    {
        Lock();
        if (GetPartitioningMode() != GRAPHPARTITIONINGMODE_OPTIMAL)
        {
            PTask::Runtime::HandleError(
                "%s:%s GRAPHPARTITIONINGMODE_OPTIMAL must be set to use this method.\n", __FILE__, __FUNCTION__);
            return;
        }

        if (m_pPartitioner != NULL)
        {
            PTask::Runtime::HandleError(
                "%s:%s Partitioner may not be configured more than once for a given graph.\n", __FILE__, __FUNCTION__);
            return;
        }

        m_pPartitioner = new Partitioner(this, numPartitions, workingDir, fileNamePrefix);
        Unlock();        
    }
    
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Set a partition of the graph explicitly. This is experimental code that
    ///             takes a partition provided as a vector of ints, created by an 
    ///             external tool. This is very brittle, (sensitive to the order in which
    ///             node identifiers are assigned) and needs a different API if we
    ///             find that this is a performance profitable approach. Currently,
    ///             the vector is essentially pasted out of a text file received from
    ///             Renato.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 11/21/2013. </remarks>
    ///
    /// <param name="vPartition">       [in,out] If non-null, the partition. </param>
    /// <param name="nPartitionHints">  The partition hints. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::SetExplicitPartition(
        __in int * vPartition,
        __in UINT nPartitionHints
        )
    {
        Lock();
        assert(m_eState == PTGS_INITIALIZING);
        if(m_eState != PTGS_INITIALIZING) {
            PTask::Runtime::MandatoryInform("Graph::SetExplicitPartition() called while graph not initializing. Ignored!");
        } else {
            m_bExplicitPartition = TRUE;
            m_pExplicitPartition = new int[nPartitionHints];
            memcpy(m_pExplicitPartition, vPartition, nPartitionHints*sizeof(int));
            m_uiExplicitPartitionElems = nPartitionHints;
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets partition identifier. </summary>
    ///
    /// <remarks>   Crossbac, 11/21/2013. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///
    /// <returns>   The partition identifier. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    Graph::GetPartitionId(
        __in Task * pTask,
        __in std::map<Task*, int>* pNodeNumberMap
        )
    {
        if(m_bExplicitPartition) {

            // explicitly partition the graph according to the vector in m_pExplicitPartition,
            // which should have a per task entry specifying its partition id. Note that we 
            // require that node ids match those used to generate the input to the partitioner,
            // which (currently) is in WriteWeightedModel. 
            // TODO/FIXME: make this less brittle! (if it works)

            assert(pNodeNumberMap != NULL);
            assert(m_pExplicitPartition != NULL);
            assert(pNodeNumberMap->size() == m_vTasks.size());
            assert(pNodeNumberMap->find(pTask) != pNodeNumberMap->end());
            assert(m_uiExplicitPartitionElems == (UINT)m_vTasks.size());

            if((m_pExplicitPartition == NULL) ||
               (m_uiExplicitPartitionElems != (UINT)m_vTasks.size())) {
                PTask::Runtime::HandleError("%s:%s: Invalid parameters\n");
                return 0;
            }

            int nNodeId = (*pNodeNumberMap)[pTask];
            int nPartitionIndex = nNodeId - 1;
            assert(nPartitionIndex >= 0 && nPartitionIndex < m_vTasks.size());
            return static_cast<UINT>(m_pExplicitPartition[nPartitionIndex]);
        }

        return pTask->GetSchedulerPartitionHint();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Partition the graph explicitly based on information provided in the
    ///             call to SetExplicitPartition
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 11/21/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::ExplicitPartition(
        VOID
        )
    {
        int nodeId=0;
        std::map<Task*, int> nodeNumberMap;    
        if(m_bExplicitPartition) {
            std::map<std::string, Task*>::iterator iter;
            for(iter=m_vTasks.begin(); iter!=m_vTasks.end() ; ++iter) {
                ++nodeId; // index from 1, not 0!
                Task * pTask = iter->second;
                assert(pTask != NULL);
                nodeNumberMap[pTask] = nodeId;
            }
        }

        int nRequiredCount = 0;
        std::set<Task*>::iterator si;
        std::map<UINT, std::set<Task*>*> vRequestedPartitionMap;
        std::map<UINT, std::set<Task*>*>::iterator pi;
        std::map<std::string, Task*>::iterator mi;
        std::set<ACCELERATOR_CLASS> vAccClasses;
        for(mi=m_vTasks.begin(); mi!=m_vTasks.end(); mi++) {
            Task * pTask = mi->second;
            UINT uiPartitionId = GetPartitionId(pTask, &nodeNumberMap);
            std::set<Task*>* pPartitionTasks = NULL;
            pi=vRequestedPartitionMap.find(uiPartitionId);
            if(pi==vRequestedPartitionMap.end()) {
                pPartitionTasks = new std::set<Task*>();
                vRequestedPartitionMap[uiPartitionId] = pPartitionTasks;
            } else {
                pPartitionTasks = pi->second;
            }
            pPartitionTasks->insert(pTask);
            ACCELERATOR_CLASS eTaskClass = pTask->GetAcceleratorClass();
            if(eTaskClass != ACCELERATOR_CLASS_HOST) 
                vAccClasses.insert(eTaskClass);
            UINT uiDepClasses = pTask->GetDependentBindingClassCount();
            for(UINT ui=0; ui<uiDepClasses; ui++) {
                ACCELERATOR_CLASS ePortClass = pTask->GetDependentAcceleratorClass(ui, nRequiredCount);
                assert(ePortClass != ACCELERATOR_CLASS_HOST); // abuse of the abstraction--shouldn't happen
                if(ePortClass != ACCELERATOR_CLASS_HOST)
                    vAccClasses.insert(ePortClass);
            }
        }

        // FIXME: TODO:
        // ------------
        // a truly general implemementation of this would assume it is possible to have different
        // resource availabilities per accelerator class, implying that that the mapping from graph
        // partition to resource is different per class. However, in reality this situation can only
        // arise if the graph is using multiple back-ends, AND has somehow disabled some GPUs for one
        // backend but not another. So for now we assume we just have one accelerator class and
        // multiple accelerators of that class to map to each partition. Complain loudly if the
        // assumption is violated. 

        if(vAccClasses.size() != 1) {
            PTask::Runtime::HandleError("%s: Unsupported use of graph partitioning in heterogenous framework graph!\n", 
                                        __FUNCTION__);
            return;
        }

        // return early if there are no classes for which have a plurality of resources
        // or if the number of partitions forces us trivially into a single partition.
        UINT uiRequestedPartitionCount = (UINT)vRequestedPartitionMap.size();
        BOOL bPartitionPlurality = uiRequestedPartitionCount > 1;
        BOOL bResourcePlurality = FALSE;
        std::set<ACCELERATOR_CLASS>::iterator aci;
        for(aci=vAccClasses.begin(); aci!=vAccClasses.end(); aci++) {
            std::set<Accelerator*> vClassSet;
            Scheduler::EnumerateEnabledAccelerators(*aci, vClassSet);
            if(vClassSet.size() > 1) {
                // at least one resource class has a plurality enabled. 
                // so we must go forward with the partition. 
                bResourcePlurality = TRUE;
                break;
            }
        }
        if(!(bPartitionPlurality && bResourcePlurality))
            return; // cannot partition!

        // for each class we want to map, create a partitioning based on
        // the resource availability for that class.
        std::map<Task*, std::map<ACCELERATOR_CLASS, Accelerator*>> vActualPartitionMap;
        for(aci=vAccClasses.begin(); aci!=vAccClasses.end(); aci++) {
                
            // find the set of resource to map
            std::set<Accelerator*> vClassSet;
            std::vector<Accelerator*> vClassList;
            Scheduler::EnumerateEnabledAccelerators(*aci, vClassSet);
            if(vClassSet.size() < 2) continue;

            // create a copy of the class set that we can access
            // by index later when we map partition requests
            vClassList.assign(vClassSet.begin(), vClassSet.end());

            // there is a policy decision to make here. Distribute 
            // actual resources round-robin to partitions? This is probably
            // the most reasonable thing to do. On the other hand, if partitions
            // are adjacent, a round-robin policy will sacrifice locality.
            // For now, assume adjacent partition ids imply some graph adjacency. 

            UINT uiPartitionIndex=0;
            std::set<Task*>::iterator tsi;
            std::map<UINT, std::set<Task*>*>::iterator rpi;
            UINT uiReqPartitionPerActualPartition = uiRequestedPartitionCount / (UINT)vClassSet.size();
            uiReqPartitionPerActualPartition = max(uiReqPartitionPerActualPartition, 1);
            for(uiPartitionIndex=0, rpi=vRequestedPartitionMap.begin(); 
                uiPartitionIndex<uiRequestedPartitionCount && rpi!=vRequestedPartitionMap.end(); 
                uiPartitionIndex++, rpi++) {
                std::set<Task*>* vAssignSet = rpi->second;
                UINT uiAssignedPartition = uiPartitionIndex / uiReqPartitionPerActualPartition;
                uiAssignedPartition = min((UINT)(vClassSet.size()-1), uiAssignedPartition);
                Accelerator * pAssignment = vClassList[uiAssignedPartition];
                for(tsi=vAssignSet->begin(); tsi!=vAssignSet->end(); tsi++) {
                    Task * pAssignTask = *tsi;
                    (vActualPartitionMap[pAssignTask])[*aci] = pAssignment;
                }
            }
        }

        // at this point we have a map from tasks to the per accelerator-class
        // assignment. It suffices to iterate over all the tasks in that map,
        // figure out how to map that assignment to the task and dependent ports,
        // and make a call to set mandatory affinities accordingly.
        std::map<Task*, std::map<ACCELERATOR_CLASS, Accelerator*>>::iterator vpmi;
        for(vpmi=vActualPartitionMap.begin(); vpmi!=vActualPartitionMap.end(); vpmi++) {

            Task * pTask = vpmi->first;
            ACCELERATOR_CLASS eTaskClass = pTask->GetAcceleratorClass();

            if(eTaskClass != ACCELERATOR_CLASS_HOST) {

                // we're dealing with an accelerator-based task. we only
                // care about making an assignment for the task, since 
                // dependent port assignments can only occur on host tasks.
                Accelerator * pTaskAssignment = vpmi->second[eTaskClass];
                pTask->SetAffinity(pTaskAssignment, AFFINITYTYPE_MANDATORY);
                assert(!pTask->HasDependentAcceleratorBindings());

            } else {

                // we are dealing with a host-task. if it has dependent ports
                // then set mandatory affinity. If its a pure host task we should
                // not have added it to the map.
                std::map<UINT, Port*>::iterator deppi;
                std::map<UINT, Port*>* oPorts = pTask->GetInputPortMap();
                for(deppi=oPorts->begin(); deppi!=oPorts->end(); deppi++) {
                    Port * pIPort = deppi->second;
                    if(pIPort->HasDependentAcceleratorBinding()) {
                        ACCELERATOR_CLASS ePortClass = pIPort->GetDependentAcceleratorClass(0);
                        Accelerator * pPortAssignment = vpmi->second[ePortClass];
                        pTask->SetDependentAffinity(pIPort, pPortAssignment, AFFINITYTYPE_MANDATORY);
                    }
                }
                oPorts = pTask->GetOutputPortMap();
                for(deppi=oPorts->begin(); deppi!=oPorts->end(); deppi++) {
                    Port * pIPort = deppi->second;
                    if(pIPort->HasDependentAcceleratorBinding()) {
                        ACCELERATOR_CLASS ePortClass = pIPort->GetDependentAcceleratorClass(0);
                        Accelerator * pPortAssignment = vpmi->second[ePortClass];
                        pTask->SetDependentAffinity(pIPort, pPortAssignment, AFFINITYTYPE_MANDATORY);
                    }
                }
            }
        }

        // cleanup the sets we allocated to map tasks to requested partitions
        for(pi=vRequestedPartitionMap.begin(); pi!=vRequestedPartitionMap.end(); pi++) {
            delete pi->second;
        }

    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Records the fact that tasks in this graph have explicit scheduler partition hints. 
    ///             This simplifies the task of establishing a partition during graph finalization:
    ///             if there are partition hints in the graph, the finalizer elides heuristic partitioning
    ///             entirely and uses the per-task hints it encounters. </summary>
    ///
    /// <remarks>   crossbac, 6/28/2013. </remarks>
    ///
    /// <param name="bHintsConfigured"> The hints configured. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::SetHasSchedulerPartitionHints(
        __in BOOL bHintsConfigured
        )
    {
        Lock();
        assert(m_eState == PTGS_INITIALIZING);
        if(m_eState != PTGS_INITIALIZING) {
            PTask::Runtime::MandatoryInform("Graph::SetHasSchedulerPartitionHints() called while graph not initializing. Ignored!");
        } else {
            m_bHasSchedulerPartitionHints = bHintsConfigured;
            if(!m_bHasSchedulerPartitionHints) {
                PTask::Runtime::Warning("Graph::SetHasSchedulerPartitionHints(FALSE) called. ???");
            }
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if tasks in this graph have explicit scheduler partition hints. 
    ///             This simplifies the task of establishing a partition during graph finalization:
    ///             if there are partition hints in the graph, the finalizer elides heuristic partitioning
    ///             entirely and uses the per-task hints it encounters. </summary>
    ///
    /// <remarks>   crossbac, 6/28/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Graph::GetHasSchedulerPartitionHints(
        VOID
        )
    {
        assert(m_eState == PTGS_INITIALIZING);
        return m_bHasSchedulerPartitionHints;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Heuristic partition. </summary>
    ///
    /// <remarks>   Crossbac, 11/21/2013. </remarks>
    ///
    /// <param name="eClass">   The class. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::HeuristicPartition(
        __in ACCELERATOR_CLASS eClass
        )
    {
        assert(eClass != ACCELERATOR_CLASS_UNKNOWN);
        if(eClass == ACCELERATOR_CLASS_UNKNOWN ||
           eClass == ACCELERATOR_CLASS_HOST)
           return;
        assert(!m_bExplicitPartition);

        // use heuristics--partition by finding exposed channels with block pools divide them according
        // the compute resources, and set mandatory affinity at those partition points. Rely on data
        // affinity downstream to create a de-facto partition. 

        std::set<Accelerator*> accs;
        std::map<ACCELERATOR_CLASS, std::set<Accelerator*>>* ppAccMap;
        ppAccMap = Scheduler::EnumerateBlockPoolAccelerators();
        std::map<ACCELERATOR_CLASS, std::set<Accelerator*>>& pAccMap = *ppAccMap;
        std::set<Accelerator*>::iterator aci;
        for(aci=pAccMap[eClass].begin(); aci!=pAccMap[eClass].end(); aci++) {
            Accelerator * pAccelerator = *aci;
            accs.insert(pAccelerator);
        }
        if(accs.size() <= 1)
            return;   // not enough accelerators to warrant partition

        
        // look for tasks that have a block pool hint on an input port
        // and add them to the list of candidates: this is not a general
        // solution, but the presence of a block pool on an input *port*
        // generally means that this is a place where there is frequent 
        // need for allocation, making it a likely cut point for a partition

        std::map<Task*, Port*> vPartitionPoints;
        map<std::string, Task*>::iterator ti;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {
            std::map<UINT, Port*>::iterator pi;
            Task * pTask = ti->second;
            assert(pTask != NULL);
            std::map<UINT, Port*>* oPorts = pTask->GetInputPortMap();
            for(pi=oPorts->begin(); pi!=oPorts->end(); pi++) {
                Port * pIPort = pi->second;
                if(pIPort->HasUpstreamChannelPool()) {
                    std::map<UINT, Port*>* dsPorts = pTask->GetOutputPortMap();
                    if(dsPorts->size() != 1)
                        continue;         // we want to choose tasks such that the cut cost is 
                                            // minimal for the partition. prefer graphs with
                                            // a single output channel
                    if((*(dsPorts->begin())).second->GetChannelCount() > 1)
                        continue;
                    if(pTask->GetAcceleratorClass() == eClass ||
                        (pIPort->HasDependentAcceleratorBinding() && 
                            pIPort->GetDependentAcceleratorClass(0) == eClass)) {
                        vPartitionPoints[pTask] = pIPort;                   
                        break;
                    }
                }
            }
        }

        // assign affinity on partition points
        if(vPartitionPoints.size() >= 1) {
            std::set<Task*> vassigned;
            UINT uiPartitionPointsPerAcc = (UINT)max(vPartitionPoints.size() / accs.size(), 1);
            std::vector<AFFINITYTYPE> afftype;
            std::set<AFFINITYTYPE> affset;
            afftype.push_back(AFFINITYTYPE_MANDATORY);
            affset.insert(AFFINITYTYPE_MANDATORY);
            while(vPartitionPoints.size()) {
                std::set<Accelerator*>::iterator si = accs.begin();
                Accelerator * pAcc = *si;
                std::vector<Accelerator*> assignset;
                std::set<Accelerator*> assignvset;
                assignset.push_back(pAcc);
                assignvset.insert(pAcc);
                UINT uiUpperBound = (UINT)(accs.size() > 1 ? uiPartitionPointsPerAcc : vPartitionPoints.size());
                for(UINT i=0; i<uiUpperBound && vPartitionPoints.size(); i++) {
                    std::map<Task*, Port*>::iterator mi = vPartitionPoints.begin();
                    if(mi == vPartitionPoints.end())
                        break;
                    Task * pTask = mi->first;
                    Port * pPort = mi->second;
                    vPartitionPoints.erase(mi);
                    if(pPort->HasDependentAcceleratorBinding())
                        pTask->SetDependentAffinity(pPort, assignset, afftype);
                    else 
                        pTask->SetAffinity(assignset, afftype);
                    vassigned.insert(pTask);
                }
                accs.erase(*si);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Round robin partition of the graph using mandatory affinity. </summary>
    ///
    /// <remarks>   Crossbac, 11/21/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::RoundRobinPartition(
        VOID
        )
    {
        // if the runtime is in round-robin mode and we have multiple accelerators,
        // *and* this graph has a strict assignment, we want to select an accelator using 
        // based on the the modulus of the assingment, and set mandatory affinity for 
        // every task and dependent task in the graph to use that accelerator. 
           
        if(!m_bStrictAcceleratorAssignment) return;
        std::set<Accelerator*> vAccelerators;
        Scheduler::EnumerateEnabledAccelerators(ACCELERATOR_CLASS_CUDA, vAccelerators);
        if(vAccelerators.size() <= 1) return; 
        m_uiAffinitizedAccelerator = m_uiAffinitizedAccelerator % vAccelerators.size();
        std::set<Accelerator*>::iterator si;
        BOOL bAffinitizedAccFound = FALSE;
        for(si=vAccelerators.begin(); si!=vAccelerators.end(); si++) {
            Accelerator * pAccelerator = (*si);
            UINT uiDevID = (UINT) pAccelerator->GetDevice();
            if(m_uiAffinitizedAccelerator == uiDevID) {
                bAffinitizedAccFound = TRUE;
                m_pStrictAffinityAccelerator = pAccelerator;
            }
        }
        assert(m_pStrictAffinityAccelerator != NULL);
        assert(bAffinitizedAccFound);
        if(!bAffinitizedAccFound) return; 

        // we have an affinitized accelarator. traverse all tasks in the
        // graph and configure mandatory affinity for non-host tasks, as
        // well as host tasks with dependent accelerator bindings. 
            
        std::map<std::string, Task*>::iterator dmi;
        for(dmi=m_vTasks.begin(); dmi!=m_vTasks.end(); dmi++) {
            Task * pTask = dmi->second;
            if(pTask->GetAcceleratorClass() == m_pStrictAffinityAccelerator->GetClass()) {
                pTask->SetAffinity(m_pStrictAffinityAccelerator, AFFINITYTYPE_MANDATORY);
            } else if(pTask->HasDependentAcceleratorBindings()) {
                std::set<Port*> vDependentPorts;
                std::set<Port*>::iterator psi;
                std::map<UINT, Port*>::iterator pi;
                std::map<UINT, Port*>* iPorts = pTask->GetInputPortMap();
                std::map<UINT, Port*>* oPorts = pTask->GetOutputPortMap();
                for(pi=oPorts->begin(); pi!=oPorts->end(); pi++) {
                    Port * pIPort = pi->second;
                    if(pIPort->HasDependentAcceleratorBinding() && 
                        pIPort->GetDependentAcceleratorClass(0) == m_pStrictAffinityAccelerator->GetClass())
                        vDependentPorts.insert(pIPort);
                }
                for(pi=iPorts->begin(); pi!=iPorts->end(); pi++) {
                    Port * pIPort = pi->second;
                    if(pIPort->HasDependentAcceleratorBinding() && 
                        pIPort->GetDependentAcceleratorClass(0) == m_pStrictAffinityAccelerator->GetClass())
                        vDependentPorts.insert(pIPort);                        
                }
                for(psi=vDependentPorts.begin();psi!=vDependentPorts.end();psi++) {
                    Port * pBindPort = *psi;
                    pTask->SetDependentAffinity(pBindPort, m_pStrictAffinityAccelerator, AFFINITYTYPE_MANDATORY);
                }
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Optimal partition in process. </summary>
    ///
    /// <remarks>   Crossbac, 3/18/2014. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::OptimalPartitionInProcess(
        VOID
        )
    {
        assert(FALSE);
        assert(g_bUseDLLPartitioner);

        // If the user has put the graph in auto partition mode, use a graph-partitioning algorithm to
        // try to find an optimal assignment that utilizes all available GPUs. Currently, more than 2
        // GPUs can be supported by recursively applying the partitioner, (with corresponding loss of
        // the optimality guarantee), but since we have some commitment from Renato to enhance the
        // partioner algorithm to lift this restriction, for now, we support just 2 partitions. 
    
        std::set<Accelerator*> vClassSet;
        Scheduler::EnumerateEnabledAccelerators(ACCELERATOR_CLASS_CUDA, vClassSet);
        if(vClassSet.size() != 2) {
            PTask::Runtime::MandatoryInform("%s::%s--%d accelerators, but partitioner has max==2.\n"
                                            "Defaulting to heuristic partition algorithm (non-optimal!)\n",
                                            __FILE__,
                                            __FUNCTION__,
                                            vClassSet.size());
        } else {
            // We can use the optimal partitioner, since we have the right number of accelerators. 
            // the partitioner.AssignPartition call below will use the SetExplicitPartition API
            // on the graph once it has arrived at a partition, so the logic below using m_bExplicitPartition
            // will be followed, assuming there is a successful partition result here.
            int nPartitions = 2;
            // int nSolutionValue = -1;
            // int nSolutionEvaluation = -1; 
            Partitioner partitioner(this, nPartitions);
            //if(partitioner.Partition(nPartitions, nSolutionValue, nSolutionEvaluation)) {
            //    partitioner.AssignPartition();
            //}
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   PTask's data aware scheduling strategy is effective for regular graph structures,
    ///             but runs into problems particularly in the presence of large graphs with cycles.
    ///             The greedy  approach of scheduling whereever data is most recent fails to plan
    ///             ahead to ensure that data on back edges is in the same memory space as data on
    ///             forward edges (some level of global view is necessary to handle this), and a
    ///             brute force search for the most performant assignment is impractical due to the
    ///             size of the graph. In such cases we want to use a graph partitioning algorithm,
    ///             and schedule partitions on available accelerators according to min-cost cuts. Udi
    ///             and Renato are currently helping with this, but in the meantime, some simple
    ///             heuristics can be used to help guide the scheduler to approximate this kind of
    ///             partitioning approach. This is a temporary fix. The method checks whether the
    ///             graph is large and contains cycles, and if so, it use some heuristics to
    ///             find transition points that would likely be chosen by a graph partitioning
    ///             algorithm, based on the presence or absence of certain types of block pools.
    ///             </summary>
    ///                     
    /// <remarks>   Crossbac, 5/3/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::Partition(
        VOID
        )
    {
        assert(LockIsHeld());
        assert(!__IsFinalized());

        if(!PTask::Runtime::MultiGPUEnvironment()) {

            // there aren't enough accelerators present or enabled to require a 
            // partition. This step becomes a no-op, but warn the programmer if a 
            // partitioning mode was selected that indicates some other expectation. 
            
            if (m_ePartitioningMode != GRAPHPARTITIONINGMODE_NONE) 
                PTask::Runtime::MandatoryInform("WARNING: %s::%s Partitioning mode %d has no effect: GPU count <= 1\n",
                                                __FILE__, 
                                                __FUNCTION__);
            return;  
        }
        
        if(PTask::Runtime::GetGraphAssignmentPolicy() == GMP_ROUND_ROBIN) {

            // if the user has requested a round-robin partition, hopefully that user
            // is a researcher, collecting baseline datapoints! Round-robin is just
            // about the worst you can do in terms of destroying locality when the
            // graph is large relative to the number of nodes in the graph. 
            
            PTask::Runtime::MandatoryInform("%s::%s using round-robin partition policy\n",
                                            __FILE__,
                                            __FUNCTION__);
            RoundRobinPartition();
            return;
        }



        switch(m_ePartitioningMode) {
        #pragma warning(push)
        #pragma warning(disable:4127)
        case GRAPHPARTITIONINGMODE_NONE:
            
            RequirePartitionModeSettings(HintsNo, NonMandatory, ExplicitNo, NonMandatory);
            return;

        case GRAPHPARTITIONINGMODE_HINTED:

            RequirePartitionModeSettings(HintsYes, Mandatory, ExplicitNo, Mandatory);
            ExplicitPartition();
            return;

        case GRAPHPARTITIONINGMODE_HEURISTIC:

            RequirePartitionModeSettings(HintsNo, Mandatory, ExplicitNo, Mandatory);
            HeuristicPartition(GetDominantAcceleratorClass());
            return;

        case GRAPHPARTITIONINGMODE_OPTIMAL:

            if(g_bUseDLLPartitioner) {
                OptimalPartitionInProcess();
                return;
            } 

            RequirePartitionModeSettings(HintsAny, DontCare, ExplicitNo, Mandatory);
            m_pPartitioner = (m_pPartitioner == NULL) ? new Partitioner(this) : m_pPartitioner;
            if (!m_pPartitioner->Partition()) {
                    PTask::Runtime::HandleError("%s::%s Partitioner->Partition failed.\n"
                                                __FILE__, 
                                                __FUNCTION__);
            }
            ExplicitPartition();      
            return;


        default:
            PTask::Runtime::HandleError("%s::%s Unknown partitioning mode %d.\n", 
                                        __FILE__, 
                                        __FUNCTION__, 
                                        m_ePartitioningMode);
            break;
        #pragma warning(pop)
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the thread operation. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///
    /// <param name="lpRoutine">    The routine. </param>
    /// <param name="lpParameter">  The parameter. </param>
    ///
    /// <returns>   The handle of the. </returns>
    ///-------------------------------------------------------------------------------------------------

    HANDLE __stdcall 
    Graph::LaunchThread(
        __in LPTHREAD_START_ROUTINE lpRoutine, 
        __in GRAPHRUNNERDESC * lpParameter
        )
    {
        BOOL bPooledThread = FALSE;
        HANDLE hThread = INVALID_HANDLE_VALUE;
        if(Runtime::HasGlobalThreadPool()) {

            // try to get a thread from the global thread pool. Note that we must request that we get the
            // thread in a "suspended" state to avoid a race configuring state that allows us to detect
            // whether the thread has been primed. The thread will check the m_lpvhTaskThreads member to
            // see if the thread is from the global pool to determine whether thread 'priming' is necessary
            // or can be elided. 
            
            hThread = ThreadPool::RequestThread(lpRoutine, lpParameter, FALSE);
            if(hThread != NULL && hThread != INVALID_HANDLE_VALUE) {
                bPooledThread = TRUE;
                m_lpvhTaskThreads[hThread] = bPooledThread;
                lpParameter->hGraphRunnerThread = hThread;
                lpParameter->bIsPooledThread = bPooledThread;
                ThreadPool::StartThread(hThread);
                return hThread;
            }
        }

        // if we got here, global thread pooling is either disabled, or we exhausted it's resources and
        // it is  unable to grow, either because there are too many threads, or it is configured with a
        // static maximum. Start our own thread directly then. Again, note that we must create the
        // thread in a suspended state to avoid a race on the m_lpvhTaskThreads map. See comment above
        // for more thorough explanation. 
                
        if(hThread == NULL || hThread == INVALID_HANDLE_VALUE) {
            hThread = CreateThread(NULL, 0, lpRoutine, lpParameter, CREATE_SUSPENDED, NULL);
            if(hThread != NULL && hThread != INVALID_HANDLE_VALUE) {
                bPooledThread = FALSE;
                m_lpvhTaskThreads[hThread] = bPooledThread;
                lpParameter->hGraphRunnerThread = hThread;
                lpParameter->bIsPooledThread = bPooledThread;
                ResumeThread(hThread);
                return hThread;
            }
        }

        // we really shouldn't get here--if we can't get a thread from either the global
        // thread pool or by starting one with the Win32 API, something truly hideous has gone
        // down. Complain and declare the situation unrecoverable. 
        assert(hThread != INVALID_HANDLE_VALUE && hThread != NULL);
        if(hThread == INVALID_HANDLE_VALUE && hThread == NULL) {
            PTask::Runtime::HandleError("%s::%s: fatal: failed to launch graphrunner thread!\n",
                                        __FILE__,
                                        __FUNCTION__);
        }
        return hThread;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Starts the graph runner threads for the single-thread mode. </summary>
    ///
    /// <remarks>   Crossbac, 3/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::LaunchGraphRunnerThreadsST(
        VOID
        )
    {
        // spawn a single thread to manage all vertices. Use
        // a task queue instead of event waits to decide which
        // task to execute next. 
        int i=0;
        m_bUseReadyQ = TRUE;
        std::map<std::string, Task*>::iterator vi;
        if(m_lpvhTaskThreads.size() == 0) {

            // create a single manual reset event for
            // signalling new tasks in the ready q.
            assert(m_phReadyQueue == NULL);
            m_phReadyQueue = new HANDLE[1];
            m_phReadyQueue[0] = CreateEvent(NULL, TRUE, FALSE, NULL);
            PTASKERTHREADDESC_ST pTaskDescs = new TASKERTHREADDESC_ST();
            pTaskDescs->pGraph = this;
            pTaskDescs->ppTasks = new Task*[m_vTasks.size()];
            pTaskDescs->nTasks = (int) m_vTasks.size();
            pTaskDescs->hGraphRunningEvent = m_hGraphRunningEvent;
            pTaskDescs->hGraphTeardownEvent = m_hGraphTeardownEvent;
            pTaskDescs->hGraphStopEvent = m_hGraphStopEvent;
            pTaskDescs->hRuntimeTerminateEvent = m_hRuntimeTerminateEvent;
            pTaskDescs->pGraphState = &m_eState;
            pTaskDescs->nThreadIdx = 0;
            pTaskDescs->nGraphRunnerThreadCount = 1;
            pTaskDescs->hReadyQ = m_phReadyQueue[0];
            for(i=0, vi=m_vTasks.begin(); vi!=m_vTasks.end(); vi++, i++) 
                pTaskDescs->ppTasks[i] = vi->second;
			m_uiNascentGraphRunnerThreads = 1;
            LaunchThread(Graph::GraphRunnerProcSTQ, pTaskDescs);
			WaitGraphRunnerThreadsAlive();

        } else {
            assert(m_lpvhTaskThreads.size() == 1);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Starts the graph runner threads for the multi-thread mode. In this case we have
    ///             some work to do to decide how many threads to use, and what the mapping is from
    ///             threads to vertices. Generally speaking, if the graph is small a 1:1 mapping is
    ///             fastest, but since threads are not free, there is a performance cliff when the
    ///             graph gets large. We try to dump of the responsibility for handling this on the
    ///             programmer by providing switches to set the policy and to manage the threshold
    ///             at which we will change from 1:1 to 1:N. Choose a mapping based on the state
    ///             of these switches and launch the thread pool.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::LaunchGraphRunnerThreadsMT(
        VOID
        )
    {
        // the graph has already been constructed and has been
        // in the running state. we can't change the mapping now!
        if(m_lpvhTaskThreads.size() != 0) 
            return; 

        UINT uiActualTaskPoolSize = 0;
        UINT uiTasks = static_cast<UINT>(m_vTasks.size());
        UINT uiRequestedTaskPoolSize = PTask::Runtime::GetTaskThreadPoolSize();
        UINT uiPolicyThreshold = PTask::Runtime::GetSchedulerThreadPerTaskThreshold();
        THREADPOOLPOLICY ePolicy = PTask::Runtime::GetTaskThreadPoolPolicy() ;

        if(ePolicy == TPP_THREADPERTASK) {

            // force to use 1:1 policy 
            // no matter what the graph size
            m_bUseReadyQ = FALSE;

        } else if(ePolicy == TPP_AUTOMATIC) {

            // the policy is set to be determined automatically:
            // we are responsible for choosing the number of threads.
            // If the number of tasks is greater than the threshold, we
            // use 1:N plus a ready q. otherwise, use 1:1

            if(uiTasks > uiPolicyThreshold) {
                uiActualTaskPoolSize = uiRequestedTaskPoolSize;
                m_bUseReadyQ = TRUE;
            } else {
                uiActualTaskPoolSize = uiTasks; // 1:1
                m_bUseReadyQ = FALSE;
            }

        } else if(ePolicy == TPP_EXPLICIT) {

            // the user has wants explicit control. This means that
            // the requested thread pool size *is* the thread pool size,
            // and we will always use the ready queue even if the pool is
            // larger than the number of tasks.
            
            assert(uiRequestedTaskPoolSize != 0);
            uiActualTaskPoolSize = uiRequestedTaskPoolSize;
            m_bUseReadyQ = TRUE;
        }

        if(m_bUseReadyQ) {

            // N:M mapping of threads -> tasks. Use a ready queue to
            // multiplex the graphrunnerproc threads across the graph vertices.
            // create per-thread ready q events, which enables the arrival
            // of new tasks on the ready q to select a single graph runner proc,
            // hopefully thereby avoiding major pile-ons when work is available. 
            
            int i;
            assert(m_phReadyQueue == NULL);
            if(PTask::Runtime::GetThreadPoolSignalPerThread()) {
                m_phReadyQueue = new HANDLE[uiActualTaskPoolSize];
                for(UINT ui=0; ui<uiActualTaskPoolSize; ui++) {
                    m_phReadyQueue[ui] = CreateEvent(NULL, FALSE, FALSE, NULL);
                }
            } else {
                m_phReadyQueue = new HANDLE[1];
                m_phReadyQueue[0] = CreateEvent(NULL, TRUE, FALSE, NULL);
            }
            std::map<std::string, Task*>::iterator vi;
			m_uiNascentGraphRunnerThreads = uiActualTaskPoolSize;
            for(UINT ui=0; ui<uiActualTaskPoolSize; ui++) {
                PTASKERTHREADDESC_ST pTaskDescs = new TASKERTHREADDESC_ST();
                pTaskDescs->pGraph = this;
                pTaskDescs->ppTasks = new Task*[m_vTasks.size()];
                pTaskDescs->nTasks = (int) m_vTasks.size();
                pTaskDescs->hGraphRunningEvent = m_hGraphRunningEvent;
                pTaskDescs->hGraphTeardownEvent = m_hGraphTeardownEvent;
                pTaskDescs->hGraphStopEvent = m_hGraphStopEvent;
                pTaskDescs->hRuntimeTerminateEvent = m_hRuntimeTerminateEvent;
                pTaskDescs->pGraphState = &m_eState;
                pTaskDescs->nThreadIdx = ui;
                pTaskDescs->nGraphRunnerThreadCount = uiActualTaskPoolSize;
                pTaskDescs->hReadyQ = PTask::Runtime::GetThreadPoolSignalPerThread() ? m_phReadyQueue[ui] : m_phReadyQueue[0];
                for(i=0, vi=m_vTasks.begin(); vi!=m_vTasks.end(); vi++, i++) 
                    pTaskDescs->ppTasks[i] = vi->second;
                HANDLE handle = LaunchThread(Graph::GraphRunnerProcMTQ, pTaskDescs);
                SetThreadPriority(handle, THREAD_PRIORITY_HIGHEST);
            }
			WaitGraphRunnerThreadsAlive();

        } else {

            // 1:1 mapping of threads -> tasks. No Ready Q required!
            m_bUseReadyQ = FALSE;
            std::map<std::string, Task*>::iterator vi;
            if(m_lpvhTaskThreads.size() == 0) {
                // never been run before...create enough threads
                // for every ptask, set the running event, and we're off!
                m_uiNascentGraphRunnerThreads = (UINT) m_vTasks.size();
                UINT uiRoleIndex = 0;
                for(vi=m_vTasks.begin(); vi!=m_vTasks.end(); vi++, uiRoleIndex++) {
                    PTASKERTHREADDESC pDesc = new TASKERTHREADDESC();
                    pDesc->pGraph = this;
                    pDesc->pTask = vi->second;
                    pDesc->nThreadIdx = uiRoleIndex;
                    pDesc->nGraphRunnerThreadCount = static_cast<UINT>(m_vTasks.size());
                    pDesc->hGraphRunningEvent = m_hGraphRunningEvent;
                    pDesc->hGraphStopEvent = m_hGraphStopEvent;
                    pDesc->hGraphTeardownEvent = m_hGraphTeardownEvent;
                    pDesc->hRuntimeTerminateEvent = m_hRuntimeTerminateEvent;
                    pDesc->pGraphState = &m_eState;
                    LaunchThread(Graph::GraphRunnerProc, pDesc);
                }
				WaitGraphRunnerThreadsAlive();
            } else {
                assert(m_lpvhTaskThreads.size() == m_vTasks.size());
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Run the graph. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="bSingleThreaded">  (optional) single thread flag. If true, the runtime will use
    ///                                 a single thread to manage all tasks in the graph. Otherwise,
    ///                                 the runtime will use a thread per task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::Run(
        BOOL bSingleThreaded
        ) 
    {
        MARKEVENT(L"graph->Run");
        Lock();
        m_bSingleThreaded = bSingleThreaded;
        Finalize();      
        if(m_eState == PTGS_INITIALIZING) {
            if(PTask::Runtime::GetUseGraphMonitorWatchdog()) {
                PTask::Runtime::Inform("Starting graph monitor thread...\n");
                m_hMonitorProc = CreateThread(NULL, NULL, Graph::GraphMonitorProc, this, NULL, 0);
                Scheduler::UpdateLastDispatchTimestamp();
            }
            if(bSingleThreaded) {
                LaunchGraphRunnerThreadsST();
            } else {
                LaunchGraphRunnerThreadsMT();
            }
            m_eState = PTGS_RUNNABLE;
            m_eState = PTGS_RUNNING;
            Scheduler::NotifyGraphRunning(this);
        } else {
            assert(m_eState == PTGS_RUNNABLE);
            m_eState = PTGS_RUNNING;
            Scheduler::EndGraphQuiescence(this);
        }
        m_bEverRan = TRUE;
        ResetEvent(m_hGraphStopEvent);
        SetEvent(m_hGraphRunningEvent);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the current state of this graph. You must hold a lock on this graph
    ///             to call this function.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   The state. </returns>
    ///-------------------------------------------------------------------------------------------------

    GRAPHSTATE 
    Graph::GetState(
        VOID
        )
    {
        assert(LockIsHeld());
        return m_eState;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Stops this graph. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::Stop(
        VOID
        )
    {
        Lock();
        __Stop();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Stops this graph. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::__Stop(
        VOID
        )
    {
        assert(LockIsHeld());
        if(__IsRunning()) {
            BOOL bWorkersQuiesced = FALSE;
            m_eState = PTGS_QUIESCING;
            SetEvent(m_hGraphStopEvent);
            ResetEvent(m_hGraphRunningEvent);
            do {
                bWorkersQuiesced = (m_uiOutstandingExecuteThreads == 0) && 
                                   (m_uiThreadsAwaitingRunnableGraph >= m_lpvhTaskThreads.size());
                if(!bWorkersQuiesced) 
                    Sleep(10);
            } while(!bWorkersQuiesced);
            Scheduler::QuiesceGraph(this, m_hGraphQuiescentEvent);
            m_eState = PTGS_RUNNABLE;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resets a graph to its initial state. </summary>
    ///
    /// <remarks>   Crossbac, 5/2/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::Reset(
        VOID
        )
    {
        MARKRANGEENTER(L"Reset");
        Lock();
        EnterCriticalSection(&m_csWaitingThreads);
        m_vWaitingThreads.clear();
        m_vOutstandingNotifications.clear();
        m_vOutstandingTasks.clear();
        m_vDeferredTasks.clear();
        m_uiReadyQSurplus = 0;
        LeaveCriticalSection(&m_csWaitingThreads);
        EnterCriticalSection(&m_csReadyQ);
        m_vReadyQ.clear();
        m_vReadySet.clear();
        LeaveCriticalSection(&m_csReadyQ);

        map<std::string, Task*>::iterator ti;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {

            std::map<UINT, Port*>::iterator pi;
            Task * pTask = ti->second;
            assert(pTask != NULL);

            // construct a list of all accelerators that may be bound to dispatch resources associated with
            // this task. This includes all accelerators of the tasks accelerator class, as well as all
            // accelerators matching all classes on ports with dependent accelerator bindings. 

            std::map<UINT, Port*>* pIPorts = pTask->GetInputPortMap();
            std::map<UINT, Port*>* pMPorts = pTask->GetMetaPortMap();
            std::map<UINT, Port*>* pCPorts = pTask->GetConstantPortMap();
            std::map<UINT, Port*>* pOPorts = pTask->GetOutputPortMap();
            for(pi=pIPorts->begin(); pi!=pIPorts->end(); pi++) 
                pi->second->Reset();
            for(pi=pMPorts->begin(); pi!=pMPorts->end(); pi++) 
                pi->second->Reset();
            for(pi=pCPorts->begin(); pi!=pCPorts->end(); pi++) 
                pi->second->Reset();
            for(pi=pOPorts->begin(); pi!=pOPorts->end(); pi++) 
                pi->second->Reset();
            
            pTask->Reset();
        }

        std::map<std::string, Channel*>::iterator ci;
        for(ci=m_vChannels.begin(); ci!=m_vChannels.end(); ci++) 
            ci->second->Reset();

        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Teardown this graph. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::Teardown(
        VOID
        )
    {
        Lock(); 
        if(__IsTearingDown() || __IsTorndown()) {
            Unlock();
            return;
        }
        if(__IsRunning()) {
            PTask::Runtime::Warning("XXX: Teardown called on running graph...forcing stop\n");
            Stop();
        }
        assert(m_eState == PTGS_INITIALIZING || m_eState == PTGS_RUNNABLE);
        recordTeardownStart();

        EnterCriticalSection(&m_csReadyQ);
        EnterCriticalSection(&m_csWaitingThreads);
        m_eState = PTGS_TEARINGDOWN;
        Scheduler::AbandonDispatches(this);
        Scheduler::NotifyGraphTeardown(this);
        SetEvent(m_hGraphTeardownEvent);
        LeaveCriticalSection(&m_csWaitingThreads);
        LeaveCriticalSection(&m_csReadyQ);

        if(m_bEverRan) {
            std::vector<HANDLE> lpvWaitThreads;
            std::map<HANDLE, BOOL>::iterator mi;
            for(mi=m_lpvhTaskThreads.begin(); mi!=m_lpvhTaskThreads.end(); mi++) 
                if(!mi->second) lpvWaitThreads.push_back(mi->first);
            Unlock();
            WaitForMultipleObjects((DWORD)lpvWaitThreads.size(), lpvWaitThreads.data(), TRUE, INFINITE);
            Lock();
            for(unsigned int i=0; i<lpvWaitThreads.size(); i++) {
                CloseHandle(lpvWaitThreads[i]); 
            }
            m_lpvhTaskThreads.clear();
        }
        ReleaseBlockPools();
        map<std::string, Channel*>::iterator ci;
        vector<Channel*> liveChannels;
        for(ci=m_vChannels.begin(); ci!=m_vChannels.end(); ci++) {
            Channel * pChannel = ci->second;
            pChannel->Drain();
            if(m_pChannelDstMap.find(pChannel) != m_pChannelDstMap.end()) {
                m_pChannelDstMap.erase(pChannel);
            }
            if(m_pChannelSrcMap.find(pChannel) != m_pChannelSrcMap.end()) {
                m_pChannelSrcMap.erase(pChannel);
            }
            if(pChannel->Release()) 
                liveChannels.push_back(ci->second);
        }
        m_vChannels.clear();
        map<Port*, Task*>::iterator pi;
        for(pi=m_pPortMap.begin(); pi!=m_pPortMap.end(); pi++) {
            // before deleting ports, first release blocks
            // owned by any output ports. This will ensure that
            // owned blocks will be returned to their pools before
            // their pool owners are de-allocated. 
            Port * pPort = pi->first;
            if(pPort->GetPortType() == INPUT_PORT) {
                InputPort* pIPort = dynamic_cast<InputPort*>(pPort);
                pIPort->ReleaseReplayableBlock();
            }
            if(pPort->GetPortType() == OUTPUT_PORT) {
                OutputPort* pOPort = (OutputPort*) pPort;
                pOPort->ReleaseDatablock();
            }
        }
        for(pi=m_pPortMap.begin(); pi!=m_pPortMap.end(); pi++) {
            Port * pPort = pi->first;
            delete pPort;
        }
        m_pPortMap.clear();
        // live channels from above should have been 
        // released by the port destructors!
        // vector<Channel*>::iterator vi;
        // for(vi=liveChannels.begin(); vi!=liveChannels.end(); ci++) {
        //    (*vi)->Release();
        // }
        map<std::string, Task*>::iterator ti;
        vector<std::string> deadTaskKeys;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {
            Task * pTask = ti->second;
            pTask->Shutdown();
            Scheduler::AbandonDispatches(pTask);
        }
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {
            Task * pTask = ti->second;
            if(pTask) {
                if(0==pTask->Release()) {
                    deadTaskKeys.push_back(ti->first);
                }
            }
        }
        for(vector<std::string>::iterator si=deadTaskKeys.begin();
            si!=deadTaskKeys.end(); si++) {
            m_vTasks.erase(*si);
        }
        Accelerator::SynchronizeDefaultContexts();
        m_eState = PTGS_TEARDOWNCOMPLETE;
        BlockPoolOwner::RetireGraph(this);
        SetEvent(m_hGraphTeardownComplete);
        Unlock();
        recordTeardownLatency();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Forcible teardown for this graph initiated by the scheduler. 
    ///             If the scheduler is exiting before the graph is deleted, we 
    ///             need to get this graph in a quiescent state and do everything
    ///             a Stop/Teardown/dtor would do with the caveat that we must leave the graph
    ///             in a state where a dtor call (typically initiated well after the
    ///             runtime exits by a GC sweep in the managed wrapper) can still delete 
    ///             the object cleanly.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::ForceTeardown(
        VOID
        )
    {
        // this is called by the scheduler *after* it has done a global
        // quiesce, so we don't need to do any work to quiesce.
        Lock(); 

        // if the graph is already torndown, it's sufficient to mark
        // this graph as forced so that we don't actually run the dtor
        EnterCriticalSection(&m_csReadyQ);
        EnterCriticalSection(&m_csWaitingThreads);
        BOOL bTornDown = (m_bForceTeardown || __IsTorndown() || m_bForcedTeardownComplete);
        m_bForcedTeardownComplete = FALSE;
        m_bForceTeardown = TRUE;
        if(!bTornDown) {
            m_eState = PTGS_TEARINGDOWN;
            SetEvent(m_hGraphTeardownEvent);
        }
        LeaveCriticalSection(&m_csWaitingThreads);
        LeaveCriticalSection(&m_csReadyQ);
        if(bTornDown) {
            Unlock();
            return;
        }

        recordTeardownStart();
        if(m_bEverRan) {
            std::vector<HANDLE> lpvTaskThreads;
            std::map<HANDLE, BOOL>::iterator mi;
            for(mi=m_lpvhTaskThreads.begin(); mi!=m_lpvhTaskThreads.end(); mi++) 
                if(!mi->second) 
                    lpvTaskThreads.push_back(mi->first);
            Unlock();
            WaitForMultipleObjects((DWORD)lpvTaskThreads.size(), 
                                   lpvTaskThreads.data(), 
                                   TRUE, INFINITE);
            Lock();
            for(unsigned int i=0; i<lpvTaskThreads.size(); i++) {
                CloseHandle(lpvTaskThreads[i]); 
                lpvTaskThreads[i] = NULL;
            }
            m_lpvhTaskThreads.clear();
        }
        ReleaseBlockPools();
        map<std::string, Channel*>::iterator ci;
        vector<Channel*> liveChannels;
        for(ci=m_vChannels.begin(); ci!=m_vChannels.end(); ci++) {
            Channel * pChannel = ci->second;
            pChannel->Drain();
            if(m_pChannelDstMap.find(pChannel) != m_pChannelDstMap.end()) {
                m_pChannelDstMap.erase(pChannel);
            }
            if(m_pChannelSrcMap.find(pChannel) != m_pChannelSrcMap.end()) {
                m_pChannelSrcMap.erase(pChannel);
            }
            if(pChannel->Release()) 
                liveChannels.push_back(ci->second);
        }
        m_vChannels.clear();
        map<Port*, Task*>::iterator pi;
        for(pi=m_pPortMap.begin(); pi!=m_pPortMap.end(); pi++) {
            // before deleting ports, first release blocks
            // owned by any output ports. This will ensure that
            // owned blocks will be returned to their pools before
            // their pool owners are de-allocated. 
            Port * pPort = pi->first;
            if(pPort->GetPortType() == INPUT_PORT) {
                InputPort* pIPort = dynamic_cast<InputPort*>(pPort);
                pIPort->ReleaseReplayableBlock();
            }
            if(pPort->GetPortType() == OUTPUT_PORT) {
                OutputPort* pOPort = (OutputPort*) pPort;
                pOPort->ReleaseDatablock();
            }
        }
        for(pi=m_pPortMap.begin(); pi!=m_pPortMap.end(); pi++) {
            Port * pPort = pi->first;
            delete pPort;
        }
        m_pPortMap.clear();
        // live channels from above should have been 
        // released by the port destructors!
        // vector<Channel*>::iterator vi;
        // for(vi=liveChannels.begin(); vi!=liveChannels.end(); ci++) {
        //    (*vi)->Release();
        // }
        map<std::string, Task*>::iterator ti;
        vector<std::string> deadTaskKeys;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {
            Task * pTask = ti->second;
            pTask->Shutdown();
            Scheduler::AbandonDispatches(pTask);
        }
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {
            Task * pTask = ti->second;
            if(pTask) {
                if(0==pTask->Release()) {
                    deadTaskKeys.push_back(ti->first);
                }
            }
        }
        for(vector<std::string>::iterator si=deadTaskKeys.begin();
            si!=deadTaskKeys.end(); si++) {
            m_vTasks.erase(*si);
        }
        Accelerator::SynchronizeDefaultContexts();
        m_eState = PTGS_TEARDOWNCOMPLETE;
        BlockPoolOwner::RetireGraph(this);
        SetEvent(m_hGraphTeardownComplete);
        Unlock();
        recordTeardownLatency();
        Destroy();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a channel. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="lpszChannelName">  (optional) [in] If non-null, name of the channel. </param>
    ///
    /// <returns>   null if it fails, else the channel. </returns>
    ///-------------------------------------------------------------------------------------------------

    Channel * 
    Graph::GetChannel(
        char * lpszChannelName
        )
    {
        Channel * pChannel = NULL;
        Lock();
        map<std::string, Channel*>::iterator mi = m_vChannels.find(lpszChannelName);
        if(mi != m_vChannels.end()) 
            pChannel = mi->second;
        Unlock();
        return pChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Traverse a (supposedly) quiescent graph looking for sticky blocks that might
    /// 			get reused when the graph becomes non-quiescent. </summary>
    ///
    /// <remarks>   Crossbac, 3/2/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::ReleaseStickyBlocks(
        VOID
        )
    {
        std::map<std::string, Task*>::iterator ti;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {
            Task * pTask = ti->second;
            pTask->ReleaseStickyBlocks();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Drain channels. </summary>
    ///
    /// <remarks>   Crossbac, 3/5/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::DrainChannels(
        VOID
        )
    {
        std::map<std::string, Channel*>::iterator ci;
        int nInitializers = 0;
        for(ci=m_vChannels.begin(); ci!=m_vChannels.end(); ci++) {
            Channel * pChannel = ci->second;
            pChannel->Lock();
            if(pChannel->GetQueueDepth() > 0) {
                std::cout << pChannel << " has non-zero queue depth...draining" << std::endl;
                pChannel->Drain();
                assert(pChannel->GetQueueDepth() != 0);
            }
            if(pChannel->GetType() == CT_INITIALIZER) {
                nInitializers++;
                pChannel->SetPropagatedControlSignal(DBCTLC_ENDITERATION);
            }
            pChannel->Unlock();
        }
        std::cout << "encountered " << nInitializers << " initializer channels" << std::endl;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notify the graph that an event has occurred that can affect the ready state of a
    ///             task. Whether or not the graph needs to respond depends on the thread-task
    ///             mapping policy for the graph. If the graph is using 1:1 mapping, then no action
    ///             is necessary: the graph runner procs will respond to the port status signals. In
    ///             all other modes, the graph runner threads share a queue of tasks that *may* be
    ///             ready. Those threads take from the front of that queue and attempt to dispatch.
    ///             Avoiding queuing for tasks that are not ready is consequently worth some effort,
    ///             but the ready check is expensive so we make conservate estimate instead. The
    ///             bReadyStateKnown flag provides a way for the caller to override that check. A
    ///             TRUE bReadyStateKnown flag means we know it can be dispatched, so enqueue it
    ///             without further ado.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/28/2013. </remarks>
    ///
    /// <param name="pTask">            [in,out] If non-null, the task. </param>
    /// <param name="bReadyStateKnown"> TRUE if the ready state is known, FALSE if
    ///                                 a check is required before enqueueing the task. 
    ///                                 </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::SignalReadyStateChange(
        __in Task * pTask,
        __in BOOL bReadyStateKnown
        )
    {
        if(m_bUseReadyQ) {

            // avoid clogging the ready queue with tasks that are not really ready. The EstimateReadyStatus
            // returns a conservative estimate without acquiring locks, meaning it will return FALSE only
            // when the task is *definitely* not ready. If the bReadyStateKnown flag is set we 
            // can skip this check because the caller already knows the task is ready.
            
            if(!bReadyStateKnown && !pTask->EstimateReadyStatus())
                return; 

            EnterCriticalSection(&m_csReadyQ);
            EnterCriticalSection(&m_csWaitingThreads);

            if(!m_bSingleThreaded && m_vOutstandingTasks.find(pTask) != m_vOutstandingTasks.end()) {

                // if we have multiple graph runner threads, it is possible
                // that this task is already in flight for a dispatch attempt. 
                // if that is the case, insert it in the deferred list. 
                // TODO: this is actually unnecessary book-keeping that helps with
                // debugging by giving us a sanity check on tasks that get signaled
                // after they've dispatched. In the future we should stop tracking this
                
                m_vDeferredTasks.insert(pTask);

            } else if(m_vReadySet.find(pTask) == m_vReadySet.end()) {

                // if the task is *not* already in the ready Q, insert it
                // in both the ready set and the queue. The Q enables a quick
                // hash-based check for presence of the task, while the queue
                // enables us to sort based on policy without impacting the
                // performance of a check for presence in the Q.

                m_vReadySet.insert(pTask);
                m_vReadyQ.push_back(pTask);
                assert(m_vReadySet.size() > 0);
                assert(m_vReadyQ.size() > 0);
                SignalTasksAvailable();

            } else {

                // if we *do* have it queued, move it to the back of the queue, since we *know* it wasn't ready
                // when we queued it (otherwise the event that triggered this call wouldn't have been raised).
                // we'd prefer that the queue track FIFO order for *ready* tasks, so it's previous entry into
                // the queue was not a real "first in" event since it arrived early. 
                
                std::deque<Task*>::iterator qpos = find(m_vReadyQ.begin(), m_vReadyQ.end(), pTask);
                assert(qpos != m_vReadyQ.end());
                m_vReadyQ.erase(qpos);
                m_vReadyQ.push_back(pTask);
                SignalTasksAvailable();

            }   

            LeaveCriticalSection(&m_csWaitingThreads);
            LeaveCriticalSection(&m_csReadyQ);

        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   No tasks are available: empty ready queue. Reset task queue events
    ///             if the runtime and graph modes dictate this must be done (the task queue
    ///             events are manually managed in some modes, and automatic in others).
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::SignalTaskQueueEmpty(
        VOID
        )
    {
        // we have a wait object per thread by default when not in single threaded graph run mode. This
        // can be overridden with a runtime-level switch. If we are single-threaded or this override is
        // on, the event is a manual reset, so set it. 
        if(!m_bUseReadyQ) return;
        if(m_bSingleThreaded || !PTask::Runtime::GetThreadPoolSignalPerThread()) {
            assert(m_vReadyQ.size() == 0);
            if(m_vReadyQ.size() == 0) {
                ResetEvent(m_phReadyQueue[0]);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Raises the Tasks available signal: notify the graph runner thread(s) that
    ///             there is something on the ready q. In single-thread mode it is sufficient
    ///             to set a single event that the graph-runner proc waits on. In multi-threaded
    ///             mode, the implementation is more nuanced, since we don't want a large number
    ///             of threads to wake up and charge pell-mell to contend for the run queue. 
    ///             When we are using the run queue with multiple threads, we maintain a list
    ///             of waiting threads, and choose randomly amongst them which thread to wake up. 
    ///             That thread is resposible for dequeueing something from the run queue and removing
    ///             itself from the waiter list as efficiently as possible to minimize contention.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::SignalTasksAvailable(
        VOID
        )
    {
        assert(m_bUseReadyQ);
        assert(m_vReadyQ.size() > 0);
        if(m_bSingleThreaded) {
            SetEvent(m_phReadyQueue[0]);
        } else {
            if(PTask::Runtime::GetThreadPoolSignalPerThread()) {
                EnterCriticalSection(&m_csWaitingThreads);
                if(m_vWaitingThreads.size() == 0) {
                    // there are no waiting threads! we can't signal
                    // anyone, so take note of this fact and signal 
                    // later when another thread becomes available. 
                    m_uiReadyQSurplus++;
                } else {
                    UINT uiIdx = 0;
                    UINT uiWaiters = static_cast<UINT>(m_vWaitingThreads.size());
                    double dRand = static_cast<double>(rand());
                    UINT uiWaiterIndex = (UINT)((dRand/(double)RAND_MAX)*uiWaiters);
                    uiWaiterIndex = min(uiWaiterIndex, ((UINT)m_vWaitingThreads.size())-1);
                    std::set<UINT>::iterator si=m_vWaitingThreads.begin();
                    while(uiIdx++<uiWaiterIndex) si++;
                    UINT uiThreadPoolIdx = *si;
                    assert(uiThreadPoolIdx >= 0 && uiThreadPoolIdx < m_lpvhTaskThreads.size());
                    m_vWaitingThreads.erase(si);
                    m_vOutstandingNotifications.insert(uiThreadPoolIdx);
                    SetEvent(m_phReadyQueue[uiThreadPoolIdx]);
                }
                LeaveCriticalSection(&m_csWaitingThreads);
            } else {
                SetEvent(m_phReadyQueue[0]);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets quiescent event. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <returns>   The quiescent event. </returns>
    ///-------------------------------------------------------------------------------------------------

    HANDLE 
    Graph::GetQuiescentEvent(
        VOID
        )
    {
        return m_hGraphQuiescentEvent;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the graph running event object. </summary>
    ///
    /// <remarks>   Crossbac, 3/14/2013. </remarks>
    ///
    /// <returns>   The graph running event. </returns>
    ///-------------------------------------------------------------------------------------------------

    HANDLE 
    Graph::GetGraphRunningEvent(
        VOID
        )
    {
        return m_hGraphRunningEvent;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the graph stop event object. </summary>
    ///
    /// <remarks>   Crossbac, 3/14/2013. </remarks>
    ///
    /// <returns>   The graph running event. </returns>
    ///-------------------------------------------------------------------------------------------------

    HANDLE 
    Graph::GetGraphStopEvent(
        VOID
        )
    {
        return m_hGraphStopEvent;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Null to empty. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="a">    a. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    inline std::string NULL_TO_EMPTY(const char *a)
    {
        return (a) ? std::string(a) : std::string("");
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Port to name. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="port">         [in,out] If non-null, the port. </param>
    /// <param name="portNameMap">  [in,out] [in,out] If non-null, the port name map. </param>
    /// <param name="portNodeNum">  [in,out] The port node number. </param>
    /// <param name="fout">         [in,out] The fout. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    inline std::string 
    portToName(
        Port* port, 
        std::map<Port*, std::string>& portNameMap, 
        int& portNodeNum, 
        std::ofstream& fout
        )
    {
        if(portNameMap.find(port) != portNameMap.end())
        {
            return portNameMap[port];
        }
        else
        {
            char str[1024];
            sprintf_s(str, 1024, "portNode_%d", portNodeNum);
            ++portNodeNum;
            std::string portNodeName(str); 
            fout 
                << "\t" << portNodeName 
                << "[label = \"" 
                << NULL_TO_EMPTY(port->GetVariableBinding()) 
                << (port->IsSticky() ?  "*":"")
                << "\"];\n";
            portNameMap[port] = portNodeName;
            return portNodeName;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets port map adjacencies. </summary>
    ///
    /// <remarks>   Crossbac, 11/20/2013. </remarks>
    ///
    /// <param name="pTask">                    [in,out] If non-null, the task. </param>
    /// <param name="pPorts">                   [in,out] If non-null, the ports. </param>
    /// <param name="pSet">                     [in,out] If non-null, the set. </param>
    /// <param name="bIgnoreExposedChannels">   The ignore exposed channels. </param>
    ///
    /// <returns>   The number of adjacencies induced by the given port map. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    Graph::GetPortMapAdjacencies(
        __in    Task * pTask,
        __in    BOOL bTaskIsSrc,
        __in    std::map<UINT, Port*>* pPorts,
        __inout std::set<Task*>* pSet,
        __in    BOOL bIgnoreExposedChannels
        )
    {
        std::set<Channel*> channels;
        std::set<Channel*>::iterator si;
        std::map<UINT, Port*>::iterator pi;
        UINT uiInitialTaskCount = (UINT)pSet->size();

        for(pi=pPorts->begin(); pi!=pPorts->end(); pi++) {
            Port * pPort = pi->second;
            UINT uiChannelCount = pPort->GetChannelCount(); 
            UINT uiControlChannelCount = pPort->GetControlChannelCount(); 
            for(UINT i=0; i<uiChannelCount; i++) 
                channels.insert(pPort->GetChannel(i));
            for(UINT i=0; i<uiControlChannelCount; i++) 
                channels.insert(pPort->GetControlChannel(i));
        }

        for(si=channels.begin(); si!=channels.end(); si++) {
            Channel * pChannel = *si;
            CHANNELENDPOINTTYPE eOtherType = bTaskIsSrc ? CE_DST : CE_SRC;
            Port * pOtherPort = pChannel->GetBoundPort(eOtherType);
            BOOL bExposedChannel = pOtherPort == NULL;
            BOOL bSelfCycle = pOtherPort && pOtherPort->GetTask() == pTask;
            if((bExposedChannel && bIgnoreExposedChannels) || bSelfCycle)
                continue;
            pSet->insert(pChannel->GetBoundPort(eOtherType)->GetTask());
        }

        UINT uiPostTaskCount = (UINT)pSet->size();
        UINT uiTaskCount = uiPostTaskCount - uiInitialTaskCount;
        return uiTaskCount;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a map from task to the channels that express adjacency according to the flag
    ///             parameters. Exposed channels never express true adjacency, but since exposed
    ///             channels always represent a potential transition across memory spaces, it may be
    ///             important to include them somehow in the weight model. The caller must free the
    ///             result.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 11/20/2013. </remarks>
    ///
    /// <param name="pMap">                     [in,out] [in,out] If non-null, the map. </param>
    /// <param name="bIgnoreExposedChannels">   The ignore exposed channels. </param>
    ///
    /// <returns>   The forward channel count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    Graph::GetAdjacencyMap(
        __out std::map<Task*, std::set<Task*>*>* &pMap,
        __in BOOL bIgnoreExposedChannels
        )
    {
        UINT nChannels = 0;
        std::map<std::string, Task*>::iterator iter;
        pMap = new std::map<Task*, std::set<Task*>*>();
            
        for(iter=m_vTasks.begin(); iter!=m_vTasks.end() ; ++iter) {
            Task * pTask = iter->second;
            std::set<Task*>* pSet = new std::set<Task*>();
            nChannels += GetPortMapAdjacencies(pTask, FALSE, pTask->GetInputPortMap(), pSet, bIgnoreExposedChannels);
            nChannels += GetPortMapAdjacencies(pTask, TRUE, pTask->GetOutputPortMap(), pSet, bIgnoreExposedChannels);
            nChannels += GetPortMapAdjacencies(pTask, FALSE, pTask->GetMetaPortMap(), pSet, bIgnoreExposedChannels);
            (*pMap)[pTask] = pSet;
        }

        return nChannels;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Free adjacency map. </summary>
    ///
    /// <remarks>   Crossbac, 11/20/2013. </remarks>
    ///
    /// <param name="pMap"> [in,out] If non-null, the first parameter. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::FreeAdjacencyMap(
        __inout std::map<Task*, std::set<Task*>*>* pMap
        )
    {
        if(pMap == NULL)
            return;
        std::map<Task*, std::set<Task*>*>::iterator mi;
        for(mi=pMap->begin(); mi!=pMap->end(); mi++) {
            delete mi->second;
        }
        delete pMap;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Writes a file that describes the vertex and edge weights in the graph to be used
    ///             as input to Renato et al's graph partitioning algorithm. Format described below.
    ///
    ///             Currently the weight for each vertex is taken from the partition hint associated with
    ///             the corresponding task (set via Task::SetSchedulerPartitionHint). Every task in the
    ///             graph must have a hint associated with it or this method will raise an error.
    ///
    ///             Currently the edge weight is the same for all edges, and defaults to 1. 
    ///
    ///
    ///             Experimental code. File format:
    ///             Assume you have a graph with n nodes and m edges. Nodes must have ids from 1 to n.
    ///             
    ///             The first line just specifies the graph size and the format (weights on vertices
    ///             and costs on edges): <n> <m> 11
    ///             (The 11 is magic sauce. This one really does go to 11.)
    ///             
    ///             This is followed by n lines, each describing the adjacency list of one vertex (in
    ///             order). Each line is of the form: <vertex_weight> <v1> <ew1> <v2> <ew2> ... <vk> <ewk>
    ///             
    ///             For example, if the v-th line is “7 17 3 15 4 3 8”, then you know that vertex v
    ///             has weight 7 and has three outgoing edges: (v,17), (v,15), and (v,3). These edges
    ///             cost 3, 4, and 8, respectively.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 11/20/2013. </remarks>
    ///
    /// <param name="filename">                 [in] Filename of the file to write the model to. </param>
    /// <param name="edgeWeight">               [in] Weight to apply to each edge. </param>
    ///
    /// <returns>   The number of vertices in the model. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Graph::WriteWeightedModel(
        __in char * filename,
        __in UINT edgeWeightX // TODO JC Drop this parameter?
        )
    {
        UNREFERENCED_PARAMETER(edgeWeightX);  

        std::ofstream fout(filename);

        int nodeId=0;
        std::map<Task*, int> nodeNumberMap;    
        std::map<int, Task*> nodeNumberBackMap;    
        std::map<int, Task*>::iterator ti;
        std::map<std::string, Task*>::iterator iter;
        for(iter=m_vTasks.begin(); iter!=m_vTasks.end() ; ++iter) {
            ++nodeId; // index from 1, not 0!
            Task * pTask = iter->second;
            assert(pTask != NULL);
            nodeNumberMap[pTask] = nodeId;
            nodeNumberBackMap[nodeId] = pTask;
        }

        // The first line just specifies the graph size and the format (weights on vertices and costs
        // on edges): <n> <m> 11. n == number of tasks, m = number of channels. If we are ignoring 
        // back-edges, finding m involves traversing the graph to look for cycles unfortunately.
        
        size_t nVertices = m_vTasks.size();
        std::map<Task*, std::set<Task*>*>* pMap = NULL;
        UINT nEdges = GetAdjacencyMap(pMap, TRUE);
        
        fout << nVertices << " " << nEdges << " 11" << std::endl;

        // Currently require either all tasks to have a partition hint (in which case use it as a weight)
        // or no tasks to have a partition hint (in which case give all a weight of 1).
        BOOL bAllTasksHaveHint = TRUE;
        BOOL bNoTasksHaveHint = TRUE;

        // This is followed by n lines, each describing the adjacency list of one vertex 
        // (in order). Each line is of the form:
        // <vertex_weight> <v1> <ew1> <v2> <ew2> … <vk,ewk>

        for(ti=nodeNumberBackMap.begin(); ti!=nodeNumberBackMap.end(); ti++) {

            Task * pTaskNode = ti->second;
            assert(pTaskNode != NULL);
            std::map<Task*, std::set<Task*>*>::iterator mi = pMap->find(pTaskNode);

            // vertex weight
            UINT vertexWeight = 1;
            if (pTaskNode->HasSchedulerPartitionHint())
            {
                vertexWeight = pTaskNode->GetSchedulerPartitionHint(NULL); // Not using cookies (yet).
                bNoTasksHaveHint = FALSE;
            } else {
                bAllTasksHaveHint = FALSE;
            }

            fout << vertexWeight;

            // adjacencies and per-adjacency weights
            std::set<Task*>::iterator si;
            std::set<Task*>* pTaskSet = mi->second;
            for(si=pTaskSet->begin(); si!=pTaskSet->end(); si++) {

                Task * pAdjacentTask = *si;
                assert(pAdjacentTask != NULL);
#ifdef DEBUG
                // if this task is adjacent, then pTaskNode should also be in
                // the adjacency list for the task: pAdjacentNode
                std::map<Task*, std::set<Task*>*>::iterator ni = pMap->find(pAdjacentTask);
                assert(ni != pMap->end());
                assert(ni->second != NULL);
                assert(ni->second->find(pTaskNode) != ni->second->end());
#endif
                int nAdjacentId = nodeNumberMap[pAdjacentTask];

                int adjacentVertexWeight = 0;
                if (pAdjacentTask->HasSchedulerPartitionHint())
                    adjacentVertexWeight = pAdjacentTask->GetSchedulerPartitionHint(NULL);
                int maxVertexWeight = max((int)vertexWeight, adjacentVertexWeight);

                int edgeWeight;
                switch(PTask::Runtime::GetOptimalPartitionerEdgeWeightScheme())
                {
                case 1:
                    edgeWeight=1;
                    break;
                case 2:
                    edgeWeight=maxVertexWeight;
                    break;
                case 3:
                    edgeWeight=maxVertexWeight;
                    edgeWeight*=edgeWeight;
                    break;
                case 4:
                    edgeWeight=maxVertexWeight*10;
                    edgeWeight*=edgeWeight;
                    break;
                case 5:
                    edgeWeight=maxVertexWeight*100;
                    edgeWeight*=edgeWeight;
                    break;
                case 6:
                    edgeWeight=maxVertexWeight*1000;
                    edgeWeight*=edgeWeight;
                    break;
                default:
                    printf("**** Unknown weight scheme %d. Exiting.\n", 
                        PTask::Runtime::GetOptimalPartitionerEdgeWeightScheme());
                    exit(1);
                }

                fout << " " << nAdjacentId << " " << edgeWeight;
            }
            fout << std::endl;
        }

        fout.close();
        FreeAdjacencyMap(pMap);

        if (!bAllTasksHaveHint && !bNoTasksHaveHint)
        {
            PTask::Runtime::HandleError(
                "%s::%s: CUrrently require either all or no tasks to have a partition hint to use the optimal partitioner.\n"
                "This graph has a mix of ones and ones without! Exiting...\n",
                __FILE__, __FUNCTION__);
            fout.close();
            exit(1);
        }

        return (UINT)m_vTasks.size();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Writes a dot file.  
    ///             To compile the DOT file generated by this function, you will need to download and
    ///             install graphviz : 1. Download and install the msi from
    ///             http://www.graphviz.org/Download_windows.php 2. To compile the DOT file, in a
    ///             command prompt type the following:
    ///                 dot -Tpng &lt;DOT file&gt; -o &lt;Output PNG file&gt;  
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="filename">             [in,out] If non-null, filename of the file. </param>
    /// <param name="drawPorts">            the draw ports. </param>
    /// <param name="bPresentation">        true to presentation. </param>
    /// <param name="bShowSchedulerHints">  true to show, false to hide the scheduler hints. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::WriteDOTFile(
        char * filename, 
        BOOL drawPorts,
        BOOL bPresentation,
        BOOL bShowSchedulerHints
        )
    {
        if(bPresentation) {
            WritePresentationDOTFile(filename, drawPorts);
            return;
        }
        std::ofstream fout(filename);
        fout << "digraph DFG {\n";
        std::map<Task*, int> nodeNumberMap;
        std::map<Port*, std::string> portNameMap;
        int nodeId=1, inputNodeNum=1, outputNodeNum=1, portNodeNum=1;
        int nMultiChannelId = 0;

        static const char * lppszAffinityColors[] = {
            "orange",
            "green",
            "red",
            "blue",
            "black",
            "purple"
//            "orange",
//            "green",
//            "red",
//            "yellow",
//            "purple",
//            "black"
        };
        static const char * lppszAffinityTextColors[] = {
            "black",
            "blue",
            "white",
            "red",
            "white",
            "white"
//            "black",
//            "blue",
//            "yellow",
//            "red",
//            "white",
//            "white"
        };
        static const UINT uiAffinityColors = sizeof(lppszAffinityColors) / sizeof(char*);

		BOOL showUnaffinitized = FALSE;
        // TODO JC Merge the two versions.
        if (bShowSchedulerHints)
        {
            // Also dump a file listing the task -> accelerator affinity.
            // Can diff with output from other partitioning schemes to see the 
            // delta's in assignment.
            std::string affinityFilename = filename;
            affinityFilename += ".task_affinities.txt";
            std::ofstream fout2(affinityFilename);
            for(std::map<std::string, Task*>::iterator iter=m_vTasks.begin(); iter!=m_vTasks.end() ; ++iter)
            {
                Task * pTask = iter->second;
                Accelerator * pAccelerator = pTask->GetAffinitizedAcceleratorHint();
                fout2 << pTask->GetTaskName() << "\t";
                if (pAccelerator != NULL)
                    fout2 << pAccelerator->GetAcceleratorId();
                else 
                    fout2 << "None";
                fout2 << "\n";
            }
            fout2.close();

            // Build node id map in single pass over tasks.
            for(std::map<std::string, Task*>::iterator iter=m_vTasks.begin(); iter!=m_vTasks.end() ; ++iter)
            {
                nodeNumberMap[iter->second] = nodeId++;
            }

			// Build set of accelerator ids mentioned in affinitization.
			// Add -1 to the set if any of tasks are unaffinitized.
			std::set<int> accIds;
			for(std::map<std::string, Task*>::iterator iter=m_vTasks.begin(); iter!=m_vTasks.end() ; ++iter)
            {
                Task * pTask = iter->second;
                Accelerator * pAccelerator = pTask->GetAffinitizedAcceleratorHint();
                if (pAccelerator != NULL)
				{
					accIds.insert(pAccelerator->GetAcceleratorId());
				} else {
					accIds.insert(-1);
				}
			}
			int numClusters = static_cast<int>(accIds.size());

            for (std::set<int>::iterator iter=accIds.begin(); iter!=accIds.end() ; ++iter)
            {
				int currentClusterAccelerator = *iter;
				if ((!showUnaffinitized) && (currentClusterAccelerator == -1))
				{
					printf("Skipping unaffinitized tasks...\n");
					continue;
				}
				int clusterWeight=0;
				if (numClusters > 1)
				{
					fout << "subgraph cluster_";
					if (currentClusterAccelerator == -1) 
						fout << "Unaffinitized";
					else
						fout << currentClusterAccelerator;
					fout << " {\n";
				}
                for(std::map<std::string, Task*>::iterator iter=m_vTasks.begin(); iter!=m_vTasks.end() ; ++iter)
                {
                    Task * pTask = iter->second;
                    std::string strTaskColor = (pTask->GetMetaPortMap()->size() > 0) ? "lightblue" : "blue";
                    std::string strTextColor = "white";
                    if(bShowSchedulerHints) {
                        Accelerator * pAccelerator = pTask->GetAffinitizedAcceleratorHint();
                        if (!pAccelerator)
						{
							if(currentClusterAccelerator != -1)
								continue;
						} else {
							if ((INT)(pAccelerator->GetAcceleratorId()) != currentClusterAccelerator)
								continue;
						}
                        strTaskColor = lppszAffinityColors[abs(currentClusterAccelerator) % uiAffinityColors];
                        strTextColor = lppszAffinityTextColors[abs(currentClusterAccelerator) % uiAffinityColors];
                    }
                    fout << "\t" 
                         << nodeNumberMap.find(pTask)->second // nodeId
                         << "[label = \"" 
                         << iter->first; 
                    if(bShowSchedulerHints) {
                        //fout << "__" << nodeId;
                        fout << " [" << pTask->GetSchedulerPartitionHint(NULL) << "]";
                        clusterWeight += pTask->GetSchedulerPartitionHint(NULL);
                    }
                    fout << "\", shape=box, style=filled, color="
                         << strTaskColor 
                         <<", fontcolor="
                         << strTextColor
                         << "];\n";
                }
				if (numClusters > 1)
				{
	                fout << "label = \"";
					if (currentClusterAccelerator == -1) 
						fout << "Unaffinitized";
					else
						fout << "GPU " << currentClusterAccelerator;
					fout << " W=" << clusterWeight << "\"; }\n";
				}
            }
        } else {
            for(std::map<std::string, Task*>::iterator iter=m_vTasks.begin(); iter!=m_vTasks.end() ; ++iter)
            {
                Task * pTask = iter->second;
                std::string strTaskColor = (pTask->GetMetaPortMap()->size() > 0) ? "lightblue" : "blue";
                std::string strTextColor = "white";
                if(bShowSchedulerHints) {
                    Accelerator * pAccelerator = pTask->GetAffinitizedAcceleratorHint();
                    if(pAccelerator) {
                        strTaskColor = lppszAffinityColors[pAccelerator->GetAcceleratorId() % uiAffinityColors];
                        strTextColor = lppszAffinityTextColors[pAccelerator->GetAcceleratorId() % uiAffinityColors];
                    }
                }
                fout << "\t" 
                     << nodeId 
                     << "[label = \"" 
                     << iter->first; 
                if(bShowSchedulerHints) {
                    fout << "__" << nodeId;
                }
                fout << "\", shape=box, style=filled, color="
                     << strTaskColor 
                     <<", fontcolor="
                     << strTextColor
                     << "];\n";
                nodeNumberMap[iter->second] = nodeId;
                ++nodeId;
            }
    }

        size_t i;
        map<std::string, Channel*>::iterator ci;
        for(i=0, ci=m_vChannels.begin(); ci!=m_vChannels.end(); ++i, ++ci)
        {
            bool bMultiChannel = false;
            Channel * pPrimaryChannel = ci->second;
            vector<Channel*> vComponentChannels;
            if(pPrimaryChannel->GetType() == CT_MULTI) {
                bMultiChannel = true;
                nMultiChannelId++;
                MultiChannel* pMChannel = (MultiChannel*) pPrimaryChannel;
                map<UINT, Channel*>* pmap = pMChannel->GetCoalescedChannelMap();
                map<UINT, Channel*>::iterator mmi;
                for(mmi=pmap->begin(); mmi!=pmap->end(); mmi++) {
                    vComponentChannels.push_back(mmi->second);
                }
            } else {
                vComponentChannels.push_back(pPrimaryChannel);
            }
            vector<Channel*>::iterator vci;
            for(vci=vComponentChannels.begin(); vci!=vComponentChannels.end(); vci++) {
                Channel * channel = *vci;

                //if (channel->ShouldDraw() == false) continue;
                Port *src = channel->GetBoundPort(PTask::CE_SRC);
                Port *dst = channel->GetBoundPort(PTask::CE_DST);
                Task *srcTask = NULL, *dstTask = NULL;
                std::map<Task*, int>::iterator srcIter, dstIter;
                if (src)
                {
                    srcTask = src->GetTask();
                    srcIter = nodeNumberMap.find(srcTask);
                }
                if (dst)
                {
                    dstTask = dst->GetTask();
                    dstIter = nodeNumberMap.find(dstTask);
					if (!showUnaffinitized && (dstTask->GetAffinitizedAcceleratorHint() == NULL))
					{
						continue;
					}
                }
                assert (!(src==NULL && dst==NULL));
                bool bPredicated = 
                    ((channel->GetPredicationType(CE_SRC) != CGATEFN_NONE) ||
                    (channel->GetPredicationType(CE_DST) != CGATEFN_NONE));
                bool bDescriptor = ((src == NULL) && (dst != NULL) && dst->IsDescriptorPort());
                bool bSticky = ((src == NULL) && (dst != NULL) && dst->IsSticky());
                if (src==NULL)
                {
                    //Create a new input node
                    std::string strColor = bPredicated ? "red" : 
                                           bDescriptor ? "gray77" : 
                                           bMultiChannel ? "darkorange" : 
                                           bSticky ? "blue" :
                                           "green";
                    char str[1024];
                    sprintf_s(str, 1024, "inputNode_%d", inputNodeNum);
                    ++inputNodeNum;
                    std::string inputNodeName(str); 
                    std::string channelName; 
                    if(bMultiChannel) {
                        const char * pBinding = dst->GetVariableBinding();
                        if(pBinding == NULL) {
                            sprintf_s(str, 1024, "(multi_%d)", nMultiChannelId);
                        } else {
                            sprintf_s(str, 1024, "%s_(multi_%d)", pBinding, nMultiChannelId);
                        }
                        channelName = str;
                    } else {
                        channelName = NULL_TO_EMPTY(dst->GetVariableBinding());
                    }
                    fout << "\t" << inputNodeName << "[label = \"" << channelName << "\", color="<<strColor<<"];\n";
                    fout << "\tedge [color="<<strColor<<"];\n";
                    fout << "\t" << inputNodeName << " -> " << dstIter->second << ";\n";

                }
                else if(dst==NULL)
                {
                    //Create a new output node
                    std::string strColor = bPredicated ? "red":"darkviolet";
                    char str[1024];
                    sprintf_s(str, 1024, "outputNode_%d", outputNodeNum);
                    ++outputNodeNum;
                    std::string outputNodeName(str); 
                    fout << "\t" << outputNodeName << "[label = \"" << NULL_TO_EMPTY(src->GetVariableBinding()) << "\", color="<<strColor<<"];\n";
                    fout << "\tedge [color="<<strColor<<"];\n";
                    fout << "\t" << srcIter->second << " -> " << outputNodeName << ";\n";
                }
                else
                {

                    std::string strColor = bPredicated ? "red" : 
                                           bSticky ? "blue" :
                                           "black";
                    assert (srcIter != nodeNumberMap.end());
                    assert (dstIter != nodeNumberMap.end());

                    if(drawPorts)
                    {
                        std::string srcPortName = portToName(src, portNameMap, portNodeNum, fout);
                        std::string dstPortName = portToName(dst, portNameMap, portNodeNum, fout);
            
                        fout << "\tedge [color="<<strColor<<"];\n";
                        fout << "\t" << srcIter->second << " -> " << srcPortName << " [weight=1000];\n";
                        fout << "\t" << srcPortName << " -> " << dstPortName << " [color="<<strColor<<"];\n";
                        fout << "\t" << dstPortName << " -> " << dstIter->second << " [weight=1000];\n";
                    }
                    else
                    {
                        fout << "\tedge [color="<<strColor<<"];\n";
                        fout << "\t" << srcIter->second << " -> " << dstIter->second << ";\n";
                    }
                }
            }
        }
        fout << "}\n";
        fout.close();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Writes a dot file to visualize affinity-based scheduler partitioning.  
    ///             To compile the DOT file generated by this function, you will need to download and
    ///             install graphviz : 1. Download and install the msi from
    ///             http://www.graphviz.org/Download_windows.php 2. To compile the DOT file, in a
    ///             command prompt type the following:
    ///                 dot -Tpng &lt;DOT file&gt; -o &lt;Output PNG file&gt;  
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="filename">         [in,out] If non-null, filename of the file. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::WritePartitionDOTFile(
        __in char * filename
        )
    {
        WriteDOTFile(filename, FALSE, FALSE, TRUE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Writes a dot file, with colors and annotations chosen to facilitate debugging of
    ///             graphs which make no forward progess. To use this the caller needs to download
    ///             and install graphviz : 1. Download and install the msi from
    ///             http://www.graphviz.org/Download_windows.php 2. To compile the DOT file, in a
    ///             command prompt type the following
    ///                dot -Tpng &lt;DOT file&gt; -o &lt;Output PNG file&gt;
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="filename">             [in,out] If non-null, filename of the file. </param>
    /// <param name="vReadyTasks">          [in,out] [in,out] If non-null, the ready tasks. </param>
    /// <param name="vReadyChannelMap">     [in,out] [in,out] If non-null, the ready channel map. </param>
    /// <param name="vBlockedChannelMap">   [in,out] [in,out] If non-null, the blocked channel map. </param>
    /// <param name="drawPorts">            (optional) the draw ports. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::WriteDiagnosticDOTFile(
        __in char * filename, 
        __in std::set<Task*>& vReadyTasks,
        __in std::map<Task*, std::set<Channel*>*>& vReadyChannelMap,
        __in std::map<Task*, std::set<Channel*>*>& vBlockedChannelMap,
        __in BOOL   drawPorts
        )
    {
        UNREFERENCED_PARAMETER(vReadyChannelMap);

        std::ofstream fout(filename);
        fout << "digraph DFG {\n";
        std::map<Task*, int> nodeNumberMap;
        std::map<Port*, std::string> portNameMap;
        int nodeId=1, inputNodeNum=1, outputNodeNum=1, portNodeNum=1;
        int nMultiChannelId = 0;
    
        for(std::map<std::string, Task*>::iterator iter=m_vTasks.begin(); iter!=m_vTasks.end() ; ++iter)
        {
            Task * pTask = iter->second;
            BOOL bBlocked = vBlockedChannelMap.find(pTask) != vBlockedChannelMap.end();
            BOOL bReady = vReadyTasks.find(pTask) != vReadyTasks.end();
            assert(!bBlocked || !bReady); 
            UINT uiDispatchCount = pTask->GetCurrentDispatchCount();
            std::string strColor = bReady ? "green" : (bBlocked ? "red" : "blue");
            std::ostringstream ss;
            ss << iter->first << "[" << uiDispatchCount << "]";
            std::string strNodeName = ss.str();
            fout << "\t" << nodeId << "[label = \"" << strNodeName << "\", shape=box, style=filled, color="<<strColor<<", fontcolor=white];\n";            
            nodeNumberMap[iter->second] = nodeId;
            ++nodeId;
        }

        size_t i;
        map<std::string, Channel*>::iterator ci;
        for(i=0, ci=m_vChannels.begin(); ci!=m_vChannels.end(); ++i, ++ci)
        {
            bool bMultiChannel = false;
            Channel * pPrimaryChannel = ci->second;
            vector<Channel*> vComponentChannels;
            if(pPrimaryChannel->GetType() == CT_MULTI) {
                bMultiChannel = true;
                nMultiChannelId++;
                MultiChannel* pMChannel = (MultiChannel*) pPrimaryChannel;
                map<UINT, Channel*>* pmap = pMChannel->GetCoalescedChannelMap();
                map<UINT, Channel*>::iterator mmi;
                for(mmi=pmap->begin(); mmi!=pmap->end(); mmi++) {
                    vComponentChannels.push_back(mmi->second);
                }
            } else {
                vComponentChannels.push_back(pPrimaryChannel);
            }
            vector<Channel*>::iterator vci;
            for(vci=vComponentChannels.begin(); vci!=vComponentChannels.end(); vci++) {
                Channel * channel = *vci;
                Port *src = channel->GetBoundPort(PTask::CE_SRC);
                Port *dst = channel->GetBoundPort(PTask::CE_DST);
                Task *srcTask = NULL, *dstTask = NULL;
                std::map<Task*, int>::iterator srcIter, dstIter;
                if (src)
                {
                    srcTask = src->GetTask();
                    srcIter = nodeNumberMap.find(srcTask);
                }
                if (dst)
                {
                    dstTask = dst->GetTask();
                    dstIter = nodeNumberMap.find(dstTask);
                }
                assert (!(src==NULL && dst==NULL));
                bool bPredicated = 
                    ((channel->GetPredicationType(CE_SRC) != CGATEFN_NONE) ||
                    (channel->GetPredicationType(CE_DST) != CGATEFN_NONE));
                bool bBlocked = false;
                channel->Lock();
                UINT uiQueueDepth = static_cast<UINT>(channel->GetQueueDepth());
                UINT uiCapacity = channel->GetCapacity();
                channel->Unlock();
                bool bFull = uiQueueDepth == uiCapacity;
                bool bEmpty = uiQueueDepth == 0;
                std::map<Task*, set<Channel*>*>::iterator srcI = vBlockedChannelMap.find(srcTask);
                std::map<Task*, set<Channel*>*>::iterator dstI = vBlockedChannelMap.find(dstTask);
                if(srcI != vBlockedChannelMap.end()) {
                    bBlocked = (srcI->second->find(channel) != srcI->second->end());
                } else if(dstI != vBlockedChannelMap.end()) {
                    bBlocked = (dstI->second->find(channel) != dstI->second->end());
                }
                bool bBlockedFull = bBlocked && bFull;
                bool bBlockedEmpty = bBlocked && bEmpty;
                bool bBlockedWithinCapacity = bBlocked && !bBlockedFull && !bBlockedEmpty;
                bool bUnblockedWithinCapacity = !bBlocked && !bFull && !bEmpty;

                std::string strColor = 
                    bBlockedFull ? "red" : 
                    bBlockedEmpty ? "orange" :
                    bBlockedWithinCapacity ? "black" :
                    bUnblockedWithinCapacity ? "purple" :
                    "green";
                std::string strWeight = 
                    bBlocked || bUnblockedWithinCapacity ? " penwidth=3 " : 
                    " penwidth=1 ";

                if (src==NULL) {
                    
                    // Create a new input node
                    char str[1024];
                    sprintf_s(str, 1024, "inputNode_%d", inputNodeNum);
                    ++inputNodeNum;
                    std::string inputNodeName(str); 
                    std::string channelName; 
                    if(bMultiChannel) {
                        const char * pBinding = dst->GetVariableBinding();
                        if(pBinding == NULL) {
                            sprintf_s(str, 1024, "(multi_%d)", nMultiChannelId);
                        } else {
                            sprintf_s(str, 1024, "%s_(multi_%d)", pBinding, nMultiChannelId);
                        }
                        channelName = str;
                    } else {
                        channelName = NULL_TO_EMPTY(dst->GetVariableBinding());
                    }
                    fout << "\t" << inputNodeName << "[label = \"" << channelName << "\", color="<<strColor<<"];\n";
                    fout << "\tedge ["<< strWeight << " color="<<strColor<<"];\n";
                    fout << "\t" << inputNodeName << " -> " << dstIter->second << ";\n";

                }
                else if(dst==NULL)
                {
                    //Create a new output node
                    std::string strColor = bPredicated ? "red":"darkviolet";
                    char str[1024];
                    sprintf_s(str, 1024, "outputNode_%d", outputNodeNum);
                    ++outputNodeNum;
                    std::string outputNodeName(str); 
                    fout << "\t" << outputNodeName << "[label = \"" << NULL_TO_EMPTY(src->GetVariableBinding()) << "\", color="<<strColor<<"];\n";
                    fout << "\tedge ["<< strWeight << " color="<<strColor<<"];\n";
                    fout << "\t" << srcIter->second << " -> " << outputNodeName << ";\n";
                }
                else
                {
                    assert (srcIter != nodeNumberMap.end());
                    assert (dstIter != nodeNumberMap.end());

                    if(drawPorts)
                    {
                        std::string srcPortName = portToName(src, portNameMap, portNodeNum, fout);
                        std::string dstPortName = portToName(dst, portNameMap, portNodeNum, fout);
            
                        fout << "\tedge ["<< strWeight << "color="<<strColor<<"];\n";
                        fout << "\t" << srcIter->second << " -> " << srcPortName << " [weight=1000];\n";
                        fout << "\t" << srcPortName << " -> " << dstPortName << " [color="<<strColor<<"];\n";
                        fout << "\t" << dstPortName << " -> " << dstIter->second << " [weight=1000];\n";
                    }
                    else
                    {
                        fout << "\tedge ["<< strWeight << "color="<<strColor<<"];\n";
                        fout << "\t" << srcIter->second << " -> " << dstIter->second << ";\n";
                    }
                }
            }
        }
        fout << "}\n";
        fout.close();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Writes a dot file, with colors and annotations chosen to facilitate debugging of
    ///             control signal propagation in graphs. To use this the caller needs to download
    ///             and install graphviz : 1. Download and install the msi from
    ///             http://www.graphviz.org/Download_windows.php 2. To compile the DOT file, in a
    ///             command prompt type the following
    ///                dot -Tpng &lt;DOT file&gt; -o &lt;Output PNG file&gt;
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="filename">             [in] non-null, filename of the file. </param>
    /// <param name="pvReadyTasks">         [in] If non-null, a list of tasks that are known to be
    ///                                     ready to dispatch. If null, no assumptions are made about
    ///                                     which tasks are ready. </param>
    /// <param name="pvReadyChannelMap">    [in] If non-null, the ready channel map. If null, no
    ///                                     assumptions are made about which channels are
    ///                                     ready/blocked. </param>
    /// <param name="pvBlockedChannelMap">  [in] If non-null, the blocked channel map. If null, no
    ///                                     assumptions are made about which channels are
    ///                                     ready/blocked. </param>
    /// <param name="drawPorts">            the draw ports. </param>
    /// <param name="pHighlightTasks">      [in,out] If non-null, the highlight tasks. </param>
    ///
    /// ### <param name="luiSignalsFilter"> (Optional) a filter for control signals the caller is
    ///                                     interested in. DBCTLC_NONE means "any", otherwise, the
    ///                                     parameter is the bitwise or of all signals for which the
    ///                                     caller would like to see paths elucidated. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::WriteControlPropagationDiagnosticDOTFile(
        __in char * filename, 
        __in std::set<Task*>* pvReadyTasks,
        __in std::map<Task*, std::set<Channel*>*>* pvReadyChannelMap,
        __in std::map<Task*, std::set<Channel*>*>* pvBlockedChannelMap,
        __in BOOL   drawPorts,
        __in std::set<Task*>* pHighlightTasks
        )
    {
        std::set<Task*> vSignalConductors;
        std::map<Task*, std::set<Port*>> vTaskSignalSources;
        std::map<Port*, std::set<Port*>> vCrossTaskPaths;
        std::set<Channel*> vInterTaskSignalChannels;
        std::set<Task*> emptyReadyTasks;
        std::map<Task*, std::set<Channel*>*> emptyReadyChannelMap;
        std::map<Task*, std::set<Channel*>*> emptyBlockedChannelMap;
        std::set<Task*>& vReadyTasks = pvReadyTasks == NULL ? emptyReadyTasks : *pvReadyTasks;
        std::map<Task*, std::set<Channel*>*>& vReadyChannelMap = pvReadyChannelMap == NULL ? emptyReadyChannelMap : *pvReadyChannelMap;
        std::map<Task*, std::set<Channel*>*>& vBlockedChannelMap = pvBlockedChannelMap == NULL ? emptyBlockedChannelMap : *pvBlockedChannelMap;
        UNREFERENCED_PARAMETER(vReadyChannelMap);

        std::ofstream fout(filename);
        fout << "digraph DFG {\n";
        std::map<Task*, int> nodeNumberMap;
        std::map<Port*, std::string> portNameMap;
        int nodeId=1, inputNodeNum=1, outputNodeNum=1, portNodeNum=1;
        int nMultiChannelId = 0;

        UINT uiCrossTaskPairCount = 0;
        for(std::map<std::string, Task*>::iterator iter=m_vTasks.begin(); iter!=m_vTasks.end() ; ++iter) {
            Task * pTask = iter->second;
            UINT uiTaskCount = pTask->GetControlPropagationPaths(vTaskSignalSources, 
                                                                 vCrossTaskPaths, 
                                                                 vInterTaskSignalChannels);
            if(uiTaskCount != 0) {
                uiCrossTaskPairCount += uiTaskCount;
                vSignalConductors.insert(pTask);
            }
        }
    
        for(std::map<std::string, Task*>::iterator iter=m_vTasks.begin(); iter!=m_vTasks.end() ; ++iter)
        {
            Task * pTask = iter->second;
            BOOL bBlocked = vBlockedChannelMap.find(pTask) != vBlockedChannelMap.end();
            BOOL bReady = vReadyTasks.find(pTask) != vReadyTasks.end();
            BOOL bSignalConductor = vSignalConductors.find(pTask) != vSignalConductors.end();
            BOOL bSignalsOccurred = ctlpwasactive(pTask);
            UINT uiDispatchCount = pTask->GetCurrentDispatchCount();
            std::string strFontColor = bSignalConductor ? "yellow" : "black";
            std::string strColor = "gray60"; 
            if(bSignalConductor) {
                strColor = bSignalsOccurred ? 
                    (bReady ? "darkslategray1" : (bBlocked ? "firebrick1" : "green")) :
                    (bReady ? "darkslategray3" : (bBlocked ? "firebrick4" : "darkgreen"));
            }
            if(pHighlightTasks != NULL) {
                if(pHighlightTasks->find(pTask) != pHighlightTasks->end()) {
                    strColor = "navyblue";
                    strFontColor = "yellow";
                } else {
                    strColor = "gray60";
                    strFontColor = "black";
                }
            }
            std::ostringstream ss;
            ss << iter->first << "[" << uiDispatchCount << "]";
            if(bSignalConductor) {
                std::set<Port*>::iterator si, ti;
                std::set<Port*>& vTaskSources = vTaskSignalSources[pTask];
                for(si=vTaskSources.begin(); si!=vTaskSources.end(); si++) {
                    Port * pSrc = *si;
                    std::set<Port*>& vPortSinks = vCrossTaskPaths[pSrc];
                    for(ti=vPortSinks.begin(); ti!=vPortSinks.end(); ti++) {
                        ss << "\\n" << pSrc->GetVariableBinding() << " -> " << (*ti)->GetVariableBinding();
                    }
                }
            }
            std::string strNodeName = ss.str();
            fout << "\t" << nodeId << "[label = \"" << strNodeName << "\", shape=box, style=filled, color="<<strColor<<", fontcolor="<<strFontColor<<"];\n";            
            nodeNumberMap[iter->second] = nodeId;
            ++nodeId;
        }

        size_t i;
        map<std::string, Channel*>::iterator ci;
        for(i=0, ci=m_vChannels.begin(); ci!=m_vChannels.end(); ++i, ++ci)
        {
            bool bMultiChannel = false;
            Channel * pPrimaryChannel = ci->second;
            vector<Channel*> vComponentChannels;
            if(pPrimaryChannel->GetType() == CT_MULTI) {
                bMultiChannel = true;
                nMultiChannelId++;
                MultiChannel* pMChannel = (MultiChannel*) pPrimaryChannel;
                map<UINT, Channel*>* pmap = pMChannel->GetCoalescedChannelMap();
                map<UINT, Channel*>::iterator mmi;
                for(mmi=pmap->begin(); mmi!=pmap->end(); mmi++) {
                    vComponentChannels.push_back(mmi->second);
                }
            } else {
                vComponentChannels.push_back(pPrimaryChannel);
            }
            vector<Channel*>::iterator vci;
            for(vci=vComponentChannels.begin(); vci!=vComponentChannels.end(); vci++) {
                Channel * channel = *vci;
                Port *src = channel->GetBoundPort(PTask::CE_SRC);
                Port *dst = channel->GetBoundPort(PTask::CE_DST);
                Task *srcTask = NULL, *dstTask = NULL;
                std::map<Task*, int>::iterator srcIter, dstIter;
                if (src)
                {
                    srcTask = src->GetTask();
                    srcIter = nodeNumberMap.find(srcTask);
                }
                if (dst)
                {
                    dstTask = dst->GetTask();
                    dstIter = nodeNumberMap.find(dstTask);
                }

                assert (!(src==NULL && dst==NULL));
                CHANNELACTIVITYSTATE eSigActivityState = ctlpgetchactstate(channel);
                CHANNELPREDICATIONSTATE ePredicationState = ctlpgetchpredstate(channel); 
                UNREFERENCED_PARAMETER(ePredicationState); 

                bool bDevnullOutChannel = ((channel->GetType() == CT_GRAPH_OUTPUT) && 
                                           (channel->GetPredicationType(CE_SRC) == CGATEFN_DEVNULL ||
                                            channel->GetPredicationType(CE_DST) == CGATEFN_DEVNULL));

                BOOL bBlocked = FALSE;
                bool bUserSpecifiedName = channel->HasUserSpecifiedName()!=0;
                std::map<Task*, set<Channel*>*>::iterator srcI = vBlockedChannelMap.find(srcTask);
                std::map<Task*, set<Channel*>*>::iterator dstI = vBlockedChannelMap.find(dstTask);
                if(srcI != vBlockedChannelMap.end()) {
                    bBlocked = (srcI->second->find(channel) != srcI->second->end());
                } else if(dstI != vBlockedChannelMap.end()) {
                    bBlocked = (dstI->second->find(channel) != dstI->second->end());
                }

                std::string strColor(ctlpgetchcolor(eSigActivityState, ePredicationState));
                std::string strWeight = eSigActivityState == cas_none ? " penwidth=1 " : " penwidth=3 ";
                std::string channelName = ctlpgetchname(channel, bBlocked);

                if (src==NULL) {
                    
                    // Create a new input node
                    char str[1024];
                    sprintf_s(str, 1024, "inputNode_%d", inputNodeNum);
                    ++inputNodeNum;
                    std::string inputNodeName(str); 
                    fout << "\t" << inputNodeName << "[shape=point];\n";
                    fout << "\tedge ["<< strWeight << " label = \"" << channelName << "\", color="<<strColor<<"] ";
                    fout << "\t" << inputNodeName << " -> " << dstIter->second << ";\n";

                }
                else if(dst==NULL)
                {
                    if(!bDevnullOutChannel) {
                        //Create a new output node
                        // std::string strColor = bRelevantPredicate ? "red":"darkviolet";
                        char str[1024];
                        sprintf_s(str, 1024, "outputNode_%d", outputNodeNum);
                        ++outputNodeNum;
                        std::string outputNodeName(str); 
                        fout << "\t" << outputNodeName << "[shape=point]\n";
                        fout << "\tedge ["<< strWeight << " label = \"" << channelName << "\", color="<<strColor<<"]";
                        fout << "\t" << srcIter->second << " -> " << outputNodeName << ";\n";
                    }
                }
                else
                {
                    assert (srcIter != nodeNumberMap.end());
                    assert (dstIter != nodeNumberMap.end());

                    if(drawPorts)
                    {
                        std::string srcPortName = portToName(src, portNameMap, portNodeNum, fout);
                        std::string dstPortName = portToName(dst, portNameMap, portNodeNum, fout);
            
                        fout << "\tedge ["<< strWeight << "color="<<strColor<<"];\n";
                        fout << "\t" << srcIter->second << " -> " << srcPortName << " [weight=1000];\n";
                        fout << "\t" << srcPortName << " -> " << dstPortName << " [color="<<strColor<<"];\n";
                        fout << "\t" << dstPortName << " -> " << dstIter->second << " [weight=1000];\n";
                    }
                    else
                    {
                        if(bUserSpecifiedName) {
                            fout << "\tedge ["<< strWeight << " label = \"" << channelName << "\", color="<<strColor<<"]";
                            fout << "\t" << srcIter->second << " -> " << dstIter->second << ";\n";
                        } else {
                            fout << "\tedge ["<< strWeight << "color="<<strColor<<"];\n";
                            fout << "\t" << srcIter->second << " -> " << dstIter->second << ";\n";
                        }
                    }
                }
            }
        }

        BOOL bShowLegend = TRUE; 
        if(bShowLegend) {
            fout << "   subgraph key{                                                                " << std::endl;
            fout << "    Legend [shape=none, margin=0, label=<                                       " << std::endl;
            fout << "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">   " << std::endl;
            fout << "     <TR>                                                                       " << std::endl;
            fout << "      <TD COLSPAN=\"5\"><B>Legend</B></TD>                                      " << std::endl;
            fout << "     </TR>                                                                      " << std::endl;
            fout << "     <TR>                                                                       " << std::endl;
            fout << "      <TD rowspan=\"4\">Tasks</TD>                                              " << std::endl;
            fout << "      <TD COLSPAN=\"1\"><B>signal path</B></TD>                                 " << std::endl;
            fout << "      <TD COLSPAN=\"3\"><B>Execution State</B></TD>                             " << std::endl;
            fout << "     </TR>                                                                      " << std::endl;
            fout << "     <TR>                                                                       " << std::endl;
            fout << "      <TD>none</TD>                                                             " << std::endl;
            fout << "      <TD BGCOLOR=\"gray60\"><FONT COLOR=\"black\">idle</FONT></TD>             " << std::endl;
            fout << "      <TD BGCOLOR=\"gray60\"><FONT COLOR=\"black\">blocked</FONT></TD>          " << std::endl;
            fout << "      <TD BGCOLOR=\"gray60\"><FONT COLOR=\"black\">ready</FONT></TD>            " << std::endl;
            fout << "     </TR>                                                                      " << std::endl;
            fout << "     <TR>                                                                       " << std::endl;
            fout << "      <TD>not exercised</TD>                                                    " << std::endl;
            fout << "      <TD BGCOLOR=\"darkgreen\"><FONT COLOR=\"yellow\">idle</FONT></TD>         " << std::endl;
            fout << "      <TD BGCOLOR=\"firebrick4\"><FONT COLOR=\"yellow\">blocked</FONT></TD>     " << std::endl;
            fout << "      <TD BGCOLOR=\"darkslategray3\"><FONT COLOR=\"yellow\">ready</FONT></TD>   " << std::endl;
            fout << "     </TR>                                                                      " << std::endl;
            fout << "     <TR>                                                                       " << std::endl;
            fout << "      <TD>exercised</TD>                                                        " << std::endl;
            fout << "      <TD BGCOLOR=\"green\"><FONT COLOR=\"yellow\">idle</FONT></TD>             " << std::endl;
            fout << "      <TD BGCOLOR=\"firebrick1\"><FONT COLOR=\"yellow\">blocked</FONT></TD>     " << std::endl;
            fout << "      <TD BGCOLOR=\"darkslategray1\"><FONT COLOR=\"yellow\">ready</FONT></TD>   " << std::endl;
            fout << "     </TR>                                                                      " << std::endl;
            fout << "                                                                                " << std::endl;
            fout << "     <TR>                                                                       " << std::endl;
            fout << "      <TD COLSPAN=\"1\" rowspan=\"4\">Channels</TD>                             " << std::endl;
            fout << "      <TD COLSPAN=\"1\"><B>signal path</B></TD>                                 " << std::endl;
            fout << "      <TD COLSPAN=\"3\"><B>Predicate State</B></TD>                             " << std::endl;
            fout << "     </TR>                                                                      " << std::endl;
            fout << "     <TR>                                                                       " << std::endl;
            fout << "      <TD>none</TD>                                                             " << std::endl;
            fout << "      <TD BGCOLOR=\""<<ctlpgetchcolor(cas_none, cps_na)<<"\"><FONT COLOR=\"black\">NA</FONT></TD>             " << std::endl;
            fout << "      <TD BGCOLOR=\""<<ctlpgetchcolor(cas_none, cps_open)<<"\"><FONT COLOR=\"black\">open</FONT></TD>             " << std::endl;
            fout << "      <TD BGCOLOR=\""<<ctlpgetchcolor(cas_none, cps_closed)<<"\"><FONT COLOR=\"black\">closed</FONT></TD>          " << std::endl;
            fout << "     </TR>                                                                      " << std::endl;
            fout << "     <TR>                                                                       " << std::endl;
            fout << "      <TD>not exercised</TD>                                                    " << std::endl;
            fout << "      <TD BGCOLOR=\""<<ctlpgetchcolor(cas_unexercised, cps_na)<<"\"><FONT COLOR=\"yellow\">NA</FONT></TD>         " << std::endl;
            fout << "      <TD BGCOLOR=\""<<ctlpgetchcolor(cas_unexercised, cps_open)<<"\"><FONT COLOR=\"yellow\">open</FONT></TD>         " << std::endl;
            fout << "      <TD BGCOLOR=\""<<ctlpgetchcolor(cas_unexercised, cps_closed)<<"\"><FONT COLOR=\"yellow\">closed</FONT></TD>     " << std::endl;
            fout << "     </TR>                                                                      " << std::endl;
            fout << "     <TR>                                                                       " << std::endl;
            fout << "      <TD>exercised</TD>                                                        " << std::endl;
            fout << "      <TD BGCOLOR=\""<<ctlpgetchcolor(cas_exercised, cps_na)<<"\"><FONT COLOR=\"yellow\">NA</FONT></TD>         " << std::endl;
            fout << "      <TD BGCOLOR=\""<<ctlpgetchcolor(cas_exercised, cps_open)<<"\"><FONT COLOR=\"yellow\">open</FONT></TD>         " << std::endl;
            fout << "      <TD BGCOLOR=\""<<ctlpgetchcolor(cas_exercised, cps_closed)<<"\"><FONT COLOR=\"yellow\">closed</FONT></TD>     " << std::endl;
            fout << "     </TR>                                                                      " << std::endl;
            fout << "    </TABLE>                                                                    " << std::endl;
            fout << "   >];                                                                          " << std::endl;
            fout << "  }    	                                                                     " << std::endl;
            fout << "  Legend -> 1[style=\"invisible\" dir=\"none\"];	                             " << std::endl;
            fout << "  Legend -> " << nodeId-1 << " [style=\"invisible\" dir=\"none\"];	             " << std::endl;
        }

        fout << "}\n";
        fout.close();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Writes a dot file.  
    ///             To compile the DOT file generated by this function, you will need to download and
    ///             install graphviz : 
    ///             1. Download and install the msi from http://www.graphviz.org/Download_windows.php 
    ///             2. To compile the DOT file, in a command prompt type the following:   
    ///                 dot -Tpng <DOT file> -o <Output PNG file>  
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="filename">     [in,out] If non-null, filename of the file. </param>
    /// <param name="drawPorts">    (optional) the draw ports. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::WritePresentationDOTFile(
        char * filename, 
        BOOL drawPorts
        )
    {
        int fontsize=96;
        std::string fontname("Arial");        

        std::ofstream fout(filename);
        fout << "digraph DFG {\n";
        fout << "\tgraph [bgcolor=black];\n";
        fout << "\tgraph [color=white];\n";
	    fout << "\tedge [color=white];\n";
        fout << "\tgraph[page=\"11,8.5\",size=\"10,6\",ratio=fill,center=1];\n";

        std::map<Task*, int> nodeNumberMap;
        std::map<Port*, std::string> portNameMap;
        int nodeId=1, inputNodeNum=1, outputNodeNum=1, portNodeNum=1;
        int nMultiChannelId = 0;
    
        for(std::map<std::string, Task*>::iterator iter=m_vTasks.begin(); iter!=m_vTasks.end() ; ++iter)
        {
            Task * pTask = iter->second;
            if(pTask->GetMetaPortMap()->size() > 0) {
                fout << "\t" << nodeId << "[label = \"" << iter->first << "\", shape=box, style=filled, color=lightblue, fontsize="<<fontsize<<", fontname="<<fontname<<", fontcolor=black];\n";
            } else {
                fout << "\t" << nodeId << "[label = \"" << iter->first << "\", shape=box, style=filled, color=blue, fontsize="<<fontsize<<", fontname="<<fontname<<", fontcolor=white];\n";
            }
            nodeNumberMap[iter->second] = nodeId;
            ++nodeId;
        }

        size_t i;
        map<std::string, Channel*>::iterator ci;
        for(i=0, ci=m_vChannels.begin(); ci!=m_vChannels.end(); ++i, ++ci)
        {
            bool bMultiChannel = false;
            Channel * pPrimaryChannel = ci->second;
            vector<Channel*> vComponentChannels;
            if(pPrimaryChannel->GetType() == CT_MULTI) {
                bMultiChannel = true;
                nMultiChannelId++;
                MultiChannel* pMChannel = (MultiChannel*) pPrimaryChannel;
                map<UINT, Channel*>* pmap = pMChannel->GetCoalescedChannelMap();
                map<UINT, Channel*>::iterator mmi;
                for(mmi=pmap->begin(); mmi!=pmap->end(); mmi++) {
                    vComponentChannels.push_back(mmi->second);
                }
            } else {
                vComponentChannels.push_back(pPrimaryChannel);
            }
            vector<Channel*>::iterator vci;
            for(vci=vComponentChannels.begin(); vci!=vComponentChannels.end(); vci++) {
                Channel * channel = *vci;

                //if (channel->ShouldDraw() == false) continue;
                Port *src = channel->GetBoundPort(PTask::CE_SRC);
                Port *dst = channel->GetBoundPort(PTask::CE_DST);
                Task *srcTask = NULL, *dstTask = NULL;
                std::map<Task*, int>::iterator srcIter, dstIter;
                if (src)
                {
                    srcTask = src->GetTask();
                    srcIter = nodeNumberMap.find(srcTask);
                }
                if (dst)
                {
                    dstTask = dst->GetTask();
                    dstIter = nodeNumberMap.find(dstTask);
                }
                assert (!(src==NULL && dst==NULL));
                bool bPredicated = 
                    ((channel->GetPredicationType(CE_SRC) != CGATEFN_NONE) ||
                    (channel->GetPredicationType(CE_DST) != CGATEFN_NONE));
                bool bDescriptor = ((src == NULL) && (dst != NULL) && dst->IsDescriptorPort());

                if (src==NULL)
                {
                    // explicitly leave out descriptor ports in presentation mode
                    if(!bDescriptor) {
                        //Create a new input node
                        std::string strColor = bPredicated ? "red" : 
                                               bDescriptor ? "gray10" : 
                                               bMultiChannel ? "orange" : 
                                               "green";
                        char str[1024];
                        sprintf_s(str, 1024, "inputNode_%d", inputNodeNum);
                        ++inputNodeNum;
                        std::string inputNodeName(str); 
                        std::string channelName; 
                        if(bMultiChannel) {
                            const char * pBinding = dst->GetVariableBinding();
                            if(pBinding == NULL) {
                                sprintf_s(str, 1024, "(multi_%d)", nMultiChannelId);
                            } else {
                                sprintf_s(str, 1024, "%s_(multi_%d)", pBinding, nMultiChannelId);
                            }
                            channelName = str;
                        } else {
                            channelName = NULL_TO_EMPTY(dst->GetVariableBinding());
                        }
                        fout << "\t" << inputNodeName << "[label = \"" << channelName << "\", fontsize="<<fontsize<<", fontname="<<fontname<<", fontcolor="<<strColor<<", color="<<strColor<<"];\n";
                        fout << "\tedge [color="<<strColor<<"];\n";
                        fout << "\t" << inputNodeName << " -> " << dstIter->second << ";\n";
                    }

                }
                else if(dst==NULL)
                {
                    //Create a new output node
                    std::string strColor = bPredicated ? "red":"violet";
                    char str[1024];
                    sprintf_s(str, 1024, "outputNode_%d", outputNodeNum);
                    ++outputNodeNum;
                    std::string outputNodeName(str); 
                    fout << "\t" << outputNodeName << "[label = \"" << NULL_TO_EMPTY(src->GetVariableBinding()) << "\", fontsize="<<fontsize<<", fontname="<<fontname<<", fontcolor="<<strColor<<", color="<<strColor<<"];\n";
                    fout << "\tedge [color="<<strColor<<"];\n";
                    fout << "\t" << srcIter->second << " -> " << outputNodeName << ";\n";
                }
                else
                {

                    std::string strColor = bPredicated ? "red" : "yellow";
                    assert (srcIter != nodeNumberMap.end());
                    assert (dstIter != nodeNumberMap.end());

                    if(drawPorts)
                    {
                        std::string srcPortName = portToName(src, portNameMap, portNodeNum, fout);
                        std::string dstPortName = portToName(dst, portNameMap, portNodeNum, fout);
            
                        fout << "\tedge [color="<<strColor<<"];\n";
                        fout << "\t" << srcIter->second << " -> " << srcPortName << " [weight=1000];\n";
                        fout << "\t" << srcPortName << " -> " << dstPortName << " [color="<<strColor<<"];\n";
                        fout << "\t" << dstPortName << " -> " << dstIter->second << " [weight=1000];\n";
                    }
                    else
                    {
                        fout << "\tedge [color="<<strColor<<"];\n";
                        fout << "\t" << srcIter->second << " -> " << dstIter->second << ";\n";
                    }
                }
            }
        }
        fout << "}\n";
        fout.close();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reports the graph state: diagnostic tool for finding problems with lack of
    ///             forward progress.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///
    /// <param name="ios">              [in,out] The ios. </param>
    /// <param name="szLabel">          [in,out] If non-null, the label. </param>
    /// <param name="pTask">            [in,out] If non-null, the task. </param>
    /// <param name="pPortMap">         [in,out] If non-null, the port map. </param>
    /// <param name="bIsInput">         true if this object is input. </param>
    /// <param name="vReadyChannels">   [in,out] The ready channels. </param>
    /// <param name="vBlockedChannels"> [in,out] The non ready channels. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::ReportBoundChannelState(
        std::ostream &ios,
        char * szLabel,
        Task * pTask,
        std::map<UINT, Port*>* pPortMap,
        BOOL bIsInput,
        std::set<Channel*>& vReadyChannels,
        std::set<Channel*>& vBlockedChannels
        )
    {
        if(pPortMap->size() == 0) return;
        ios << "\t" << szLabel << " for " << pTask << ":"<< std::endl;
        std::map<UINT, Port*>::iterator pi;
        for(pi=pPortMap->begin(); pi!=pPortMap->end(); pi++) {
            Port* pPort = pi->second;
            pPort->Lock();
            BOOL bSticky = pPort->IsSticky();
            BOOL bOccupied = pPort->IsOccupied();
            BOOL bReady = bIsInput ? bOccupied : !bOccupied;
            UINT uiChannelCount = pPort->GetChannelCount(); 
            UINT uiControlChannelCount = pPort->GetControlChannelCount(); 
            for(UINT i=0; i<uiChannelCount; i++) {
                Channel * pChannel = pPort->GetChannel(i);
                pChannel->Lock();
                UINT uiCapacity = static_cast<UINT>(pChannel->GetCapacity());
                UINT uiQueued = static_cast<UINT>(pChannel->GetQueueDepth());
                BOOL bPredicated = pChannel->IsPredicated(CE_SRC) ||
                                    pChannel->IsPredicated(CE_DST);
                if(bIsInput) {
                    if(uiQueued > 0) 
                        vReadyChannels.insert(pChannel);
                    else 
                        vBlockedChannels.insert(pChannel);
                } else {
                    if(uiQueued == uiCapacity)
                        vBlockedChannels.insert(pChannel);
                    else 
                        vReadyChannels.insert(pChannel);
                }
                ios << "\t\t"
                    << (bReady?"** ":"   ")
                    << (bSticky?"S ":"  ")
                    << pPort->GetVariableBinding() << " "
                    << pChannel << ": " << uiQueued << "/" 
                    << uiCapacity
                    << (bPredicated ? " (PRED)":"")
                    << std::endl;
                pChannel->Unlock();
            }
            for(UINT i=0; i<uiControlChannelCount; i++) {
                Channel * pChannel = pPort->GetControlChannel(i);
                pChannel->Lock();
                UINT uiCapacity = static_cast<UINT>(pChannel->GetCapacity());
                UINT uiQueued = static_cast<UINT>(pChannel->GetQueueDepth());
                BOOL bPredicated = pChannel->IsPredicated(CE_SRC) ||
                                    pChannel->IsPredicated(CE_DST);
                if(bIsInput) {
                    if(uiQueued > 0) 
                        vReadyChannels.insert(pChannel);
                    else 
                        vBlockedChannels.insert(pChannel);
                } else {
                    if(uiQueued == uiCapacity)
                        vBlockedChannels.insert(pChannel);
                    else 
                        vReadyChannels.insert(pChannel);
                }
                ios << "\t\t"
                    << (bReady?"** ":"   ")
                    << (bSticky?"S ":"  ")
                    << pPort->GetVariableBinding() << " "
                    << pChannel << "(control): " << uiQueued << "/" 
                    << uiCapacity
                    << (bPredicated ? " (PRED)":"")
                    << std::endl;
                pChannel->Unlock();
            }
            pPort->Unlock();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Stream insertion operator for graph state enum. </summary>
    ///
    /// <remarks>   crossbac, 7/2/2013. </remarks>
    ///
    /// <param name="os">       [in,out] The operating system. </param>
    /// <param name="eState">   The state. </param>
    ///
    /// <returns>   The shifted result. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::ostream& operator <<(
        std::ostream& os, 
        const GRAPHSTATE& eState
        )
    {
        os << GraphStateToString(eState);
        return os;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reports the graph state: diagnostic tool for finding problems with lack of
    ///             forward progress.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///
    /// <param name="ios">                  [in,out] The ios. </param>
    /// <param name="lpszDOTFilePath">      [in,out] If non-null, full pathname of the dot file. </param>
    /// <param name="bForceGraphDisplay">   true to force graph display. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::ReportGraphState(
        std::ostream &ios,
        char * lpszDOTFilePath,
        BOOL bForceGraphDisplay
        )
    {
        ios << "Dumping graph state for " 
            << m_lpszGraphName << "["
            << m_eState << "]"
            << std::endl << std::endl << std::endl;
        std::set<Task*> vReadyTasks;
        std::map<std::string, Task*>::iterator iter;
        std::map<Task*, std::set<Channel*>*> vReadyChannelMap;
        std::map<Task*, std::set<Channel*>*> vBlockedChannelMap;       
        for(iter=m_vTasks.begin(); iter!=m_vTasks.end() ; ++iter) {
            Task * pTask = iter->second;
            BOOL bReady = pTask->IsReadyForDispatch(&m_eState);
            std::set<Channel*>* pvReadyChannels = new std::set<Channel*>();
            std::set<Channel*>* pvBlockedChannels = new std::set<Channel*>();
            std::map<UINT, Port*>* pInputPorts = pTask->GetInputPortMap();
            std::map<UINT, Port*>* pConstantPorts = pTask->GetConstantPortMap();
            std::map<UINT, Port*>* pOutputPorts = pTask->GetOutputPortMap();
            std::map<UINT, Port*>* pMetaPorts = pTask->GetMetaPortMap();
            ios << pTask << (bReady?"(READY?!!!!)":"") << ":"<< std::endl;
            if(bReady) vReadyTasks.insert(pTask);
            ReportBoundChannelState(ios, "inputs", pTask, pInputPorts, TRUE, *pvReadyChannels, *pvBlockedChannels);
            ReportBoundChannelState(ios, "constants", pTask, pConstantPorts, TRUE, *pvReadyChannels, *pvBlockedChannels);
            ReportBoundChannelState(ios, "meta-ins", pTask, pMetaPorts, TRUE, *pvReadyChannels, *pvBlockedChannels);
            ReportBoundChannelState(ios, "outputs", pTask, pOutputPorts, FALSE, *pvReadyChannels, *pvBlockedChannels);
            if(pvReadyChannels->size() && pvBlockedChannels->size()) {
                vReadyChannelMap[pTask] = pvReadyChannels;
                vBlockedChannelMap[pTask] = pvBlockedChannels;
                std::set<Channel*>::iterator si;
                BOOL bFirst = TRUE;
                ios << "\twaiting for: ";
                for(si=pvBlockedChannels->begin(); si!=pvBlockedChannels->end(); si++) {
                    if(!bFirst) ios << ", ";
                    Channel * pChannel = *si;
                    switch(pChannel->GetType()) {
                    case CT_GRAPH_INPUT: ios << "external-input"; break;
                    case CT_GRAPH_OUTPUT: ios << "external-pull"; break;
                    case CT_INITIALIZER: ios << "predicate"; break;
                    case CT_INTERNAL: 
                        if(pTask == pChannel->GetBoundPort(CE_DST)->GetTask()) {
                            ios << pChannel->GetBoundPort(CE_SRC)->GetTask()->GetTaskName();
                        } else {
                            ios << pChannel->GetBoundPort(CE_DST)->GetTask()->GetTaskName();
                        }
                        break;
                    default:
                        break;
                    }
                    bFirst = FALSE;
                }
            } else {
                // Heuristic prune of candidate tasks:
                // this task's state is apparently uninteresting
                // for diagnosing hung graphs: either all channels
                // are ready or all blocked. We are typically looking 
                // for a single missed state transition or event on a channel
                // and this task will either fire or will likely require a lot of
                // state change in the graph to become ready. Hence we ignore it.
                delete pvReadyChannels;
                delete pvBlockedChannels;
            }
            ios << std::endl << std::endl << std::endl << std::endl;
        }
        if(lpszDOTFilePath != NULL) {
            WriteDiagnosticDOTFile(lpszDOTFilePath, vReadyTasks, vReadyChannelMap, vBlockedChannelMap, FALSE);
            ios << "Diagnostic graph DOT file saved in " << lpszDOTFilePath << std::endl;
            if(bForceGraphDisplay) {
                char szCommandLine[1024];
                sprintf_s(szCommandLine, 1024, "dot -Tpng %s -o %s.png", lpszDOTFilePath, lpszDOTFilePath);
                system(szCommandLine);
                sprintf_s(szCommandLine, 1024, "start %s.png", lpszDOTFilePath);
                system(szCommandLine);
            }
            if(PTask::Runtime::GetSignalProfilingEnabled()) {
                //  ccomps -x hunggraphCP.dot | dot | gvpack -array_u | neato -Tpng -n2 -o tryit.png
                char * lpszCPDOTFilePath = "c:\\temp\\hunggraphCP.dot";
                ios << "Control signal diagnostic graph DOT file saved in " << lpszCPDOTFilePath << std::endl;
                WriteControlPropagationDiagnosticDOTFile(lpszCPDOTFilePath, &vReadyTasks, &vReadyChannelMap, &vBlockedChannelMap, FALSE);
                if(bForceGraphDisplay) {
                    char szCommandLine[1024];
                    sprintf_s(szCommandLine, 1024, "ccomps -x %s | dot | gvpack -array_u | neato -Tpng -n2 -o %s.png", lpszCPDOTFilePath, lpszCPDOTFilePath);
                    system(szCommandLine);
                    sprintf_s(szCommandLine, 1024, "start %s.png", lpszCPDOTFilePath);
                    system(szCommandLine);
                }
                std::stringstream * phist = SignalProfiler::GetHistory();
                ios << "SIGNAL HISTORY: "<< std::endl << phist->str() << std::endl;
                delete phist;
            }
        }
        std::map<Task*, std::set<Channel*>*>::iterator mi;
        for(mi=vReadyChannelMap.begin(); mi!=vReadyChannelMap.end(); mi++)
            delete mi->second;
        for(mi=vBlockedChannelMap.begin(); mi!=vBlockedChannelMap.end(); mi++)
            delete mi->second;
        ios << "Press any key to un-pause the graph" << std::endl;
        getc(stdin);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pause and report graph state: another diagnostics tool. Reset the graph
    ///             running event, and set the probe graph event. This will cause the monitor
    ///             thread to dump the graph and diagnostics DOT file. The event synchronization
    ///             require to do this right (that is, to ensure the graph quiesces before we dump)
    ///             is non-trivial; so we do it in the most unprincipled way possible; Sleep!
    ///             TODO: fix this so we can actually probe the graph from other processes.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::PauseAndReportGraphState(
        VOID
        )
    {
        ResetEvent(m_hGraphRunningEvent);
        Sleep(5000);
        SetEvent(m_hProbeGraphEvent);
        Sleep(15000);
        SetEvent(m_hGraphRunningEvent);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets (a pointer to) the graph name. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the name. </returns>
    ///-------------------------------------------------------------------------------------------------

    const char * 
    Graph::GetName(
        VOID
        )
    {
        return m_lpszGraphName;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Request a pooled block. </summary>
    ///
    /// <remarks>   crossbac, 8/21/2013. </remarks>
    ///
    /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
    /// <param name="uiDataSize">       Size of the data. </param>
    /// <param name="uiMetaSize">       Size of the meta. </param>
    /// <param name="uiTemplateSize">   Size of the template. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Graph::RequestPooledBlock(
        __in DatablockTemplate * pTemplate,
        __in UINT                uiDataSize,
        __in UINT                uiMetaSize,
        __in UINT                uiTemplateSize
        )
    {
        if(m_pScopedPoolManager != NULL)
            return m_pScopedPoolManager->RequestBlock(pTemplate, uiDataSize, uiMetaSize, uiTemplateSize);
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Require block pool. </summary>
    ///
    /// <remarks>   crossbac, 8/20/2013. </remarks>
    ///
    /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
    /// <param name="nDataSize">        Size of the data. </param>
    /// <param name="nMetaSize">        Size of the meta. </param>
    /// <param name="nTemplateSize">    Size of the template. </param>
    /// <param name="nBlocks">          (Optional) The blocks. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Graph::RequireBlockPool(
        __in DatablockTemplate * pTemplate,
        __in int                 nDataSize, 
        __in int                 nMetaSize, 
        __in int                 nTemplateSize,
        __in int                 nBlocks
        )
    {        
        Lock();
        if(m_pScopedPoolManager == NULL) {
            m_pScopedPoolManager = new ScopedPoolManager(this);
        }
        Unlock();
        return m_pScopedPoolManager->RequireBlockPool(pTemplate,
                                                      nDataSize, 
                                                      nMetaSize, 
                                                      nTemplateSize, 
                                                      nBlocks);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Require block pool. </summary>
    ///
    /// <remarks>   crossbac, 8/20/2013. </remarks>
    ///
    /// <param name="nDataSize">        Size of the data. </param>
    /// <param name="nMetaSize">        Size of the meta. </param>
    /// <param name="nTemplateSize">    Size of the template. </param>
    /// <param name="nBlocks">          (Optional) The blocks. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Graph::RequireBlockPool(
        __in int                 nDataSize, 
        __in int                 nMetaSize, 
        __in int                 nTemplateSize,
        __in int                 nBlocks
        )
    {
        Lock();
        if(m_pScopedPoolManager == NULL) {
            m_pScopedPoolManager = new ScopedPoolManager(this);
        }
        Unlock();
        return m_pScopedPoolManager->RequireBlockPool(nDataSize, 
                                                      nMetaSize, 
                                                      nTemplateSize, 
                                                      nBlocks);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Require block pool. </summary>
    ///
    /// <remarks>   crossbac, 8/20/2013. </remarks>
    ///
    /// <param name="pTemplate">    [in,out] If non-null, the template. </param>
    /// <param name="nBlocks">      (Optional) The blocks. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Graph::RequireBlockPool(
        __in DatablockTemplate * pTemplate,
        __in int                 nBlocks
        )
    {
        return RequireBlockPool(pTemplate, 0, 0, 0, nBlocks);
    }        

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Serialize the graph structure to a file such that it can be deserialized in another 
    ///             process and used as if the graph had been instantiated there directly with calls to 
    ///             the graph construction API.
    ///
    ///             Only the graph topology (Tasks, Ports, Channels etc) and static information about
    ///             those entities, including scheduling-related properties, are stored. Datablocks
    ///             and Task executions in flight are not recorded.
    ///             </summary>
    ///
    /// <remarks>   jcurrey, 5/5/2013. </remarks>
    ///
    /// <param name="filename">   The name of the file to serialize the graph to. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::Serialize(const char * filename)
    {
#ifdef XMLSUPPORT
        XMLWriter* writer = new XMLWriter(filename);
        writer->WriteGraph(this);
        delete writer;
#else        
        PTask::Runtime::HandleError("%s::%s(%d): PTask build does not support XML serialization/deserialization\n"
                                    "\t cannot save %s to %s\n",
                                    __FILE__,
                                    __FUNCTION__,
                                    __LINE__,
                                    m_lpszGraphName,
                                    filename);
#endif
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deserializes the given file. </summary>
    ///
    /// <remarks>   jcurrey. </remarks>
    ///
    /// <param name="filename"> Filename of the file. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::Deserialize(const char * filename)
    {
#ifdef XMLSUPPORT
        XMLReader* reader = new XMLReader(filename);
        reader->ReadGraph(this);
        delete reader;
#else
        PTask::Runtime::HandleError("%s::%s(%d): PTask build does not support XML serialization/deserialization\n"
                                    "\t cannot read graph from %s\n",
                                    __FILE__,
                                    __FUNCTION__,
                                    __LINE__,
                                    filename);
#endif    
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synthesize control propagation paths required to ensure that all graph objects
    ///             whose behavior may be predicated by the specified set of signals (bitwise-or'd
    ///             together)
    ///             are reachable along some control propagation path. This is heuristic,
    ///             experimental code--generally, specifying such paths is a task we leave to the
    ///             programmer. In many simple cases, we conjecture that we can synthesize the
    ///             required paths using well known graph traversal algorithms. However, this is
    ///             decidedly conjecture, the programmer is advised to use this API at his/her own
    ///             risk.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/3/2014. </remarks>
    ///
    /// <param name="luiControlSignals">    [in] Bit-wise or of all control signals of interest.
    ///                                     Technically, these can be collected automatically by
    ///                                     traversing the graph as well: if DBCTLC_NONE is specified, we
    ///                                     traverse the graph to examine predicated objects and collect
    ///                                     the set of "signals-of-interest". </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Graph::SynthesizeControlPropagationPaths(
        __in CONTROLSIGNAL luiControlSignals
        )
    {
        if(luiControlSignals == DBCTLC_NONE)
            luiControlSignals = GetControlSignalsOfInterest();

        BOOL bSuccess = TRUE;
        std::set<Task*> vInputTerminals;
        std::set<Task*> vOutputTerminals;
        std::map<Task*, std::map<size_t, std::vector<std::vector<Task*>>>> vPaths;
        std::map<Task*, std::map<size_t, std::vector<std::vector<Task*>>>>::iterator vppi;

        GetControlSignalTerminalsOfInterest(vInputTerminals, vOutputTerminals, luiControlSignals); 
        FindControlSignalPaths(vInputTerminals, vOutputTerminals, vPaths);

        for(vppi=vPaths.begin(); vppi!=vPaths.end(); vppi++) {

            Task * pTask = vppi->first;
            if(vOutputTerminals.find(pTask) != vOutputTerminals.end()) {
                std::map<size_t, std::vector<std::vector<Task*>>>::iterator vpmi;
                vpmi=vppi->second.begin();
                std::vector<std::vector<Task*>>& vTaskPaths = vpmi->second;
                std::vector<Task*>& vFirstPath = vTaskPaths.front();
                bSuccess &= SynthesizeControlPropagationPath(vFirstPath);
            }
        }
        
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Searches for the super-set of all "control signals of interest" for a graph.  
    ///             A control signal is "of interest" if there exists an object within this graph
    ///             whose behavior is predicated in some way by the presence or absence of a given
    ///             signal. This function traverses the graph and returns the bit-wise OR of all such
    ///             signals.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/7/2014. </remarks>
    ///
    /// <returns>   The bitwise OR of all found control signals of interest. </returns>
    ///-------------------------------------------------------------------------------------------------

    CONTROLSIGNAL 
    Graph::GetControlSignalsOfInterest(
        VOID
        )
    {
        CONTROLSIGNAL luiSuperset = DBCTLC_NONE;
        map<std::string, Task*>::iterator ti;
        map<CONTROLSIGNAL, Channel*>::iterator mi;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++)
            luiSuperset |= (ti->second)->GetControlSignalsOfInterest();        
        luiSuperset |= m_luiTriggerControlSignals;
        return luiSuperset;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   find the set of terminal tasks which must be reachable along some control
    ///             propagation path for the given set of control signals to correctly exercise
    ///             all predicates in graph for the given (bit-wise OR of) control signal(s).
    ///             If no control signal is specified, this API will first search the graph to
    ///             collect the set of all control signals of interest.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/7/2014. </remarks>
    ///
    /// <param name="vTasks">               [in,out] [in,out] If non-null, the tasks. </param>
    /// <param name="luiSignalsOfInterest"> (Optional) the lui signals of interest. </param>
    ///
    /// <returns>   the number of terminals found. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Graph::GetControlSignalOutputTerminalsOfInterest(
        __inout std::set<Task*>& vTasks,
        __in    CONTROLSIGNAL luiSignalsOfInterest
        )
    {
        if(luiSignalsOfInterest == DBCTLC_NONE) 
            luiSignalsOfInterest = GetControlSignalsOfInterest(); 
        map<std::string, Task*>::iterator ti;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {
            Task * pTask = ti->second;
            CONTROLSIGNAL luiTaskSignalsOfInterest = pTask->GetControlSignalsOfInterest();
            if(luiSignalsOfInterest & luiTaskSignalsOfInterest)
                vTasks.insert(pTask);
        }
        return static_cast<UINT>(vTasks.size());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   find the set of input terminal tasks which are candidates for source nodes
    ///             along control propagation paths. Note that there is no need to know which signals
    ///             are of interest, as this candidate set is the same regardless. The automated signal
    ///             router try to find paths to output terminals that begin from these terminals. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/7/2014. </remarks>
    ///
    /// <param name="vTasks">                   [in,out] [in,out] If non-null, the tasks. </param>
    /// <param name="bFilterConstantEdgeNodes"> true to ignore input edges with apparent
    ///                                         const semantics on all exposed channels. </param>
    ///
    /// <returns>   the number of terminals found. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Graph::GetControlSignalInputTerminalsOfInterest(
        __inout std::set<Task*>& vTasks,
        __in    BOOL bFilterConstantEdgeNodes
        )
    {
        // find all the exposed input edges. A task is an exposed input if it has a graph input channel
        // which could be used as the source end of a control propagation path. Such channels can be
        // connected to either normal or control channels. The criterion can be further restricted to
        // require that the bound input be non-constant, which boils down to requiring that the set
        // returned contains only tasks with inputs that can be used without the hazard of potential
        // unintentional re-use of control signals (e.g. because we are routing signals from a "sticky"
        // input). 
            
        map<std::string, Task*>::iterator ti;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {

            Task * pTask = ti->second; 
            BOOL bKnownInputTask = FALSE;
            std::set<Channel*> vTaskInputs;
            std::set<Channel*>::iterator vsi;
            pTask->GetInboundChannels(vTaskInputs);

            for(vsi=vTaskInputs.begin(); vsi!=vTaskInputs.end() && !bKnownInputTask; vsi++) {

                Channel * pChannel = *vsi;
                assert(pChannel != NULL);
                if(pChannel == NULL) continue;
                if(pChannel->GetType() == CHANNELTYPE::CT_GRAPH_INPUT) {

                    Port * pPort = pChannel->GetBoundPort(CE_DST);
                    assert(pPort != NULL);
                    if(pPort == NULL) continue;
                    if( pPort->IsInputParameter() && 
                        !pPort->IsInitializerParameter() && 
                        (!bFilterConstantEdgeNodes || !pPort->IsConstantSemantics())) {
                        bKnownInputTask = TRUE;
                        vTasks.insert(pTask);
                    }
                    
                }
            }
        }
        return static_cast<UINT>(vTasks.size());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   find the sets on input and output terminal tasks for control signal propagation
    ///             path synthesis. Input terminals are any task with exposed input channel (see
    ///             channel.h for a precise notion of "exposed"). find the set of terminal tasks
    ///             which must be reachable along some control propagation path for the given set of
    ///             control signals to correctly exercise all predicates in graph for the given (bit-
    ///             wise OR of) control signal(s). If no control signal is specified, this API will
    ///             first search the graph to collect the set of all control signals of interest.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/7/2014. </remarks>
    ///
    /// <param name="vInputTerminals">      [in,out] If non-null, the tasks. </param>
    /// <param name="vOutputTerminals">     [in,out] If non-null, the output terminals. </param>
    /// <param name="luiSignalsOfInterest"> the bitwise or of all signals of interest. </param>
    ///
    /// <returns>   the number of terminals found. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Graph::GetControlSignalTerminalsOfInterest(
        __inout std::set<Task*>& vInputTerminals,
        __inout std::set<Task*>& vOutputTerminals,
        __in    CONTROLSIGNAL luiSignalsOfInterest
        )
    {
        if(luiSignalsOfInterest == DBCTLC_NONE) 
            luiSignalsOfInterest = GetControlSignalsOfInterest(); 

        map<std::string, Task*>::iterator ti;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {

            // find all the exposed input and output edges a task is an exposed input if it has a graph
            // input channel which could be used as the source end of a control propagation path. Such
            // channels can be connected to either normal or control channels. A task is an output edge if
            // it has a graph output channel. Optionally, the criterion can be further restricted to
            // require a non-trivial predicate be defined for at least one channel on that node. 
            
            Task * pTask = ti->second; 
            CONTROLSIGNAL luiTaskSignalsOfInterest = pTask->GetControlSignalsOfInterest();
            if(luiSignalsOfInterest & luiTaskSignalsOfInterest)
                vOutputTerminals.insert(pTask);

            BOOL bKnownInputTask = FALSE;
            std::set<Channel*>::iterator vsi;
            std::set<Channel*> vInputChannels;
            std::set<Channel*> vOutputChannels;
            pTask->GetInboundChannels(vInputChannels);

            for(vsi=vInputChannels.begin(); vsi!=vInputChannels.end() && !bKnownInputTask; vsi++) {

                Channel * pChannel = *vsi;
                assert(pChannel != NULL);
                if(pChannel == NULL) continue;
                if(pChannel->GetType() == CHANNELTYPE::CT_GRAPH_INPUT) {

                    Port * pPort = pChannel->GetBoundPort(CE_DST);
                    assert(pPort != NULL);
                    if(pPort == NULL) continue;
                    bKnownInputTask |=  (pPort->IsInputParameter() && 
                                        !pPort->IsInitializerParameter() && 
                                        !pPort->IsConstantSemantics());
                    if(bKnownInputTask) {
                        assert(pPort->GetPortType() == INPUT_PORT);
                        vInputTerminals.insert(pTask);                    
                    }
                }
            }
        }
        return static_cast<UINT>(vInputTerminals.size() + vOutputTerminals.size());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synthesize a control propagation path that traverses the given list
    ///             of tasks. This was written as part of support for automatic inference/construction
    ///             of control propagaion paths, which is fundamentally heuristic. However,
    ///             if the programmer knows the path and is willing to trust the runtime to
    ///             make good choices about which ports and channels to use to synthesize a
    ///             path, it can be called from external code.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/3/2014. </remarks>
    ///
    /// <param name="vPath">    [in,out] [in,out] If non-null, full pathname of the file. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Graph::SynthesizeControlPropagationPath(
        __in std::vector<Task*>& vPath
        )
    {
        assert(vPath.size() > 0);
        if(vPath.size() == 0) return FALSE;
        
        Task * pCurrentTask = NULL;
        Task * pSuccessorTask = NULL;
        Channel * pPredecessorChannel = NULL;
        UINT uiPathLength = static_cast<UINT>(vPath.size());
        for(UINT ui=0; ui<uiPathLength; ui++) {

            pCurrentTask = vPath[ui];
            pSuccessorTask = (ui < uiPathLength - 1) ? vPath[ui+1] : NULL;
            pPredecessorChannel = (ui==0) ? NULL : pPredecessorChannel;
            if(!SynthesizeControlPropagationHop(pCurrentTask, &pPredecessorChannel, pSuccessorTask))
                return FALSE;
        }
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synthesize one hop in a control propagation path. This was written as part of
    ///             support for automatic inference/construction of control propagaion paths, which
    ///             is fundamentally heuristic. Do not call from external code.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/3/2014. </remarks>
    ///
    /// <param name="pCurrentTask">         [in,out] [in,out] If non-null, full pathname of the file. </param>
    /// <param name="ppPredecessorChannel"> [in,out] On entry, the inbound channel.
    ///                                              On exit, outbound channel selected on pCurrentTask to
    ///                                              extend the control propagation path across this
    ///                                              taskNode. </param>
    /// <param name="pSuccessorTask">       [in,out] If non-null, the immediate successor task in the
    ///                                     path. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Graph::SynthesizeControlPropagationHop(
        __in    Task *     pCurrentTask,
        __inout Channel ** ppPredecessorChannel,
        __in    Task *     pSuccessorTask
        )
    {
        assert(pCurrentTask);
        assert(ppPredecessorChannel != NULL);
        if(!pCurrentTask) return FALSE;
        
        Channel * pInboundChannel = *ppPredecessorChannel;
        if(pInboundChannel == NULL) {
            
            // this must be the first task on the propagation path,
            // so there is no incoming hop. So we want to select a
            // suitable exposed input channel. 

            std::set<Channel*> vInputs;
            std::set<Channel*>::iterator vsi;
            pCurrentTask->GetInboundChannels(vInputs);
            for(vsi=vInputs.begin(); vsi!=vInputs.end() && !pInboundChannel; vsi++) {
                Channel * pChannel = *vsi;
                if(pChannel != NULL && pChannel->GetType() == CHANNELTYPE::CT_GRAPH_INPUT) {
                    Port * pDstPort = pChannel->GetBoundPort(CE_DST);
                    assert(pDstPort != NULL);
                    pInboundChannel = ((pDstPort->GetPortType() == PORTTYPE::INPUT_PORT) && 
                                       (!pDstPort->IsSticky())) ? 
                                       pChannel : NULL;
                }
            }
        }

        // if we got to here, there should be a non-null inbound channel, either because it was handed
        // to us as part of the call, or because we searched the current task's bound inputs to make
        // the selection. If neither of these conditions hold, we cannot create the desired control
        // propagation path. The assert here might be a little too conservative, but it simplifies
        // debugging automatic path creation, and it's probably a quicker way to bring the problem to
        // the programmer's attention than by returning failure if we have been handed an a graph for
        // which the requested path cannot be created. The control propagation source for this hop
        // is the port on the dest end of the inbound channel. 
        
        assert(pInboundChannel != NULL);
        if(pInboundChannel == NULL)
            return FALSE;
        Port * pControlSourcePort = pInboundChannel->GetBoundPort(CE_DST); 
        assert(pControlSourcePort != NULL);
        assert(pControlSourcePort->GetPortType() == PORTTYPE::INPUT_PORT);
        assert(!pControlSourcePort->IsSticky());

        // to find an outbound port to complete the hop across this task we need to 
        // choose one.  If the successor task is null it means this is the last hop, 
        // so we need to choose one from amongst all available exposed output channels. 
        // If there are no such channels we will not create the hop, but this is not
        // necessarily failure, since it is not guaranteed that the final hop in the path 
        // must be an edge task. If there is a successor task speceifed choose one
        // from amongst all channels bound to that task at the dst end.

        std::set<Channel*>::iterator si;
        std::set<Channel*> vOutboundChannels;
        std::set<Channel*> vSelectedOutboundChannels;
        pCurrentTask->GetOutboundChannels(vOutboundChannels);
        for(si=vOutboundChannels.begin(); si!=vOutboundChannels.end(); si++) {

            Channel * pChannel = *si;
            Port * pPort = pChannel->GetBoundPort(CE_DST);
            CHANNELTYPE eChannelType = pChannel->GetType();
            if(pSuccessorTask == NULL && eChannelType == CT_GRAPH_OUTPUT && pChannel->HasNonTrivialPredicate())
                vSelectedOutboundChannels.insert(pChannel);
            else if(pPort != NULL && pPort->GetTask() == pSuccessorTask)
                vSelectedOutboundChannels.insert(pChannel);
        }

        // an empty outbound channel list means there is nothing
        // left to do. It's also an error if this isn't the terminal
        // hop on the requested control propagation path.

        if(vOutboundChannels.size() == 0) {

            // this is not necessarily a problem
            // for the terminal task. If it's not the terminal
            // task in the path, then we can't create the path.
            assert(pSuccessorTask == NULL);
            return pSuccessorTask == NULL;
        }

        // now we need to do the bindings for this hop. If this is the terminal 
        // task in the path, conservatively, we need to perform the requested binding
        // for *all* outbound channels. If this is an internal hop, then one channel
        // should be quite sufficient. 
        
            
        *ppPredecessorChannel = NULL;
        int nBindingsPerformed = 0;
        for(si=vSelectedOutboundChannels.begin(); si!=vSelectedOutboundChannels.end(); si++) {

            Channel * pChannel = *si;
            Port * pDstPort = pChannel->GetBoundPort(CE_SRC);
            assert(this == pChannel->GetGraph());
            assert(pDstPort != NULL);
            assert(pSuccessorTask != NULL || pChannel->GetType() == CT_GRAPH_OUTPUT);
            assert(pSuccessorTask == NULL || pChannel->GetBoundPort(CE_DST) != NULL);
            BindControlPropagationPort(pControlSourcePort, pDstPort);
            nBindingsPerformed++;
            
            *ppPredecessorChannel = pSuccessorTask == NULL ? NULL : pChannel;
            if(pSuccessorTask != NULL) 
                break;  // a single binding suffices 
                        // if this is not the last task
        }

        return nBindingsPerformed > 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   OBSOLETE! Use FindControlSignalPaths Searches for paths from input tasks (those
    ///             with exposed input channels)
    ///             to output tasks (those with exposed output channels). Returns a comprehensive
    ///             list. Since the primary use of this API is to find good candidate control
    ///             propagation paths, the bFilterUnpredicatedEdgeNodes tells the method whether or
    ///             not to include paths that terminate in tasks that have no non-trivial predication
    ///             on bound output channels. If bDiscardSuboptimalPaths is true, the output map will
    ///             not include paths with length greater than that of the shortest path for the
    ///             given input/output task. If bMapKeyedByTerminusTask is true, then the output map
    ///             is keyed by the endpoint of each path, otherwise by the input. If
    ///             bFilterConstantEdgeNodes is true, the input edge node detection will attempt to
    ///             filter out exposed input channels that are setup (apparently) to be used as
    ///             constants (bound to ports that have no inout binding and the sticky property).
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/1/2014. </remarks>
    ///
    /// <param name="vInputTasks">          [in,out] [in,out] If non-null, the input tasks. </param>
    /// <param name="vOutputTasks">         [in,out] [in,out] If non-null, the output tasks. </param>
    /// <param name="vPrioritizedPaths">    [in,out] [in,out] If non-null, the paths. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::FindControlSignalPaths(
        __inout std::set<Task*>& vInputTasks,
        __inout std::set<Task*>& vOutputTasks,
        __inout std::map<Task*, std::map<size_t, std::vector<std::vector<Task*>>>>& vPrioritizedPaths
        )
    {
        BOOL bDiscardSuboptimalPaths = TRUE;
        BOOL bMapKeyedByTerminusTask = TRUE;
        BOOL bVerbose = PTask::Runtime::IsVerbose();
        CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
        pTimer->reset();
        
        size_t uiPathsFound = 0;
        std::set<Task*>::iterator ssi;
        std::map<Task*, double> vTaskTimes;
        std::map<Task*, size_t> vTaskPaths;
        std::map<Task*, TASKPATHSTATS> vTaskPathStats;
        std::vector<std::vector<Task*>> vPaths;        
        double dPathFindingStart = pTimer->elapsed(false);
        for(ssi=vInputTasks.begin(); ssi!=vInputTasks.end(); ssi++) {

            double dTaskStart = pTimer->elapsed(false);
            std::set<Task*> vVisited;
            size_t oPathsFound = vPaths.size(); 
            FindAllOutboundPaths(*ssi, vVisited, vPaths, vOutputTasks);
            double dTaskEnd = pTimer->elapsed(false);
            vTaskTimes[*ssi] = dTaskEnd-dTaskStart;
            vTaskPaths[*ssi] = vPaths.size() - oPathsFound;
            uiPathsFound = vPaths.size();
        }
        PrioritizeOutboundPaths(vPaths, vPrioritizedPaths, vTaskPathStats, bDiscardSuboptimalPaths, bMapKeyedByTerminusTask);
        for(std::set<Task*>::iterator soi=vOutputTasks.begin(); soi!=vOutputTasks.end(); soi++) {
            // force creation of empty path list for any output tasks we didn't reach.
            std::map<size_t, std::vector<std::vector<Task*>>>& vTaskEntry=vPrioritizedPaths[*soi];
            if(vTaskEntry.size() != 0) {}
        }
        double dPathFindingEnd = pTimer->elapsed(false);

        if(bVerbose) {            

            // if so-requested, report on the endeavor        
            std::cout << "enumerated paths from " << vInputTasks.size() << " input edge tasks to "
                      << vOutputTasks.size() << " output edge tasks in " << dPathFindingEnd - dPathFindingStart 
                      << " msec." << std::endl << "enumerated " << uiPathsFound << " raw paths in aggregate.";
            std::map<Task*, double>::iterator mti;
            for(mti=vTaskTimes.begin(); mti!=vTaskTimes.end(); mti++) 
                std::cout << "   " << vTaskPaths[mti->first] << " paths for " << mti->first->GetTaskName() << ": " << mti->second << " msec." << std::endl;
            PrintOutboundPaths(std::cout, vPrioritizedPaths, vTaskPathStats);
        }

        delete pTimer;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   OBSOLETE! Use FindControlSignalPaths
    ///             Searches for paths from input tasks (those with exposed input channels)
    ///             to output tasks (those with exposed output channels). Returns a comprehensive
    ///             list. Since the primary use of this API is to find good candidate control
    ///             propagation paths, the bFilterUnpredicatedEdgeNodes tells the method whether or
    ///             not to include paths that terminate in tasks that have no non-trivial predication
    ///             on bound output channels. If bDiscardSuboptimalPaths is true, the output map will
    ///             not include paths with length greater than that of the shortest path for the
    ///             given input/output task. If bMapKeyedByTerminusTask is true, then the output map
    ///             is keyed by the endpoint of each path, otherwise by the input. If bFilterConstantEdgeNodes
    ///             is true, the input edge node detection will attempt to filter out exposed 
    ///             input channels that are setup (apparently) to be used as constants (bound to
    ///             ports that have no inout binding and the sticky property). 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/1/2014. </remarks>
    ///
    /// <param name="vInputTasks">                  [in,out] [in,out] If non-null, the input tasks. </param>
    /// <param name="vOutputTasks">                 [in,out] [in,out] If non-null, the output tasks. </param>
    /// <param name="vPrioritizedPaths">            [in,out] [in,out] If non-null, the paths. </param>
    /// <param name="bFilterUnpredicatedEdgeNodes"> (Optional) the filter unpredicated edge nodes. </param>
    /// <param name="bFilterConstantEdgeNodes">     (Optional) the filter constant edge nodes. </param>
    /// <param name="bDiscardSuboptimalPaths">      (Optional) the discard suboptimal paths. </param>
    /// <param name="bMapKeyedByTerminusTask">      (Optional) the map keyed by terminus task. </param>
    /// <param name="bVerbose">                     (Optional) the verbose. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::FindExposedPathTasksEx(
        __inout std::set<Task*>& vInputTasks,
        __inout std::set<Task*>& vOutputTasks,
        __inout std::map<Task*, std::map<size_t, std::vector<std::vector<Task*>>>>& vPrioritizedPaths,
        __in    BOOL bFilterUnpredicatedEdgeNodes,
        __in    BOOL bFilterConstantEdgeNodes,
        __in    BOOL bDiscardSuboptimalPaths,
        __in    BOOL bMapKeyedByTerminusTask,
        __in    BOOL bVerbose
        )
    {
        CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
        pTimer->reset();
        double dStart = pTimer->elapsed(false); 

        map<std::string, Task*>::iterator ti;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {

            // find all the exposed input and output edges a task is an exposed input if it has a graph
            // input channel which could be used as the source end of a control propagation path. Such
            // channels can be connected to either normal or control channels. A task is an output edge if
            // it has a graph output channel. Optionally, the criterion can be further restricted to
            // require a non-trivial predicate be defined for at least one channel on that node. 
            
            Task * pTask = ti->second; 
            std::map<UINT, Port*>::iterator mpi;
            std::map<UINT, Port*>* pInputPortMap = pTask->GetInputPortMap();
            std::map<UINT, Port*>* pOutputPortMap = pTask->GetOutputPortMap(); 
            BOOL bKnownInputTask = FALSE;
            BOOL bKnownOutputTask = FALSE;            
            for(mpi=pInputPortMap->begin(); mpi!=pInputPortMap->end() && !bKnownInputTask; mpi++) {

                Port * pIPort = mpi->second;
                UINT uiChannelCount = pIPort->GetChannelCount();
                UINT uiControlCount = pIPort->GetControlChannelCount(); 
                std::set<Channel*> vInputs;
                std::set<Channel*>::iterator vsi;
                for(UINT ui=0; ui<uiChannelCount; ui++) 
                    vInputs.insert(pIPort->GetChannel(ui));
                for(UINT ui=0; ui<uiControlCount; ui++) 
                    vInputs.insert(pIPort->GetControlChannel(ui));

                for(vsi=vInputs.begin(); vsi!=vInputs.end() && !bKnownInputTask; vsi++) {
                    Channel * pChannel = *vsi;
                    if(pChannel != NULL && pChannel->GetType() == CHANNELTYPE::CT_GRAPH_INPUT) {
                        Port * pDstPort = pChannel->GetBoundPort(CE_DST);
                        assert(pDstPort != NULL);
                        PORTTYPE ePortType = pDstPort->GetPortType();
                        if(ePortType == PORTTYPE::INPUT_PORT) {

                            // constraint: we are bound to an input port
                            // this effectively filters ports with semantics that
                            // make them a poor choice for propagation paths,
                            // namely, sticky ones. 
                            bKnownInputTask = TRUE; 
                            if(bFilterConstantEdgeNodes) {

                                // additionally try to detect situations where the user
                                // is using a traditional input channel -> input port construct
                                // in a way that is actually a constant; again stickiness is 
                                // problematic for control paths. 
                                InputPort * pIDstPort = dynamic_cast<InputPort*>(pDstPort);
                                bKnownInputTask = (pIDstPort->GetInOutConsumer() == NULL) &&
                                                   !pIDstPort->IsSticky();
                            }
                        }
                    }
                }
            }

            for(mpi=pOutputPortMap->begin(); mpi!=pOutputPortMap->end() && !bKnownOutputTask; mpi++) {
                Port * pOPort = mpi->second; 
                UINT uiChannelCount = pOPort->GetChannelCount(); 
                for(UINT ui=0; ui<uiChannelCount && !bKnownOutputTask; ui++) {
                    Channel * pChannel = pOPort->GetChannel(ui);
                    if(pChannel && pChannel->GetType() == CHANNELTYPE::CT_GRAPH_OUTPUT) {
                        bKnownOutputTask = TRUE;
                        if(bFilterUnpredicatedEdgeNodes) {
                            CHANNELPREDICATE eSrcPredicate = pChannel->GetPredicationType(CE_SRC);
                            CHANNELPREDICATE eDstPredicate = pChannel->GetPredicationType(CE_DST);
                            BOOL bPredicated = (eSrcPredicate != CGATEFN_NONE) || (eDstPredicate != CGATEFN_NONE);
                            BOOL bDevNull = (eSrcPredicate == CGATEFN_DEVNULL) || (eDstPredicate == CGATEFN_DEVNULL);
                            bKnownOutputTask = bPredicated && !bDevNull;
                        } 
                    }
                }
            }

            if(bKnownInputTask) vInputTasks.insert(pTask); 
            if(bKnownOutputTask) vOutputTasks.insert(pTask); 
        }
        double dEdgesFound = pTimer->elapsed(false); 

        // having found all edge nodes in the graph,
        // for each input edge, enumerate all paths to each output edge
        // update per task enumeration times to report later.
        
        size_t uiPathsFound = 0;
        std::set<Task*>::iterator ssi;
        std::map<Task*, double> vTaskTimes;
        std::map<Task*, size_t> vTaskPaths;
        std::map<Task*, TASKPATHSTATS> vTaskPathStats;
        std::vector<std::vector<Task*>> vPaths;        
        double dPathFindingStart = pTimer->elapsed(false);
        for(ssi=vInputTasks.begin(); ssi!=vInputTasks.end(); ssi++) {

            double dTaskStart = pTimer->elapsed(false);
            std::set<Task*> vVisited;
            size_t oPathsFound = vPaths.size(); 
            FindAllOutboundPaths(*ssi, vVisited, vPaths, vOutputTasks);
            double dTaskEnd = pTimer->elapsed(false);
            vTaskTimes[*ssi] = dTaskEnd-dTaskStart;
            vTaskPaths[*ssi] = vPaths.size() - oPathsFound;
            uiPathsFound = vPaths.size();
        }
        PrioritizeOutboundPaths(vPaths, vPrioritizedPaths, vTaskPathStats, bDiscardSuboptimalPaths, bMapKeyedByTerminusTask);
        for(std::set<Task*>::iterator soi=vOutputTasks.begin(); soi!=vOutputTasks.end(); soi++) {
            // force creation of empty path list for any output tasks we didn't reach.
            std::map<size_t, std::vector<std::vector<Task*>>>& vTaskEntry=vPrioritizedPaths[*soi];
            if(vTaskEntry.size() != 0) {}
        }
        double dPathFindingEnd = pTimer->elapsed(false);

        // if so-requested, report on the endeavor        
        if(bVerbose) {            
            std::cout << "enumerated edge nodes in " << dEdgesFound - dStart << " msec."  << std::endl;
            std::cout << "enumerated paths from " << vInputTasks.size() << " input edge tasks to "
                      << vOutputTasks.size() << " output edge tasks in " << dPathFindingEnd - dPathFindingStart 
                      << " msec." << std::endl << "enumerated " << uiPathsFound << " raw paths in aggregate.";
            std::map<Task*, double>::iterator mti;
            for(mti=vTaskTimes.begin(); mti!=vTaskTimes.end(); mti++) 
                std::cout << "   " << vTaskPaths[mti->first] << " paths for " << mti->first->GetTaskName() << ": " << mti->second << " msec." << std::endl;
            PrintOutboundPaths(std::cout, vPrioritizedPaths, vTaskPathStats);
        }

        // cleanup
        delete pTimer;

    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets immediate successors. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2014. </remarks>
    ///
    /// <param name="pTask">        [in,out] If non-null, the task. </param>
    /// <param name="vSuccessors">  [in,out] [in,out] If non-null, the successors. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::GetImmediateSuccessors(
        __in Task * pTask,
        __inout std::set<Task*>& vSuccessors
        )
    {
        assert(pTask != NULL);
        if(pTask == NULL) return; 

        std::map<UINT, Port*>::iterator mpi;
        std::map<UINT, Port*>* pOutputPortMap = pTask->GetOutputPortMap(); 
        for(mpi=pOutputPortMap->begin(); mpi!=pOutputPortMap->end(); mpi++) {
            Port * pOPort = mpi->second; 
            UINT uiChannelCount = pOPort->GetChannelCount(); 
            for(UINT ui=0; ui<uiChannelCount; ui++) {
                Channel * pChannel = pOPort->GetChannel(ui);
                Port * pDstPort = pChannel->GetBoundPort(CE_DST); 
                if(pDstPort != NULL) {
                    Task * pSuccessor = pDstPort->GetTask();
                    if(pSuccessor != NULL && pSuccessor != pTask)
                        vSuccessors.insert(pSuccessor);
                }
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets transitive closure of successors. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2014. </remarks>
    ///
    /// <param name="pTask">        [in,out] If non-null, the task. </param>
    /// <param name="vSuccessors">  [in,out] [in,out] If non-null, the successors. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::GetAllSuccessors(
        __in Task * pTask,
        __inout std::set<Task*>& vVisited,
        __inout std::set<Task*>& vSuccessors
        )
    {
        assert(pTask != NULL);
        if(pTask == NULL) return; 
        std::set<Task*>::iterator si;
        std::set<Task*> localSuccessors; 
        vVisited.insert(pTask); 
        GetImmediateSuccessors(pTask, localSuccessors); 
        for(si=localSuccessors.begin(); si!=localSuccessors.end(); si++) {
            Task * pSucc = *si; 
            vSuccessors.insert(pSucc); 
            if(vVisited.find(pSucc) == vVisited.end()) {
                GetAllSuccessors(pSucc, vVisited, vSuccessors);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets immediate predecessors. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2014. </remarks>
    ///
    /// <param name="pTask">        [in,out] If non-null, the task. </param>
    /// <param name="vSuccessors">  [in,out] [in,out] If non-null, the successors. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::GetImmediatePredecessors(
        __in Task * pTask,
        __inout std::set<Task*>& predecessors
        )
    {
        assert(pTask != NULL);
        if(pTask == NULL) return; 

        std::map<UINT, Port*>::iterator mpi;
        std::map<UINT, Port*>* pInputPortMap = pTask->GetInputPortMap();
        for(mpi=pInputPortMap->begin(); mpi!=pInputPortMap->end(); mpi++) {
            Port * pIPort = mpi->second;
            UINT uiChannelCount = pIPort->GetChannelCount();
            UINT uiControlCount = pIPort->GetControlChannelCount(); 
            for(UINT ui=0; ui<uiChannelCount; ui++) {
                Channel * pChannel = pIPort->GetChannel(ui);
                if(pChannel != NULL) {
                    Port * pPort = pChannel->GetBoundPort(CE_SRC); 
                    if(pPort != NULL) {
                        Task * pSrcTask = pPort->GetTask(); 
                        if(pSrcTask && pSrcTask != pTask)
                            predecessors.insert(pSrcTask);
                    }
                }
            }
            for(UINT ui=0; ui<uiControlCount; ui++) {
                Channel * pChannel = pIPort->GetControlChannel(ui);
                if(pChannel != NULL) {
                    Port * pPort = pChannel->GetBoundPort(CE_SRC); 
                    if(pPort != NULL) {
                        Task * pSrcTask = pPort->GetTask(); 
                        if(pSrcTask && pSrcTask != pTask)
                            predecessors.insert(pSrcTask);
                    }
                }
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets immediate successors. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2014. </remarks>
    ///
    /// <param name="pTask">        [in,out] If non-null, the task. </param>
    /// <param name="vSuccessors">  [in,out] [in,out] If non-null, the successors. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::GetAllPredecessors(
        __in Task * pTask,
        __inout std::set<Task*>& vVisited,
        __inout std::set<Task*>& vPredecessors
        )
    {
        assert(pTask != NULL);
        if(pTask == NULL) return; 
        std::set<Task*>::iterator si;
        std::set<Task*> localPreds; 
        vVisited.insert(pTask); 
        GetImmediatePredecessors(pTask, localPreds); 
        for(si=localPreds.begin(); si!=localPreds.end(); si++) {
            Task * pPred = *si; 
            vPredecessors.insert(pPred); 
            if(vVisited.find(pPred) == vVisited.end()) {
                GetAllPredecessors(pPred, vVisited, vPredecessors);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Finds all outbound paths from a given task to tasks on the "edge" of the graph. A
    ///             graph is on the edge if it has exposed output channels. An optional terminals of
    ///             interest parameter filters paths that terminate in edge task not in the specified
    ///             set of "interesting tasks".
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/1/2014. </remarks>
    ///
    /// <param name="pTask">                [in,out] If non-null, the task. </param>
    /// <param name="vVisited">             [in,out] [in,out] If non-null, the successors. </param>
    /// <param name="vPaths">               [in,out] [in,out] If non-null, the paths. </param>
    /// <param name="vTerminalsOfInterest"> [in,out] (Optional) If non-null, (Optional) the terminals
    ///                                     of interest. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::FindAllOutboundPaths(
        __in    Task * pTask,
        __inout std::set<Task*>& vVisited,
        __inout std::vector<std::vector<Task*>>& vPaths,
        __in    std::set<Task*>& vTerminalsOfInterest
        )
    {
        assert(vTerminalsOfInterest.size() > 0);
        assert(pTask != NULL);
        if(pTask == NULL) return; 
        std::set<Task*>::iterator si;
        std::set<Task*> localSuccessors; 

        if(vVisited.find(pTask) != vVisited.end())
            return;

        // collect all paths from all successors to 
        // terminals of interest....
        vVisited.insert(pTask);         
        GetImmediateSuccessors(pTask, localSuccessors); 
        std::vector<std::vector<Task*>> succOutbound;
        std::set<Task*> localSuccsVisited;
        for(si=localSuccessors.begin(); si!=localSuccessors.end(); si++) {
            Task * pSucc = *si; 
            if(pSucc != pTask && vVisited.find(pSucc) == vVisited.end()) {
                std::set<Task*> succVisited;
                succVisited.insert(vVisited.begin(), vVisited.end());
                succVisited.insert(localSuccsVisited.begin(), localSuccsVisited.end());
                FindAllOutboundPaths(pSucc, succVisited, succOutbound, vTerminalsOfInterest);
                localSuccsVisited.insert(pSucc);
            }
        }

        // for each outbound path from a successor, add a path entry to vPaths that prepends pTask to
        // that path. if there are no such outbound paths, then this task is a terminus of the search.
        // In that case, add a path of length one including this task (assuming it is a terminal of
        // interest). 

        if(succOutbound.size() > 0) {

            std::vector<std::vector<Task*>>::iterator vi; 
            for(vi=succOutbound.begin(); vi!=succOutbound.end(); vi++) {

                // add a new path that concatenates this
                // task with the successor path. 
                std::vector<Task*>::iterator vsi;
                std::vector<Task*> pPath;
                std::vector<Task*>& pSuccPath = *vi;
                pPath.push_back(pTask);
                for(vsi=pSuccPath.begin(); vsi!=pSuccPath.end(); vsi++) {
                    pPath.push_back(*vsi);
                }
                vPaths.push_back(pPath);
            }

        } 

        // If this node is a terminus we are interested in, add a path of length one that includes it,
        // since we want paths that end with this one, even if it is part of a path that leads to
        // another terminus that we are interested in. This is a common pattern in DNN graphs.
        if(vTerminalsOfInterest.find(pTask) != vTerminalsOfInterest.end()) {
            std::vector<Task*> pPath;
            pPath.push_back(pTask); 
            vPaths.push_back(pPath);
        }        
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Print outbound paths. </summary>
    ///
    /// <remarks>   crossbac, 7/2/2014. </remarks>
    ///
    /// <param name="ios">                  [in,out] The ios. </param>
    /// <param name="vPrioritizedPaths">    [in,out] [in,out] If non-null, the paths. </param>
    /// <param name="vPathStats">           [in,out] [in,out] If non-null, the path stats. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::PrintOutboundPaths(
        __inout std::ostream &ios,
        __in    std::map<Task*, std::map<size_t, std::vector<std::vector<Task*>>>>& vPrioritizedPaths,
        __in    std::map<Task*, TASKPATHSTATS>& vPathStats
        )
    {
        std::map<size_t, std::vector<std::vector<Task*>>>::iterator vmpi;
        std::map<Task*, std::map<size_t, std::vector<std::vector<Task*>>>>::iterator vppi;
        for(vppi=vPrioritizedPaths.begin(); vppi!=vPrioritizedPaths.end(); vppi++) {

            Task * pEndpoint = vppi->first;
            TASKPATHSTATS& stats = vPathStats[pEndpoint];
            ios << pEndpoint->GetTaskName() << ":" << std::endl
                << "  OPT=" << stats.uiOptimalLength << "(" << stats.uiOptimalPathCount << " unique) " 
                << " max=" << stats.uiMaxPathLength << " tot: " << stats.uiTotalPathCount
                << std::endl;
            
            for(vmpi=vppi->second.begin(); vmpi!=vppi->second.end(); vmpi++) {

                std::vector<std::vector<Task*>>& vPaths = vmpi->second;
                std::vector<std::vector<Task*>>::iterator vi; 
                for(vi=vPaths.begin(); vi!=vPaths.end(); vi++) {
                    std::vector<Task*>::iterator vsi;
                    std::vector<Task*>& pPath = *vi;
                    ios << "  * ";
                    bool bFirst = true;
                    int nHops = 0;
                    for(vsi=pPath.begin(); vsi!=pPath.end(); vsi++) {
                        if(!bFirst) ios << " -> ";
                        bFirst = false;
                        if(++nHops % 4 == 0)
                            ios << std::endl << "    ";
                        ios << (*vsi)->GetTaskName();
                    }
                    ios << std::endl;
                }
            }
        }       
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Prioritize outbound paths. Based on the vector of paths, create a map from Task
    ///             to paths prioritized to prefer shorter paths. The bMapKeyedByTerminusTask
    ///             controls whether the keys in the map are the terminus or the origin of the task.
    ///             If bDiscardSuboptimalPaths is true, then paths with length greater than the
    ///             shortest are discarded.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/2/2014. </remarks>
    ///
    /// <param name="vPaths">                   [in,out] [in,out] If non-null, the paths. </param>
    /// <param name="vPrioritizedPaths">        [in,out] [in,out] If non-null, the prioritized paths. </param>
    /// <param name="vPathStats">               [in,out] [in,out] If non-null, the path stats. </param>
    /// <param name="bDiscardSuboptimalPaths">  The discard suboptimal paths. </param>
    /// <param name="bMapKeyedByTerminusTask">  The map keyed by terminus task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::PrioritizeOutboundPaths(
        __in    std::vector<std::vector<Task*>>& vPaths,
        __inout std::map<Task*, std::map<size_t, std::vector<std::vector<Task*>>>>& vPrioritizedPaths,
        __inout std::map<Task*, TASKPATHSTATS>& vPathStats,
        __in    BOOL bDiscardSuboptimalPaths,
        __in    BOOL bMapKeyedByTerminusTask
        )
    {
        std::vector<std::vector<Task*>>::iterator vvi;
        std::map<Task*, std::map<size_t, std::vector<std::vector<Task*>>>>::iterator vppi;
        std::map<Task*, size_t> vTaskMinPathLengths;
        std::map<Task*, std::set<size_t>> vStaleOptima;
        std::map<Task*, size_t>::iterator vpli;
        std::map<Task*, TASKPATHSTATS>::iterator sti;

        // since this method can be called multiple times to update the priority map, there is no
        // guarantee that sub-optimal entries were actually created here, so keeping a valid min
        // lengths entry and a "stale optimum map" requires us to examine the state of the priority map
        // on entry to ensure we catch any previous entries that are made stale by this update. 
        
        if(bDiscardSuboptimalPaths) {

            for(vppi=vPrioritizedPaths.begin(); vppi!=vPrioritizedPaths.end(); vppi++) {
                Task * pTask = vppi->first;
                if(vPathStats.find(pTask) == vPathStats.end()) 
                    vPathStats[pTask] = TASKPATHSTATS(pTask);
                std::map<size_t, std::vector<std::vector<Task*>>>& vTaskEntry = vppi->second;
                std::map<size_t, std::vector<std::vector<Task*>>>::iterator vmpi;
                for(vmpi=vTaskEntry.begin(); vmpi!=vTaskEntry.end(); vmpi++) {
                    size_t uiPathLength = vmpi->first;
                    std::map<Task*, size_t>::iterator mpli = vTaskMinPathLengths.find(pTask);
                    size_t uiCurrentMin = (mpli==vTaskMinPathLengths.end()) ? uiPathLength : mpli->second;
                    vTaskMinPathLengths[pTask] = min(uiPathLength, uiCurrentMin); 
                    vPathStats[pTask].uiMaxPathLength = max(uiPathLength, vPathStats[pTask].uiMaxPathLength);
                    vPathStats[pTask].uiOptimalLength = min(uiPathLength, uiCurrentMin); 
                    if(uiPathLength < uiCurrentMin)
                        vStaleOptima[pTask].insert(uiCurrentMin);
                }
            }
        }

        
        // go through all known paths and insert them keyed by task and secondarily by path length in
        // the map. if we have been asked to discard sub-optimal paths, try to avoid inserting anything
        // with length greater than the current known best for each task. Since we can encounter paths
        // in an order that requires us to update our current known best, we will do a cleanup pass
        // later if necessary to remove sub-optimal paths. 
        
        for(vvi=vPaths.begin(); vvi!=vPaths.end(); vvi++) {

            // find the task by which this path should be
            // keyed, and figure out the secondary key: path length.
            
            std::vector<Task*>& path = *vvi;
            size_t uiPathLength = path.size(); 
            Task * pKeyTask = bMapKeyedByTerminusTask ? (*(path.rbegin())) : (*(path.begin()));
            BOOL bShouldInsertEntry = TRUE;   // can be negated if we are discarding sub opts.
            vPathStats[pKeyTask].uiMaxPathLength = max(uiPathLength, vPathStats[pKeyTask].uiMaxPathLength);
            vPathStats[pKeyTask].uiOptimalLength = min(uiPathLength, vPathStats[pKeyTask].uiOptimalLength);
            vPathStats[pKeyTask].uiTotalPathCount++;

            if(bDiscardSuboptimalPaths) {

                // figure out if this path is sub-optimal according to our 
                // current view, and avoid inserting it if so. if we have
                // to update our current view the optimal length, we will
                // clean up stale entries later. 
                
                vpli = vTaskMinPathLengths.find(pKeyTask);
                if(vpli != vTaskMinPathLengths.end()) { 
                    size_t uiCurrentTaskOptimal = vpli->second;
                    bShouldInsertEntry = (uiPathLength <= uiCurrentTaskOptimal);
                    BOOL bStaleOptimum = uiPathLength < uiCurrentTaskOptimal;
                    if(bStaleOptimum) 
                        vStaleOptima[pKeyTask].insert(uiCurrentTaskOptimal);
                }
                if(bShouldInsertEntry) 
                    vTaskMinPathLengths[pKeyTask] = uiPathLength; 
            }
            if(bShouldInsertEntry) 
                vPrioritizedPaths[pKeyTask][uiPathLength].push_back(path);
        }

        if(bDiscardSuboptimalPaths) {
            
            // go through the map and weed out any path entries that were 
            // entered before we encountered the true optima for each keyed task. 
            
            if(vStaleOptima.size() > 0) {
                std::map<Task*, std::set<size_t>>::iterator soi;
                for(soi=vStaleOptima.begin(); soi!=vStaleOptima.end(); soi++) {

                    Task * pTask = soi->first;
                    vppi = vPrioritizedPaths.find(pTask);
                    assert(vppi!=vPrioritizedPaths.end());
                    std::map<size_t, std::vector<std::vector<Task*>>>& vTaskPaths = vppi->second;
                    std::set<size_t>::iterator ssoi;
                    for(ssoi=soi->second.begin(); ssoi!=soi->second.end(); ssoi++) {
                        size_t uiSizeToRemove = *ssoi;
                        vTaskPaths.erase(uiSizeToRemove);
                    }
                    assert(vTaskPaths.size() == 1);
                }
            }
        }

        // update stats
        for(vppi=vPrioritizedPaths.begin(); vppi!=vPrioritizedPaths.end(); vppi++) {
            Task * pTask = vppi->first;
            size_t uiTaskOptimal = vTaskMinPathLengths[pTask];
            size_t uiOptimalPathCount = vppi->second[uiTaskOptimal].size();
            vPathStats[pTask].uiOptimalPathCount = uiOptimalPathCount;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check to see if the graph is well-formed. This is not an exhaustive check, but a
    ///             collection of obvious sanity checks. If the bFailOnWarning flag is set, then the
    ///             runtime will exit the process if it finds anything wrong with the graph.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="bVerbose">         true for verbose console output. </param>
    /// <param name="bFailOnWarning">   (optional) fail on warning flag: if set, exit the process
    ///                                 when malformed graph elements are found. </param>
    ///
    /// <returns>   PTRESULT--use PTSUCCESS/PTFAILED macros:
    /// 			PTASK_OK:   the graph is well-formed 
    ///             PTASK_ERR_GRAPH_MALFORMED: the graph is malformed in a way that cannot be
    ///                                        tolerated by the runtime. Or the the issue may be
    ///                                        tolerable but the user requested fail on warning.
    ///             PTASK_WARNING_GRAPH_MALFORMED: the graph is malformed in a way that can be
    ///                                        tolerated by the runtime. 
    /// 			</returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT 
    Graph::CheckGraphSemantics(
        BOOL bVerbose,
        BOOL bFailOnWarning
        )
    {
        Lock();

        std::stringstream ss;
        UINT nErrors = 0;
        UINT nWarnings = 0;
        map<Channel*, Port*>::iterator cpi;
        size_t nTaskCount = m_vTasks.size();
        size_t nChannelCount = m_vChannels.size();
        size_t nPortCount = m_pPortMap.size();

        // are there actual Tasks in the graph?
        // if not the graph is well-formed, but trivially so, and
        // a warning is most likely in order.
        if(!nTaskCount) {
            nWarnings++;
            ss << "No tasks are in the graph. It is well-formed, but "
                << "not useful for computation. Was this the intent?"
                << endl;
        }

        // if there are nodes but no channels, this is
        // an obvious problem--no communication can occur.
        if(nChannelCount == 0 && nTaskCount > 0) {
            nErrors++;
            ss << "No channels are in the graph. It is well-formed, but "
                << "the nodes cannot communicate." 
                << endl;
        }

        // if there are nodes but no ports, this is
        // an obvious problem--no communication can occur.
        if(nPortCount == 0 && nTaskCount > 0) {
            nErrors++;
            ss << "No ports are in the graph. It is well-formed, but "
                << "the nodes cannot communicate." 
                << endl;
        }

        // check that every task has at least one input. A meta port, constant port, or initializer
        // port can be an input port. Initializer ports are stored in the input port map though, so we
        // need no explicit map to traverse that part of the structure. Technically, we allow tasks
        // with no outputs, but it is probably worth warning the user when there is a node with no
        // output. 
        std::map<std::string, Task*>::iterator ti;
        std::map<UINT, Port*>::iterator portiter;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {
            UINT nTotalInputs = 0;
            UINT nTotalOutputs = 0;
            Task * pTask = ti->second;
            vector<Port*> vAllInputs;
            for(portiter=pTask->GetInputPortMap()->begin(); portiter!=pTask->GetInputPortMap()->end(); portiter++) 
                vAllInputs.push_back(portiter->second);
            for(portiter=pTask->GetConstantPortMap()->begin(); portiter!=pTask->GetConstantPortMap()->end(); portiter++) 
                vAllInputs.push_back(portiter->second);
            for(portiter=pTask->GetMetaPortMap()->begin(); portiter!=pTask->GetMetaPortMap()->end(); portiter++) 
                vAllInputs.push_back(portiter->second);
            for(vector<Port*>::iterator pi=vAllInputs.begin(); pi!=vAllInputs.end(); pi++) {
                nTotalInputs++;
                Port* pPort = *pi;
                if(pPort->GetTask() != pTask) {
                    nErrors++;
                    ss << pPort 
                        << "has a malformed binding to " << pTask
                        << endl;
                }
                // every port must be bound to at least one channel,
                // unless it is an initializer port. An initializer
                // port should not be bound to a channel. 
                if(!pPort->CheckSemantics(&ss, this)) {
                    nErrors++;
                }
            }

            for(portiter=pTask->GetOutputPortMap()->begin();
                portiter!=pTask->GetOutputPortMap()->end();
                portiter++) 
            {
                nTotalOutputs++;
                Port* pPort = portiter->second;
                if(pPort->GetTask() != pTask) {
                    nErrors++;
                    ss << pPort 
                        << "has a malformed binding to " << pTask
                        << endl;
                }
                // every port must be bound to at least one channel,
                // unless it is an initializer port. An initializer
                // port should not be bound to a channel. 
                if(!pPort->CheckSemantics(&ss, this)) {
                    nErrors++;
                }
            }

            if(nTotalInputs == 0) {
                nWarnings++;
                ss << pTask 
                    << "has no ports consuming input "
                    << endl;
            }

            if(nTotalOutputs == 0) {
                nWarnings++;
                ss << pTask 
                    << "has no ports producing Output "
                    << endl;
            }

        }

        // check that all destination port bindings are 
        // in correctly formed. This means that the channel port destination
        // map should have channels whose bound port is the same as the 
        for(cpi=m_pChannelDstMap.begin(); cpi!=m_pChannelDstMap.end(); cpi++) {
            Channel * pChannel = cpi->first;
            Port * pPort = cpi->second;
            if(pChannel->GetBoundPort(CE_DST) != pPort) {
                ss << pChannel
                    << " binding to Port(" << pPort->GetVariableBinding() << ")"
                    << " is malformed " << endl;
                nErrors++;
            }
            if(!pChannel->CheckSemantics(&ss, this)) {
                nErrors++;
            }
        }

        // check that all source port bindings are 
        // in correctly formed. This means that the channel port source
        // map should have channels whose bound port is the same as the entry  
        for(cpi=m_pChannelSrcMap.begin(); cpi!=m_pChannelSrcMap.end(); cpi++) {
            Channel * pChannel = cpi->first;
            Port * pPort = cpi->second;
            if(pChannel->GetBoundPort(CE_SRC) != pPort) {
                ss << pChannel
                    << " binding to Port(" << pPort->GetVariableBinding() << ")"
                    << " is malformed " << endl;
                nErrors++;
            }
            if(!pChannel->CheckSemantics(&ss, this)) {
                nErrors++;
            }
        }

        Unlock();

        if(bVerbose && ss.str().length() > 0) {
            PTask::Runtime::Warning(ss.str());
        }

        if(nErrors > 0) {
            return PTASK_ERR_GRAPH_MALFORMED;
        }
        
        if(nWarnings > 0) {
            return bFailOnWarning ?
                PTASK_ERR_GRAPH_MALFORMED :
                PTASK_WARNING_GRAPH_MALFORMED;
        }

        return PTASK_OK;
    }

#ifdef DISPATCH_COUNT_DIAGNOSTICS

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an expected invocation counts. </summary>
    ///
    /// <remarks>   Crossbac, 2/27/2012. </remarks>
    ///
    /// <param name="pvInvocationCounts">   [in,out] If non-null, the pv invocation counts. </param>
    /// <param name="nScalar">              The scalar. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::SetExpectedDispatchCounts(
        std::map<std::string, UINT> * pvInvocationCounts,
        UINT nScalar
        )
    {
        m_vDispatchCounts = pvInvocationCounts;
        std::map<std::string, UINT>::iterator vi;
        for(vi=pvInvocationCounts->begin(); vi!=pvInvocationCounts->end(); vi++) {
            std::map<std::string, Task*>::iterator ti=m_vTasks.find(vi->first);
            if(ti!=m_vTasks.end()) {
                Task * pTask = ti->second;
                pTask->SetExpectedDispatchCount(vi->second * nScalar);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the block count for blocks queued in channels in the graph. 
    /// 			Optionally provide per-channel counts.</summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="pvOutstanding">    [in,out] If non-null, the pv outstanding. </param>
    ///
    /// <returns>   The oustanding block count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Graph::GetOustandingBlockCounts(
        std::map<Channel*, UINT> * pvOutstanding
        )
    {
        UINT nOutstanding = 0;
        std::map<std::string, Channel*>::iterator mi;
        for(mi=m_vChannels.begin(); mi!=m_vChannels.end(); mi++) {
            Channel * pChannel = mi->second;
            pChannel->Lock();
            UINT nChannelOutstanding = (UINT) pChannel->GetQueueDepth();
            nOutstanding += nChannelOutstanding;
            if(pvOutstanding && nChannelOutstanding) {
                (*pvOutstanding)[pChannel] = nChannelOutstanding;
            }
            pChannel->Unlock();
        }
        return nOutstanding;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Find all ports with non-zero control codes. When a graph is stopped or in some
    ///             quiescent state, it should generally be the case that no active control codes are
    ///             left lingering: this kind of situation can lead to control codes associated with
    ///             a previous stream or iteration affecting control flow for subsequent ones, which
    ///             is both undesirable and extremely hard to debug.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/2/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::CheckPortControlCodes(
        VOID
        )
    {
        std::map<std::string, Task*>::iterator ti;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {
            Task * pTask = ti->second;
            pTask->CheckPortControlCodes();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Find all channels with non-zero control codes. When a graph is stopped or in some
    ///             quiescent state, it should generally be the case that no active control codes are
    ///             left lingering: this kind of situation can lead to control codes associated with
    ///             a previous stream or iteration affecting control flow for subsequent ones, which
    ///             is both undesirable and extremely hard to debug.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/2/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------
        
    void 
    Graph::CheckChannelControlCodes(
        VOID
        )
    {
        std::map<std::string, Channel*>::iterator ti;
        for(ti=m_vChannels.begin(); ti!=m_vChannels.end(); ti++) {
            Channel * pChannel = ti->second;
            UINT uiCode = pChannel->GetPropagatedControlCode();
            if(uiCode != DBCTL_NONE) {
                std::cout << pChannel << " has control code " << std::hex << uiCode << std::dec << std::endl;
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check that block pools contain only datablocks with no control signals. </summary>
    ///
    /// <remarks>   Crossbac, 3/2/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::CheckBlockPoolStates(
        VOID
        )
    {
        std::map<std::string, Task*>::iterator ti;
        for(ti=m_vTasks.begin(); ti!=m_vTasks.end(); ti++) {
            Task * pTask = ti->second;
            pTask->CheckBlockPoolStates();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Traverse a (supposedly) quiescent graph looking for active control signals </summary>
    ///
    /// <remarks>   Crossbac, 3/2/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::CheckControlSignals(
        VOID
        )
    {
        Lock();
        CheckPortControlCodes();
        CheckChannelControlCodes();
        CheckBlockPoolStates();
        Unlock();
    }

#endif

#ifdef PROFILE_GRAPHS
    void Graph::OnTaskThreadAlive()                         { if(m_pGraphProfiler) m_pGraphProfiler->OnTaskThreadAlive();                    }
    void Graph::OnTaskThreadExit()                          { if(m_pGraphProfiler) m_pGraphProfiler->OnTaskThreadExit();                     }
    void Graph::OnTaskThreadBlockRunningGraph()             { if(m_pGraphProfiler) m_pGraphProfiler->OnTaskThreadBlockRunningGraph();        }
    void Graph::OnTaskThreadWakeRunningGraph()              { if(m_pGraphProfiler) m_pGraphProfiler->OnTaskThreadWakeRunningGraph();         }
    void Graph::OnTaskThreadBlockTasksAvailable()           { if(m_pGraphProfiler) m_pGraphProfiler->OnTaskThreadBlockTasksAvailable();      }
    void Graph::OnTaskThreadWakeTasksAvailable()            { if(m_pGraphProfiler) m_pGraphProfiler->OnTaskThreadWakeTasksAvailable();       }
    void Graph::OnTaskThreadDequeueAttempt()                { if(m_pGraphProfiler) m_pGraphProfiler->OnTaskThreadDequeueAttempt();           }
    void Graph::OnTaskThreadDequeueComplete(Task * pTask)   { if(m_pGraphProfiler) m_pGraphProfiler->OnTaskThreadDequeueComplete(pTask);     }
    void Graph::OnTaskThreadDispatchAttempt()               { if(m_pGraphProfiler) m_pGraphProfiler->OnTaskThreadDispatchAttempt();          }
    void Graph::OnTaskThreadDispatchComplete(BOOL bSuccess) { if(m_pGraphProfiler) m_pGraphProfiler->OnTaskThreadDispatchComplete(bSuccess); }
#else
    void Graph::OnTaskThreadAlive()                         { }
    void Graph::OnTaskThreadExit()                          { }
    void Graph::OnTaskThreadBlockRunningGraph()             { }
    void Graph::OnTaskThreadWakeRunningGraph()              { }
    void Graph::OnTaskThreadBlockTasksAvailable()           { }
    void Graph::OnTaskThreadWakeTasksAvailable()            { }
    void Graph::OnTaskThreadDequeueAttempt()                { }
    void Graph::OnTaskThreadDequeueComplete(Task * pTask)   { UNREFERENCED_PARAMETER(pTask); }
    void Graph::OnTaskThreadDispatchAttempt()               { }
    void Graph::OnTaskThreadDispatchComplete(BOOL bSuccess) { UNREFERENCED_PARAMETER(bSuccess); }
#endif

};