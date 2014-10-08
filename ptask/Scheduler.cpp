///-------------------------------------------------------------------------------------------------
// file:	Scheduler.cpp
//
// summary:	Implements the scheduler class
///-------------------------------------------------------------------------------------------------

#include "Scheduler.h"
#include "port.h"
#include "datablock.h"
#include "dxaccelerator.h"
#include "cuaccelerator.h"
#include "claccelerator.h"
#include "hostaccelerator.h"
#include "task.h"
#include "PTaskRuntime.h"
#include "graph.h"
#include "hrperft.h"
#include "nvtxmacros.h"
#include "cuhdr.h"
#include "instrumenter.h"
#include <iomanip>
#include <vector>
#include <algorithm>
#include <assert.h>
using namespace std;

#define QueueLockIsHeld()       (m_nQueueLockDepth > 0)
#define AcceleratorLockIsHeld() (m_nAcceleratorMapLockDepth > 0)
#define QuiescenceLockIsHeld()  (m_nQuiescenceLockDepth > 0)

#ifdef DEBUG
#define CHECK_MANDATORY_ASSIGNMENTS(a,b,c,d)    __CheckDispatchConstraints(a,b,c,d)
#else
#define CHECK_MANDATORY_ASSIGNMENTS(a,b,c,d) 
#endif // DEBUG

namespace PTask {

#pragma warning(disable:4503)

    Scheduler * g_pScheduler = NULL;
    SCHEDULINGMODE g_ePendingMode = SCHEDMODE_DATADRIVEN;
    BOOL g_bPendingScheduleModeChange = FALSE;
    UINT uiSchedThreadIdx = 0;
    UINT uiSchedInitCallsPerAS = 0;
    std::map<ACCELERATOR_CLASS, std::set<int>*> Scheduler::s_vDenyListDeviceIDs;
    std::map<ACCELERATOR_CLASS, std::set<int>*> Scheduler::s_vAllowListDeviceIDs;


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Ptask scheduler thread. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="p">    The p. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD WINAPI 
    ptask_scheduler_thread(
        LPVOID p
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Scheduler::Scheduler(
        VOID
        )
    {
        m_eSchedulingMode = SCHEDMODE_DATADRIVEN;
        if(g_bPendingScheduleModeChange) {
            g_bPendingScheduleModeChange = FALSE;
            m_eSchedulingMode = g_ePendingMode;
        }
        m_bAlive = TRUE;
        m_bLiveGraphs = FALSE;
        m_hTaskQueue = CreateEvent(NULL, FALSE, FALSE, L"PTask::Scheduler::TaskQueueEvent");
        m_hSchedulerTerminateEvent = CreateEvent(NULL, TRUE, FALSE, L"PTask::Scheduler::SchedulerTerminateEvent");
        m_hGlobalQuiescentEvent = CreateEvent(NULL, TRUE, FALSE, L"PTask::Scheduler::QuiescentEvent");
        m_hOKToSchedule = CreateEvent(NULL, TRUE, FALSE, L"PTask::Scheduler::OKToScheduleEvent");
        m_hAcceleratorQueue = CreateEvent(NULL, FALSE, FALSE, L"PTask::Scheduler::AcceleratorQueue");
        m_hRuntimeTerminateEvent = PTask::Runtime::GetRuntimeTerminateEvent();

        m_bOKToSchedule = FALSE;
        m_bTasksAvailable = FALSE;        
        m_bCrossRuntimeSharingChecksRequired = FALSE;
        InitializeCriticalSection(&m_csQ);
        InitializeCriticalSection(&m_csAccelerators);
        InitializeCriticalSection(&m_csStatistics);
        InitializeCriticalSection(&m_csDispatchTimestamp);
        InitializeCriticalSection(&m_csQuiescentState);
        m_bQuiescenceInProgress = FALSE;
        m_dwLastDispatchTimestamp = 0xFFFFFFFF;
        m_nQueueLockDepth = 0;
        m_nAcceleratorMapLockDepth = 0;
        m_nQuiescenceLockDepth = 0;
        m_pDeviceManager = new AcceleratorManager();

        m_vhAcceleratorsAvailable[ACCELERATOR_CLASS_DIRECT_X]  = CreateEvent(NULL, TRUE, FALSE, L"PTask::Scheduler::DXAcceleratorAvailableEvent");
        m_vhAcceleratorsAvailable[ACCELERATOR_CLASS_OPEN_CL]   = CreateEvent(NULL, TRUE, FALSE, L"PTask::Scheduler::CLAcceleratorAvailableEvent");
        m_vhAcceleratorsAvailable[ACCELERATOR_CLASS_CUDA]      = CreateEvent(NULL, TRUE, FALSE, L"PTask::Scheduler::CUAcceleratorAvailableEvent");
        m_vhAcceleratorsAvailable[ACCELERATOR_CLASS_REFERENCE] = CreateEvent(NULL, TRUE, FALSE, L"PTask::Scheduler::RefAcceleratorsAvailableEvent");
        m_vhAcceleratorsAvailable[ACCELERATOR_CLASS_HOST]      = CreateEvent(NULL, TRUE, FALSE, L"PTask::Scheduler::HostAcceleratorsAvailableEvent");
        m_vhAcceleratorsAvailable[ACCELERATOR_CLASS_SUPER]     = CreateEvent(NULL, TRUE, FALSE, L"PTask::Scheduler::SuperAcceleratorsAvailableEvent");

        m_vEnabledAccelerators[ACCELERATOR_CLASS_DIRECT_X]     = new std::set<Accelerator*>();
        m_vEnabledAccelerators[ACCELERATOR_CLASS_OPEN_CL]      = new std::set<Accelerator*>();
        m_vEnabledAccelerators[ACCELERATOR_CLASS_CUDA]         = new std::set<Accelerator*>();
        m_vEnabledAccelerators[ACCELERATOR_CLASS_REFERENCE]    = new std::set<Accelerator*>();
        m_vEnabledAccelerators[ACCELERATOR_CLASS_HOST]         = new std::set<Accelerator*>();
        m_vEnabledAccelerators[ACCELERATOR_CLASS_SUPER]        = new std::set<Accelerator*>();
        
        m_vAvailableAccelerators[ACCELERATOR_CLASS_DIRECT_X]   = new std::set<Accelerator*>();
        m_vAvailableAccelerators[ACCELERATOR_CLASS_OPEN_CL]    = new std::set<Accelerator*>();
        m_vAvailableAccelerators[ACCELERATOR_CLASS_CUDA]       = new std::set<Accelerator*>();
        m_vAvailableAccelerators[ACCELERATOR_CLASS_REFERENCE]  = new std::set<Accelerator*>();
        m_vAvailableAccelerators[ACCELERATOR_CLASS_HOST]       = new std::set<Accelerator*>();
        m_vAvailableAccelerators[ACCELERATOR_CLASS_SUPER]      = new std::set<Accelerator*>();
        
        m_vbAcceleratorsAvailable[ACCELERATOR_CLASS_DIRECT_X]  = FALSE;
        m_vbAcceleratorsAvailable[ACCELERATOR_CLASS_OPEN_CL]   = FALSE;
        m_vbAcceleratorsAvailable[ACCELERATOR_CLASS_CUDA]      = FALSE;
        m_vbAcceleratorsAvailable[ACCELERATOR_CLASS_REFERENCE] = FALSE;
        m_vbAcceleratorsAvailable[ACCELERATOR_CLASS_HOST]      = FALSE;
        m_vbAcceleratorsAvailable[ACCELERATOR_CLASS_SUPER]     = FALSE;

        InterlockedIncrement(&uiSchedInitCallsPerAS);
        CreateAccelerators();
        m_uiThreadCount = Runtime::GetSchedulerThreadCount();
        if(m_uiThreadCount == 0) {
            PTask::Runtime::HandleError("%s::%s(%d): scheduler created with thread count of 0! no tasks will ever get scheduled!\n",
                                        __FILE__,
                                        __FUNCTION__,
                                        __LINE__);
        }
        m_phThreads = new HANDLE[m_uiThreadCount];
        SpawnSchedulerThreads();
        srand((UINT)::GetTickCount());
        nDispatchTotal = 0;
        nDependentDispatchTotal = 0;
        nDependentAcceleratorDispatchTotal = 0;
        m_bInGlobalQuiescentState = FALSE;
        m_bWaitingForGlobalQuiescence = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Scheduler::~Scheduler(
        void
        )
    {
        assert(!m_bAlive && "did you call Scheduler::shutdown?");
        if(m_phThreads != NULL && m_uiThreadCount) {
            for(UINT ui=0; ui<m_uiThreadCount; ui++) {
                CloseHandle(m_phThreads[ui]);
            }
            delete [] m_phThreads;
        }
        CloseHandle(m_hTaskQueue);
        CloseHandle(m_vhAcceleratorsAvailable[ACCELERATOR_CLASS_DIRECT_X] );
        CloseHandle(m_vhAcceleratorsAvailable[ACCELERATOR_CLASS_OPEN_CL]  );
        CloseHandle(m_vhAcceleratorsAvailable[ACCELERATOR_CLASS_CUDA]     );
        CloseHandle(m_vhAcceleratorsAvailable[ACCELERATOR_CLASS_REFERENCE]);
        CloseHandle(m_vhAcceleratorsAvailable[ACCELERATOR_CLASS_HOST]     );
        CloseHandle(m_vhAcceleratorsAvailable[ACCELERATOR_CLASS_SUPER]    );
        CloseHandle(m_hSchedulerTerminateEvent);
        CloseHandle(m_hGlobalQuiescentEvent);
        CloseHandle(m_hOKToSchedule); 
        CloseHandle(m_hAcceleratorQueue);
        DumpDispatchStatistics();
        assert(!QueueLockIsHeld() && "deleting scheduler object with Q lock held by another thread!");
        assert(!AcceleratorLockIsHeld() && "deleting scheduler object with accelerators lock held by another thread!");
        DeleteCriticalSection(&m_csQ);
        DeleteCriticalSection(&m_csAccelerators);
        DeleteCriticalSection(&m_csDispatchTimestamp);
        DeleteCriticalSection(&m_csQuiescentState);
        delete m_pDeviceManager;

        Accelerator::DeinitializeTLSContextManagement();
        for(set<Accelerator*>::iterator si=m_vMasterAcceleratorList.begin();
            si!= m_vMasterAcceleratorList.end(); si++) {
            Accelerator * pAcc = *si;
            delete pAcc;
        }

        map<ACCELERATOR_CLASS, std::set<Accelerator*>*>::iterator ami;
        for(ami=m_vEnabledAccelerators.begin(); ami!=m_vEnabledAccelerators.end(); ami++)
            delete ami->second;
        for(ami=m_vAvailableAccelerators.begin(); ami!=m_vAvailableAccelerators.end(); ami++)
            delete ami->second;
        std::map<std::string, std::map<Accelerator*, int>*>::iterator mi;
        EnterCriticalSection(&m_csStatistics);
        std::map<std::string, std::map<std::string, std::map<Accelerator*, int>*>*>::iterator tdi;

        for(tdi=m_vDispatches.begin(); tdi!=m_vDispatches.end(); tdi++) {
            std::map<std::string, std::map<Accelerator*, int>*>* pGraphDispatchMap = tdi->second;
            for(mi=pGraphDispatchMap->begin(); mi!=pGraphDispatchMap->end(); mi++) {
                std::map<Accelerator*, int>* pMap = mi->second;
                delete pMap;
            }
            pGraphDispatchMap->clear();
            delete pGraphDispatchMap;
        }
        m_vDispatches.clear();

        for(tdi=m_vDependentDispatches.begin(); tdi!=m_vDependentDispatches.end(); tdi++) {
            std::map<std::string, std::map<Accelerator*, int>*>* pGraphDispatchMap = tdi->second;
            for(mi=pGraphDispatchMap->begin(); mi!=pGraphDispatchMap->end(); mi++) {
                std::map<Accelerator*, int>* pMap = mi->second;
                delete pMap;
            }
            pGraphDispatchMap->clear();
            delete pGraphDispatchMap;
        }
        m_vDependentDispatches.clear();
        LeaveCriticalSection(&m_csStatistics);
        DeleteCriticalSection(&m_csStatistics);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the scheduler. </summary>
    ///
    /// <remarks>   crossbac, 5/10/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Scheduler::Initialize(
        VOID
        ) 
    {
        if(g_pScheduler == NULL) {
            g_pScheduler = new Scheduler();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the scheduler. </summary>
    ///
    /// <remarks>   crossbac, 5/10/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Scheduler::Destroy(
        VOID
        ) 
    {
        if(g_pScheduler != NULL) {
            delete g_pScheduler;
            g_pScheduler = NULL;
        }
        std::map<ACCELERATOR_CLASS, std::set<int>*>::iterator mi;
        for(mi=s_vDenyListDeviceIDs.begin(); mi!=s_vDenyListDeviceIDs.end(); mi++) 
            delete mi->second;
        for(mi=s_vAllowListDeviceIDs.begin(); mi!=s_vAllowListDeviceIDs.end(); mi++) 
            delete mi->second;
        s_vDenyListDeviceIDs.clear();
        s_vAllowListDeviceIDs.clear();
    }        

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pGraph' has outstanding dispatches. </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///
    /// <returns>   true if outstanding dispatches, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::HasUnscheduledDispatches(
        Graph * pGraph
        ) 
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL) {
            return g_pScheduler->__HasUnscheduledDispatches(pGraph);
        }  
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pTask' has outstanding dispatches. </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///
    /// <returns>   true if outstanding dispatches, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::HasUnscheduledDispatches(
        Task * pTask
        ) 
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL) {
            return g_pScheduler->__HasUnscheduledDispatches(pTask);
        }  
        return FALSE;    
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pGraph' has outstanding dispatches. </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///
    /// <returns>   true if outstanding dispatches, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::__HasUnscheduledDispatches(
        Graph * pGraph
        ) 
    {
        BOOL bResult = FALSE;
        assert(QuiescenceLockIsHeld());
        assert(QueueLockIsHeld());

        deque<Task*>::iterator di;
        for(di=m_vReadyQ.begin(); di!=m_vReadyQ.end(); di++) {
            if((*di)->GetGraph() == pGraph) {
                bResult = TRUE;
                break;
            }
        }

        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pTask' has outstanding dispatches. </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///
    /// <returns>   true if outstanding dispatches, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::__HasUnscheduledDispatches(
        Task * pTask
        ) 
    {
        BOOL bResult = FALSE;
        assert(QuiescenceLockIsHeld());
        assert(QueueLockIsHeld());

        deque<Task*>::iterator di;
        for(di=m_vReadyQ.begin(); di!=m_vReadyQ.end(); di++) {
            if(*di == pTask) {
                bResult = TRUE;
                break;
            }
        }

        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pGraph' has deferred dispatches. </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///
    /// <returns>   true if outstanding dispatches, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::__HasDeferredDispatches(
        Graph * pGraph
        ) 
    {
        BOOL bResult = FALSE;
        assert(QuiescenceLockIsHeld());
        assert(QueueLockIsHeld());

        deque<Task*>::iterator di;
        for(di=m_vDeferredQ.begin(); di!=m_vDeferredQ.end(); di++) {
            if((*di)->GetGraph() == pGraph) {
                bResult = TRUE;
                break;
            }
        }

        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pTask' has outstanding dispatches. </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///
    /// <returns>   true if outstanding dispatches, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::__HasDeferredDispatches(
        Task * pTask
        ) 
    {
        BOOL bResult = FALSE;
        assert(QuiescenceLockIsHeld());
        assert(QueueLockIsHeld());

        deque<Task*>::iterator di;
        for(di=m_vDeferredQ.begin(); di!=m_vDeferredQ.end(); di++) {
            if(*di == pTask) {
                bResult = TRUE;
                break;
            }
        }

        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pGraph' has deferred dispatches. </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///
    /// <returns>   true if outstanding dispatches, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::__HasInflightDispatches(
        Graph * pGraph
        ) 
    {
        assert(AcceleratorLockIsHeld());
        std::map<Accelerator*, Task*>::iterator mi;        
        for(mi=m_vInflightAccelerators.begin(); mi!=m_vInflightAccelerators.end(); mi++) {
            Task * pTask = mi->second;
            if(pTask->GetGraph() == pGraph) 
                return TRUE;
        }

        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pTask' has outstanding dispatches. </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///
    /// <returns>   true if outstanding dispatches, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::__HasInflightDispatches(
        Task * pTask
        ) 
    {
        assert(AcceleratorLockIsHeld());
        std::map<Accelerator*, Task*>::iterator mi;        
        for(mi=m_vInflightAccelerators.begin(); mi!=m_vInflightAccelerators.end(); mi++) {
            if(mi->second == pTask) 
                return TRUE;
        }

        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Shuts down the scheduler. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::Shutdown(
        VOID
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL) {
            g_pScheduler->__Shutdown();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Shuts down the scheduler. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    HANDLE
    Scheduler::ShutdownAsync(
        VOID
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL) {
            return g_pScheduler->__ShutdownAsync();
        }
        return INVALID_HANDLE_VALUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Quiesce the scheduler. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::GlobalQuiesce(
        VOID
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL) {
            g_pScheduler->__GlobalQuiesce();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Quiesce the scheduler. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    HANDLE
    Scheduler::GlobalQuiesceAsync(
        VOID
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL) {
            return g_pScheduler->__GlobalQuiesceAsync();
        }
        return INVALID_HANDLE_VALUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Queisce the scheduler with respect only to a given graph. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///
    /// <param name="pGraph">           [in,out] If non-null, the graph. </param>
    /// <param name="hQuiescentEvent">  The quiescent event. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::QuiesceGraph(
        __in Graph * pGraph, 
        __in HANDLE hQuiescentEvent
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL) {
            return g_pScheduler->__QuiesceGraph(pGraph, hQuiescentEvent);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Un-Quiesce the scheduler. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::EndGlobalQuiescence(
        VOID
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL) {
            g_pScheduler->__EndGlobalQuiescence();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Un-Quiesce the scheduler. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::EndGraphQuiescence(
        Graph * pGraph
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL) {
            g_pScheduler->__EndGraphQuiescence(pGraph);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Un-Quiesce the scheduler. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__EndGlobalQuiescence(
        VOID
        )
    {
        LockQuiescenceState();
        LockTaskQueues();

        if(m_bAlive) { 
            assert(m_bQuiescenceInProgress);
            assert(!m_bOKToSchedule);
            m_bInGlobalQuiescentState = FALSE;
            m_bWaitingForGlobalQuiescence = FALSE;
            if(m_vDeferredQ.size()) {
                std::deque<Task*> newDeferredQ;
                while(m_vDeferredQ.size()) {
                    Task * pTask = m_vDeferredQ.front();
                    m_vDeferredQ.pop_front();
                    if(m_vRunningGraphs.find(pTask->GetGraph()) != m_vRunningGraphs.end())
                        m_vReadyQ.push_back(pTask);
                    else
                        newDeferredQ.push_back(pTask);
                }
                m_vDeferredQ.assign(newDeferredQ.begin(), newDeferredQ.end());
            }
            __UpdateQuiescenceHint();
            __UpdateOKToScheduleView();            
            __UpdateTaskQueueView();
            ResetEvent(m_hGlobalQuiescentEvent);
        }

        UnlockTaskQueues();
        UnlockQuiescenceState();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Un-Quiesce the scheduler. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__EndGraphQuiescence(
        Graph * pGraph
        )
    {
        LockQuiescenceState();
        assert(m_bQuiescenceInProgress);

        assert(m_vQuiescedGraphs.find(pGraph) != m_vQuiescedGraphs.end());
        assert(m_vLiveGraphs.find(pGraph) != m_vLiveGraphs.end());
        assert(m_vRunningGraphs.find(pGraph) == m_vRunningGraphs.end());
        assert(m_vGraphsWaitingForQuiescence.find(pGraph) == m_vGraphsWaitingForQuiescence.end());

        assert(pGraph->m_eState == PTGS_RUNNING);
        LockTaskQueues();
        std::set<Graph*>::iterator si=m_vQuiescedGraphs.find(pGraph);
        if(si != m_vQuiescedGraphs.end()) {
            m_vQuiescedGraphs.erase(si);
            m_vRunningGraphs.insert(pGraph);
            std::deque<Task*> newDeferredQ;
            while(m_vDeferredQ.size()) {
                Task * pTask = m_vDeferredQ.front();
                m_vDeferredQ.pop_front();
                if(pTask->GetGraph() == pGraph) {
                    m_vReadyQ.push_back(pTask); 
                } else {
                    newDeferredQ.push_back(pTask);
                }
            }        
            m_vDeferredQ.assign(newDeferredQ.begin(), newDeferredQ.end());
        }
        __UpdateQuiescenceHint();
        __UpdateOKToScheduleView();
        __UpdateTaskQueueView();

        UnlockTaskQueues();
        UnlockQuiescenceState();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Shuts down the scheduler. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__Shutdown(
        VOID
        )
    {
        LockQuiescenceState();
        BOOL bAliveTransition = FALSE;
        if(m_bAlive) {
            bAliveTransition = TRUE;
            UnlockQuiescenceState();
            HANDLE hQuiesceEvent = __ShutdownAsync();
            DWORD dwWait = WaitForSingleObject(hQuiesceEvent, INFINITE);
            if(dwWait == WAIT_ABANDONED || dwWait == WAIT_FAILED) {
                PTask::Runtime::HandleError("%s:%s: Wait failed with %d\n",
                                            __FUNCTION__,
                                            "WaitForSingleObject",
                                            dwWait);
            }
            LockQuiescenceState();
        }
        LockTaskQueues();
        LockAcceleratorMaps();
        if(m_bAlive || bAliveTransition) {

            assert(m_bInGlobalQuiescentState);
            assert(m_vInflightAccelerators.size() == 0);
            assert(m_vReadyQ.size() == 0);
            m_bAlive = FALSE;       
            SetEvent(m_hSchedulerTerminateEvent);
            std::set<Graph*>::iterator si;
            std::set<Graph*> vDanglingGraphs;

            vDanglingGraphs.insert(m_vNascentGraphs.begin(), m_vNascentGraphs.end());
            vDanglingGraphs.insert(m_vLiveGraphs.begin(), m_vLiveGraphs.end());
            vDanglingGraphs.insert(m_vRetiredGraphs.begin(), m_vRetiredGraphs.end());
            vDanglingGraphs.insert(m_vRunningGraphs.begin(), m_vRunningGraphs.end());
            vDanglingGraphs.insert(m_vQuiescedGraphs.begin(), m_vQuiescedGraphs.end());
            vDanglingGraphs.insert(m_vGraphsWaitingForQuiescence.begin(), m_vGraphsWaitingForQuiescence.end());
            for(si=vDanglingGraphs.begin(); si!=vDanglingGraphs.end(); si++) {
                __AbandonDispatches(*si);
                (*si)->ForceTeardown();
            }

            m_vNascentGraphs.clear();
            m_vRetiredGraphs.clear();
            m_vRunningGraphs.clear();
            m_vQuiescedGraphs.clear();
            m_vGraphsWaitingForQuiescence.clear();
            m_vLiveGraphs.clear();
            m_vReadyQ.clear();
            m_vDeferredQ.clear();
        }
        UnlockAcceleratorMaps();
        UnlockTaskQueues();
        UnlockQuiescenceState();
        WaitForMultipleObjects(m_uiThreadCount, m_phThreads, TRUE, INFINITE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Shuts down the scheduler asynchronously. This version returns
    /// 			a wait handle for the caller, who can respond by calling 
    /// 			shutdown again, with the guarantee that the scheduler is quiescent.
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    HANDLE
    Scheduler::__ShutdownAsync(
        VOID
        )
    {
        HANDLE hWaitHandle = INVALID_HANDLE_VALUE;
        LockQuiescenceState();
        LockTaskQueues();
        LockAcceleratorMaps();        

        if(m_bAlive) {

            hWaitHandle = m_hGlobalQuiescentEvent;
            BOOL bQuiescent = (m_vReadyQ.size() == 0 && m_vInflightAccelerators.size() == 0);

            if(bQuiescent) {

                // We are already quiescent, do the book keeping for global quiescence, shut down the thread
                // pools, and set the event before returning it. Note that we must release the locks first,
                // since we are going to perform a blocking wait; re-acquire them subsequently for form's sake.
                
                m_bInGlobalQuiescentState = TRUE;
                m_bWaitingForGlobalQuiescence = FALSE;
                m_bAlive = FALSE;
                m_bOKToSchedule = FALSE;
                ResetEvent(m_hOKToSchedule);
                SetEvent(m_hGlobalQuiescentEvent);
                SetEvent(m_hSchedulerTerminateEvent);

                UnlockAcceleratorMaps();
                UnlockTaskQueues();
                UnlockQuiescenceState();
                if(m_uiThreadCount > 0) {
                    DWORD dwWait = WaitForMultipleObjects(m_uiThreadCount, m_phThreads, TRUE, INFINITE);
                    switch(dwWait) {
                    case WAIT_TIMEOUT:   PTask::Runtime::MandatoryInform("Scheduler::ShutdownAsync(): wait timed out on thread pool objects\n"); break; 
                    case WAIT_ABANDONED: PTask::Runtime::MandatoryInform("Scheduler::ShutdownAsync(): wait abandoned on thread pool objects\n"); break; 
                    case WAIT_FAILED:    PTask::Runtime::MandatoryInform("Scheduler::ShutdownAsync(): wait FAILED on thread pool objects\n"); break; 
                    default:             break;
                    }
                }
                LockQuiescenceState();
                LockTaskQueues();
                LockAcceleratorMaps();

            } else {

                // there is outstanding work to be done. wait for the
                // task queues and inflight lists to be empty. Reset the global quiescence event
                // so that we can return it to the caller to wait on. 

                m_bInGlobalQuiescentState = FALSE;
                m_bWaitingForGlobalQuiescence = TRUE;
                ResetEvent(m_hGlobalQuiescentEvent);
            }
        }

        UnlockAcceleratorMaps();
        UnlockTaskQueues();
        UnlockQuiescenceState();
        return m_hGlobalQuiescentEvent;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check whether there are multiple back-end GPU frameworks
    ///             in use in this graph, and if so, notify the scheduler so it
    ///             can avoid over committing devices that are shared through 
    ///             different accelerator objects (with different subtypes). 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::FindCrossRuntimeDependences(
        VOID
        )
    {
        LockAcceleratorMaps();

        m_vLiveAcceleratorClasses.clear();
        if(m_bLiveGraphs) {
            std::set<Graph*>::iterator si;
            for(si= m_vLiveGraphs.begin(); si!=m_vLiveGraphs.end(); si++) {
                Graph * pGraph = *si;
                std::set<ACCELERATOR_CLASS>::iterator aci;
                std::set<ACCELERATOR_CLASS>* pFrameworks = pGraph->GetBackendFrameworks();
                for(aci=pFrameworks->begin(); aci!=pFrameworks->end(); aci++) {
                    m_vLiveAcceleratorClasses.insert(*aci);
                }                
            }
        }

        set<ACCELERATOR_CLASS> vGPUFrameworks;
        set<ACCELERATOR_CLASS>::iterator fi;
        for(fi=m_vLiveAcceleratorClasses.begin(); fi!=m_vLiveAcceleratorClasses.end(); fi++) {
            // figure out if its possible for a device
            // to be shared through multiple frameworks given what we 
            // know about all the live graphs. If there is a super-task,
            // assert for now--what we care about is it's underlying framework.
            // basically we care only if there are DX, CL, and/or CU Tasks
            // in the superset. Otherwise, we can skip checking availability
            // at the physical device level. 
            switch(*fi) {
            case ACCELERATOR_CLASS_SUPER:     assert(FALSE); break; // not a backend framework. how did this happen?
            case ACCELERATOR_CLASS_DIRECT_X:  vGPUFrameworks.insert(*fi); break;
            case ACCELERATOR_CLASS_OPEN_CL:   vGPUFrameworks.insert(*fi); break;
            case ACCELERATOR_CLASS_CUDA:      vGPUFrameworks.insert(*fi); break;
            case ACCELERATOR_CLASS_REFERENCE:
            case ACCELERATOR_CLASS_HOST:
            case ACCELERATOR_CLASS_UNKNOWN:
                break;
            }
        }
        m_bCrossRuntimeSharingChecksRequired = (vGPUFrameworks.size() > 1);

        UnlockAcceleratorMaps();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notifies a new graph. </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Scheduler::NotifyGraphCreate(
        __in Graph * pGraph
        )
    {
        if(g_pScheduler == NULL) {
            PTask::Runtime::MandatoryInform("WARNING: %s::%s called with runtime uninitialized!\n",
                                            __FILE__,
                                            __FUNCTION__);
        }
        if(g_pScheduler != NULL)
            g_pScheduler->__NotifyGraphCreate(pGraph);     
    }   

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notifies a dtor'ed graph. </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Scheduler::NotifyGraphDestroy(
        __in Graph * pGraph
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL)
            g_pScheduler->__NotifyGraphDestroy(pGraph);     
    }   

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notifies the scheduler that a new graph has been created. 
    ///             Requires the queue lock but not the accelerator lock.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Scheduler::__NotifyGraphCreate(
        __in Graph * pGraph
        )
    {
        LockQuiescenceState();

        assert(pGraph->m_eState == PTGS_INITIALIZING);
        assert(m_vLiveGraphs.find(pGraph) == m_vLiveGraphs.end());
        assert(m_vNascentGraphs.find(pGraph) == m_vNascentGraphs.end());
        assert(m_vRunningGraphs.find(pGraph) == m_vRunningGraphs.end());
        assert(m_vQuiescedGraphs.find(pGraph) == m_vQuiescedGraphs.end());
        assert(m_vGraphsWaitingForQuiescence.find(pGraph) == m_vGraphsWaitingForQuiescence.end());
        assert(m_vRetiredGraphs.find(pGraph) == m_vRetiredGraphs.end());
        m_vNascentGraphs.insert(pGraph);

        UnlockQuiescenceState();
    }   

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notify the scheduler that there is a live graphs for which
    ///             it is ok to schedule tasks. Take note of what backend frameworks
    ///             are represented so we can figure out if we need to do cross
    ///             framework sharing checks on underlying devices.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Scheduler::NotifyGraphRunning(
        __in Graph * pGraph
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL)
            g_pScheduler->__NotifyGraphRunning(pGraph);     
    }   

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notify the scheduler that there is a live graphs for which
    ///             it is ok to schedule tasks. Take note of what backend frameworks
    ///             are represented so we can figure out if we need to do cross
    ///             framework sharing checks on underlying devices.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__NotifyGraphRunning(
        __in Graph * pGraph
        )
    {
        LockQuiescenceState();

        // this should only be called when the graph enters the running
        // state for the first time, at which point it becomes live. 
        // We expect it only in the nascent graphs list
        assert(pGraph->m_eState == PTGS_RUNNING);
        assert(m_vLiveGraphs.find(pGraph) == m_vLiveGraphs.end());
        assert(m_vNascentGraphs.find(pGraph) != m_vNascentGraphs.end());
        assert(m_vRunningGraphs.find(pGraph) == m_vRunningGraphs.end());
        assert(m_vQuiescedGraphs.find(pGraph) == m_vQuiescedGraphs.end());
        assert(m_vGraphsWaitingForQuiescence.find(pGraph) == m_vGraphsWaitingForQuiescence.end());
        assert(m_vRetiredGraphs.find(pGraph) == m_vRetiredGraphs.end());

        m_vNascentGraphs.erase(pGraph);
        m_vRunningGraphs.insert(pGraph);
        m_vLiveGraphs.insert(pGraph);
        m_bLiveGraphs = TRUE;

        // update our view of cross-platform dependences.
        FindCrossRuntimeDependences();

        __UpdateQuiescenceHint();
        __UpdateOKToScheduleView();
        LockTaskQueues();
        __UpdateTaskQueueView();
        UnlockTaskQueues();
        UnlockQuiescenceState();       
    }   

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notify the scheduler that there are no live graphs. Block until there is one.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::NotifyGraphTeardown(
        __in Graph * pGraph
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL)
            g_pScheduler->__NotifyGraphTeardown(pGraph);   
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notify the scheduler that there are no live graphs. Block until there is one.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__NotifyGraphTeardown(
        __in Graph * pGraph
        )
    {
        LockQuiescenceState();

        // when the graph is torn down, the user (and/or indeed, the graph teardown
        // code, should be sure to quiesce the graph and abandon and dispatches
        // before we get to this point. If this is not the case, then complain 
        // loudly and then clean up, grudgingly. 
        
        LockTaskQueues();
        if(__HasUnscheduledDispatches(pGraph) || __HasDeferredDispatches(pGraph)) {
            //assert(!__HasUnscheduledDispatches(pGraph));
            //assert(!__HasDeferredDispatches(pGraph));
            PTask::Runtime::MandatoryInform("WARNING: Scheduler::NotifyGraphTeardown: teardown with unscheduled or deferred tasks!\n");
            __AbandonDispatches(pGraph);
        }
        UnlockTaskQueues();

        assert(pGraph->m_eState == PTGS_TEARINGDOWN);
        assert(m_vLiveGraphs.find(pGraph) != m_vLiveGraphs.end() || 
               m_vNascentGraphs.find(pGraph) != m_vNascentGraphs.end());
        assert(m_vRunningGraphs.find(pGraph) != m_vRunningGraphs.end() ||
               m_vQuiescedGraphs.find(pGraph) != m_vQuiescedGraphs.end() ||
               m_vGraphsWaitingForQuiescence.find(pGraph) != m_vGraphsWaitingForQuiescence.end() ||
               m_vNascentGraphs.find(pGraph) != m_vNascentGraphs.end());
        assert(m_vRetiredGraphs.find(pGraph) == m_vRetiredGraphs.end());

        if(m_vRunningGraphs.find(pGraph) != m_vRunningGraphs.end())
            m_vRunningGraphs.erase(pGraph);
        if(m_vQuiescedGraphs.find(pGraph) != m_vQuiescedGraphs.end())
            m_vQuiescedGraphs.erase(pGraph);
        if(m_vGraphsWaitingForQuiescence.find(pGraph) != m_vGraphsWaitingForQuiescence.end()) {
            PTask::Runtime::MandatoryInform("WARNING: Scheduler::NotifyGraphTeardown: tearing down quiescing graph!\n");
            m_vGraphsWaitingForQuiescence.erase(pGraph);
        }
        if(m_vLiveGraphs.find(pGraph) != m_vLiveGraphs.end())
            m_vLiveGraphs.erase(pGraph);
        if(m_vNascentGraphs.find(pGraph) != m_vNascentGraphs.end())
            m_vNascentGraphs.erase(pGraph);

        std::map<Graph*, HANDLE>::iterator mi;
        mi = m_vhQuiescentEvents.find(pGraph);
        if(mi!=m_vhQuiescentEvents.end())
            m_vhQuiescentEvents.erase(mi);
        mi = m_vhShutdownEvents.find(pGraph);
        if(mi!=m_vhShutdownEvents.end()) 
            m_vhShutdownEvents.erase(mi);

        m_vRetiredGraphs.insert(pGraph);
        m_bLiveGraphs = m_vLiveGraphs.size() > 0;
        assert(!m_bLiveGraphs ||
               (m_vRunningGraphs.size() || 
                m_vQuiescedGraphs.size() || 
                m_vGraphsWaitingForQuiescence.size()));

        // recompute the backend frameworks in use:
        FindCrossRuntimeDependences();

        __UpdateQuiescenceHint();
        __UpdateOKToScheduleView();

        if(!m_bLiveGraphs) {
            // no live graphs left--drain the async contexts
            // on all accelerators that support async APIs.
            assert(m_vLiveGraphs.size() == 0);
            assert(m_vRunningGraphs.size() == 0);
            assert(m_vQuiescedGraphs.size() == 0);
            assert(m_vReadyQ.size() == 0);
            assert(m_vDeferredQ.size() == 0);
            std::map<ACCELERATOR_CLASS, std::set<Accelerator*>*>::iterator mi;
            for(mi=m_vEnabledAccelerators.begin(); mi!=m_vEnabledAccelerators.end(); mi++) {
                std::set<Accelerator*>* pAccSet = mi->second;
                std::set<Accelerator*>::iterator si;
                for(si=pAccSet->begin(); si!=pAccSet->end(); si++) {
                    Accelerator * pAccelerator = *si;
                    if(pAccelerator->SupportsExplicitAsyncOperations()) {
                        pAccelerator->Synchronize();
                    }
                }
            }
        }

        UnlockQuiescenceState();
    }   

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notifies the scheduler that a graph is being destroyed. </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__NotifyGraphDestroy(
        __in Graph * pGraph
        )
    {
        LockQuiescenceState();

        // when the graph is torn down, the user (and/or indeed, the graph teardown
        // code, should be sure to quiesce the graph and abandon and dispatches
        // before we get to this point. If this is not the case, then complain 
        // loudly and then clean up, grudgingly. 
        
        LockTaskQueues();
        if(__HasUnscheduledDispatches(pGraph) || __HasDeferredDispatches(pGraph)) {
            assert(!__HasUnscheduledDispatches(pGraph));
            assert(!__HasDeferredDispatches(pGraph));
            PTask::Runtime::MandatoryInform("WARNING: Scheduler::NotifyGraphDestroy: dtor called with unscheduled or deferred tasks!\n");
            __AbandonDispatches(pGraph);
        }
        UnlockTaskQueues();

        assert(pGraph->m_eState == PTGS_TEARDOWNCOMPLETE);
        assert(m_vLiveGraphs.find(pGraph) == m_vLiveGraphs.end());
        assert(m_vNascentGraphs.find(pGraph) == m_vNascentGraphs.end());
        assert(m_vRunningGraphs.find(pGraph) == m_vRunningGraphs.end());
        assert(m_vQuiescedGraphs.find(pGraph) == m_vQuiescedGraphs.end());
        assert(m_vGraphsWaitingForQuiescence.find(pGraph) == m_vGraphsWaitingForQuiescence.end());
        assert(m_vRetiredGraphs.find(pGraph) != m_vRetiredGraphs.end());

        if(m_vRetiredGraphs.find(pGraph) != m_vRetiredGraphs.end())
            m_vRetiredGraphs.erase(pGraph);
        
        m_bLiveGraphs = m_vLiveGraphs.size() > 0;
        assert(!m_bLiveGraphs ||
               (m_vRunningGraphs.size() || 
                m_vQuiescedGraphs.size() || 
                m_vGraphsWaitingForQuiescence.size()));

        __UpdateQuiescenceHint();
        __UpdateOKToScheduleView();

        UnlockQuiescenceState();
    }   

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Is this task dispatchable based on the graph state? </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///
    /// <returns>   true if dispatchable, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::__IsDispatchable(
        __in Task * pTask
        )
    {
        // queue lock required. We can check the graph state, but we cannot enforce the invariant that
        // the graph lock must be held. So if the task is from a graph that is still in the running set,
        // assume we can dispatch it, even if the graph state has been changed (which should be
        // followed shortly by a notification that we should move the graph to the teardown list or
        // quiescing lists. 
        
        assert(QuiescenceLockIsHeld());
        Graph * pGraph = pTask->GetGraph();
        BOOL bRunnableGraph = (m_vRunningGraphs.find(pGraph) != m_vRunningGraphs.end()) ||
                              (m_vGraphsWaitingForQuiescence.find(pGraph) != m_vGraphsWaitingForQuiescence.end());
        return bRunnableGraph;
    }   

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Is this task dispatch deferrable based on the graph state? </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///
    /// <returns>   true if dispatchable, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::__IsDeferrable(
        __in Task * pTask
        )
    {
        assert(QuiescenceLockIsHeld());
        Graph * pGraph = pTask->GetGraph();
        return (m_vQuiescedGraphs.find(pGraph) != m_vQuiescedGraphs.end() ||
                m_vGraphsWaitingForQuiescence.find(pGraph) != m_vGraphsWaitingForQuiescence.end());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   quiesce the scheduler. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__GlobalQuiesce(
        VOID
        )
    {
        HANDLE hQuiesceEvent = __GlobalQuiesceAsync();
        HANDLE vWaitHandles[] = { 
            m_hSchedulerTerminateEvent,
            m_hRuntimeTerminateEvent,
            hQuiesceEvent };
        DWORD dwWaitHandles = sizeof(vWaitHandles) / sizeof(HANDLE);
        DWORD dwWait = WaitForMultipleObjects(dwWaitHandles, vWaitHandles, FALSE, INFINITE);
        switch(dwWait) {
        case WAIT_TIMEOUT:      PTask::Runtime::MandatoryInform("__GlobalQuiesce: Unexpected Wait TIMEOUT!..ignoring\n");
        case WAIT_ABANDONED:    PTask::Runtime::MandatoryInform("__GlobalQuiesce: Wait ABANDONED!..ignoring\n");
        case WAIT_FAILED:       PTask::Runtime::MandatoryInform("__GlobalQuiesce: Wait FAILED!..ignoring\n");
        case WAIT_OBJECT_0 + 0: PTask::Runtime::MandatoryInform("__GlobalQuiesce: scheduler shutdown during quiescence wait!\n");
        case WAIT_OBJECT_0 + 1: PTask::Runtime::MandatoryInform("__GlobalQuiesce: runtime shutdown during quiescence wait!\n");
        case WAIT_OBJECT_0 + 2: break;
        default: break; 
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Quiesce the scheduler asynchronously. This version returns
    /// 			a wait handle for the caller, who can respond by calling 
    /// 			methods that require quiescence.
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    HANDLE
    Scheduler::__GlobalQuiesceAsync(
        VOID
        )
    {
        LockQuiescenceState();
        LockTaskQueues();
        LockAcceleratorMaps();
        BOOL bQuiescent = (m_vReadyQ.size() == 0 && m_vInflightAccelerators.size() == 0);
        if(bQuiescent) {
            m_bInGlobalQuiescentState = TRUE;
            m_bWaitingForGlobalQuiescence = FALSE;
            m_bOKToSchedule = FALSE;
            ResetEvent(m_hOKToSchedule);            
            SetEvent(m_hGlobalQuiescentEvent);
            __UpdateQuiescenceHint();
        } else {
            m_bInGlobalQuiescentState = FALSE;
            m_bWaitingForGlobalQuiescence = TRUE;
            __UpdateQuiescenceHint();
            ResetEvent(m_hGlobalQuiescentEvent);
        }
        UnlockAcceleratorMaps();
        UnlockTaskQueues();
        UnlockQuiescenceState();
        return m_hGlobalQuiescentEvent;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pGraph' is quiescent. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///
    /// <returns>   true if quiescent, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::IsGraphQuiescent(
        __in Graph * pGraph
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL)
            return g_pScheduler->__IsGraphQuiescent(pGraph);
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pGraph' is quiescent. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///
    /// <returns>   true if quiescent, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::__IsGraphQuiescent(
        __in Graph * pGraph
        )
    {        
        // we require the queue lock and the accelerator for this call, since
        // the ready queue and the inflight list must be checked.

        assert(QuiescenceLockIsHeld());
        assert(QueueLockIsHeld());
        assert(AcceleratorLockIsHeld());

        BOOL bInflightTasks = __HasInflightDispatches(pGraph);
        BOOL bUnscheduledTasks = __HasUnscheduledDispatches(pGraph);        

        return !bInflightTasks && !bUnscheduledTasks;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Quiesce the scheduler asynchronously with respect to a specific graph. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Scheduler::__QuiesceGraph(
        Graph * pGraph,
        HANDLE hQuiescentEvent
        )
    {
        // on entry, we assume the graph lock is
        // held. We need to drop it to wait for quiescence,
        // and then reacquire it before returning.
        assert(pGraph->LockIsHeld());
        HANDLE hWaitEvent = __QuiesceGraphAsync(pGraph, hQuiescentEvent);
        HANDLE vWaitHandles[] = { m_hRuntimeTerminateEvent, m_hSchedulerTerminateEvent, hWaitEvent };
        DWORD dwWaitHandles = sizeof(vWaitHandles) / sizeof(HANDLE);
        
        int nLockDepth = pGraph->Unlock();
        if(pGraph->LockIsHeld()) {
            assert(!pGraph->LockIsHeld());
            PTask::Runtime::HandleError("%s:%s: lock-depth =%d!\n",
                                        __FUNCTION__,
                                        pGraph->m_lpszGraphName,
                                        nLockDepth);
        }
        WaitForMultipleObjects(dwWaitHandles, vWaitHandles, FALSE, INFINITE);
        pGraph->Lock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Quiesce the scheduler asynchronously with respect to a specific graph. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    HANDLE
    Scheduler::__QuiesceGraphAsync(
        Graph * pGraph,
        HANDLE hQuiescentEvent
        )
    {
        LockQuiescenceState();

        assert(m_vLiveGraphs.find(pGraph) != m_vLiveGraphs.end());
        assert(m_vQuiescedGraphs.find(pGraph) == m_vQuiescedGraphs.end());
        assert(m_vRunningGraphs.find(pGraph) != m_vRunningGraphs.end());
        assert(m_vGraphsWaitingForQuiescence.find(pGraph) == m_vGraphsWaitingForQuiescence.end());

        LockTaskQueues();
        std::deque<Task*> newReadyQ;
        while(m_vReadyQ.size()) {
            Task * pTask = m_vReadyQ.front();
            m_vReadyQ.pop_front();
            if(pTask->GetGraph() == pGraph) {
                m_vDeferredQ.push_back(pTask);
            } else {
                newReadyQ.push_back(pTask);
            }
        }
        m_vReadyQ.assign(newReadyQ.begin(), newReadyQ.end());

        LockAcceleratorMaps();
        if(__HasInflightDispatches(pGraph)) {
            ResetEvent(hQuiescentEvent);
            m_vGraphsWaitingForQuiescence.insert(pGraph);
            m_vhQuiescentEvents[pGraph] = hQuiescentEvent;
        } else {
            SetEvent(hQuiescentEvent);
            m_vQuiescedGraphs.insert(pGraph);
        }
        m_vRunningGraphs.erase(pGraph);
        __UpdateQuiescenceHint();
        __UpdateOKToScheduleView();
        __UpdateTaskQueueView();

        UnlockAcceleratorMaps();
        UnlockTaskQueues();
        UnlockQuiescenceState();
        return hQuiescentEvent;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Updates our view of the "can schedule" state. 
    ///             We can schedule if there are runnable graphs or quiescing graphs,
    ///             and we are not in a wait for global quiescence.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/9/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__UpdateOKToScheduleView(
        VOID
        )
    {
        assert(QuiescenceLockIsHeld());
        BOOL bOldSchedulerState = m_bOKToSchedule;
        m_bOKToSchedule = !m_bInGlobalQuiescentState &&
                          ((m_vRunningGraphs.size() > 0) || 
                           (m_vGraphsWaitingForQuiescence.size() > 0));
        BOOL bStateChange = bOldSchedulerState != m_bOKToSchedule;
        if(bStateChange && m_bOKToSchedule) {
            SetEvent(m_hOKToSchedule);
        } else if(bStateChange && !m_bOKToSchedule) {
            ResetEvent(m_hOKToSchedule);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Updates the quiescence hint. </summary>
    ///
    /// <remarks>   Crossbac, 7/9/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__UpdateQuiescenceHint(
        VOID
        )
    {
        // cache a hint for whether we need to examine graph
        // state data structures (and potentially acquire more locks)
        // to make scheduling decisions. 
        assert(QuiescenceLockIsHeld());
        m_bQuiescenceInProgress = m_bInGlobalQuiescentState || 
                                  m_bWaitingForGlobalQuiescence ||
                                  m_vGraphsWaitingForQuiescence.size() ||
                                  m_vQuiescedGraphs.size();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Spawn scheduler threads. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Scheduler::SpawnSchedulerThreads(
        VOID
        )
    {
        for(UINT ui=0; ui<m_uiThreadCount; ui++) {
            m_phThreads[ui] = CreateThread(NULL, NULL, ptask_scheduler_thread, this, NULL, 0);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pAccelerator' is on a deny or allow restriction list. this is a helper
    ///             function for deciding whether to enable an accelerator at init time.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/19/2014. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    /// <param name="pList">        [in,out] If non-null, the list. </param>
    ///
    /// <returns>   true if on the list, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::IsOnRestrictionList(
        __in Accelerator * pAccelerator, 
        __in std::set<int>* pList
        )
    {
        assert(pAccelerator != NULL);
        if(pAccelerator == NULL || pList == NULL)
            return FALSE;

        int nPSDeviceID = reinterpret_cast<int>(pAccelerator->GetDevice());
#ifdef DEBUG
        ACCELERATOR_CLASS eClass = pAccelerator->GetClass();
        if(eClass == ACCELERATOR_CLASS_CUDA) {
            CUAccelerator * pCUAccelerator = reinterpret_cast<CUAccelerator*>(pAccelerator);
            assert(nPSDeviceID == pCUAccelerator->GetDeviceId());
        }
#endif
        return pList->find(nPSDeviceID) != pList->end();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pAccelerator' is deny listed. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2014. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   true if deny listed, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::IsDenyListed(
        __in Accelerator * pAccelerator
        )
    {
        assert(pAccelerator != NULL);
        if(pAccelerator == NULL) return FALSE;
        std::set<int>* pDenyList = NULL;
        GetRestrictionLists(pAccelerator->GetClass(), NULL, &pDenyList, FALSE);
        return IsOnRestrictionList(pAccelerator, pDenyList);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pAccelerator' is deny listed. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2014. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   true if deny listed, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::IsAllowListed(
        __in Accelerator * pAccelerator
        )
    {
        assert(pAccelerator != NULL);
        if(pAccelerator == NULL) return FALSE;
        std::set<int>* pAllowList = NULL;
        GetRestrictionLists(pAccelerator->GetClass(), &pAllowList, NULL, FALSE);
        return IsOnRestrictionList(pAccelerator, pAllowList);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pAccelerator' is an enablable accelerator based
    ///             on allow-list/deny-list static informations configured by 
    ///             user code before the scheduler is created (PTask::Runtime::Initialize is called).
    ///             If a allow list exists for the given class, the accelerator must be on the allow
    ///             list. If a deny list exists, it must not be on the deny list. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/19/2014. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   true if enablable candidate, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::IsEnablable(
        __in Accelerator* pAccelerator
        )
    {
        std::set<int>* pDenyList = NULL;
        std::set<int>* pAllowList = NULL;
        GetRestrictionLists(pAccelerator->GetClass(), &pAllowList, &pDenyList, FALSE);
        BOOL bDenyListHasEntries = pDenyList != NULL && pDenyList->size() > 0;
        BOOL bAllowListHasEntries = pAllowList != NULL && pAllowList->size() > 0;

        // no deny or allow list entries for this class. 
        // this means there are no disable/enable entries,
        // so we will enable every accelerator of this class
        if(!bDenyListHasEntries && !bAllowListHasEntries) 
            return TRUE; 

        BOOL bDenyListed = bDenyListHasEntries && IsOnRestrictionList(pAccelerator, pDenyList);
        BOOL bAllowListed = bAllowListHasEntries && IsOnRestrictionList(pAccelerator, pAllowList); 

        static BOOL bBLWLWarningShown = FALSE;
        if(bDenyListHasEntries && bAllowListHasEntries && !bBLWLWarningShown) {
            PTask::Runtime::MandatoryInform("Black and allow lists for %s Accelerators *both* have entries! \n" 
                                            " This is an API violation, PTask behavior undefined...\n",
                                            AccClassString(pAccelerator->GetClass()));
            bBLWLWarningShown = TRUE;
        }

        if(bAllowListed && bDenyListed) {
            PTask::Runtime::MandatoryInform("%s is both deny and allow listed. Disabling %s...\n" 
                                            " This is an API violation, (PTask preferring deny list entry)...\n",
                                            pAccelerator->GetDeviceName(),
                                            pAccelerator->GetDeviceName());
            return FALSE;
        }

        return ((bDenyListHasEntries && !bDenyListed && bAllowListed) ||
                (bAllowListHasEntries && bAllowListed && !bDenyListed));
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds all the accelerator objects in the list
    /// 			to the schedulers accelator-tracking data structures. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="accelerators"> [in,out] [in,out] If non-null, the accelerators. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::AddAccelerators(
        __in ACCELERATOR_CLASS eClass,
        __in std::vector<Accelerator*> &vaccelerators
        )
    {
        LockAcceleratorMaps();

        // the input is *all known* accelerators of a given class. Here, we add them to the scheduler
        // data structures, and  if the concurrency is artificially limited by the runtime, we only add
        // a subset to the enabled list. sort the accelerate list.  First weed out denylisted ones. 

        std::vector<Accelerator*> rejected;
        std::vector<Accelerator*> selected;
        std::vector<Accelerator*> denylist;
        std::vector<Accelerator*> allowlist;
        std::vector<Accelerator*>::iterator vi;

        for(vi=vaccelerators.begin(); vi!=vaccelerators.end(); vi++) {
            assert((*vi)->GetClass() == eClass);
            if(IsEnablable(*vi)) {
                allowlist.push_back(*vi);
            } else {
                Accelerator * pAcc = *vi;
                PTask::Runtime::Inform("Disabling accelerator: %s based on deny/allow enable lists...\n", pAcc->GetDeviceName());
                denylist.push_back(*vi);
            }
        }

        // figure out which concurrency limit applies this list of accelerators.
        // ptask supports a different concurrency limit API for host tasks.
        
		BOOL bHostAccelerators = vaccelerators.size() && (*vaccelerators.begin())->IsHost();
        int nConcurrencyLimit = bHostAccelerators ? 
			PTask::Runtime::GetMaximumHostConcurrency() :
			PTask::Runtime::GetMaximumConcurrency();

        if(nConcurrencyLimit != 0 && allowlist.size() > (UINT) nConcurrencyLimit) {
            if (PTask::Runtime::IsVerbose()) {
                PTask::Runtime::Inform("Maximum (%s) concurrency limited!" 
                                       " Using only (%d) of (%d) accelerators.\n",
                                       (bHostAccelerators ? "HostAcc":"GPU"),
                                       nConcurrencyLimit,
                                       allowlist.size());
            }
        }


        Accelerator::SelectBestAccelerators(nConcurrencyLimit, allowlist, selected, rejected);
        for(vi=denylist.begin(); vi!=denylist.end(); vi++)
            rejected.push_back(*vi);

        if(selected.size() == 0 && rejected.size() > 0) {

            // the user disabled everything. Nothing is going to work!
            // Better at least inform the user about this. 
            
            PTask::Runtime::MandatoryInform("XXXXX"
                                            "XXXXX  %s::%s(%d): All %d available %s accelerators are disabled because they \n"  
                                            "XXXXX  where deny/allow-list filtered or rejected by limits on concurrency/accelerator attributes. \n"
                                            "XXXXX  PTask will be unable to dispatch tasks of this class unless \n"
                                            "XXXXX  some are dynamically enabled with PTask::Runtime::DynamicEnableAccelerator!\n"
                                            "XXXXX",
                                            __FILE__,
                                            __FUNCTION__,
                                            __LINE__,
                                            static_cast<int>(rejected.size()),
                                            AccClassString(eClass));
        }
        
        for(vi=rejected.begin(); vi!=rejected.end(); vi++) {
            // track this accelerator object, but do not add
            // it to the enabled or available list.
            Accelerator * pAccelerator = *vi;
            UINT uiAcceleratorId = pAccelerator->GetAcceleratorId();
            m_vMasterAcceleratorList.insert(pAccelerator);
            m_vAcceleratorMap[uiAcceleratorId] = pAccelerator;
            m_pDeviceManager->AddDevice(pAccelerator);
        }

        for(vi=selected.begin(); vi!=selected.end(); vi++) {
            Accelerator * pAccelerator = *vi;
            ACCELERATOR_CLASS eClass = pAccelerator->GetClass();
            UINT uiAcceleratorId = pAccelerator->GetAcceleratorId();
            m_vMasterAcceleratorList.insert(pAccelerator);
            m_vEnabledAccelerators[eClass]->insert(pAccelerator);
            m_vAvailableAccelerators[eClass]->insert(pAccelerator);
            m_vAcceleratorMap[uiAcceleratorId] = pAccelerator;
            m_pDeviceManager->AddDevice(pAccelerator);
            if(eClass != ACCELERATOR_CLASS_HOST || !m_accmap[eClass].size())
                m_accmap[eClass].insert(pAccelerator);
        }
        UnlockAcceleratorMaps();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enables/disables the use of the given accelerator by the scheduler: use to
    ///             restrict/manage the pool of devices actually used by PTask to dispatch tasks.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    /// <param name="bEnabled">     true to enable, false to disable. </param>
    ///
    /// <returns>   PTASK_OK for successful addition/removal to/from the deny list. 
    ///             PTASK_ERR_UNINITIALIZED if the runtime is already initialized.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT 
    Scheduler::SetAcceleratorEnabled(
        Accelerator * pAccelerator, 
        BOOL bEnabled
        )
    {   
        if(!PTask::Runtime::IsInitialized())
            return PTASK_ERR_UNINITIALIZED;
        if(g_pScheduler == NULL) 
            return PTASK_ERR_UNINITIALIZED;
        return g_pScheduler->__SetAcceleratorEnabled(pAccelerator, bEnabled);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   get the deny and allow lists relevant to the given accelerator. 
    ///             helper function. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2014. </remarks>
    ///
    /// <param name="eClass">           the accelerator class. </param>
    /// <param name="pAllowList">       [in,out] If non-null, list of allows. </param>
    /// <param name="pDenyList">       [in,out] If non-null, list of denys. </param>
    /// <param name="bCreateIfAbsent">  The create if absent. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Scheduler::GetRestrictionLists(
        __in ACCELERATOR_CLASS eClass,
        __out std::set<int>** pAllowList,
        __out std::set<int>** pDenyList,
        __in BOOL bCreateIfAbsent
        )
    {
        std::set<int>* pClassDenyList = NULL;
        std::set<int>* pClassAllowList = NULL;
        std::map<ACCELERATOR_CLASS, std::set<int>*>::iterator mi;

        if(pDenyList != NULL) {
            mi=s_vDenyListDeviceIDs.find(eClass);
            if(mi==s_vDenyListDeviceIDs.end()) {
                if(bCreateIfAbsent) {
                    pClassDenyList = new std::set<int>();
                    s_vDenyListDeviceIDs[eClass] = pClassDenyList;
                }
            } else {
                pClassDenyList = mi->second;
            }
            *pDenyList = pClassDenyList;
        }

        if(pAllowList != NULL) {
            mi=s_vAllowListDeviceIDs.find(eClass);
            if(mi==s_vAllowListDeviceIDs.end()) {
                if(bCreateIfAbsent) {
                    pClassAllowList = new std::set<int>();
                    s_vAllowListDeviceIDs[eClass] = pClassAllowList;
                }
            } else {
                pClassAllowList = mi->second;
            }
            *pAllowList = pClassAllowList;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enables/disables the use of the given accelerator by the scheduler: this version
    ///             adds/removes entries from a static deny list. On startup the scheduler checks
    ///             devices against this deny list to enable/disable them. Do not call after
    ///             PTask::Runtime::Initialized is called.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="eClass">       If non-null, the accelerator. </param>
    /// <param name="nPSDeviceID">  platform specific device id. </param>
    /// <param name="bEnable">      true to enable, false to disable. </param>
    ///
    /// <returns>   PTASK_OK for successful addition/removal to/from the deny list. 
    ///             PTASK_ERR_ALREADY_INITIALIZED if the runtime is already initialized.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT 
    Scheduler::EnableAccelerator(
        ACCELERATOR_CLASS eClass,
        int nPSDeviceID,
        BOOL bEnabled
        )
    {   
        if(PTask::Runtime::IsInitialized())
            return PTASK_ERR_ALREADY_INITIALIZED;
        if(g_pScheduler != NULL) 
            return PTASK_ERR_ALREADY_INITIALIZED;

        std::set<int>* pClassDenyList = NULL;
        std::set<int>* pClassAllowList = NULL;
        GetRestrictionLists(eClass, &pClassAllowList, &pClassDenyList, TRUE);
        assert(pClassAllowList != NULL);
        assert(pClassDenyList != NULL); 

        if(bEnabled) {
            // put the accelerator on the allow
            // list, erase it from the deny list if present
            pClassAllowList->insert(nPSDeviceID);
            pClassDenyList->erase(nPSDeviceID);
        } else {
            pClassDenyList->insert(nPSDeviceID);
            pClassAllowList->erase(nPSDeviceID);
        }

        return PTASK_OK;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pAccelerator' is enabled. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   true if enabled, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::IsEnabled(
        Accelerator * pAccelerator
        )
    {
        static BOOL bLiveGraphWarning = FALSE;
        if(!PTask::Runtime::IsInitialized())
            return PTASK_ERR_UNINITIALIZED;
        if(g_pScheduler == NULL) 
            return PTASK_ERR_UNINITIALIZED;
        g_pScheduler->LockQuiescenceState();
        if(g_pScheduler->m_bLiveGraphs) {
            if(!bLiveGraphWarning) {
                PTask::Runtime::Inform("WARNING %s(%s) called with live graphs: result has no freshness guarantees\n"
                                       "     ---suppressing future warnings about this\n",
                                       __FUNCTION__,
                                       pAccelerator->GetDeviceName());
                bLiveGraphWarning = TRUE;
            }
        }
        g_pScheduler->LockAcceleratorMaps();
        BOOL bEnabled = g_pScheduler->__IsEnabled(pAccelerator);
        g_pScheduler->UnlockAcceleratorMaps();
        g_pScheduler->UnlockQuiescenceState();
        return bEnabled;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enables/disables the use of the given accelerator by the scheduler: use to
    ///             restrict/manage the pool of devices actually used by PTask to dispatch tasks.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    /// <param name="bEnabled">     true to enable, false to disable. </param>
    ///
    /// <returns>   success/failure </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT 
    Scheduler::__SetAcceleratorEnabled(
        Accelerator * pAccelerator, 
        BOOL bEnabled
        )
    {
        if(pAccelerator == NULL) 
            return PTASK_ERR_INVALID_PARAMETER;
        if(m_vMasterAcceleratorList.find(pAccelerator) == m_vMasterAcceleratorList.end())
            return PTASK_ERR_NOT_FOUND;

        LockAcceleratorMaps();
        PTRESULT ptresult = PTASK_OK;
        ACCELERATOR_CLASS eClass = pAccelerator->GetClass();
        std::set<Accelerator*>* pClassSet = m_vEnabledAccelerators[eClass];
        std::set<Accelerator*>* pAvailableSet = m_vAvailableAccelerators[eClass];

        if(bEnabled && (pClassSet->find(pAccelerator) != pClassSet->end())) {

            UINT uiMaxConcurrency = PTask::Runtime::GetMaximumConcurrency();
            if(uiMaxConcurrency > 0) {

                // the caller is trying to enable an additional device, and the number of devices PTask can use
                // is being artifically limited by a call to SetMaximumConcurrency. If enabling this
                // accelerator will cause us to exceed the artificial limit, then we must fail the request.
                // Since we impose this limit separately per accelerator class, we must first count up the
                // currently enabled accelerators for the corresponding class so we can then determine whether
                // the new enabled count will exceed the limit. Note that we *cannot* assume the current enable
                // count is equal to the imposed limit, since it is perfectly acceptable to *disable*
                // accelerators when there is a limit in place;  so a previous call to disable something can
                // leave us in a state where that assumption would be wrong. 

                UINT uiCurrentlyEnabled = 0;
                ACCELERATOR_CLASS eClass = pAccelerator->GetClass();
                std::set<Accelerator*>::iterator si;
                for(si=pClassSet->begin(); si!=pClassSet->end(); si++) {
                    Accelerator * pOtherAcc = *si;
                    if(pOtherAcc->GetClass() == eClass) {
                        assert(pOtherAcc != pAccelerator);
                        uiCurrentlyEnabled++;
                    }
                }

                if(uiCurrentlyEnabled+1 > uiMaxConcurrency) {

                    // enabling this accelerator would cause us to exceed the limit. 
                    // so we must fail the request. Give the user some console feedback
                    // too, since we can't count on the programmer to check the return.
                    PTask::Runtime::Inform("\tAn attempt was made to enable an accelerator that would exceed the\n"
                                           "\tlimit previously imposed by a call to PTask::Runtime::SetMaximumConcurrency.\n"
                                           "\tYou must disable another accelerator first before this accelerator can be enabled.\n"
                                           "\tAlternatively, remove your call to SetMaximumConcurrency!\n");
                    UnlockAcceleratorMaps();
                    return PTASK_ERR_TOO_MANY_ACCELERATORS;
                }

            }
        }


        if((bEnabled && (pClassSet->find(pAccelerator) != pClassSet->end())) ||
           (!bEnabled && (pClassSet->find(pAccelerator) == pClassSet->end()))) {

            // if the accelerator is already in the requested state, there is nothing to be done
            // because the caller is trying enable/disable an already enabled/disabled accelerator
            ptresult = PTASK_OK;

        } else if(m_vInflightAccelerators.find(pAccelerator) != m_vInflightAccelerators.end()) {

            // the caller is trying to disable an accelerator, but it is currently 
            // being used in a dispatch. The caller needs to wait until the accelerator 
            // is not in use and retry the call. 
            assert(pClassSet->find(pAccelerator) != pClassSet->end());
            assert(pClassSet->find(pAccelerator) == pClassSet->end());
            ptresult = PTASK_ERR_INFLIGHT;

        } else {

            if(bEnabled) {

                // make sure all the accelerator lists are consistent with the state
                // transition being requested. It should not be already enabled or in flight
                assert(pClassSet->find(pAccelerator) == pClassSet->end());
                assert(m_vInflightAccelerators.find(pAccelerator) == m_vInflightAccelerators.end());
                assert(m_vMasterAcceleratorList.find(pAccelerator) != m_vMasterAcceleratorList.end());
                assert(pAvailableSet->find(pAccelerator) == pAvailableSet->end());
                pClassSet->insert(pAccelerator);
                pAvailableSet->insert(pAccelerator);
                ptresult = PTASK_OK;

            } else {

                // make sure all the accelerator lists are consistent with the state
                // transition being requested. It should not be already enabled or in flight
                assert(pClassSet->find(pAccelerator) != pClassSet->end());
                assert(m_vInflightAccelerators.find(pAccelerator) == m_vInflightAccelerators.end());
                assert(m_vMasterAcceleratorList.find(pAccelerator) != m_vMasterAcceleratorList.end());
                assert(pAvailableSet->find(pAccelerator) != pAvailableSet->end());
                pClassSet->erase(pAccelerator);
                pAvailableSet->erase(pAccelerator);
                ptresult = PTASK_OK;
            }

        }

        if(pAvailableSet->size() == 0) {
            m_vbAcceleratorsAvailable[eClass] = FALSE;
            ResetEvent(m_vhAcceleratorsAvailable[eClass]);
        } else {
            m_vbAcceleratorsAvailable[eClass] = TRUE;
            SetEvent(m_vhAcceleratorsAvailable[eClass]);
            SetEvent(m_hAcceleratorQueue);
        }
        UnlockAcceleratorMaps();
        return PTASK_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Locks the scheduler: by locking the task queues and accelerator lists,
    ///             ensuring no scheduling can occur. Use with caution. </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::LockScheduler(
        VOID
        )
    {
        if(g_pScheduler != NULL)
            g_pScheduler->__LockScheduler();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unlocks the scheduler by unlocking the task queues and accelerator lists,
    ///             ensuring no scheduling can occur. Use with caution. </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::UnlockScheduler(
        VOID
        )
    {
        if(g_pScheduler != NULL) 
            g_pScheduler->__UnlockScheduler();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Locks the scheduler: by locking the task queues and accelerator lists,
    ///             ensuring no scheduling can occur. Use with caution. </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__LockScheduler(
        VOID
        )
    {
        LockQuiescenceState();
        LockTaskQueues();
        LockAcceleratorMaps();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unlocks the scheduler by unlocking the task queues and accelerator lists,
    ///             ensuring no scheduling can occur. Use with caution. </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__UnlockScheduler(
        VOID
        )
    {
        UnlockAcceleratorMaps();
        UnlockTaskQueues();
        UnlockQuiescenceState();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pAccelerator' is enabled. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   true if enabled, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::__IsEnabled(
        Accelerator * pAccelerator
        )
    {
        assert(AcceleratorLockIsHeld());
        ACCELERATOR_CLASS eClass = pAccelerator->GetClass();
        BOOL bResult = m_vEnabledAccelerators[eClass]->find(pAccelerator) != m_vEnabledAccelerators[eClass]->end();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates accelerators for all devices and runtimes in the environment </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Scheduler::CreateAccelerators(
        VOID
        ) 
    {
        // create directX accelerators
        if(PTask::Runtime::GetUseDirectX()) {
#ifndef DIRECTXCOMPILESUPPORT
            PTask::Runtime::Inform("PTask build does not support DirectX compilation...enumerating DX accelerators anyway!\n");
#endif
            vector<Accelerator*> dxdevices;
            DXAccelerator::EnumerateAccelerators(dxdevices);
            AddAccelerators(ACCELERATOR_CLASS_DIRECT_X, dxdevices);
        }

#ifdef CUDA_SUPPORT
        if(PTask::Runtime::GetUseCUDA()) {
            PTask::Runtime::Inform("PTask build supports CUDA, enumerating CUDA devices...\n");
            vector<Accelerator*> candidateAccelerators;
            CUAccelerator::EnumerateAccelerators(candidateAccelerators);
            AddAccelerators(ACCELERATOR_CLASS_CUDA, candidateAccelerators);
            LockAcceleratorMaps();

            // enable p2p support up front where it is supported
            std::map<ACCELERATOR_CLASS, std::set<Accelerator*>*>::iterator mi;
            mi=m_vEnabledAccelerators.find(ACCELERATOR_CLASS_CUDA);
            if(mi!=m_vEnabledAccelerators.end()) {
                
                std::set<Accelerator*>* pCUDAAccs = mi->second;
                std::set<Accelerator*>::iterator si;
                std::set<Accelerator*>::iterator ti;
                if(pCUDAAccs != NULL && pCUDAAccs->size() > 0) {
                    Accelerator * pA = *(pCUDAAccs->begin());
                    BOOL bShouldDefault = !PTask::Runtime::GetApplicationThreadsManagePrimaryContext();
                    pA->InitializeTLSContext(PTTR_APPLICATION, bShouldDefault, FALSE);
                }

                for(si=pCUDAAccs->begin(); si!=pCUDAAccs->end(); si++) {
                    Accelerator * pA = *si;
                    pA->Lock();
                    for(ti=pCUDAAccs->begin(); ti!=pCUDAAccs->end(); ti++) {
                        Accelerator * pB = *ti;
                        if(pA != pB) {
                            pB->Lock();
                            pA->SupportsDeviceToDeviceTransfer(pB);
                            pB->Unlock();
                        }
                    }
                    pA->Unlock();
                }
            }
            UnlockAcceleratorMaps();           
        }
#else
        PTask::Runtime::Inform("PTask build does not support CUDA...\n");
#endif

#ifdef OPENCL_SUPPORT
        // get any open CL devices
        if(PTask::Runtime::GetUseOpenCL()) {
            PTask::Runtime::Inform("PTask build supports OpenCL, enumerating OpenCL devices...\n");
            vector<Accelerator*> cldevices;
            CLAccelerator::EnumerateAccelerators(cldevices);
            AddAccelerators(cldevices);
        }
#else
        PTask::Runtime::Inform("PTask build does not support OpenCL...\n");
#endif

        // create host accelerator objects
        if(PTask::Runtime::GetUseHost()) {
            std::vector<Accelerator*> candidates;
            HostAccelerator::EnumerateAccelerators(candidates);            
            AddAccelerators(ACCELERATOR_CLASS_HOST, candidates);            
        }

        // make sure the runtime knows what devices we are dealing with. 
        LockAcceleratorMaps();
        UINT nDevices = m_pDeviceManager->GetPhysicalAcceleratorCount();
        PTask::Runtime::SetPhysicalAcceleratorCount(nDevices);
        UnlockAcceleratorMaps();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Select best accelerator 
    ///             "Best" is determined by the following sort order (somewhat arbitrarily) is:
    ///             * highest runtime version support
    ///             * support for concurrent kernels
    ///             * highest core count
    ///             * fastest core clock
    ///             * biggest memory
    ///             * device enumeration order (ensuring we will usually choose   
    ///               the same physical device across multiple back ends)        
    ///               </summary> 
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="candidates">   [in,out] [in,out] If non-null, the candidates. </param>
    /// <param name="target">       Target for the. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Scheduler::SelectBestAccelerator(
        vector<Accelerator*>& candidates, 
        ACCELERATOR_CLASS target
        )
    {
        // call with accelerator lock held!
        assert(AcceleratorLockIsHeld());
        Accelerator * pBest = NULL;
        // randomize the selection by searching 
        // sometimes from the front and sometimes from the
        // back. In the case where there are multiple identical
        // graphics cards, this prevents the first one in the list
        // from becoming a hot spot.
        switch(target) {
        case ACCELERATOR_CLASS_DIRECT_X: {
            if(rand() % 2) {
                vector<Accelerator*>::iterator vi;
                for(vi=candidates.begin(); vi!=candidates.end(); vi++) {
                    DXAccelerator * pDXAccelerator = (DXAccelerator*) *vi;
                    DXAccelerator * pDXBest = (DXAccelerator*) pBest;
                    if(pBest == NULL) {
                        pBest = pDXAccelerator;
                    } else {
                        if(pDXAccelerator->GetFeatureLevel() > pDXBest->GetFeatureLevel()) {
                            pBest = pDXAccelerator;
                        }
                    }
                }
                return pBest;
            } else {
                vector<Accelerator*>::reverse_iterator vi(candidates.end());
                vector<Accelerator*>::reverse_iterator rlast(candidates.begin());
                while(vi != rlast) {
                    DXAccelerator * pDXAccelerator = (DXAccelerator*) *vi++;
                    DXAccelerator * pDXBest = (DXAccelerator*) pBest;
                    if(pBest == NULL) {
                        pBest = pDXAccelerator;
                    } else {
                        if(pDXAccelerator->GetFeatureLevel() > pDXBest->GetFeatureLevel()) {
                            pBest = pDXAccelerator;
                        }
                    }
                }
                return pBest;
            }
                                         }
#ifdef CUDA_SUPPORT
        case ACCELERATOR_CLASS_CUDA: {
            if(rand() % 2) {
                vector<Accelerator*>::iterator vi;
                for(vi=candidates.begin(); vi!=candidates.end(); vi++) {
                    CUAccelerator * pCUAccelerator = (CUAccelerator*) *vi;
                    CUAccelerator * pCUBest = (CUAccelerator*) pBest;
                    if(pBest == NULL) {
                        pBest = pCUAccelerator;
                    } else {
                        int nCandidateCores =  pCUAccelerator->GetCoreCount();
                        int nBestCores = pCUBest->GetCoreCount();
                        if(nCandidateCores > nBestCores)
                            pBest = pCUAccelerator;
                    }
                }
                return pBest;
            } else {
                vector<Accelerator*>::reverse_iterator vi(candidates.end());
                vector<Accelerator*>::reverse_iterator rlast(candidates.begin());
                while(vi != rlast) {
                    CUAccelerator * pCUAccelerator = (CUAccelerator*) *vi++;
                    CUAccelerator * pCUBest = (CUAccelerator*) pBest;
                    if(pBest == NULL) {
                        pBest = pCUAccelerator;
                    } else {
                        int nCandidateCores =  pCUAccelerator->GetCoreCount();
                        int nBestCores = pCUBest->GetCoreCount();
                        if(nCandidateCores > nBestCores)
                            pBest = pCUAccelerator;
                    }
                }
                return pBest;
            }
        }
#endif
        default:									 
        case ACCELERATOR_CLASS_OPEN_CL:
        case ACCELERATOR_CLASS_REFERENCE:
        case ACCELERATOR_CLASS_HOST:
        case ACCELERATOR_CLASS_UNKNOWN:
            return candidates[0];
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return a map from class to accelerator objects for all
    ///             accelerators which have a unique memory space. This function is
    ///             not to be used when the graph is running, but rather, should be 
    ///             used at block pool creation to simplify finding candidate memory
    ///             spaces for block pooling.  
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/30/2013. </remarks>
    ///
    /// <param name="accmap">   [in,out] [in,out] If non-null, the accmap. </param>
    ///
    /// <returns>   The number of found accelerators. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::map<ACCELERATOR_CLASS, std::set<Accelerator*>> * 
    Scheduler::EnumerateBlockPoolAccelerators(
        VOID
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler == NULL) 
            return 0;
        g_pScheduler->__EnumerateBlockPoolAccelerators(g_pScheduler->m_accmap);
        return &g_pScheduler->m_accmap;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return a map from class to accelerator objects for all
    ///             accelerators which have a unique memory space. This function is
    ///             not to be used when the graph is running, but rather, should be 
    ///             used at block pool creation to simplify finding candidate memory
    ///             spaces for block pooling.  
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/30/2013. </remarks>
    ///
    /// <param name="accmap">   [in,out] [in,out] If non-null, the accmap. </param>
    ///
    /// <returns>   The number of found accelerators. </returns>
    ///-------------------------------------------------------------------------------------------------

    int 
    Scheduler::__EnumerateBlockPoolAccelerators(
        std::map<ACCELERATOR_CLASS, std::set<Accelerator*>>& accmap
        )
    {
        static ACCELERATOR_CLASS classes[] = {
            ACCELERATOR_CLASS_DIRECT_X,
            ACCELERATOR_CLASS_OPEN_CL,
            ACCELERATOR_CLASS_CUDA ,
            ACCELERATOR_CLASS_REFERENCE,
            ACCELERATOR_CLASS_HOST,
            ACCELERATOR_CLASS_UNKNOWN
            };
        static int nClasses = sizeof(classes)/sizeof(ACCELERATOR_CLASS);
        int nFound = 0;
        if(accmap.size()==0) {
            for(int i=0; i<nClasses; i++) {
                ACCELERATOR_CLASS eClass = classes[i];
                std::set<Accelerator*> accs;
                __FindEnabledCapableAccelerators(eClass, accs);
                size_t nAccelerators = accs.size();
                if(nAccelerators) {
                    if(eClass == ACCELERATOR_CLASS_HOST) {
                        accmap[eClass].insert(*accs.begin());
                        nFound++;
                    } else {
                        accmap[eClass] = accs;
                        nFound += (int)nAccelerators;
                    }
                }
            }
        }
        return nFound;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enumerate available accelerators. This is a test method, and provides a list of
    ///             currently available accelerator objects by class; the method can be called
    ///             without a lock, which means that any list returned is not guaranteed to be
    ///             consistent with the scheduler's current state--in other words, the results are
    ///             unactionable unless the runtime is known to be in a quiescent state (e.g. during
    ///             test scenarios!)
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="accClass"> The accumulate class. </param>
    /// <param name="vaccs">    [in,out] [in,out] If non-null, the vaccs. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::EnumerateAvailableAccelerators(
        ACCELERATOR_CLASS accClass, 
        std::set<Accelerator*>& vaccs
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler == NULL) 
            return FALSE;
        if(g_pScheduler->m_vLiveGraphs.size() > 0) {
            PTask::Runtime::Warning("EnumerateAvailableAccelerators called!\n"
                                    "\tThis is a test method, and provides a list of currently available\n"
                                    "\taccelerator objects by class; the method can be called without a lock,\n"
                                    "\twhich means that any list returned is not guaranteed to be consistent with\n"
                                    "\tthe scheduler's current state--in other words, the results are unactionable\n"
                                    "\tunless the runtime is known to be in a quiescent state (e.g. during test scenarios!)");
        }
        return g_pScheduler->__FindEnabledCapableAccelerators(accClass, vaccs) > 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enumerate enabled accelerators for the given class. This provides a list of
    ///             currently enabled accelerator objects by class; the method can be called
    ///             without a lock, so if the caller requires a consistent view (with no lock, 
    ///             another thread may enable or disable an accelerator after this list returned),
    ///             LockScheduler should be called.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 9/8/2013. </remarks>
    ///
    /// <param name="accClass"> The accumulate class. </param>
    /// <param name="vaccs">    [in,out] [in,out] If non-null, the vaccs. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::EnumerateEnabledAccelerators(
        ACCELERATOR_CLASS accClass, 
        std::set<Accelerator*>& vaccs
        )
    {
        if(g_pScheduler != NULL) 
            return g_pScheduler->__FindEnabledCapableAccelerators(accClass, vaccs) > 0;
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enumerate all accelerators. This provides a list of
    ///             all known accelerator objects by class, whether the accelerators are actually
    ///             in use by the scheduler or not. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="accClass"> The accumulate class. </param>
    /// <param name="vaccs">    [in,out] [in,out] If non-null, the vaccs. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::EnumerateAllAccelerators(
        ACCELERATOR_CLASS accClass, 
        std::set<Accelerator*>& vaccs
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler == NULL) 
            return FALSE;
        std::vector<Accelerator*> vvaccs;
        if(EnumerateAllAccelerators(accClass, vvaccs)) {
            vaccs.insert(vvaccs.begin(), vvaccs.end());
            return TRUE;
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enumerate all accelerators. This provides a list of
    ///             all known accelerator objects by class, whether the accelerators are actually
    ///             in use by the scheduler or not. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="accClass"> The accumulate class. </param>
    /// <param name="vaccs">    [in,out] [in,out] If non-null, the vaccs. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::EnumerateAllAccelerators(
        ACCELERATOR_CLASS accClass, 
        std::vector<Accelerator*>& vaccs
        )
    {        
        if(g_pScheduler == NULL) {
            PTask::Runtime::MandatoryInform("%s::%s(%s) called with uninitialized runtime...no result!\n",
                                            __FILE__,
                                            __FUNCTION__,
                                            AccClassString(accClass));
            return FALSE;
        }

        if(accClass == ACCELERATOR_CLASS_UNKNOWN) {
            vaccs.assign(g_pScheduler->m_vMasterAcceleratorList.begin(), 
                         g_pScheduler->m_vMasterAcceleratorList.end());
        } else {
            set<Accelerator*>::iterator si;
            for(si=g_pScheduler->m_vMasterAcceleratorList.begin();
                si!=g_pScheduler->m_vMasterAcceleratorList.end(); si++) {
                if((*si)->GetClass() == accClass)
                    vaccs.push_back(*si);
            }
        }

        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Searches for the first capable accelerators. return a list of all the
    ///             accelerators in the system that should be capable of executing code for the given
    ///             accelerator class, regardless of whether that accelerator is currently in flight
    ///             or not.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="target">       Target class of the task being matched. </param>
    /// <param name="candidates">   [in,out] the candidate list. </param>
    ///
    /// <returns>   The found capable accelerators. </returns>
    ///-------------------------------------------------------------------------------------------------

    int 
    Scheduler::__FindEnabledCapableAccelerators(
        ACCELERATOR_CLASS target,
        std::set<Accelerator*>& candidates
        ) 
    {
        // need not be called with accelerator lock if it's being
        // called to find an accelerator to compile or create
        // device-side buffers. 

        set<Accelerator*>::iterator si;
        for(si=m_vEnabledAccelerators[target]->begin(); si!=m_vEnabledAccelerators[target]->end(); si++) {
            candidates.insert(*si);
        }
        if(candidates.size() == 0 && PTask::Runtime::GetUseReferenceDrivers()) {
            // if we wanted a DX hardware accelerator but didn't find
            // one, go ahead and use the reference driver. 
            if(target == ACCELERATOR_CLASS_DIRECT_X)
                target = ACCELERATOR_CLASS_REFERENCE;
            for(si=m_vEnabledAccelerators[target]->begin(); si!=m_vEnabledAccelerators[target]->end(); si++) {
                candidates.insert(*si);
            }
        }
        return (int) candidates.size();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Searches for the first capable accelerators. return a list of all the
    ///             accelerators in the system that should be capable of executing code for the given
    ///             accelerator class, regardless of whether that accelerator is currently in flight
    ///             or not.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="target">       Target class of the task being matched. </param>
    /// <param name="candidates">   [in,out] the candidate list. </param>
    ///
    /// <returns>   The found capable accelerators. </returns>
    ///-------------------------------------------------------------------------------------------------

    int 
    Scheduler::__FindAllCapableAccelerators(
        ACCELERATOR_CLASS target,
        std::set<Accelerator*>& candidates
        ) 
    {
        // need not be called with accelerator lock if it's being
        // called to find an accelerator to compile or create
        // device-side buffers. 
        set<Accelerator*>::iterator si;
        for(si=m_vMasterAcceleratorList.begin(); si!=m_vMasterAcceleratorList.end(); si++) {
            Accelerator * pAccelerator = *si;
            if(pAccelerator->GetClass() == target) {
                candidates.insert(*si);
            }
        }
        if(candidates.size() == 0 && PTask::Runtime::GetUseReferenceDrivers()) {
            // if we wanted a DX hardware accelerator but didn't find
            // one, go ahead and use the reference driver. 
            if(target == ACCELERATOR_CLASS_DIRECT_X)
                target = ACCELERATOR_CLASS_REFERENCE;
            for(si=m_vMasterAcceleratorList.begin(); si!=m_vMasterAcceleratorList.end(); si++) {
                if((*si)->GetClass() == target) {
                    candidates.insert(*si);
                }
            }
        }
        return (int) candidates.size();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Searches for the first capable accelerators. return a list of all the
    ///             accelerators in the system that should be capable of executing code for the given
    ///             accelerator class, regardless of whether that accelerator is currently in flight
    ///             or not.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="target">       Target class of the task being matched. </param>
    /// <param name="candidates">   [in,out] the candidate list. </param>
    ///
    /// <returns>   The found capable accelerators. </returns>
    ///-------------------------------------------------------------------------------------------------

    int 
    Scheduler::FindAllCapableAccelerators(
        ACCELERATOR_CLASS target,
        std::set<Accelerator*>& candidates
        ) 
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler == NULL) 
            return 0;
        return g_pScheduler->__FindAllCapableAccelerators(target, candidates);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Searches for the first capable accelerators. return a list of all the
    ///             accelerators in the system that should be capable of executing code for the given
    ///             accelerator class, regardless of whether that accelerator is currently in flight
    ///             or not.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="target">       Target class of the task being matched. </param>
    /// <param name="candidates">   [in,out] the candidate list. </param>
    ///
    /// <returns>   The found capable accelerators. </returns>
    ///-------------------------------------------------------------------------------------------------

    int 
    Scheduler::FindEnabledCapableAccelerators(
        ACCELERATOR_CLASS target,
        std::set<Accelerator*>& candidates
        ) 
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler == NULL) 
            return 0;
        return g_pScheduler->__FindEnabledCapableAccelerators(target, candidates);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Searches for the strongest available accelerator for the given
    ///             class. Generally we want to avoid this being our only criterion
    ///             because locality is a first class consideration. Generally,
    ///             we should be looking at the task.
    ///              </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="target">   Target class of the task being matched. </param>
    ///
    /// <returns>   null if it fails, else the found available accelerator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Scheduler::FindStrongestAvailableAccelerator(
        ACCELERATOR_CLASS target
        ) 
    {
        // call with accelerator lock held!
        assert(AcceleratorLockIsHeld());

        if(m_vAvailableAccelerators[target]->size() == 0) {
            if(target == ACCELERATOR_CLASS_HOST) {
                PTask::Runtime::MandatoryInform("XXX: Scheduler has no host accelerators available!\n");
            }
            return NULL;
        }
        std::set<Accelerator*>* pAvailable = m_vAvailableAccelerators[target];
        assert(pAvailable->size());

        // if its a host accelerator that is being requested then don't bother with a bunch 
        // of extra work building temporary lists and sorting by accelerator strength.
        // they are all the same. In fact we should have an unlimited number of them.

        if(target == ACCELERATOR_CLASS_HOST) {
            return *(pAvailable->begin());
        }

        // if there is only on instance of the desired type our job is simple
        // too. There is a subtlety, which is the presence of multiple backend GPU
        // frameworks--if there are no cross-framework sharing dependences, then
        // we know we can always just return the singular item. 

        if(pAvailable->size() == 1 && !m_bCrossRuntimeSharingChecksRequired) {
            return *(pAvailable->begin());
        }

        // if we get here, we have to choose among many, or the potential for
        // an underlying device to be busy despite the presence of an accelerator
        // in the available list, so we need to do some extra work to decide

        vector<Accelerator*> candidates;
        set<Accelerator*>::iterator si;
        if(m_bCrossRuntimeSharingChecksRequired) {
            for(si=pAvailable->begin(); si!=pAvailable->end(); si++) {
                Accelerator * pAccelerator = *si;
                PhysicalDevice * pDevice = pAccelerator->GetPhysicalDevice();
                pDevice->Lock();
                if(!pDevice->IsBusy()) {
                    candidates.push_back(*si);
                }
                pDevice->Unlock();
            }
        } else {
            candidates.assign(pAvailable->begin(), pAvailable->end());
        }
        if(candidates.size() > 0) {
            return SelectBestAccelerator(candidates, target);
        }

        // if we wanted a DX hardware accelerator but didn't find
        // one, go ahead and use the reference driver. 
        if(target == ACCELERATOR_CLASS_DIRECT_X && PTask::Runtime::GetUseReferenceDrivers())
            target = ACCELERATOR_CLASS_REFERENCE;
        for(si=m_vAvailableAccelerators[target]->begin(); si!=m_vAvailableAccelerators[target]->end(); si++) {
            if((*si)->GetClass() == target) {
                if(!PTask::Runtime::GetUseReferenceDrivers()) {
                    PTask::Runtime::HandleError("runtime not configured for reference drivers!");
                    return NULL;
                }
                return (*si);
            }
        }
        return NULL;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   If there is a scheduling choice that must be made based
    /// 			on affinity the inability of the system to migrate
    /// 			unmarshallable datablocks, find an accelerator that 
    /// 			fits the constraints, if possible.  Call with accelerator
    /// 			lock held.
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task. </param>
    ///
    /// <returns>   null if it fails, else the found affinitized accelerator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Scheduler::FindAffinitizedAccelerator(
        Task* pTask
        ) 
    {
        assert(AcceleratorLockIsHeld());

        BOOL bConstrained = FALSE;
        Accelerator * pAssignment = NULL;    
        ACCELERATOR_CLASS eClass = pTask->GetAcceleratorClass();
        std::set<Accelerator*>* pAvailableSet = m_vAvailableAccelerators[eClass];
        if(Runtime::MultiGPUEnvironment()) {
            std::set<Accelerator*> constraints;
            std::set<Accelerator*> preferences;
            if(pTask->CollectSchedulingConstraints(constraints, preferences)) {
                size_t nConstraints = constraints.size();
                if(nConstraints > 0) {
                    if(nConstraints > 1) {
                        Runtime::HandleError("%s: conflicting scheduling constraints!\n", __FUNCTION__);
                    } else {
                        bConstrained = TRUE;
                        Accelerator * pAccelerator = *(constraints.begin());
                        if(pAvailableSet->find(pAccelerator) != pAvailableSet->end()) {
                            pAssignment = pAccelerator;
                            return pAssignment;
                        }
                    }
                }
                std::set<Accelerator*>::iterator pi;
                for(pi=preferences.begin(); pi!=preferences.end(); pi++) {
                    Accelerator * pPreferred = *pi;
                    if(pAvailableSet->find(pPreferred) != pAvailableSet->end()) {
                        pAssignment = pPreferred;
                        return pAssignment;
                    }
                }
            }
        }
        return pAssignment;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Searches for the first available accelerator for the given ptask. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTask">                    [in] non-null, the task. </param>
    /// <param name="pnDependentAccelerators">  [in,out] If non-null, the pn dependent accelerators. </param>
    /// <param name="pppDependentAccelerators"> [in,out] If non-null, the ppp dependent accelerators. </param>
    ///
    /// <returns>   null if it fails, else the found available accelerator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Scheduler::FindStrongestAvailableAccelerators(
        Task* pTask,
        int * pnDependentAccelerators,
        Accelerator *** pppDependentAccelerators
        )
    {
        // Return the most powerful available accelerator that matches the accelerator class of task.
        // This simple variant just returns the first accelerator with the given class call with
        // accelerator lock held. Assign any dependent accelerators as well, using the same policy. 
        
        assert(AcceleratorLockIsHeld());

        Accelerator * pAssignment = FindAffinitizedAccelerator(pTask);
        if(pAssignment == NULL) {
            ACCELERATOR_CLASS acc = pTask->GetAcceleratorClass();
            pAssignment = FindStrongestAvailableAccelerator(acc);
        }
        
        if(pAssignment != NULL) {

            // attempt to assign any dependent accelerators. If we cannot satisfy the task's 
            // dependent accelerator requirements, we cannot dispatch it, so we drop the 
            // dispatch accelerator assignment as well.    
                     
            if(!AssignDependentAccelerators(pTask,
                                            pnDependentAccelerators,
                                            pppDependentAccelerators,
                                            pAssignment,
                                            NULL)) {
                return NULL;
            }
        }

        return pAssignment;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Choose best based on histogram. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <param name="pCandidates">      [in,out] If non-null, the candidates. </param>
    /// <param name="pInputHistogram">  [in,out] If non-null, a histogram mapping accelerators to
    ///                                 accelerators to number of inputs for this task which are
    ///                                 already materialized there. When this is null, ignore it.
    ///                                 When it is available, if it is possible to choose dependent
    ///                                 assignments to maximize locality based on this histogram, do
    ///                                 it. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Scheduler::SelectMaxLocalityAccelerator(
        std::set<Accelerator*>* pCandidates,
        std::map<Accelerator*, UINT>* pInputHistogram
        )
    {
        assert(pCandidates != NULL);
        if(pCandidates == NULL) return NULL;
        size_t nCandidates = pCandidates->size();
        if(nCandidates == 0)
            return NULL;
        if(nCandidates == 1 || pInputHistogram == NULL)
            return *(pCandidates->begin());
        Accelerator * pBestFit = NULL;
        UINT uiBestScore = 0;
        std::set<Accelerator*>::iterator ai;
        std::map<Accelerator*, UINT>::iterator mi;
        for(ai=pCandidates->begin(); ai!=pCandidates->end(); ai++) {
            Accelerator * pA = *ai;
            mi=pInputHistogram->find(pA);
            if(mi!=pInputHistogram->end()) {
                UINT uiScore = mi->second; 
                if(pBestFit == NULL || uiScore > uiBestScore) {
                    pBestFit = pA;
                    uiBestScore = uiScore;
                }
            } else {
                if(pBestFit == NULL)
                    pBestFit = pA;
            }
        }
        return pBestFit;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Assign dependent accelerators. If this task requires resources in addition to the
    ///             dispatch accelerator, assign them. Return true if an assignment can be made
    ///             (indicating that this task can be dispatched).
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/9/2012. </remarks>
    ///
    /// <param name="pTask">                    [in] non-null, the task. </param>
    /// <param name="pnDependentAccelerators">  [in,out] If non-null, the pn dependent accelerators. </param>
    /// <param name="pppDependentAccelerators"> [in,out] If non-null, the ppp dependent accelerators. </param>
    /// <param name="pDispatchAccelerator">     [in,out] If non-null, the assignment. </param>
    /// <param name="pInputHistogram">          [in,out] If non-null, a histogram mapping
    ///                                         accelerators to accelerators to number of inputs for
    ///                                         this task which are already materialized there. When
    ///                                         this is null, ignore it. When it is available, if it
    ///                                         is possible to choose dependent assignments to
    ///                                         maximize locality based on this histogram, do it. 
    ///                                         </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::AssignDependentAccelerators(
        Task * pTask,
        int * pnDependentAccelerators,
        Accelerator *** pppDependentAccelerators,
        Accelerator * pDispatchAccelerator,
        std::map<Accelerator*, UINT>* pInputHistogram
        )
    {
        assert(AcceleratorLockIsHeld());

        UNREFERENCED_PARAMETER(pDispatchAccelerator);
        Accelerator ** ppDepAccs = NULL;
        pTask->Lock();
        int nDepCount = pTask->GetDependentBindingClassCount();
        pTask->Unlock();

        if(nDepCount == 0) {

            // if the number of dependent accelerator classes is zero, then the dispatch accelerator is the
            // only execution resource required by this task. We have trivially satisfied the dependent
            // accelerator constraints, so we return TRUE to indicate it is possible to continue setting up
            // for dispatch. 
            
            *pnDependentAccelerators = 0;
            *pppDependentAccelerators = 0;
             return TRUE;

        } else {

            // allocate space for the accelerator list
            ppDepAccs = (Accelerator**) calloc(nDepCount, sizeof(Accelerator*));
        }

        // There are dependences on other accelerators. Currently, we require that those accelerators be
        // of a different class than the dispatch accelerator. We only add the accelerators to the task 
        // if we can collect all that are needed without draining the scheduler's resources.
        
        BOOL bDependencesAssigned = FALSE;  
        BOOL bDependencesAssignable = TRUE; 
        BOOL bMandatoryConstraints = FALSE;
        BOOL bPreferenceConstraints = FALSE; 
        BOOL bAffinityConstraints = FALSE;
        std::vector<Accelerator*> vCandidateDepAccs;

        bAffinityConstraints = pTask->HasDependentAffinities();
        if(bAffinityConstraints) {

            pTask->Lock();
            std::set<Accelerator*> vMandatorySet;
            std::set<Accelerator*> vPreferenceSet;
            std::map<Port*, Accelerator*> vMandatoryMap;
            std::map<Port*, Accelerator*> vPreferenceMap;


            // collect all the constraints associated with this task. The return value,
            // indicates whether any affinities constrain the scheduler, and if so, the
            // port->contraint and constrained assignment sets are populated by the
            // call to collect them. Since we've already made a call to see if this
            // task has dependent affnities, the return value bloody well better be TRUE!
            BOOL bConstrained = pTask->CollectDependentSchedulingConstraints(vMandatoryMap, 
                                                                             vMandatorySet, 
                                                                             vPreferenceMap, 
                                                                             vPreferenceSet);
            UINT uiMandatoryConstraints = static_cast<UINT>(vMandatorySet.size());
            UINT uiPreferenceConstraints = static_cast<UINT>(vPreferenceSet.size());
            bMandatoryConstraints = uiMandatoryConstraints != 0;
            bPreferenceConstraints = uiPreferenceConstraints != 0;

            if(!bConstrained) assert(bConstrained);                              // see above. shouldn't be here otherwise
            assert(uiMandatoryConstraints > 0 || bPreferenceConstraints > 0);    // can't have both mandatory and preference
            assert(uiMandatoryConstraints == 0 || bPreferenceConstraints == 0);  // should have at least one of the constraint types
            assert(uiMandatoryConstraints <= 1);                                 // can't have a mandatory assignment to more than 1 target

            if(bMandatoryConstraints) {

                assert(!bPreferenceConstraints);
                if(uiMandatoryConstraints != (UINT)nDepCount) {

                    // currently this is an error condition. We don't allow more than one
                    // distinct dependent accelerator to be assigned to a single task.
                    // Complain loudly about the abuse. 

                    pTask->Unlock();
                    PTask::Runtime::HandleError("%s:%s:%d: vMandatorySet.size(%d) != nDepCount(%d)\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                __LINE__,
                                                vMandatorySet.size(),
                                                nDepCount);

                    bDependencesAssignable = FALSE;
                    bDependencesAssigned = FALSE;
                    free(ppDepAccs);
                    return FALSE;
                }

                // if the mandatory set size equals the depcount then all that remains is
                // to check whether the members of that set are available. If they are not
                // available, we will not be able to schedule this task until an outstanding
                // dispatch frees the mandatory accelerator. 

                assert(nDepCount == 1);
                bDependencesAssignable = TRUE;
                Accelerator * pMandatoryDepAcc = *vMandatorySet.begin();
                ACCELERATOR_CLASS eClass = pMandatoryDepAcc->GetClass();
                BOOL bMandatoryAvailable = m_vAvailableAccelerators[eClass]->find(pMandatoryDepAcc) != 
                                            m_vAvailableAccelerators[eClass]->end();                                
                bDependencesAssigned = bMandatoryAvailable;
                bDependencesAssignable = bMandatoryAvailable;
                if(bMandatoryAvailable) 
                    vCandidateDepAccs.push_back(pMandatoryDepAcc);

            }

            if(bPreferenceConstraints) {

                // we are dealing with a soft constraint. We know a list of 
                // candidates to choose from, actual assignment performed below. 
                // make sure that we don't also have mandatory constraints.
               
                assert(!bMandatoryConstraints);
                vCandidateDepAccs.assign(vPreferenceSet.begin(), vPreferenceSet.end());
                bDependencesAssigned = TRUE;
                bDependencesAssignable = vPreferenceSet.size() >= nDepCount;
            }

            pTask->Unlock();
        }

        if(!bDependencesAssigned && bDependencesAssignable) {

            // we know a satisfying assignment must exist, but we have not added any accelerators to the
            // candidate pool. We should really only get to this point if the task has no affinities
            // configured, or if the preferential affinity does not include sufficient candidates to
            // complete the assignment. In either case, our stratagy is to find a suitable candidate based
            // on the underlying scheduling strategy. We call SelectMaxLocality which will fall-back to
            // other policies if the data aware policy is not in use. This area of the code needs a re-
            // write, since the function names don't accurately reflect dynamic policy.
            
            pTask->Lock();
            int nRequiredCount = 0;
            ACCELERATOR_CLASS accClass = pTask->GetDependentAcceleratorClass(0, nRequiredCount);
            assert(nRequiredCount == 1 && "only one dependence per class currently supported!"); 
            assert(pDispatchAccelerator->GetClass() != accClass);
            assert(!bMandatoryConstraints);
            assert(vCandidateDepAccs.size() == 0 || bPreferenceConstraints);

            std::set<Accelerator*>* pAv = m_vAvailableAccelerators[accClass];
            size_t nClassCount = pAv->size();
            if(nClassCount == 0) {

                // nothing is available matching the accelerator class requirements for
                // the dependent assignment. Declare defeat. 
                bDependencesAssigned = FALSE;
                bDependencesAssignable = FALSE;

            } else { 

                // find the accelerator that matches the underlying policy most
                // closely. Generally this means look for the one that preserves locality,
                // but the call to SelectMaxLocalityAccelerator below will fall back to other
                // policies if the scheduler is not configured to use the data aware policy. 
                
                Accelerator * pAcc = SelectMaxLocalityAccelerator(pAv, pInputHistogram);
                vCandidateDepAccs.push_back(pAcc);
                bDependencesAssigned = TRUE;
            }

            pTask->Unlock();
        }

        if(vCandidateDepAccs.size() >= nDepCount && bDependencesAssigned) {

            // we've found a satisfying assignment for all the 
            // task's dependent accelerators. The candidate pool can be larger than
            // the number of actual dependences iff the task has preferential 
            // affinity rather than mandatory, 

            pTask->Lock();
            for(int i=0; i<nDepCount; i++) {
                ppDepAccs[i] = vCandidateDepAccs[i];
            }
            *pnDependentAccelerators = nDepCount;
            *pppDependentAccelerators = ppDepAccs;
            pTask->Unlock();
            return TRUE;
        }

        // could not satisfy the dependent accelerator constraints. 
        // this should force the dispatch attempt to fail.         
        free(ppDepAccs);
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Claim an accelerator: take it off the available list, mark it inflight, and mark
    ///             its physical device busy.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/10/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
    /// <param name="pTask">        [in,out] If non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Scheduler::__ClaimAccelerator(
        Accelerator * pAccelerator,
        Task * pTask
        ) 
    {
        assert(AcceleratorLockIsHeld());

        assert(pAccelerator != NULL);
        if(pAccelerator != NULL) {
            ACCELERATOR_CLASS eClass = pAccelerator->GetClass();
            assert(m_vAvailableAccelerators[eClass]->find(pAccelerator) != m_vAvailableAccelerators[eClass]->end());
            assert(m_vInflightAccelerators.find(pAccelerator) == m_vInflightAccelerators.end());

            std::map<Accelerator*, Task*>::iterator mi;
            for(mi=m_vInflightAccelerators.begin(); mi!=m_vInflightAccelerators.end(); mi++) {
                if(mi->second == pTask && mi->first->GetClass() == eClass) {
                    PTask::Runtime::HandleError("%s: concurrent inflight attempts for %s!\n", 
                                                __FUNCTION__,
                                                pTask->GetTaskName());
                }
            }

            m_vAvailableAccelerators[eClass]->erase(pAccelerator);
            m_vInflightAccelerators[pAccelerator] = pTask;

            if(m_bCrossRuntimeSharingChecksRequired) {
                PhysicalDevice * pDev = pAccelerator->GetPhysicalDevice();
                pDev->Lock();
                assert(!pDev->IsBusy());
                pDev->SetBusy(TRUE);
                pDev->Unlock();
            }
            if(m_vAvailableAccelerators[eClass]->size() == 0) {
                ResetEvent(m_vhAcceleratorsAvailable[eClass]);
                m_vbAcceleratorsAvailable[eClass] = FALSE;
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the accelerator described by pAccelerator. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::ReleaseAccelerator(
        Accelerator * pAccelerator
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL) {
            LockScheduler();
            g_pScheduler->__ReleaseAccelerator(pAccelerator);
            UnlockScheduler();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the accelerator described by pAccelerator. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__ReleaseAccelerator(
        Accelerator * pAccelerator
        )
    {
        assert(AcceleratorLockIsHeld());
        assert(pAccelerator != NULL);
        if(pAccelerator != NULL) {
            ACCELERATOR_CLASS eClass = pAccelerator->GetClass();
            assert(m_vAvailableAccelerators[eClass]->find(pAccelerator) == m_vAvailableAccelerators[eClass]->end());
            assert(m_vInflightAccelerators.find(pAccelerator) != m_vInflightAccelerators.end());
            m_vAvailableAccelerators[eClass]->insert(pAccelerator);
            m_vInflightAccelerators.erase(pAccelerator);
            if(m_bCrossRuntimeSharingChecksRequired) {
                PhysicalDevice * pDev = pAccelerator->GetPhysicalDevice();
                pDev->Lock();
                assert(pDev->IsBusy());
                pDev->SetBusy(FALSE);
                pDev->Unlock();
            }
            SetEvent(m_hAcceleratorQueue);
            m_vbAcceleratorsAvailable[eClass] = TRUE;
            SetEvent(m_vhAcceleratorsAvailable[eClass]);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Match an accelerator to a ptask based on current scheduling mode. In the general
    ///             case, matching returns a single accelerator, and leaves the dependent accelerator
    ///             list set to NULL. If the task has dependences on other accelerators (e.g. as many
    ///             CUDA+thrust-based HostTasks do), the dependency list will come back non-empty as
    ///             well).
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTask">                    [in] non-null, the task. </param>
    /// <param name="pnDependentAccelerators">  [in,out] non-null, the number of dependent
    ///                                         accelerators. </param>
    /// <param name="pppDependentAccelerators"> [out] If non-null, the dependent accelerators. The
    ///                                         caller is *NOT* responsible for freeing this list, as
    ///                                         it will be freed by the runtime after dispatch
    ///                                         completes. </param>
    ///
    /// <returns>   null if it fails, else a pointer to the dispatch accelerator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Scheduler::MatchAccelerators(
        Task * pTask, 
        int * pnDependentAccelerators,
        Accelerator *** pppDependentAccelerators
        )
    {
        // call with accelerator lock held! 
        // ---------------------------------
        // Return the most powerful available accelerator that matches
        // the accelerator class of the task, according to the current scheduling policy. 
        assert(AcceleratorLockIsHeld());

        Accelerator * pAssignment = NULL;

        if(PTask::Runtime::MultiGPUEnvironment() && 
            PTask::Runtime::GetSchedulingMode() == SCHEDMODE_DATADRIVEN) {

            // the data-driven scheduling mode is only interesting in the presence of multiple accelerators:
            // if there is just one, all view are in either that GPU's memory space or the CPUs, so
            // scheduling to prefer a particular GPU is meaningless. Hence, we invoke the DATADRIVEN
            // scheduler only if there is more than one. 
            
            pAssignment = MatchAcceleratorsDataAware(pTask, 
                                                     pnDependentAccelerators,
                                                     pppDependentAccelerators);
        } else {

            // default policies schedule without regard to locality. These policies are invoked if there is
            // only one accelerator, or if the user explicitly put the runtime in a mode other than DATADRIVEN. 
            pAssignment = FindStrongestAvailableAccelerators(pTask,
                                                             pnDependentAccelerators,
                                                             pppDependentAccelerators);
        }  
        CHECK_MANDATORY_ASSIGNMENTS(pTask, pAssignment, pnDependentAccelerators, pppDependentAccelerators);
        return pAssignment;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Is the given accelerator on the available list? </summary>
    ///
    /// <remarks>   crossbac, 6/28/2013. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   true if available, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::__IsAvailable(
        __in Accelerator * pAccelerator
        )
    {
        assert(AcceleratorLockIsHeld());
        assert(pAccelerator != NULL);
        if(pAccelerator != NULL) {
            ACCELERATOR_CLASS eClass = pAccelerator->GetClass();            
            std::set<Accelerator*>* pAvailSet = m_vAvailableAccelerators[eClass];
            std::set<Accelerator*>::iterator mi = pAvailSet->find(pAccelerator);
            return (mi != pAvailSet->end());
        } 
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Match an accelerator to a ptask using the "data aware" policy. In the general
    ///             case, matching returns a single accelerator, and leaves the dependent accelerator
    ///             list set to NULL. If the task has dependences on other accelerators (e.g. as many
    ///             CUDA+thrust-based HostTasks do), the dependency list will come back non-empty as
    ///             well).
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTask">                    [in] non-null, the task. </param>
    /// <param name="pnDependentAccelerators">  [in,out] non-null, the number of dependent
    ///                                         accelerators. </param>
    /// <param name="pppDependentAccelerators"> [out] If non-null, the dependent accelerators. The
    ///                                         caller is *NOT* responsible for freeing this list, as
    ///                                         it will be freed by the runtime after dispatch
    ///                                         completes. </param>
    ///
    /// <returns>   null if it fails, else a pointer to the dispatch accelerator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Scheduler::MatchAcceleratorsDataAware(
        Task * pTask, 
        int * pnDependentAccelerators,
        Accelerator *** pppDependentAccelerators
        )
    {
        // call with accelerator lock held.
        assert(AcceleratorLockIsHeld());
        *pnDependentAccelerators = 0;
        *pppDependentAccelerators = NULL;

        // Schedule preferring locality first as long as there is potential to benefit
        // from it in the current execution environment. In general this potential derives
        // from the presence of multiple accelerators capable of running the same task. 
        // Hence, we don't bother trying to deal with locality if: 
        // 1) there are no available accelerators 
        // 2) there is only one accelerator of the given class in the universe.
        
        ACCELERATOR_CLASS eClass = pTask->GetAcceleratorClass();
        set<Accelerator*>* pAvailSet = m_vAvailableAccelerators[eClass];
        if(eClass == ACCELERATOR_CLASS_HOST && pAvailSet->size() == 0) {
            PTask::Runtime::MandatoryInform("XXXX: WARNING: Scheduler: No available Host accelerator! Consider reprovisioning!\n");
        }
        if(pAvailSet->size() == 0)
            return NULL;

        if(eClass == ACCELERATOR_CLASS_HOST && !pTask->HasDependentAcceleratorBindings()) {
            // if this is an host task, and has no dependent accelerator bindings, we
            // can make an assignment trivially, since there is no reason to prefer
            // on "host" accelerator over another. 
            *pnDependentAccelerators = 0;
            *pppDependentAccelerators = NULL;
            return (*(pAvailSet->begin()));
        }


        BOOL bConstrained = FALSE;
        BOOL bMandatoryAssignable = FALSE;
        Accelerator * pAssignment = NULL;      
        Accelerator * pTaskMandatory = NULL;
        Accelerator * pDepMandatory = NULL;
        std::set<Accelerator*> preferences;
        std::set<Accelerator*> constraints;

        if(pTask->HasSchedulingConstraints(&pTaskMandatory, &pDepMandatory, bMandatoryAssignable)) {

            if(bMandatoryAssignable) {

                // fast path result--mandatory affinity is set such that we have only
                // one possible way to assign accelerators to this task, and it is simple
                // enough to be determined at the time the check for constraints is made.
                // If mandatory dependent assignments must be made they are known an the
                // task is either host task, or there is a mandatory assignment for it too. 

                if(eClass == ACCELERATOR_CLASS_HOST) {

                    assert(pTaskMandatory == NULL);
                    assert(pDepMandatory != NULL);
                    if(!__IsAvailable(pDepMandatory)) 
                        return NULL;

                    // the dependent accelerator is available!
                    *pnDependentAccelerators = 1;
                    *pppDependentAccelerators = (Accelerator**) calloc(1, sizeof(Accelerator*));
                    (*pppDependentAccelerators)[0] = pDepMandatory;
                    return (*(pAvailSet->begin()));

                } else {

                    // if the task is not a host task and we have a mandatory
                    // assignment, check for availability and proceed accordingly.
                    assert(pDepMandatory==NULL);
                    assert(!pTask->HasDependentAcceleratorBindings());
                    return __IsAvailable(pTaskMandatory) ? pTaskMandatory : NULL;
                }
            }

            // if we couldn't make a fast path assignment we go on to make a more detailed examination of
            // the constraints imposed by the task structure: in particular, the presence of preferences
            // (which are non-mandatory) may require us to choose amongst many options. The call to
            // CollectSchedulingConstraints may discover mandatory constraints we did not know about above.
            
            if(pTask->CollectSchedulingConstraints(constraints, preferences)) {

                bConstrained = TRUE; 
                if(constraints.size() > 0) {
                    // we discovered a mandatory top-level constraint
                    // we did not find in the previous call. 
                    assert(constraints.size() == 1);
                    pTaskMandatory = *(constraints.begin());
                    if(!__IsAvailable(pTaskMandatory))
                        return NULL;
                    pAssignment = pTaskMandatory;
                    if(AssignDependentAccelerators(pTask,
                                                   pnDependentAccelerators,
                                                   pppDependentAccelerators,
                                                   pAssignment,
                                                   NULL)) {
                        return pAssignment;
                    } else {
                        return NULL;
                    }
                }
            }
        }

        map<Accelerator*, UINT> histogram;
        BOOL bLocalityImpossible = (BuildInputHistogram(pTask, histogram) == 0);

        // try to take care of a few common cases up front. For example, 
        // if this is a host task, don't bother working hard to select the
        // accelerator for the base task--take the first available and
        // start working on any dependent ports.
        
        if(eClass == ACCELERATOR_CLASS_HOST) {
            pAssignment = (*(pAvailSet->begin()));
            return AssignDependentAccelerators(pTask,
                                               pnDependentAccelerators,
                                               pppDependentAccelerators,
                                               pAssignment,
                                               &histogram) ? pAssignment : NULL;
        }

        // To schedule for locality, get a histogram mapping accelerators materialized input counts.
        // The histogram builder will return the number of histogram buckets, so that a zero return
        // value means no inputs are materialized, so there there is no way to make a locality-
        // preferential choice. 
        
        if(bLocalityImpossible) {

            if(preferences.size() > 0) {

                // if the hist map is empty, it means that all our inputs are in the CPU memory domain. It
                // would be good to look downstream of course, but for now, just assume it's a crapshoot if we
                // can't make a decision based on our producer. Take anything from the affinitized group if
                // there is something

                pAssignment = (*(preferences.begin()));
                return AssignDependentAccelerators(pTask,
                                                   pnDependentAccelerators,
                                                   pppDependentAccelerators,
                                                   pAssignment,
                                                   &histogram) ? pAssignment : NULL;
            }

            // No preferences can be determined. Schedule using default policy.
            return FindStrongestAvailableAccelerators(pTask, pnDependentAccelerators, pppDependentAccelerators);
        }

        // we have a non-empty histogram. If we can make an assigment where the accelerator is present
        // in both the histogram and the affinitized accelerator list, do so. 

        if(preferences.size() > 0) {

            // if we can make a best fit matching, and satisfy dependent accelerator
            // constraints, then do so and return. Otherwise, check to see if we've been waiting...
            if(NULL != (pAssignment = FindDataAwareBestFit(histogram, preferences))) {
                return AssignDependentAccelerators(pTask,
                                                   pnDependentAccelerators,
                                                   pppDependentAccelerators,
                                                   pAssignment,
                                                   &histogram) ? pAssignment : NULL;
            }

            // if we haven't been waiting too long, go ahead and insist that we are willing to wait for
            // something matches our policy and the specified affinity. 
            if(pTask->GetEffectivePriority() <= PTask::Runtime::GetIgnoreLocalityThreshold())
                 return NULL;
        }

        // the affinitized accelerators are not among choose from amongst all available accelerators,
        // OR there were no affinitized accelerators, OR we have crossed the ignore locality threshold.
        // collect a list of everyone available and assign from there. 

        if(NULL != (pAssignment = FindDataAwareBestFit(histogram, *pAvailSet))) {    
            return AssignDependentAccelerators(pTask,
                                                pnDependentAccelerators,
                                                pppDependentAccelerators,
                                                pAssignment,
                                                &histogram) ? pAssignment : NULL;
        }

        // the best fit is not available. If our effective priority is high, indicating that either we
        // have high static priority or that this ptask has been waiting for a while, then just use
        // whatever is available. Otherwise, return NULL, and hope the accelerator we want becomes
        // available soon. 
        if(pTask->GetEffectivePriority() > PTask::Runtime::GetIgnoreLocalityThreshold()) {
             return FindStrongestAvailableAccelerators(pTask, 
                                                       pnDependentAccelerators, 
                                                       pppDependentAccelerators);
        }

        // no assignment possible under the current policy. 
        // return NULL to indicate we should try to schedule
        // something else instead if possible.
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Look at the input blocks for the given ptask, and build a histogram mapping accelerators
    ///     to the number of inputs that are currently up to date in that memory domain. Return the
    ///     number of histogram buckets, (a zero return value indicates there is no way to make a
    ///     good assignment based on locality alone because no  
    ///     inputs are up to date on any accelerators).
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. 
    /// 			call with accelerator lock held.
    /// 			</remarks>
    ///
    /// <param name="pTask">        [in] non-null, the task. </param>
    /// <param name="histogram">    [in,out] non-null, the histogram. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Scheduler::BuildInputHistogram(
        Task* pTask, 
        std::map<Accelerator*, UINT> &hist
        )
    {
        // Look at the ptask's inputs. Choose the accelerator from the
        // given list, in whose memory domain the most inputs are materialized.
        // If no candidate has any inputs materialized, do not make an
        // assignment (since that will be handled by the caller's logic).

        map<UINT, Port*>::iterator mi;
        map<UINT, Port*>* pPortMap = pTask->GetInputPortMap();
        assert(pPortMap != NULL);
        for(mi=pPortMap->begin(); mi!=pPortMap->end(); mi++) {
            Port * pPort = mi->second;
            if(pPort->GetPortType() == INITIALIZER_PORT ||
                pPort->GetPortType() == META_PORT) {
                // initializer ports produce values
                // in response to a pull, which makes them
                // a) irrelevant to the matching decision
                // b) guaranteed to return null on a peek.
                // meta ports are irrelevant because the runtime
                // consumes them--so communication is the same
                // regardless of where we schedule the ptask
                continue;
            }
            Datablock * pBlock = pPort->Peek();
            if(pBlock == NULL) continue;
            pBlock->Lock();
            std::vector<Accelerator*> vaccs;
            UINT nAccelerators = pBlock->GetValidViewAccelerators(vaccs);
            pBlock->Unlock();
            if(nAccelerators != 0) {
                for(std::vector<Accelerator*>::iterator sti=vaccs.begin(); sti!=vaccs.end(); sti++) {
                    Accelerator * pAccelerator = *sti;
                    if(pTask->GetAcceleratorClass() != pAccelerator->GetClass())
                        continue;
                    int nCount = 0;
                    if(hist.find(pAccelerator) != hist.end()) 
                        nCount = hist[pAccelerator];
                    nCount++;
                    hist[pAccelerator] = nCount;
                }
            }
        }
        return (UINT) hist.size();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Given a list of candidates, find the one that is the best fit, where "best"
    ///             adheres to the data-aware policy. We want the accelerator where the most inputs
    ///             are already up-to- date.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="histogram">    [in] hist: accelerator->(materialized input count). </param>
    /// <param name="candidates">   [in] non-null, the candidates. </param>
    ///
    /// <returns>   null if it fails, else the found accelerator. </returns>
    ///
    /// ### <param name="pTask">    [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Scheduler::FindDataAwareBestFit(
        std::map<Accelerator*, UINT> &histogram,
        std::set<Accelerator*> &candidates
        )
    {
        // Look at the ptask's inputs. Choose the accelerator from the given list, in whose memory
        // domain the most inputs are materialized. If no candidate has any inputs materialized, do not
        // make an assignment (since that will be handled by the caller's logic). find the best-scoring
        // accelerator that is actually present in the candidate list.
         
        UINT nBestCount = 0;
        Accelerator * pBestFit = NULL;
        map<Accelerator*, UINT>::iterator ai;
        for(ai=histogram.begin(); ai!=histogram.end(); ai++) {
            Accelerator * pCandidate = ai->first;
            UINT uiCandidateScore = ai->second;
            if(uiCandidateScore > nBestCount && candidates.find(pCandidate) != candidates.end()) {
                pBestFit = pCandidate;
                nBestCount = uiCandidateScore;
            }
        }
        // might be null!
        return pBestFit;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Begins a dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Scheduler::ScheduleOrEnqueue(
        Task * pTask,
        BOOL& bQueued,
        BOOL bBypassQ
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler == NULL)
            return FALSE;
        return g_pScheduler->__ScheduleOrEnqueue(pTask, bQueued, bBypassQ);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Begins a dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Scheduler::__ScheduleOrEnqueue(
        Task * pTask,
        BOOL& bQueued,
        BOOL bBypassQ
        )
    {
        // add this task to the ready queue,
        // and signal the scheduler thread that
        // there is work available. 
        LockQuiescenceState();
        LockTaskQueues();

        bQueued = FALSE;
        BOOL bScheduled = FALSE;
        if(m_bWaitingForGlobalQuiescence || m_bInGlobalQuiescentState) {
            m_vDeferredQ.push_back(pTask);
        } else {
            if(AssignDispatchAccelerator(pTask, bBypassQ)) {
                bScheduled = TRUE;
            } else {
                if(__IsDeferrable(pTask)) {
                    m_vDeferredQ.push_back(pTask);
                } else if(!__IsDispatchable(pTask)) {
                    PTask::Runtime::MandatoryInform("%s(%s) graph is not dispatchable/deferrable...deferring anyway...\n",
                                                    __FUNCTION__,
                                                    pTask->GetTaskName());
                    m_vDeferredQ.push_back(pTask);
                } else {
                    m_vReadyQ.push_back(pTask);
                    bQueued = TRUE;
                    // reset the usage timer - timing the wait in the ready queue begins now.
                    pTask->GetUsageTimer()->reset();
                    __UpdateTaskQueueView();
                }
                __UpdateQuiescenceWaiters(pTask->GetGraph());
            }
        }
        __UpdateQuiescenceHint();
        __UpdateOKToScheduleView();
        UnlockTaskQueues();
        UnlockQuiescenceState();
        return bScheduled;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Begins a dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Scheduler::BeginDispatch(
        Task * pTask
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler == NULL)
            return FALSE;
        return g_pScheduler->__BeginDispatch(pTask);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Begins a dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Scheduler::__BeginDispatch(
        Task * pTask
        )
    {
        // add this task to the ready queue,
        // and signal the scheduler thread that
        // there is work available. 
        LockQuiescenceState();
        LockTaskQueues();

        BOOL bQueued = FALSE;
        if(m_bWaitingForGlobalQuiescence || m_bInGlobalQuiescentState) {
            m_vDeferredQ.push_back(pTask);
        } else {
            if(__IsDeferrable(pTask)) {
                m_vDeferredQ.push_back(pTask);
            } else if(!__IsDispatchable(pTask)) {
                PTask::Runtime::MandatoryInform("%s(%s) graph is not dispatchable/deferrable...deferring anyway...\n",
                                                __FUNCTION__,
                                                pTask->GetTaskName());
                m_vDeferredQ.push_back(pTask);
            } else {
                m_vReadyQ.push_back(pTask);
                bQueued = TRUE;
                // reset the usage timer - timing the wait in the ready queue begins now.
                pTask->GetUsageTimer()->reset();
                __UpdateTaskQueueView();
            }
            __UpdateQuiescenceWaiters(pTask->GetGraph());
        }
        __UpdateQuiescenceHint();
        __UpdateOKToScheduleView();
        UnlockTaskQueues();
        UnlockQuiescenceState();
        return bQueued;
    }



    ///-------------------------------------------------------------------------------------------------
    /// <summary>   The given task has completed dispatch. We want to move its dispatch accelerator
    ///             and any dependent accelerator objects from the inflight list to the available
    ///             list, signal that dispatch has completed, and that accelerators are available.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task that is completing. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::AbandonCurrentDispatch(
        Task * pTask
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler == NULL)
            return;
        return g_pScheduler->__AbandonCurrentDispatch(pTask);
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   The given task has completed dispatch. We want to move its dispatch accelerator
    ///             and any dependent accelerator objects from the inflight list to the available
    ///             list, signal that dispatch has completed, and that accelerators are available.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task that is completing. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::CompleteDispatch(
        Task * pTask
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler == NULL)
            return;
        return g_pScheduler->__CompleteDispatch(pTask);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   The given task has completed dispatch. We want to move its dispatch accelerator
    ///             and any dependent accelerator objects from the inflight list to the available
    ///             list, signal that dispatch has completed, and that accelerators are available.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task that is completing. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__CompleteDispatch(
        Task * pTask
        )
    {
        assert(pTask != NULL);
        MARKRANGEENTER(L"__CompleteDispatch");
        LockAcceleratorMaps();
        MARKRANGEENTER(L"__CompleteDispatch(locked)");
        __ReleaseDispatchResources(pTask);
        UnlockAcceleratorMaps();
        MARKRANGEEXIT();
        __UpdateQuiescenceWaiters(pTask->GetGraph());
        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   The given task has completed dispatch. We want to move its dispatch accelerator
    ///             and any dependent accelerator objects from the inflight list to the available
    ///             list, signal that dispatch has completed, and that accelerators are available.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task that is completing. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__AbandonCurrentDispatch(
        Task * pTask
        )
    {
        assert(pTask != NULL);
        LockQuiescenceState();
        LockTaskQueues();
        LockAcceleratorMaps();

        // abandoning a dispatch cannot rely on getting the dispatch accelerator member from the task,
        // since the scheduler might have assigned an accelerator, but the dispatch was abandoned
        // before the task set it's m_pDispatchAccelerator member. So there could still be entries in
        // the inflight list for the task after we complete the normal procedure for releasing dispatch
        // resources. Follow that normal procedure first, then go ferret out any such dangling entries,
        // and forcibly reset any dispatch accelerators the task is holding. 

        __ReleaseDispatchResources(pTask);
        __ReleaseDanglingDispatchResources(pTask);

        // finally, it is unlikely, but not impossible that we abandan a dispatch while waiting for
        // quiescence. This shouldn't happen since waiting for quiescence is *supposed* to allow
        // inflight tasks to complete rather than killin them in mid-flight. However, the runtime can
        // be terminated at any time or a graph destructor can be called by the user (which they are
        // not supposed to do but may be unable to avoid) which trumps all of our rules about what
        // order things should be torn down. Consequently, we need to duplicate any of the
        // book keeping relevant to quiescence from the normal dispatch complete pattern here.
        
        __UpdateQuiescenceWaiters(pTask->GetGraph());
        __UpdateQuiescenceHint();
        __UpdateOKToScheduleView();

        UnlockAcceleratorMaps();
        UnlockTaskQueues();
        UnlockQuiescenceState();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the dispatch resources held by a task on a normal 
    ///             dispatch completion. </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__ReleaseDispatchResources(
        __in Task * pTask
        )
    {
        // accelerator lock and queue lock must be held!
        assert(AcceleratorLockIsHeld());
        pTask->Lock();
        for(int i=0; i<pTask->GetDependentBindingClassCount(); i++) {
            Accelerator * pAccelerator = pTask->GetAssignedDependentAccelerator(i);
            if(pAccelerator) {
                __ReleaseAccelerator(pAccelerator);
            }
        }
        pTask->ReleaseDependentAccelerators();
        pTask->Unlock();
        Accelerator * pAccelerator = pTask->GetDispatchAccelerator();
        if(pAccelerator != NULL) {
            __ReleaseAccelerator(pTask->GetDispatchAccelerator());
        }

    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases any dispatch resources held by a task on a failed dispatch completion
    ///             (abandoned dispatch can leave things in the inflight list for a task that the
    ///             task doesn't know about).
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__ReleaseDanglingDispatchResources(
        __in Task * pTask
        )
    {
        // abandoning a dispatch cannot rely on getting the dispatch accelerator member from the task,
        // since the scheduler might have assigned an accelerator, but the dispatch was abandoned
        // before the task set it's m_pDispatchAccelerator member. So there could still be entries in
        // the inflight list for the task after we complete the normal procedure for releasing dispatch
        // resources. Go ferret out any such dangling entries, and forcibly reset any dispatch
        // accelerators the task is holding. 
        
        // Frankly, this should not happen. If we abandon a dispatch for a task, it means we have been
        // awakened from a wait on the scheduler to assign us a dispatch accelerator. If we have been
        // awakened with them claimed, the Release should be sufficient to clean up the inflight list.
        // This is reasonable defensive cleanup, but we assert if we actually find something that needs
        // to get cleaned up. 
      
        assert(AcceleratorLockIsHeld());
        
        std::set<Accelerator*>::iterator si;
        std::map<Accelerator*, Task*>::iterator mi;
        std::set<Accelerator*> vDanglingAccelerators;
        for(mi=m_vInflightAccelerators.begin(); mi!=m_vInflightAccelerators.end(); mi++) {
            if(mi->second == pTask) {
                assert(false);
                vDanglingAccelerators.insert(mi->first);
            }
        }
        for(si=vDanglingAccelerators.begin(); si!=vDanglingAccelerators.end(); si++) {
            Accelerator * pAccelerator = *si;
            if(pAccelerator != NULL) {
                __ReleaseAccelerator(pAccelerator);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   On dispatch completion or abandonment, if there are graphs waiting for quiescence
    ///             or the scheduler is waiting for global quiescence, update our view of such
    ///             outstanding requests, signaling waiters if we have achieved quiescence.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__UpdateQuiescenceWaiters(
        __in Graph * pGraph
        )
    {
        // we must hold the queue lock to perform this update, but do not require the accelerator lock
        // unless we are going to check the inflight set. Since we protect the flags and lists that
        // manage waiting for global or graph quiescence with the queue lock, in the common case, we
        // can elide the accelerator lock. 
        
        LockQuiescenceState();

        assert(m_bQuiescenceInProgress || ((m_vGraphsWaitingForQuiescence.size() == 0) &&
                                           !m_bWaitingForGlobalQuiescence));

        if(m_bQuiescenceInProgress) {

            LockTaskQueues();

            if(m_bWaitingForGlobalQuiescence) {

                // if we are waiting for global quiescece, we have to check the 
                // inflight list, so we must lock the accelerator maps. check the 
                // ready queue first, since it may allow us to avoid taking an
                // accelerator lock. 
            
                if(m_vReadyQ.size() == 0) {
                    LockAcceleratorMaps();
                    if(m_vInflightAccelerators.size() == 0) {
                        m_bInGlobalQuiescentState = TRUE;
                        m_bWaitingForGlobalQuiescence = FALSE;
                        __UpdateOKToScheduleView();
                        SetEvent(m_hGlobalQuiescentEvent);
                    }
                    UnlockAcceleratorMaps();
                }
            }

            std::map<Graph*, HANDLE>::iterator mi = m_vhQuiescentEvents.find(pGraph);
            if(mi != m_vhQuiescentEvents.end()) {

                assert(m_vGraphsWaitingForQuiescence.find(pGraph) != m_vGraphsWaitingForQuiescence.end());

                LockAcceleratorMaps();
                if(__IsGraphQuiescent(pGraph)) {     
                    assert(m_vLiveGraphs.find(pGraph) != m_vLiveGraphs.end());
                    assert(m_vRunningGraphs.find(pGraph) == m_vRunningGraphs.end());
                    assert(m_vGraphsWaitingForQuiescence.find(pGraph) != m_vGraphsWaitingForQuiescence.end());
                    assert(m_vQuiescedGraphs.find(pGraph) == m_vQuiescedGraphs.end());
                    m_vGraphsWaitingForQuiescence.erase(pGraph);
                    m_vQuiescedGraphs.insert(pGraph);
                    __UpdateOKToScheduleView();
                    SetEvent(mi->second);
                    m_vhQuiescentEvents.erase(mi);
                    __UpdateQuiescenceHint();
                }
                UnlockAcceleratorMaps();
            }

            UnlockTaskQueues();

        }
        UnlockQuiescenceState();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Compare tasks by effprio. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    struct compare_tasks_by_effprio {     
        bool operator() (const Task * lhs, const Task * rhs) { 
            return ((Task*)lhs)->GetEffectivePriority() > 
                ((Task*)rhs)->GetEffectivePriority(); 
        } 
    };

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sort run queue. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Scheduler::SortRunQueue(
        VOID
        )
    {
        // take a pass over the ptask queue, recompute
        // effective priority for each task, and then sort
        // the queue based on effective priority. 
        // call with queue lock held!
        assert(QueueLockIsHeld());
        MARKRANGEENTER(L"SortRunQueue");
        record_sort_q_entry();

        PTASKUSAGESTATS avg;
        PTASKUSAGESTATS stat;
        memset(&avg, 0, sizeof(avg));
        memset(&stat, 0, sizeof(stat));
        int nSamples = 0;
        deque<Task*>::iterator di;
        assert(m_vReadyQ.size());
        for(di=m_vReadyQ.begin(); di!=m_vReadyQ.end(); di++) {
            Task * pTask = (*di);
            // update current wait time before recompute stats.
            pTask->OnUpdateWait();
            pTask->GetUsageStats(&stat);
            avg.dAverageDispatchTime += stat.dAverageDispatchTime;
            avg.dAverageWaitTime += stat.dAverageWaitTime;
            avg.dLastDispatchTime += stat.dLastDispatchTime;
            avg.dLastWaitTime += stat.dLastWaitTime;
            avg.dAverageOSPrio += stat.dAverageOSPrio;
            nSamples++;
        }
        if(nSamples > 0) {
            avg.dAverageDispatchTime /= nSamples;
            avg.dAverageWaitTime /= nSamples;
            avg.dLastDispatchTime /= nSamples;
            avg.dLastWaitTime /= nSamples;
            avg.dAverageOSPrio /= nSamples;
            for(di=m_vReadyQ.begin(); di!=m_vReadyQ.end(); di++) {
                Task * pTask = (*di);
                pTask->ComputeEffectivePriority(avg.dAverageDispatchTime, 
                                                avg.dLastWaitTime, 
                                                avg.dAverageWaitTime,
                                                avg.dAverageOSPrio);
            }
            std::sort(m_vReadyQ.begin(), m_vReadyQ.end(), compare_tasks_by_effprio());
        }      
        record_sort_q_exit();
        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Removes pTask from the run queue and deferred queue.  </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::AbandonDispatches(
        Task * pTask
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler == NULL)
            return;
        return g_pScheduler->__AbandonDispatches(pTask);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Removes pTask from the run queue. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__AbandonDispatches(
        Task * pTask
        )
    {
        LockQuiescenceState();
        LockTaskQueues();   

        while(m_vReadyQ.size()) {
            deque<Task*>::iterator qi = find(m_vReadyQ.begin(), m_vReadyQ.end(), pTask);
            if(qi == m_vReadyQ.end()) 
                break;
            m_vReadyQ.erase(qi);
        }
        while(m_vDeferredQ.size()) {
            deque<Task*>::iterator qi = find(m_vDeferredQ.begin(), m_vDeferredQ.end(), pTask);
            if(qi == m_vDeferredQ.end()) 
                break;
            m_vDeferredQ.erase(qi);
        }
        __UpdateTaskQueueView();
        __UpdateQuiescenceWaiters(pTask->GetGraph());

        UnlockTaskQueues(); 
        UnlockQuiescenceState();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Moves pTask from the run queue to the deferred queue.  </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::DeferDispatches(
        Task * pTask
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler == NULL)
            return;
        return g_pScheduler->__DeferDispatches(pTask);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Moves pTask from the run queue to the deferred queue. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__DeferDispatches(
        Task * pTask
        )
    {
        // remove any dispatches for this task that are queued but not inflight. 
        LockQuiescenceState();
        LockTaskQueues();   

        while(m_vReadyQ.size()) {
            deque<Task*>::iterator qi = find(m_vReadyQ.begin(), m_vReadyQ.end(), pTask);
            if(qi == m_vReadyQ.end()) 
                break;
            m_vReadyQ.erase(qi);
            m_vDeferredQ.push_back(*qi);
        }
        __UpdateTaskQueueView();
        __UpdateQuiescenceWaiters(pTask->GetGraph());

        UnlockTaskQueues(); 
        UnlockQuiescenceState();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Updates the tasks ready view. </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__UpdateTaskQueueView(
        VOID
        )
    {
        assert(QuiescenceLockIsHeld());
        assert(QueueLockIsHeld());
        m_bTasksAvailable = m_vReadyQ.size() > 0;
        if(m_bTasksAvailable) {
            SetEvent(m_hTaskQueue);
        }
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Removes all tasks from the given graph from the run queue. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::AbandonDispatches(
        Graph * pGraph
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler == NULL)
            return;
        return g_pScheduler->__AbandonDispatches(pGraph);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Removes all tasks from the given graph from the run queue. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__AbandonDispatches(
        Graph * pGraph
        )
    {
        LockQuiescenceState();
        LockTaskQueues();        

        if(m_vReadyQ.size() || m_vDeferredQ.size()) {
            std::deque<Task*> newReadyQ;
            std::deque<Task*> newDeferredQ;
            deque<Task*>::iterator qi;
            for(qi=m_vReadyQ.begin(); qi!=m_vReadyQ.end(); qi++) {
                if((*qi)->GetGraph() != pGraph) 
                    newReadyQ.push_back(*qi);
            }
            for(qi=m_vDeferredQ.begin(); qi!=m_vDeferredQ.end(); qi++) {
                if((*qi)->GetGraph() == pGraph) {
                    newDeferredQ.push_back(*qi);
                }
            }
            m_vReadyQ.clear();
            m_vDeferredQ.clear();
            m_vReadyQ.assign(newReadyQ.begin(), newReadyQ.end());
            m_vDeferredQ.assign(newDeferredQ.begin(), newDeferredQ.end());
            __UpdateTaskQueueView();
            __UpdateQuiescenceWaiters(pGraph);
        }

        UnlockTaskQueues(); 
        UnlockQuiescenceState();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Moves all tasks from the given graph from the run queue to the
    ///             deferredQ. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///
    /// <param name="pTask">    [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::DeferDispatches(
        Graph * pGraph
        )
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler == NULL)
            return;
        return g_pScheduler->__DeferDispatches(pGraph);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Moves all tasks from the given graph from the run queue to the deferred queue.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///
    /// <param name="pGraph">   [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__DeferDispatches(
        Graph * pGraph
        )
    {
        LockQuiescenceState();
        LockTaskQueues();        

        deque<Task*>::iterator qi;
        std::deque<Task*> newReadyQ;
        for(qi=m_vReadyQ.begin(); qi!=m_vReadyQ.end(); qi++) {
            if((*qi)->GetGraph() == pGraph) 
                m_vDeferredQ.push_back(*qi);
            else
                newReadyQ.push_back(*qi);
        }            
        m_vReadyQ.clear();
        m_vReadyQ.assign(newReadyQ.begin(), newReadyQ.end());
        __UpdateQuiescenceWaiters(pGraph);
        __UpdateTaskQueueView();

        UnlockTaskQueues(); 
        UnlockQuiescenceState();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets live graph count. </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <returns>   The live graph count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Scheduler::GetLiveGraphCount(
        VOID
        )
    {
        if(g_pScheduler != NULL)
            return g_pScheduler->__GetLiveGraphCount();
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the running graph count. Locks are released before return, so
    /// 			the result is not guaranteed fresh--use only to inform heuristics. 
    /// 			</summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <returns>   The running graph count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Scheduler::GetRunningGraphCount(
        VOID
        )
    {
		UINT nGraphs = 0;
        if(g_pScheduler != NULL) {
			g_pScheduler->LockQuiescenceState();
            nGraphs = g_pScheduler->__GetRunningGraphCount();
			g_pScheduler->UnlockQuiescenceState();
		}
        return nGraphs;
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
    Scheduler::PauseAndReportGraphStates(
        VOID
        )
    {
        if(g_pScheduler != NULL) {
            g_pScheduler->__LockScheduler();
            g_pScheduler->__PauseAndReportGraphStates();
            g_pScheduler->__UnlockScheduler();
        }
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
    Scheduler::__PauseAndReportGraphStates(
        VOID
        )
    {
        assert(QueueLockIsHeld());
        assert(AcceleratorLockIsHeld());
        std::set<Graph*>::iterator gi;
        for(gi=m_vLiveGraphs.begin(); gi!=m_vLiveGraphs.end(); gi++) {
            (*gi)->PauseAndReportGraphState();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets live graph count. </summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <returns>   The live graph count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Scheduler::__GetLiveGraphCount(
        VOID
        )
    {
        //assert(QuiescenceLockIsHeld());
        //assert(QueueLockIsHeld());
        return (UINT) m_vLiveGraphs.size();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the running graph count. Locks are released before return, so
    /// 			the result is not guaranteed fresh--use only to inform heuristics. 
    /// 			</summary>
    ///
    /// <remarks>   crossbac, 6/24/2013. </remarks>
    ///
    /// <returns>   The running graph count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Scheduler::__GetRunningGraphCount(
        VOID
        )
    {
        assert(QuiescenceLockIsHeld());
        return (UINT) m_vRunningGraphs.size();
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Determines if we can required accelerators *may be* available and 
    ///             it is worth trying to do more work to attempt an assignment. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/14/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Scheduler::RequiredAcceleratorsAvailable(
        VOID
        ) 
    {
        assert(AcceleratorLockIsHeld());
        std::set<ACCELERATOR_CLASS>::iterator si;
        for(si=m_vLiveAcceleratorClasses.begin(); si!=m_vLiveAcceleratorClasses.end(); si++) {
            if(m_vAvailableAccelerators[*si]->size())
                return TRUE;
        }
        return FALSE;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Schedules. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::Schedule(
        VOID
        ) 
    {
        record_schedule_entry();
        MARKRANGEENTER(L"Schedule");

        // if the queue is non-empty and there are available accelerators, select the highest priority
        // ptask available and pair it with the most capable accelerator that fits its needs. Do this
        // by removing the accelerator from the available queue, and setting the tasks dispatch
        // accelerator member to the accelerator we have chosen, subsequently setting the dispatch
        // signal for that ptask. The execute thread for that ptask will wake up, find that it has an
        // accelerator and dispatch using it. The ptask is responsible for returning the accelerator to
        // the available queue by calling CompleteDispatch().
        // --------------------------------------------------
        // Regarding dependent accelerators: some tasks use accelerators other than those assigned as
        // the dispatch accelerator through APIs that are not visible to PTask. In this case
        // correctness when the runtime is in multi-threaded mode depends on the programmer making
        // those dependences visible so that the scheduler does not concurrently assign those devices
        // to other tasks. If the MatchAccelerator call returns a non-empty list of dependent
        // accelerators, those must also be removed from the available list. 
        
        Task * pTask = NULL;
        Accelerator* pAccelerator = NULL;
        int nDependentAccelerators = 0;
        Accelerator** ppDependentAccelerators = NULL;
        
        LockQuiescenceState();
        LockTaskQueues();

        if(m_vReadyQ.size()) {

            LockAcceleratorMaps();

            if(!PTask::Runtime::GetScheduleChecksClassAvailability() || RequiredAcceleratorsAvailable()) {

                if (PTask::Runtime::GetSchedulingMode() != SCHEDMODE_FIFO)
                    SortRunQueue();

                MARKRANGEENTER(L"Attempt-assign");
                deque<Task*>::iterator qi = m_vReadyQ.begin();
                do {
                    assert(qi != m_vReadyQ.end());

                    // start from the front of the list and look for a valid pairing. If MatchAccelerators returns
                    // NULL, it means that the task is constrained to use accelerator(s) that are not currently
                    // available. In that case move on to the next task in the list. 
                    
                    MARKRANGEENTER(L"Match-disp");
                    pTask = *qi;
                    nDependentAccelerators = 0;
                    pAccelerator = MatchAccelerators(pTask, &nDependentAccelerators, &ppDependentAccelerators);
                    MARKRANGEEXIT();

                    if(pAccelerator != NULL) {

                        // if we have actually assigned an accelerator, mark it as inflight, 
                        // and remove it from the available accelerator list until the dispatch completes.                    
                        MARKRANGEENTER(L"Match-deps+assign");
                        __ClaimAccelerator(pAccelerator, pTask);
                        UpdateDispatchStatistics(pAccelerator, pTask, nDependentAccelerators, ppDependentAccelerators);

                        if(nDependentAccelerators != 0) {

                            // if this task requires resources in addition to the dispatch accelerator,
                            // mark them inflight and remove them from the available list as well.
                            pTask->Lock();
                            nDependentDispatchTotal++;
                            for(int i=0; i<nDependentAccelerators; i++) {
                                assert(ppDependentAccelerators[i] != NULL);
                                if(ppDependentAccelerators[i] != NULL) {                            
                                    __ClaimAccelerator(ppDependentAccelerators[i], pTask);
                                    pTask->AssignDependentAccelerator(i, ppDependentAccelerators[i]);
                                    UpdateDependentDispatchStatistics(ppDependentAccelerators[i], pTask);
                                }
                            }
                            pTask->Unlock();
                            free(ppDependentAccelerators);
                        }

                        // remove this task from the run queue to indicate
                        // that it has been dequeued and is ready to dispatch.
                    
                        m_vReadyQ.erase(qi);
                        MARKRANGEEXIT();

                    } else {

                        // couldn't find an assignment for this task,
                        // try the next one in the queue.
                        pTask = NULL;
                        qi++;
                    }

                } while((pAccelerator == NULL) && (qi != m_vReadyQ.end()));
                MARKRANGEEXIT();
            }
            UnlockAcceleratorMaps();
        }
        __UpdateTaskQueueView();
        if(pTask && pAccelerator) {
            pTask->Lock();
            pTask->SetDispatchAccelerator(pAccelerator);
            pTask->SignalDispatch();
            pTask->Unlock();
        }
        UnlockTaskQueues(); 
        UnlockQuiescenceState();
        MARKRANGEEXIT();
        record_schedule_exit();
        return (pTask && pAccelerator);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Schedules. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Scheduler::AssignDispatchAccelerator(
        __inout Task * pTask,
        __in    BOOL bBypassQ
        ) 
    {
        record_schedule_entry();
        MARKRANGEENTER(L"Schedule(Task)");

        BOOL bOkToSchedule = FALSE;
        Accelerator* pAccelerator = NULL;
        int nDependentAccelerators = 0;
        Accelerator** ppDependentAccelerators = NULL;        
        LockQuiescenceState();
        LockTaskQueues();
        if(m_vReadyQ.size() == 0 || bBypassQ) {
            assert(std::find(m_vReadyQ.begin(), m_vReadyQ.end(), pTask) == m_vReadyQ.end());
            bOkToSchedule = TRUE;
        }
        UnlockTaskQueues(); 
        UnlockQuiescenceState();

        if(bOkToSchedule) {
            LockAcceleratorMaps();
            MARKRANGEENTER(L"Attempt-assign");
            nDependentAccelerators = 0;
            pAccelerator = MatchAccelerators(pTask, &nDependentAccelerators, &ppDependentAccelerators);
            if(pAccelerator != NULL) {
                MARKRANGEENTER(L"Match-deps+assign");
                __ClaimAccelerator(pAccelerator, pTask);
                UpdateDispatchStatistics(pAccelerator, pTask, nDependentAccelerators, ppDependentAccelerators);
                if(nDependentAccelerators != 0) {
                    pTask->Lock();
                    nDependentDispatchTotal++;
                    for(int i=0; i<nDependentAccelerators; i++) {
                        assert(ppDependentAccelerators[i] != NULL);
                        if(ppDependentAccelerators[i] != NULL) {                            
                            __ClaimAccelerator(ppDependentAccelerators[i], pTask);
                            pTask->AssignDependentAccelerator(i, ppDependentAccelerators[i]);
                            UpdateDependentDispatchStatistics(ppDependentAccelerators[i], pTask);
                        }
                    }
                    pTask->Unlock();
                    free(ppDependentAccelerators);
                }
                MARKRANGEEXIT();
            } 
            MARKRANGEEXIT();
            UnlockAcceleratorMaps();
        }
        if(pTask && pAccelerator) {
            pTask->Lock();
            pTask->SetDispatchAccelerator(pAccelerator);
            pTask->Unlock();
        }
        MARKRANGEEXIT();
        record_schedule_exit();
        return (pAccelerator != NULL);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Schedule thread proc. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD 
    Scheduler::ScheduleThreadProc(
        VOID
        ) 
    {
        NAMETHREAD(L"SchedulerThread");
        HANDLE vCanScheduleHandles[] = { 
            m_hRuntimeTerminateEvent,
            m_hSchedulerTerminateEvent,
            m_hOKToSchedule 
        };
        HANDLE vWaitHandles[] = {
            m_hRuntimeTerminateEvent,
            m_hSchedulerTerminateEvent,
            m_hTaskQueue,
            m_hAcceleratorQueue
        };
        DWORD dwCanScheduleHandles = sizeof(vCanScheduleHandles) / sizeof(HANDLE);
        DWORD dwWaitHandles = sizeof(vWaitHandles) / sizeof(HANDLE);
        BOOL bShutdown = FALSE;

        UINT uiThreadId = InterlockedIncrement(&uiSchedThreadIdx)-1;
        assert((uiThreadId < m_uiThreadCount) || (uiSchedInitCallsPerAS > 1)); 
        Accelerator::InitializeTLSContextManagement(PTTR_SCHEDULER, uiThreadId, m_uiThreadCount, FALSE);

        while(m_bAlive && !bShutdown) {

            // printf("entering first wait...\n");
            MARKRANGEENTER(L"Scheduler-wait-runnable");
            DWORD dwRunnableWait = WaitForMultipleObjects(dwCanScheduleHandles, vCanScheduleHandles, FALSE, INFINITE);
            switch(dwRunnableWait) {
            case WAIT_ABANDONED:     bShutdown = TRUE; continue;
            case WAIT_FAILED:        bShutdown = TRUE; continue;
            case WAIT_TIMEOUT:       bShutdown = TRUE; continue;
            case WAIT_OBJECT_0 + 0:  bShutdown = TRUE; continue;
            case WAIT_OBJECT_0 + 1:  bShutdown = TRUE; continue;
            case WAIT_OBJECT_0 + 2:  break; // we can schedule!
            }
            MARKRANGEEXIT();
                
            if(!Schedule()) {

                MARKRANGEENTER(L"Scheduler-wait-resources");
                DWORD dwScheduleWait = WaitForMultipleObjects(dwWaitHandles, vWaitHandles, FALSE, INFINITE);
                switch(dwScheduleWait) {
                case WAIT_ABANDONED:    bShutdown = TRUE; continue;
                case WAIT_FAILED:       bShutdown = TRUE; continue;
                case WAIT_TIMEOUT:      bShutdown = TRUE; continue;
                case WAIT_OBJECT_0 + 0: bShutdown = TRUE; continue;
                case WAIT_OBJECT_0 + 1: bShutdown = TRUE; continue;
                case WAIT_OBJECT_0 + 2: break;
                case WAIT_OBJECT_0 + 3: break;
                }
                MARKRANGEEXIT();
            }
        }

        Accelerator::DeinitializeTLSContextManagement();
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Ptask scheduler thread. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="p">    The p. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD WINAPI 
    ptask_scheduler_thread(
        LPVOID p
        ) 
    {
        Scheduler * pScheduler = (Scheduler*) p;
        return pScheduler->ScheduleThreadProc();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Locks the queue. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::LockTaskQueues(
        VOID
        )
    {
        MARKRANGEENTER(L"LockTaskQueues");
        EnterCriticalSection(&m_csQ);
        m_nQueueLockDepth++;
        MARKRANGEEXIT();
        if(m_nQueueLockDepth == 1) {
            MARKRANGEENTER(L"QueueLockHeld");
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unlocks the queue. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::UnlockTaskQueues(
        VOID
        )
    {
        assert(QueueLockIsHeld());
        m_nQueueLockDepth--;
        assert(m_nQueueLockDepth >= 0);
        if(m_nQueueLockDepth == 0) {
            MARKRANGEEXIT(); // queue lock held
        }
        LeaveCriticalSection(&m_csQ);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Locks the queue. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::LockQuiescenceState(
        VOID
        )
    {
        MARKRANGEENTER(L"LockQuiescenceState");
        EnterCriticalSection(&m_csQuiescentState);
        assert(m_nQuiescenceLockDepth >= 0);
        m_nQuiescenceLockDepth++;
        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unlocks the queue. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::UnlockQuiescenceState(
        VOID
        )
    {
        assert(m_nQuiescenceLockDepth > 0);
        m_nQuiescenceLockDepth--;
        LeaveCriticalSection(&m_csQuiescentState);
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Locks the accelerators. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::LockAcceleratorMaps(
        VOID
        )
    {
        MARKRANGEENTER(L"Lock-Accelerators");
        EnterCriticalSection(&m_csAccelerators);
        m_nAcceleratorMapLockDepth++;
        MARKRANGEEXIT();
        if(m_nAcceleratorMapLockDepth == 1) {
            MARKRANGEENTER(L"Acc-Lock-Held");
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unlocks the accelerators. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::UnlockAcceleratorMaps(
        VOID
        )
    {
        assert(AcceleratorLockIsHeld());
        m_nAcceleratorMapLockDepth--;
        assert(m_nAcceleratorMapLockDepth >= 0);
        if(m_nAcceleratorMapLockDepth == 0) {
            MARKRANGEEXIT();
        }
        LeaveCriticalSection(&m_csAccelerators);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Updates the dispatch statistics. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pDispatchAccelerator">     [in] non-null, the accelerator. </param>
    /// <param name="pTask">                    [in] non-null, the task. </param>
    /// <param name="nDependentAccelerators">   The dependent accelerators. </param>
    /// <param name="ppDependentAccelerators">  [in,out] If non-null, the dependent accelerators. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Scheduler::UpdateDispatchStatistics(
        __in Accelerator * pDispatchAccelerator, 
        __in Task * pTask,
        __in int nDependentAccelerators,
        __in Accelerator** ppDependentAccelerators
        )
    {
        // call with accelerator/dipatch lock held. 
        EnterCriticalSection(&m_csStatistics);
        if(m_vAcceleratorDispatches.find(pDispatchAccelerator) == m_vAcceleratorDispatches.end())
            m_vAcceleratorDispatches[pDispatchAccelerator] = 0;
        std::string strTaskName(pTask->GetTaskName());
        Graph * pGraph = pTask->GetGraph();
        std::string strGraphName(pGraph->m_lpszGraphName);
        std::map<std::string, std::map<Accelerator*, int>*>* pGraphMap;
        std::map<std::string, std::map<std::string, std::map<Accelerator*, int>*>*>::iterator tdi = m_vDispatches.find(strGraphName);        
        if(tdi == m_vDispatches.end()) {
            pGraphMap = new std::map<std::string, std::map<Accelerator*, int>*>();
            m_vDispatches[strGraphName] = pGraphMap;
        } else {
            pGraphMap = tdi->second;
        }
        if(pGraphMap->find(strTaskName) == pGraphMap->end()) {
            (*pGraphMap)[strTaskName] = new std::map<Accelerator*, int>();
        }
        std::map<Accelerator*, int>* pvTaskDispatchMap = (*pGraphMap)[strTaskName];
        assert(pvTaskDispatchMap != NULL);
        if(pvTaskDispatchMap->find(pDispatchAccelerator) == pvTaskDispatchMap->end()) {
            (*pvTaskDispatchMap)[pDispatchAccelerator] = 0;
        }
        (*pvTaskDispatchMap)[pDispatchAccelerator] = (*pvTaskDispatchMap)[pDispatchAccelerator] + 1;

        if(nDependentAccelerators) {
            Accelerator* pDependentAccelerator = ppDependentAccelerators[0];
            assert(pTask->HasDependentAcceleratorBindings());
            tdi = m_vDependentDispatches.find(strGraphName);
            if(tdi==m_vDependentDispatches.end()) {
                pGraphMap = new std::map<std::string, std::map<Accelerator*, int>*>();
                m_vDependentDispatches[strGraphName] = pGraphMap;
            } else {
                pGraphMap = tdi->second;
            }
            if(pGraphMap->find(strTaskName) == pGraphMap->end()) {
                (*pGraphMap)[strTaskName] = new std::map<Accelerator*, int>();
            }
            std::map<Accelerator*, int>* pvTaskDepDispatchMap = (*pGraphMap)[strTaskName];
            assert(pvTaskDepDispatchMap != NULL);
            if(pvTaskDepDispatchMap->find(pDependentAccelerator) == pvTaskDepDispatchMap->end()) {
                (*pvTaskDepDispatchMap)[pDependentAccelerator] = 0;
            }
            (*pvTaskDepDispatchMap)[pDependentAccelerator] = (*pvTaskDepDispatchMap)[pDependentAccelerator] + 1;
        }

        m_vAcceleratorDispatches[pDispatchAccelerator] = m_vAcceleratorDispatches[pDispatchAccelerator] + 1;
        nDispatchTotal++;
        LeaveCriticalSection(&m_csStatistics);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Updates dispatch statistics for accelerators scheduled as dependent accelerators
    ///             (not the dispatch accelerator, and used by a task through APIs other than PTask:
    ///             e.g. HostTasks that use CUDA).
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
    /// <param name="pTask">        [in] non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::UpdateDependentDispatchStatistics(
        Accelerator * pAccelerator, 
        Task * pTask
        )
    {
        // call with accelerator/dispatch lock held. 
        EnterCriticalSection(&m_csStatistics);
        if(m_vDependentAcceleratorDispatches.find(pAccelerator) == m_vDependentAcceleratorDispatches.end())
            m_vDependentAcceleratorDispatches[pAccelerator] = 0;
        std::string strTaskName(pTask->GetTaskName());
        m_vDependentAcceleratorDispatches[pAccelerator] = m_vDependentAcceleratorDispatches[pAccelerator] + 1;
        nDependentAcceleratorDispatchTotal++;
        LeaveCriticalSection(&m_csStatistics);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps a dispatch statistics. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void    
    Scheduler::DumpDispatchStatistics(
        VOID
        )
    {
        if (!PTask::Runtime::IsVerbose() && !PTask::Runtime::GetTaskProfileMode()) 
            return;

        const int nFillWidth = 50;
        map<Accelerator*, int>::iterator ai;
        map<std::string, std::map<Accelerator*, int>*>::iterator ti;
        map<std::string, std::map<Accelerator*, int>*>::iterator dti;
        std::map<std::string, std::map<std::string, std::map<Accelerator*, int>*>*>::iterator tdi;
        std::map<std::string, std::map<std::string, std::map<Accelerator*, int>*>*>::iterator tddi;
        std::ios_base::fmtflags original_flags = std::cerr.flags();

        std::cerr << endl << "DISPATCH STATISTICS:" << endl;
        std::cerr << setfill('-') << setw(nFillWidth) << "-" << std::endl;
        std::cerr.setf(std::ios_base::left, std::ios_base::adjustfield);
        std::cerr.width(nFillWidth);
        std::cerr.fill(' ');
        std::cerr 
            << "TOTAL DISPATCHES"
            << nDispatchTotal 
            << std::endl;

        EnterCriticalSection(&m_csStatistics);
        for(ai=m_vAcceleratorDispatches.begin();
            ai!=m_vAcceleratorDispatches.end();
            ai++) {
            std::cerr.width(nFillWidth);
            std::cerr.fill(' ');
            std::cerr 
                << ai->first->GetDeviceName() 
                << ai->second 
                << std::endl;
        }

        int nGraphs = 0;
        for(tdi=m_vDispatches.begin(); tdi!=m_vDispatches.end(); tdi++) {
            map<string, int> vTaskDispatchTotals;
            map<string, string> vTaskDispatchDetails;
            std::cerr 
                << tdi->first
                << std::endl;
            std::map<std::string, std::map<Accelerator*, int>*>* pGraphMap = tdi->second;
            std::map<std::string, std::map<Accelerator*, int>*>* pGraphDepDMap = NULL;
            tddi = m_vDependentDispatches.find(tdi->first);
            if(tddi != m_vDependentDispatches.end()) 
                pGraphDepDMap = tddi->second;

            nGraphs++;
            for(ti=pGraphMap->begin();
                ti!=pGraphMap->end();
                ti++) {
                string strTask = ti->first;
                map<Accelerator*, int>* pTaskMap = ti->second;
                map<Accelerator*, int>::iterator ei;
                int nTaskTotal = 0;
                stringstream ssPerAcceleratorCount;
                ssPerAcceleratorCount << "\t(";
                bool bFirst = true;
                for(ei=pTaskMap->begin(); ei!=pTaskMap->end(); ei++) {
                    Accelerator * pAccelerator = ei->first;
                    int nAccCount = ei->second;
                    nTaskTotal += nAccCount;
                    if(!bFirst) {
                        ssPerAcceleratorCount << ", ";
                    }
                    bFirst = false;
                    ssPerAcceleratorCount 
                        << "A" << pAccelerator->GetAcceleratorId() 
                        << ":" << nTaskTotal;
                }
                map<Accelerator*, int>* pDepMap = NULL;
                if(pGraphDepDMap) {
                    dti = pGraphDepDMap->find(strTask);
                    if(dti != pGraphDepDMap->end()) {
                        pDepMap = dti->second;
                        for(ei=pDepMap->begin(); ei!=pDepMap->end(); ei++) {
                            Accelerator * pAccelerator = ei->first;
                            int nAccCount = ei->second;
                            nTaskTotal += nAccCount;
                            if(!bFirst) {
                                ssPerAcceleratorCount << ", ";
                            }
                            bFirst = false;
                            ssPerAcceleratorCount 
                                << "DepA" << pAccelerator->GetAcceleratorId() 
                                << ":" << nTaskTotal;
                        }
                    }
                }
                ssPerAcceleratorCount << ")";
                vTaskDispatchTotals[strTask] = nTaskTotal;
                vTaskDispatchDetails[strTask] = ssPerAcceleratorCount.str();
            }
            for(ti=pGraphMap->begin();
                ti!=pGraphMap->end();
                ti++) {
                std::cerr.width(nFillWidth);
                std::cerr.fill(' ');
                int nTotal = vTaskDispatchTotals[ti->first];
                string strDetails = vTaskDispatchDetails[ti->first];
                std::cerr 
                    << ti->first 
                    << nTotal
                    << strDetails
                    << std::endl;
            }
        }

        std::cerr.width(nFillWidth);
        std::cerr.fill(' ');
        std::cerr 
            << "total dependent dispatches" 
            << nDependentDispatchTotal 
            << std::endl;

        std::cerr.width(nFillWidth);
        std::cerr.fill(' ');
        std::cerr 
            << "total accelerators assigned in dispatches" 
            << nDependentAcceleratorDispatchTotal 
            << std::endl;

        for(ai=m_vDependentAcceleratorDispatches.begin();
            ai!=m_vDependentAcceleratorDispatches.end();
            ai++) {
            std::cerr.width(nFillWidth);
            std::cerr.fill(' ');
            std::cerr 
                << ai->first->GetDeviceName() 
                << ai->second 
                << std::endl;
        }
        std::cerr.flags(original_flags);
        LeaveCriticalSection(&m_csStatistics);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the global scheduling mode. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="mode"> The mode. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::SetSchedulingMode(
        SCHEDULINGMODE mode
        ) 
    {
        if(g_pScheduler != NULL) {
            g_pScheduler->__SetSchedulingMode(mode);
        } else {
            g_ePendingMode = mode;
            g_bPendingScheduleModeChange = TRUE;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the global scheduling mode. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   The scheduling mode. </returns>
    ///-------------------------------------------------------------------------------------------------

    SCHEDULINGMODE 
    Scheduler::GetSchedulingMode(
        VOID
        ) 
    {
        if(g_pScheduler != NULL) {
            return g_pScheduler->__GetSchedulingMode();
        }
        return SCHEDMODE_DATADRIVEN;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the global scheduling mode. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="mode"> The mode. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__SetSchedulingMode(
        SCHEDULINGMODE mode
        ) 
    { 
        m_eSchedulingMode = mode; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the global scheduling mode. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   The scheduling mode. </returns>
    ///-------------------------------------------------------------------------------------------------

    SCHEDULINGMODE 
    Scheduler::__GetSchedulingMode(
        VOID
        ) 
    { 
        return m_eSchedulingMode; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Find the accelerator with the given identifier. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="uiAcceleratorId">  Identifier for the accelerator. </param>
    ///
    /// <returns>   null if no such accelerator is found, 
    /// 			else the accelerator object with the specified id. </returns>
    ///-------------------------------------------------------------------------------------------------    

    Accelerator * 
    Scheduler::GetAcceleratorById(
        UINT uiAcceleratorId
        ) 
    {
        assert(g_pScheduler != NULL);
        if(g_pScheduler != NULL) {
            return g_pScheduler->__GetAcceleratorById(uiAcceleratorId);
        }
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Find the accelerator with the given identifier. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="uiAcceleratorId">  Identifier for the accelerator. </param>
    ///
    /// <returns>   null if no such accelerator is found, 
    /// 			else the accelerator object with the specified id. </returns>
    ///-------------------------------------------------------------------------------------------------    

    Accelerator * 
    Scheduler::__GetAcceleratorById(
        UINT uiAcceleratorId
        ) 
    {
        Accelerator * pAccelerator = NULL;
        LockAcceleratorMaps();
        std::map<UINT, Accelerator*>::iterator ai;
        ai = m_vAcceleratorMap.find(uiAcceleratorId);
        if(ai != m_vAcceleratorMap.end()) {
            pAccelerator = ai->second;
        }
        UnlockAcceleratorMaps();
        return pAccelerator;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the last dispatch timestamp. </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///
    /// <returns>   The last dispatch timestamp. </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD 
    Scheduler::GetLastDispatchTimestamp(
        VOID
        )
    {
        return g_pScheduler->__GetLastDispatchTimestamp();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Updates the last dispatch timestamp. </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///
    /// <returns>   The last dispatch timestamp. </returns>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::UpdateLastDispatchTimestamp(
        VOID
        )
    {
        g_pScheduler->__UpdateLastDispatchTimestamp();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the last dispatch timestamp. </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///
    /// <returns>   The last dispatch timestamp. </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD 
    Scheduler::__GetLastDispatchTimestamp(
        VOID
        )
    {
        // strict consistency is not required here, since this
        // timestamp is only used by a watchdog thread to detect
        // lack of forward progress. We use a lock to ensure we
        // get a partial view of the timestamp itself, but the
        // consumer gets no guarantees about the freshness of this
        // result, obviously.
        DWORD dwValue;
        EnterCriticalSection(&m_csDispatchTimestamp);
        dwValue = m_dwLastDispatchTimestamp;
        LeaveCriticalSection(&m_csDispatchTimestamp);
        return dwValue;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Updates the last dispatch timestamp. </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///
    /// <returns>   The last dispatch timestamp. </returns>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__UpdateLastDispatchTimestamp(
        VOID
        )
    {
        EnterCriticalSection(&m_csDispatchTimestamp);
        m_dwLastDispatchTimestamp = ::GetTickCount();
        LeaveCriticalSection(&m_csDispatchTimestamp);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check dispatch constraints. Debug utility--checks whether dispatch
    ///             and dependent accelerator assignments conform to the affinity 
    ///             the user has specified for the given task. </summary>
    ///
    /// <remarks>   Crossbac, 11/15/2013. </remarks>
    ///
    /// <param name="pTask">                    [in,out] If non-null, the task. </param>
    /// <param name="pDispatchAssignment">      [in,out] If non-null, the dispatch assignment. </param>
    /// <param name="pnDependentAccelerators">  [in,out] If non-null, the pn dependent accelerators. </param>
    /// <param name="pppDependentAccelerators"> [in,out] If non-null, the PPP dependent accelerators. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Scheduler::__CheckDispatchConstraints(
        __in Task * pTask,
        __in Accelerator * pDispatchAssignment,
        __in int * pnDependentAccelerators,
        __in Accelerator *** pppDependentAccelerators
        )
    {
        // a NULL dispatch assignment means the dispatch
        // attempt failed. which means that trivially, no
        // affinity constraints are violated by the assignment
        if(pDispatchAssignment == NULL) 
            return; 

        Accelerator * pMandatory = pTask->GetMandatoryAccelerator();
        BOOL bMandatoryViolation = (pMandatory != NULL) && 
                                   (pMandatory != pDispatchAssignment);

        if(!bMandatoryViolation && pTask->HasMandatoryDependentAffinities()) {

            // no violations were found for the dispatch accelerator
            // assignment, but dependent accelerators must be checked as well.

            pTask->Lock();
            Accelerator * pAssignment = pTask->GetAssignedDependentAccelerator(0);
            Accelerator * pRequirement = pTask->GetMandatoryDependentAccelerator();
            pTask->Unlock();

            if((pAssignment != NULL && pRequirement != NULL) && (pAssignment != pRequirement)) {
                bMandatoryViolation = TRUE;                        
            } else {
                bMandatoryViolation = (*pnDependentAccelerators != 1 || (*pppDependentAccelerators)[0] != pRequirement);
            }
        }

        if(bMandatoryViolation) {
            PTask::Runtime::HandleError("%s:%s:%d constraints check failed for %s->%s!\n",
                                        __FILE__,
                                        __FUNCTION__,
                                        __LINE__,
                                        pTask->GetTaskName(),
                                        pDispatchAssignment->GetDeviceName());                                        
        }

    }

};
