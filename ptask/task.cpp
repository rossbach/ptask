//--------------------------------------------------------------------------------------
// File: Task.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "Task.h"
#include "taskprofiler.h"
#include "dispatchcounter.h"
#include "graph.h"
#include "hrperft.h"
#include "shrperft.h"
#include "InputPort.h"
#include "OutputPort.h"
#include "MetaPort.h"
#include "Scheduler.h"
#include "MemorySpace.h"
#include <map>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <assert.h>
#include "Tracer.h"
#include "nvtxmacros.h"
#include "ptasklynx.h"
#include "ptprofsupport.h"
#include "extremetrace.h"
#include "instrumenter.h"
#include "CompiledKernel.h"
#include "DatablockProfiler.h"
#include "ptgc.h"
#include "signalprofiler.h"
#include "multichannel.h"
using namespace std;
using namespace PTask::Runtime;

#ifdef DISPATCH_COUNT_DIAGNOSTICS        
#define dispcnt_init()               DispatchCounter::InitializeDispatchCounting()
#define dispcnt_report()             DispatchCounter::DumpDispatchCounts()
#define dispcnt_verify(x)            DispatchCounter::Verify(x)
#define dispcnt_record_dispatch(p)   RecordDispatch()
#define dispcnt_check_port_control() CheckPortControlCodes()
#define dispcnt_check_block_pools()  CheckBlockPoolStates()
#define dispcnt_set_expected(c)      SetExpectedDispatchCount(c)
#define dispcnt_reset()              m_pDispatchCounter->Reset()
#define dispcnt_get_dispnum()        (m_nActualDispatches+1)
#else
#define dispcnt_init()               
#define dispcnt_report()             
#define dispcnt_verify(x)            (0)
#define dispcnt_record_dispatch(p)   
#define dispcnt_check_port_control() 
#define dispcnt_check_block_pools()  
#define dispcnt_set_expected(c)      
#define dispcnt_reset()             
#define dispcnt_get_dispnum()        (m_pTask->m_nDispatchNumber) 
#endif

#ifdef CHECK_CRITICAL_PATH_ALLOC
#define critical_path_check_binding_trace(x, y, z)  
// #define critical_path_check_binding_trace(x, y, z)  PTask::Runtime::MandatoryInform((x),(y),(z))
#else
#define critical_path_check_binding_trace(x, y, z)  
#endif

// #define COLLECT_AGG_DATA
#if defined(ADHOC_STATS) && defined(COLLECT_AGG_DATA)
#define cond_record_stream_agg_entry() {                                               \
    if(!strncmp((m_lpszTaskName),"accumulator_",strlen("accumulator_"))) {             \
        record_stream_agg_entry(m_lpszTaskName);                                       \
    }                                                                                  \
    if(!strncmp((m_lpszTaskName),"CountUniqueKeys_",strlen("CountUniqueKeys")) &&      \
        strncmp((m_lpszTaskName),"CountUniqueKeys_24",strlen("CountUniqueKeys_24"))) { \
        record_stream_agg_exit(m_lpszTaskName);                                        \
    }}             
#define cond_record_stream_agg_exit() {                                                \
    if(!strncmp((m_lpszTaskName),"Select_12",strlen("Select_12"))) {                   \
        record_stream_agg_exit(m_lpszTaskName);                                        \
        printf("left agg phase\n");                                                    \
    }}                                                                                  
                                                                                       
#else
#define cond_record_stream_agg_entry()
#define cond_record_stream_agg_exit()
#endif

#ifdef DEBUG
#define COMPLAIN(x) PTask::Runtime::MandatoryInform(x)
#define GET_OSTENSIBLE_DEPENDENCES(deps, buf)  { (deps) = &m_vIncomingDepBuffers[buf]; }
#define CLEAR_OSTENSIBLE_DEPENDENCES() { m_vIncomingDepBuffers.clear(); m_vIncomingDuplicateDeps.clear(); }
#define FIND_DUPLICATE_DEPENDENCES() {                                                    \
    std::set<AsyncDependence*> superset;                                                  \
        std::map<PBuffer*, std::set<AsyncDependence*>>::iterator dsi;                     \
        for(dsi=m_vIncomingDepBuffers.begin(); dsi!=m_vIncomingDepBuffers.end(); dsi++) { \
            std::set<AsyncDependence*>::iterator asi;                                     \
            for(asi=dsi->second.begin(); asi!=dsi->second.end(); asi++) {                 \
                if(superset.find(*asi) != superset.end()) {                               \
                    m_vIncomingDuplicateDeps[*asi] = TRUE;                                \
                } else {                                                                  \
                    superset.insert(*asi);                                                \
                } } } }

#define VALIDATE_EARLY_RESOLUTION(pDependences, pBuffer, pbAlreadyResolved)   {                             \
    if(pbAlreadyResolved && (*pbAlreadyResolved) && pDependences) {                                         \
        std::set<AsyncDependence*>::iterator adi=pDependences->begin();                                     \
        if(m_vIncomingDuplicateDeps.find(*adi) == m_vIncomingDuplicateDeps.end()) {                         \
            COMPLAIN("WARNING: wait async resolved between check/wait, but dep was not a duplicate!\n");    \
        } } }

#else 
#define COMPLAIN(x) 
#define GET_OSTENSIBLE_DEPENDENCES(deps, buf)  
#define CLEAR_OSTENSIBLE_DEPENDENCES()         
#define VALIDATE_EARLY_RESOLUTION(pDependences, pBuffer, pbAlreadyResolved)
#define FIND_DUPLICATE_DEPENDENCES()
#endif

namespace PTask {

    static double s_gWeight_CurrentWaitBump = 3.5;
    static double s_gWeight_DecayedWaitBump = 1.0;
    static double s_gWeight_DispatchTimeBump = 1.0;
    static double s_gWeight_OSPrioBump = 1.0;

#if (defined(DANDELION_DEBUG) || defined(DUMP_INTERMEDIATE_BLOCKS))
#define dump_block(b, sz)   (b)->DebugDump(&g_ss, &g_sslock, m_lpszTaskName, (sz), 0)
#define dump_buffer(p, buf) (buf)->DebugDump(&g_ss, &g_sslock, m_lpszTaskName, p->GetVariableBinding())
#define dump_dispatch_info(x,y)  DumpDispatchInfo(x,y)
#else
#define dump_block(pDest, strType) 
#define dump_buffer(port, pBuffer) 
#define dump_dispatch_info(x,y)  DumpDispatchInfo(x,y)
#endif

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="hRuntimeTerminateEvt"> Handle of the terminate event. </param>
    /// <param name="hGraphTeardownEvent">  Handle of the graph's teardown event. When a task's
    ///                                     dispatch thread enters a wait (e.g. on a port status
    ///                                     event or dispatch event), it must also wait on the
    ///                                     graph's teardown event to ensure that dispatch threads
    ///                                     are awakened and can exit before the runtime's data
    ///                                     structures are cleaned up. </param>
    /// <param name="hGraphStopEvent">      Handle of the graph's stop event. When a task's dispatch
    ///                                     thread enters a wait (e.g. on a port status event or
    ///                                     dispatch event), it must also wait on the stop to ensure
    ///                                     that dispatch threads can exit before the runtime's data
    ///                                     structures are cleaned up. </param>
    /// <param name="hGraphRunningEvent">   Handle of the graph's running event. </param>
    /// <param name="pCompiledKernel">  CompiledKernel associated with this task. </param>
    ///-------------------------------------------------------------------------------------------------

    Task::Task(
        __in HANDLE hRuntimeTerminateEvent, 
        __in HANDLE hGraphTeardownEvent,
        __in HANDLE hGraphStopEvent, 
        __in HANDLE hGraphRunningEvent,
        __in CompiledKernel * pCompiledKernel
        ) : ReferenceCounted()
    {
        m_pGraph = NULL;
        m_pCompiledKernel = pCompiledKernel;
        m_nEstimatorElemsPerThread = PTGE_DEFAULT_ELEMENTS_PER_THREAD;
        m_nEstimatorGroupSizeX     = PTGE_DEFAULT_BASIC_GROUP_X;
        m_nEstimatorGroupSizeY     = PTGE_DEFAULT_BASIC_GROUP_Y;
        m_nEstimatorGroupSizeZ     = PTGE_DEFAULT_BASIC_GROUP_Z;
        m_bHasImplicitMetaChannelBindings = FALSE;
        m_lpszTaskName = NULL;
        m_nPriority = DEF_PTASK_PRIO;
        m_nEffectivePriority = DEF_PTASK_PRIO;
        m_nProxyOSPrio = DEF_OS_PRIO;
        m_pUsageTimer = new CHighResolutionTimer(gran_msec);
        m_dLastDispatchTime = 0.0;
        m_dAverageDispatchTime = 0.0;
        m_dLastWaitTime = 0.0;
        m_dAverageWaitTime = 0.0;
        m_pDispatchAccelerator = NULL;
        m_hPortStatusEvent = CreateEvent(NULL, FALSE, TRUE, NULL);
        m_hDispatchEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
        m_hDispatchThread = NULL;
        m_hRuntimeTerminateEvent = hRuntimeTerminateEvent;
        m_hGraphRunningEvent = hGraphRunningEvent;
        m_hGraphTeardownEvent = hGraphTeardownEvent;
        m_hGraphStopEvent = hGraphStopEvent;
        m_tEstimatorType = NO_SIZE_ESTIMATOR;
        m_lpfnEstimator = NULL;
        m_bProducesUnmigratableData = FALSE;
        m_pMandatoryAccelerator = NULL;
        m_pMandatoryDependentAccelerator = NULL;
        m_bMandatoryDependentAcceleratorValid = FALSE;
        m_nRecordCardinality = 1;
        m_nDispatchIterationCount = DEFAULT_DISPATCH_ITERATIONS;
        m_bHasPredicatedDataFlow = FALSE;
        m_bInflight = FALSE;
        m_bShutdown = FALSE;
        m_nDispatchNumber = 0;
        m_bRequestDependentPSObjects = FALSE;
        m_bHasDependentAffinities = FALSE;
        m_uiDependentBindingClasses = MAXDWORD32;
        m_uiSchedulerPartitionHint = 0;
        m_lpvSchedulerCookie = NULL;
        m_bSchedulerPartitionHintSet = FALSE;
        m_pDispatchCounter = NULL;
        m_pTaskProfile = NULL;
        m_bUserCodeAllocatesMemory = FALSE;
        m_bDependentPortSetValid = FALSE;
        m_bMandatoryConstraintsCacheValid = FALSE;
        m_bPreferenceConstraintsCacheValid = FALSE;
        InitializeCriticalSection(&m_csDispatchLock);
        InitializeInstanceDispatchCounter();
        InitializeTaskInstanceProfile();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Task::~Task() {
        assert(!LockIsHeld());
        map<Accelerator*, AsyncContext*>::iterator si;
        for(si=m_vDispatchAsyncContexts.begin(); si!=m_vDispatchAsyncContexts.end(); si++) {
            AsyncContext * pAsyncContext = si->second;
            Accelerator * pAccelerator = si->first;
            assert(pAsyncContext != NULL);
            assert(pAccelerator != NULL);
            assert((!pAccelerator->SupportsExplicitAsyncOperations()) ||
                   (ASYNCCTXT_ISEXECCTXT(pAsyncContext->GetAsyncContextType())));
            pAccelerator->Lock();
            pAsyncContext->Lock();
            if(pAccelerator->SupportsExplicitAsyncOperations()) {
                pAsyncContext->SynchronizeContext();
                pAsyncContext->TruncateOutstandingQueue(TRUE);
            } 
            pAccelerator->ReleaseTaskAsyncContext(this, pAsyncContext);
            pAsyncContext->Unlock();
            pAccelerator->Unlock();
            pAsyncContext->Release();
        }
        CloseHandle(m_hPortStatusEvent);
        CloseHandle(m_hDispatchEvent);
        DeleteCriticalSection(&m_csDispatchLock);
        delete m_pUsageTimer;
        MergeTaskInstanceStatistics();
        DeinitializeTaskInstanceProfile();
        std::map<AsyncContext*, std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>>*>::iterator ii;
        for(ii=m_vIncomingPendingBufferOps.begin(); ii!=m_vIncomingPendingBufferOps.end(); ii++) {
            std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>>* pSet = ii->second;
            if(pSet != NULL) { 
                if(pSet->size() != 0) 
                    pSet->clear();
                delete pSet;
            }
        }
        m_vIncomingPendingBufferOps.clear();
        std::map<AsyncContext*, std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>>*>::iterator mi;
        for(mi=m_vOutgoingPendingBufferOps.begin(); mi!=m_vOutgoingPendingBufferOps.end(); mi++) {
            std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>>* pSet = mi->second;
            if(pSet != NULL) { 
                if(pSet->size() != 0) 
                    pSet->clear();
                delete pSet;
            }
        }
        if(m_lpszTaskName)
            delete [] m_lpszTaskName;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resets a graph to its initial state. </summary>
    ///
    /// <remarks>   Crossbac, 5/2/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::Reset(
        VOID
        )
    {
        // TODO--nothing important for re-use in DNN
        assert(FALSE && "Task::Reset() unimplemented!");
        PTask::Runtime::MandatoryInform("%s unimplemented...ignoring call...\n",
                                        __FUNCTION__);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Shuts down this object and frees any resources it is using. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Task::Shutdown(
        VOID
        )
    {
        EnterCriticalSection(&m_csDispatchLock);
        m_bShutdown = TRUE;
        LeaveCriticalSection(&m_csDispatchLock);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if this task produces unmigratable data. This is the case if the
    ///             datablocks manipulated by the task contain pointers to device-side memory: a host-
    ///             side or other-device-side view of such pointers will not be meaningful, so we
    ///             cannot materialize views in other memory spaces.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Task::ProducesUnmigratableData(
        VOID
        )
    {
        return m_bProducesUnmigratableData;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the task name. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="s">    [in] non-null, the name of the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SetTaskName(
        char * s
        ) 
    {
        if(m_lpszTaskName)
            delete [] m_lpszTaskName;
        size_t len = strlen(s)+1;
        m_lpszTaskName = new char[len];
        strcpy_s(m_lpszTaskName, len, s);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the task name. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the task name. </returns>
    ///-------------------------------------------------------------------------------------------------

    const char * 
    Task::GetTaskName(
        VOID
        ) 
    {
        return (const char*) m_lpszTaskName;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the port status event. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <returns>   The port status event. </returns>
    ///-------------------------------------------------------------------------------------------------

    HANDLE 
    Task::GetPortStatusEvent(
        VOID
        ) 
    { 
        return m_hPortStatusEvent; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Raises a Port status change signal. This event should be set any time a change
    ///             occurs that might cause the task state to transition from waiting to ready (e.g.
    ///             an upstream channel push just occurred).
    ///             
    ///             The known ready flag is used to inform queueing of the next dispatch attempt,
    ///             based on the thread-to-task mapping in use by the graph. If we are using 1:1
    ///             mapping, the flag is meaningless because each task has it's own thread and does
    ///             not impact other tasks by doing a ready-state check that does not lead to a
    ///             dispatch because the task is not really ready. In all other modes, this signal is
    ///             accompanies by placement on a queue for graph runner procs to make dispatch
    ///             attempts, so there is some motivation to avoid queueing tasks before they are
    ///             truly ready. Since that check is expensive and requires complex synchronization,
    ///             we make a conservative estimate at enqueue time and skip queuing tasks for which
    ///             we can be sure we will receive a subsequent signal. When set, the known ready
    ///             flag allows us to skip that check and is computed at a time when we already hold
    ///             all the locks we need.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="bKnownReady">  True if we know for sure the task is ready. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SignalPortStatusChange(
        __in BOOL bKnownReady
        ) 
    { 
        SetEvent(m_hPortStatusEvent); 
        if(m_pGraph != NULL && m_pGraph->IsRunning())
            m_pGraph->SignalReadyStateChange(this, bKnownReady);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Raises the Dispatch signal, telling this task that it has been
    /// 			assigned an accelerator by the scheduler. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SignalDispatch(
        VOID
        ) 
    { 
        SetEvent(m_hDispatchEvent); 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the enter ready queue action. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::OnEnterReadyQueue(
        VOID
        )
    {
        Lock();
        m_pUsageTimer->reset();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Updates statistics for wait time on the ready Q. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Task::OnUpdateWait(
        VOID
        )
    {
        Lock();
        double dWaitTime = m_pUsageTimer->elapsed(false);
        m_dLastWaitTime = dWaitTime;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the begin dispatch action. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::OnBeginDispatch(
        VOID
        ) 
    {
        Lock();
        double dWaitTime = m_pUsageTimer->elapsed(false);
        double dExpTotal = (dWaitTime * D0_WEIGHT) + 
                           (m_dLastWaitTime * D1_WEIGHT) + 
                           (m_dAverageWaitTime * D2_WEIGHT);
        m_dLastWaitTime = dWaitTime;
        m_dAverageWaitTime = dExpTotal;
        m_pUsageTimer->reset();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the complete dispatch action. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::OnCompleteDispatch(
        VOID
        ) 
    {
        MARKRANGEENTER(L"OnCompleteDispatch");
        Lock();
        MARKRANGEENTER(L"OnCompleteDispatch(locked)");
        double dDispatchTime = m_pUsageTimer->elapsed(false);
        double dExpTotal = (dDispatchTime * D0_WEIGHT) + 
                           (m_dLastDispatchTime * D1_WEIGHT) + 
                           (m_dAverageDispatchTime * D2_WEIGHT);
        m_dLastDispatchTime = dDispatchTime;
        m_dAverageDispatchTime = dExpTotal;        
        MARKRANGEEXIT();
        Unlock();
        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets usage statistics. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="pStats">   The stats. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::GetUsageStats(
        PPTASKUSAGESTATS pStats
        )
    {
        Lock();
        pStats->dAverageDispatchTime = m_dAverageDispatchTime;
        pStats->dAverageWaitTime = m_dAverageWaitTime;
        pStats->dLastDispatchTime = m_dLastDispatchTime;
        pStats->dLastWaitTime = m_dLastWaitTime;
        pStats->dAverageOSPrio = 0.0;
        if(m_nProxyOSPrio != DEF_OS_PRIO && m_hDispatchThread != NULL) {
            pStats->dAverageOSPrio = (double) GetThreadPriority(m_hDispatchThread);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Calculates the effective priority. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="dAvgDispatch">     The average dispatch. </param>
    /// <param name="dAvgCurrentWait">  The average current wait. </param>
    /// <param name="dAvgDecayedWaits"> The average decayed waits. </param>
    /// <param name="dAvgOSPrio">       The average operating system prio. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::ComputeEffectivePriority(
        double dAvgDispatch, 
        double dAvgCurrentWait, 
        double dAvgDecayedWaits,
        double dAvgOSPrio
        ) 
    {
        // we can't preempt anyone, making priority a very coarse tool. 
        // GOAL: we want effective priority to be high if:
        // a) priority is high
        // b) wait time is high
        // c) gpu usage is low
        // d) os proxy priority is high
        // So the approach is to perturb the base priority
        // positively in proportion to the difference between this task's
        // wait time and the average, and negatively in proportion to the
        // delta between this task's dispatch time and the average. 
        Lock();
        int nWaitBump = 0;
        int nDispatchBump = 0;
        int nOSPrioBump = 0;
        double dCurrentWaitDelta = -777;
        if(dAvgCurrentWait > 0) {
            dCurrentWaitDelta = (m_dLastWaitTime - dAvgCurrentWait) / dAvgCurrentWait;
            nWaitBump = (int) ceil(s_gWeight_CurrentWaitBump * dCurrentWaitDelta);
        }
        double dDecayedWaitDelta = -777;
        if(dAvgDecayedWaits > 0 &&
            m_dAverageWaitTime != 0.0) 
        {
            dDecayedWaitDelta = (m_dAverageWaitTime - dAvgDecayedWaits) / dAvgDecayedWaits;
            nWaitBump += (int) ceil(s_gWeight_DecayedWaitBump * dDecayedWaitDelta);
        }
        if(dAvgDispatch > 0) {
            double dDispatchDelta = m_dAverageDispatchTime - dAvgDispatch;
            nDispatchBump = (int) ceil(s_gWeight_DispatchTimeBump * dDispatchDelta);
        }
        if(dAvgOSPrio > 0 && m_nProxyOSPrio != DEF_OS_PRIO) {
            // OS proxy prio attempts to avoid priority laundering by
            // assigning the priority of the client process to the threads
            // that manage its PTasks. If this value is actually set, we
            // bump the effective priority proportionally to the delta
            // between it and the other waiters. 
            int osprio = GetThreadPriority(m_hDispatchThread);
            double dOSPrioDelta = osprio - dAvgOSPrio;
            nOSPrioBump = (int) ceil(s_gWeight_OSPrioBump * dOSPrioDelta);
        }
        m_nEffectivePriority = m_nPriority + nWaitBump + nDispatchBump + nOSPrioBump;
        m_nEffectivePriority = max(m_nEffectivePriority, MIN_PTASK_PRIO);
        m_nEffectivePriority = min(m_nEffectivePriority, MAX_PTASK_PRIO);
/*        printf("[Task %s: CurrentWaitTime=%1f, avgWaitTime=%.1f, currentWaitDelta=%.1f, decayedWaitDelta=%.1f, basePrio=%d, ePrio=%d, waitBump=%d, dispatchBump=%d, OSprioBump=%d]\n",
            m_lpszTaskName, m_dLastWaitTime, m_dAverageWaitTime, dCurrentWaitDelta, dDecayedWaitDelta,
            m_nPriority, m_nEffectivePriority, nWaitBump, nDispatchBump, nOSPrioBump);
*/        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the dispatch accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SetDispatchAccelerator(
        Accelerator * pAccelerator
        )
    {
        Lock();
        m_pDispatchAccelerator = pAccelerator;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the priority for this task. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="nPrio">        The prio. </param>
    /// <param name="nOSProxyPrio"> The operating system proxy prio. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SetPriority(
        int nPrio,
        int nOSProxyPrio
        )
    {
        Lock();
        m_nPriority = nPrio;
        m_nEffectivePriority = nPrio;
        m_nProxyOSPrio = nOSProxyPrio;
        if(m_nProxyOSPrio != DEF_OS_PRIO && m_hDispatchThread != NULL) {
            SetThreadPriority(m_hDispatchThread, nOSProxyPrio);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the effective priority. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="nPrio">    The prio. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SetEffectivePriority(
        int nPrio
        )
    {
        Lock();
        m_nEffectivePriority = nPrio;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the task priority. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <returns>   The priority. </returns>
    ///-------------------------------------------------------------------------------------------------

    int
    Task::GetPriority(
        VOID
        )
    {
        int prio;
        Lock();
        prio = m_nPriority;
        Unlock();
        return prio;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the effective priority. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <returns>   The effective priority. </returns>
    ///-------------------------------------------------------------------------------------------------

    int
    Task::GetEffectivePriority(
        VOID
        )
    {
        int prio;
        Lock();
        prio = m_nEffectivePriority;
        Unlock();
        return prio;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a dispatch thread. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="h">    Handle of the dispatch thread. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SetDispatchThread(
        HANDLE h
        )
    {
        Lock();
        m_hDispatchThread = h;
        if(m_nProxyOSPrio != DEF_OS_PRIO) {
            SetThreadPriority(m_hDispatchThread, m_nProxyOSPrio);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the dispatch thread. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <returns>   The dispatch thread handle. </returns>
    ///-------------------------------------------------------------------------------------------------

    HANDLE 
    Task::GetDispatchThread(
        VOID
        )
    {
        return m_hDispatchThread;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind port. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    /// <param name="index">    Zero-based index of the variable to bind. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    Task::BindPort(
        Port * pPort, 
        PORTINDEX index
        ) 
    {
        assert(pPort != NULL);
        assert(index != UNBOUND_PORT);
        if(pPort == NULL || index == UNBOUND_PORT)
            return E_FAIL;
        pPort->Lock();
        if(!pPort->IsMarshallable()) {
            m_bProducesUnmigratableData = TRUE;
        }
        map<UINT, Port*>* pMap = NULL;
        switch(pPort->GetPortType()) {
        case INPUT_PORT: pMap = &m_mapInputPorts; break;
        case OUTPUT_PORT: 
            pPort->Unlock();
            return BindOutputPort(pPort, index);
        case STICKY_PORT: pMap = &m_mapConstantPorts; break;
        case META_PORT: pMap = &m_mapMetaPorts; break;
        case INITIALIZER_PORT: pMap = &m_mapInputPorts; break;
        }
        if(pMap->find(index) != pMap->end()) {
            Port* pOldPort = (*pMap)[index];
            delete pOldPort;
        } 
        (*pMap)[index] = pPort;
        pPort->BindTask(this, index);
        pPort->Unlock();
        return S_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind an output port to a given variable in the ptask, specified by ordinal
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/8/2013. </remarks>
    ///
    /// <param name="pPort">    port to bind </param>
    /// <param name="index">    index in the function signature of the variable to bind to the port </param>
    ///
    /// <returns>   success/failure. </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    Task::BindOutputPort(
        Port * pPort, 
        PORTINDEX index
        ) 
    {
        pPort->Lock();
        map<UINT, Port*>::iterator mi = m_mapOutputPorts.find(index);
        if(mi != m_mapOutputPorts.end()) {	
            Port * pOldPort = mi->second;
            m_mapOutputPorts.erase(index);
            delete pOldPort;
        }
        m_mapOutputPorts[index] = pPort;

        // binding an output port is somewhat more subtle because it requires to create a platform
        // specific buffer at the port that eventually becomes part of a Datablock. Since the actual
        // buffer is platform-specific, it is difficult to perform the binding before the port knows
        // what kind of task/accelerator it is attached to. 
        OutputPort * pOPort = (OutputPort*) pPort;
        pOPort->BindTask(this, index);
        pPort->Unlock();
        return S_OK;	
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   unbind a port from a given variable in the ptask, which is specified by ordinal
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/8/2013. </remarks>
    ///
    /// <param name="t">        The PORTTYPE to process. </param>
    /// <param name="index">    Zero-based index of the  port. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port * 
    Task::UnbindPort(
        PORTTYPE t, 
        PORTINDEX index
        )
    {
        assert(index != UNBOUND_PORT);
        if(index == UNBOUND_PORT)
            return NULL;
        switch(t) {
        case INPUT_PORT: return UnbindPort(m_mapInputPorts, index);
        case OUTPUT_PORT: return UnbindPort(m_mapOutputPorts, index);
        case STICKY_PORT: return UnbindPort(m_mapConstantPorts, index);
        }
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   unbind a port from a given variable in the ptask. find the required ordinal by
    ///             using our port maps.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/8/2013. </remarks>
    ///
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    ///
    /// <returns>   success/fail </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    Task::UnbindPort(
        Port * pPort
        )
    {
        HRESULT hrInputs = UnbindPort(m_mapInputPorts, pPort);
        HRESULT hrOutputs = UnbindPort(m_mapOutputPorts, pPort);
        HRESULT hrConst = UnbindPort(m_mapConstantPorts, pPort);
        return FAILED(hrInputs) ? hrInputs :
               FAILED(hrOutputs) ? hrOutputs :
               FAILED(hrConst) ? hrConst :
               S_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   unbind a port from a given variable in the ptask. find the port, based on the
    ///             ordinal.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/8/2013. </remarks>
    ///
    /// <param name="pmap">     [in,out] [in,out] If non-null, the pmap. </param>
    /// <param name="index">    Zero-based index of the. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port*
    Task::UnbindPort(
        std::map<UINT, Port*> &pmap, 
        PORTINDEX index
        ) 
    {
        if(pmap.find(index) != pmap.end()) {
            Port * pResult = pmap[index];
            pmap.erase(index);
            return pResult;
        }
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   unbind a port from a given variable in the ptask by removing it from the map.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/8/2013. </remarks>
    ///
    /// <param name="pmap">     [in,out] [in,out] If non-null, the pmap. </param>
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    ///
    /// <returns>   success/fail. </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    Task::UnbindPort(
        std::map<UINT, Port*> &pmap, 
        Port * pPort
        )
    {
        vector<UINT> eraseidx;
        map<UINT, Port*>::iterator mi;
        for(mi = pmap.begin(); mi!= pmap.end(); mi++) {
            if(mi->second == pPort) {
                eraseidx.push_back(mi->first);
            }
        }
        assert(eraseidx.size() >= 1);
        if(eraseidx.size() > 0) {
            for(vector<UINT>::iterator vi=eraseidx.begin();
                vi != eraseidx.end(); vi++) {
                    pmap.erase(*vi);
            }
            return S_OK;
        }
        return E_FAIL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enqueue ready. put this PTask on the ready queue, indicating it is ready to begin
    ///             dispatch.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Task::ScheduleOrEnqueue(
        BOOL& bQueued,
        BOOL bBypassQueue
        ) 
    {
        // return Scheduler::BeginDispatch(this);
        return Scheduler::ScheduleOrEnqueue(this, bQueued, bBypassQueue);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Received control block. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *      
    Task::ReceivedControlBlock(
        VOID
        ) 
    {
        map<UINT, Port*>::iterator mi;
        for(mi=m_mapInputPorts.begin(); 
            mi!=m_mapInputPorts.end();
            mi++) {
            Port * pPort = mi->second;
            Datablock * pBlock = pPort->Peek();
            if(pBlock != NULL) {
                pBlock->Lock();
                if(pBlock->IsControlToken()) {
                    pBlock->AddRef();
                    pBlock->Unlock();
                    return pBlock;
                }
                pBlock->Unlock();
            }
        }
        for(mi=m_mapConstantPorts.begin(); 
            mi!=m_mapConstantPorts.end();
            mi++) {
            Port * pPort = mi->second;
            Datablock * pBlock = pPort->Peek();
            if(pBlock != NULL) {
                pBlock->Lock();
                if(pBlock->IsControlToken()) {
                    pBlock->AddRef();
                    pBlock->Unlock();
                    return pBlock;
                }
                pBlock->Unlock();
            }
        }
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Propagate control blocks. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <param name="pCtlBlock">    [in,out] If non-null, the control block. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Task::PropagateControlBlocks(
        Datablock * pCtlBlock
        )
    {
        // Pull all the inputs and release. 
        // push the control block to all outputs.
        assert(FALSE); // obsolete!
        map<UINT, Port*>::iterator mi;
        for(mi=m_mapInputPorts.begin(); 
            mi!=m_mapInputPorts.end();
            mi++) {
            InputPort * pPort = (InputPort*) mi->second;
            Datablock * pBlock = pPort->Pull();
            if(pBlock != pCtlBlock)
                pBlock->Release();
        }
        // and the constant ports have available data
        for(mi=m_mapConstantPorts.begin(); 
            mi!=m_mapConstantPorts.end();
            mi++) {
            Port * pPort = (InputPort*) mi->second;
            Datablock * pBlock = pPort->Peek();
            if(pBlock != pCtlBlock)
                pBlock->Release();
        }
        for(mi=m_mapOutputPorts.begin();
            mi!=m_mapOutputPorts.end(); mi++) {
            OutputPort * pPort = (OutputPort*) mi->second;
            pPort->Push(pCtlBlock);
        }
        pCtlBlock->Release();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   "Estimates" the readiness of this task for dispatch. When the runtime is using
    ///             thread pools (or single-thread mode), the graph runner threads share a ready
    ///             queue, and tasks are enqueued in response to port status signals. This means that
    ///             tasks will almost always visit the ready queue more than once on their way to
    ///             dispatch because we are effectively using the graph runner procs to both test
    ///             readiness and dispatch, which are activities that almost certainly have different
    ///             optimal policies for the order of the queue (Note this is in constrast to the 1:1
    ///             thread:task mapping where non-ready tasks *attempting* to dispatch cannot impede
    ///             progress for actually ready task by clogging the ready queue). The *obvious*
    ///             design response is to avoid actually queuing tasks if they are not ready. There
    ///             is problem with this approach: tasks are signaled as candidates for the ready
    ///             queue when events like dispatch or user-space block pushing happen, both of which
    ///             are activities that typically involve non-trivial locking. Checking a task for
    ///             readiness can also involve locking particularly if there is predication on any of
    ///             the channels or ports that can cause a "ready" conclusion to become stale.
    ///             
    ///             The solution: compromise. Estimate the readiness of tasks and don't queue them if
    ///             we can be sure they aren't ready based on that estimate. The strategy is: for
    ///             tasks whose ready state can be checked without acquiring locks (because only a
    ///             subsequent dispatch can cause a transition out of the ready state), return the
    ///             actual ready state by traversing the state of the ports and channels bound to the
    ///             task. If we cannot make a conclusion because locks would be required, ASSUME the
    ///             task is ready. We'll occasionally try to dispatch something that is not ready.
    ///             But most of the time, we'll avoid clogging the ready queues and forcing the graph
    ///             runner threads to waste work.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/20/2013. 
    ///             WARNING: THIS METHOD returns an UNRELIABLE ANSWER. It elides locks and may 
    ///             claim a task is ready when it is not. Don't use it without 
    ///             that caveat in mind.
    ///             </remarks>
    ///
    /// <returns>   FALSE if we can be *sure* the task is not ready without taking locks.
    ///             TRUE if it is ready, or if we can't be sure without locking graph structures.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::EstimateReadyStatus(
        VOID
        )
    {
        AssembleIOLockList();
        if(m_bHasPredicatedDataFlow)
            return TRUE;
        BOOL bAllReady = TRUE;
        if(!(bAllReady &= IsPortMapReadyNoLock(&m_mapInputPorts))) return FALSE;
        if(!(bAllReady &= IsPortMapReadyNoLock(&m_mapMetaPorts))) return FALSE;
        if(!(bAllReady &= IsPortMapReadyNoLock(&m_mapConstantPorts))) return FALSE;
        if(!(bAllReady &= IsPortMapReadyNoLock(&m_mapOutputPorts))) return FALSE;
        return bAllReady;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   is this task ready for dispatch?. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <param name="pbAlive">  [in] non-null, the system alive flag (for early return). </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::IsReadyForDispatch(
        GRAPHSTATE * pGraphState
        )
    {
        BOOL bReady = FALSE;
        MARKRANGEENTER(L"IsReady");
        MARKRANGEENTER(L"IsReady:task-lock");
        Lock();
        MARKRANGEEXIT();
        if(HasPredicatedControlFlow()) {
            // if there are upstream data sources that have predicated flow or multiple input channels, it
            // is possible for the port state to transition from ready to not ready without a dispatch
            // occurring. In this case, we have to make sure that we at least provide a consistent view
            // (which may change later), but this involves locking all ports in order. Since the common
            // case is simple control flow, we'd prefer to avoid that expense. 
            bReady = IsReadyAtomic(pGraphState, FALSE);
        } else {
            // control flow structures are simple, so the only event that can force a transition
            // from ready to not ready for a data source or sink is for this task to dispatch. 
            // In short, because readiness is monotonic before dispatch, we don't need to lock
            // a bunch of resources to provide the correct answer. 
            bReady = IsReadyNoLock(pGraphState);
        }
        Unlock();
        MARKRANGEEXIT();
        return bReady;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Assemble port lock sets. </summary>
    ///
    /// <remarks>   crossbac, 6/21/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::AssemblePortLockSets(
        VOID
        )
    {
        map<UINT, Port*>::iterator mi;
        for(mi=m_mapMetaPorts.begin(); mi!=m_mapMetaPorts.end(); mi++) 
            mi->second->AssembleChannelLockSet();
        for(mi=m_mapConstantPorts.begin(); mi!=m_mapConstantPorts.end(); mi++) 
            mi->second->AssembleChannelLockSet();
        for(mi=m_mapInputPorts.begin(); mi!=m_mapInputPorts.end(); mi++) 
            mi->second->AssembleChannelLockSet();
        for(mi=m_mapOutputPorts.begin(); mi!=m_mapOutputPorts.end(); mi++) 
            mi->second->AssembleChannelLockSet();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Acquires the port locks described by pbAlive. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <param name="pbAlive">  [in] non-null, the system alive flag (for early return). </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    void
    Task::AssembleIOLockList(
        VOID
        )
    {
        if(m_vLockableIOResources.size() == 0) {
            assert(LockIsHeld());
            tpprofile_enter(AssembleIOLockList);
            // first assemble a list of all ports and channels that must be locked. 
            // we use a set here, which is ordered by default, so we have no additional
            // need to sort the assembled list of resources. We assemble this once,
            // on the assumption that the structure of the task does not change.
            set<Port*> vAllPorts;
            set<Port*>::iterator si;
            map<UINT, Port*>::iterator mi;
            for(mi=m_mapMetaPorts.begin(); mi!=m_mapMetaPorts.end(); mi++) 
                vAllPorts.insert(mi->second);
            for(mi=m_mapConstantPorts.begin(); mi!=m_mapConstantPorts.end(); mi++) 
                vAllPorts.insert(mi->second);
            for(mi=m_mapInputPorts.begin(); mi!=m_mapInputPorts.end(); mi++) 
                vAllPorts.insert(mi->second);
            for(mi=m_mapOutputPorts.begin(); mi!=m_mapOutputPorts.end(); mi++) 
                vAllPorts.insert(mi->second);
            for(si=vAllPorts.begin(); si!=vAllPorts.end(); si++) {
                Port * pPort = *si; 
                m_vLockableIOResources.insert(pPort); 
                PORTTYPE eType = pPort->GetPortType();
                int nChannelCount = 0;
                int nUpstreamPredicates = 0;
                for(UINT i=0; i<pPort->GetChannelCount(); i++) {
                    Channel * pChannel = pPort->GetChannel(i);
                    m_vLockableIOResources.insert(pChannel);
                    nChannelCount++;
                    if((pChannel->GetPredicationType(CE_SRC) != CGATEFN_NONE) ||
                        (pChannel->GetPredicationType(CE_DST) != CGATEFN_NONE))
                        nUpstreamPredicates++;
                }
                for(UINT i=0; i<pPort->GetControlChannelCount(); i++) {
                    Channel * pChannel = pPort->GetControlChannel(i);
                    m_vLockableIOResources.insert(pChannel);
                    nChannelCount++;
                    if((pChannel->GetPredicationType(CE_SRC) != CGATEFN_NONE) ||
                        (pChannel->GetPredicationType(CE_DST) != CGATEFN_NONE))
                        nUpstreamPredicates++;
                }
                BOOL bMultipleDataSources = 
                    (nChannelCount > 1 && eType != OUTPUT_PORT) || 
                    (nChannelCount > 0 && eType == INITIALIZER_PORT);
                BOOL bMonotonic = (nUpstreamPredicates == 0) && !bMultipleDataSources; 
                m_bHasPredicatedDataFlow |= !bMonotonic;
            }
            tpprofile_exit(AssembleIOLockList);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pbAlive' is ready atomic. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <param name="pbAlive">              [in] non-null, the system alive flag (for early return). </param>
    /// <param name="bExitWithLocksHeld">   true to exit with locks held on success </param>
    ///
    /// <returns>   true if ready atomic, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::IsReadyAtomic(
        GRAPHSTATE * pGraphState,
        BOOL bExitWithLocksHeld
        )
    {
        // we need to lock everything that 
        BOOL bReady = TRUE;
        assert(LockIsHeld());
        AssembleIOLockList();
        set<Lockable*>::iterator si;
        set<Lockable*> pLockList;
        MARKRANGEENTER(L"IsReadyAtomic-LockAcq");
        for(si=m_vLockableIOResources.begin(); si!=m_vLockableIOResources.end(); si++) {
            // start acquiring locks, but don't keep acquiring if
            // know we will have to return FALSE. Note that we 
            // don't have to check ready state of any channels, we
            // just have to lock them (since they may influence port state).
            Lockable * pLockable = *si;
            MARKRANGEENTER(L"IsReadyAtomic-ind-resource-lock");
            pLockable->Lock();
            MARKRANGEEXIT();
            pLockList.insert(pLockable);
            Port * pPort = dynamic_cast<Port*>(pLockable);
            if(pPort != NULL) {
                // std::cout << "\tWarpImage[" << pPort << "]:\t";
                if(pPort->GetPortType() == OUTPUT_PORT) {
                    if(!(bReady &= IsGraphRunning(pGraphState) && !pPort->IsOccupied())) {
                        // std::cout << " NOT READY" << std::endl;
                        break;
                    }
                } else {
                    if(!(bReady &= IsGraphRunning(pGraphState) && pPort->IsOccupied())) {
                        // std::cout << " NOT READY" << std::endl;
                        break; 
                    }
                }
                // std::cout << " READY" << std::endl;
            }
        }
        MARKRANGEEXIT();
        if(!bReady || !bExitWithLocksHeld) {
            for(si=pLockList.begin(); si!=pLockList.end(); si++) {
                (*si)->Unlock(); 
            }
        }
        return bReady; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has predicated control flow. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <returns>   true if predicated control flow, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::HasPredicatedControlFlow(
        VOID
        )
    {
        assert(LockIsHeld());
        AssembleIOLockList(); // sets the value of flag below. 
        return m_bHasPredicatedDataFlow;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the given port map has all ready ports. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <returns>   true if port map ready, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::IsPortMapReady(
        std::map<UINT, Port*>* pPortMap,
        GRAPHSTATE * pGraphState
        )
    {
        assert(LockIsHeld());
        BOOL bAllReady = TRUE;
        map<UINT, Port*>::iterator mi;
        for(mi=pPortMap->begin(); 
            mi!=pPortMap->end() && bAllReady && IsGraphRunning(pGraphState); 
            mi++) {
            Port * pPort = mi->second;
            if(pPort->GetPortType() == OUTPUT_PORT) 
                bAllReady &= !pPort->IsOccupied() && IsGraphRunning(pGraphState);
            else 
                bAllReady &= pPort->IsOccupied() && IsGraphRunning(pGraphState);
            if(!bAllReady) return FALSE;
        }
        return bAllReady;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the given port map has all ready ports. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <returns>   true if port map ready, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::IsPortMapReadyNoLock(
        std::map<UINT, Port*>* pPortMap
        )
    {
        map<UINT, Port*>::iterator mi;
        for(mi=pPortMap->begin(); mi!=pPortMap->end(); mi++) {
            Port * pPort = mi->second;
            if(pPort->GetPortType() == OUTPUT_PORT) {
                if(pPort->IsOccupied())
                    return FALSE;
            } else {
                if(!pPort->IsOccupied())
                    return FALSE;
            }
            
        }
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   non-atomic check: is this task ready for dispatch?. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <param name="pbAlive">  [in] non-null, the system alive flag (for early return). </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::IsReadyNoLock(
        GRAPHSTATE * pGraphState
        )
    {
        // before the if the input ports are available note that we needn't explicitly check meta port
        // availability, since they are occupied by definition. 
        // -----------------------------------------------------
        // Note: by non-atomic, we mean there is no requirement to hold port locks while
        // doing the check. We still need to hold the task lock while doing this. 
        assert(LockIsHeld());
        MARKRANGEENTER(L"IsReadyNoLock");
        BOOL bAllReady = TRUE;
        if(bAllReady) bAllReady &= IsPortMapReady(&m_mapInputPorts, pGraphState);
        if(bAllReady) bAllReady &= IsPortMapReady(&m_mapMetaPorts, pGraphState);
        if(bAllReady) bAllReady &= IsPortMapReady(&m_mapConstantPorts, pGraphState);
        if(bAllReady) bAllReady &= IsPortMapReady(&m_mapOutputPorts, pGraphState);
        MARKRANGEEXIT();
        return bAllReady;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Schedules. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <param name="pGraphState">  [in,out] The first parameter. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator *
    Task::Schedule(
        GRAPHSTATE * pGraphState
        )
    {
        UNREFERENCED_PARAMETER(pGraphState);
        m_nDispatchNumber++;
        tpprofile_enter(Schedule);
        MARKRANGEENTER(L"Schedule");
        Accelerator * pDispatchAccelerator = NULL;
        assert(Scheduler::GetSchedulingMode() != SCHEDMODE_COOPERATIVE);

        // put ourselves on the scheduler's ready queue, and then wait for notification that the
        // scheduler has actually assigned us an acclerator, meaning we can call run. 
        BOOL bQueued = FALSE;
        BOOL bScheduled = ScheduleOrEnqueue(bQueued, TRUE);
        BOOL bExitBeforeDispatch = FALSE;
        BOOL bStopped = FALSE;
        BOOL bTerminate = FALSE;

        if(!bScheduled && bQueued) {
            HANDLE vHandles[] = { m_hDispatchEvent, 
                                  m_hRuntimeTerminateEvent, 
                                  m_hGraphTeardownEvent, 
                                  m_hGraphStopEvent };
            DWORD dwHandles = sizeof(vHandles) / sizeof(HANDLE);
            DWORD dwWait = WAIT_TIMEOUT;

            // we're on the ready q and will only wake up under 3 conditions:
            // 1. The scheduler has assigned the dispatch resources we require (accelerator(s))
            // 2. The user program has called Graph::Stop on this graph
            // 3. The user program is shutting down the runtime. 
            // Cases 2 and 3 are worth distinguishing for visibility/debug, but
            // we handle them identically. In particular, if the user calls Stop while
            // we are waiting for an accelerator, there is probably a problem with the
            // program--why is there are task ready at this time? The call should probably
            // quiesce the scheduler with respect to the task first because if we abandon a
            // dispatch it is difficult to make guarantees about the graph state. 
            // Conversely, a terminate is OK--we are cleaning up!
            
            tpprofile_enter(BlockedOnReadyQ);
            MARKRANGEENTER(L"BlockedOnReadyQ");
            do {
                record_wait_acc_entry();
                dwWait = WaitForMultipleObjects(dwHandles, vHandles, FALSE, INFINITE);
                record_wait_acc_exit();
                bExitBeforeDispatch = (dwWait != WAIT_OBJECT_0);
                bStopped = (dwWait == WAIT_OBJECT_0 + 3);  // pause--leave things in the deferred queue.
                bTerminate = (dwWait == WAIT_OBJECT_0 + 1) || (dwWait == WAIT_OBJECT_0 + 2); // hard stop-shut down
                if(bExitBeforeDispatch && bQueued) {
                    PTask::Runtime::MandatoryInform("Task::Schedule(%s) aborting dispatch: %s event raised!\n",
                                                    m_lpszTaskName,
                                                    ((dwWait == WAIT_OBJECT_0+3)?"STOP":"TERMINATE"));
                }
            } while(dwWait == WAIT_TIMEOUT && !bExitBeforeDispatch);
            tpprofile_exit(BlockedOnReadyQ);
            MARKRANGEEXIT();
        }

        if(bExitBeforeDispatch) {

            // either the graph was stopped while we were on the ready Q or the
            // user started shutting down while we were on the ready Q. Chances
            // are we hold some important locks though, so we need to inform the
            // scheduler that we are giving up, but don't require it to be in the same
            // state we would require if we were going forward with the dispatch. 
            // In short, we don't know if the dispatch resources were assigned or 
            // not when we got the stop or terminate signal.

            MARKRANGEENTER(L"Abandon-dispatch");
            Scheduler::AbandonCurrentDispatch(this);
            if(bTerminate) {
                // get ourselves out of the run queue
                Scheduler::AbandonDispatches(this);
            } else {
                // we are likey in the deferred queue
                // reschedule ourselves for when the
                // graph is put back into the run state.
                assert(bStopped);
                SignalPortStatusChange(TRUE);
            }
            Lock();
            pDispatchAccelerator = NULL;
            m_pDispatchAccelerator = NULL;            
            Unlock();
            MARKRANGEEXIT();

        } else {

            // the scheduler has assigned a dispatch accelerator and any dependent
            // resources we require. Go forward and execute the task!
            
            MARKRANGEENTER(L"Scheduled-dispatch");
            OnBeginDispatch();
            Dispatch();
            OnCompleteDispatch();
            pDispatchAccelerator = m_pDispatchAccelerator;
            m_pDispatchAccelerator = NULL;
            MARKRANGEEXIT();
        }
        tpprofile_exit(Schedule);
        MARKRANGEEXIT();
        return pDispatchAccelerator;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Propagate data movement after dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <param name="pDispatchAccelerator"> [in] non-null, the dispatch accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::PropagateDataflow(
        Accelerator * pDispatchAccelerator
        )
    {
        // if we didn't get an accelerator, the
        // dispatch did not actually occur!
        if(!pDispatchAccelerator) 
            return; 

        // materialize any outputs that should be visible in
        // the host domain, and perform push for any datablocks
        // on outbound ports
        MARKRANGEENTER(L"PropagateDataflow");
        tpprofile_enter(PropagateDataflow);
        BOOL bOutputPortsHaveCapacity = TRUE;
        map<UINT, Port*>::iterator mi;
        for(mi=m_mapOutputPorts.begin(); mi!=m_mapOutputPorts.end(); mi++) {

            OutputPort * pPort = (OutputPort*) mi->second;
            InputPort* pInOutProducer = (InputPort*)pPort->GetInOutProducer();
            if(pInOutProducer != NULL) {

                // if this is an output port for an in/out parameter port pair, then the 
                // datablock was already pushed in ReleaseInflightDatablocks, so there is 
                // nothing remaining to be done. 
                dump_block((pPort->GetDestinationBuffer()), "output");
                continue;
            }

            BOOL bMaterializeViewCandidate = 
                pPort->HasDownstreamHostConsumer() && 
                GetPortTargetClass(pPort) != ACCELERATOR_CLASS_HOST;
            bMaterializeViewCandidate |= PTask::Runtime::GetDebugMode();

            Datablock * pDest = pPort->GetDestinationBuffer();
            MARKRANGEENTER(L"entLockViewForSync");
            Accelerator * pMostRecent = pDest->LockForViewSync(bMaterializeViewCandidate);
            if(pPort->GetControlPropagationSource() != NULL) {
                pPort->Lock();
                CONTROLSIGNAL luiPropagatedSignal = pPort->GetPropagatedControlSignals(); 
                pDest->SetControlSignal(luiPropagatedSignal);
                ctlpegress(this, pDest);
                pPort->ClearAllPropagatedControlSignals();
                pPort->Unlock();
            }            
            if(bMaterializeViewCandidate) {
                MaterializeDownstreamViews(pDest, pPort);
            }
            pDest->UnlockForViewSync(bMaterializeViewCandidate, pMostRecent);
            MARKRANGEEXIT();
            pPort->Push(pDest);
            pPort->ReleaseDestinationBuffer();
            pPort->ClearAllPropagatedControlSignals();
            bOutputPortsHaveCapacity &= pPort->IsOccupied();
        }

        if(PTask::Runtime::GetForceSynchronous())
            pDispatchAccelerator->Synchronize(this);
        tpprofile_exit(PropagateDataflow);
        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes a task. Do not return until it has executed unless the
    ///             runtime or the graph is torn down. This Execute method should only 
    ///             be used when the runtime is using the 1:1 threads:tasks mapping
    ///             because the thread calling execute will block on the port status
    ///             event if the task is not ready. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pGraphState">     [in] non-null, the system alive flag (for early return). </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::Execute(
        __in GRAPHSTATE * pGraphState
        ) 
    {
        BOOL bTerminate = FALSE;
        BOOL bAllReady = IsReadyForDispatch(pGraphState);

        tpprofile_enter(BlockedNotReady);
        while(!bAllReady &&                     // the graph is not ready
              IsGraphAlive(pGraphState) &&      // graph state suggests it is ok to dispatch
              !m_bShutdown &&                   // the task is not marked for shutdown
              !bTerminate                       // we have not been signaled for termination
              ) {

            // this variant of the execute method expects it's own dedicated thread for calling this method
            // in a tight loop--in such a case it is safe to block when we determine the task is is not yet
            // ready because doing so does not make the thread unavailable for other tasks. Wait for the
            // port status event to be signaled, which will occur when operations occur on ports or
            // channels connected to this task. 
            // 
            HANDLE vHandles[] = { m_hPortStatusEvent, m_hRuntimeTerminateEvent, m_hGraphTeardownEvent, m_hGraphStopEvent };
            DWORD dwHandles = sizeof(vHandles)/sizeof(HANDLE);
            DWORD dwWait = WaitForMultipleObjects(dwHandles, vHandles, FALSE, INFINITE);
            switch(dwWait) {
            case WAIT_TIMEOUT: assert(FALSE); continue;
            case WAIT_OBJECT_0 + 0: bAllReady = IsReadyForDispatch(pGraphState); break;
            case WAIT_OBJECT_0 + 1: bTerminate = TRUE; continue; 
            case WAIT_OBJECT_0 + 2: bTerminate = TRUE; continue;
            case WAIT_OBJECT_0 + 3: bTerminate = TRUE; continue;
            }
        }
        tpprofile_exit(BlockedNotReady);

        assert(bAllReady || !IsGraphRunning(pGraphState));
        if(!IsGraphRunning(pGraphState) || m_bShutdown || bTerminate) return;

        // will block until dispatch is complete.
        // returns accelerator on which the task ran.
        Accelerator * pDispatchAccelerator = Schedule(pGraphState);
        if(!IsGraphAlive(pGraphState) || m_bShutdown) // did a shutdown start while we were blocked?
            return;

        PropagateDataflow(pDispatchAccelerator);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Attempt to execute a task: if all it's inputs and outputs are ready, schedule it
    ///             and dispatch it. If the task is not ready, return without dispatching. This
    ///             method should be used for all cases when the 1:1 thread:task mapping is *not* in
    ///             use.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pGraphState">  [in] non-null, the system alive flag (for early return). </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Task::AttemptExecute(
        __in GRAPHSTATE * pGraphState
        )
    {
        BOOL bDispatchSucceeded = FALSE;
        if(IsReadyForDispatch(pGraphState) && IsGraphRunning(pGraphState) && !m_bShutdown) {

            // if we are ready (ie all inputs are available and outputs have capacity, call schedule. This
            // method will put us on the scheduler's ready q, which will block until an accelerator is
            // assigned and will subsequently dispatch us. If the dispatch is abandoned before it completes
            // a NULL dispatch accelerator is returned, allowing the PropagateDataflow method to elide its
            // work. If the task is still ready for dispatch (either because upstream queues already have
            // more blocks, or because we abandoned the dispatch and are *still* ready)
            // then raise a port status signal, which will enqueue us once once again. 
            
            Accelerator * pDispatchAccelerator = Schedule(pGraphState);
            bDispatchSucceeded = pDispatchAccelerator != NULL;
            if(!IsGraphAlive(pGraphState) || m_bShutdown) 
                return bDispatchSucceeded;
            PropagateDataflow(pDispatchAccelerator);
            if(IsReadyForDispatch(pGraphState))
                SignalPortStatusChange(TRUE);
        }
        return bDispatchSucceeded;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize downstream views. </summary>
    ///
    /// <remarks>   Crossbac, 9/28/2012. </remarks>
    ///
    /// <param name="pBlock">   If non-null, the block. </param>
    /// <param name="pPort">    If non-null, the port. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::MaterializeDownstreamViews(
        __in Datablock * pBlock, 
        __in Port * pPort
        )
    {

        #define GETLASTACC(pLastAcc, uiPortMemorySpaceID)       \
            if(!pLastAcc) {                                     \
                pLastAcc = pBlock->GetMostRecentAccelerator();  \
                uiPortMemorySpaceID = pLastAcc ?                \
                    pLastAcc->GetMemorySpaceId() :              \
                    m_pDispatchAccelerator->GetMemorySpaceId();}

        // look at all channels connected to the output port from this in/out pair,
        // and determine whether the channel is connected to anything that warrants an investment
        // of our effort in eager materialization of a view we know will be needed by a downstream
        // consumer. Three structures can give rise to this need:
        // 1) An output channel that has been configued to use an eager materialization policy
        // 2) An internal channel connected to a meta-port (which we typically know requires
        //    a valid host view to perform allocation before the next dispatch).
        // 3) The runtime is in the programmer-controlled debug mode, which always
        //    materializes views of data on internal channels.

        MARKRANGEENTER(L"MaterializeDownstreamViews");
        tpprofile_enter(RIBMaterializeViews);
        Accelerator * pLastAcc = NULL;
        UINT uiPortMemorySpaceID = 0;
        BOOL bViewMaterialized = FALSE;
        OutputPort * pConsumer = (OutputPort*) pPort;
        BOOL bDownstreamHostConsumer = pConsumer->HasDownstreamHostConsumer();
        BOOL bViewMaterializeCandidate = bDownstreamHostConsumer || Runtime::GetDebugMode();
        UNREFERENCED_PARAMETER(bViewMaterializeCandidate);

        assert(pBlock != NULL);
        assert(pBlock->LockIsHeld());
        assert(bViewMaterializeCandidate); // shouldn't get called otherwise

        BOOL bForceSynchronous = Runtime::GetDebugMode();
        BOOL bMaterializeView = Runtime::GetDebugMode();
        for(UINT i=0; i<pConsumer->GetChannelCount() && !bMaterializeView && bDownstreamHostConsumer; i++) {
                    
            Channel * pChannel = pConsumer->GetChannel(i);
            VIEWMATERIALIZATIONPOLICY policy;
            Port * pDownstreamPort = NULL;

            // a port can be connected to multiple downstream channels, and any one of them can trigger
            // this materialization. Only force a materialization if the same logic for a previous channel
            // on this port hasn't done it. 

            switch(pChannel->GetType()) {
            case CT_GRAPH_OUTPUT: 
                policy = pChannel->GetViewMaterializationPolicy();
                if(policy == VIEWMATERIALIZATIONPOLICY_EAGER) {
                    GETLASTACC(pLastAcc, uiPortMemorySpaceID);
                    bMaterializeView |= (uiPortMemorySpaceID != HOST_MEMORY_SPACE_ID);
                }
                break;
            case CT_INTERNAL:
                pDownstreamPort = pChannel->GetBoundPort(CE_DST);
                GETLASTACC(pLastAcc, uiPortMemorySpaceID);
                if(Runtime::GetEagerMetaPortMode() && pDownstreamPort->GetPortType() == META_PORT) {
                    MetaPort* pMetaPort = reinterpret_cast<MetaPort*>(pDownstreamPort);
                    BOOL bStaticAllocationSize = pMetaPort->IsStaticAllocationSize();
                    bMaterializeView |= (!bStaticAllocationSize && (uiPortMemorySpaceID != HOST_MEMORY_SPACE_ID));
                } else {
                    Task * pDownstreamTask = pDownstreamPort->GetTask();
                    if(pDownstreamTask != NULL &&
                       GetPortTargetClass(pDownstreamPort) == ACCELERATOR_CLASS_HOST &&
                       pDownstreamPort->GetPortType() == INPUT_PORT) {
                        bMaterializeView |= bDownstreamHostConsumer;
                    }
                }
                break;
            default:
                break;
            }
        }
                    
        if(bMaterializeView) {

            // we need to create a host view because the consumer on this channel will want one (or because
            // we're in debug mode). We only want a synchronous transfer if we're in debug mode though,
            // since the thread that will want the view may not need it for a while (e.g. waiting for inputs)

            MARKRANGEENTER(L"ViewUpdate-chk");
            if(pBlock->RequiresViewUpdate(HOST_MEMORY_SPACE_ID, BSTATE_SHARED)) {
                tpprofile_enter(RIBSyncHost);
                BOOL bAcquireViewLocks = FALSE; // we should have them already!
                assert(pLastAcc->LockIsHeld());
                AsyncContext * pTransferContext = GetOperationAsyncContext(pLastAcc, ASYNCCTXT_XFERDTOH);
                pBlock->SynchronizeHostView(pTransferContext, 
                                            BSTATE_SHARED, 
                                            bForceSynchronous,
                                            bAcquireViewLocks);
                tpprofile_exit(RIBSyncHost);
            }
            MARKRANGEEXIT();
            bViewMaterialized = TRUE;
        }
        tpprofile_exit(RIBMaterializeViews);
        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets control propagation paths. </summary>
    ///
    /// <remarks>   crossbac, 6/26/2014. </remarks>
    ///
    /// <param name="vTaskSignalSources">   [in,out] [in,out] If non-null, the task signal sources. </param>
    /// <param name="vPaths">               [in,out] On exit, a map from src port to the set of dest
    ///                                     ports reachable by control signals on the src port. </param>
    /// <param name="vSignalChannels">      [in,out] the set of channels that can carry outbound
    ///                                     control signals. </param>
    ///
    /// <returns>   The number of control propagation pairs. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Task::GetControlPropagationPaths(
        __inout std::map<Task*, std::set<Port*>>& vTaskSignalSources,
        __inout std::map<Port*, std::set<Port*>>& vPaths,
        __inout std::set<Channel*>& vSignalChannels
        )
    {
        UINT uiIncrementalPairCount = 0;
        std::map<UINT, Port*>::iterator mi;
        for(mi=m_mapOutputPorts.begin(); mi!=m_mapOutputPorts.end(); mi++) {
            OutputPort * pDstPort = reinterpret_cast<OutputPort*>(mi->second);
            assert(pDstPort->GetPortType() == OUTPUT_PORT);
            Port * pSrcPort = pDstPort->GetControlPropagationSource();
            if(pSrcPort != NULL) {
                vTaskSignalSources[this].insert(pSrcPort);
                std::set<Port*>& vDestSet = vPaths[pSrcPort];
                if(vDestSet.find(pDstPort) == vDestSet.end()) {
                    vDestSet.insert(pDstPort);
                    uiIncrementalPairCount++;
                }
                for(UINT ui=0; ui<pDstPort->GetChannelCount(); ui++) {
                    Channel * pOutboundChannel = pDstPort->GetChannel(ui);
                    if(pOutboundChannel != NULL)
                        vSignalChannels.insert(pOutboundChannel);
                }
            }
        }
        return uiIncrementalPairCount;
    }

    //--------------------------------------------------------------------------------------
    // release our reference to any datablocks that are in flight, typically causing
    //    inputs to get freed, unless the caller has kept a reference to them. 
    //--------------------------------------------------------------------------------------
    void
    Task::ReleaseInflightDatablocks(
        VOID
        ) 
    {
        MARKRANGEENTER(L"ReleaseInflightDatablocks");
        tpprofile_enter(ReleaseInflightDatablocks);

        std::map<Port*, Datablock*>::iterator mi;
        for(mi=m_vInflightDatablockMap.begin(); mi!=m_vInflightDatablockMap.end(); mi++) {

            Datablock * pBlock = mi->second;
            InputPort * pPort = reinterpret_cast<InputPort*>(mi->first);
            OutputPort* pInOutConsumer = (OutputPort*) pPort->GetInOutConsumer();
            if(pInOutConsumer != NULL) {

                BOOL bViewMaterializeCandidate = 
                    GetPortTargetClass(pInOutConsumer) != ACCELERATOR_CLASS_HOST && 
                    pInOutConsumer->HasDownstreamHostConsumer();

                Accelerator * pMostRecent = pBlock->LockForViewSync(bViewMaterializeCandidate);

                // if we are dealing with an in/out consumer pair, this method is responsible for propagating
                // any control signals that must flow along this path, and for materializing any views that are
                // going to be required by downstream consumers. 

                assert(((OutputPort*)pBlock->GetDestinationPort()) == pInOutConsumer);
                if(pInOutConsumer->GetControlPropagationSource() != NULL) {
                    pInOutConsumer->Lock();
                    pBlock->SetControlSignal(pInOutConsumer->GetPropagatedControlSignals());
                    ctlpegress(this, pBlock);
                    pInOutConsumer->ClearAllPropagatedControlSignals();
                    pInOutConsumer->Unlock();
                }

                if(bViewMaterializeCandidate) {
                    MaterializeDownstreamViews(pBlock, pInOutConsumer);
                }

                pInOutConsumer->Push(pBlock);
                pBlock->SetDestinationPort(NULL);
                pInOutConsumer->ReleaseDestinationBuffer();
                pInOutConsumer->ClearAllPropagatedControlSignals();

                pBlock->UnlockForViewSync(bViewMaterializeCandidate, pMostRecent);
            }

            pBlock->Release();

        }
        m_vInflightDatablockMap.clear();
        m_vInflightBlockSet.clear();

        tpprofile_exit(ReleaseInflightDatablocks);
        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets output port map. </summary>
    ///
    /// <remarks>   crossbac, 6/27/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the output port map. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::map<UINT, Port*>* 
    Task::GetOutputPortMap(
        VOID
        )
    {
        return &m_mapOutputPorts;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the input port map. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the input port map. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::map<UINT, Port*>* 
    Task::GetInputPortMap(
        VOID
        )
    {
        return &m_mapInputPorts;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the constant port map. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the input port map. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::map<UINT, Port*>* 
    Task::GetConstantPortMap(
        VOID
        )
    {
        return &m_mapConstantPorts;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the Meta port map. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the input port map. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::map<UINT, Port*>* 
    Task::GetMetaPortMap(
        VOID
        )
    {
        return &m_mapMetaPorts;
    }

    //--------------------------------------------------------------------------------------
    // Find and connect any input/output port pairs that collaborate to 
    // implement inout parameters.
    //--------------------------------------------------------------------------------------
    void 
    Task::ResolveInOutPortBindings(
        VOID
        )
    {
        Lock();
        std::map<UINT, Port*>::iterator mii;
        for(mii=m_mapInputPorts.begin(); mii!=m_mapInputPorts.end(); mii++) {
            InputPort * pInputPort = (InputPort*) mii->second;
            if(pInputPort->IsInOutParameter()) {
                int nOutputPortIndex = pInputPort->GetInOutRoutingIndex();
                std::map<UINT, Port*>::iterator moi = m_mapOutputPorts.find(nOutputPortIndex);
                if(moi == m_mapOutputPorts.end()) {
                    printf("Warning: malformed Task port structure: InOut Input port %s"
                           "has no corresponding output port!\n", pInputPort->GetVariableBinding());
                    assert(false);
                    return;
                } else {
                    OutputPort * pOutputPort = (OutputPort*) moi->second;
                    pInputPort->SetInOutConsumer(pOutputPort);
                    pOutputPort->SetInOutProducer(pInputPort);
                }
            }
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Find and connect any meta/output port pairs that collaborate to implement
    ///             variable length output. This is important because we don't want to pool data
    ///             blocks on outputs that get their allocation sizes from a meta port (because we
    ///             can't do so correctly!)
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::ResolveMetaPortBindings(
        VOID
        )
    {
        Lock();
        std::map<UINT, Port*>::iterator mii;
        for(mii=m_mapMetaPorts.begin(); mii!=m_mapMetaPorts.end(); mii++) {
            MetaPort * pPort = (MetaPort*) mii->second;
            if(pPort->GetMetaFunction() == MF_ALLOCATION_SIZE) {
                int nOutputPortIndex = pPort->GetFormalParameterIndex();
                std::map<UINT, Port*>::iterator moi = m_mapOutputPorts.find(nOutputPortIndex);
                if(moi == m_mapOutputPorts.end()) {
                    printf("Warning: malformed Task port structure: Meta port %s"
                            "has no corresponding output port!\n", pPort->GetVariableBinding());
                    assert(false);
                    return;
                } else {
                    OutputPort * pOutputPort = (OutputPort*) moi->second;
                    pOutputPort->SetAllocatorPort(pPort);
                    pPort->SetAllocationPort(pOutputPort);
                }
            } 
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if there are scheduling constraints for this task 
    ///             that are the result of calls to set affinity. Collecting all
    ///             constraints and choosing amongst available resources can be 
    ///             complex, and there are a few fast-path cases we want to handle
    ///             without requiring the scheduler to examine the task in detail. 
    ///             In particular, if there are mandatory assignments that can 
    ///             satisfied from cached pointers, we want to avoid repeated traversals
    ///             to figure out the same information. If there are constraints
    ///             to satisfy of any form, this method will return true. If there
    ///             are mandatory constraints that have already been discovered, 
    ///             on exit the ppTaskMandatory and ppDepMandatory pointers will be
    ///             set. On exit, a TRUE value for bAssignable indicates that the
    ///             scheduler can proceed directly to trying to acquire those mandatory
    ///             resources. Any other condition means the scheduler has to take the
    ///             full path to examine constraints and choose. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/28/2013. </remarks>
    ///
    /// <param name="ppTaskMandatory">  [in,out] If non-null, the task mandatory. </param>
    /// <param name="ppDepMandatory">   [in,out] If non-null, the dep mandatory. </param>
    /// <param name="bAssignable">      [in,out] The assignable. </param>
    ///
    /// <returns>   true if scheduling constraints, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::HasSchedulingConstraints(
        __out Accelerator ** ppTaskMandatory,
        __out Accelerator ** ppDepMandatory,
        __out BOOL& bAssignable
        )
    {
        if(!Runtime::MultiGPUEnvironment())
            return FALSE;

        Lock();
        bAssignable = FALSE;
        *ppTaskMandatory = NULL;
        *ppDepMandatory = NULL;
        BOOL bConstrained = ((m_pMandatoryAccelerator != NULL) ||
                             (m_vAffinities.size() != 0) ||
                             m_bHasDependentAffinities ||
                             m_bProducesUnmigratableData);

        if(m_pMandatoryAccelerator != NULL) {

            // try to make a fast path assignment. 
            // if there is a mandatory accelerator we can 
            // exit right away, assuming there are no dependent
            // ports that must be assigned, or there are dependent
            // ports whose mandatory assignment is also known.

            if(!HasDependentAcceleratorBindings()) {
                
                // there is no dependent accelerator required,
                // so we can proceed directly to assign the task accelerator
                *ppTaskMandatory = m_pMandatoryAccelerator;
                *ppDepMandatory = NULL;
                bAssignable = TRUE;

            } else {

                if(m_bMandatoryDependentAcceleratorValid && m_pMandatoryDependentAccelerator != NULL) {

                    // dependent accelerators are required, but we know the result
                    // of that constraint search already, so again we can proceed
                    // proceed directly to assign the task accelerator and dependent accs.
                    *ppTaskMandatory = m_pMandatoryAccelerator;
                    *ppDepMandatory = m_pMandatoryDependentAccelerator;
                    bAssignable = TRUE;
                }
            } 

        } else {

            // we dont have a mandatory accelerator set. However, there is still a
            // fast path case: if this is a host task, we can still check if the dependent
            // assignments are known already, if they exist.

            if(GetAcceleratorClass() == ACCELERATOR_CLASS_HOST && !HasDependentAcceleratorBindings()) {
                
                if(m_bMandatoryDependentAcceleratorValid && m_pMandatoryDependentAccelerator != NULL) {

                    // dependent accelerators are required, but we know the result
                    // of that constraint search already, so again we can proceed
                    // proceed directly to assign the task accelerator and dependent accs.
                    *ppTaskMandatory = NULL;
                    *ppDepMandatory = m_pMandatoryDependentAccelerator;
                    bAssignable = TRUE;
                }
            } 

        }

        Unlock();
        return bConstrained;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Collect scheduling constraints. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="vMandatoryConstraints">    [out] non-null, mandatory constraints. </param>
    /// <param name="vPreferences">             [out] non-null, preferred accelerators. </param>
    ///
    /// <returns>   true if there are constraints, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::CollectSchedulingConstraints(
        std::set<Accelerator*> &vMandatoryConstraints, 
        std::set<Accelerator*> &vPreferences
        )
    {
        BOOL bConstraints = FALSE;
        Lock();
        // if there are mandatory scheduling constraints imposed by the programmer,
        // we don't really need to invest the effort to collect preferences, or
        // find constraints that arise dynamically from upstream unmigratable
        // blocks. If the user wants it on a given accelerator, then we put it there.
        if(m_pMandatoryAccelerator != NULL) {
            vMandatoryConstraints.insert(m_pMandatoryAccelerator);
            Unlock();
            return TRUE;
        }
        // discover constraints. 
        map<Accelerator*, AFFINITYTYPE>::iterator ai;
        for(ai=m_vAffinities.begin(); ai!=m_vAffinities.end(); ai++) {
            switch(ai->second) {
            default:
            case AFFINITYTYPE_NONE: 
                break;
            case AFFINITYTYPE_WEAK:
            case AFFINITYTYPE_STRONG:
                // currently lump these together, until we
                // have a scheduler that can tell the difference.
                vPreferences.insert(ai->first);
                break;
            case AFFINITYTYPE_MANDATORY:
                vMandatoryConstraints.insert(ai->first);
                vPreferences.clear();
                Unlock();
                return TRUE;
            }
        }
        // now look for mandatory constraints that arise
        // from upstream unmigratable datablocks. 
        std::map<UINT, Port*>::iterator mii;
        for(mii=m_mapInputPorts.begin(); mii!=m_mapInputPorts.end(); mii++) {
            Port * pPort = mii->second;
            Datablock * pInputBlock = pPort->Peek();
            if(pInputBlock != NULL) {
                pInputBlock->AddRef();
                pInputBlock->Lock();
                if(!pInputBlock->IsMarshallable()) {
                    // we have to schedule this task on
                    // the accelerator that produced this
                    // datablock. 
                    Accelerator * pAccelerator = pInputBlock->GetMostRecentAccelerator();
                    if(pAccelerator != NULL) {
                        // not migratable. 
                        vMandatoryConstraints.insert(pAccelerator);
                        bConstraints = TRUE;
                    }
                }
                pInputBlock->Unlock();
                pInputBlock->Release();
            }
        }
        Unlock();
        if(vMandatoryConstraints.size() > 1) {
            assert(FALSE);
            PTask::Runtime::HandleError("%s:%s constraints are irresovlable!\n",
                                        __FUNCTION__,
                                        m_lpszTaskName);
        }

        // if we are restricted to an accelerator,
        // then we are constrained.
        bConstraints = (vMandatoryConstraints.size() > 0);
        if(vMandatoryConstraints.size() > 1) {
            assert(false);
            Runtime::HandleError("%s: %s: conflicting scheduling constraints!\n",
                                 __FUNCTION__,
                                 m_lpszTaskName);
        }
        return bConstraints;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Populate dependent port set. </summary>
    ///
    /// <remarks>   Crossbac, 11/15/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void            
    Task::PopulateDependentPortSet(
        VOID
        )
    {
        if(m_bDependentPortSetValid)
            return; 

        std::map<UINT, Port*>::iterator pi;
        for(pi=m_mapInputPorts.begin(); pi!=m_mapInputPorts.end(); pi++) 
            if(pi->second->HasDependentAcceleratorBinding()) 
                m_vDependentPorts.insert(pi->second);
        for(pi=m_mapOutputPorts.begin(); pi!=m_mapOutputPorts.end(); pi++) 
            if(pi->second->HasDependentAcceleratorBinding()) 
                m_vDependentPorts.insert(pi->second);
        for(pi=m_mapConstantPorts.begin(); pi!=m_mapConstantPorts.end(); pi++) 
            if(pi->second->HasDependentAcceleratorBinding()) 
                m_vDependentPorts.insert(pi->second);

        m_bDependentPortSetValid = TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Populate caches of scheduling constraints. </summary>
    ///
    /// <remarks>   Crossbac, 11/15/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void            
    Task::PopulateConstraintsCaches(
        VOID
        )
    {
        PopulateDependentPortSet();
        if(!m_bMandatoryConstraintsCacheValid || !m_bPreferenceConstraintsCacheValid) {
            
            std::set<Port*>::iterator si;
            for(si=m_vDependentPorts.begin(); si!=m_vDependentPorts.end(); si++) {
                Port * pPort = *si;
                Accelerator * pAccelerator = pPort->GetMandatoryDependentAccelerator();
                if(pAccelerator != NULL) {
                    m_vMandatoryConstraintMap[pPort] = pAccelerator;
                    m_vMandatoryConstraintSet.insert(pAccelerator);
                } else {
                    std::map<Accelerator*, AFFINITYTYPE> * pAffinities = pPort->GetDependentAffinities();
                    if(pAffinities != NULL && pAffinities->size()) {
                        if(pAffinities->size() > 1) {
                            PTask::Runtime::Warning("Choosing blindly amongst multiple dependent affinities!");
                        }
                        Accelerator * pAccelerator = pAffinities->begin()->first;
                        m_vPreferenceConstraintMap[pPort] = pAccelerator;
                        m_vPreferenceConstraintSet.insert(pAccelerator);
                    }
                }
            }

            if(m_vMandatoryConstraintSet.size()) {
                m_pMandatoryDependentAccelerator = *m_vMandatoryConstraintSet.begin();
                m_bMandatoryDependentAcceleratorValid = TRUE;
            } else {
                m_pMandatoryDependentAccelerator = NULL;
                m_bMandatoryDependentAcceleratorValid = TRUE;
            }

            m_bMandatoryConstraintsCacheValid = TRUE;
            m_bPreferenceConstraintsCacheValid = TRUE;
        }

    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Collect the scheduling constraints for dependent ports on a given task. The task
    ///             itself may have constraints, but when tasks have depenendent ports, affinity may
    ///             be specified for those dependent bindings. This routine collects maps (port-
    ///             >accelerator) of preferred and mandatory scheduling constraints induced by those
    ///             dependent port bindings.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="vMandatoryConstraints">    [out] non-null, mandatory constraints. </param>
    /// <param name="vMandatorySet">            [out] the set the accelerators represented in
    ///                                         the output map vMandatoryConstraints
    ///                                         belongs to. </param>
    /// <param name="vPreferences">             [out] non-null, preferred accelerators. </param>
    /// <param name="vPreferenceSet">           [out] the set of accelerators represented in the
    ///                                         preference map.</param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::CollectDependentSchedulingConstraints(
        __out std::map<Port*, Accelerator*> &vMandatoryConstraintMap, 
        __out std::set<Accelerator*> &vMandatorySet,
        __out std::map<Port*, Accelerator*> &vPreferenceMap,
        __out std::set<Accelerator*> &vPreferenceSet
        )
    {
        BOOL bConstraints = FALSE;
        Lock();
        if(m_bHasDependentAffinities) {

            PopulateDependentPortSet();
            PopulateConstraintsCaches();
            vMandatorySet.insert(m_vMandatoryConstraintSet.begin(), m_vMandatoryConstraintSet.end());
            vMandatoryConstraintMap.insert(m_vMandatoryConstraintMap.begin(), m_vMandatoryConstraintMap.end());
            vPreferenceMap.insert(m_vPreferenceConstraintMap.begin(), m_vPreferenceConstraintMap.end());
            vPreferenceSet.insert(m_vPreferenceConstraintSet.begin(), m_vPreferenceConstraintSet.end());

        }
        Unlock();

        if(vMandatorySet.size() > 1) {
            assert(FALSE);
            PTask::Runtime::HandleError("%s: %s: Dependent port task constraints are irresovlable!\n",
                                        __FUNCTION__,
                                        m_lpszTaskName);
        }
        bConstraints = (vMandatorySet.size() > 0 || vPreferenceSet.size() > 0);
        return bConstraints;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates an asynchronous context for the task. Create the cuda stream for this
    ///             ptask.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 12/20/2011.
    ///             
    ///             This method is required of all subclasses, and abstracts the work associated with
    ///             managing whatever framework-level asynchrony abstractions are supported by the
    ///             backend target. For example, CUDA supports the "stream", while DirectX supports
    ///             an ID3D11ImmediateContext, OpenCL has command queues, and so on.
    ///             </remarks>
    ///
    /// <param name="pAccelerator"> [in] non-null, the CUDA-capable acclerator to which the stream is
    ///                             bound. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::CreateDispatchAsyncContext(
        __in Accelerator * pAccelerator
        )
    {
        return __createDispatchAsyncContext(pAccelerator);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates an asynchronous context for the task. </summary>
    ///
    /// <remarks>   crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pAccelerator"> [in] If non-null, the accelerator. </param>
    /// <param name="vContextMap">  [out] the context map. </param>
    ///
    /// <returns>   true for success, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Task::__createDispatchAsyncContext(
        __in Accelerator * pAccelerator
        ) 
    {
        BOOL bSuccess = FALSE;
        pAccelerator->Lock();

        Lock();

        std::map<Accelerator*, AsyncContext*>::iterator mi;
        mi=m_vDispatchAsyncContexts.find(pAccelerator);
        assert(mi==m_vDispatchAsyncContexts.end());

        if(mi==m_vDispatchAsyncContexts.end()) {

            AsyncContext * pAsyncContext = NULL;
            pAsyncContext = CreateAsyncContext(pAccelerator, ASYNCCTXT_TASK);
            pAsyncContext->AddRef();
            m_vDispatchAsyncContexts[pAccelerator] = pAsyncContext;

        } else {

            PTask::Runtime::MandatoryInform("%s: %s: redundant call to create dispatch async context on %s!\n",
                                            __FUNCTION__,
                                            m_lpszTaskName,
                                            pAccelerator->GetDeviceName());
        }

        Unlock();
        pAccelerator->Unlock();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if dependent bindings on this task expect to have
    ///             platform specific objects (such as device-id, streams, etc) passed
    ///             to it at dispatch time. </summary>
    ///
    /// <remarks>   Crossbac, 7/11/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::DependentBindingsRequirePSObjects(
        VOID
        )
    {
        return m_bRequestDependentPSObjects;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a platform-specific asynchronous context for the task. </summary>
    ///
    /// <remarks>   crossbac, 12/20/2011.
    ///             
    ///             This method is required of all subclasses, and abstracts the work associated with
    ///             managing whatever framework-level asynchrony abstractions are supported by the
    ///             backend target. For example, CUDA supports the "stream", while DirectX supports
    ///             an ID3D11ImmediateContext, OpenCL has command queues, and so on.
    ///             </remarks>
    ///
    /// <param name="pAccelerator">         [in,out] If non-null, the accelerator. </param>
    /// <param name="eAsyncContextType">    Type of the asynchronous context. </param>
    ///
    /// <returns>   null if it fails, else the new async context object. </returns>
    ///-------------------------------------------------------------------------------------------------

    AsyncContext * 
    Task::CreateAsyncContext(
        __in Accelerator * pAccelerator,
        __in ASYNCCONTEXTTYPE eAsyncContextType
        )
    {
        return pAccelerator->CreateAsyncContext(this, eAsyncContextType);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets an async context for the proposed operation. Generally, we prefer to avoid
    ///             scheduling execution and data transfer in the same async context because that
    ///             forces serialization. Often it is necessary serialization (e.g. producer-consumer
    ///             relationship between task execution and data xfer), but often it is not (e.g.
    ///             subsequent task invocations using different data should not wait for transfers on
    ///             data produced by this execution). This method returns an async context in which
    ///             we should schedule a given operation.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pAccelerator">     [in] non-null, the dispatch accelerator. </param>
    /// <param name="eOperationType">   Type of operation. </param>
    /// <param name="pTransferTarget">  (optional) [in,out] If non-null, the transfer target. </param>
    ///
    /// <returns>   null if it fails, else the async context to use for the operation. </returns>
    ///-------------------------------------------------------------------------------------------------

    AsyncContext *
    Task::GetOperationAsyncContext(
        __in Accelerator * pAccelerator,
        __in ASYNCCONTEXTTYPE eAsyncContextType
        )
    {
        // do some basic sanity checking--if the accelerator is null, we've got a serious problem--
        // where are we attempting to queue this operation? If the operation is a kernel execution,
        // then there should be no transfer target. Similarly, if there is a transfer target, we'd
        // better be trying to queue a data transfer. Complain if there is a malformed combination,
        // and then refuse to execute the operation. 
        
        AsyncContext * pAsyncContext = NULL;
        assert(pAccelerator != NULL);
        if(eAsyncContextType == ASYNCCTXT_TASK) {

            // the request is for the lauch context for this task.
            // It should be present in the dispatch async context map.
            // If it is *not* something has gone horribly wrong. 
            // Assuming we find it, it should either be a task context
            // (if the accelerator has an explicit Async API), or a default
            // context (if the accelerator does not support explicit asynchrony).
            
            std::map<Accelerator*, AsyncContext*>::iterator mi;
            mi = m_vDispatchAsyncContexts.find(pAccelerator);
            if(mi == m_vDispatchAsyncContexts.end()) {

                if(!pAccelerator->SupportsExplicitAsyncOperations()) {

                    // no problem, this is a task for an accelerator with no async 
                    // API. just get the default async context object from the 
                    // accelerator and go with that. 
                    // pAsyncContext = pAccelerator->GetAsyncContext(ASYNCCTXT_DEFAULT);
                    pAsyncContext = NULL;

                } else {
                
                    // this port/task combination is bound to something that actually
                    // supports asynchrony. We should have been able to find the context.
                    // assert(FALSE && "missing async context!");
                    PTask::Runtime::MandatoryInform("%s: %s: could not find task async context for %s!\n",
                                                    __FUNCTION__,
                                                    m_lpszTaskName,
                                                    pAccelerator->GetDeviceName());
                }

            } else {

                pAsyncContext = mi->second;
                assert(pAsyncContext != NULL);
                assert((pAsyncContext->GetAsyncContextType() == ASYNCCTXT_TASK && 
                        pAsyncContext->SupportsExplicitAsyncOperations()) ||
                       (pAsyncContext->GetAsyncContextType() == ASYNCCTXT_DEFAULT &&
                        (!pAsyncContext->SupportsExplicitAsyncOperations())));
            }

        } else {

            // the proposed operation is a memory transfer. In which case
            // we defer to the accelerator for an appropriate context, assuming
            // it supports an explicit API for async ops.
            
            pAsyncContext = pAccelerator->GetAsyncContext(eAsyncContextType);
            assert(pAsyncContext != NULL);
            assert((pAsyncContext->SupportsExplicitAsyncOperations() &&
                    ASYNCCTXT_ISXFERCTXT(pAsyncContext->GetAsyncContextType())) ||
                   (!pAsyncContext->SupportsExplicitAsyncOperations() && 
                    ASYNCCTXT_ISDEFAULT(pAsyncContext->GetAsyncContextType())));

        }
        return pAsyncContext;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a synchronization timestamp. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="p">    [in,out] If non-null, the p. </param>
    ///
    /// <returns>   The synchronization timestamp. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Task::GetSynchronizationTimestamp(
        Accelerator * p
        )
    {
        UNREFERENCED_PARAMETER(p);
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a geometry estimator. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="lpfn"> The lpfn. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SetGeometryEstimator(
        LPFNGEOMETRYESTIMATOR lpfn
        ) 
    {
        m_tEstimatorType = USER_DEFINED_ESTIMATOR;
        m_lpfnEstimator = lpfn;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Sets a canonical geometry estimator. </summary>
    ///
    /// <remarks>	crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="t">			  	The type of the port. </param>
    /// <param name="nElemsPerThread">	The elems per thread. </param>
    /// <param name="nGroupSizeX">	  	The group size x coordinate. </param>
    /// <param name="nGroupSizeY">	  	The group size y coordinate. </param>
    /// <param name="nGroupSizeZ">	  	The group size z coordinate. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SetCanonicalGeometryEstimator(
        __in GEOMETRYESTIMATORTYPE t,
        __in int nElemsPerThread,
        __in int nGroupSizeX,
        __in int nGroupSizeY,
        __in int nGroupSizeZ
        ) 
    {
        m_tEstimatorType = t;
        m_nEstimatorElemsPerThread = nElemsPerThread;
        m_nEstimatorGroupSizeX = nGroupSizeX;
        m_nEstimatorGroupSizeY = nGroupSizeY;
        m_nEstimatorGroupSizeZ = nGroupSizeZ;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a geometry estimator. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <returns>   The geometry estimator. </returns>
    ///-------------------------------------------------------------------------------------------------

    LPFNGEOMETRYESTIMATOR 
    Task::GetGeometryEstimator(
        VOID
        ) 
    {
        return m_lpfnEstimator;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a canonical geometry estimator. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="pnElemsPerThread"> [in,out] If non-null, the elems per thread. </param>
    ///
    /// <returns>   The canonical geometry estimator. </returns>
    ///-------------------------------------------------------------------------------------------------

    GEOMETRYESTIMATORTYPE 
    Task::GetCanonicalGeometryEstimator(
        __out int * pnElemsPerThread,
        __out int * pnGroupSizeX,
        __out int * pnGroupSizeY,
        __out int * pnGroupSizeZ
        ) 
    {
        *pnElemsPerThread = m_nEstimatorElemsPerThread;
        *pnGroupSizeX = m_nEstimatorGroupSizeX;
        *pnGroupSizeY = m_nEstimatorGroupSizeY;
        *pnGroupSizeZ = m_nEstimatorGroupSizeZ;
        return m_tEstimatorType;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a dispatch accelerator. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the dispatch accelerator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Task::GetDispatchAccelerator(
        VOID
        ) 
    { 
        return m_pDispatchAccelerator; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an accelerator class. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="cls">  The cls. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SetAcceleratorClass(
        ACCELERATOR_CLASS cls
        ) 
    { 
        m_eAcceleratorClass = cls; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets an accelerator class. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <returns>   The accelerator class. </returns>
    ///-------------------------------------------------------------------------------------------------

    ACCELERATOR_CLASS 
    Task::GetAcceleratorClass(
        VOID
        ) 
    { 
        return m_eAcceleratorClass; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a usage timer. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the usage timer. </returns>
    ///-------------------------------------------------------------------------------------------------

    CHighResolutionTimer *
    Task::GetUsageTimer(
        VOID
        ) 
    { 
        return m_pUsageTimer; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets task-accelerator affinity. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pAccelerator"> [in] the accelerator. </param>
    /// <param name="affinityType"> Type of the affinity. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::SetAffinity(
        Accelerator* pAccelerator, 
        AFFINITYTYPE affinityType
        )
    {        
        BOOL bSuccess = FALSE;
        Lock();
        if(affinityType == AFFINITYTYPE_NONE) {
            if(m_vAffinities.find(pAccelerator) != m_vAffinities.end()) {
                m_vAffinities.erase(pAccelerator);
                bSuccess = TRUE;
            }
        } else {
            bSuccess = TRUE;
            m_vAffinities[pAccelerator] = affinityType;
            if(affinityType == AFFINITYTYPE_MANDATORY) {
                m_pMandatoryAccelerator = pAccelerator;
            }
        }        
        Unlock();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets task-accelerator affinity. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="vAccelerators">    [in,out] non-null, the accelerators. </param>
    /// <param name="pvAffinityTypes">  [in,out] List of types of affinities. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::SetAffinity(
        std::vector<Accelerator*> &vAccelerators, 
        std::vector<AFFINITYTYPE> &pvAffinityTypes
        )
    {
        BOOL bSuccess = TRUE;
        assert(vAccelerators.size() == pvAffinityTypes.size());
        std::vector<Accelerator*>::iterator ai = vAccelerators.begin();
        std::vector<AFFINITYTYPE>::iterator afi = pvAffinityTypes.begin();
        while(ai != vAccelerators.end() && afi != pvAffinityTypes.end()) {
            Accelerator * pAccelerator = *ai;
            AFFINITYTYPE affinityType = *afi;
            if(pAccelerator->GetClass() != GetAcceleratorClass()) {
                assert(FALSE);
                PTask::Runtime::Warning("attempt to affinitize accelerator that cannot run the given task!");
            } else {
                bSuccess &= SetAffinity(pAccelerator, affinityType);
            }
            ai++;
            afi++;
        }
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets task-accelerator affinity for dependent ports. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pAccelerator"> [in] the accelerator. </param>
    /// <param name="affinityType"> Type of the affinity. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::SetDependentAffinity(
        Port * pPort, 
        Accelerator* pAccelerator, 
        AFFINITYTYPE affinityType
        )
    {
        assert(pPort != NULL);
        if(pPort == NULL)
            return FALSE;
        BOOL bSuccess = pPort->SetDependentAffinity(pAccelerator, affinityType);
        m_bHasDependentAffinities |= bSuccess;
        if(m_bHasDependentAffinities && affinityType == AFFINITYTYPE_MANDATORY) {

            // try to cache any mandatory affinities to enable
            // the scheduler to avoid complex constraint solving 
            // if there is only one possible solution. 
            
            if(m_bMandatoryDependentAcceleratorValid &&
               m_pMandatoryDependentAccelerator != NULL && 
               m_pMandatoryDependentAccelerator != pAccelerator) {

                // this mandatory dependent assignment conflicts with the "known" 
                // cached version of the dependent assignment--complain, and then
                // invalidate our cache of this information so the scheduler will 
                // examine all constraints at dispatch time. 
                    
                assert(FALSE);
                PTask::Runtime::MandatoryInform("Conflicting mandatory dependent affinity for %s\n",
                                                m_lpszTaskName);
                m_pMandatoryDependentAccelerator = NULL;
                m_bMandatoryDependentAcceleratorValid= FALSE;

            } else {

                // we know dependent assignments unambigously.
                m_pMandatoryDependentAccelerator = pAccelerator;
                m_bMandatoryDependentAcceleratorValid = TRUE;
            }
        }
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets task-accelerator affinity. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="vAccelerators">    [in,out] non-null, the accelerators. </param>
    /// <param name="pvAffinityTypes">  [in,out] List of types of affinities. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::SetDependentAffinity(
        Port * pPort, 
        std::vector<Accelerator*> &vAccelerators, 
        std::vector<AFFINITYTYPE> &pvAffinityTypes
        )
    {
        assert(pPort != NULL);
        if(pPort == NULL)
            return FALSE;
        BOOL bSuccess = pPort->SetDependentAffinity(vAccelerators, pvAffinityTypes);
        m_bHasDependentAffinities |= bSuccess;
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a mandatory accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the mandatory accelerator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Task::GetMandatoryAccelerator(
        VOID
        )
    {
        return m_pMandatoryAccelerator;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind all meta ports. Metaports do not bind to variables in shader code, but
    ///             provide information for the runtime about other ports. We envision multiple use
    ///             cases, but the only one currently supported is where a datablock on a metaport
    ///             provides an integer- valued allocation size for another output port on the ptask.
    ///             Hence, this function looks at all metaports, and performs output datablock
    ///             allocation as needed.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Task::BindMetaPorts(
        VOID
        ) 
    {
        size_t nMetaPorts = m_mapMetaPorts.size();
        if(nMetaPorts == 0)
            return TRUE;

        tpprofile_enter(BindMetaPorts);        
        vector<Port*>::iterator vi;
        for(vi=m_vMetaPortDispatchOrder.begin(); 
            vi!=m_vMetaPortDispatchOrder.end(); 
            vi++) {
            
            // get the meta port, and perform it's meta function. 
            // MetaPort::PerformMetaFunction assumes the dispatch 
            // accelerator is locked. 
            MetaPort* pPort = reinterpret_cast<MetaPort*>(*vi);
            critical_path_check_binding_trace("%s:BindMeta(%s)\n", m_lpszTaskName, pPort->GetVariableBinding());
            OutputPort * pAllocationPort = reinterpret_cast<OutputPort*>(pPort->GetAllocationPort());
            Accelerator * pPortBindAccelerator = m_pDispatchAccelerator;
            if(pAllocationPort && pAllocationPort->HasDependentAcceleratorBinding()) 
                pPortBindAccelerator = GetAssignedDependentAccelerator(pAllocationPort);
            pPort->PerformMetaFunction(pPortBindAccelerator);
        }
        tpprofile_exit(BindMetaPorts);
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   When the graph is complete, (indicated because Graph.Run was called), this method
    ///             is called on every task to allow tasks to perform and one-time initializations
    ///             that cannot be performed without knowing that the structure of the graph is now
    ///             static. The base class method performs some operations such as sorting metaports
    ///             and then calls the platform specific version, which is required of every 
    ///             subclass, since there is some diversity in what is required here. For example,
    ///             in CUDA we must compute parameter byte offsets; in OpenCL, we do nothing...etc.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/5/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::OnGraphComplete(
        VOID
        )
    {
        Lock();
        SortMetaPorts();         // find a consistent order for meta port traversal
        AssemblePortLockSets();  // let ports find a sort order for upstream channels
        AssembleIOLockList();    // find a sort order for all graph objects required for dispatch
        PlatformSpecificOnGraphComplete();
        Unlock();
    }

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Executes the worker thread alive action. </summary>
	///
	/// <remarks>	crossbac, 8/13/2013. </remarks>
	///
	/// <param name="dwThreadId">	Identifier for the thread. </param>
	///-------------------------------------------------------------------------------------------------

	void
	Task::OnWorkerThreadEntry(
		DWORD dwThreadId
		)
	{
		Lock();
		InvokeStaticInitializers(dwThreadId);
		Unlock();
	}

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Executes the static initializers on a different thread, and waits for the result.
	/// 			</summary>
	///
	/// <remarks>	crossbac, 8/13/2013. </remarks>
	///
	/// <param name="dwThreadId">	Identifier for the thread. </param>
	///-------------------------------------------------------------------------------------------------

	void
	Task::InvokeStaticInitializers(
		DWORD dwThreadId
		)
	{
		if(m_pCompiledKernel->HasStaticInitializer()) {
			if(m_pCompiledKernel->InitializerRequiresPSObjects()) {
				std::set<Accelerator*> vAccelerators;
				ACCELERATOR_CLASS eClass = m_pCompiledKernel->GetInitializerRequiredPSClass();
				Scheduler::FindEnabledCapableAccelerators(eClass, vAccelerators);
				m_pCompiledKernel->InvokeInitializer(dwThreadId, vAccelerators);
			} else {
                assert(FALSE);
                printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
                printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
                printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
                printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
				m_pCompiledKernel->InvokeInitializer(dwThreadId);
			}
		}
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sorts meta ports into an order that guarantees they can perform their operations
    ///             at dispatch time. Usually, the metaport visitation order does not matter. However,
    ///             when there are multiple allocator ports, bound to output ports that collaborate
    ///             in a descriptor port relationship, we cannot allocate the block until all the
    ///             metaports involved have determined the channel sizes for the block. Consequently,
    ///             in this situation, we must be sure to process all descriptor ports before non-
    ///             descriptor ports. This condition is sufficient to ensure that we can always
    ///             perform the block allocation at the described port once we arrive, and since the
    ///             order does not change once the graph is running, we perform this operation as
    ///             part of OnGraphComplete(), which is invoked just before the graph enters the
    ///             RUNNING state.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 2/15/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::SortMetaPorts(
        VOID
        )
    {
        Lock();
        assert(m_vMetaPortDispatchOrder.size() == 0);
        map<UINT, Port*>::iterator mi;        
        vector<Port*>::iterator vi;
        vector<Port*> vSortTail;
        for(mi=m_mapMetaPorts.begin(); mi!= m_mapMetaPorts.end(); mi++) {
            BOOL bAdded = FALSE;
            MetaPort* pMPort = reinterpret_cast<MetaPort*>(mi->second);
            if(pMPort->GetMetaFunction() == MF_ALLOCATION_SIZE) {
                OutputPort * pOPort = reinterpret_cast<OutputPort*>(pMPort->GetAllocationPort());
                if(pOPort->IsDescriptorPort()) {
                    m_vMetaPortDispatchOrder.push_back(pMPort);
                    bAdded = TRUE;
                }
            }
            if(!bAdded) {
                vSortTail.push_back(pMPort);
            }
            // if this port performs allocation for output ports with other output descriptor ports, make a
            // note of it: the information is only needed at dispatch time, but since it's non-trivial to
            // compute and requires graph traversal, we pre-compute it before running the graph.
            pMPort->FindCollaboratingMetaPorts();
        }
        for(vi=vSortTail.begin(); vi!=vSortTail.end(); vi++)
            m_vMetaPortDispatchOrder.push_back(*vi);
        Unlock();
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Acquires any locks required to deal with incoming asynchronous dependences. 
    ///             If the block has outstanding dependences we may require an accelerator lock
    ///             that would not otherwise be required, and since we must acquire accelerator
    ///             locks up front, we have to check this condition in advance. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///
    /// <param name="pBlock">                   [in,out] If non-null, the block. </param>
    /// <param name="pDispatchAccelerator">     [in,out] If non-null, the dispatch accelerator. </param>
    /// <param name="pBlockTargetAccelerator">  [in,out] If non-null, the block target accelerator. </param>
    /// <param name="eProposedBlockOperation">  The proposed block operation. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    Task::AcquireIncomingAsyncDependenceLocks(
        __in Datablock * pBlock,
        __in Accelerator * pDispatchAccelerator,
        __in Accelerator * pBlockTargetAccelerator,
        __in ASYNCHRONOUS_OPTYPE eProposedBlockOperation
        )
    {
        assert(pBlock != NULL);
        assert(pBlock->LockIsHeld());
        assert(pDispatchAccelerator != NULL);
        assert(pBlockTargetAccelerator != NULL);

        // if neither the block's target accelerator (the accelerator bound to the port, which can
        // differ from the task's dispatch accelerator when there are dependent ports), nor the task
        // dispatch accelerator support an explicit asynchronous API, then we may need to lock
        // accelerators with outstanding operations on this block to enable a backend framework call to
        // allow those operations to complete. PTask has several modes for dealing with this kind of
        // dependence, so PTask::Runtime settings may make this check unnecessary. 
        
        if(!PTask::Runtime::GetTaskDispatchLocksIncomingAsyncSources())
            return;

        if(!pBlockTargetAccelerator->SupportsExplicitAsyncOperations() && 
            !pDispatchAccelerator->SupportsExplicitAsyncOperations() &&
            pBlock->HasOutstandingAsyncDependences(pBlockTargetAccelerator, eProposedBlockOperation)) {

            // this block has outstanding operations queued asynchronously that require a wait
            // before an operation of the proposed type can begin. Find the accerators in whose 
            // contexts all such operations were queued (there can be multiple if there are readers
            // on a block and this operation is a write) and add them to the lock list. 

            std::set<Accelerator*>::iterator sadi;
            std::set<Accelerator*>* pDepAccs = NULL;
            pDepAccs = pBlock->GetOutstandingAsyncAcceleratorsPointer(pBlockTargetAccelerator, 
                                                                      eProposedBlockOperation);

            if(pDepAccs != NULL) {
                for(sadi=pDepAccs->begin(); sadi!=pDepAccs->end(); sadi++) {
                    m_vRequiredAcceleratorLocks.insert(*sadi);
                    m_vOutstandingAsyncOpAccelerators.insert(*sadi);
                    assert((*sadi)->SupportsExplicitAsyncOperations());
                }
            }
        }
    }

    static BOOL gbCtxChange = FALSE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Acquire locks on all resources that will need to be locked during task dispatch.
    ///             We acquire these locks all at once because we need to do it in order to prevent
    ///             deadlock, making it difficult to defer acquisition for some objects until the
    ///             time the resource is actually accessed. The following resources must be locked
    ///             for dispatch:
    ///             
    ///             1. The dispatch accelerator.   
    ///             2. Any datablocks that will be written by accelerator code during dispatch.  
    ///             3. Any datablocks that require migration--that is, any datablock for  
    ///                that will be bound as input or in/out whose most recent view exists on an
    ///                accelerator other than the dispatch accelerator.
    ///             
    ///             Note that this does not include datablocks whose view is either up to date, or
    ///             whose most recent view is in host memory. These blocks can be locked as
    ///             encountered and unlocked after updates are performed because we do not need any
    ///             locks other than the dispatch accelerator lock and the datablock lock to perform
    ///             these operations. And any concurrent writers attempting to write such blocks will
    ///             hold locks for the duration of dispatch, which will block updates of shared views.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void        
    Task::AcquireDispatchResourceLocks(
        VOID
        )
    {
        tpprofile_enter(AcquireDispatchResourceLocks);
        MARKRANGEENTER(L"AcquireDispatchResourceLocks");

        // if the last dispatch left us some blocks, that's a bug

        assert(m_vInputDatablockLocks.size() == 0);
		assert(m_vRequiredDatablockLocks.size() == 0);
        assert(m_vRequiredAcceleratorLocks.size() == 0);
        assert(m_vBlocksRequiringMigration.size() == 0);
        assert(m_vBlindWriteInflightBlocks.size() == 0);
        assert(m_vWriteableInflightBlocks.size() == 0);
        assert(m_vOutstandingAsyncOpAccelerators.size() == 0);

        // the dispatch accelerator lock is always required
        // for dispatch, so add it. If it's null, something
        // has gone very very wrong. 

        assert(m_pDispatchAccelerator != NULL);
        m_vRequiredAcceleratorLocks.insert(m_pDispatchAccelerator);

        // look at all our inputs by peeking every input port. peek can return null when we are ready
        // for dispatch if the port is an initializer port, but we will never have to migrate a block
        // received on an initializer port. Every block that requires migration must be added
        // to the list of required dispatch locks, along with every accelerator that will be involved
        // in migration. We also add any blocks that are part of an inout pair because we need exclusive
        // permissions on those blocks and have to hold the until dispatch completes.

        map<UINT, Port*>::iterator mi;
        for(mi=m_mapInputPorts.begin(); mi!= m_mapInputPorts.end(); mi++) {

			InputPort* pPort = dynamic_cast<InputPort*>(mi->second);
            Port * pInOutConsumer = pPort->GetInOutConsumer();
            Datablock * pBlock = pPort->Peek();
            if(pBlock == NULL) continue; // initializer port!

            pBlock->Lock();
            m_vInputDatablockLocks[pBlock] = pPort;
            m_vRequiredDatablockLocks.insert(pBlock);
            ASYNCHRONOUS_OPTYPE eOpType = pInOutConsumer != NULL ? 
                OT_LAUNCH_TARGET_WRITABLE : OT_LAUNCH_TARGET_READABLE;
            if(pInOutConsumer != NULL) m_vWriteableInflightBlocks.insert(pBlock);
            Accelerator * pTargetAccelerator = GetTargetMemorySpace(pPort);

            if(pBlock->RequiresMigration(pTargetAccelerator)) {

				// if the block is up-to-date elsewhere (where elsewhere means another accelerator) we must
				// lock the block and the accelerator we are going to migrate from. moreover, if the target is
				// not the dispatch accelerator, because this block is bound to a port with a dependent
				// accelerator, we need to add the target accelerator to the list. 

				assert(m_vBlocksRequiringMigration.find(pBlock) == m_vBlocksRequiringMigration.end());
                m_vBlocksRequiringMigration[pBlock] = pTargetAccelerator; 
                Accelerator * pSourceAccelerator = pBlock->GetMostRecentAccelerator();
                assert(pSourceAccelerator != NULL);
                m_vRequiredAcceleratorLocks.insert(pSourceAccelerator);
				if(pTargetAccelerator != m_pDispatchAccelerator)
					m_vRequiredAcceleratorLocks.insert(pTargetAccelerator);

            } else {

                // even if the block doesn't require migration, we may need an
                // accelerator lock to perform a blocking wait on an outstanding depenence. 
                // This can happen if no resources on the current task have contexts that are
                // capable of async operations (e.g. it's a pure host task), but an upstream
                // task that did support async pushed a block with outstanding dependences. 
                // We will need to wait synchronously in such cases, which will require accelerator
                // locks. Find all such Accelerators and make sure we can track and lock them. 

                AcquireIncomingAsyncDependenceLocks(pBlock, m_pDispatchAccelerator, pTargetAccelerator, eOpType);
            }

            pBlock->Unlock();			
        }

        // Now look at all our output ports. We require locks on all written blocks.  
        // Note that we never check for migration here: if a block on an output port
        // requires migration its because it was part of an inout pair, and was therefore
        // handled by the input port map traversal. 

        for(mi=m_mapOutputPorts.begin(); mi!= m_mapOutputPorts.end(); mi++) {

            OutputPort* pPort = dynamic_cast<OutputPort*>(mi->second);
            Accelerator * pTargetAccelerator = GetTargetMemorySpace(pPort);

            if(pPort->GetInOutProducer() == NULL) {
                
                // If this port is *not* part of an in/out pair, we need to lock the block on this output port.
                // There is one caveat here: if this port has an allocator port (an upstream meta-port), then
                // the block we would lock hasn't been allocated yet because allocation doesn't occur until the
                // metaport bind time, which is *after* lock acquisition. Fortunately, if the block isn't
                // allocated yet, then it is technically private until it is pushed into downstream channels,
                // which is after dispatch. In short, we don't need locks for such blocks because they are
                // unreachable from other threads until after dispatch. Worse, if we call GetDestinationBuffer
                // on a port with an upstream allocator, we can wind up performing an allocation for something
                // that is just released when the metaport gets bound! Hence, we perform this work only for
                // ports without an upstream allocator.
                // ------------------------------------
                // FIXME: TODO: Such blocks *will* wind up in flight, and will be blind written. we need to add
                // them to those lists once the allocation is actually performed.
                // ------------------------------------. 
                
                if(pPort->GetAllocatorPort() == NULL) { 
                    Datablock * pDestBlock = pPort->GetDestinationBuffer(pTargetAccelerator);
                    m_vWriteableInflightBlocks.insert(pDestBlock);
                    m_vBlindWriteInflightBlocks.insert(pDestBlock);
                    m_vRequiredDatablockLocks.insert(pDestBlock);
					if(pTargetAccelerator != m_pDispatchAccelerator) {
						m_vRequiredAcceleratorLocks.insert(pTargetAccelerator);
					}
                }

            } else {

                // if this port is the out in an in/out pair, then the block it will handle would be locked in
                // the traversal of the input ports. Moreoever, since no blocks have been bound yet, the block
                // on the destination buffer is *not* the one that will be put there when we finally call bind
                // inputs. Hence, there are no meaningful invariants to check about the destination buffer for
                // this output port. 

            }
        }

        // if this task has dependent accelerators (indicating it requires more than
        // one execution context), all of its dependent accelerators must be acquired too.
        Lock();
        for(int i=0; i<GetDependentBindingClassCount(); i++) {
            Accelerator * pDepAcc = GetAssignedDependentAccelerator(i);
            assert(pDepAcc != NULL);
            m_vRequiredAcceleratorLocks.insert(pDepAcc);
        }
        Unlock();

        // Note we never need to migrate a block on a constant port or lock it 
        // for the duration of dispatch. This is because the contract with accelerator-side
        // code is that such data will never be written and therefore no view will ever
        // be invalidated. Most accelerators actually can't violate this contract, but
        // HostAccelerator dispatches can...achtung!
        // -----------------------------------------
        // We have an exhaustive list of everything we'll need to lock for dispatch
        // of this task, and it's already in sorted order because we use std::set.
        // Lock ordering discipline is Accelerator->Datablock, so first acquire all the
        // Accelerator locks and then all the Datablock locks.
        std::set<Accelerator*, compare_accelerators_by_memspace_id>::iterator ai;
        std::set<Datablock*>::iterator di;
        std::map<Datablock*, Port*>::iterator dpi;
        BOOL bLockSuccess = FALSE;

        while(!bLockSuccess) {

            // there is a fundamental challenge here, in that the lock-ordering discipline
            // for datablocks and accelerators requires we lock accelerators first. which means
            // that we have to release locks on the blocks that we used to compute the set of
            // locks we need. A consequence of this is that the blocks can change state out from
            // under us by the time we get to this phase--for example a block that required migration
            // might no longer require it because another task performed the migration already,
            // or vice-versa. So as we acquire locks, we update our view of the state, and if 
            // something has changed, we release all the locks, add any missing locks, and retry.
            // The lock sets are monotonically increasing as a result, so we may acquire locks
            // we do not actually need--however, this does allow us to guarantee forward progress.
            
            BOOL bMissedAcceleratorLock = FALSE;
            std::set<Accelerator*> vMissingLocks;
            for(ai=m_vRequiredAcceleratorLocks.begin(); ai!=m_vRequiredAcceleratorLocks.end(); ai++) 
                (*ai)->Lock();
            for(di=m_vRequiredDatablockLocks.begin(); di!=m_vRequiredDatablockLocks.end(); di++) {
                Datablock * pBlock = *di;
                dpi = m_vInputDatablockLocks.find(pBlock);
                pBlock->Lock();

                if(dpi!=m_vInputDatablockLocks.end()) {                

                    InputPort * pPort = (InputPort*)dpi->second;                    
                    ASYNCHRONOUS_OPTYPE eOpType = pPort->GetInOutConsumer() != NULL ? 
                        OT_LAUNCH_TARGET_WRITABLE : OT_LAUNCH_TARGET_READABLE;
                    Accelerator * pTargetAccelerator = GetTargetMemorySpace(pPort);
                    if(pBlock->RequiresMigration(pTargetAccelerator)) {

                        // if the block requires migration, and it did not previously, we may be
                        // missing an accelerator lock for the source accelerator. Add it to the
                        // set of missing locks, and put the block on the migration list. 

                        Accelerator * pSource = pBlock->GetMostRecentAccelerator();
                        if(m_vRequiredAcceleratorLocks.find(pSource) == m_vRequiredAcceleratorLocks.end()) {
                            bMissedAcceleratorLock = TRUE;
                            vMissingLocks.insert(pSource);
                        }
                        m_vBlocksRequiringMigration[pBlock] = pTargetAccelerator;

                    } else {

                        // if the block does not require migration, and we previously thought
                        // it did, remove it from the list so we can avoid double-migrating
                        // blocks and other needless work. Note that we do not make any attempt
                        // to shrink the lock sets in response to this case. Re-do the check
                        // for incoming dependences that require additional locks. 
                        
                        std::map<Datablock*, Accelerator*>::iterator brmi;
                        brmi = m_vBlocksRequiringMigration.find(pBlock);
                        if(brmi != m_vBlocksRequiringMigration.end()) {
                            m_vBlocksRequiringMigration.erase(brmi);
                        }

                        AcquireIncomingAsyncDependenceLocks(pBlock, m_pDispatchAccelerator, pTargetAccelerator, eOpType);
                    }
                }
            }           

            bLockSuccess = !bMissedAcceleratorLock;
            if(bMissedAcceleratorLock) {
                
                // if we missed a required lock, we need to retry. release all the datablock
                // locks and accelerator locks, add all entries in the missing lock set to
                // the required acccelerator lock set, and retry
                for(di=m_vRequiredDatablockLocks.begin(); di!=m_vRequiredDatablockLocks.end(); di++) 
                    (*di)->Unlock();
                for(ai=m_vRequiredAcceleratorLocks.begin(); ai!=m_vRequiredAcceleratorLocks.end(); ai++) 
                    (*ai)->Unlock();
                std::set<Accelerator*>::iterator msi;
                for(msi=vMissingLocks.begin(); msi!=vMissingLocks.end(); msi++)
                    m_vRequiredAcceleratorLocks.insert(*msi);
            }
        }

        m_pDispatchAccelerator->MakeDeviceContextCurrent();

        MARKRANGEEXIT();
        tpprofile_exit(AcquireDispatchResourceLocks);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the dispatch resource locks aqcuired
    /// 			in the call to AcquireDispatchResourceLocks, but which were
    /// 			not already released during another phase of dispatch.
    /// 			It should be the case that by releasing all the blocks on the exclusive
    /// 			inflight list we arrive at an empty required list because any blocks
    /// 			whose locks were required for migration but needing only shared state
    /// 			access should have been released after migration. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void        
    Task::ReleaseDispatchResourceLocks(
        VOID
        )
    {
        MARKRANGEENTER(L"ReleaseDispatchResourceLocks");
        tpprofile_enter(ReleaseDispatchResourceLocks);

        m_pDispatchAccelerator->ReleaseCurrentDeviceContext();

        std::set<Datablock*>::iterator di;
        m_vBlindWriteInflightBlocks.clear();
        for(di=m_vRequiredDatablockLocks.begin(); di!=m_vRequiredDatablockLocks.end(); di++) {
            Datablock * pDatablock = (*di);
            assert(pDatablock->LockIsHeld());
            pDatablock->Unlock();
        }
        m_vWriteableInflightBlocks.clear();
        m_vRequiredDatablockLocks.clear();
        m_vInputDatablockLocks.clear();
        
        // accelerators other than the dispatch accelerator should also have been removed after
        // migration--we don't want to hold locks on other accelerators while we execute, unless they
        // are locked becausee the task has "dependence" on them. (E.g. host tasks using CUDA). 
        assert(m_vRequiredAcceleratorLocks.size() == 1 || m_vDependentAcceleratorAssignments.size() > 0);
        std::set<Accelerator*, compare_accelerators_by_memspace_id>::iterator ai;
        for(ai=m_vRequiredAcceleratorLocks.begin();ai!=m_vRequiredAcceleratorLocks.end(); ai++) {
            Accelerator * pAccelerator = *ai;
            assert(pAccelerator->LockIsHeld());
            pAccelerator->Unlock();
        }
        m_vRequiredAcceleratorLocks.clear();
        tpprofile_exit(ReleaseDispatchResourceLocks);
        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Early release for exclusive dispatch access to pAccelerator. </summary>
    ///
    /// <remarks>   Crossbac, 1/5/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    Task::ReleaseExclusiveDispatchAccess(
        Accelerator * pAccelerator
        )
    {
        // this must have been an accelerator used to migrate!
        assert(pAccelerator != NULL);
        assert(pAccelerator != m_pDispatchAccelerator);
        assert(m_vRequiredAcceleratorLocks.size() > 1);
        assert(m_vRequiredAcceleratorLocks.find(pAccelerator) != m_vRequiredAcceleratorLocks.end());
        m_vRequiredAcceleratorLocks.erase(pAccelerator);
        pAccelerator->Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Perform the actual data transfers associated with datablock migration. All such
    ///             blocks and the accelerator objects required to complete the migration should be
    ///             locked already. The work of migration is performed by the source accelerator.
    ///             Accelerator subclasses can implement Migrate if their underlying APIs have
    ///             support for transfer paths other than through host memory. If the source
    ///             accelerator subclass does not implement migrate, the default implemention
    ///             Accelerator::Migrate will handle transfering data through host memory.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    Task::MigrateInputs(
        VOID
        )
    {
        tpprofile_enter(MigrateInputs);

        std::set<Datablock*>::iterator di;
        std::map<Datablock*, Accelerator*>::iterator mi;

        for(mi=m_vBlocksRequiringMigration.begin(); mi!=m_vBlocksRequiringMigration.end(); mi++) {

            // look at the block and verify that its lock is held and that it has a valid need to migrate--
            // then call pBlock->Migrate, which will defer to any custom migration implementations provided
            // by the accelerator subclasses involved. If the block is on the exclusive dispatch list, that
            // means that it will be written during dispatch so we need to request exclusive permissions,
            // and we need to retain the lock on the block. Conversely, if it is not on the exclusive
            // dispatch list, we can request shared permissions, and release the lock on the block. 
            
            Datablock * pBlock = mi->first;
			Accelerator * pDispatchAccelerator = mi->second;
            assert(pDispatchAccelerator != NULL);
            assert(pDispatchAccelerator->LockIsHeld());
            assert(pBlock->LockIsHeld());
            Accelerator * pSourceAccelerator = pBlock->GetMostRecentAccelerator();
            assert(pSourceAccelerator != NULL);
			if(!pSourceAccelerator->LockIsHeld()) {
				assert(pSourceAccelerator->LockIsHeld());            
			}

            BOOL bExclusive = m_vWriteableInflightBlocks.find(pBlock) != m_vWriteableInflightBlocks.end();
            BUFFER_COHERENCE_STATE uiRequestedPerms = bExclusive ? BSTATE_EXCLUSIVE : BSTATE_SHARED;
            pBlock->Migrate(pDispatchAccelerator, 
                            pSourceAccelerator, 
                            GetOperationAsyncContext(pDispatchAccelerator, ASYNCCTXT_XFERDTOD),                            
                            uiRequestedPerms);
            
        }

        // CJR: bugfix 10/25/2012:  empty the blocks requiring migration list here. This was previously
        // done in ReleaseExclusiveDispatchAccess(), but it is possible that a block was migrated
        // successfully but stayed on the exclusive dispatch lock-list. Most notably, if a block is
        // migrated into exclusive state, we must hold the lock until dispatch completes, the block
        // will will not be on the releasable list, which leaves the block on the migrate list across
        // dispatches, causing havoc. Hence, empty the migrate list before releasing any locks. 

        m_vBlocksRequiringMigration.clear();

        // release any accelerator locks other than the dispatch accelerator and any dependent
        // accelerator assignements made on specific ports. we have no reason to bind up those
        // resources now that migration is complete, unless the target memory space has a null async
        // context, in which case we must hold the locks until after all input bindings are complete. 

        std::set<Accelerator*, compare_accelerators_by_memspace_id>::iterator ai;
        std::set<Accelerator*, compare_accelerators_by_memspace_id> vReleaseableAccelerators;
        for(ai=m_vRequiredAcceleratorLocks.begin(); ai!=m_vRequiredAcceleratorLocks.end(); ai++) {
            Accelerator * pAccelerator = (*ai);
            if(pAccelerator != m_pDispatchAccelerator && 
               m_vOutstandingAsyncOpAccelerators.find(pAccelerator) == m_vOutstandingAsyncOpAccelerators.end() &&
               m_vAllDependentAssignments.find(pAccelerator) == m_vAllDependentAssignments.end()) {

                // this accelerator is locked because it was a migration source, *and* it is not assigned to a
                // dependent port. We can release the lock now iff the dispatch accelerator has a an async
                // context that supports real async commands, because in that case we can use *it* to wait for
                // the copy commands to complete without re-acquiring the source accelerator lock. Otherwise,
                // we must defer the lock release until after we have bound all inputs, because we will require
                // the lock to drain the source stream, and re-acquiring these locks later risks deadlock. TODO:
                // technically, we could actually look for cases where we have another async context that we
                // can use among the dependent assignments...

                if(m_pDispatchAccelerator->SupportsExplicitAsyncOperations()) {
                    vReleaseableAccelerators.insert(pAccelerator);
                } else {
                    m_vMigrationSources.insert(pAccelerator);
                }
            }
        }

        // ...and release them.
        for(ai=vReleaseableAccelerators.begin(); ai!=vReleaseableAccelerators.end(); ai++) {
            Accelerator * pAccelerator = (*ai);
            ReleaseExclusiveDispatchAccess(pAccelerator);
        }

        assert(m_vRequiredAcceleratorLocks.size() == 1 || 
               m_vDependentAcceleratorAssignments.size()>0 ||
               m_vOutstandingAsyncOpAccelerators.size()>0 ||
               m_vMigrationSources.size() > 0);

        //BOOL bEarlyRelease = TRUE;
        //for(ai=m_vMigrationSources.begin(); ai!=m_vMigrationSources.end(); ai++) {
        //    Accelerator * pAccelerator = (*ai);
        //    if(pAccelerator->GetClass() != ACCELERATOR_CLASS_DIRECT_X)
        //        bEarlyRelease = FALSE;
        //}
        //if(bEarlyRelease) {
        //    ReleaseMigrationSourceLocks();
        //}

        tpprofile_exit(MigrateInputs);
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Release any locks that were required *only* to deal with incoming asynchronous 
    ///             dependences. If we needed an accelarator lock to make a backend frame work call
    ///             to wait for outanding operations on a block to complete, but did not need it
    ///             for any other reason release it. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void        
    Task::ReleaseIncomingAsyncDependenceLocks(
        VOID
        )
    {
        assert(PTask::Runtime::GetTaskDispatchLocksIncomingAsyncSources() || 
               m_vOutstandingAsyncOpAccelerators.size() == 0);

        std::set<Accelerator*>::iterator ai;
        for(ai=m_vOutstandingAsyncOpAccelerators.begin(); ai!=m_vOutstandingAsyncOpAccelerators.end(); ai++) {
            Accelerator * pAccelerator = *ai;
            if(pAccelerator != m_pDispatchAccelerator && 
               m_vMigrationSources.find(*ai) == m_vMigrationSources.end() &&
               m_vAllDependentAssignments.find(pAccelerator) == m_vAllDependentAssignments.end()) {
                ReleaseExclusiveDispatchAccess(pAccelerator);
            }
        }
        m_vOutstandingAsyncOpAccelerators.clear();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases any locks held on accelerators that were taken because the 
    ///             accelerator was a migration source. When the dispatch accelerator (or
    ///             dependent port accelerator) has a non-null async context, we can release
    ///             these locks after queing the copy, because we can make the target wait
    ///             for the source without re-acquiring the lock on the source accelerator. 
    ///             When there is a null async context, waiting for those copies requires us
    ///             to drain the source command queue for the given stream, so if we do not 
    ///             hold the lock until this is done, we risk deadlock for such cases. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/28/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::ReleaseMigrationSourceLocks(
        VOID
        )
    {
        std::set<Accelerator*>::iterator ai;
        for(ai=m_vMigrationSources.begin(); ai!=m_vMigrationSources.end(); ai++) {
            Accelerator * pAccelerator = (*ai);
            assert(m_pDispatchAccelerator != pAccelerator);
            assert(m_vAllDependentAssignments.find(pAccelerator) == m_vAllDependentAssignments.end());
            ReleaseExclusiveDispatchAccess(pAccelerator);
        }
        m_vMigrationSources.clear();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a block to the inflight list. </summary>
    ///
    /// <remarks>   Crossbac, 1/5/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    Task::AddToInflightList(
        Port * pInputSourcePort,
        Datablock * pBlock
        )
    {
        if(m_vInflightBlockSet.find(pBlock) != m_vInflightBlockSet.end()) {

            // this block is already bound to another input port. Generally speaking this is super-wierd.
            // However, it's not outright insane, and as long as the multiple binding does not induce any
            // read/write conflicts, then there is no real correctness danger (as far as PTask can tell.)
            // Check for that condition and complain if there is a problem. If the port we are currently
            // binding is a writer, we already know there is a problem.
                       
            InputPort * pConflictingPort = NULL;
            InputPort * pInputPort = reinterpret_cast<InputPort*>(pInputSourcePort);
            BOOL bReadWriteHazard = pInputPort != NULL && pInputPort->GetInOutConsumer();

            if(!bReadWriteHazard) {

                // traverse the existing bindings. The presence of any 
                // writer means we've got a semantic hazard to report
                
                std::map<Port*, Datablock*>::iterator mi;
                for(mi=m_vInflightDatablockMap.begin(); mi!=m_vInflightDatablockMap.end(); mi++) {
                    pConflictingPort = reinterpret_cast<InputPort*>(mi->first);
                    bReadWriteHazard = pConflictingPort && pConflictingPort->GetInOutConsumer();
                    if(bReadWriteHazard) 
                        break;
                }
            }

            if(bReadWriteHazard) {
                PTask::Runtime::MandatoryInform("XXXX Task:%s->Port:%s,%s: Read Write hazard on same block instance!!!!\n",
                                                m_lpszTaskName,
                                                pInputPort->GetVariableBinding(),
                                                pConflictingPort->GetVariableBinding());
            }
        }

        assert(m_vInflightDatablockMap.find(pInputSourcePort) == m_vInflightDatablockMap.end());
        m_vInflightDatablockMap[pInputSourcePort] = pBlock;
        m_vInflightBlockSet.insert(pBlock);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Force sealed. </summary>
    ///
    /// <remarks>   crossbac, 5/2/2013. </remarks>
    ///
    /// <param name="pDestBlock">   [in,out] If non-null, destination block. </param>
    /// <param name="pPort">        [in,out] If non-null, the port. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Task::ForceSealed(
        Datablock * pDestBlock,
        Port * pPort
        )
    {
        // Some blocks require "sealing". We should probably assert here, and force the programmer to
        // fix their code if we encounter an unsealed block. However, it is easily fixed. This can
        // occur under the following conditions: if we need to write to an output channel channel other
        // than data, the only way we can tell we must allocate buffers additional channel buffers is
        // by examining the block allocated on the port.
        // --------------------------------------------------------------
        // we can tell there will be a metadata channel from a few things. 
        // a. the destination block explicitly has such a channel (the buffer exists)
        // b. the output port template describes a record stream
        // --------------------------------------------------------------
        // Currently, we will fail if we encounter an output that requires a template channel. This is
        // bad, and we need a better way to handle this big-time.
        // --------------------------------------------------------------
        // XXX:TODO:FIXME: we should not be guessing when we need to allocate non-data channels on
        // internal channels! 

        assert(pPort != NULL);
        DatablockTemplate * pPortTemplate = pPort->GetTemplate();
        DatablockTemplate * pBlockTemplate = pDestBlock->GetTemplate();
        if(!pPortTemplate || !pBlockTemplate) 
            return;

        if(pDestBlock->HasMetadataChannel() || pPortTemplate->DescribesRecordStream()) {
            if(!pDestBlock->IsSealed()) {
                pDestBlock->SetRecordCount(m_nRecordCardinality);
                pDestBlock->Seal(m_nRecordCardinality, 
                                 pDestBlock->GetTemplate()->GetDatablockByteCount(DBDATA_IDX),
                                 m_nRecordCardinality * sizeof(long) * 2,
                                 0);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates the buffer asynchronous dependences. </summary>
    ///
    /// <remarks>   Crossbac, 5/26/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::CreateBufferAsyncDependences(
        VOID
        )
    {
        // create new sync points for buffers that have been bound as writeable resources for this
        // dispatch. First go through the per-accelerator map to find buffers fow which we have
        // committed to create sync points and create those sync points. Next, clear the buffer list
        // for any subsequent dispatches so we needn't new up a fresh std::set to track these
        // dependences on every dispatch. 
        
        MARKRANGEENTER(L"CreateBufferAsyncDependences");
        std::map<AsyncContext*, SyncPoint*> syncpoints;
        std::set<std::pair<PBuffer*,ASYNCHRONOUS_OPTYPE>>* pBuffers;
        std::map<AsyncContext*,std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>>*>::iterator ai;
        std::set<std::pair<PBuffer*,ASYNCHRONOUS_OPTYPE>>::iterator ci;

        for(ai=m_vOutgoingPendingBufferOps.begin(); ai!=m_vOutgoingPendingBufferOps.end(); ai++) {
            pBuffers = ai->second;
            AsyncContext * pAsyncContext = ai->first;
            BOOL bUseableContext = (pAsyncContext != NULL &&
                                    pAsyncContext->SupportsExplicitAsyncOperations());
            if(bUseableContext && pBuffers->size()) {
                SyncPoint * pSP = NULL;
                pAsyncContext->Lock();
                pSP = pAsyncContext->CreateSyncPoint(NULL);
                pAsyncContext->Unlock();
                assert(pAsyncContext->GetDeviceContext()->LockIsHeld());
                for(ci=pBuffers->begin(); ci!=pBuffers->end(); ci++) {
                    PBuffer * pBuffer = ci->first;
                    ASYNCHRONOUS_OPTYPE eOperation = ci->second;
                    pBuffer->AddOutstandingDependence(pAsyncContext, eOperation, pSP);
                }
            }
            pBuffers->clear();
        }
        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dispatches a task. 
    ///               
    ///             1. make sure that any device-resident inputs are actually resident on the
    ///                dispatch accelerator (something we can't know until the dispatch accelerator
    ///                is selected by the scheduler). Because ports may represent inputs that are
    ///                bound on *other* dependent accelerators (e.g. host tasks using CUDA), we must
    ///                also resolve the mapping from port->dependent accelerator before deciding if/what
    ///                to migrate data between memory spaces. 
    ///               
    ///             2. bind the actual shader binary, inputs, outputs and constant buffers
    ///               
    ///             3. dispatch
    ///               
    ///             4. unbind the actual shader binary, inputs, outputs and constant buffers
    ///               
    ///             5. release our reference to any datablocks that are in flight, typically causing
    ///                inputs to get freed, unless the caller has kept a reference to them.     
    ///                
    ///             Each step ultimately defers to a "PlatformSpecific*" implementation which makes
    ///             the necessary API calls for the given GPGPU backend to set up the dispatch. 
    ///             
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Task::Dispatch( 
        VOID
        )
    {
        Scheduler::UpdateLastDispatchTimestamp();
        log_dispatch_enter();
        tpprofile_enter(Dispatch);

        int ordinal = 0;
        BOOL bSuccess = FALSE;
        MARKRANGEENTER(L"PrepareDispatch");
        MarkInflightDispatch();
        ResolveDependentPortBindings();
        AttemptLocklessWaitOutstandingDeps();
        AcquireDispatchResourceLocks();
        MigrateInputs();
        EstimateDispatchDimensions();
        PreDispatchMemoryPressureCheck();
        MARKRANGEEXIT();
        
        MARKRANGEENTER(L"DispBindings");
        bSuccess = BindInputs(ordinal);
        if(bSuccess) bSuccess = BindMetaPorts();
        if(bSuccess) bSuccess = BindOutputs(ordinal);
        if(bSuccess) bSuccess = BindConstants(ordinal);
        if(bSuccess) bSuccess = BindExecutable();
        if(bSuccess) bSuccess = PlatformSpecificFinalizeBindings();
        MARKRANGEEXIT();

        WaitIncomingBufferAsyncDependences();
        ReleaseMigrationSourceLocks();
        
        if(!bSuccess) {
            ReleaseDispatchResourceLocks();
            CompleteInflightDispatch();
            PTask::Runtime::HandleError("%s: %s: Bindings failed at dispatch!\n", 
                                        __FUNCTION__,
                                        m_lpszTaskName);
            tpprofile_exit(Dispatch);
            return FALSE;
        }

        tpprofile_enter(PSDispatch);
        
        UINT nDispatchIterationCount = GetDispatchIterationCount();
        for(UINT nIteration=0; nIteration<nDispatchIterationCount && bSuccess; nIteration++) {
            dump_dispatch_info(nIteration, nDispatchIterationCount);
            dispcnt_record_dispatch(this);
            init_task_code_instrumentation(this);
            cond_record_stream_agg_entry();
			record_dispatch_entry();
            MARKTASKENTER(m_lpszTaskName);
            bSuccess = PlatformSpecificDispatch();
            MARKTASKEXIT();
			record_dispatch_exit();
            finalize_task_code_instrumentation(this);
        }       
        
        tpprofile_exit(PSDispatch);

        tpprofile_enter(DispatchTeardown);
        MARKRANGEENTER(L"PostDispatch");
        CreateBufferAsyncDependences();
        UnbindExecutable();
        UnbindOutputs();
        UnbindInputs();
        UnbindConstants();
        UnbindMetaPorts();
        ReleaseDispatchResourceLocks();
        Scheduler::CompleteDispatch(this);
        ReleaseInflightDatablocks();
        MARKRANGEEXIT();
        tpprofile_exit(DispatchTeardown);

        log_dispatch_exit();
        CompleteInflightDispatch();
        tpprofile_exit(Dispatch);
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initialize instrumentation. </summary>
    ///
    /// <remarks>   t-nailaf, 06/10/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void Task::InitializeInstrumentation() { }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Finalize instrumentation (cleanup allocated resources, etc). </summary>
    ///
    /// <remarks>   t-nailaf, 06/10/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void Task::FinalizeInstrumentation() {}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Attempt a lightweight wait for outstanding async deps on incoming blocks. The
    ///             general strategy in PTask for dealing with RW dependences amongst tasks is to
    ///             detect sharing conflicts and use backend async APIs to explicitly order
    ///             conflicting operations on shared blocks. Where such APIs are present, maximal
    ///             asychrony is possible. However, there is a difficult case: tasks that require
    ///             only dispatch resources that support no explicit asynchrony managment, which
    ///             consume data produced by tasks using resources that do. Such cases fundamentally
    ///             require synchrony (the consumer cannot proceed until all conflicting outstanding
    ///             operations are known to have resolved), which in turn requires performance-
    ///             sapping calls that synchronize with devices and harm the performance of other
    ///             tasks.
    ///             
    ///             The challenge is to perform such synchronization with minimal impact--making
    ///             backend framework calls typically requires a lock on an accelerator instance that
    ///             encapsulate a device context, so while such waits should not impact the
    ///             performance of other tasks from a threading standpoint, locked accelerators cause
    ///             a defacto serialization because no other task can use the encapsulated device for
    ///             the duration of the wait. Async APIs differ by framework though--in particular,
    ///             it appears that the CUDA API we use to deal with these dependences
    ///             (cuEventSynchronize) does not require the device context that created the event
    ///             to be current, and is purportedly threadsafe, so it is plausible that we can
    ///             perform such waits on independent threads without acquiring locks that block
    ///             forward progress of other tasks.
    ///             
    ///             This method (called at the beginning of dispatch), if the runtime setting
    ///             PTask::Runtime::SetTaskDispatchLocklessIncomingDepWait(TRUE) has been called,
    ///             will attempt all such required synchronous waits without acquiring accelerator
    ///             locks *before* the normal protocol for pre-dispatch lock acquisition is run.
    ///             
    ///             This method is experimental--the CUDA APIs lack sufficient detail to predict
    ///             whether this is truly safe, so we're finding out empirically.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void        
    Task::AttemptLocklessWaitOutstandingDeps(
        VOID
        )
    {
        // If PTask doesn't have this mode turned on, do nothing
        if(!PTask::Runtime::GetTaskDispatchLocklessIncomingDepWait())
            return;

        // if the dispatch accelerator supports async APIs, this is a NOP
        assert(!PTask::Runtime::GetTaskDispatchLocksIncomingAsyncSources()); 
        if(m_pDispatchAccelerator->SupportsExplicitAsyncOperations())
            return;

        map<UINT, Port*>::iterator mi;
        for(mi=m_mapInputPorts.begin(); mi!= m_mapInputPorts.end(); mi++) {

			InputPort* pPort = dynamic_cast<InputPort*>(mi->second);
            Accelerator * pTargetAccelerator = GetTargetMemorySpace(pPort);
            if(pTargetAccelerator->SupportsExplicitAsyncOperations())
                continue;   // ASYNC API support--nothing to do 

            // no async apis for either the dispatch accelerator or the
            // port-bind accelerator--we have to examine the block. 
            Datablock * pBlock = pPort->Peek();
            if(pBlock == NULL) continue; 
            Port * pInOutConsumer = pPort->GetInOutConsumer();
            ASYNCHRONOUS_OPTYPE eOpType = pInOutConsumer != NULL ? 
                OT_LAUNCH_TARGET_WRITABLE : OT_LAUNCH_TARGET_READABLE;

            pBlock->Lock();
            if(pBlock->HasOutstandingAsyncDependences(pTargetAccelerator, eOpType)) {

                // this block has outstanding operations queued asynchronously that require a wait
                // before an operation of the proposed type can begin. Attempt to wait those
                // outstanding operations without acquiring accelerator locks. 
                assert(!pBlock->RequiresMigration(pTargetAccelerator));  // true->ASYNC API present!
#ifdef NVPROFILE
                std::stringstream ssRange;
                ssRange 
                    << "ll-wait:" 
                    << m_lpszTaskName 
                    << "-" << pPort->GetVariableBinding() 
                    << ASYNCOP_ISREAD(eOpType) ? "(R)":"(W)";
#endif
                MARKTASKENTER(ssRange.str().c_str());
                pBlock->LocklessWaitOutstanding(pTargetAccelerator, eOpType);
                MARKTASKEXIT();
            }
            pBlock->Unlock();			
        }

    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record pending buffer asynchronous dependences. Save the ones
    ///             for which we must wait before dispatch so that we can wait them
    ///             en masse before the dispatch calls, and save the ones which will
    ///             be outstanding after dispatch so we can create new dependences
    ///             after dispatch.  
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/24/2013. </remarks>
    ///
    /// <param name="pBuffer">          [in] non-null, the buffer. </param>
    /// <param name="pAccelerator">     [in] non-null, the accelerator. </param>
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="eOperationType">   Type of the operation. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::WaitIncomingBufferAsyncDependences(
        VOID
        )
    {
        FIND_DUPLICATE_DEPENDENCES();            
        std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>>::iterator si;
        std::map<AsyncContext*, std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>>*>::iterator ii;
        for(ii=m_vIncomingPendingBufferOps.begin();ii!=m_vIncomingPendingBufferOps.end();ii++) {

            // choosing an async context to perform the wait has some subtleties-- if the async context
            // that is mapped to these buffers does not support asynchronous waiting, we can sometimes use
            // the async context of the dispatch accelerator. 

            AsyncContext * pBufferAsyncContext = ii->first;            
            AsyncContext * pWaitAsyncContext = pBufferAsyncContext;
            if(pBufferAsyncContext != NULL && !pBufferAsyncContext->SupportsExplicitAsyncOperations()) {
                AsyncContext * pDispatchAsyncContext = m_pDispatchAccelerator->GetAsyncContext(ASYNCCTXT_DEFAULT);
                if(pDispatchAsyncContext != NULL && pDispatchAsyncContext->SupportsExplicitAsyncOperations()) {
                    pWaitAsyncContext = pDispatchAsyncContext;
                }
            }
            std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>> * pBuffers = ii->second;
            for(si=pBuffers->begin(); si!=pBuffers->end(); si++) {
                PBuffer * pBuffer = si->first;

                // if we have a null async context here, we had better have a lock 
                // on an accelerator capable with a context that we can use for the wait. 
                // asserting with precision here is cumbersome: instead assert that
                // we have a lock on more than just the dispatch accelerator. 
                
                assert(pWaitAsyncContext != NULL || m_vRequiredAcceleratorLocks.size() > 1 ||
                       PTask::Runtime::GetTaskDispatchLocklessIncomingDepWait());

                std::set<AsyncDependence*>* pDependences = NULL;
                BOOL * pbAlreadyResolved = NULL;
                GET_OSTENSIBLE_DEPENDENCES(pDependences, pBuffer);
                pBuffer->WaitOutstandingAsyncOperations(pWaitAsyncContext, si->second, pDependences, pbAlreadyResolved); 
                VALIDATE_EARLY_RESOLUTION(pDependences, pBuffer, pbAlreadyResolved);
            }
            pBuffers->clear();
        } 
        CLEAR_OSTENSIBLE_DEPENDENCES();
        ReleaseIncomingAsyncDependenceLocks();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record a pending buffer asynchronous dependence. This means that a buffer has
    ///             been bound as an out or inout variable to a given port, meaning that any
    ///             subsequent users of the buffer on a different stream will need to wait until the
    ///             dispatch has been completed. This method adds the buffer to a list of map
    ///             (accelerator->buffer) which will be used subsequently to create sync points in
    ///             CreateBufferAsyncDependences.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/24/2013. </remarks>
    ///
    /// <param name="pBuffer">              [in] non-null, the buffer. </param>
    /// <param name="pPortBindAccelerator"> [in,out] If non-null, the port bind accelerator. </param>
    /// <param name="eOperation">           The operation. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::RecordPendingBufferAsyncDependences(
        __in PBuffer * pBuffer,
        __in Accelerator * pPortBindAccelerator, 
        __in ASYNCHRONOUS_OPTYPE eOperation
        )
    {
        std::set<AsyncDependence*>* pDependences = NULL; 
        GET_OSTENSIBLE_DEPENDENCES(pDependences, pBuffer);
        AsyncContext * pLaunchContext = GetOperationAsyncContext(pPortBindAccelerator, ASYNCCTXT_TASK);

        if(pBuffer->ContextRequiresSync(pLaunchContext, eOperation, pDependences)) {
            std::map<AsyncContext*, std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>>*>::iterator ii;
            ii = m_vIncomingPendingBufferOps.find(pLaunchContext);
            if(ii == m_vIncomingPendingBufferOps.end()) { 
                m_vIncomingPendingBufferOps[pLaunchContext] = new std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>>();
            } 
            m_vIncomingPendingBufferOps[pLaunchContext]->insert(make_pair(pBuffer,eOperation));
        }

        assert(m_pDispatchAccelerator->IsHost() || (m_pDispatchAccelerator == pPortBindAccelerator));
        if(!pPortBindAccelerator->IsHost()) {
            assert(!(m_pDispatchAccelerator->IsHost() && pPortBindAccelerator->IsHost()));
            std::map<AsyncContext*, std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>>*>::iterator mi;
            mi = m_vOutgoingPendingBufferOps.find(pLaunchContext);
            if(mi == m_vOutgoingPendingBufferOps.end()) { 
                m_vOutgoingPendingBufferOps[pLaunchContext] = new std::set<std::pair<PBuffer*, ASYNCHRONOUS_OPTYPE>>();
            } 
            m_vOutgoingPendingBufferOps[pLaunchContext]->insert(make_pair(pBuffer,eOperation));
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind constants. </summary>
    ///
    /// <remarks>   Crossbac, 1/5/2012. </remarks>
    ///
    /// <param name="ordinal">  [in,out] The ordinal. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Task::BindConstants(
        int &ordinal
        ) 
    {
        trace2("(%s)->BindConstants\n", m_lpszTaskName);
        m_nActualConstantCount = 0;
        UINT nConstantPorts = (UINT) m_mapConstantPorts.size();
        if(nConstantPorts == 0) 
            return TRUE;

        tpprofile_enter(BindConstants);
        map<UINT, Port*>::iterator mi;
        for(mi=m_mapConstantPorts.begin(); 
            mi!=m_mapConstantPorts.end(); mi++) {		

            // pull the block we want from the port
            // and lock it. The addref is unnecessary
            // but harmless, so leave it for now...
            UINT uiPortIndex = mi->first;
            Port * pPort = mi->second;
			critical_path_check_binding_trace("%s:BindConstant(%s)\n", m_lpszTaskName, pPort->GetVariableBinding());
            Datablock * pBlock = pPort->Pull();
            pBlock->AddRef();
            pBlock->Lock();
            pBlock->RecordBinding(pPort, this);     

            BOOL bMultiChannel = 
                pBlock->HasMetadataChannel() ||    // not supported for constants. what would it mean?
                pBlock->HasTemplateChannel();      // not supported for constants. what would it mean?
            if(bMultiChannel) {
                assert(FALSE);
                PTask::Runtime::HandleError("%s: attempt to use multi-channel input to constant memory!",
                                            __FUNCTION__);
                tpprofile_exit(BindConstants);
                return FALSE;
            }

            // if we are binding to constant memory, this block
            // needs to be immutable, or the buffer creation will fail
            if(!(pBlock->GetAccessFlags() & PT_ACCESS_IMMUTABLE)) {
                BUFFERACCESSFLAGS flags = pBlock->GetAccessFlags();
                pBlock->SetAccessFlags(flags | PT_ACCESS_IMMUTABLE);
            }

            // it is not necessarily the case that we need to materialize device side buffer: this platform
            // prefers to communicate arguments by value in the binding/dispatch params, rather than by
            // creating an explicit device-side buffer. we only want a device-side buffer if: a) the port
            // is not a formal parameter b) the port is a formal parameter that neither an integer or float
            // (thus requiring a buffer)             
            BOOL bScalarBinding = 
                    m_pDispatchAccelerator->SupportsByvalArguments() &&
                    pPort->IsFormalParameter() && 
                    (pPort->GetParameterType() == PTPARM_INT ||
                     pPort->GetParameterType() == PTPARM_FLOAT ||
                     pPort->GetParameterType() == PTPARM_BYVALSTRUCT);

            PBuffer * pBindBuffer = NULL;
            
            if(bScalarBinding) {

                // Use the host-side buffer. It better exist, or there is no value for this constant!
                pBindBuffer = pBlock->GetPlatformBuffer(HOST_MEMORY_SPACE_ID, DBDATA_IDX);

            } else {

                // make sure the accelerator-side view of the block is up-to-date. We use the block as a
                // constant in GPU code, so we don't need exclusive permissions--request shared. Propagate any
                // control information received with the block, and then bind the appropriate view the constant
                // buffer resources. 
                BOOL bPopulate = TRUE;
                pBlock->UpdateView(m_pDispatchAccelerator, 
                                   GetOperationAsyncContext(m_pDispatchAccelerator, ASYNCCTXT_XFERHTOD),
                                   bPopulate, 
                                   BSTATE_SHARED,
                                   DBDATA_IDX,
                                   DBDATA_IDX);
                pBlock->PropagateControlInformation(pPort);
                pBindBuffer = pBlock->GetPlatformBuffer(m_pDispatchAccelerator, DBDATA_IDX);                
            }

            dump_buffer(pPort, pBindBuffer);
            PlatformSpecificBindConstant(pPort, ordinal, m_nActualConstantCount, pBindBuffer);
            m_nActualConstantCount++;
            ordinal++;

            pBlock->Unlock();
            pBlock->Release();
            uiPortIndex++;
        }
        tpprofile_exit(BindConstants);
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind outputs for an imminent shader dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::BindOutputs(
        int &ordinal
        ) 
    {
        /// <summary> The number of actual outputs. Each port is a single
        /// 		  logical output, but since datablocks may have multiple channels,
        /// 		  and each channel will have a buffer that must bind to an
        /// 		  *actual* output in shader code, the number of actual outputs
        /// 		  can be greater than the number of output ports. 
        /// 		  </summary>
        m_nActualOutputCount = 0;

        size_t nOutputPorts = m_mapOutputPorts.size();
        if(nOutputPorts == 0)
            return TRUE; // ok to have no outputs. weird though.

        /// <summary> Zero-based index of the port we are binding resources for.
        /// 		  See comment above for nActualInputs for details about why
        /// 		  bound output counts are different from the number of output ports.
        /// 		  </summary>
        size_t uiPortIndex = 0;

        tpprofile_enter(BindOutputs);
        for(map<UINT, Port*>::iterator mi=m_mapOutputPorts.begin(); mi!= m_mapOutputPorts.end(); mi++) {

            // get the output port and grab the datablock that will be written on the current dispatch. 
            OutputPort* pPort = (OutputPort*) mi->second;
			critical_path_check_binding_trace("%s:BindOutput(%s)\n", m_lpszTaskName, pPort->GetVariableBinding());

            // figure out which memory space we must bind. In the overwhelming majority of
            // cases, the port bind accelerator is just the dispatch accelerator. However, for
            // tasks with dependent accelerator bindings (e.g. host tasks using CUDA), the port
            // may actually be binding a resource in a memory space other than the dispatch.
            // Figure out the target space, so we can get the right platform buffer when we bind.
            Accelerator * pPortBindAccelerator = m_pDispatchAccelerator;
            if(pPort->HasDependentAcceleratorBinding()) 
                pPortBindAccelerator = GetAssignedDependentAccelerator(pPort);

            Datablock * pDestBlock = pPort->GetDestinationBuffer(pPortBindAccelerator);
            UINT uiBindTargetChannel = pPort->GetDestinationChannel(pPortBindAccelerator);

            pDestBlock->Lock();
            pDestBlock->RecordBinding(pPort, this);    

            // if this output port is marked unmarshallable (because the device side code writting it is
            // producing data such as pointers which are only valid on the device), be sure to mark the
            // block as unmarshallable so that the scheduler doesn't schedule any consumers of the block on
            // another  device. 
            pDestBlock->SetMarshallable(pPort->IsMarshallable());

            // if this port is part of the control network, the output control code will have been set
            // during the bind of the input port producing the control signal. Here, we simply set the
            // output block's control code accordingly. 
            pPort->Lock();
            CONTROLSIGNAL luiCtlCode = pPort->GetPropagatedControlSignals();
            pDestBlock->SetControlSignal(luiCtlCode);
            ctlpegress(this, pDestBlock);
            // std::cout << m_lpszTaskName << "->" << pPort << " setting out ctl signal to " << luiCtlCode << std::endl;
            pPort->Unlock();

            // If the block has no device side buffers, create them. Note that we don't need to do anything
            // if the it does have device-side buffers because either the block is in/out (in which case
            // the buffers were created in BindInputs), or the block is a write-only resource for this
            // dispatch, so it is perfectly safe to leave whatever is already in those buffers alone: it's
            // only going to get overwritten. 
            if(!pDestBlock->HasBuffers(pPortBindAccelerator)) {

                // Some blocks require "sealing". 
                // if we need to write to an output metadata channel, allocate the buffer
                // we can tell there will be a metadata channel from a few things. 
                // a. the destination block explicitly has such a channel (the buffer exists)
                // b. the output port template describes a record stream
                ForceSealed(pDestBlock, pPort); // harmless if not required.

                // Create accelerator-side buffers. BindInputs creates accelerator buffers
                // for in/out variables, so if there are no accelerator buffers, this better
                // not be the out side of an in/out pair. If there is an initial value committed
                // for blocks on this port, an HtoD transfer may occur--in most cases the async
                // context should be ignored.
                
                AsyncContext * pUpdateAsyncContext = GetOperationAsyncContext(pPortBindAccelerator, ASYNCCTXT_XFERHTOD);
                pDestBlock->AllocateBuffers(pPortBindAccelerator, pUpdateAsyncContext, FALSE);
                assert(pPort->GetInOutProducer() == NULL);
            }

            // now bind buffers to parameters. The notion of inferring meta-data and template data bindings
            // from parameter ordering is (rightly) obsolete. This datablock may be the target block for
            // multiple output ports on this task, (each of which targets a different channel such as data,
            // metadata, etc.), but each binding of an output block must explicitly specify the target
            // channel. get the platform-specific buffer associated with this block, find its buffer object-
            // -it better be there because CreateAcceleratorBuffers should create it based on the access
            // flags. 
                
            PBuffer * pBuffer = pDestBlock->GetPlatformBuffer(pPortBindAccelerator, uiBindTargetChannel);
            RecordPendingBufferAsyncDependences(pBuffer, pPortBindAccelerator, OT_LAUNCH_TARGET_WRITABLE);
            PlatformSpecificBindOutput(pPort, ordinal, m_nActualOutputCount, pBuffer);
            pBuffer->MarkDirty(TRUE);
            m_nActualOutputCount++;
            ordinal++;

            // technically, we should set this block to have a coherent view *after*
            // the dispatch (indicating data is valid there), but we do it here for a few reasons:
            // 1) doing it after the dispatch requires costly traversal of the port
            //    data structures to find the output block again.
            // 2) the bind/dispatch/unbind sequence is done with the accelerator
            //    lock held, so it's safe to do it here from a concurrency standpoint
            // 3) dispatch does not return success/failure, so doing it later does not
            //    provide us with additional error-handling information.
            BOOL bAcquireSuccess = pDestBlock->AcquireExclusive(pPortBindAccelerator);
            assert(bAcquireSuccess); UNREFERENCED_PARAMETER(bAcquireSuccess);

            // we may have to estimate the dispatch dimensions here because the task may use only unordered
            // access views. The call to estimate will do nothing after an estimate is made, so it is
            // harmless to call it here if the dimensions wer already estimated by previous bind call. 
            EstimateDispatchDimensions(pDestBlock);

            UNREFERENCED_PARAMETER(uiPortIndex);
            pDestBlock->SetProducerTask(this);
            pDestBlock->Unlock();
        }
        tpprofile_exit(BindOutputs);
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind inputs. </summary>
    ///
    /// <remarks>   FIXME: TODO: 1. Implement pooling on input ports! For blocks with precisely
    ///                             determined input sizes this removes allocation from the critical
    ///                             path.
    ///             
    ///             FIXME: TODO: 2. Decouple input materialization from dispatch. Currently a task
    ///                             is ready for dispatch when all its inputs have non-null
    ///                             datablocks available. This is too conservative. A task *should* be
    ///                             ready for input materialization when all its inputs have non-null
    ///                             blocks, and is only ready for *dispatch* when all
    ///                             materializations have completed. Basically, we should modify the
    ///                             state machine to include a "MATERIALIZING" state in addition to
    ///                             those outlined in the original PTask SOSP paper. 
    ///             
    ///             crossbac, 12/23/2011.
    ///             </remarks>
    ///
    /// <param name="ordinal">  [in,out] The ordinal. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Task::BindInputs(
        int &ordinal
        ) 
    {
        UNREFERENCED_PARAMETER(ordinal);
        BOOL bInputOnlyTask = m_mapOutputPorts.size() == 0;

        /// <summary> The number of actual inputs. The datablock is a single
        /// 		  logical input, but since it may have multiple channels,
        /// 		  and each channel will have a buffer that must bind to an
        /// 		  *actual* input in shader code, the number of actual inputs
        /// 		  can be greater than the number of input ports. It can also
        /// 		  be less than the number of input ports, because in/out ports
        /// 		  must be bound to writable shader code resources (unordered
        /// 		  access views) so we perform some materialization for in/outs
        /// 		  here, but binding is deferred to BindOutputs.</summary>
        m_nActualInputCount = 0;

        size_t nInputPorts = m_mapInputPorts.size(); 
        if(nInputPorts == 0)
            return TRUE; // ok to have no inputs. weird, but ok. 

        /// <summary> Zero-based index of the port we are binding resources for.
        /// 		  See comment above for nActualInputs for details about why
        /// 		  bound input counts are different from the number of input ports.
        /// 		  </summary>
        size_t uiPortIndex = 0;
        map<UINT, Port*>::iterator mi;

        tpprofile_enter(BindInputs);
        for(mi=m_mapInputPorts.begin();mi!= m_mapInputPorts.end(); mi++) {
                
            // get the port and the consumer port
            // if this input port is part of an in/out pair
            
            Port* pPort = mi->second; 
			critical_path_check_binding_trace("%s:BindInput(%s)\n", m_lpszTaskName, pPort->GetVariableBinding());
            OutputPort * pInOutConsumer = (OutputPort*)((InputPort*)pPort)->GetInOutConsumer(); 

            // figure out which memory space we must bind. In the overwhelming majority of
            // cases, the port bind accelerator is just the dispatch accelerator. However, for
            // tasks with dependent accelerator bindings (e.g. host tasks using CUDA), the port
            // may actually be binding a resource in a memory space other than the dispatch.
            // Figure out the target space, so we can get the right platform buffer when we bind.
            
            Accelerator * pPortBindAccelerator = m_pDispatchAccelerator;
            if(pPort->HasDependentAcceleratorBinding()) 
                pPortBindAccelerator = GetAssignedDependentAccelerator(pPort);
            ASYNCHRONOUS_OPTYPE eLaunchOpType = OT_LAUNCH_TARGET_READABLE;
            BOOL bPureHostPort = pPortBindAccelerator->IsHost() &&
                                 m_pDispatchAccelerator == pPortBindAccelerator;
                
            // pull the input data block, lock it. Use the block to estimate the dispatch thread group
            // dimensions: if the programmer has made the geometry explicit for this task, or the
            // dimensions have already been estimated, this will cause no change to the dispatch geometry.
            // Finally, if there is a control signal on the block, propagate it. We do this before updating
            // the acclerator view because control information propagates independently from data, so need
            // for exclusive or shared permissions is irrelevant to control routing. 
            
            Datablock * pBlock = pPort->Pull();                     // get the input block.
            ctlpingress(this, pBlock);                              // for signal profiler...
            pBlock->Lock();                                         // lock it
            EstimateDispatchDimensions(pBlock);                     // set thread group sizes
            pBlock->PropagateControlInformation(pPort);             // route control information.
            pBlock->RecordBinding(pPort, this, pInOutConsumer);     // track migration (coherence profiler)        

            assert(!pBlock->IsScalarParameter());    // should use constant port for this!

            // Update the dispatch accelerator's view of the data. If this is part of an in/out pair, then
            // we must assume the block will be written so we update it with a request for modifiable
            // (exclusive) permissions. Otherwise, the update can request a shared copy. Note that if this
            // port is part of an in/out pair, the destination port to the corresponding output port. Note
            // that we will not do the device-side bind of an in/out here: DirectX's Unordered Access Views
            // and Shader Resource Views are not trivial to map onto PTask's in/out variable abstraction.
            // In essence, a UAV *is* an in/out byref variable, but it can only by bound to an output stage
            // of the pipeline. 
            
            if(pInOutConsumer != NULL) {

                // If this port is part of an in/out pair, then the very same block object will be bound to an
                // output port below. The output port should set any control signals on the block if the output
                // port is part of the control propagation path. However, if this block is carrying a control
                // signal and the output port is *not*, we need to be sure that we've cleared the signal on
                // this block, otherwise, a signal will be produced on the output port when it should not be,
                // which can lead to all manner of unpleasantness. 
                
                pBlock->ClearAllControlSignals();  // post-dispatch will reset based on port, if need be

                // if this port is part of an in/out pair, then it's going to get written on dispatch, so we
                // add the block to the exclusive permissions list rather than requesting an update now.
                // Because this is an in/out binding, it is *not* a blind write. Note also that we don't bind
                // to any GPU-side resources yet--this is deferred to BindOutputs. Instead, we just make this
                // block the "destination buffer" for the output port. 
                
                BOOL bPopulate = TRUE;
                pBlock->SetDestinationPort(pInOutConsumer);
                pBlock->SetProducerTask(this);
                pInOutConsumer->SetDestinationBuffer(pBlock);

                // update the accelerator's view. we want the view backed by the most recent data because this
                // is a read-write variable, and we need exclusive access, which will ensure that other
                // accelerator's views are invalidated. This block should have been locked at the
                // beginning of dispatch because we need to hold a lock on it until dispatch completes.
                
                if(pBlock->RequiresViewUpdate(pPortBindAccelerator, BSTATE_EXCLUSIVE)) {
                    AsyncContext * pUpdateAsyncContext = GetOperationAsyncContext(pPortBindAccelerator, ASYNCCTXT_XFERHTOD);
                    if(bInputOnlyTask && bPureHostPort) {
                        MARKRANGEENTER(L"BindInputs.UpdateView(EXCL)");
                    }
                    pBlock->UpdateView(pPortBindAccelerator, 
                                       pUpdateAsyncContext,
                                       bPopulate, 
                                       BSTATE_EXCLUSIVE,
                                       DBDATA_IDX,
                                       DBDATA_IDX);
                    if(bInputOnlyTask && bPureHostPort) {
                        MARKRANGEEXIT();
                    }
                }
                pBlock->SetMarshallable(pInOutConsumer->IsMarshallable());
                eLaunchOpType = OT_LAUNCH_TARGET_WRITABLE;

            } else {

                // Don't deal with blocks that have absolutely no data. 
                assert(pBlock->HasValidChannels());

                // This block is being used as an input, so we do not need exclusive permissions on the block.
                // Update the accelerator view with shared permissions. we are actually going to bind this
                // block to directX execution resources. So get the platform-specific buffers associated with
                // all channels in use by this block and this accelerator and bind to shader resource views.
                // Note that if we have meta data and template buffers in this block, they are bound as well by
                // iterating the channel indeces. When we update the accelerator view, we set the populate flag
                // to true because this task is a consumer of the data (not a blind write). 

                BOOL bPopulate = TRUE;
                Datablock::ChannelIterator iter;

                if(pBlock->RequiresViewUpdate(pPortBindAccelerator, BSTATE_SHARED)) {
                    AsyncContext * pUpdateAsyncContext = GetOperationAsyncContext(pPortBindAccelerator, ASYNCCTXT_XFERHTOD);
                    if(bInputOnlyTask && bPureHostPort) {
                        MARKRANGEENTER(L"BindInputs.UpdateView(SHR)");
                    }
                    pBlock->UpdateView(pPortBindAccelerator, 
                                       pUpdateAsyncContext,
                                       bPopulate, 
                                       BSTATE_SHARED,
                                       DBDATA_IDX,
                                       DBDATA_IDX);
                    if(bInputOnlyTask && bPureHostPort) {
                        MARKRANGEEXIT();
                    }
                }

                eLaunchOpType = OT_LAUNCH_TARGET_READABLE;
                for(iter=pBlock->FirstChannel(); iter!=pBlock->LastChannel(); iter++) {

                    // Get the PBuffer for this channel, do some basic sanity checks, (it should have been
                    // materialized by calls above if this is a valid channel for the block), dump some of its
                    // contents if we are debugging, and bind it to a shader resource view. 
                    PBuffer * pBuffer = pBlock->GetPlatformBuffer(pPortBindAccelerator, iter);
                    assert(pBuffer != NULL);
                    RecordPendingBufferAsyncDependences(pBuffer, pPortBindAccelerator, eLaunchOpType);

                    dump_buffer(pPort, pBuffer);
                    PlatformSpecificBindInput(pPort, ordinal, m_nActualInputCount, pBuffer);
                    m_nActualInputCount++;
                    ordinal++;
                    
                    // if this task does not have implicity meta channel bindings,
                    // we can leave this loop early: even if there is a valid meta channel
                    // or template channel on this block, we don't have anything to bind it to.
                    if(!GetHasImplicitMetaChannelBindings())
                        break;
                }
            }

            // mark the block as being in flight so that we can
            // release it or push it into the appropriate output
            // channels when dispatch completes. 
            AddToInflightList(pPort, pBlock);
            pBlock->Unlock();
            uiPortIndex++;
        }
        tpprofile_exit(BindInputs);
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind shader. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void Task::UnbindExecutable() {}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind inputs. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void Task::UnbindInputs(
        VOID
        ) 
    {
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind Meta Ports. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void Task::UnbindMetaPorts(
        VOID
        ) 
    {
        map<UINT, Port*>::iterator mi;
        for(mi=m_mapMetaPorts.begin(); mi!=m_mapMetaPorts.end(); mi++) {
            MetaPort* pPort = (MetaPort*) mi->second;
            if(pPort->GetMetaFunction() == MF_SIMPLE_ITERATOR) {
                ResetDispatchIterationCount();
                break;
            }
            pPort->FinalizeMetaFunction(m_pDispatchAccelerator);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind outputs. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void Task::UnbindOutputs() {}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind constants. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void Task::UnbindConstants() {}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Estimate dispatch dimensions. </summary>
    ///
    /// <remarks>   Crossbac, 1/5/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::EstimateDispatchDimensions(
        Datablock * pBlock
        ) 
    { 
        UNREFERENCED_PARAMETER(pBlock);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Estimate dispatch dimensions. </summary>
    ///
    /// <remarks>   Crossbac, 1/5/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::EstimateDispatchDimensions(
        VOID
        ) 
    { 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the dispatch iteration count. This is the number of times the platform-
    ///             specific dispatch routine should invoke acclerator code for this dispatch. The
    ///             default is 1. If the Task has a MetaPort whose meta-function is of type
    ///             MF_SIMPLE_ITERATOR, then blocks received on that port will be interpreted as an
    ///             integer-valued iteration count. Dispatch will call GetIterationCount before
    ///             dispatching, and will clear the count (reset to 1) after.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/10/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The dispatch iteration count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Task::GetDispatchIterationCount(
        VOID
        )
    {
        return m_nDispatchIterationCount;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the dispatch iteration count. This is the number of times the platform-
    ///             specific dispatch routine should invoke acclerator code for this dispatch. The
    ///             default is 1. If the Task has a MetaPort whose meta-function is of type
    ///             MF_SIMPLE_ITERATOR, then blocks received on that port will be interpreted as an
    ///             integer-valued iteration count. Dispatch will call GetIterationCount before
    ///             dispatching, and will clear the count (reset to 1) after.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/10/2012. </remarks>
    ///
    /// <param name="nIterations">  The iterations. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SetDispatchIterationCount(
        UINT nIterations
        )
    {
        // by asserting here, we are checking that two conditions hold. First, that the dispatch
        // iteration count was cleared after the last dispatch, and second, that only one MetaPort per
        // task can control a simple iterator. 
        assert(m_nDispatchIterationCount == DEFAULT_DISPATCH_ITERATIONS);
        m_nDispatchIterationCount = nIterations;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resets the dispatch iteration count. This is the number of times the platform-
    ///             specific dispatch routine should invoke acclerator code for this dispatch. The
    ///             default is 1. If the Task has a MetaPort whose meta-function is of type
    ///             MF_SIMPLE_ITERATOR, then blocks received on that port will be interpreted as an
    ///             integer-valued iteration count. Dispatch will call GetIterationCount before
    ///             dispatching, and will clear the count (reset to 1) after.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/10/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::ResetDispatchIterationCount(
        VOID
        )
    {
        m_nDispatchIterationCount = DEFAULT_DISPATCH_ITERATIONS;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enter dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::MarkInflightDispatch(
        VOID
        )
    {
        EnterCriticalSection(&m_csDispatchLock);
        assert(!m_bInflight);
        m_bInflight = TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Complete dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::CompleteInflightDispatch(
        VOID
        )
    {
        assert(m_bInflight);
        m_bInflight = FALSE;
        LeaveCriticalSection(&m_csDispatchLock);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the sticky blocks described by vPortMap. </summary>
    ///
    /// <remarks>   Crossbac, 3/4/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::ReleaseStickyBlocks(
        VOID
        )
    {
        map<UINT, Port*>::iterator mi;
        for(mi=m_mapInputPorts.begin(); mi!=m_mapInputPorts.end(); mi++) {
            InputPort * pPort = dynamic_cast<InputPort*>(mi->second);
            pPort->ReleaseReplayableBlock();
        }
        for(mi=m_mapOutputPorts.begin(); mi!=m_mapOutputPorts.end(); mi++) {
            OutputPort * pPort = dynamic_cast<OutputPort*>(mi->second);
            pPort->ReleaseDatablock();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has explicit meta channel bindings. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <returns>   true if explicit meta channel bindings, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::GetHasImplicitMetaChannelBindings(
        VOID
        )
    {
        return m_bHasImplicitMetaChannelBindings;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the has explicit meta channel bindings. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="b">    true to b. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SetHasImplicitMetaChannelBindings(
        BOOL b
        )
    {
        m_bHasImplicitMetaChannelBindings = b;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the graph. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the graph. </returns>
    ///-------------------------------------------------------------------------------------------------

    Graph * 
    Task::GetGraph(
        VOID
        )
    {
        return m_pGraph;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Returns the current dispatch count for the task. This is a diagnostic tool; we
    ///             return the value without taking a lock or requiring the task to be in a quiescent
    ///             state, so the return value has no consistency guarantees. Use
    ///             GetDispatchStatistics when you need a consistent view. Use this when you need a
    ///             reasonable estimate (e.g. when debugging!)
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/16/2013. </remarks>
    ///
    /// <returns>   The current dispatch count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Task::GetCurrentDispatchCount(
        VOID
        )
    {
        return m_nDispatchNumber;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a graph. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SetGraph(
        Graph * pGraph
        )
    {
        assert(m_pGraph == NULL);
        m_pGraph = pGraph;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a dependent accelerator class. This interface provides a way for the
    ///             programmer to make a task's dependences on accelerators other than the dispatch
    ///             accelerator explicit, and is required for correctness if a task uses more than
    ///             one accelerator (for example, a HostTask using a GPU through CUDA+thrust). When
    ///             the scheduler is preparing a task for dispatch it must acquire, in addition to
    ///             the dispatch accelerator, an accelerator object of the appropriate class for
    ///             every entry in the m_vDependentAcceleratorClasses list before dispatch.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/9/2012. </remarks>
    ///
    /// <param name="cls">                          The accelerator class. </param>
    /// <param name="nRequiredInstances">           The number of instances of the class required. </param>
    /// <param name="bRequestDependentPSObjects">   True if platform-specific context objects should
    ///                                             be provided in the task entry point from platform-
    ///                                             specific dispatch. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::BindDependentAcceleratorClass(
        __in ACCELERATOR_CLASS cls, 
        __in int nRequiredInstances,
        __in BOOL bRequestDependentPSObjects
        ) 
    {
        Lock();
        BOOL bSuccess = FALSE;

        std::map<ACCELERATOR_CLASS, int>::iterator mi;
        mi=m_vDependentAcceleratorRequirements.find(cls);
        int nBoundInstances = ((mi==m_vDependentAcceleratorRequirements.end()) ? 0 : mi->second);

        if(nRequiredInstances != 1 || nBoundInstances > 0) {            

            // make sure the bind request is well-formed. Currently we support 
            // only 1 instance of each class per dependent class. If this is a redundant
            // binding, complain--we could probably recover, but it's a likely programmer
            // error if this occurs, so fail the request. 
            
            assert(nRequiredInstances == 1 && 
                   "multiple dependent instances per accelerator class not supported.");
            assert(m_vDependentAcceleratorRequirements.find(cls) ==
                   m_vDependentAcceleratorRequirements.end()); 
            Runtime::HandleError("%s: malformed dependent binding: %d of %s, with %d(MAX=1) already bound!\n",
                                 __FUNCTION__,
                                 nRequiredInstances,
                                 AccClassString(cls),
                                 nBoundInstances);

        } else {

            // record the fact that we require the given number of instances of the specified accelerator
            // class *in addition* to the dispatch accelerator in order to execute. 

            m_vDependentAcceleratorRequirements[cls] = nRequiredInstances;
            m_bRequestDependentPSObjects = bRequestDependentPSObjects;
            std::set<Accelerator*> acclist;
            std::set<Accelerator*>::iterator ai;
            Scheduler::FindEnabledCapableAccelerators(cls, acclist);
            for(ai=acclist.begin(); ai!=acclist.end(); ai++) {
                Accelerator * pDependentAccelerator = *ai;
                __createDispatchAsyncContext(pDependentAccelerator);
            }                

            bSuccess = TRUE;
        }
        Unlock();

        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the number dependent accelerator classes. The scheduler must acquire an
    ///             accelerator object for each depenendent accelerator class entry in addition to
    ///             the dispatch accelerator before dispatching a task. See AddDependentAccelerator
    ///             class documentation for more details.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/9/2012. </remarks>
    ///
    /// <returns>   The number of dependent accelerator classes. </returns>
    ///-------------------------------------------------------------------------------------------------

    int  
    Task::GetDependentBindingClassCount(
        VOID
        )
    {
        // no lock required. This list should never change after it's been constructed.
        // we try to cache this information though, which does require lock if we have not
        // yet cached it...
        if(m_uiDependentBindingClasses == MAXDWORD32) {
            Lock();
            if(m_uiDependentBindingClasses == MAXDWORD32) {
                m_uiDependentBindingClasses = static_cast<UINT>(m_vDependentAcceleratorRequirements.size());
            }
            Unlock();
        }
        assert(m_vDependentAcceleratorRequirements.size() == (size_t)m_uiDependentBindingClasses);
        return static_cast<int>(m_uiDependentBindingClasses);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has dependent accelerator bindings. </summary>
    ///
    /// <remarks>   crossbac, 7/8/2013. </remarks>
    ///
    /// <returns>   true if dependent accelerator bindings, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Task::HasDependentAcceleratorBindings(
        VOID
        )
    {
        if(m_uiDependentBindingClasses == MAXDWORD32)
            return GetDependentBindingClassCount() > 0;
        return m_uiDependentBindingClasses > 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the dependent accelerator assigned at the given index. The scheduler must
    ///             acquire an accelerator object for each depenendent accelerator class entry in
    ///             addition to the dispatch accelerator before dispatching a task. See
    ///             AddDependentAccelerator class documentation for more details.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/9/2012. </remarks>
    ///
    /// <param name="nIndex">       The index. </param>
    ///
    /// <returns>   The dependent accelerator, or null if none is assigned. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Task::GetAssignedDependentAccelerator(
        __in int nIndex
        )
    {
        int nEnumerated = 0;
        assert(LockIsHeld());
        assert(nIndex < m_vDependentAcceleratorRequirements.size());
        if(nIndex < 0 || nIndex >= m_vDependentAcceleratorRequirements.size())
            return NULL;
        std::map<ACCELERATOR_CLASS, std::vector<Accelerator*>*>::iterator mi;
        for(mi=m_vDependentAcceleratorAssignments.begin();
            mi!=m_vDependentAcceleratorAssignments.end(); 
            mi++) {
            if(nEnumerated == nIndex) {
                std::vector<Accelerator*>* pVector = mi->second;
                assert(pVector != NULL);
                assert(pVector->size() == 1);
                Accelerator * pAssignment = pVector->at(0);
                return pAssignment;
            }
            nEnumerated++;
        }
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the dependent accelerator assigned to the given port. The scheduler must
    ///             acquire an accelerator object for each depenendent accelerator class entry in
    ///             addition to the dispatch accelerator before dispatching a task. See
    ///             AddDependentAccelerator class documentation for more details.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/9/2012. </remarks>
    ///
    /// <param name="pPort">    [in,out] The port. </param>
    ///
    /// <returns>   The dependent accelerator, or null if none is assigned. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Task::GetAssignedDependentAccelerator(
        __in Port * pPort
        )    
    {        
        Lock();
        Accelerator * pResult = NULL;
        assert(pPort->HasDependentAcceleratorBinding());
        std::map<Port*, Accelerator*>::iterator mi;
        mi = m_vPortDependentAcceleratorAssignments.find(pPort);
        assert(mi!=m_vPortDependentAcceleratorAssignments.end());
        if(mi!=m_vPortDependentAcceleratorAssignments.end()) {
            assert(mi->second != NULL);
            pResult = mi->second;
        }
        Unlock();
        return pResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the target memory space for blocks bound to the given port. </summary>
    ///
    /// <remarks>   crossbac, 5/21/2012. </remarks>
    ///
    /// <param name="pPort">    [in,out] The port. </param>
    ///
    /// <returns>   null if it fails, else the target memory space. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Task::GetTargetMemorySpace(
        Port * pPort
        )
    {
        assert(pPort != NULL);
        if(!pPort->HasDependentAcceleratorBinding())
            return m_pDispatchAccelerator;
        std::map<Port*, Accelerator*>::iterator mi;
        mi = m_vPortDependentAcceleratorAssignments.find(pPort);
        if(mi==m_vPortDependentAcceleratorAssignments.end())
            return m_pDispatchAccelerator;
        return mi->second;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets target port class. If there is a dependent binding for this
    ///             port, return the appropriate class. Otherwise, return the 
    ///             task accelerator class.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/3/2013. </remarks>
    ///
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    ///
    /// <returns>   The dependent port class. </returns>
    ///-------------------------------------------------------------------------------------------------

    ACCELERATOR_CLASS 
    Task::GetPortTargetClass(
        __in Port * pPort
        )
    {
        assert(pPort != NULL);
        Task * pTask = pPort->GetTask();
        assert(pTask != NULL);
        if(!pPort->HasDependentAcceleratorBinding())
            return pTask->m_eAcceleratorClass;
        std::map<Port*, ACCELERATOR_CLASS>::iterator mi;
        mi = pTask->m_vPortDependentAcceleratorRequirements.find(pPort);
        if(mi != pTask->m_vPortDependentAcceleratorRequirements.end())
            return mi->second;
        return pTask->m_eAcceleratorClass;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the dependent accelerator class at the given index. The scheduler must
    ///             acquire an accelerator object for each depenendent accelerator class entry in
    ///             addition to the dispatch accelerator before dispatching a task. See
    ///             AddDependentAccelerator class documentation for more details.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/9/2012. </remarks>
    ///
    /// <param name="nIndex">       The index. </param>
    /// <param name="nRequired">    [in,out] The number of instances of that class required for
    ///                             dispatch. Currently required/assumed to be one. </param>
    ///
    /// <returns>   The dependent accelerator class. </returns>
    ///-------------------------------------------------------------------------------------------------

    ACCELERATOR_CLASS 
    Task::GetDependentAcceleratorClass(
        __in int nIndex, 
        __out int &nRequired 
        )
    {
        int nEnumerated = 0;
        assert(nIndex < m_vDependentAcceleratorRequirements.size());
        if(nIndex < 0 || nIndex >= m_vDependentAcceleratorRequirements.size())
            return ACCELERATOR_CLASS_UNKNOWN;

        std::map<ACCELERATOR_CLASS, int>::iterator mi;
        for(mi=m_vDependentAcceleratorRequirements.begin(); 
            mi!=m_vDependentAcceleratorRequirements.end(); 
            mi++) {
            if(nEnumerated == nIndex) {
                nRequired = mi->second;
                return mi->first;
            }
            nEnumerated++;
        }
        return ACCELERATOR_CLASS_UNKNOWN;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Assign the dependent accelerator at the given index. The scheduler must acquire
    ///             an accelerator object for each depenendent accelerator class entry in addition to
    ///             the dispatch accelerator before dispatching a task. See AddDependentAccelerator
    ///             class documentation for more details.        
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/9/2012. </remarks>
    ///
    /// <param name="idx">          The index. </param>
    /// <param name="pAccelerator"> [in] the accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::AssignDependentAccelerator(
        __in int idx, 
        __in Accelerator * pAccelerator
        )
    {
        UNREFERENCED_PARAMETER(idx);
        tpprofile_enter(AssignDependentAccelerator);
        ACCELERATOR_CLASS cls = pAccelerator->GetClass();
        std::map<ACCELERATOR_CLASS, int>::iterator mi;
        std::map<ACCELERATOR_CLASS, std::vector<Accelerator*>*>::iterator ai;
        assert(LockIsHeld());
        assert(HasDependentAcceleratorBindings());
        assert(idx == 0 && "multiple assignments of same dependent acc class are not supported!");
        mi = m_vDependentAcceleratorRequirements.find(cls);
        assert(mi != m_vDependentAcceleratorRequirements.end());
        assert(mi->second == 1 && "multiple assignments of same dependent acc class are not supported!");
        assert(m_vDependentAcceleratorAssignments.find(cls) == m_vDependentAcceleratorAssignments.end()); // already assigned?
        std::vector<Accelerator*>* pVector = new std::vector<Accelerator*>();
        pVector->push_back(pAccelerator);
        m_vAllDependentAssignments.insert(pAccelerator);
        m_vDependentAcceleratorAssignments[cls] = pVector;
        tpprofile_exit(AssignDependentAccelerator);
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases all dependent accelerators. </summary>
    ///
    /// <remarks>   crossbac, 5/9/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void  
    Task::ReleaseDependentAccelerators(
        VOID
        )
    {
        // due to lock ordering requirements, acquire and release of dependent accelerator
        // locks must be part of the protocol for acquiring all dispatch locks. Consequently,
        // the release of depenendent accelerators by the scheduler just means releasing the
        // memory acquired to hold the list of assigned dependents, because the actual locks
        // are released elsewhere.
        Lock();
        int nReleased = 0;
        std::map<ACCELERATOR_CLASS, std::vector<Accelerator*>*>::iterator ai;
        for(ai=m_vDependentAcceleratorAssignments.begin();
            ai!=m_vDependentAcceleratorAssignments.end();
            ai++) {
            assert(ai->second != NULL);                 // better be there!
            assert(ai->second->size() == 1);            // NS: multiple dependences on same resource class...
            vector<Accelerator*>* vAssigned = ai->second;
            delete vAssigned;
            nReleased++;
        }
        m_vDependentAcceleratorAssignments.clear();
        m_vPortDependentAcceleratorAssignments.clear();
        m_vAllDependentAssignments.clear();

        // if there are dependences we better have released something!
        // actually, this is ok if we are abandoning a dispatch.
        // assert(GetDependentAcceleratorClassCount() == 0 || nReleased > 0);

        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resolve dependent accelerator port bindings. Before dispatch, tasks with
    ///             dependent accelerators, (e.g. host tasks using cuda) must be bound to both
    ///             dispatch and dependent accelerators. In order to bind the appropriate platform-
    ///             specific buffer instance before the platform-specific dispatch, we need to
    ///             resolve the resource assignments from the scheduler to the actual per-port
    ///             bindings. This method looks at the dependent assignments and distributes them
    ///             accordingly across the ports which have dependent bindings.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/21/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::ResolveDependentPortBindings(
        VOID
        )
    {
        std::map<Port*, ACCELERATOR_CLASS>::iterator mi;
        std::map<ACCELERATOR_CLASS, std::vector<Accelerator*>*>::iterator ai;

        for(mi=m_vPortDependentAcceleratorRequirements.begin(); 
            mi!=m_vPortDependentAcceleratorRequirements.end(); 
            mi++) {

            Port * pBoundPort = mi->first;
            ACCELERATOR_CLASS cls = mi->second;
            assert(pBoundPort != NULL);                 // meaningless
            assert(cls != GetAcceleratorClass());       // NS: dependence on another instance of this class...            
            assert(m_vPortDependentAcceleratorAssignments.find(pBoundPort) ==
                   m_vPortDependentAcceleratorAssignments.end()); // not already assigned!

            ai = m_vDependentAcceleratorAssignments.find(cls);
            assert(ai != m_vDependentAcceleratorAssignments.end());
            assert(ai->second != NULL);                 // better be there!
            assert(ai->second->size() == 1);            // NS: multiple dependences on same resource class...
            vector<Accelerator*>* vAssigned = ai->second;

            Accelerator * pDepAcc = vAssigned->at(0);
            assert(pDepAcc != NULL);
            assert(pDepAcc->GetClass() == cls);
            m_vPortDependentAcceleratorAssignments[pBoundPort] = pDepAcc;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resolve dependent port requirements. </summary>
    ///
    /// <remarks>   crossbac, 5/21/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::ResolveDependentPortRequirements(
        VOID
        )
    {
        Lock();
        std::map<UINT, Port*>::iterator mii;
        for(mii=m_mapInputPorts.begin(); mii!=m_mapInputPorts.end(); mii++) {
            Port * pPort = mii->second;
            if(pPort->HasDependentAcceleratorBinding()) {
                ACCELERATOR_CLASS cls = pPort->GetDependentAcceleratorClass(0);
                m_vPortDependentAcceleratorRequirements[pPort] = cls;
            }
        }
        for(mii=m_mapOutputPorts.begin(); mii!=m_mapOutputPorts.end(); mii++) {
            Port * pPort = mii->second;
            if(pPort->HasDependentAcceleratorBinding()) {
                ACCELERATOR_CLASS cls = pPort->GetDependentAcceleratorClass(0);
                m_vPortDependentAcceleratorRequirements[pPort] = cls;
            }
        }
        Unlock();
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has dependent affinities. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <returns>   true if dependent affinities, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::HasDependentAffinities(
        VOID
        )
    {
        return m_bHasDependentAffinities;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a scheduler hint for this task. This is largely an experimental interface to
    ///             provide a way for the programmer to give the scheduler hints about how it can
    ///             deal with large graphs. Generally PTask hopes to find an optimal or near optimal
    ///             schedule (by which we mean mapping from tasks to compute resources) dynamically
    ///             by reacting to performance history approximations and dynamic data movement. With
    ///             large, or complex graphs, this does not always work for a few reasons: 
    ///             
    ///             1. When PTask thinks a particular task should go on an accelerator because its  
    ///                inputs are already materialized there, and that accelerator is unavailable, 
    ///                it blocks that task for a while (with a configurable threshold) but then eventually
    ///                schedules it someplace non-optimal to avoid starving the task. With a large graph,
    ///                time spent in the ready Q can be long, and the starvation avoidance mechanism
    ///                winds up causing compromises in locality that can propagate deep into the graph. 
    ///                
    ///             2. The experimental scheduler that does a brute force search of the assignment space   
    ///                has combinatorial complexity. Searching the entire space simply takes too long. 
    ///             
    ///             3. The graph partitioning scheduler relies on the presence of block pools at   
    ///                exposed channels as a heuristic to find partition points. This works, but only
    ///                if the programmer has actually bothered to configure pools there. 
    ///                
    ///             While we are working on better graph partitioning algorithms, if the programmer
    ///             actually knows something that can help the scheduler, there ought to be a way
    ///             to let PTask know. 
    ///             
    ///             PTask interprets the uiPartitionHint as a unique identifier for the partition 
    ///             the graph should be mapped to (where all tasks in a partion should have mandatory
    ///             affinity for the same GPU), and interprets the user cookie pointer as an opaque
    ///             pointer to any additional data structure that a particular (modularized) scheduler 
    ///             can use to maintain state.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/28/2013. </remarks>
    ///
    /// <param name="uiSchedulerHint">  The scheduler hint. </param>
    /// <param name="lpvSchedulerCookie">    [in,out] If non-null, the lpv user cookie. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SetSchedulerPartitionHint(
        __in UINT   uiPartitionHint,
        __in void * lpvSchedulerCookie
        )
    {
        m_uiSchedulerPartitionHint = uiPartitionHint;
        m_lpvSchedulerCookie = lpvSchedulerCookie;
        m_bSchedulerPartitionHintSet = TRUE;
        assert(m_pGraph != NULL);
        if(m_pGraph != NULL)
            m_pGraph->SetHasSchedulerPartitionHints(TRUE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets any scheduler hint for this task. This is largely an experimental interface
    ///             to provide a way for the programmer to give the scheduler hints about how it can
    ///             deal with large graphs. Generally PTask hopes to find an optimal or near optimal
    ///             schedule (by which we mean mapping from tasks to compute resources) dynamically
    ///             by reacting to performance history approximations and dynamic data movement. With
    ///             large, or complex graphs, this does not always work for a few reasons:
    ///             
    ///             1. When PTask thinks a particular task should go on an accelerator because its  
    ///                inputs are already materialized there, and that accelerator is unavailable, it
    ///                blocks that task for a while (with a configurable threshold) but then
    ///                eventually schedules it someplace non-optimal to avoid starving the task. With
    ///                a large graph, time spent in the ready Q can be long, and the starvation
    ///                avoidance mechanism winds up causing compromises in locality that can
    ///                propagate deep into the graph.
    ///             
    ///             2. The experimental scheduler that does a brute force search of the assignment
    ///             space
    ///                has combinatorial complexity. Searching the entire space simply takes too long.
    ///             
    ///             3. The graph partitioning scheduler relies on the presence of block pools at
    ///                exposed channels as a heuristic to find partition points. This works, but only
    ///                if the programmer has actually bothered to configure pools there.
    ///             
    ///             While we are working on better graph partitioning algorithms, if the programmer
    ///             actually knows something that can help the scheduler, there ought to be a way to
    ///             let PTask know.
    ///             
    ///             PTask interprets the uiPartitionHint as a unique identifier for the partition the
    ///             graph should be mapped to (where all tasks in a partion should have mandatory
    ///             affinity for the same GPU), and interprets the user cookie pointer as an opaque
    ///             pointer to any additional data structure that a particular (modularized)
    ///             scheduler can use to maintain state.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/28/2013. </remarks>
    ///
    /// <param name="lppvSchedulerCookie">   If non-null, on exit will contain a pointer to the user
    ///                                     cookie pointer provided in the corresponding setter. </param>
    ///
    /// <returns>   The scheduler partition hint. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Task::GetSchedulerPartitionHint(
        __out void ** lppvSchedulerCookie
        )
    {
        UINT uiHint = 0;
        if(lppvSchedulerCookie) 
            *lppvSchedulerCookie = NULL;
        if(m_bSchedulerPartitionHintSet)  {
            if(lppvSchedulerCookie != NULL) 
                *lppvSchedulerCookie = m_lpvSchedulerCookie;
            uiHint = m_uiSchedulerPartitionHint;
        }
        return uiHint;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this task has scheduler partition hint. </summary>
    ///
    /// <remarks>   crossbac, 6/28/2013. </remarks>
    ///
    /// <returns>   true if scheduler partition hint, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Task::HasSchedulerPartitionHint(
        VOID
        )
    {
        return m_bSchedulerPartitionHintSet;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   If affinity for a particular accelerator has been configured for this
    ///             task, either on a dependent port or otherwise, return it. This should
    ///             not be used by the scheduler, since affinity for more than one accelerator
    ///             is possible either because of soft affinity or because the task needs
    ///             multiple accelerator classes (e.g. a host task with affinity for a particular
    ///             CPU and a particular dependent GPU). This method is a helper for heuristics
    ///             and visualizers. Achtung. </summary>
    ///
    /// <remarks>   Crossbac, 11/15/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the first affinitized accelerator found. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Task::GetAffinitizedAcceleratorHint(
        void
        )
    {
        if(m_pMandatoryAccelerator != NULL)
            return m_pMandatoryAccelerator;
        if(m_bHasDependentAffinities) {
            if(!m_bMandatoryDependentAcceleratorValid) 
                PopulateConstraintsCaches();
            if(m_pMandatoryDependentAccelerator != NULL)
                return m_pMandatoryDependentAccelerator;
            if(m_vPreferenceConstraintSet.size()) 
                return *(m_vPreferenceConstraintSet.begin());
        }
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets mandatory dependent accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 11/15/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the mandatory dependent accelerator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Task::GetMandatoryDependentAccelerator(
        VOID
        )
    {
        if(HasMandatoryDependentAffinities())
            return m_pMandatoryDependentAccelerator;
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'uiIndex' has mandatory dependent affinities. </summary>
    ///
    /// <remarks>   Crossbac, 11/15/2013. </remarks>
    ///
    /// <returns>   true if mandatory dependent affinities, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::HasMandatoryDependentAffinities(
        VOID
        )
    {
        if(m_bHasDependentAffinities) {
            if(!m_bMandatoryDependentAcceleratorValid) 
                PopulateConstraintsCaches();
            return m_pMandatoryDependentAccelerator != NULL;
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   ToString() operator. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="os">       [in,out] The operating system. </param>
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///
    /// <returns>   The shifted result. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::ostream& operator <<(std::ostream &os, Task * pTask) { 
        if(pTask == NULL) {
            os << "task:null";
            return os;
        }
        const char * lpszTask = const_cast<const char*>(pTask->GetTaskName());
        os << lpszTask;
        return os;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps dispatch information. </summary>
    ///
    /// <remarks>   crossbac, 6/6/2012. </remarks>
    ///
    /// <param name="nIteration">       The iteration. </param>
    /// <param name="nMaxIterations">   The maximum iterations. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Task::DumpDispatchInfo(
        UINT nIteration,
        UINT nMaxIterations
        )
    {
        if(Runtime::GetDispatchLoggingEnabled()) {
            if(nMaxIterations == DEFAULT_DISPATCH_ITERATIONS) {                                    
                std::cout                                                             
                    << "T[" << std::hex << std::setw(4)                               
                    << ::GetCurrentThreadId()                                         
                    << std::dec                                                       
                    << "] dispatch: "      
                    << m_pGraph->GetName() 
                    << "::"
                    << m_lpszTaskName
                    << "(dev:"                                                        
                    << m_pDispatchAccelerator->GetAcceleratorId();
                if(m_vDependentAcceleratorAssignments.size()) {
                    bool bFirst = true;
                    std::cout << " dep[";
                    std::map<ACCELERATOR_CLASS, std::vector<Accelerator*>*>::iterator si; 
                    for(si=m_vDependentAcceleratorAssignments.begin();                    
                        si!=m_vDependentAcceleratorAssignments.end(); si++) {             
                        std::vector<Accelerator*>::iterator vi;        
                        for(vi=si->second->begin();vi!=si->second->end();vi++) {
                            if(!bFirst)
                                std::cout << ",";
                            bFirst = false;
                            std::cout << (*vi)->GetAcceleratorId();            
                        }                                                                 
                    }                                                                     
                    std::cout << "]";  
                }
                std::cout                                                             
                    << ")"                                                           
                    << endl;                                                          
            } else {                                                                  
                std::cout                                                             
                    << "dispatch: "                                                   
                    << this->m_lpszTaskName                                           
                    << "(si=" << (nIteration) << ")"                                           
                    << endl;                                                          
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initialises the dispatch counting. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::InitializeInstanceDispatchCounter(
        VOID
        )
    {
        assert(m_pDispatchCounter == NULL);
#ifdef DISPATCH_COUNT_DIAGNOSTICS
        m_pDispatchCounter = new DispatchCounter(this);
#endif
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Verify dispatch counts against a prediction for every task in the graph. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="pvInvocationCounts">   [in,out] If non-null, the pv invocation counts. </param>
    ///
    /// <returns>   true if the actual and predicted match, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Task::VerifyDispatchCounts(
        std::map<std::string, UINT> * pvInvocationCounts
        )
    {
        if(!Runtime::g_bTaskProfile || pvInvocationCounts == NULL) return FALSE;
        return DispatchCounter::Verify(pvInvocationCounts); 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::RecordDispatch(
        VOID
        )
    {
        if(Runtime::g_bTaskProfile && m_pDispatchCounter != NULL) 
            m_pDispatchCounter->RecordDispatch();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an expected dispatch count. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="nCount">   Number of. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Task::SetExpectedDispatchCount(
        UINT nCount
        )
    {
        if(Runtime::g_bTaskProfile && m_pDispatchCounter != NULL) 
            m_pDispatchCounter->SetExpectedDispatchCount(nCount);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Find all ports within this given map containing non-zero control codes. </summary>
    ///
    /// <remarks>   Crossbac, 3/2/2012. </remarks>
    ///
    /// <param name="vPortMap"> [in,out] [in,out] If non-null, the port map. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::CheckPortControlCodes(
        std::map<UINT, Port*> &vPortMap
        )
    {
        if(Runtime::g_bTaskProfile && m_pDispatchCounter != NULL) {
            map<UINT, Port*>::iterator mi;
            for(mi=vPortMap.begin(); mi!=vPortMap.end(); mi++) {
                Port * pPort = mi->second;
                pPort->CheckControlCodes();
            }
        }
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
    Task::CheckPortControlCodes(
        VOID
        )
    {
        if(Runtime::g_bTaskProfile && m_pDispatchCounter != NULL) {
            CheckPortControlCodes(m_mapInputPorts);
            CheckPortControlCodes(m_mapOutputPorts);
            CheckPortControlCodes(m_mapConstantPorts);
            CheckPortControlCodes(m_mapMetaPorts);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check that block pools contain only datablocks with no control signals. </summary>
    ///
    /// <remarks>   Crossbac, 3/2/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::CheckBlockPoolStates(
        VOID
        )
    {
        if(Runtime::g_bTaskProfile && m_pDispatchCounter != NULL) {
            map<UINT, Port*>::iterator mi;
            for(mi=m_mapOutputPorts.begin(); mi!=m_mapOutputPorts.end(); mi++) {
                OutputPort * pPort = dynamic_cast<OutputPort*>(mi->second);
                pPort->CheckBlockPoolStates();
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Get the CompiledKernel associated with this task. </summary>
    ///
    /// <remarks>   jcurrey, 5/5/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    CompiledKernel *
    Task::GetCompiledKernel()
    {
        return m_pCompiledKernel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the task profiling. </summary>
    ///
    /// <remarks>   Crossbac, 9/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::InitializeTaskProfiling(
        BOOL bTabularOutput
        )
    {
        TaskProfile::Initialize(bTabularOutput);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deinitialize task profiling. </summary>
    ///
    /// <remarks>   Crossbac, 9/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::DeinitializeTaskProfiling(
        VOID
        )
    {
        TaskProfile::Deinitialize();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the task profiling. </summary>
    ///
    /// <remarks>   Crossbac, 9/21/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::InitializeTaskInstanceProfile(
        VOID
        )
    {
        Lock();
        assert(m_pTaskProfile == NULL);
        if(PTask::Runtime::GetTaskProfileMode() && (m_pTaskProfile == NULL)) {
            m_pTaskProfile = new TaskProfile(this);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deinitialize task profiling. </summary>
    ///
    /// <remarks>   Crossbac, 9/21/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::DeinitializeTaskInstanceProfile(
        VOID
        )
    {
        Lock();
        assert(m_pTaskProfile == NULL || PTask::Runtime::GetTaskProfileMode());
        if(PTask::Runtime::GetTaskProfileMode() && (m_pTaskProfile != NULL)) {
            m_pTaskProfile->DeinitializeInstanceProfile();
            delete m_pTaskProfile;
            m_pTaskProfile = NULL;
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Merge task instance statistics. </summary>
    ///
    /// <remarks>   Crossbac, 9/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::MergeTaskInstanceStatistics(
        VOID
        )
    {
        if(PTask::Runtime::GetTaskProfileMode() && (m_pTaskProfile != NULL)) 
            m_pTaskProfile->MergeTaskInstanceStatistics();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps a task instance profile statistics. </summary>
    ///
    /// <remarks>   Crossbac, 7/17/2013. </remarks>
    ///
    /// <param name="ss">   [in,out] The ss. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::DumpTaskInstanceProfile(
        std::ostream& ss
        )
    {
        if(PTask::Runtime::GetTaskProfileMode() && (m_pTaskProfile != NULL)) 
            m_pTaskProfile->DumpTaskProfile(ss);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Provide an API for letting the user tell us if a task implementation
    ///             allocates memory through APIs that are not visible to PTask. The canonical
    ///             examples of this are HostTasks that use thrust or cublas: if there is 
    ///             high memory pressure, ptask can sometimes avert an OOM in user code by forcing
    ///             a gc sweep before attempting a dispatch. We can only do this if the programmer
    ///             lets us know that temporary space will be required.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 9/10/2013. </remarks>
    ///
    /// <param name="bState">   true to state. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::SetUserCodeAllocatesMemory(
        BOOL bState
        )
    {
        // no lock required since graph construction
        // is single-threaded and this member is assumed to
        // be immutable once the graph is running
        m_bUserCodeAllocatesMemory = bState;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return true if a task implementation
    ///             allocates memory through APIs that are not visible to PTask. The canonical
    ///             examples of this are HostTasks that use thrust or cublas: if there is 
    ///             high memory pressure, ptask can sometimes avert an OOM in user code by forcing
    ///             a gc sweep before attempting a dispatch. We can only do this if the programmer
    ///             lets us know that temporary space will be required.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 9/10/2013. </remarks>
    ///
    /// <param name="bState">   true to state. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::UserCodeAllocatesMemory(
        void
        )
    {
        // no lock required since this is assumed to
        // be immutable once the graph is running
        return m_bUserCodeAllocatesMemory;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check if memory pressure warrants a pre-dispatch GC sweep. Currently this
    ///             only occurs for tasks that are explicitly marked as allocators of temporary
    ///             buffers in user code (ie. allocation that is invisible to ptask), and is an
    ///             attempt to avert OOMs that we cannot catch and remedy after the fact (OOM
    ///             in user-code will cause an exception that we can catch, but which cannot
    ///             recover from).
    ///             </summary>
    ///
    /// <remarks>   crossbac, 9/10/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Task::PreDispatchMemoryPressureCheck(
        void
        )
    {
        // we are only interested in tasks for which it is possible
        // for user code to do memory allocation. For all other tasks,
        // memory allocation is done by PTask, so we can address failures
        // post-facto for those. Only host tasks marked as allocators
        // require this check. 
        if(m_eAcceleratorClass != ACCELERATOR_CLASS_HOST ||
           !HasDependentAcceleratorBindings() ||
           !UserCodeAllocatesMemory()) 
           return;

        assert(m_vAllDependentAssignments.size() == 1);
        Accelerator * pDepAcc = *(m_vAllDependentAssignments.begin());
        assert(pDepAcc != NULL);
        assert(!pDepAcc->IsHost());
        UINT uiMemSpaceID = pDepAcc->GetMemorySpaceId();
        UINT uiAllocPercent = MemorySpace::GetAllocatedPercent(uiMemSpaceID);
        UINT uiThresholdPercent = Runtime::GetGCSweepThresholdPercent();

        PTask::Runtime::Inform("%s::%s(memspace_%d)--%d%% allocated\n",
                               __FILE__,
                               __FUNCTION__,
                               uiMemSpaceID,
                               uiAllocPercent);

        if(uiAllocPercent >= uiThresholdPercent) {
        
            PTask::Runtime::MandatoryInform("%s::%s(memspace_%d)--%d%% allocated...forcing GC sweep...",
                                            __FILE__,
                                            __FUNCTION__,
                                            uiMemSpaceID,
                                            uiAllocPercent);
            DWORD dwGCStart = GetTickCount();
            GarbageCollector::ForceGC(uiMemSpaceID);
            DWORD dwSweepTicks = GetTickCount() - dwGCStart;
            UINT uiNewAllocPercent = MemorySpace::GetAllocatedPercent(uiMemSpaceID);
            PTask::Runtime::MandatoryInform("%d%% allocated (%d%% recovered in %d ms)\n",
                                            uiNewAllocPercent,
                                            uiAllocPercent-uiNewAllocPercent, 
                                            dwSweepTicks);


            if(uiNewAllocPercent > 95) {
                PTask::Runtime::MandatoryInform("%s::%s(memspace_%d)--%d%% allocated...GC sweep ineffective...dumping DB alloc state...\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                uiMemSpaceID,
                                                uiNewAllocPercent);
                DatablockProfiler::Report(std::cout);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets all inbound channels. </summary>
    ///
    /// <remarks>   crossbac, 7/3/2014. </remarks>
    ///
    /// <param name="vChannels">    [in,out] [in,out] If non-null, the channels. </param>
    ///
    /// <returns>   The inbound channels. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Task::GetInboundChannels(
        __inout std::set<Channel*>& vChannels
        )
    {
        std::map<UINT, Port*>::iterator mpi;
        std::set<std::map<UINT, Port*>*> vPortMaps;
        std::set<std::map<UINT, Port*>*>::iterator mi;
        vPortMaps.insert(GetInputPortMap());
        vPortMaps.insert(GetConstantPortMap()); 
        vPortMaps.insert(GetMetaPortMap());

        for(mi=vPortMaps.begin(); mi!=vPortMaps.end(); mi++) {
            
            std::map<UINT, Port*>* pInputPortMap = *mi;        
            for(mpi=pInputPortMap->begin(); mpi!=pInputPortMap->end(); mpi++) {

                Port * pIPort = mpi->second;
                UINT uiChannelCount = pIPort->GetChannelCount();
                UINT uiControlCount = pIPort->GetControlChannelCount(); 
                for(UINT ui=0; ui<uiChannelCount; ui++) 
                    vChannels.insert(pIPort->GetChannel(ui));
                for(UINT ui=0; ui<uiControlCount; ui++) 
                    vChannels.insert(pIPort->GetControlChannel(ui));
            }
        }
        return static_cast<UINT>(vChannels.size());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the set of all outbound channels. </summary>
    ///
    /// <remarks>   crossbac, 7/3/2014. </remarks>
    ///
    /// <param name="vChannels">    [in,out] [in,out] If non-null, the channels. </param>
    ///
    /// <returns>   The outbound channels. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Task::GetOutboundChannels(
        __inout std::set<Channel*>& vChannels
        )
    {
        std::map<UINT, Port*>::iterator mpi;
        std::map<UINT, Port*>* pOutputPortMap = GetOutputPortMap(); 
        for(mpi=pOutputPortMap->begin(); mpi!=pOutputPortMap->end(); mpi++) {
            Port * pOPort = mpi->second; 
            UINT uiChannelCount = pOPort->GetChannelCount(); 
            for(UINT ui=0; ui<uiChannelCount; ui++) 
                vChannels.insert(pOPort->GetChannel(ui));
        }
        return static_cast<UINT>(vChannels.size());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the super-set of all "control signals of interest" for this graph object.  
    ///             A control signal is "of interest" if the behavior of this object is is predicated
    ///             in some way by the presence or absence of a given signal. This function returns
    ///             the bit-wise OR of all such signals.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/7/2014. </remarks>
    ///
    /// <returns>   The bitwise OR of all found control signals of interest. </returns>
    ///-------------------------------------------------------------------------------------------------

    CONTROLSIGNAL 
    Task::GetControlSignalsOfInterest(
        VOID
        )
    {
        CONTROLSIGNAL luiSignals = DBCTLC_NONE;
        luiSignals |= GetControlSignalsOfInterest(m_mapInputPorts);
        luiSignals |= GetControlSignalsOfInterest(m_mapConstantPorts);
        luiSignals |= GetControlSignalsOfInterest(m_mapMetaPorts);
        luiSignals |= GetControlSignalsOfInterest(m_mapOutputPorts);        
        return luiSignals;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the super-set of all "control signals of interest" for this graph object.  
    ///             A control signal is "of interest" if the behavior of this object is is predicated
    ///             in some way by the presence or absence of a given signal. This function returns
    ///             the bit-wise OR of all such signals.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/7/2014. </remarks>
    ///
    /// <returns>   The bitwise OR of all found control signals of interest. </returns>
    ///-------------------------------------------------------------------------------------------------

    CONTROLSIGNAL 
    Task::GetControlSignalsOfInterest(
        __in std::map<UINT, Port*>& pPortMap
        )
    {
        CONTROLSIGNAL luiSignals = DBCTLC_NONE;
        std::map<UINT, Port*>::iterator mpi;
        for(mpi=pPortMap.begin(); mpi!=pPortMap.end(); mpi++) {
            Port * pPort = mpi->second;
            luiSignals |= pPort->GetControlSignalsOfInterest();
        }
        return luiSignals;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this task is a "terminal". Generally, a task is a terminal if it has
    ///             exposed input or output channels. More precisely, a task is terminal if it is a
    ///             sink or source for any channels whose ability to produce or consume blocks is
    ///             under control of the user program. For input and output channels, this is an
    ///             obvious property. Initializer channels with predicates may behave just like an
    ///             input channel since their ability to produce blocks for a task is controlled by
    ///             predicates on blocks produced by the user program. Consequently, a task with
    ///             predicated initializers but no actual exposed input channels is also a terminal.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/7/2014. </remarks>
    ///
    /// <returns>   true if terminal, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Task::IsTerminal(
        VOID
        )
    {
        BOOL bTerminal = FALSE;
        MultiChannel * pMChannel = NULL;
        std::set<Channel*>::iterator ci;
        std::set<Channel*> vChannels;
        GetInboundChannels(vChannels);
        GetOutboundChannels(vChannels);
        for(ci=vChannels.begin(); ci!=vChannels.end(); ci++) {
            CHANNELTYPE eType = (*ci)->GetType();
            switch(eType) {
            case CHANNELTYPE::CT_INTERNAL: continue;
            case CHANNELTYPE::CT_GRAPH_INPUT: return TRUE;
            case CHANNELTYPE::CT_GRAPH_OUTPUT: return TRUE;
            case CHANNELTYPE::CT_INITIALIZER: if((*ci)->HasNonTrivialPredicate()) return TRUE; continue;
            case CHANNELTYPE::CT_MULTI: 
                pMChannel = dynamic_cast<MultiChannel*>(*ci);
                if(pMChannel && pMChannel->HasExposedComponentChannel())
                    return TRUE;
                continue;
            }
        }
        return bTerminal;
    }

};

