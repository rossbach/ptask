///-------------------------------------------------------------------------------------------------
// file:	PTaskRuntime.cpp
//
// summary:	Implements the task runtime class
///-------------------------------------------------------------------------------------------------

#include "AsyncContext.h"
#include "PTaskRuntime.h"
#include "InputPort.h"
#include "OutputPort.h"
#include "StickyPort.h"
#include "MetaPort.h"
#include "InitializerPort.h"
#include "ptaskutils.h"
#include "PTaskRuntime.h"
#include "task.h"
#include "graph.h"
#include "Scheduler.h"
#include "ptgc.h"
#include <assert.h>
#include <set>
#include <fstream>
#include <iostream>
#include <string>
#include "MemorySpace.h"
#include "Tracer.h"
#include "shrperft.h"
#include "ptasklynx.h"
#include "ptprofsupport.h"
#include "nvtxmacros.h"
#include "taskprofiler.h"
#include "dispatchcounter.h"
#include "RefCountProfiler.h"
#include "ChannelProfiler.h"
#include "CoherenceProfiler.h"
#include "DatablockProfiler.h"
#include "GraphProfiler.h"
#include "CompiledKernel.h"
#include "datablocktemplate.h"
#include "graphInputChannel.h"
#include "graphOutputChannel.h"
#include "internalChannel.h"
#include "InitializerChannel.h"
#include "multichannel.h"
#include "Recorder.h"
#include "instrumenter.h"
#include "GlobalPoolManager.h"
#include "GlobalBlockPool.h"
#include "ThreadPool.h"
#include "signalprofiler.h"
#include <map>
#include <tuple>
using namespace std;	

DECLARE_NVTX_GLOBALS();
DECLARE_NVTX_INIT();
#define PTVALIDSYNCHANDLE(x) (((x)!=INVALID_HANDLE_VALUE) && ((x)!=NULL))

namespace PTask { 

    namespace Runtime {

        const BOOL                          DEFAULT_GRAPHS_ARE_MUTABLE = FALSE;
        const BOOL                          DEFAULT_APP_THREADS_MANAGE_PRIMARY_CONTEXT = FALSE;
        const UINT                          DEFAULT_ASYNC_CONTEXT_GC_QUERY_THRESHOLD = 10;
        const BOOL                          DEFAULT_SCHEDULE_CHECKS_CLASS_AVAILABILITY = FALSE;
        const int                           DEFAULT_DATA_BUFFER_SIZE = 1024 * 1024 * 16;
        const int                           DEFAULT_META_BUFFER_SIZE = 1024 * 32;
        const int                           DEFAULT_TEMPLATE_BUFFER_SIZE = 1024;
        const GRAPHASSIGNMENTPOLICY         DEFAULT_GRAPH_ASSIGNMENT_POLICY = GMP_USER_DEFINED;
        const BLOCKPOOLRESIZEPOLICY         DEFAULT_BLOCK_POOL_BLOCK_RESIZE_POLICY = BPRSP_EXIT_POOL;
        const BLOCKRESIZEMEMSPACEPOLICY     DEFAULT_BLOCK_RESIZE_MEMSPACE_POLICY = BRSMSP_RELEASE_DEVICE_BUFFERS;
        const SCHEDULINGMODE                DEFAULT_SCHEDMODE = SCHEDMODE_DATADRIVEN;
        const int                           DEFAULT_MAX_CONCURRENCY = 0;
		const int							DEFAULT_HOST_CONCURRENCY = 4;
        const BOOL                          DEFAULT_BLOCK_POOLS_ENABLED = TRUE;
        const BOOL                          DEFAULT_DEBUG_MODE = FALSE;
        const int                           DEFAULT_GCBATCH_SIZE = 20;
        const int                           DEFAULT_CHANNEL_CAPACITY = 4;
        const int                           DEFAULT_INTERNAL_BLOCK_POOL_SIZE = DEFAULT_CHANNEL_CAPACITY;
        const UINT                          DEFAULT_INIT_CHANNEL_POOL_SIZE = 4;
        const UINT                          DEFAULT_INPUT_CHANNEL_POOL_SIZE = 4;
        const UINT                          DEFAULT_BLOCK_POOL_GROW_INCREMENT = 4;
        const UINT                          DEFAULT_LOGGING_LEVEL = 0;
        const BOOL                          DEFAULT_DISPATCH_LOGGING = FALSE;
        const BOOL                          DEFAULT_DISPATCH_TRACING = FALSE;
        const BOOL                          DEFAULT_FORCE_SYNCHRONOUS = FALSE;
        const BOOL                          DEFAULT_EXTREME_TRACE = FALSE;
        const BOOL                          DEFAULT_COHERENCE_PROFILE = FALSE;
        const BOOL                          DEFAULT_TASK_PROFILE = FALSE;
        const BOOL                          DEFAULT_PAGE_LOCKING_ENABLED = TRUE;
        const BOOL                          DEFAULT_PAGE_LOCKING_AGGRESSIVE = FALSE;
        const BOOL                          DEFAULT_SIGNAL_PROFILING_ENABLED = FALSE;
        const BOOL                          DEFAULT_DEBUG_ASYNCHRONY = FALSE;
        const BOOL                          DEFAULT_PROFILE_PBUFFERS = FALSE;
        const BOOL                          DEFAULT_EAGER_META_PORTS = FALSE;
        const UINT                          DEFAULT_THREAD_POOL_SIZE = 4;                           
        const UINT                          DEFAULT_THREAD_POOL_PER_THREAD_MAX_THRESHOLD = 32;      // probably should set this based on CPU-count!
        const DWORD                         DEFAULT_GRAPH_WATCHDOG_THRESHOLD = 10000;
        const UINT                          DEFAULT_SCHEDULER_THREADS = 1;
        const BOOL                          DEFAULT_SET_CUDA_HEAP_SIZE = FALSE;
        const UINT                          DEFAULT_CUDA_HEAP_SIZE = 1024*1024*8; // 8MB, the real CUDA default
        const BOOL                          DEFAULT_TRACK_DEVICE_ALLOCATION = FALSE;
        const BOOL                          DEFAULT_USE_GRAPH_WATCHDOG = FALSE;
        const BOOL                          DEFAULT_TASK_PROFILE_VERBOSE = TRUE;
        const BOOL                          DEFAULT_TASK_POOL_READY_SORT_BY_PRIO = TRUE;
        const GRAPHPARTITIONINGMODE         DEFAULT_GRAPH_PARTIONING_MODE = GRAPHPARTITIONINGMODE_NONE;
        const BOOL                          DEFAULT_PROVISION_POOLS_FOR_CAPACITY = TRUE;
        const BOOL                          DEFAULT_USE_REFERENCE_DRIVERS = FALSE;
        const int                           DEFAULT_MIN_DIRECT_X_FEATURE_LEVEL = 11;
        const BOOL                          DEFAULT_LOCK_INCOMING_ASYNC_SOURCES = FALSE;
        const BOOL                          DEFAULT_TASK_POOL_SIGNAL_PER_WORKER = TRUE;
        const BOOL                          DEFAULT_READY_CHECK_INCLUDES_INCOMING_ASYNC_DEPS = FALSE;
        const BOOL                          DEFAULT_LOCKLESS_INCOMING_ASYNC_WAIT = TRUE;
        const BOOL                          DEFAULT_REFCOUNT_PROFILING_ENABLED = FALSE;
        const BOOL                          DEFAULT_DATABLOCK_PROFILING_ENABLED = FALSE;
        const BOOL                          DEFAULT_COHERENCE_PROFILING_ENABLED = FALSE;
        const BOOL                          DEFAULT_TASK_PROFILING_ENABLED = FALSE;
        const BOOL                          DEFAULT_PBUFFER_PROFILING_ENABLED = FALSE;
        const BOOL                          DEFAULT_INVOCATION_COUNTING_ENABLED = FALSE;
        const BOOL                          DEFAULT_CHANNEL_PROFILING_ENABLED = FALSE;
        const BOOL                          DEFAULT_BLOCKPOOL_PROFILING_ENABLED = FALSE;
        const BOOL                          DEFAULT_DEBUG_LOGGING_ENABLED = FALSE;
        const VIEWMATERIALIZATIONPOLICY     DEFAULT_VIEWMATERIALIZATIONPOLICY = VIEWMATERIALIZATIONPOLICY_ON_DEMAND;
        const VIEWMATERIALIZATIONPOLICY     DEFAULT_OUTPUTVIEWMATERIALIZATIONPOLICY = VIEWMATERIALIZATIONPOLICY_EAGER;
        const THREADPOOLPOLICY              DEFAULT_THREAD_POOL_POLICY = TPP_AUTOMATIC;
        const BOOL                          DEFAULT_PBUFFER_CLEAR_ON_CREATE = TRUE;
        const BOOL                          DEFAULT_PROFILE_PS_DISPATCH = FALSE;
        const UINT                          DEFAULT_GLOBAL_THREAD_POOL_SIZE = DEFAULT_THREAD_POOL_SIZE;
        const BOOL                          DEFAULT_GLOBAL_THREAD_POOL_GROWABLE = FALSE;
        const BOOL                          DEFAULT_PRIME_GLOBAL_THREAD_POOL = TRUE;
        const BOOL                          DEFAULT_AGGRESSIVE_RELEASE_MODE = FALSE;
        const UINT                          DEFAULT_GC_SWEEP_THRESHOLD_PERCENT = 85;
        const BOOL                          DEFAULT_EXIT_ON_RUNTIME_ERROR = TRUE;
        const BOOL                          DEFAULT_WARN_CRITICAL_PATH_ALLOC = 
#ifdef CHECK_CRITICAL_PATH_ALLOC
                                                        TRUE;
#else
                                                        FALSE;
#endif
        const BOOL                          DEFAULT_ADHOC_INSTRUMENTATION_ENABLED = 
#ifdef ADHOC_STATS
                                                        TRUE;
#else
                                                        FALSE;
#endif
        const BOOL                          DEFAULT_VERBOSE = 
#ifdef DEBUG
                                                        TRUE;
#else
                                                        FALSE;
#endif
        const DATABLOCKAFFINITYPOLICY       DEFAULT_DATABLOCK_AFFINITY_POLICY = DAP_BOUND_TASK_AFFINITY;
        const BOOL                          DEFAULT_HARMONIZE_INIITIAL_VALUE_COHERENCE_STATE = TRUE;

        /// <summary> A cache of Datablock templates in the system </summary>
        static std::set<DatablockTemplate*> g_vTemplates;

        /// <summary> A cache of all compiled kernels </summary>
        static std::vector<CompiledKernel*> g_vKernels;

        /// <summary> true if the PTask runtime is initialized </summary>
        BOOL g_bPTaskInitialized = FALSE;

        /// <summary> Global uid counter </summary>
        UINT g_uiUIDCounter = 0;

        /// <summary> Lock for global uid counter</summary>
        CRITICAL_SECTION csUID;

        /// <summary> true to enable, false to disable use of reference drivers </summary>
        BOOL g_bEnableReferenceDrivers = DEFAULT_USE_REFERENCE_DRIVERS;

        /// <summary> The minimum direct x feature level to accept when 
        /// 		  creating DXAcclerator objects.
        /// 		  </summary>
        int g_nMinimumDirectXFeatureLevel = DEFAULT_MIN_DIRECT_X_FEATURE_LEVEL;

        /// <summary> The logging level. Inform calls with lower
        /// 		  logging level will produce no output. </summary>
        UINT g_nLoggingLevel = DEFAULT_LOGGING_LEVEL;

        /// <summary> true to put the runtime in verbose mode.</summary>
        BOOL g_bVerbose = DEFAULT_VERBOSE;

        /// <summary>   lock protecting the subsystem trace list. </summary>
        CRITICAL_SECTION g_csTraceList;

        /// <summary>   The set of subsystems for which trace is active. </summary>
        std::set<std::string> g_vActiveTraceSystems;

        BOOL g_bExitOnRuntimeFailure = TRUE;

        DEBUGDUMPTYPE g_nDefaultDumpType = dt_float;
        int g_nDefaultDumpStride = 1;
        int g_nDefaultDumpLength = 16;

        /// <summary>   Handle of the runtime terminate event. </summary>
        HANDLE g_hRuntimeTerminateEvent = INVALID_HANDLE_VALUE;
        HANDLE g_hRuntimeInitialized = INVALID_HANDLE_VALUE;
        HANDLE g_hRuntimeMutex = INVALID_HANDLE_VALUE;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Warning for setting global runtime settings 
        /// 			after the runtime is already initialized. 
        /// 			In such a case the setting will have no affect
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="strCaller">    [in] non-null, the calling function. </param>
        ///-------------------------------------------------------------------------------------------------

        void warnIfInitialized(
            char * strCaller
            )
        {
            if(g_bPTaskInitialized) {
                std::cerr 
                    << "WARNING: " 
                    << strCaller 
                    << " has no effect after runtime is initialized!"
                    << std::endl;
            }
        }

        BOOL g_bGraphsAreMutable = DEFAULT_GRAPHS_ARE_MUTABLE;
        BOOL g_bAppThreadsManagePrimaryContext = DEFAULT_APP_THREADS_MANAGE_PRIMARY_CONTEXT;
        BOOL g_bScheduleChecksClassAvailability = DEFAULT_SCHEDULE_CHECKS_CLASS_AVAILABILITY;
        GRAPHASSIGNMENTPOLICY g_eGraphAssignmentPolicy = DEFAULT_GRAPH_ASSIGNMENT_POLICY;
        BLOCKPOOLRESIZEPOLICY g_eBlockPoolBlockResizePolicy = DEFAULT_BLOCK_POOL_BLOCK_RESIZE_POLICY;
        BLOCKRESIZEMEMSPACEPOLICY g_eBlockResizeMemspacePolicy = DEFAULT_BLOCK_RESIZE_MEMSPACE_POLICY;
        DATABLOCKAFFINITYPOLICY g_eDatablockAffinityPolicy = DEFAULT_DATABLOCK_AFFINITY_POLICY;
        UINT g_uiGCSweepThresholdPercent = DEFAULT_GC_SWEEP_THRESHOLD_PERCENT;
        UINT g_uiGlobalThreadPoolSize = DEFAULT_GLOBAL_THREAD_POOL_SIZE;
        BOOL g_bPrimeGlobalThreadPool = DEFAULT_PRIME_GLOBAL_THREAD_POOL;
        BOOL g_bGlobalThreadPoolGrowable = DEFAULT_GLOBAL_THREAD_POOL_GROWABLE;
        BOOL g_bWarnCriticalPathAlloc = DEFAULT_WARN_CRITICAL_PATH_ALLOC;
        BOOL g_bBlockPoolsEnabled = DEFAULT_BLOCK_POOLS_ENABLED;
        UINT g_uiDescriptorBlockPoolSize = DEFAULT_INPUT_CHANNEL_POOL_SIZE * 10;
        BOOL g_bAggressiveReleaseMode = DEFAULT_AGGRESSIVE_RELEASE_MODE;
        BOOL g_bHarmonizeInitialValueCoherenceState = DEFAULT_HARMONIZE_INIITIAL_VALUE_COHERENCE_STATE;
        UINT g_optimalPartitionerEdgeWeightScheme = 1;
        BOOL g_bDisableDtoHTransfers = FALSE;
        UINT g_uiAsyncContextGCQueryThreshold = DEFAULT_ASYNC_CONTEXT_GC_QUERY_THRESHOLD;
        BOOL GetGraphMutabilityMode()                                          { return g_bGraphsAreMutable;}
        void SetGraphMutabilityMode(BOOL bGraphsAreMutable)                    { warnIfInitialized(__FUNCTION__); g_bGraphsAreMutable = bGraphsAreMutable; }
        BOOL GetApplicationThreadsManagePrimaryContext()                       { return g_bAppThreadsManagePrimaryContext; }
        void SetApplicationThreadsManagePrimaryContext(BOOL bUseCtxt)          { warnIfInitialized(__FUNCTION__); if(!g_bPTaskInitialized) g_bAppThreadsManagePrimaryContext = bUseCtxt; }
        UINT GetAsyncContextGCQueryThreshold()                                 { return g_uiAsyncContextGCQueryThreshold; }
        void SetAsyncContextGCQueryThreshold(UINT uiThreshold)                 { g_uiAsyncContextGCQueryThreshold = uiThreshold; }
        GRAPHASSIGNMENTPOLICY GetGraphAssignmentPolicy()                       { return g_eGraphAssignmentPolicy; }
        void SetGraphAssignmentPolicy(GRAPHASSIGNMENTPOLICY policy)            { g_eGraphAssignmentPolicy = policy; }
        UINT GetGCSweepThresholdPercent()                                      { return g_uiGCSweepThresholdPercent; } 
        void SetGCSweepThresholdPercent(UINT uiPercent)                        { g_uiGCSweepThresholdPercent = uiPercent; }
        void SetAggressiveReleaseMode(BOOL bEnable)                            { g_bAggressiveReleaseMode = bEnable; }
        BOOL GetAggressiveReleaseMode()                                        { return g_bAggressiveReleaseMode; }
        void SetSizeDescriptorPoolSize(UINT uiBlocks)                          { warnIfInitialized(__FUNCTION__); g_uiDescriptorBlockPoolSize = uiBlocks; }
        UINT GetSizeDescriptorPoolSize()                                       { return g_uiDescriptorBlockPoolSize; }
        UINT GetGlobalThreadPoolSize()                                         { return g_uiGlobalThreadPoolSize; }
        BOOL GetPrimeGlobalThreadPool()                                        { return g_bPrimeGlobalThreadPool; }
        BOOL GetGlobalThreadPoolGrowable()                                     { return g_bGlobalThreadPoolGrowable; }
        void SetGlobalThreadPoolSize(UINT ui)                                  { warnIfInitialized(__FUNCTION__); g_uiGlobalThreadPoolSize = ui; }
        void SetPrimeGlobalThreadPool(BOOL b)                                  { warnIfInitialized(__FUNCTION__); g_bPrimeGlobalThreadPool = b; }
        void SetGlobalThreadPoolGrowable(BOOL b)                               { warnIfInitialized(__FUNCTION__); g_bGlobalThreadPoolGrowable = b; }
        BOOL HasGlobalThreadPool()                                             { return GetGlobalThreadPoolSize() > 0; }
        BOOL GetBlockPoolsEnabled()                                            { return g_bBlockPoolsEnabled; }
        void SetBlockPoolsEnabled(BOOL b)                                      { warnIfInitialized(__FUNCTION__); g_bBlockPoolsEnabled = b; }
        BOOL GetCriticalPathAllocMode()                                        { return g_bWarnCriticalPathAlloc; }
        BLOCKPOOLRESIZEPOLICY GetBlockPoolBlockResizePolicy()                  { return g_eBlockPoolBlockResizePolicy; }
        void SetBlockPoolBlockResizePolicy(BLOCKPOOLRESIZEPOLICY p)            { g_eBlockPoolBlockResizePolicy = p; }
        BLOCKRESIZEMEMSPACEPOLICY GetBlockResizeMemorySpacePolicy()            { return g_eBlockResizeMemspacePolicy; }
        void SetBlockResizeMemorySpacePolicy(BLOCKRESIZEMEMSPACEPOLICY policy) { g_eBlockResizeMemspacePolicy = policy; }
        BOOL GetScheduleChecksClassAvailability()                              { return g_bScheduleChecksClassAvailability; }
        void SetScheduleChecksClassAvailability(BOOL bCheck)                   { g_bScheduleChecksClassAvailability = bCheck; }
        DATABLOCKAFFINITYPOLICY GetDatablockAffinitiyPolicy()                  { return g_eDatablockAffinityPolicy; }
        void SetDatablockAffinitiyPolicy(DATABLOCKAFFINITYPOLICY policy)       { g_eDatablockAffinityPolicy = policy; }
        BOOL GetHarmonizeInitialValueCoherenceState()                          { return g_bHarmonizeInitialValueCoherenceState; }
        void SetHarmonizeInitialValueCoherenceState(BOOL b)                    { g_bHarmonizeInitialValueCoherenceState = b; }
        UINT GetOptimalPartitionerEdgeWeightScheme()                           { return g_optimalPartitionerEdgeWeightScheme; }
        void SetOptimalPartitionerEdgeWeightScheme(UINT s)                     { g_optimalPartitionerEdgeWeightScheme = s; }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets disable device to host xfer. This is a debug/instrumentation
        ///             setting that simply elides backend DtoH xfer calls. The only use
        ///             is estimating DtoH xfer impact on performance--it will clearly 
        ///             result in wrong answers for any workload that wants results back
        ///             from a GPU! Use with caution.</summary>
        ///
        /// <remarks>   Crossbac, 3/25/2014. </remarks>
        ///
        /// <param name="bDisable"> true to disable, false to enable. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetDisableDeviceToHostXfer(
            __in BOOL bDisable
            )
        {
            g_bDisableDtoHTransfers = bDisable; 
            if(g_bDisableDtoHTransfers) {
                PTask::Runtime::MandatoryInform("\n");
                PTask::Runtime::MandatoryInform("\n");
                PTask::Runtime::MandatoryInform("\n");
                PTask::Runtime::MandatoryInform("WARNING: %s(TRUE) called! is this intentional?\n", __FUNCTION__);
                PTask::Runtime::MandatoryInform("\n");
                PTask::Runtime::MandatoryInform("\n");
                PTask::Runtime::MandatoryInform("\n");
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets disable device to host xfer. This is a debug/instrumentation
        ///             setting that simply elides backend DtoH xfer calls. The only use
        ///             is estimating DtoH xfer impact on performance--it will clearly
        ///             result in wrong answers for any workload that wants results back
        ///             from a GPU! Use with caution.</summary>
        ///
        /// <remarks>   Crossbac, 3/25/2014. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        GetDisableDeviceToHostXfer(
            VOID
            )
        {
            return g_bDisableDtoHTransfers;
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   enable/disable warnings on critical path allocation. </summary>
        ///
        /// <remarks>   crossbac, 8/28/2013. </remarks>
        ///
        /// <param name="bEnable">  true to enable, false to disable. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetCriticalPathAllocMode(
            BOOL bEnable
            )             
        {
#ifdef CHECK_CRITICAL_PATH_ALLOC
            g_bWarnCriticalPathAlloc = bEnable; 
#else 
            if(bEnable) {
                PTask::Runtime::MandatoryInform("%s::%s(true) call has no effect!\n" 
                                                "Build does not support warnings on critical path allocation!",
                                                __FILE__,
                                                __FUNCTION__);
            }
            g_bWarnCriticalPathAlloc = false;
#endif
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Require block pool. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <param name="nDataSize">        Size of the data. </param>
        /// <param name="nMetaSize">        Size of the meta. </param>
        /// <param name="nTemplateSize">    Size of the template. </param>
        /// <param name="nBlocks">          The blocks. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        RequireBlockPool(
            __in int nDataSize, 
            __in int nMetaSize, 
            __in int nTemplateSize,
            __in int nBlocks
            )
        {
            warnIfInitialized(__FUNCTION__);
            if(!g_bPTaskInitialized) {
                GlobalPoolManager::RequireBlockPool(nDataSize, nMetaSize, nTemplateSize, nBlocks);
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Require block pool. </summary>
        ///
        /// <remarks>   crossbac, 8/20/2013. </remarks>
        ///
        /// <param name="nBlocks">          The blocks. </param>
        /// <param name="nDataSize">        Size of the data. </param>
        /// <param name="nMetaSize">        Size of the meta. </param>
        /// <param name="nTemplateSize">    Size of the template. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        RequireBlockPool(
            __in DatablockTemplate * pTemplate,
            __in int nBlocks
            )
        {
            warnIfInitialized(__FUNCTION__);
            if(!g_bPTaskInitialized) {
                GlobalPoolManager::RequireBlockPool(pTemplate, nBlocks);
            }
        }

        // logging facility. User can specify the iostream to use when generating trace output. The
        // primary motivation is that console output generates system calls and can affect scheduling
        // in a way that makes finding concurrency problems more difficult. The user can set the
        // iostream to a string stream if they want to. Default iostream is cout. 
        
#if (defined(DANDELION_DEBUG) || defined(DUMP_INTERMEDIATE_BLOCKS))
        BOOL g_bDebugLoggingSupported = TRUE;
#ifdef TRACE_STRINGSTREAM
        std::stringstream g_ss;
#else
        std::ostream& g_ss = std::cout;
#endif
        CRITICAL_SECTION g_sslock;
#define InitializeDebugLogging() { InitializeCriticalSection(&g_sslock); }
#define TerminateDebugLogging() { DeleteCriticalSection(&g_sslock); }
#else
        BOOL g_bDebugLoggingSupported = FALSE;
#define InitializeDebugLogging()
#define TerminateDebugLogging()
#endif


        // debug mode has two purposes:
        // 1) Always materialize host-side buffers for inner nodes
        //    so that intermediate results are visible to a debugger
        // 2) Emulate lower bound on overhead for a "modular" design
        //    that doesn't use ptasks and therefore must copy data back
        //    and forth between kernel invocations. Because of this 
        //    latter mode, we want to not only materialize data host-side,
        //    but force it to copy it back for it's consumer. To this it
        //    suffices to set mark the data block incoherent.
        BOOL g_bExtremeTrace = DEFAULT_EXTREME_TRACE;
        BOOL g_bCoherenceProfile = DEFAULT_COHERENCE_PROFILE;
        BOOL g_bTaskProfile = DEFAULT_TASK_PROFILE;
        BOOL g_bTaskProfileVerbose = DEFAULT_TASK_PROFILE_VERBOSE;
        BOOL g_bPageLockingEnabled = DEFAULT_PAGE_LOCKING_ENABLED;
        BOOL g_bAggressivePageLocking = DEFAULT_PAGE_LOCKING_AGGRESSIVE;
        BOOL g_bForceSynchronous = DEFAULT_FORCE_SYNCHRONOUS;
        BOOL g_bDebugMode = DEFAULT_DEBUG_MODE;
        BOOL g_bDispatchLoggingEnabled = DEFAULT_DISPATCH_LOGGING;
        BOOL g_bDispatchTracingEnabled = DEFAULT_DISPATCH_TRACING;
        BOOL g_bDebugAsynchrony = DEFAULT_DEBUG_ASYNCHRONY;
        BOOL g_bProfilePBuffers = DEFAULT_PROFILE_PBUFFERS;
        BOOL g_bEagerMetaPorts = DEFAULT_EAGER_META_PORTS;
        BOOL g_bTrackDeviceAllocation = DEFAULT_TRACK_DEVICE_ALLOCATION;
        BOOL g_bUseGraphWatchdog = DEFAULT_USE_GRAPH_WATCHDOG;
        BOOL g_bProvisionBlockPoolsForCapacity = DEFAULT_PROVISION_POOLS_FOR_CAPACITY;
        DWORD g_dwGraphWatchdogThreshold = DEFAULT_GRAPH_WATCHDOG_THRESHOLD;
        THREADPOOLPOLICY g_eThreadPoolPolicy = DEFAULT_THREAD_POOL_POLICY;
        UINT g_nTaskThreadPoolSize = DEFAULT_THREAD_POOL_SIZE;
        UINT g_nTaskPerThreadPolicyThreshold = DEFAULT_THREAD_POOL_PER_THREAD_MAX_THRESHOLD;
        UINT g_uiSchedulerThreadCount = DEFAULT_SCHEDULER_THREADS;
        BOOL g_bUserDefinedCUDAHeapSize = DEFAULT_SET_CUDA_HEAP_SIZE;
        UINT g_uiCUDAHeapSize = DEFAULT_CUDA_HEAP_SIZE;
        BOOL g_bInitCublas = FALSE;
        BOOL g_bSortThreadPoolQueues = DEFAULT_TASK_POOL_READY_SORT_BY_PRIO;
        GRAPHPARTITIONINGMODE g_eDefaultGraphPartitioningMode = (GRAPHPARTITIONINGMODE)DEFAULT_GRAPH_PARTIONING_MODE;
        BOOL g_bTaskDispatchLocksIncomingAsyncSources = DEFAULT_LOCK_INCOMING_ASYNC_SOURCES;
        BOOL g_bThreadPoolSignalPerThread = DEFAULT_TASK_POOL_SIGNAL_PER_WORKER;
        BOOL g_bTaskDispatchReadyCheckIncomingAsyncDeps = DEFAULT_READY_CHECK_INCLUDES_INCOMING_ASYNC_DEPS;
        BOOL g_bTaskDispatchLocklessIncomingDepWait = DEFAULT_LOCKLESS_INCOMING_ASYNC_WAIT;
        VIEWMATERIALIZATIONPOLICY g_eDefaultViewMaterializationPolicy = DEFAULT_VIEWMATERIALIZATIONPOLICY;
        VIEWMATERIALIZATIONPOLICY g_eDefaultOutputViewMaterializationPolicy = DEFAULT_OUTPUTVIEWMATERIALIZATIONPOLICY;

        BOOL g_bSignalProfilingEnabled       = DEFAULT_SIGNAL_PROFILING_ENABLED;
        BOOL g_bRCProfilingEnabled           = DEFAULT_REFCOUNT_PROFILING_ENABLED;
        BOOL g_bDBProfilingEnabled           = DEFAULT_DATABLOCK_PROFILING_ENABLED;
        BOOL g_bCTProfilingEnabled           = DEFAULT_COHERENCE_PROFILING_ENABLED;
        BOOL g_bTPProfilingEnabled           = DEFAULT_TASK_PROFILING_ENABLED;
        BOOL g_bPBufferProfilingEnabled      = DEFAULT_PBUFFER_PROFILING_ENABLED;
        BOOL g_bInvocationCountingEnabled    = DEFAULT_INVOCATION_COUNTING_ENABLED;
        BOOL g_bChannelProfilingEnabled      = DEFAULT_CHANNEL_PROFILING_ENABLED;
        BOOL g_bBlockPoolProfilingEnabled    = DEFAULT_BLOCKPOOL_PROFILING_ENABLED;
        BOOL g_bEnableDebugLogging           = DEFAULT_DEBUG_LOGGING_ENABLED;
        BOOL g_bAdhocInstrumentationEnabled  = DEFAULT_ADHOC_INSTRUMENTATION_ENABLED;
        BOOL g_bPBufferClearOnCreatePolicy   = DEFAULT_PBUFFER_CLEAR_ON_CREATE;
        BOOL g_bProfilePSDispatch            = DEFAULT_PROFILE_PS_DISPATCH;

        BOOL GetRCProfilingEnabled()                      { return g_bRCProfilingEnabled; }
        BOOL GetDBProfilingEnabled()                      { return g_bDBProfilingEnabled; }
        BOOL GetCTProfilingEnabled()                      { return g_bCTProfilingEnabled; }
        BOOL GetTPProfilingEnabled()                      { return g_bTPProfilingEnabled; }
        BOOL GetPBufferProfilingEnabled()                 { return g_bPBufferProfilingEnabled; }
        BOOL GetInvocationCountingEnabled()               { return g_bInvocationCountingEnabled; }
        BOOL GetChannelProfilingEnabled()                 { return g_bChannelProfilingEnabled; }
        BOOL GetBlockPoolProfilingEnabled()               { return g_bBlockPoolProfilingEnabled; }
        BOOL GetEnableDebugLogging()                      { return g_bEnableDebugLogging; }
        BOOL GetAdhocInstrumentationEnabled()             { return g_bAdhocInstrumentationEnabled; }
        BOOL GetProfilePSDispatch()                       { return g_bProfilePSDispatch; }
        BOOL GetSignalProfilingEnabled()                  { return g_bSignalProfilingEnabled; }

        void SetRCProfilingEnabled(BOOL bEnable)          { SET_PROFILER_MODE(g_bRCProfilingSupported, bEnable, g_bRCProfilingEnabled); }
        void SetDBProfilingEnabled(BOOL bEnable)          { SET_PROFILER_MODE(g_bDBProfilingSupported, bEnable, g_bDBProfilingEnabled); }
        void SetCTProfilingEnabled(BOOL bEnable)          { SET_PROFILER_MODE(g_bCTProfilingSupported, bEnable, g_bCTProfilingEnabled); }
        void SetTPProfilingEnabled(BOOL bEnable)          { SET_PROFILER_MODE(g_bTPProfilingSupported, bEnable, g_bTPProfilingEnabled); }
        void SetPBufferProfilingEnabled(BOOL bEnable)     { SET_PROFILER_MODE(g_bPBufferProfilingSupported, bEnable, g_bPBufferProfilingEnabled); }
        void SetInvocationCountingEnabled(BOOL bEnable)   { SET_PROFILER_MODE(g_bInvocationCountingSupported, bEnable, g_bInvocationCountingEnabled); }
        void SetChannelProfilingEnabled(BOOL bEnable)     { SET_PROFILER_MODE(g_bChannelProfilingSupported, bEnable, g_bChannelProfilingEnabled); }
        void SetBlockPoolProfilingEnabled(BOOL bEnable)   { SET_PROFILER_MODE(g_bBlockPoolProfilingSupported, bEnable, g_bBlockPoolProfilingEnabled); }
        void SetEnableDebugLogging(BOOL bEnable)          { SET_PROFILER_MODE(g_bDebugLoggingSupported, bEnable, g_bEnableDebugLogging); }
        void SetAdhocInstrumentationEnabled(BOOL bEnable) { SET_PROFILER_MODE(g_bAdhocInstrumentationSupported, bEnable, g_bAdhocInstrumentationEnabled); }
        void SetProfilePSDispatch(BOOL bEnable)           { SET_PROFILER_MODE(g_bAdhocInstrumentationSupported, bEnable, g_bProfilePSDispatch); }
        void SetSignalProfilingEnabled(BOOL bEnable)      { SET_PROFILER_MODE(g_bSignalProfilingSupported, bEnable, g_bSignalProfilingEnabled); }
        void RegisterSignalForProfiling(CONTROLSIGNAL luiControlSignal)     { warnIfInitialized(__FUNCTION__); if(!g_bPTaskInitialized && g_bSignalProfilingSupported) SignalProfiler::RegisterSignal(luiControlSignal); }
        void UnregisterSignalForProfiling(CONTROLSIGNAL luiControlSignal)   { warnIfInitialized(__FUNCTION__); if(!g_bPTaskInitialized && g_bSignalProfilingSupported) SignalProfiler::RegisterSignal(luiControlSignal, FALSE); }

        BOOL GetPBufferClearOnCreatePolicy()              { return g_bPBufferClearOnCreatePolicy; }
        BOOL SetPBufferClearOnCreatePolicy(BOOL bClearOnCreate) { g_bPBufferClearOnCreatePolicy = bClearOnCreate; return bClearOnCreate; }

        BOOL GetDebugMode() { return g_bDebugMode; }
        void SetDebugMode(BOOL mode) { g_bDebugMode = mode; }
        BOOL GetDispatchLoggingEnabled() { return g_bDispatchLoggingEnabled; }
        void SetDispatchLoggingEnabled(BOOL bEnabled) { g_bDispatchLoggingEnabled = bEnabled; } 
        BOOL GetDispatchTracingEnabled() { return g_bDispatchTracingEnabled; }
        void SetDispatchTracingEnabled(BOOL bEnabled) { g_bDispatchTracingEnabled = bEnabled; } 
        BOOL GetEagerMetaPortMode() { return g_bEagerMetaPorts; }
        void SetEagerMetaPortMode(BOOL b) { g_bEagerMetaPorts = b; }
        BOOL GetTaskDispatchLocksIncomingAsyncSources() { return g_bTaskDispatchLocksIncomingAsyncSources; }
        BOOL GetThreadPoolSignalPerThread() { return g_bThreadPoolSignalPerThread; }
        BOOL GetTaskDispatchReadyCheckIncomingAsyncDeps() { return g_bTaskDispatchReadyCheckIncomingAsyncDeps; }
        BOOL GetTaskDispatchLocklessIncomingDepWait() { return g_bTaskDispatchLocklessIncomingDepWait; }
        void SetTaskDispatchLocksIncomingAsyncSources(BOOL b) { g_bTaskDispatchLocksIncomingAsyncSources = b; }
        void SetThreadPoolSignalPerThread(BOOL b) { g_bThreadPoolSignalPerThread = b; }
        void SetTaskDispatchReadyCheckIncomingAsyncDeps(BOOL b) { g_bTaskDispatchReadyCheckIncomingAsyncDeps = b; }
        void SetTaskDispatchLocklessIncomingDepWait(BOOL b) { g_bTaskDispatchLocklessIncomingDepWait = b; }

	    INSTRUMENTATIONMETRIC g_InstrumentationMetric = NONE;
        BOOL g_bInstrumented = false;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Controls whether PTask profiles management of platform-specific buffer objects
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------
    
        void 
        SetProfilePlatformBuffers(
            BOOL b
            )
        {
#ifndef PROFILE_PBUFFERS
            g_bProfilePBuffers = FALSE;
            if(b) {
                Warning("\n\n"
                        "\t***************************************************\n"
                        "\t  ineffectual attempt to enable buffer profiling! *\n"
                        "\t*   PTASK BUFFER PROFILING MODE NOT BE ENABLED!   *\n"
                        "\t*    #define PROFILE_PBUFFERS required in build   *\n"
                        "\t***************************************************\n\n");
            }
#else
            g_bProfilePBuffers = b;
#endif
#ifndef DEBUG
            if(Runtime::g_bProfilePBuffers) {
                Warning("\n\n"
                        "\t***********************************************\n"
                        "\t*   PTASK BUFFER PROFILING MODE ENABLED!      *\n"
                        "\t*   THIS MODE IS NOT FOR USE IN PRODUCTION!   *\n"
                        "\t*        IS THIS SETTING INTENTIONAL?         *\n"
                        "\t***********************************************\n\n");
            }
#endif
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Controls whether PTask profiles management of platform-specific buffer objects
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------
    
        BOOL 
        GetProfilePlatformBuffers(
            VOID
            )
        {
            return g_bProfilePBuffers;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets default view materialization policy for channels. </summary>
        ///
        /// <remarks>   Crossbac, 7/9/2013. </remarks>
        ///
        /// <returns>   The default view materialization policy. </returns>
        ///-------------------------------------------------------------------------------------------------

        VIEWMATERIALIZATIONPOLICY 
        GetDefaultViewMaterializationPolicy(
            VOID
            )
        {
            return g_eDefaultViewMaterializationPolicy;
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the default view materialization policy for exposed output channels. </summary>
        ///
        /// <remarks>   Crossbac, 7/9/2013. </remarks>
        ///
        /// <returns>   The default output view materialization policy. </returns>
        ///-------------------------------------------------------------------------------------------------

        VIEWMATERIALIZATIONPOLICY 
        GetDefaultOutputViewMaterializationPolicy(
            VOID
            )
        {
            return g_eDefaultOutputViewMaterializationPolicy;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets default view materialization policy for channels. </summary>
        ///
        /// <remarks>   Crossbac, 7/9/2013. </remarks>
        ///
        /// <param name="ePolicy">  The policy. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetDefaultViewMaterializationPolicy(
            __in VIEWMATERIALIZATIONPOLICY ePolicy 
            )
        {
            warnIfInitialized(__FUNCTION__);
            g_eDefaultViewMaterializationPolicy = ePolicy;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets default view materialization policy for channels. </summary>
        ///
        /// <remarks>   Crossbac, 7/9/2013. </remarks>
        ///
        /// <param name="ePolicy">  The policy. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetDefaultOutputViewMaterializationPolicy(
            __in VIEWMATERIALIZATIONPOLICY ePolicy 
            )
        {
            warnIfInitialized(__FUNCTION__);
            g_eDefaultOutputViewMaterializationPolicy = ePolicy;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Get the PTask runtime mode for introducing synchrony after sensitive
        ///             operations like dispatch and dependence waiting. When the runtime is in
        ///             "ForceSynchonous" mode, backend API calls are inserted to synchronize device
        ///             contexts with the host. This is intended as a debug tool to rule out race-
        ///             conditions in PTask when faced with undesirable or difficult-to-understand
        ///             results from programs written for PTask.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/12. </remarks>
        ///
        /// <param name="b">    true if the runtime should be in force-sync mode, false otherwise. </param>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        GetForceSynchronous(
            VOID
            )
        {
            return Runtime::g_bForceSynchronous;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Control the PTask runtime mode for introducing synchrony after sensitive
        ///             operations like dispatch and dependence waiting. When the runtime is in
        ///             "ForceSynchonous" mode, backend API calls are inserted to synchronize device
        ///             contexts with the host. This is intended as a debug tool to rule out race-
        ///             conditions in PTask when faced with undesirable or difficult-to-understand
        ///             results from programs written for PTask.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/12. </remarks>
        ///
        /// <param name="b">    true if the runtime should be in force-sync mode, false otherwise. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetForceSynchronous(
            BOOL b
            )
        {
            Runtime::g_bForceSynchronous = b;
#ifndef DEBUG
            if(Runtime::g_bForceSynchronous) {
                Warning("\n\n"
                        "\t***********************************************\n"
                        "\t*   PTASK FORCE SYNCHRONOUS MODE ENABLED!     *\n"
                        "\t*   THIS MODE IS NOT FOR USE IN PRODUCTION!   *\n"
                        "\t*        IS THIS SETTING INTENTIONAL?         *\n"
                        "\t***********************************************\n\n");
            }
#endif
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the extreme trace mode. Extreme trace mode logs every API call to the trace
        ///             provider. For the mode to work PTASK must be built with EXTREME_TRACE
        ///             preprocessor macro! Otherwise, the trace mode is always off.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        GetExtremeTraceMode(
            VOID
            )
        {
#ifndef EXTREME_TRACE
            return FALSE;
#else
            return Runtime::g_bExtremeTrace;
#endif
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the extreme trace mode. Extreme trace mode logs every API call to the trace
        ///             provider. For the mode to work PTASK must be built with EXTREME_TRACE
        ///             preprocessor macro! Otherwise, the trace mode is always off.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------
    
        void 
        SetExtremeTraceMode(
            BOOL b
            )
        {
#ifndef EXTREME_TRACE
            if(b) {
                Warning("\n\n"
                        "\t***********************************************\n"
                        "\t*   CANNOT ENABLE PTASK EXTREME TRACING!      *\n"
                        "\t*   (PTASK MUST BE BUILT WITH EXTREME_TRACE)  *\n"
                        "\t***********************************************\n\n");
            }
            Runtime::g_bExtremeTrace = FALSE;
#else
            Runtime::g_bExtremeTrace = b;
#ifndef DEBUG
            if(Runtime::g_bExtremeTrace) {
                Warning("\n\n"
                        "\t***********************************************\n"
                        "\t*   PTASK EXTREME TRACING MODE ENABLED!       *\n"
                        "\t*   THIS MODE IS NOT FOR USE IN PRODUCTION!   *\n"
                        "\t*        IS THIS SETTING INTENTIONAL?         *\n"
                        "\t***********************************************\n\n");
            }
#endif
#endif
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the coherence profile mode. The mode logs every coherence transition
        /// 			and task/port binding for every datablock. For the mode to work PTASK must be built 
        /// 			with PROFILE_MIGRATION. Otherwise the mode is always off.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        GetCoherenceProfileMode(
            VOID
            )
        {
#ifndef PROFILE_MIGRATION
            return FALSE;
#else
            return Runtime::g_bCoherenceProfile;
#endif
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the coherence profile mode. The mode logs every coherence transition
        /// 			and task/port binding for every datablock. For the mode to work PTASK must be built 
        /// 			with PROFILE_MIGRATION. Otherwise the mode is always off.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------
    
        void 
        SetCoherenceProfileMode(
            BOOL b
            )
        {
#ifndef PROFILE_MIGRATION
            if(b) {
                Warning("\n\n"
                        "\t***********************************************\n"
                        "\t* CANNOT ENABLE PTASK COHERENCE PROFILING!    *\n"
                        "\t* (PTASK MUST BE BUILT WITH PROFILE_MIGRATION)*\n"
                        "\t***********************************************\n\n");
            }
            Runtime::g_bCoherenceProfile = FALSE;
#else
            Runtime::g_bCoherenceProfile = b;
#ifndef DEBUG
            if(Runtime::g_bCoherenceProfile) {
                Warning("\n\n"
                        "\t***********************************************\n"
                        "\t*   PTASK COHERENCE PROFILE MODE ENABLED!     *\n"
                        "\t*   THIS MODE IS NOT FOR USE IN PRODUCTION!   *\n"
                        "\t*        IS THIS SETTING INTENTIONAL?         *\n"
                        "\t***********************************************\n\n");
            }
#endif
#endif
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the task profile mode. The mode captures per-dispatch timings for various
        ///             task-critical activities such as buffer allocation, binding, transfer, dispatch,
        ///             etc. For the mode to work, PTask must be built with PROFILE_TASKS. This API is
        ///             available regardless but does not affect the behavior of the runtime if the build
        ///             doesn't support the profiling option.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///
        /// <param name="bEnable">  true to enable, false to disable. </param>
        /// <param name="bConcise"> true for concise statistics. </param>
        ///-------------------------------------------------------------------------------------------------
    
        void 
        SetTaskProfileMode(
            BOOL bEnable,
            BOOL bConcise
            )
        {
#ifndef PROFILE_TASKS
            UNREFERENCED_PARAMETER(bConcise);
            if(bEnable) {
                Warning("\n\n"
                        "\t***********************************************\n"
                        "\t* CANNOT ENABLE PTASK PER-TASK PROFILING!     *\n"
                        "\t* (PTASK MUST BE BUILT WITH PROFILE_TASKS)    *\n"
                        "\t***********************************************\n\n");
            }
            Runtime::g_bTaskProfile = FALSE;
#else
            Runtime::g_bTaskProfile = bEnable;
            Runtime::g_bTaskProfileVerbose = !bConcise;
#ifndef DEBUG
            if(Runtime::g_bTaskProfile) {
                Warning("\n\n"
                        "\t***********************************************\n"
                        "\t*   PTASK PER-TASK PROFILE MODE ENABLED!      *\n"
                        "\t*   THIS MODE IS NOT FOR USE IN PRODUCTION!   *\n"
                        "\t*        IS THIS SETTING INTENTIONAL?         *\n"
                        "\t***********************************************\n\n");
            }
#endif
#endif
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the task profile mode. The mode captures per-dispatch timings for various
        ///             task-critical activities such as buffer allocation, binding, transfer, dispatch,
        ///             etc. For the mode to work, PTask must be built with PROFILE_TASKS. This API is
        ///             available regardless but always returns false if the build doesn't support the
        ///             profiling option.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        GetTaskProfileMode(
            VOID
            )
        {
#ifndef PROFILE_TASKS
            return FALSE;
#else
            return Runtime::g_bTaskProfile;
#endif        
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Controls whether PTask attempts to automatically manage page-locked buffer
        /// 			management for devices that support page-locked buffer allocation. If it
        /// 			is set to true, PTask will allocate page-locked buffers whenever it detects 
        /// 			that it may be profitable to do so. When it is off, PTask will never allocated
        /// 			a page-locked host-buffer directly. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------
    
        void 
        SetPageLockingEnabled(
            BOOL b
            )
        {
            g_bPageLockingEnabled = b;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Controls whether PTask attempts to automatically manage page-locked buffer
        /// 			management for devices that support page-locked buffer allocation. If it
        /// 			is set to true, PTask will allocate page-locked buffers whenever it detects 
        /// 			that it may be profitable to do so. When it is off, PTask will never allocated
        /// 			a page-locked host-buffer directly. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------
    
        BOOL 
        GetPageLockingEnabled(
            VOID
            )
        {
            return g_bPageLockingEnabled;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Controls whether PTask attempts to always use page-locked host buffers
        /// 			for devices that support page-locked buffer allocation. If it
        /// 			is set to true, PTask will allocate page-locked buffers whenever it detects 
        /// 			that it may be possible to use the resulting buffer in an async API call. 
        ///             When it is off, PTask will only allocate a page-locked host-buffer when 
        ///             the programmer requests it explicitly for a given block. Generally speaking,
        ///             setting this to true is profitable for workloads with moderate memory traffic, 
        ///             but since performance drops off quickly when too much page-locked memory is allocated,
        ///             this is not a setting that should be used without careful consideration. 
        ///             </summary>
        ///             
        /// <remarks>   crossbac, 4/28/2013. </remarks>
        ///
        /// <param name="b">    true to b. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetAggressivePageLocking(
            BOOL bAggressive
            )
        {
            g_bAggressivePageLocking = bAggressive;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Controls whether PTask attempts to always use page-locked host buffers
        /// 			for devices that support page-locked buffer allocation. If it
        /// 			is set to true, PTask will allocate page-locked buffers whenever it detects 
        /// 			that it may be possible to use the resulting buffer in an async API call. 
        ///             When it is off, PTask will only allocate a page-locked host-buffer when 
        ///             the programmer requests it explicitly for a given block. Generally speaking,
        ///             setting this to true is profitable for workloads with moderate memory traffic, 
        ///             but since performance drops off quickly when too much page-locked memory is allocated,
        ///             this is not a setting that should be used without careful consideration. 
        ///             </summary>
        ///             
        /// <remarks>   crossbac, 4/28/2013. </remarks>
        ///
        /// <param name="b">    true to b. </param>
        ///-------------------------------------------------------------------------------------------------
    
        BOOL 
        GetAggressivePageLocking(
            VOID
            )
        {
            return g_bAggressivePageLocking;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Controls whether PTask emits debug text when synchronous transfers occur
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------
    
        void 
        SetDebugAsynchronyMode(
            BOOL b
            )
        {
            g_bDebugAsynchrony = b;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Controls whether PTask emits debug text when synchronous transfers occur
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------
    
        BOOL 
        GetDebugAsynchronyMode(
            VOID
            )
        {
            return g_bDebugAsynchrony;
        }

        // tools to artificially limit the number of accelerators
        // PTask will use. In practice, setting a limit here would be silly.
        // However, it is useful for benchmarking to be able to force PTask
        // to use fewer than the total available number of GPUs. 
        /// <summary> The physical accelerators </summary>
        UINT g_nPhysicalAccelerators = 0;
        /// <summary> The maximum number of accelerators to use </summary>
        int g_nMaximumConcurrency = DEFAULT_MAX_CONCURRENCY; // 0 means no limit
		int g_nMaximumHostConcurrency = DEFAULT_HOST_CONCURRENCY; // 4

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Given the name of a file containing accelerator
        /// 			code, return the class of an accelerator
        /// 			capable of running it. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="szAcceleratorCodeFileName">    Filename of the accelerator code file. </param>
        ///
        /// <returns>   On success, the ACCELERATOR_CLASS of a matching accelerator.
        /// 			On failure, ACCELERATOR_CLASS unknown. </returns>
        ///-------------------------------------------------------------------------------------------------

        ACCELERATOR_CLASS
        GetAcceleratorClass(
            const char * szAcceleratorCodeFileName
            ) 
        {
            return ptaskutils::SelectAcceleratorClass(szAcceleratorCodeFileName);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queries if we can execute the kernel 'szAcceleratorCodeFileName' (or think we
        ///             can). This boils down to checking for platform support for the accelerator class
        ///             that would be chosen to compile and run the file. The runtime may or may not be
        ///             started with that support enabled by the programmer, and the runtime environment
        ///             may or may not be able to find an accelerator device for it even if the support
        ///             is enabled in PTask. For example, clients using RDP typically can only use host
        ///             tasks. Machines with nVidia Tesla cards cannot run DirectX programs without the
        ///             reference driver, etc. This should be called after the runtime is initialized,
        ///             and will check whether an appropriate device could be found by the scheduler.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///
        /// <param name="szAcceleratorCodeFileName">    Filename of the accelerator code file. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        CanExecuteKernel(
            const char * szAcceleratorCodeFileName
            )
        {
            if(!Runtime::g_bPTaskInitialized) {
                Warning("Runtime::CanExecuteKernel called before runtime initialization!\n"
                        "Returning FALSE, but the result is technically inconclusive!\n");
                return FALSE;
            }
            ACCELERATOR_CLASS accClass = GetAcceleratorClass(szAcceleratorCodeFileName);
            std::set<Accelerator*> vaccs;
            Scheduler::FindEnabledCapableAccelerators(accClass, vaccs);
            if(vaccs.size() == 0) {
                return FALSE;
            }
#ifndef DIRECTXCOMPILESUPPORT
            if(accClass == ACCELERATOR_CLASS_DIRECT_X) {
                MandatoryInform("PTask::Runtime::CanExecuteKernel called on HLSL code (%s),"
                                "but ptask build has DX support compiled out!\n", 
                                szAcceleratorCodeFileName);
                return FALSE;
            }
#endif
            return TRUE;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enumerate accelerators of a given class. 
        /// 			Call this function from user code to find out what accelerators
        /// 			are available to run a particular piece of accelerator code.
        /// 			To enumerate all available accelerators, pass ACCELERATOR_CLASS_UNKNOWN.
        /// 			Caller is responsible for:
        /// 			1. Incrementing the enumeration index.  
        /// 			2. Freeing the returned descriptor using free().  
        /// 			The function returns PTASK_OK until no more accelerators
        /// 			are found.
        /// 		    </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="eAcceleratorClass">        The accelerator class. Pass ACCELERATOR_CLASS_UNKNOWN
        /// 										to enumerate accelerators of all types.</param>
        /// <param name="uiAcceleratorIndex">       Index of the accelerator being enumerated. </param>
        /// <param name="ppAcceleratorDescriptor">  [out] If non-null, an ACCELERATOR_DESCRIPTOR
        /// 										describing the accelerator at that index. The 
        /// 										caller must free the descriptor. If null, the function
        /// 										return code indicates whether an accelerator exists
        /// 										at that index, but does not provide a descriptor.
        /// 										</param>
        ///
        /// <returns>   PTASK_OK if an accelerator of the given class exists at the
        /// 			specified enumeration index, otherwise PTASK_ERR_NOT_FOUND. </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        EnumerateAccelerators(
            ACCELERATOR_CLASS eAcceleratorClass,
            UINT uiEnumerationIndex,
            ACCELERATOR_DESCRIPTOR ** ppAcceleratorDescriptor
            )
        {
            std::vector<Accelerator*> acceleratorList;
            Scheduler::EnumerateAllAccelerators(eAcceleratorClass, acceleratorList);
            if(acceleratorList.size() < uiEnumerationIndex + 1) 
                return PTASK_ERR_NOT_FOUND;
            if(ppAcceleratorDescriptor == NULL) 
                return PTASK_OK;  // accelerator exists, but do not populate descriptor.
            Accelerator * pAccelerator = acceleratorList.at(uiEnumerationIndex);
            *ppAcceleratorDescriptor = (ACCELERATOR_DESCRIPTOR*) malloc(sizeof(ACCELERATOR_DESCRIPTOR));
            (*ppAcceleratorDescriptor)->pAccelerator = pAccelerator;
            (*ppAcceleratorDescriptor)->accClass = eAcceleratorClass;
            (*ppAcceleratorDescriptor)->bSupportsConcurrentKernels = pAccelerator->SupportsConcurrentKernels();
            (*ppAcceleratorDescriptor)->nClockRate = pAccelerator->GetCoreClockRate();
            (*ppAcceleratorDescriptor)->nCoreCount = pAccelerator->GetCoreCount();
            (*ppAcceleratorDescriptor)->nMemorySize = pAccelerator->GetGlobalMemorySize();
            (*ppAcceleratorDescriptor)->nPlatformIndex = pAccelerator->GetPlatformIndex();
            (*ppAcceleratorDescriptor)->uiAcceleratorId = pAccelerator->GetAcceleratorId();
            (*ppAcceleratorDescriptor)->nRuntimeVersion = pAccelerator->GetPlatformSpecificRuntimeVersion();
            (*ppAcceleratorDescriptor)->bEnabled = Scheduler::IsEnabled(pAccelerator);
            strcpy_s((*ppAcceleratorDescriptor)->szDescription, 256, pAccelerator->GetDeviceName());
            return PTASK_OK;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the platform specific runtime version for a given accelerator class.
        ///             Currently assumes that all devices of a particular class have the same version,
        ///             which is sufficient for all the needs we currently have (this API is used to
        ///             select compiler settings for autogenerated code), but when there are GPUs with
        ///             e.g. different compute capabilities, this API will assert and return failure. In
        ///             the future, we may need to enumerate all available versions, but at the moment
        ///             that is a lot of needless complexity to support a use case we don't have.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 8/29/2013. </remarks>
        ///
        /// <param name="eAcceleratorClass">    The accelerator class. </param>
        /// <param name="uiPSRuntimeVersion">   [in,out] The ps runtime version. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        GetPlatformSpecificRuntimeVersion(
            ACCELERATOR_CLASS eAcceleratorClass,
            UINT& uiPSRuntimeVersion
            ) 
        {
            uiPSRuntimeVersion = 0xFFFFFFFF;
            std::vector<Accelerator*>::iterator vi;
            std::vector<Accelerator*> acceleratorList;
            Scheduler::EnumerateAllAccelerators(eAcceleratorClass, acceleratorList);
            if(acceleratorList.size() == 0) 
                return FALSE;

            BOOL bVersionValid = FALSE;
            UINT uiCandidateVersion = 0xFFFFFFFF;
            for(vi=acceleratorList.begin(); vi!=acceleratorList.end(); vi++) {
                // require all instances of a class to match to use this API.
                UINT uiAcceleratorRTVersion = (*vi)->GetPlatformSpecificRuntimeVersion();
                if(bVersionValid && uiCandidateVersion != uiAcceleratorRTVersion) {
                    assert(FALSE);
                    return FALSE;
                }
                uiCandidateVersion = uiAcceleratorRTVersion;
                bVersionValid = true;
            }
            
            if(!bVersionValid)
                return FALSE;
            uiPSRuntimeVersion = uiCandidateVersion;
            return TRUE;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Disables an accelerator before PTask initialization: this means
        ///             PTask will deny list it, and if it encounters the accelerator
        ///             at initialization time, it immediately disables it. </summary>
        ///
        /// <remarks>   crossbac, 6/19/2014. </remarks>
        ///
        /// <param name="eAcceleratorClass">    The accelerator class. </param>
        /// <param name="nPSDeviceID">          Identifier for the ps device. </param>
        ///
        /// <returns>   PTASK_OK for successful addition to the deny list. 
        ///             PTASK_ERR_ALREADY_INITIALIZED if the runtime is already initialized.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        DisableAccelerator(
            ACCELERATOR_CLASS eAcceleratorClass,
            int  nPSDeviceID
            )
        {
            if(g_bPTaskInitialized)
                return PTASK_ERR_ALREADY_INITIALIZED;
            return Scheduler::EnableAccelerator(eAcceleratorClass, nPSDeviceID, FALSE);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enable/disables an accelerator before PTask initialization: on disable, this means
        ///             PTask will deny list it, and if it encounters the accelerator
        ///             at initialization time, it immediately calls the dynamic API to disable it. 
        ///             Enable is a NO-OP unless there is already a deny list entry for it. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/19/2014. </remarks>
        ///
        /// <param name="eAcceleratorClass">    The accelerator class. </param>
        /// <param name="nPSDeviceID">          Identifier for the ps device. </param>
        /// <param name="bEnable">              (Optional) the enable. </param>
        ///
        /// <returns>   PTASK_OK for successful addition/removal to/from the deny list. 
        ///             PTASK_ERR_ALREADY_INITIALIZED if the runtime is already initialized.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        EnableAccelerator(
            ACCELERATOR_CLASS eAcceleratorClass,
            int  nPSDeviceID,
            BOOL bEnable
            )
        {
            if(g_bPTaskInitialized)
                return PTASK_ERR_ALREADY_INITIALIZED;
            return Scheduler::EnableAccelerator(eAcceleratorClass, nPSDeviceID, bEnable);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Disables the accelerator indicated by the descriptor. "Disabled" means the
        ///             scheduler will not dispatch work on that accelerator.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pAcceleratorDescriptor">   [in,out] If non-null, information describing the
        ///                                         accelerator. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        DynamicDisableAccelerator(
            __in ACCELERATOR_DESCRIPTOR * pAcceleratorDescriptor
            )
        {
            if(pAcceleratorDescriptor == NULL)
                return PTASK_ERR_INVALID_PARAMETER;
            Accelerator * pAccelerator = reinterpret_cast<Accelerator*>(pAcceleratorDescriptor->pAccelerator);
            if(pAccelerator == NULL)
                return PTASK_ERR_INVALID_PARAMETER;
            return Scheduler::SetAcceleratorEnabled(pAccelerator, FALSE);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enables the accelerator indicated by the descriptor. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pAcceleratorDescriptor">   [in,out] If non-null, information describing the
        ///                                         accelerator. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        DynamicEnableAccelerator(
            __in ACCELERATOR_DESCRIPTOR * pAcceleratorDescriptor
            )
        {
            if(pAcceleratorDescriptor == NULL)
                return PTASK_ERR_INVALID_PARAMETER;
            Accelerator * pAccelerator = reinterpret_cast<Accelerator*>(pAcceleratorDescriptor->pAccelerator);
            if(pAccelerator == NULL)
                return PTASK_ERR_INVALID_PARAMETER;
            return Scheduler::SetAcceleratorEnabled(pAccelerator, TRUE);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Disables the accelerator. "Disabled" means the scheduler will not
        ///             dispatch work on that accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        DynamicDisableAccelerator(
            __in void * pvAccelerator
            )
        {
            Accelerator * pAccelerator = reinterpret_cast<Accelerator*>(pvAccelerator);
            if(pAccelerator == NULL)
                return PTASK_ERR_INVALID_PARAMETER;
            return Scheduler::SetAcceleratorEnabled(pAccelerator, FALSE);    
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enables the accelerator. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        DynamicEnableAccelerator(
            __in void * pvAccelerator
            )
        {
            Accelerator * pAccelerator = reinterpret_cast<Accelerator*>(pvAccelerator);
            if(pAccelerator == NULL)
                return PTASK_ERR_INVALID_PARAMETER;
            return Scheduler::SetAcceleratorEnabled(pAccelerator, TRUE);
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        ///     Sets task-accelerator affinity. Given an accelerator id, set the affinity between the ptask
        ///     and that accelerator to the given affinity type.
        /// </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pTask">            [in] non-null, the task. </param>
        /// <param name="uiAcceleratorId">  [in] accelerator identifier. </param>
        /// <param name="eAffinityType">   [in] affinity type. </param>
        ///
        /// <returns>
        ///     PTRESULT--use PTSUCCESS macro to check success. return PTASK_OK on success. returns
        ///     PTASK_ERR_INVALID_PARAMETER if the affinity combination requested cannot be provided by
        ///     the runtime.
        /// </returns>
        ///
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        SetTaskAffinity(
            Task * pTask,
            UINT uiAcceleratorId,
            AFFINITYTYPE eAffinityType
            )
        {
            if(pTask == NULL) {
                assert(pTask != NULL);
                return PTASK_ERR_INVALID_PARAMETER;
            }
            Accelerator * pAccelerator = Scheduler::GetAcceleratorById(uiAcceleratorId);
            if(pAccelerator == NULL) {
                PTask::Runtime::Warning("Attempt to set affinity for non-existent accelerator!");
                return PTASK_ERR_NOT_FOUND;
            }
            if(!pTask->SetAffinity(pAccelerator, eAffinityType)) {
                return PTASK_ERR_INVALID_PARAMETER;
            } 
            return PTASK_OK;
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets task-accelerator affinity. Given a list of accelerator ids, set the
        /// 			affinity between the ptask and each accelerator in the list
        /// 			to the affinity type at the same index in the affinity type
        /// 			list. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pTask">            [in] non-null, the task. </param>
        /// <param name="pvAcceleratorIds"> [in] non-null, list of accelerator identifiers</param>
        /// <param name="pvAffinityTypes">  [in] non-null, list of affinity types. </param>
        /// <param name="nAcceleratorIds">  List of identifiers for the accelerators. </param>
        ///
        /// <returns>   PTRESULT--use PTSUCCESS macro to check success. 
        /// 			return PTASK_OK on success.
        /// 			returns PTASK_ERR_INVALID_PARAMETER if the affinity
        /// 			combination requested cannot be provided by the runtime.</returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        SetTaskAffinity(
            Task * pTask,
            UINT * pvAcceleratorIds,
            AFFINITYTYPE * pvAffinityTypes,
            UINT nAcceleratorIds
            )
        {
            if(pTask == NULL || 
                pvAcceleratorIds == NULL || 
                pvAffinityTypes == NULL ||
                nAcceleratorIds == 0) {
                return PTASK_ERR_INVALID_PARAMETER;
            }

            for(UINT i=0; i<nAcceleratorIds; i++) {
                PTRESULT pr = SetTaskAffinity(pTask, pvAcceleratorIds[i], pvAffinityTypes[i]);
                if(PTFAILED(pr))
                    return pr;
            }
            return PTASK_OK;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the maximum concurrency. 
        /// 			Setting to 0 means do not artificially limit the runtime.
        /// 			Setting to > 0 means use no greater than X physical devices.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="nGPUs">    The gp us. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetMaximumConcurrency(
            int nGPUs
            ) 
        { 
            warnIfInitialized("SetMaximumConcurrency"); 
            g_nMaximumConcurrency = nGPUs; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the maximum concurrency setting. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   The maximum concurrency. </returns>
        ///-------------------------------------------------------------------------------------------------

        int GetMaximumConcurrency() { 
            return g_nMaximumConcurrency; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the maximum number of host accelerator objects.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="nGPUs">    The gp us. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetMaximumHostConcurrency(
            int nHostAccelerators
            ) 
        { 
            warnIfInitialized("SetMaximumHostConcurrency"); 
            g_nMaximumHostConcurrency = nHostAccelerators; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the maximum concurrency setting. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   The maximum concurrency. </returns>
        ///-------------------------------------------------------------------------------------------------

        int GetMaximumHostConcurrency() { 
            return g_nMaximumHostConcurrency; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return the size of the task thread pool. A return value of zero means the
        ///             scheduler is using a thread per-task, which is the traditional PTask
        ///             approach. However, this approach is non-performant for very graphs because
        ///             windows doesn't (can't?) handle large thread counts well. Consequently,
        ///             we provide a switch for the programmer to manage the thread pool size explicitly
        ///             as well as switches to cause the runtime to choose the size automatically
        ///             based on a tunable parameter. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetTaskThreadPoolSize(
            VOID
            ) 
        {
            return g_nTaskThreadPoolSize;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Set the size of the task thread pool. A return value of zero means the scheduler
        ///             is using a thread per-task, which is the traditional PTask approach. However,
        ///             this approach is non-performant for very graphs because windows doesn't (can't?)
        ///             handle large thread counts well. Consequently, we provide a switch for the
        ///             programmer to manage the thread pool size explicitly as well as switches to cause
        ///             the runtime to choose the size automatically based on a tunable parameter.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="uiThreadPoolSize"> Size of the thread pool. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetTaskThreadPoolSize(
            UINT uiThreadPoolSize
            )
        {
            warnIfInitialized("SetTaskThreadPoolSize");
            if(uiThreadPoolSize == 0) {
                if(g_eThreadPoolPolicy != TPP_THREADPERTASK) {
                    PTask::Runtime::Inform("\nPTask detected a call to set the task thread pool size to 0,\n"
                                           "but the thread pool management policy was set to TPP_EXPLICIT or TPP_AUTOMATIC\n"
                                           "...changing policy to TPP_THREADPERTASK.\n");
                    g_eThreadPoolPolicy = TPP_THREADPERTASK;
                }
            }
            g_nTaskThreadPoolSize = uiThreadPoolSize;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return the size of the task thread pool. A return value of zero means the
        ///             scheduler is using a thread per-task, which is the traditional PTask
        ///             approach. However, this approach is non-performant for very graphs because
        ///             windows doesn't (can't?) handle large thread counts well. Consequently,
        ///             we provide a switch for the programmer to manage the thread pool size explicitly
        ///             as well as switches to cause the runtime to choose the size automatically
        ///             based on a tunable parameter. 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        THREADPOOLPOLICY 
        GetTaskThreadPoolPolicy(
            VOID
            )
        {
            return g_eThreadPoolPolicy;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Set the size of the task thread pool. A return value of zero means the scheduler
        ///             is using a thread per-task, which is the traditional PTask approach. However,
        ///             this approach is non-performant for very graphs because windows doesn't (can't?)
        ///             handle large thread counts well. Consequently, we provide a switch for the
        ///             programmer to manage the thread pool size explicitly as well as switches to cause
        ///             the runtime to choose the size automatically based on a tunable parameter.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="uiThreadPoolSize"> Size of the thread pool. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetTaskThreadPoolPolicy(
            THREADPOOLPOLICY ePolicy
            )
        {
            warnIfInitialized("SetTaskThreadPoolPolicy");
            if(ePolicy == TPP_EXPLICIT) {
                if(g_nTaskThreadPoolSize == 0) {
                    PTask::Runtime::Inform("\nPTask detected a call to set the task thread pool policy to explicit,\n"
                                           "but the thread pool size is 0. Changing size to 1 to ensure that the runtime\n"
                                           "will actually function if no corresponding call to set the pool size is made!\n");
                    g_nTaskThreadPoolSize = 1;
                }
            } 
            g_eThreadPoolPolicy = ePolicy;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return the threshold at which the runtime will change the thread:task
        ///             cardinality from 1:1 to 1:N. For small graphs the former is more performant,
        ///             while for large graphs, the latter is. The knee of the curve is likely 
        ///             platform-dependent, so we need an API to control this. 
        ///             
        ///             TODO: FIXME: Make PTask choose a good initial value based on CPU count
        ///             rather than taking a hard-coded default.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetSchedulerThreadPerTaskThreshold(
            VOID
            )
        {
            return g_nTaskPerThreadPolicyThreshold;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Set the threshold at which the runtime will change the thread:task
        ///             cardinality from 1:1 to 1:N. For small graphs the former is more performant,
        ///             while for large graphs, the latter is. The knee of the curve is likely 
        ///             platform-dependent, so we need an API to control this. 
        ///             
        ///             TODO: FIXME: Make PTask choose a good initial value based on CPU count
        ///             rather than taking a hard-coded default.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetSchedulerThreadPerTaskThreshold(
            UINT uiMaxTasks
            )
        {
            warnIfInitialized("SetSchedulerThreadPerTaskThreshold");
            g_nTaskPerThreadPolicyThreshold = uiMaxTasks;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the scheduler's thread count. </summary>
        ///
        /// <remarks>   Crossbac, 3/17/2013. </remarks>
        ///
        /// <returns>   The scheduler thread count. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetSchedulerThreadCount(
            VOID
            )
        {
            return g_uiSchedulerThreadCount;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the scheduler thread count. </summary>
        ///
        /// <remarks>   Crossbac, 3/17/2013. </remarks>
        ///
        /// <param name="uiThreads">    The threads. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetSchedulerThreadCount(
            UINT uiThreads
            )
        {
            warnIfInitialized("SetSchedulerThreadCount");
            g_uiSchedulerThreadCount = uiThreads;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the (ostensible) cuda heap size. </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///
        /// <returns>   size in bytes. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetCUDAHeapSize() {
            return g_uiCUDAHeapSize;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the cuda heap size. </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///
        /// <param name="uiSizeBytes">  heap size </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetCUDAHeapSize(
            UINT uiSizeBytes
            )
        {
            g_bUserDefinedCUDAHeapSize = TRUE;
            g_uiCUDAHeapSize = uiSizeBytes;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Cublas has pathological start up cost, so lazy initialization 
        ///             winds up accruing to ptask execution time. This can be avoided
        ///             by forcing PTask to initialize cublas early, at the cost of some
        ///             unwanted dependences on it.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetInitializeCublas(
            BOOL bInitialize
            )
        {
            g_bInitCublas = bInitialize;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   When we use thread pooling, a lot of the scheduler's policy becomes
        ///             ineffective because tasks are queued up waiting not for GPUs but for
        ///             task threads. Technically, we need the same logic we have dealing with
        ///             the scheduler run queue to be present in the graph runner procs.
        ///             For now, just make it possible to sort by priority so we don't wind up
        ///             defaulting to pure FIFO behavior when thread pooling is in use.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetThreadPoolPriorityQueues(
            BOOL bSortQueue
            )
        {
            g_bSortThreadPoolQueues = bSortQueue;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   When we use thread pooling, a lot of the scheduler's policy becomes
        ///             ineffective because tasks are queued up waiting not for GPUs but for
        ///             task threads. Technically, we need the same logic we have dealing with
        ///             the scheduler run queue to be present in the graph runner procs.
        ///             For now, just make it possible to sort by priority so we don't wind up
        ///             defaulting to pure FIFO behavior when thread pooling is in use.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 4/30/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        GetThreadPoolPriorityQueues(
            VOID
            )
        {
            return g_bSortThreadPoolQueues;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        ///     Set the default partitioning mode assigned to subsequently created Graph instances.
        ///     Should pass only one of the following values:
        ///       GRAPHPARTITIONINGMODE_NONE = 0:
        ///         The runtime will not partition graphs across multiple available accelerators.
        ///       GRAPHPARTITIONINGMODE_HINTS = 1:
        ///         The runtime will partition graphs across multiple available accelerators,
        ///         according to hints given explicitly by the application via PTask::SetSchedulerPartitionHint().
        ///       GRAPHPARTITIONINGMODE_HEURISTIC = 2:
        ///         The runtime will partition graphs across multiple available accelerators,
        ///         available accelerators, using a set of experimental heuristics.
        ///       AUTOPARTITIONMODE_OPTIMAL = 2:
        ///         The runtime will attempt to auto-partition graphs across multiple
        ///         available accelerators, using a graph cut algorithm that finds the min-cut.
        ///
        ///     The default at runtime initialization is GRAPHPARTITIONINGMODE_NONE.
        /// </summary>
        ///
        /// <remarks>   jcurrey, 1/27/2014. </remarks>
        ///
        /// <param name="mode"> The default graph partitioning mode. </param>
        ///
        ///-------------------------------------------------------------------------------------------------
    
        void
        SetDefaultGraphPartitioningMode(
            int mode
            )
        {
            g_eDefaultGraphPartitioningMode = (GRAPHPARTITIONINGMODE)mode;
        }
    
        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        ///     Get the default partitioning mode assigned to subsequently created Graph instances. 
        ///     Will return one of the following values:
        ///       GRAPHPARTITIONINGMODE_NONE = 0:
        ///         The runtime will not partition graphs across multiple available accelerators.
        ///       GRAPHPARTITIONINGMODE_HINTS = 1:
        ///         The runtime will partition graphs across multiple available accelerators,
        ///         according to hints given explicitly by the application via PTask::SetSchedulerPartitionHint().
        ///       GRAPHPARTITIONINGMODE_HEURISTIC = 2:
        ///         The runtime will partition graphs across multiple available accelerators,
        ///         available accelerators, using a set of experimental heuristics.
        ///       AUTOPARTITIONMODE_OPTIMAL = 2:
        ///         The runtime will attempt to auto-partition graphs across multiple
        ///         available accelerators, using a graph cut algorithm that finds the min-cut.
        ///
        ///     The default at runtime initialization is GRAPHPARTITIONINGMODE_NONE.
        /// </summary>
        ///
        /// <remarks>   jcurrey, 1/27/2014. </remarks>
        ///
        /// <returns>   The graph partitioning mode. </returns>
        ///-------------------------------------------------------------------------------------------------
    
         int GetDefaultGraphPartitioningMode()
         {
             return g_eDefaultGraphPartitioningMode;
         }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enables/Disables fine-grain memory allocation tracking per memory space.
        ///             </summary>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetTrackDeviceMemory(
            BOOL bTrack
            )
        {
            g_bTrackDeviceAllocation = bTrack;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return the enables/disables state of fine-grain memory allocation tracking per
        ///             memory space.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        GetTrackDeviceMemory(
            VOID
            )
        {
            return g_bTrackDeviceAllocation;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets use graph monitor watchdog. </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        GetUseGraphMonitorWatchdog(
            VOID
            )
        {
            return g_bUseGraphWatchdog;
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets use graph monitor watchdog. </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///
        /// <param name="bUseWatchdog"> true to use watchdog. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetUseGraphMonitorWatchdog(
            BOOL bUseWatchdog
            )
        {
            g_bUseGraphWatchdog = bUseWatchdog;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets dispatch watchdog threshold. </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///
        /// <returns>   The dispatch watchdog threshold. </returns>
        ///-------------------------------------------------------------------------------------------------

        DWORD 
        GetDispatchWatchdogThreshold(
            VOID
            )
        {
            return g_dwGraphWatchdogThreshold;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets dispatch watchdog threshold. </summary>
        ///
        /// <remarks>   Crossbac, 3/16/2013. </remarks>
        ///
        /// <param name="dwThreshold">  The threshold. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetDispatchWatchdogThreshold(
            DWORD dwThreshold
            )
        {
            g_dwGraphWatchdogThreshold = dwThreshold;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return TRUE if there are multiple gpus in the environment. 
        /// 			Note that the scheduler must be initialized before this
        /// 			can return a meaningful result. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL MultiGPUEnvironment() { 
            if(!Runtime::g_bPTaskInitialized) {
                assert(false);
                ErrorMessage("checking for multi-GPU environment before the runtime is initialized!");
                return FALSE;
            }
            if(Runtime::g_nPhysicalAccelerators == 0) {
                assert(false);
                ErrorMessage("no accelerators present in the system!");
                return FALSE;

            }
            return g_nMaximumConcurrency != 1 && g_nPhysicalAccelerators > 1; 
        }

        // the "ignore locality threshold" is the effective priority
        // of a PTask for which we will assign an accelerator even if 
        // it is not the best fit for that PTasks data placement. In general,
        // the runtime (when using data aware scheduling) will choose the
        // accelerator in which the majority of a PTask's inputs are already
        // materialized. Sometimes, this will mean a runnable PTask does not
        // run, because it is waiting for it's *preferred* accelerator to become
        // available. Because of the priority aging mechanism, eventually a long
        // wait time will force its effective priority high. Once it's effective
        // priority passes the ignore locality threshold, the runtime will 
        // schedule it on any available accelerator, regardless of whether
        // it will have to migrate data from GPU to GPU or not. 
        /// <summary> The ignore locality threshold </summary>
        int g_nIgnoreLocalityThreshold = 6;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the ignore locality threshold. 
        /// 			See comment above for details.</summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="n">    The threshold value to set. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetIgnoreLocalityThreshold(
            int n
            ) 
        { 
            g_nIgnoreLocalityThreshold = n; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the ignore locality threshold. 
        /// 			See comment above.</summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   The ignore locality threshold. </returns>
        ///-------------------------------------------------------------------------------------------------

        int GetIgnoreLocalityThreshold(
            VOID
            ) 
        { 
            return g_nIgnoreLocalityThreshold; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   The default channel capacity control for default channel capacity. there is a
        ///             trade-off where high channel capacity can allow a single-threaded ptask driver
        ///             program to get pipeline parallelism benefits (by first pushing all its inputs and
        ///             and then pulling all its outputs)
        ///             but when the producer can produce inputs at a rate that is greater than the
        ///             consumer can consume, there is a noticeable bump in memory usage. In general, we
        ///             prefer to leave the channel capacity high, but we want the programmer to have the
        ///             ability to control this when needed.
        ///             </summary>
        ///-------------------------------------------------------------------------------------------------
        
        int g_nDefaultChannelCapacity = DEFAULT_CHANNEL_CAPACITY;


        /// <summary>   The default initializer channel pool size. </summary>
        UINT g_uiDefaultInitChannelPoolSize = DEFAULT_INIT_CHANNEL_POOL_SIZE;

        /// <summary>   The default input channel pool size. </summary>
        UINT g_uiDefaultInputChannelPoolSize = DEFAULT_INPUT_CHANNEL_POOL_SIZE;

        /// <summary>   The default block pool grow increment. </summary>
        UINT g_uiDefaultBlockPoolGrowIncrement = DEFAULT_BLOCK_POOL_GROW_INCREMENT;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the default channel capacity. 
        /// 			See comment for g_nDefaultChannelCapacity
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   The default channel capacity. </returns>
        ///-------------------------------------------------------------------------------------------------

        int GetDefaultChannelCapacity(
            VOID
            ) 
        { 
            return g_nDefaultChannelCapacity; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the default channel capacity. 
        /// 			See comment for g_nDefaultChannelCapacity
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="nDefaultChannelCapacity">  The default channel capacity. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetDefaultChannelCapacity(
            int nDefaultChannelCapacity
            ) 
        { 
            warnIfInitialized("SetDefaultChannelCapacity"); 
            g_nDefaultChannelCapacity = nDefaultChannelCapacity; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the default size of block pools for initializer channels.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The default channel capacity. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetDefaultInitChannelBlockPoolSize(
            VOID
            )
        {
            return g_uiDefaultInitChannelPoolSize;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the default size for block pools on initializer channels. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiPoolSize">   The desired pool size. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetDefaultInitChannelBlockPoolSize(
            UINT uiPoolSize
            )
        {
            warnIfInitialized("SetDefaultInitChannelBlockPoolSize");
            g_uiDefaultInitChannelPoolSize = uiPoolSize;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the default size of block pools for input channels.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The default channel capacity. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetDefaultInputChannelBlockPoolSize(
            VOID
            )
        {
            return g_uiDefaultInputChannelPoolSize;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the default size for block pools on input channels. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="uiPoolSize">   The desired pool size. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetDefaultInputChannelBlockPoolSize(
            UINT uiPoolSize
            )
        {
            warnIfInitialized("SetDefaultInputChannelBlockPoolSize");
            g_uiDefaultInputChannelPoolSize = uiPoolSize;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the default grow increment for block pools.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <returns>   The default channel capacity. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GetDefaultBlockPoolGrowIncrement(
            VOID
            )
        {
            return g_uiDefaultBlockPoolGrowIncrement;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the default grow increment for block pools. </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="n">    The desired pool size. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetDefaultBlockPoolGrowIncrement(
            UINT n
            )
        {
            warnIfInitialized("SetDefaultBlockPoolGrowIncrement");
            g_uiDefaultBlockPoolGrowIncrement = n;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets provision block pools for capacity. When this mode is set
        ///             the runtime will try to allocate block pools such that they
        ///             can satisfy all downstream request without allocation. This
        ///             requires being able to fill the capacity of all downstream channels
        ///             along inout consumer paths. This is a handy tool, but should not
        ///             be used unless you are actually attempting tune channel capacities
        ///             because the default channel capacity is very large. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/21/2013. </remarks>
        ///
        /// <param name="bProvision">   true to provision. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetProvisionBlockPoolsForCapacity(
            __in BOOL bProvision
            )
        {
            warnIfInitialized("SetProvisionBlockPoolsForCapacity");
            g_bProvisionBlockPoolsForCapacity = bProvision;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets provision block pools for capacity. When this mode is set
        ///             the runtime will try to allocate block pools such that they
        ///             can satisfy all downstream request without allocation. This
        ///             requires being able to fill the capacity of all downstream channels
        ///             along inout consumer paths. This is a handy tool, but should not
        ///             be used unless you are actually attempting tune channel capacities
        ///             because the default channel capacity is very large. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/21/2013. </remarks>
        ///
        /// <param name="bProvision">   true to provision. </param>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        GetProvisionBlockPoolsForCapacity(
            VOID
            )
        {
            return g_bProvisionBlockPoolsForCapacity;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Size of the datablock garbage collector batch garbage collection control.
        ///             Datablocks that are no longer referenced are queued and deleted by a low priority
        ///             thread. Some level of batching is clearly advantageous. This parameter controls
        ///             the length the queue must reach before the GC thread will get unblocked. TODO:
        ///             There are likely to be low-memory situations in which we would prefer to
        ///             prioritize GC activities. We need to build some support for this sort of thing.
        ///             </summary>
        ///-------------------------------------------------------------------------------------------------

        int g_nGCBatchSize = DEFAULT_GCBATCH_SIZE;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the gc batch size. 
        /// 			See comment for g_nGCBatchSize above </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   The gc batch size. </returns>
        ///-------------------------------------------------------------------------------------------------

        int GetGCBatchSize() { return g_nGCBatchSize; }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a gc batch size.
        /// 			See comment for g_nGCBatchSize above </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="n">    The batch size value to set. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetGCBatchSize(int n) { g_nGCBatchSize = n; }

        
        ///-------------------------------------------------------------------------------------------------
        /// <summary> 
        ///     Size of the internal channel block pool.
        ///     control for internal channel block pooling. To avoid
        ///     device-side memory-allocation overheads, output ports
        ///     maintain a block pool. Releases from downstream channels
        ///     return the block to the port's pool when the refcount 
        ///     drops to zero rather than releasing them. 
        /// </summary>
        ///-------------------------------------------------------------------------------------------------
        int g_nICBlockPoolSize = DEFAULT_INTERNAL_BLOCK_POOL_SIZE;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the internal channel block pool size. 
        /// 			See the comment for g_nICBlockPoolSize above.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   The ic block pool size. </returns>
        ///-------------------------------------------------------------------------------------------------

        int GetICBlockPoolSize() { return g_nICBlockPoolSize; } 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets an ic block pool size. 
        /// 			See the comment for g_nICBlockPoolSize above.
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="n">    The block pool size. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetICBlockPoolSize(
            int n
            ) 
        { 
            warnIfInitialized("SetICBlockPoolSize"); 
            g_nICBlockPoolSize = n; 
        }

        // control which runtimes are used. 
        // handy if you know you only care about
        // a particular runtime, scheduler data
        // structures can be simplified. Use them
        // all by default
        BOOL g_bUseHost = TRUE;
        BOOL g_bUseCUDA = TRUE;
        BOOL g_bUseOpenCL = TRUE;
        BOOL g_bUseDirectX = TRUE;
        BOOL GetUseHost() { return g_bUseHost; }
        BOOL GetUseCUDA() { return g_bUseCUDA; }
        BOOL GetUseOpenCL() { return g_bUseOpenCL; }
        BOOL GetUseDirectX() { return g_bUseDirectX; }
        void SetUseHost(BOOL b) { warnIfInitialized("SetUseHost"); g_bUseHost = b; }
        void SetUseCUDA(BOOL b) { warnIfInitialized("SetUseCUDA"); g_bUseCUDA = b; }
        void SetUseOpenCL(BOOL b) { warnIfInitialized("SetUseCUDA"); g_bUseOpenCL = b; }
        void SetUseDirectX(BOOL b) { warnIfInitialized("SetUseCUDA"); g_bUseDirectX = b; }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the scheduling mode. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <returns>   The scheduling mode. </returns>
        ///-------------------------------------------------------------------------------------------------

        int GetSchedulingMode() { 
            return Scheduler::GetSchedulingMode();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the scheduling mode. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="mode"> The mode. </param>
        ///-------------------------------------------------------------------------------------------------

        void SetSchedulingMode(int mode) {
            return Scheduler::SetSchedulingMode((SCHEDULINGMODE) mode);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is initialized. </summary>
        ///
        /// <remarks>   Crossbac, 7/12/2012. </remarks>
        ///
        /// <returns>   true if initialized, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL IsInitialized(
            VOID
            ) 
        { 
            return g_bPTaskInitialized; 
        }

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Allocate global pools. </summary>
		///
		/// <remarks>	crossbac, 8/14/2013. </remarks>
		///-------------------------------------------------------------------------------------------------

		void
		AllocateGlobalPools(
			VOID
			)
		{            
            GlobalPoolManager::RequireBlockPool(Datablock::m_gSizeDescriptorTemplate, g_uiDescriptorBlockPoolSize);
            GlobalPoolManager::Create();
		}

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Destroys the global pools. </summary>
		///
		/// <remarks>	crossbac, 8/14/2013. </remarks>
		///-------------------------------------------------------------------------------------------------

		void 
		DestroyGlobalPools(		
			VOID
			)
		{
            GlobalPoolManager::Destroy();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>
        ///     Initializes the ptask runtime. this includes starting up the scheduler, enumerating
        ///     available accelerators, and setting up data structures that map schedule calls back to
        ///     output endpoints.
        /// </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="">void</param>
        ///
        /// <returns>   PTRESULT code indicating outcome </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        Initialize(
            VOID
            ) 
        {            
            if(g_bPTaskInitialized) 
                return PTASK_ERR;
            INITNVTX();
            MARKRANGEENTER(L"PTask::Initialize");
            if(IsVerbose()) 
                PrintRuntimeConfiguration(std::cout);            
            CreateRuntimeSyncObjects(TRUE);
            InitializeCriticalSection(&g_csTraceList);
            Port::InitializeGlobal();
            INITRECORDER();
            Instrumenter::Initialize();
            SignalProfiler::Initialize();
            ReferenceCountedProfiler::Initialize(PTask::Runtime::GetRCProfilingEnabled());
            DatablockProfiler::Initialize(PTask::Runtime::GetDBProfilingEnabled());
            CoherenceProfiler::Initialize(PTask::Runtime::GetCTProfilingEnabled());
            TaskProfile::Initialize();
            DispatchCounter::Initialize();
            PBuffer::InitializeProfiler();
            ChannelProfiler::Initialize(PTask::Runtime::GetChannelProfilingEnabled());
            MemorySpace::InitializeMemorySpaces();
            GarbageCollector::CreateGC();
            Scheduler::Initialize();
            ThreadPool::Create(GetGlobalThreadPoolSize(), GetPrimeGlobalThreadPool(), GetGlobalThreadPoolGrowable());
            Datablock::m_gSizeDescriptorTemplate = new DatablockTemplate("global_size_desc", sizeof(int), PTPARM_INT);
            BlockPoolOwner::InitializeBlockPoolManager();
			AllocateGlobalPools();
            g_uiUIDCounter = 0;
            InitializeCriticalSection(&csUID);
            InitializeDebugLogging();
            g_bPTaskInitialized = TRUE;
            SetEvent(g_hRuntimeInitialized);
            UnlockRuntime();
            MARKRANGEEXIT();
            return PTASK_OK;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Deletes the templates. </summary>
        ///
        /// <remarks>   crossbac, 7/9/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        static void
        DeleteTemplates(
            VOID
            ) 
        {
            set<DatablockTemplate*>::iterator si;
            for(si=g_vTemplates.begin(); 
                si!=g_vTemplates.end(); si++) {
                    delete (*si);
            }
            g_vTemplates.clear();
            delete Datablock::m_gSizeDescriptorTemplate;
            Datablock::m_gSizeDescriptorTemplate = NULL;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destroys the kernels. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void DestroyKernels(
            VOID
            )
        {
            vector<CompiledKernel*>::iterator ki;
            for(ki=g_vKernels.begin(); 
                ki!=g_vKernels.end(); ki++) {
                    CompiledKernel* pKernel = *ki;
                    delete pKernel;
            }
            g_vKernels.clear();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Locks the runtime. </summary>
        ///
        /// <remarks>   Crossbac, 7/22/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        LockRuntime(
            VOID
            )
        {
            // The contract with PTask::Runtime::Initialize and PTask::Runtime::Terminate is that they are
            // expected to be called in sequential (or at least thread-safe) calling contexts preferably
            // once per-address space, even if multiple graphs with overlapping lifetimes are to be used in
            // that address space. However, there are a lot of ways the user can violate that contract, so
            // we make at least some effort to guard against multiple concurrent runtime-level init/
            // teardown callers. We use a global mutex around initialize/teardown to attempt to enforce this
            // contract. This method locks that mutex, but expects it to exist!

            if((g_hRuntimeMutex == INVALID_HANDLE_VALUE) ||
               (g_hRuntimeMutex == NULL)) {

                // some other thread already closed down the runtime out from under this thread. This is a
                // gross violation  of the init/terminate contract for ptask, complain vigorously about it! 
                   
                PTask::Runtime::HandleError("XXXX:\n"
                                            "XXXX:  %s:%s: attempt to lock non-existent PTask runtime mutex!\n"
                                            "XXXX:  Multiple concurrent/overlapping PTask::Initialize/Teardown\n"
                                            "XXXX:  cycles initiated in the same address space? Cannot recover...\n" 
                                            "XXXX:  PLEASE INIT/TEARDOWN PTASK ONLY ONCE PER PROCESS IF POSSIBLE!\n"
                                            "XXXX:\n"
                                            __FILE__,
                                            __FUNCTION__);

            } else {

                WaitForSingleObject(g_hRuntimeMutex, INFINITE);
            }        
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Unlocks the runtime. </summary>
        ///
        /// <remarks>   Crossbac, 7/22/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        UnlockRuntime(
            VOID
            )
        {
            // The contract with PTask::Runtime::Initialize and PTask::Runtime::Terminate is that they are
            // expected to be called in sequential (or at least thread-safe) calling contexts preferably
            // once per-address space, even if multiple graphs with overlapping lifetimes are to be used in
            // that address space. However, there are a lot of ways the user can violate that contract, so
            // we make at least some effort to guard against multiple concurrent runtime-level init/
            // teardown callers. We use a global mutex around initialize/teardown to attempt to enforce this
            // contract. This method *unlocks* that mutex, but expects it to exist!

            if((g_hRuntimeMutex == INVALID_HANDLE_VALUE) ||
               (g_hRuntimeMutex == NULL)) {

                // some other thread already closed down the runtime out from under this thread. This is a
                // gross violation  of the init/terminate contract for ptask, complain vigorously about it! 
                   
                PTask::Runtime::HandleError("XXXX:\n"
                                            "XXXX:  %s:%s: attempt to unlock non-existent PTask runtime mutex!\n"
                                            "XXXX:  Multiple concurrent/overlapping PTask::Initialize/Teardown\n"
                                            "XXXX:  cycles initiated in the same address space? Cannot recover...\n" 
                                            "XXXX:  PLEASE INIT/TEARDOWN PTASK ONLY ONCE PER PROCESS IF POSSIBLE!\n"
                                            "XXXX:\n"
                                            __FILE__,
                                            __FUNCTION__);

            } else {

                ReleaseMutex(g_hRuntimeMutex);
            }        
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates runtime synchronise objects. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void CreateRuntimeSyncObjects(
            BOOL bLockedOnExit
            )
        {
            // The contract with PTask::Runtime::Initialize and PTask::Runtime::Terminate is that they are
            // expected to be called in sequential (or at least thread-safe) calling contexts preferably
            // once per-address space, even if multiple graphs with overlapping lifetimes are to be used in
            // that address space. However, there are a lot of ways the user can violate that contract, so
            // we make at least some effort to guard against multiple concurrent runtime-level init/
            // teardown callers. 

            if(g_hRuntimeMutex == INVALID_HANDLE_VALUE) {

                HANDLE hRuntimeMutex = CreateMutex(NULL, bLockedOnExit, L"PTask::Runtime.BigRuntimeLock");
                DWORD dwCreateResult = GetLastError();
                if(dwCreateResult == ERROR_ALREADY_EXISTS) {

                    // the handle already existed. that means it was created by another thread, and the return
                    // value here points to the same sync object. If we have a distinct handle to the same sync
                    // object, close it, and then wait on the existing handle. Since this is a gross violation
                    // of the init/terminate contract for ptask, complain vigorously about it!
                   
                    PTask::Runtime::MandatoryInform("XXXX:\n"
                                                    "XXXX:  %s:%s: multiple concurrent/overlapping PTask::Initialize/Teardown\n"
                                                    "XXXX:  cycles initiated in the same address space! Attempting to recover...\n" 
                                                    "XXXX:  PLEASE INIT/TEARDOWN PTASK ONLY ONCE PER PROCESS IF POSSIBLE!\n"
                                                    "XXXX:\n"
                                                    __FILE__,
                                                    __FUNCTION__);
                    
                    if(hRuntimeMutex != NULL && 
                       hRuntimeMutex != INVALID_HANDLE_VALUE && 
                       hRuntimeMutex != g_hRuntimeMutex && 
                       g_hRuntimeMutex != INVALID_HANDLE_VALUE) {

                        // another thread already opened a handle to the same lock, and got
                        // a distinct handle value. Close our handle and wait on the original.

                        CloseHandle(hRuntimeMutex);
                        assert(g_hRuntimeMutex != NULL && g_hRuntimeMutex != INVALID_HANDLE_VALUE);
                        WaitForSingleObject(g_hRuntimeMutex, INFINITE);
                    }

                } else {

                    if(hRuntimeMutex == NULL) {

                        // the mutex create failed for some reason. this is fatal.
                        // complain vociferously about it and then appeal to the runtime.
                        PTask::Runtime::HandleError("XXXX:\n"
                                                    "XXXX:  %s:%s:%s initialization failure: could not create PTask BKL mutex\n"
                                                    "XXXX:\n"
                                                    __FILE__,
                                                    __FUNCTION__, 
                                                    "CreateMutex");
                    } else {
                        
                        // we successfully create the mutex and we hold the lock since we
                        // requested ownership as part of the API call to create it. All good. 
                        g_hRuntimeMutex = hRuntimeMutex;

                    }
                }

            } else {

                // the runtime lock already exists. Go ahead and wait on it. 
                assert(g_hRuntimeMutex != NULL && g_hRuntimeMutex != INVALID_HANDLE_VALUE);
                WaitForSingleObject(g_hRuntimeMutex, INFINITE);                
            }

            // we hold the PTask's "BKL". create the runtime 
            // terminate event if it does not exist already,

            if(!PTVALIDSYNCHANDLE(g_hRuntimeTerminateEvent)) {
                g_hRuntimeTerminateEvent = CreateEvent(NULL, TRUE, FALSE, L"PTask::Runtime.Terminate.Event");                
            }

            if(!PTVALIDSYNCHANDLE(g_hRuntimeInitialized)) {
                g_hRuntimeInitialized = CreateEvent(NULL, TRUE, FALSE, L"PTask::Runtime.Initialized.Event");                
            }

            // if the caller doesn't want us to exit with the 
            // lock held, then release the runtime mutex. 
            if(!bLockedOnExit) {
                ReleaseMutex(g_hRuntimeMutex);
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destroys the runtime synchronise objects. </summary>
        ///
        /// <remarks>   Crossbac, 7/19/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void DestroyRuntimeSyncObjects(
            BOOL bLockedOnEntry
            )
        {
            assert(PTVALIDSYNCHANDLE(g_hRuntimeTerminateEvent));
            assert(PTVALIDSYNCHANDLE(g_hRuntimeMutex));
            assert(PTVALIDSYNCHANDLE(g_hRuntimeInitialized));

            // The contract with PTask::Runtime::Initialize and PTask::Runtime::Terminate is that they are
            // expected to be called in sequential (or at least thread-safe) calling contexts preferably
            // once per-address space, even if multiple graphs with overlapping lifetimes are to be used in
            // that address space. However, there are a lot of ways the user can violate that contract, so
            // we make at least some effort to guard against multiple concurrent runtime-level init/
            // teardown callers. 

            if(!PTVALIDSYNCHANDLE(g_hRuntimeTerminateEvent) || 
               !PTVALIDSYNCHANDLE(g_hRuntimeMutex) ||
               !PTVALIDSYNCHANDLE(g_hRuntimeInitialized)) {

                // some other thread already closed down the runtime out from under this thread. This is a
                // gross violation  of the init/terminate contract for ptask, complain vigorously about it! 
                   
                PTask::Runtime::HandleError("XXXX:\n"
                                            "XXXX:  %s:%s: multiple concurrent/overlapping PTask::Initialize/Teardown\n"
                                            "XXXX:  cycles initiated in the same address space! Attempting to recover...\n" 
                                            "XXXX:  PLEASE INIT/TEARDOWN PTASK ONLY ONCE PER PROCESS IF POSSIBLE!\n"
                                            "XXXX:\n"
                                            __FILE__,
                                            __FUNCTION__);

            } else {

                if(!bLockedOnEntry) {

                    // this really shouldn't be called this way, but just in case, acquire the runtime mutex before
                    // destroying the handles to these sync objects. The mutex is supposed to protect init/teardown
                    // work, and this call is one of the last steps in the the latter, so the runtime mutex really
                    // should be held already. 

                    WaitForSingleObject(g_hRuntimeMutex, INFINITE);
                }

                CloseHandle(g_hRuntimeTerminateEvent);
                g_hRuntimeTerminateEvent = INVALID_HANDLE_VALUE;
                CloseHandle(g_hRuntimeInitialized);
                g_hRuntimeInitialized = INVALID_HANDLE_VALUE;
                HANDLE hRuntimeMutex = g_hRuntimeMutex;
                g_hRuntimeMutex = INVALID_HANDLE_VALUE;
                ReleaseMutex(hRuntimeMutex);
                CloseHandle(hRuntimeMutex);
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   provide a report on a subsystem. </summary>
        ///
        /// <remarks>   crossbac, 9/10/2013. </remarks>
        ///
        /// <param name="eSubSystem">   The sub system. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        SubsystemReport(
            __in    PTASKSUBSYSTEM eSubSystem
            )                
        {
            return SubsystemReport(eSubSystem, std::cerr);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   provide a report on a subsystem. </summary>
        ///
        /// <remarks>   crossbac, 9/10/2013. </remarks>
        ///
        /// <param name="eSubSystem">   The sub system. </param>
        /// <param name="ss">           [in,out] The ss. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        SubsystemReport(
            __in    PTASKSUBSYSTEM eSubSystem,
            __inout std::ostream&  ss
            )                
        {
            switch(eSubSystem) {
            case PTSYS_TASKS: TaskProfile::Report(ss); return TRUE;
            case PTSYS_TASK_MIGRATION : TaskProfile::MigrationReport(ss); return TRUE;
            case PTSYS_PBUFFERS: PBuffer::ProfilerReport(ss); return TRUE;
            case PTSYS_DATABLOCKS: DatablockProfiler::Report(ss); return TRUE; 
            case PTSYS_COHERENCE: CoherenceProfiler::Report(ss); return TRUE;
            case PTSYS_CHANNELS: ChannelProfiler::Report(ss); return TRUE;
            case PTSYS_DISPATCH: DispatchCounter::Report(ss); return TRUE;
            case PTSYS_REFCOUNT_OBJECTS: ReferenceCountedProfiler::Report(ss); return TRUE;
            case PTSYS_ADHOC_INSTRUMENTATION: Instrumenter::Report(ss); return TRUE;
            default: break; 
            }

            // if we got here, the call asked for a report from a subsystem
            // we don't know about. Complain and recover
            MandatoryInform("%s:%s(%d): unknown system: cannot report!\n",
                            __FILE__,
                            __FUNCTION__,
                            (int)eSubSystem);
            return FALSE; // failed
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Terminates 
        /// 			the runtime </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name=""> The. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        Terminate(
            VOID
            ) 
        {
            MARKRANGEENTER(L"PTask::Terminate");
            if(!g_bPTaskInitialized) {
                PTask::Runtime::Warning("WARNING: Runtime::Terminate called on uninitialized runtime instance (ignoring)...\n");
                return PTASK_ERR_UNINITIALIZED;
            }
            LockRuntime();
            Scheduler::Shutdown();
            ResetEvent(g_hRuntimeInitialized);
            SetEvent(g_hRuntimeTerminateEvent);
            DestroyKernels();
            DestroyGlobalPools();
            BlockPoolOwner::DestroyBlockPoolManager();
            GarbageCollector::ForceGC();
            GarbageCollector::DestroyGC();    
            Scheduler::Destroy();
            ThreadPool::Destroy();
            DeleteCriticalSection(&csUID);
            TerminateDebugLogging();
            g_bPTaskInitialized = FALSE;
            TaskProfile::MigrationReport(std::cerr);
            TaskProfile::Report(std::cerr);
            TaskProfile::Deinitialize();            
            PBuffer::ProfilerReport(std::cerr);
            PBuffer::DeinitializeProfiler();
            DatablockProfiler::Report(std::cerr);
            DatablockProfiler::Deinitialize();
            CoherenceProfiler::Report(std::cerr);
            CoherenceProfiler::Deinitialize();
            ChannelProfiler::Report(std::cerr);
            ChannelProfiler::Deinitialize();
            DispatchCounter::Report(std::cerr);
            DispatchCounter::Deinitialize();
            SignalProfiler::Report(std::cerr);
            SignalProfiler::Deinitialize();
            DeleteCriticalSection(&g_csTraceList);
            DeleteTemplates();
            ReferenceCountedProfiler::Report(std::cerr);
            ReferenceCountedProfiler::Deinitialize();
            DESTROYRECORDER();           
            MemorySpace::UnregisterMemorySpaces();
            Instrumenter::Report(std::cerr);
            Instrumenter::Destroy();
            Port::DestroyGlobal();
            DestroyRuntimeSyncObjects(TRUE);
            MARKRANGEEXIT();
            return PTASK_OK;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a compiled kernel. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="lpszFile">             [in,out] If non-null, the file. </param>
        /// <param name="lpszOperation">        [in,out] If non-null, the operation. </param>
        /// <param name="lpszCompilerOutput">   [in,out] If non-null, the compiler output. </param>
        /// <param name="uiCompilerOutput">     The compiler output. </param>
        /// <param name="tgx">                  The tgx. </param>
        /// <param name="tgy">                  The tgy. </param>
        /// <param name="tgz">                  The tgz. </param>
        ///
        /// <returns>   null if it fails, else the compiled kernel. </returns>
        ///-------------------------------------------------------------------------------------------------

        CompiledKernel * 
        GetCompiledKernel(
            char * lpszFile, 
            char * lpszOperation,
            char * lpszCompilerOutput,
            int uiCompilerOutput,
            int tgx, 
            int tgy, 
            int tgz
            )
		{
			return GetCompiledKernelEx(lpszFile,
									   lpszOperation,
									   NULL,
									   NULL,
									   ACCELERATOR_CLASS_UNKNOWN,
									   lpszCompilerOutput,
									   uiCompilerOutput,
									   tgx,
									   tgy,
									   tgz);
		}

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Compiles accelerator source code to create a PTask binary. This variant
		/// 			accomodates tasks that may have a global initializer routine that must be called
		/// 			before the graph enters the run state (e.g. creation of block pools).
		/// 			</summary>
		///
		/// <remarks>	The function accepts a file name and an operation in the file to build a binary
		/// 			for. For example, "foo.hlsl" and "vectoradd" will compile the vectoradd() shader
		/// 			in foo.hlsl. On success the function will create platform-specific binary and
		/// 			module objects that can be later used by the runtime to invoke the shader code.
		/// 			The caller can provide a buffer for compiler output, which if present, the
		/// 			runtime will fill *iff* the compilation fails.
		/// 			***
		/// 			NB: Thread group dimensions are optional parameters here. This is because some
		/// 			runtimes require them statically, and some do not. DirectX requires thread-group
		/// 			sizes to be specified statically to enable compiler optimizations that cannot be
		/// 			used otherwise. CUDA and OpenCL allow runtime specification of these parameters.
		/// 			***
		/// 			If an initializer file and entry point are provided, the runtime will load the
		/// 			corresponding binary and call the entry point upon graph completion.
		/// 			</remarks>
		///
		/// <param name="lpszFile">						[in] filename+path of source. cannot be null. </param>
		/// <param name="lpszOperation">				[in] Function name in source file. non-null. </param>
		/// <param name="lpszInitializerBinary">		[in] filename+path for initializer DLL. null OK. </param>
		/// <param name="lpszInitializerEntryPoint">	[in] entry point for initializer code. null OK. </param>
		/// <param name="eInitializerPSClass">			The initializer ps class. </param>
		/// <param name="lpszCompilerOutput">			[in,out] (optional)  On failure, the compiler
		/// 											output. </param>
		/// <param name="uiCompilerOutput">				[in] length of buffer supplied for compiler
		/// 											output. </param>
		/// <param name="tgx">							thread group X dimensions. (see remarks) </param>
		/// <param name="tgy">							thread group Y dimensions. (see remarks) </param>
		/// <param name="tgz">							thread group Z dimensions. (see remarks) </param>
		///
		/// <returns>	a new compiled kernel object if it succeeds, null if it fails. </returns>
		///-------------------------------------------------------------------------------------------------

		CompiledKernel * 
		GetCompiledKernelEx(
			__in char *            lpszFile, 
			__in char *            lpszOperation, 
			__in char *            lpszInitializerBinary,
			__in char *            lpszInitializerEntryPoint,
			__in ACCELERATOR_CLASS eInitializerPSClass,
			__in char *            lpszCompilerOutput,
			__in int               uiCompilerOutput,
			__in int               tgx, 
			__in int               tgy, 
			__in int               tgz
			)
        {	
            static BOOL bWarningShown = FALSE;
            set<Accelerator*> vAccelerators;
            set<Accelerator*>::iterator vi;
            ACCELERATOR_CLASS accClass = ptaskutils::SelectAcceleratorClass(lpszFile);
            if(ACCELERATOR_CLASS_DIRECT_X != accClass) {
                if(tgx != 1 || tgy != 1 || tgz != 1) {
                    // if the thread group sizes are not the default
                    // parameters, the user might be expecting them 
                    // to have some effect. They won't if the target
                    // runtime doesn't handle thread group sizes as
                    // a compile-time specification. 
                    PTask::Runtime::Warning("WARNING: Compile-time specification of thread group size is currently only supported for DirectX kernels");
                }
            } else {
                if(tgx == 1 && tgy == 1 && tgz == 1 && !bWarningShown) {
                    // if the thread group sizes are not the default
                    // parameters, the user might be expecting them 
                    // to have some effect. They won't if the target
                    // runtime doesn't handle thread group sizes as
                    // a compile-time specification. 
                    PTask::Runtime::Warning("PERFORMANCE WARNING:");
                    PTask::Runtime::Warning("DirectX uses Compile-time specification of thread group sizes.");
                    PTask::Runtime::Warning("Taking the default thread group sizes in PTask::Runtime::GetCompiledKernel");
                    PTask::Runtime::Warning("is unlikely to yield optimal performance.");
                    bWarningShown = TRUE;
                }
            }
            Scheduler::FindEnabledCapableAccelerators(accClass, vAccelerators);
            CompiledKernel * pKernel = new CompiledKernel(lpszFile, 
													      lpszOperation, 
														  lpszInitializerBinary, 
														  lpszInitializerEntryPoint,
														  eInitializerPSClass);

			int nSucceeded = 0;
            for(vi=vAccelerators.begin(); vi!=vAccelerators.end(); vi++) {
                void * pShader = NULL;
                void * pModule = NULL;
                Accelerator * pAccelerator = (*vi);
                if(pAccelerator->Compile(lpszFile, lpszOperation, &pShader, &pModule, lpszCompilerOutput, uiCompilerOutput, tgx, tgy, tgz)) {
                    pKernel->SetPlatformSpecificBinary(pAccelerator, pShader);
                    pKernel->SetPlatformSpecificModule(pAccelerator, pModule);
                    nSucceeded++;
                }
            }
            if(!nSucceeded) {
                std::string strError("Compile failed for ");
                strError += lpszFile;
                strError += lpszOperation;
                if(PTask::Runtime::IsVerbose()) {
                    PTask::Runtime::Warning(strError.c_str());
                }
                delete pKernel;
                pKernel = NULL;
            } else if(pKernel != NULL) {
                g_vKernels.push_back(pKernel);
            }
            return pKernel;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a compiled kernel object for a provided host function. </summary>
        ///
        /// <remarks>   jcurrey, 2/24/2014. </remarks>
        ///
        /// <param name="lpszOperation">   [in] Function name, used only for identification. Cannot be null.</param>
        /// <param name="lpfn">            [in] Function pointer. Cannot be null.</param>
        ///
        /// <returns> A new compiled kernel object if it succeeds, null if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        CompiledKernel *
        GetHostFunctionCompiledKernel(
            char * lpszOperation,
            FARPROC lpfn)

        {
            set<Accelerator*> vAccelerators;
            Scheduler::FindEnabledCapableAccelerators(ACCELERATOR_CLASS_HOST, vAccelerators);
            if (vAccelerators.size() < 1) {
                PTask::Runtime::HandleError("GetHostCompiledKernel called but no host accelerators found.");
                return NULL;
            }

            CompiledKernel * pKernel = new CompiledKernel("HostFunction",
                                                          lpszOperation,
                                                          NULL,
                                                          NULL,
                                                          ACCELERATOR_CLASS_UNKNOWN);

            set<Accelerator*>::iterator vi;
            for(vi=vAccelerators.begin(); vi!=vAccelerators.end(); vi++) {
                Accelerator * pAccelerator = (*vi);
                pKernel->SetPlatformSpecificModule(pAccelerator, "HostFunction");
                pKernel->SetPlatformSpecificBinary(pAccelerator, lpfn);
            }

            g_vKernels.push_back(pKernel);
            return pKernel;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a port. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="type">                     The type. (input, output, sticky, meta, initializer) </param>
        /// <param name="pTemplate">                [in,out] If non-null, the datablock template. </param>
        /// <param name="uiId">                     unique id for the port (arbitrary) </param>
        /// <param name="lpszVariableBinding">      [in] If non-null, name of variable to which this port
        ///                                         is bound in shader (optional parameter, default = NULL) </param>
        /// <param name="nKernelParameterIndex">    other than -1 means this port is bound to a formal
        ///                                         parameter with the given 0-based index. -1 (default)
        ///                                         means bound to a global. (optional parameter, default = -
        ///                                         1) </param>
        /// <param name="nInOutParmOutputIndex">    other than -1 means this port describes a ref
        ///                                         parameter, meaning a modified input block should be
        ///                                         pushed to the output at the given index after kernel
        ///                                         dispatch (optional parameter, default = -1) </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Port * 
        CreatePort(
            PORTTYPE type,
            DatablockTemplate * pTemplate, 
            UINT uiId,
            char * lpszVariableBinding,
            UINT nKernelParameterIndex, 
            UINT nInOutParmOutputIndex
            ) 
        {
            switch(type) {
            case INPUT_PORT: 
                return InputPort::Create(pTemplate, 
                                        uiId, 
                                        lpszVariableBinding, 
                                        nKernelParameterIndex, 
                                        nInOutParmOutputIndex);
            case OUTPUT_PORT: 
                return OutputPort::Create(pTemplate, 
                                        uiId, 
                                        lpszVariableBinding, 
                                        nKernelParameterIndex, 
                                        nInOutParmOutputIndex);
            case STICKY_PORT: 
                return StickyPort::Create(pTemplate, 
                                        uiId, 
                                        lpszVariableBinding, 
                                        nKernelParameterIndex, 
                                        nInOutParmOutputIndex);
            case META_PORT: 
                return MetaPort::Create(pTemplate, 
                                        uiId, 
                                        lpszVariableBinding, 
                                        nKernelParameterIndex, 
                                        nInOutParmOutputIndex);
            case INITIALIZER_PORT: 
                return InitializerPort::Create(pTemplate, 
                                        uiId, 
                                        lpszVariableBinding, 
                                        nKernelParameterIndex, 
                                        nInOutParmOutputIndex);

            default:
                assert(FALSE);
                break;
            }
            return NULL;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a new datablock template. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="lpszType">         [in,out] If non-null, the type. </param>
        /// <param name="uiStride">         The stride. </param>
        /// <param name="x">                The x dimensions. </param>
        /// <param name="y">                The y dimensions. </param>
        /// <param name="z">                The z dimensions. </param>
        /// <param name="bRecordStream">    true if the template describes a record stream. </param>
        /// <param name="bRaw">             true if the template describes raw blocks. </param>
        ///
        /// <returns>   null if it fails, else the datablock template. </returns>
        ///-------------------------------------------------------------------------------------------------

        DatablockTemplate * 
        GetDatablockTemplate(
            char * lpszType, 
            unsigned int uiStride, 
            unsigned int x, 
            unsigned int y, 
            unsigned int z,
            bool bRecordStream,
            bool bRaw
            )
        {
            // TODO:
            // XXXX: 
            // check to see if we've already got a template that matches!
            DatablockTemplate * pResult = new DatablockTemplate(lpszType, uiStride, x, y, z, bRecordStream, bRaw);
            pResult->AddRef();
            g_vTemplates.insert(pResult);
            return pResult;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a datablock template. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="lpszTemplateName">     [in] If non-null, name of the template. </param>
        /// <param name="uiElementStride">      [in] The element stride in bytes. </param>
        /// <param name="uiElementsX">          [in] Number of elements in X dimension. </param>
        /// <param name="uiElementsY">          [in] Number of elements in Y dimension. </param>
        /// <param name="uiElementsZ">          [in] Number of elements in Z dimension. </param>
        /// <param name="uiPitch">              [in] The row pitch. </param>
        /// <param name="bIsRecordStream">      [in] true if this object is record stream. </param>
        /// <param name="bIsByteAddressable">   [in] true if this object is byte addressable. </param>
        ///-------------------------------------------------------------------------------------------------

        DatablockTemplate * 
        GetDatablockTemplate(
            __in char *       lpszTemplateName, 
            __in unsigned int uiElementStride, 
            __in unsigned int uiElementsX, 
            __in unsigned int uiElementsY, 
            __in unsigned int uiElementsZ,
            __in unsigned int uiPitch,
            __in bool         bIsRecordStream,
            __in bool         bIsByteAddressable
            )
        {
            // TODO:
            // XXXX: 
            // check to see if we've already got a template that matches!
            DatablockTemplate * pResult = 
                new DatablockTemplate(lpszTemplateName,
                                      uiElementStride,
                                      uiElementsX,
                                      uiElementsY,
                                      uiElementsZ,
                                      uiPitch,
                                      bIsRecordStream,
                                      bIsByteAddressable);
            pResult->AddRef();
            g_vTemplates.insert(pResult);
            return pResult;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a datablock template. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="lpszTemplateName">     [in] If non-null, name of the template. </param>
        /// <param name="pBufferDims">          [in] The element stride in bytes. </param>
        /// <param name="uiNumBufferDims">      [in] Number of elements in X dimension. </param>
        /// <param name="bIsRecordStream">      [in] true if this object is record stream. </param>
        /// <param name="bIsByteAddressable">   [in] true if this object is byte addressable. </param>
        ///-------------------------------------------------------------------------------------------------

        DatablockTemplate * 
        GetDatablockTemplate(
            __in char *             lpszTemplateName, 
            __in BUFFERDIMENSIONS * pBufferDims, 
            __in unsigned int       uiNumBufferDims, 
            __in bool               bIsRecordStream,
            __in bool               bIsByteAddressable
            )
        {
            // TODO:
            // XXXX: 
            // check to see if we've already got a template that matches!
            DatablockTemplate * pResult = 
                new DatablockTemplate(lpszTemplateName,
                                      pBufferDims,
                                      uiNumBufferDims,
                                      bIsRecordStream,
                                      bIsByteAddressable);
            pResult->AddRef();
            g_vTemplates.insert(pResult);
            return pResult;
        }        


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a new datablock template. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="lpszType"> [in,out] If non-null, the type. </param>
        /// <param name="uiStride"> The stride. </param>
        /// <param name="pttype">   The pttype. </param>
        ///
        /// <returns>   null if it fails, else the datablock template. </returns>
        ///-------------------------------------------------------------------------------------------------

        DatablockTemplate * 
        GetDatablockTemplate(
            char * lpszType, 
            unsigned int uiStride,
            PTASK_PARM_TYPE pttype
            )
        {
            // For scalar parameters only
            DatablockTemplate * pResult = new DatablockTemplate(lpszType, uiStride, pttype);
            pResult->AddRef();
            g_vTemplates.insert(pResult);
            return pResult;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets an existing datablock template by type name. </summary>
        ///
        /// <remarks>   jcurrey, 5/8/2013. </remarks>
        ///
        /// <param name="lpszType"> The name of the template type looking for. </param>
        ///
        /// <returns>   null if it fails, else the datablock template. </returns>
        ///-------------------------------------------------------------------------------------------------

        DatablockTemplate * 
        GetDatablockTemplate(
            char * lpszType
            )
        {
            DatablockTemplate * pResult = nullptr;
            std::set<DatablockTemplate*>::iterator templateIter;
            for(templateIter=g_vTemplates.begin(); templateIter!=g_vTemplates.end(); ++templateIter) {
                DatablockTemplate * pTemplate = *templateIter;
                if (!strcmp(pTemplate->GetTemplateName(), lpszType))
                {
                    // Sanity check that there are not multiple templates matching the same name.
                    assert(nullptr == pResult);
                    pResult = pTemplate;
                }
            }
            return pResult;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate datablock. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTemplate">            [in] If non-null, the datablock template. </param>
        /// <param name="pInitData">            [in] If non-null, a buffer with initial data.</param>
        /// <param name="cbInitData">           [in] the number of bytes in the initial data. </param>
        /// <param name="pDestChannel">         [in] If non-null, destination channel. </param>
        /// <param name="flags">                (optional) buffer permissions flags. </param>
        /// <param name="uiBlockControlCode">   (optional) The block control code. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Datablock * 
        AllocateDatablock(
            __in DatablockTemplate *    pTemplate,
            __in void *                 pInitData,
            __in UINT                   cbInitData,
            __in Channel *              pDestChannel,
            __in BUFFERACCESSFLAGS      flags,
            __in CONTROLSIGNAL          uiBlockControlCode
            )
        {            
            if(pInitData != NULL && cbInitData == 0) {
                assert(cbInitData != 0 && "non-null init buffer requires non-zero byte count!");
                printf("AllocateDatablock::non-null init buffer requires non-zero byte count!\n");
                return NULL;
            }
            if(pInitData == NULL && cbInitData != 0) {
                assert(cbInitData == 0 && "null init buffer requires zero byte count!");
                printf("AllocateDatablock::null init buffer requires zero byte count!\n");
                return NULL;
            }

            UNREFERENCED_PARAMETER(cbInitData);

            // if the flags specify a raw buffer, the template needs to 
            // as well. The converse is not true, because we allow default
            // flag parameter of 0. A null template pointer is the same 
            // as specifying raw. 
            
            BOOL bInferFlags = !flags; // must infer them!
            assert(!((flags & PT_ACCESS_BYTE_ADDRESSABLE) && (pTemplate && !pTemplate->IsByteAddressable())));
            if(bInferFlags && pTemplate->IsByteAddressable()) flags |= PT_ACCESS_BYTE_ADDRESSABLE;

            // if we know something about the channel that is the datablock target, then we can make some
            // intelligent decisions about choosing where to copy any init data. In particular, if the
            // channel is connected to an input port or a constant port, we can copy our data directly into
            // a view that is materialized for the device. 
            
            Port * pPort = NULL;
            Task * pTask = NULL;
            Datablock * pBlock = NULL;
            HOSTMEMORYEXTENT extent(pInitData, cbInitData, FALSE);
            if(pDestChannel != NULL) {
                if(pDestChannel->GetType() == CT_GRAPH_INPUT) {
                    pPort = pDestChannel->GetBoundPort(CE_DST);
                    pTask = pPort->GetTask();
                    if(pTask) {

                        // if there is only one target accelerator in the system for this ptask's accelerator class,
                        // then we might as well create device-side buffers eagerly. Conversely, if we actually have an
                        // option to do late binding of ptasks to accelerators, then we cannot allocate device-side
                        // buffers until dispatch time. Pass NULL as the pAccelerator parameter to new data block to
                        // indicate device buffers must be created later. 
                        
                        Accelerator * pAccelerator = NULL;
                        set<Accelerator*> vAccelerators;
                        ACCELERATOR_CLASS accClass = pTask->GetAcceleratorClass();
                        Scheduler::FindEnabledCapableAccelerators(accClass, vAccelerators);
                        size_t nAccelerators = vAccelerators.size();
                        if(nAccelerators > 0) {
                            // FIXME: TODO: this isn't exactly right. 
                            // we want to be able to be able to materialize according
                            // to usage scenario, which means the constructor is too
                            // coarse an interface. For now, this is not a big issue because
                            // most users won't have multiple CUDA cards.
                            if(accClass == ACCELERATOR_CLASS_CUDA || nAccelerators == 1) {
                                pAccelerator = *(vAccelerators.begin());
                            }
                        }

                        if(pPort->GetPortType() == INPUT_PORT) {

                            // if we have an input port, we need to infer flags
                            // whether there is an accelerator selected or not!
                            
                            Port * pConsumer = ((InputPort*)pPort)->GetInOutConsumer();
                            if(bInferFlags) {
                                if(pConsumer != NULL) {

                                    // if this input port is part of an in/out producer/consumer pair
                                    // then we need to make sure the access flags are correctly inferred. 
                                    flags |= PT_ACCESS_ACCELERATOR_READ;  // this variable is bound to 
                                    flags |= PT_ACCESS_ACCELERATOR_WRITE; // an in and out, so we need RW on accelerator.

                                } else {

                                    flags = (PT_ACCESS_HOST_WRITE | PT_ACCESS_ACCELERATOR_READ);

                                }
                            }
                        }

                        // FIXME: TODO: 
                        // -------------
                        // There are many opportunities for eager materialization being neglected here. Implement them:
                        // 1. Correct handling of eager materialization for blocks on ports with dependent accelerator.
                        // 2. Simultaneous materialization in multiple memory spaces. Likely to be profitable for 
                        //    blocks that are consumed on input ports (and are therefore read-only) on multiple tasks
                        //    (which can consequently be scheduled simultaneously).

                        if(pAccelerator) {

                            // If the domain of accelerators on which we may need a view of this new
                            // block is sufficiently restricted that we can infer before the task is dispatch-ready 
                            // what accelerators will definitely need views of the block, then let's materialize the view
                            // eagerly. Currently, we *only* make that inference in situations where:
                            // 
                            // 1. There is a single accelerator in the execution environment of the target class
                            // 2. A single downstream task is going to consume the block
                            // 3. The port on that task, to which the block will be bound has no dependent accelerator
                            //    binding. More directly, this means that the task is not going to defer work to another
                            //    accelerator, as host-tasks often do with CUDA code. 
                            //    
                            // If, in fact this is a context in which we can profitably do an eager materialization,
                            // we need to get the proper AsyncContext object from the target task. Otherwise, we assume
                            // a NULL async context, which will force any transfer to be synchronous. 

                            AsyncContext * pAsyncContext = NULL;
                            ACCELERATOR_CLASS accTaskClass = pAccelerator->GetClass();
                            ACCELERATOR_CLASS accPortClass = accTaskClass;
                            if(pPort->HasDependentAcceleratorBinding()) {
                                accPortClass = pPort->GetDependentAcceleratorClass(0);
                            }

                            if(accPortClass == accTaskClass && pTask != NULL) {

                                // if this task is just going to migrate data from a dispatch accelerator to the dependent
                                // accelerator for this port, don't attempt an async transfer. We could presumably handle the
                                // case where there is only one accelerator of the dependent type, but in general we don't know
                                // what accelerator will be bound to this port until dispatch time, so hold off on that
                                // optimization for now. The predicate for this control block just ensures that we are only
                                // materializing views eagerly for tasks that have no dependent accelerators, in execution
                                // environments where there is only a single accelerator of that class. 
                                pAsyncContext = pTask->GetOperationAsyncContext(pAccelerator, ASYNCCTXT_XFERHTOD);
                            }                            

                            switch(pPort->GetPortType()) {

                            case INPUT_PORT: 

                                pBlock = Datablock::CreateDatablock(pAsyncContext, 
                                                                    pTemplate,
                                                                    &extent,
                                                                    flags,
                                                                    uiBlockControlCode);
                                pBlock->AddRef();   // assume the caller wants a reference to this thing!
                                return pBlock;

                            case STICKY_PORT:

                                if(!flags) flags = (PT_ACCESS_HOST_WRITE | PT_ACCESS_ACCELERATOR_READ | PT_ACCESS_IMMUTABLE);
                                pBlock = GlobalPoolManager::RequestBlock(pTemplate, 0, 0, 0);
                                if(pBlock != NULL) {
                                    pBlock->AddRef();
                                    pBlock->Lock();
                                    void * pData = pBlock->GetDataPointer(TRUE, FALSE);
                                    memcpy(pData, extent.lpvAddress, extent.uiSizeBytes);
                                    pBlock->SetControlSignal(uiBlockControlCode);
                                    pBlock->SetAccessFlags(flags);
                                    pBlock->Unlock();
                                    return pBlock;
                                }
                                pBlock =  Datablock::CreateDatablock(pAsyncContext, 
                                                                     pTemplate,
                                                                     &extent,
                                                                     flags,
                                                                     uiBlockControlCode);
                                pBlock->AddRef();   // assume the caller wants a reference to this thing!
                                return pBlock;

                            default:
                                // we can't allocate device-side buffers yet for this port.
                                // break here and allocate in host memory space for now. 
                                break;
                            }
                        }
                    }
                }

            } 
            
            if(pBlock == NULL && pTemplate != NULL) {
                pBlock = GlobalPoolManager::RequestBlock(pTemplate, 0, 0, 0);
                if(pBlock != NULL) {
                    if(pPort != NULL && !flags && pPort->GetPortType() == STICKY_PORT) 
                        flags = (PT_ACCESS_HOST_WRITE | PT_ACCESS_ACCELERATOR_READ | PT_ACCESS_IMMUTABLE);
                    pBlock->AddRef();
                    pBlock->Lock();
                    void * pData = pBlock->GetDataPointer(TRUE, FALSE);
                    memcpy(pData, extent.lpvAddress, extent.uiSizeBytes);
                    pBlock->SetControlSignal(uiBlockControlCode);
                    pBlock->SetAccessFlags(flags);
                    pBlock->Unlock();
                    return pBlock;
                }
            }

            assert(pBlock == NULL);
            pBlock = Datablock::CreateDatablock(NULL, pTemplate, &extent, flags, uiBlockControlCode);
            pBlock->AddRef();
            return pBlock;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate a new datablock, with an effort to ensure that it is possible
        ///             to use the resulting block with async APIs for whatever backend is the target. 
        ///             In particular, when there are multiple accelerators that can be bound to the
        ///             target port, the runtime has a tough decision with respect to materializing
        ///             on the device: we don't know which device will be bound to execute the target
        ///             task at the time we are allocating the datablock. If the bMaterializeAll parameter
        ///             is true, we will actually create potentially many device-side buffers: one for
        ///             every possible execution context. If it is false, *and* there is a choice about
        ///             the device target, we defer the device copy, but ensure that we allocate a
        ///             host-side copy that enables async copy later (ie, make sure the host buffer
        ///             is pinned!). 
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/19/2011. </remarks>
        ///
        /// <param name="pTemplate">            [in] If non-null, a template describing geometry of the
        ///                                     allocated block. </param>
        /// <param name="pInitData">            [in,out] If non-null, information describing the
        ///                                     initialise. </param>
        /// <param name="cbInitData">           count of bytes in pInitData if it is provided. </param>
        /// <param name="pDestChannel">         [in] If non-null, the channel this block will be pushed
        ///                                     into. </param>
        /// <param name="flags">                Buffer access flags for the block. </param>
        /// <param name="uiBlockControlCode">   a block control code (such as DBCTL_EOF). </param>
        /// <param name="bMaterializeAll">      (optional) the materialize all. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Datablock * 
        AllocateDatablockAsync(
            DatablockTemplate * pTemplate,
            void * pInitData,
            UINT cbInitData,
            Channel * pDestChannel,
            BUFFERACCESSFLAGS flags,
            CONTROLSIGNAL uiBlockControlCode,
            BOOL bMaterializeAll
            )
        {            
            if(pInitData != NULL && cbInitData == 0) {
                assert(cbInitData != 0 && "non-null init buffer requires non-zero byte count!");
                printf("AllocateDatablock::non-null init buffer requires non-zero byte count!\n");
                return NULL;
            }
            if(pInitData == NULL && cbInitData != 0) {
                assert(cbInitData == 0 && "null init buffer requires zero byte count!");
                printf("AllocateDatablock::null init buffer requires zero byte count!\n");
                return NULL;
            }

            UNREFERENCED_PARAMETER(cbInitData);

            // if the flags specify a raw buffer, the template needs to 
            // as well. The converse is not true, because we allow default
            // flag parameter of 0. A null template pointer is the same 
            // as specifying raw. 
            
            BOOL bInferFlags = !flags; // must infer them!
            assert(!((flags & PT_ACCESS_BYTE_ADDRESSABLE) && (pTemplate && !pTemplate->IsByteAddressable())));
            if(bInferFlags && pTemplate->IsByteAddressable()) flags |= PT_ACCESS_BYTE_ADDRESSABLE;

            // if we know something about the channel that is the datablock target, then we can make some
            // intelligent decisions about choosing where to copy any init data. In particular, if the
            // channel is connected to an input port or a constant port, we can copy our data directly into
            // a view that is materialized for the device. 
            
            static int nAllocs = 0;
            wchar_t szInstr[256];
            wsprintf(szInstr, L"alloc-%d", ++nAllocs);
            MARKRANGEENTER(szInstr);        

            Datablock * pBlock = NULL;
            HOSTMEMORYEXTENT extent(pInitData, cbInitData, FALSE);
            if(pDestChannel != NULL) {
                if(pDestChannel->GetType() == CT_GRAPH_INPUT) {
                    Port * pPort = pDestChannel->GetBoundPort(CE_DST);
                    Task * pTask = pPort->GetTask();
                    if(pTask) {

                        // if there is only one target accelerator in the system for this ptask's accelerator class,
                        // then we might as well create device-side buffers eagerly. Conversely, if we actually have an
                        // option to do late binding of ptasks to accelerators, then we cannot allocate device-side
                        // buffers until dispatch time. Pass NULL as the pAccelerator parameter to new data block to
                        // indicate device buffers must be created later. 
                        
                        set<Accelerator*> vAccelerators;
                        ACCELERATOR_CLASS accTargetClass = ACCELERATOR_CLASS_UNKNOWN;
                        ACCELERATOR_CLASS accTaskClass = pTask->GetAcceleratorClass();
                        ACCELERATOR_CLASS accPortClass = pPort->GetDependentAcceleratorClass(0);
                        Accelerator * pTaskMandatory = pTask->GetMandatoryAccelerator();
                        Accelerator * pPortMandatory = pPort->GetMandatoryDependentAccelerator();

                        if(pTaskMandatory != NULL ||  pPortMandatory != NULL) {

                            // if the user has specified a mandatory accelerator,
                            // there is no need to speculate about where we are going
                            // to need views. 
                            
                            assert(pTaskMandatory == NULL || 
                                   pPortMandatory == NULL || 
                                   pTaskMandatory == pPortMandatory); 
                            accTargetClass = pPortMandatory == NULL ? 
                                                pTaskMandatory->GetClass() :
                                                pPortMandatory->GetClass();
                            Accelerator * pAssignment = pPortMandatory ? pPortMandatory : pTaskMandatory;
                            vAccelerators.insert(pAssignment);
                            
                        } else {

                            // try to figure out all possible locations
                            // where we may need to materialize a view
                            
                            if(pPort->HasDependentAcceleratorBinding()) {
                                Scheduler::FindEnabledCapableAccelerators(accPortClass, vAccelerators);
                                accTargetClass = accPortClass;
                            } else {
                                Scheduler::FindEnabledCapableAccelerators(accTaskClass, vAccelerators);
                                accTargetClass = accTaskClass;
                            }
                        }

                        if(pPort->GetPortType() == INPUT_PORT) {

                            // if we have an input port, we need to infer flags
                            // whether there is an accelerator selected or not!
                            
                            Port * pConsumer = ((InputPort*)pPort)->GetInOutConsumer();
                            if(bInferFlags) {
                                if(pConsumer != NULL) {

                                    // if this input port is part of an in/out producer/consumer pair
                                    // then we need to make sure the access flags are correctly inferred. 
                                    flags |= PT_ACCESS_ACCELERATOR_READ;  // this variable is bound to 
                                    flags |= PT_ACCESS_ACCELERATOR_WRITE; // an in and out, so we need RW on accelerator.

                                } else {

                                    flags = (PT_ACCESS_HOST_WRITE | PT_ACCESS_ACCELERATOR_READ);

                                }
                            }
                        }

                        // FIXME: TODO: 
                        // -------------
                        // There are many opportunities for eager materialization being neglected here. Implement them:
                        // 1. Correct handling of eager materialization for blocks on ports with dependent accelerator.
                        // 2. Simultaneous materialization in multiple memory spaces. Likely to be profitable for 
                        //    blocks that are consumed on input ports (and are therefore read-only) on multiple tasks
                        //    (which can consequently be scheduled simultaneously).

                        if(vAccelerators.size()) {

                            GraphInputChannel * pIChannel = reinterpret_cast<GraphInputChannel*>(pDestChannel);
                            if(pIChannel->HasBlockPool()) {

                                // we would prefer to get a block without having to allocate one, and there are a few
                                // conditions that must be met if we going to take one out of a block pool on this channel:
                                // 1. The dimensions must match the user request dimensions. 
                                // 2. There must be a block in the pool for which there are no outstanding
                                //    conflicting operations.
                                // The fact that it is possible for a pool to contain blocks that do not meet criterion 2 is
                                // somewhat counter-intuitive, but this can arise because PTask runs so far ahead of the actual
                                // GPU command queue. Blocks wind up back in the block pool because we have satisfied the
                                // ordering constraints for all operations on its underlying buffers and no user code is
                                // referencing the block anymore, but the actual command queue entries we have so carefully
                                // ordered have not yet actually completed. Or, more precisely, we do not know whether they
                                // have completed or not. Generally PTask ensures correctness under these conditions by forcing
                                // any request for a buffer in such blocks to wait until the dependences resolve. We'd strongly
                                // prefer *not* to do that here. We could check the pool for condition two, but the subsequent
                                // choice about what to do in response features only unattractive options: allocating a buffer
                                // instead of synchronizing on an existing one will also sync the GPU driver due to the GPU
                                // side malloc. 
                                
                                if(cbInitData == pIChannel->GetTemplate()->GetDatablockByteCount()) {
                                    pBlock = pIChannel->GetPooledBlock();
                                    if(pBlock != NULL && pInitData != NULL) {

                                        BOOL bLocks = FALSE;
                                        if(vAccelerators.size()) {
                                            Accelerator * pCand = *vAccelerators.begin();
                                            if(!pCand->IsHost() && 
                                               PTask::Runtime::GetDefaultViewMaterializationPolicy() == VIEWMATERIALIZATIONPOLICY_EAGER) {
                                                std::set<Accelerator*>::iterator si;
                                                for(si=vAccelerators.begin(); si!=vAccelerators.end(); si++) {
                                                    bLocks = TRUE;
                                                    (*si)->Lock();
                                                }
                                            }
                                        }

                                        pBlock->Lock();
                                        // we need a pointer to a writeable host buffer,
                                        // but expect to overwrite its contents, so no view update is appropriate.
                                        // We need outstanding readers on this buffer to be done so we can
                                        // safely write it (implying we are a synchronous consumer), we are
                                        // not calling from dispatch context, and we do want the coherence state
                                        // correctly updated to reflect our exclusive access.
                                        BOOL bUpdateView = FALSE;
                                        BOOL bExclusive = TRUE;
                                        void * pData = pBlock->GetDataPointer(bExclusive, bUpdateView);
                                        memset(pData, 0, pIChannel->GetTemplate()->GetDatablockByteCount());
                                        memcpy(pData, pInitData, cbInitData);
                                        pBlock->SetControlSignal(uiBlockControlCode);

                                        if(bLocks) {
                                            std::set<Accelerator*>::iterator si;
                                            for(si=vAccelerators.begin(); si!=vAccelerators.end(); si++) {
                                                pBlock->UpdateView((*si),
                                                                   (*si)->GetAsyncContext(ASYNCCONTEXTTYPE::ASYNCCTXT_XFERHTOD),
                                                                   TRUE,
                                                                   BUFFER_COHERENCE_STATE::BSTATE_SHARED,
                                                                   0,
                                                                   0);
                                            }
                                        }
                                        pBlock->Unlock();

                                        if(bLocks) {
                                            std::set<Accelerator*>::iterator si;
                                            for(si=vAccelerators.begin(); si!=vAccelerators.end(); si++) 
                                                (*si)->Unlock();
                                        }

                                    }
                                    if(pBlock) {
                                        MARKRANGEEXIT();
                                        return pBlock;
                                    }
                                } else {
                                    pBlock = NULL;
                                }
                            }

                            if(!pBlock) {

                                // If the domain of accelerators on which we may need a view of this new
                                // block is sufficiently restricted that we can infer before the task is dispatch-ready 
                                // what accelerators will definitely need views of the block, then let's materialize the view
                                // eagerly. Currently, we *only* make that inference in situations where:
                                // 
                                // 1. There is a single accelerator in the execution environment of the target class
                                // 2. A single downstream task is going to consume the block
                                // 3. The port on that task, to which the block will be bound has no dependent accelerator
                                //    binding. More directly, this means that the task is not going to defer work to another
                                //    accelerator, as host-tasks often do with CUDA code. 
                                //    
                                // If, in fact this is a context in which we can profitably do an eager materialization,
                                // we need to get the proper AsyncContext object from the target task. Otherwise, we assume
                                // a NULL async context, which will force any transfer to be synchronous. 

                                std::set<Accelerator*>::iterator ai;
                                std::set<AsyncContext*> vAsyncContexts;
                                for(ai=vAccelerators.begin(); ai!=vAccelerators.end(); ai++) {
                                    vAsyncContexts.insert(pTask->GetOperationAsyncContext(*ai, ASYNCCTXT_XFERHTOD));
                                }                            

                                switch(pPort->GetPortType()) {

                                case INPUT_PORT: 

                                    //pBlock = Datablock::CreateDatablock((*(vAsyncContexts.begin())), 
                                    //                                    pTemplate,
                                    //                                    &extent,
                                    //                                    flags,
                                    //                                    uiBlockControlCode);

                                    pBlock = Datablock::CreateDatablockAsync(vAsyncContexts, 
                                                                             pTemplate,
                                                                             &extent,
                                                                             flags,
                                                                             uiBlockControlCode,
                                                                             bMaterializeAll,
                                                                             bMaterializeAll,
                                                                             TRUE);
                                    pBlock->AddRef();   // assume the caller wants a reference to this thing!
                                    MARKRANGEEXIT();
                                    return pBlock;

                                case STICKY_PORT:

                                    if(!flags) flags = (PT_ACCESS_HOST_WRITE | PT_ACCESS_ACCELERATOR_READ | PT_ACCESS_IMMUTABLE);
                                    pBlock =  Datablock::CreateDatablock(NULL, 
                                                                         pTemplate,
                                                                         &extent,
                                                                         flags,
                                                                         uiBlockControlCode);
                                    pBlock->AddRef();   // assume the caller wants a reference to this thing!
                                    MARKRANGEEXIT();
                                    return pBlock;

                                default:
                                    // we can't allocate device-side buffers yet for this port.
                                    // break here and allocate in host memory space for now. 
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            pBlock = Datablock::CreateDatablock(NULL, pTemplate, &extent, flags, uiBlockControlCode);
            pBlock->AddRef();
            MARKRANGEEXIT();
            return pBlock;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate datablock. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pTemplate">            [in,out] If non-null, the datablock template. </param>
        /// <param name="uiDataSize">           Size of the data. </param>
        /// <param name="uiMetaSize">           Size of the meta. </param>
        /// <param name="uiTemplateSize">       Size of the template. </param>
        /// <param name="uiBlockControlCode">   The block control code. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Datablock * 
        AllocateDatablock(
            __in DatablockTemplate * pTemplate,
            __in UINT                uiDataSize,
            __in UINT                uiMetaSize,
            __in UINT                uiTemplateSize,
            __in CONTROLSIGNAL       uiBlockControlCode
            )
        {            
			Datablock * pBlock = NULL;
            pBlock = GlobalPoolManager::RequestBlock(pTemplate, 
                                                     uiDataSize,
                                                     uiMetaSize,
                                                     uiTemplateSize);

			if(pBlock != NULL) {
				if(uiBlockControlCode) {
					pBlock->Lock();
					pBlock->SetControlSignal(uiBlockControlCode);
					pBlock->Unlock();
				}
			}
			if(!pBlock) {
				pBlock = Datablock::CreateDatablock(pTemplate, 
                                                    uiDataSize, 
                                                    uiMetaSize, 
                                                    uiTemplateSize, 
													PT_ACCESS_DEFAULT,
                                                    uiBlockControlCode);
			}
            pBlock->AddRef();
            return pBlock;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Allocate datablock. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="uiBlockControlCode">   The block control code. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Datablock * 
        AllocateDatablock(
            __in CONTROLSIGNAL uiBlockControlCode
            )
        {            
            Datablock * pBlock = Datablock::CreateControlBlock(uiBlockControlCode);
            pBlock->AddRef();
            return pBlock;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Free datablock. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        FreeDatablock(
            Datablock * pBlock
            )
        {
            if(pBlock != NULL)
                pBlock->Release();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Handle runtime error. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="lpszError">    [in,out] If non-null, the error. </param>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        HandleError(
            const char * szMessage,
            ...
            ) 
        {
            va_list args;     
            va_start(args,szMessage);             
            vfprintf(stdout, szMessage, args);     
            va_end(args); 
            fflush(stdout);
            if(g_bExitOnRuntimeFailure) {
                exit(1);
                // return FALSE;
            } 
            return TRUE; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Handle runtime error. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="strError"> The error. </param>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        HandleError(
            std::string strError
            ) {
            ErrorMessage(strError.c_str());
            assert(false);
            if(g_bExitOnRuntimeFailure) {
                exit(1);
            }
            return TRUE;
        }

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Check a (supposed) invariant, and complain if the invariant does not hold. </summary>
		///
		/// <remarks>	Crossbac, 10/2/2012. </remarks>
		///
		/// <param name="bCondition">   	condition to check. </param>
		/// <param name="lpszErrorText">	If non-null, the error text to emit if the invariant fails. </param>
		///-------------------------------------------------------------------------------------------------

		BOOL
		CheckInvariantCondition(
			__in BOOL bCondition,
			__in char * lpszErrorText
			)
		{
			if(bCondition) return TRUE;
			if(lpszErrorText) Warning(lpszErrorText);
			assert(FALSE && lpszErrorText);
			return FALSE;
		}

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Check a (supposed) invariant, and complain if the invariant does not hold.  </summary>
		///
		/// <remarks>	Crossbac, 10/2/2012. </remarks>
		///
		/// <param name="bCondition">  	true to condition. </param>
		/// <param name="strErrorText">	The error text. </param>
		///-------------------------------------------------------------------------------------------------

		BOOL
		CheckInvariantCondition(
			BOOL bCondition,
			std::string strErrorText
			)
		{
			if(bCondition) return TRUE;
			Warning(strErrorText);
			assert(FALSE && strErrorText.c_str());
			return FALSE;
		}


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a minimum direct x coordinate feature level. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name=""> The. </param>
        ///
        /// <returns>   The minimum direct x coordinate feature level. </returns>
        ///-------------------------------------------------------------------------------------------------

        int GetMinimumDirectXFeatureLevel(
            VOID
            ) 
        {
            return g_nMinimumDirectXFeatureLevel;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a use reference drivers. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name=""> The. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        GetUseReferenceDrivers(
            VOID
            )
        {
            return g_bEnableReferenceDrivers;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a minimum direct x coordinate feature level. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="nLevel">   The level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetMinimumDirectXFeatureLevel(
            int nLevel
            )
        {
            g_nMinimumDirectXFeatureLevel = nLevel;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a use reference drivers. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="bUseReferenceDrivers"> true to use reference drivers. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetUseReferenceDrivers(
            BOOL bUseReferenceDrivers
            )
        {
            g_bEnableReferenceDrivers = bUseReferenceDrivers;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a logging level. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name=""> The. </param>
        ///
        /// <returns>   The logging level. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT GetLoggingLevel(
            VOID
            )
        {
            return g_nLoggingLevel;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a logging level. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="nLevel">   The level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetLoggingLevel(
            UINT nLevel
            )
        {
            g_nLoggingLevel = nLevel;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Error message. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        ErrorMessage(
            const char * szMessage,
            UINT nLogLevel
            )
        {
            UNREFERENCED_PARAMETER(nLogLevel);
            std::cerr << szMessage << std::endl;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Warnings. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        Warning(
            const char * szMessage,
            UINT nLogLevel
            )
        {
            if(nLogLevel >= g_nLoggingLevel || IsVerbose()) {
                std::cerr << szMessage << std::endl;
            }
        }       

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Informs. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        Inform(
            const std::string &szMessage
            )
        {
            if(IsVerbose()) {
                std::cerr << szMessage << std::endl;
            }
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Informs. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        Inform(
            const char * szMessage,
            ...
            )
        {
            if(IsVerbose()) {
                va_list args;     
                va_start(args,szMessage);             
                vfprintf(stdout, szMessage, args);     
                va_end(args); 
                fflush(stdout);
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Informs. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        MandatoryInform(
            const char * szMessage,
            ...
            )
        {
            va_list args;     
            va_start(args,szMessage);             
            vfprintf(stdout, szMessage, args);     
            va_end(args); 
            fflush(stdout);
            return TRUE;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Traces. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        Trace(
            const char * szMessage, 
            UINT nLogLevel
            )
        {
            if(nLogLevel >= g_nLoggingLevel || IsVerbose()) {
                std::cout << szMessage << std::endl;
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Error message. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        ErrorMessage(
            const WCHAR * szMessage, 
            UINT nLogLevel
            )
        {
            if(nLogLevel >= g_nLoggingLevel || IsVerbose()) {
                std::cerr << szMessage << std::endl;
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Warnings. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        Warning(
            const WCHAR * szMessage, 
            UINT nLogLevel
            )
        {
            if(nLogLevel >= g_nLoggingLevel || IsVerbose()) {
                std::cerr << szMessage << std::endl;
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Informs. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        Inform(
            const WCHAR * szMessage, 
            UINT nLogLevel
            )
        {
            if(IsVerbose() && nLogLevel >= g_nLoggingLevel) {
                std::cout << szMessage << std::endl;
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Informs. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        MandatoryInform(
            const WCHAR * szMessage,
            ...
            )
        {
            assert(false && "var args not implemented!");
            std::cout << szMessage << std::endl;
            return TRUE;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Traces. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        Trace(
            const WCHAR * szMessage, 
            UINT nLogLevel
            )
        {
            if(nLogLevel >= g_nLoggingLevel || IsVerbose()) {
                std::cout << szMessage << std::endl;
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Logs. </summary>
        ///
        /// <remarks>	Rossbach, 2/15/2012. </remarks>
        ///
        /// <param name="fmt">	[in,out] If non-null, describes the format to use. </param>
        ///
        /// <returns>	. </returns>
        ///-------------------------------------------------------------------------------------------------

        VOID 
        Trace(
            char * szSubsystemName,
            char * fmt, 
            ...
            )
        {
            EnterCriticalSection(&Runtime::g_csTraceList);
            if(g_vActiveTraceSystems.find(std::string(szSubsystemName)) != g_vActiveTraceSystems.end()) {
                fprintf(stdout, "%s\t", szSubsystemName);
                va_list args;     
                va_start(args,fmt);             
                vfprintf(stdout, fmt,args);     
                va_end(args); 
                fflush(stdout);
            }
            LeaveCriticalSection(&Runtime::g_csTraceList);
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Turn tracing on for a given subsystem. </summary>
        ///
        /// <remarks>   crossbac, 5/21/2012. </remarks>
        ///
        /// <param name="lpszSubsystemName">    [in,out] If non-null, name of the subsystem. </param>
        /// <param name="bTrace">               (optional) the trace. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        TraceSubsystem(
            __in char * lpszSubsystemName, 
            __in BOOL bTrace
            )
        {
            EnterCriticalSection(&g_csTraceList);
            if(bTrace) {
                g_vActiveTraceSystems.insert(std::string(lpszSubsystemName));
            } else {
                std::set<std::string>::iterator si; 
                si = g_vActiveTraceSystems.find(std::string(lpszSubsystemName));
                if(si!=g_vActiveTraceSystems.end()) {
                    g_vActiveTraceSystems.erase(si);
                }
            }
            LeaveCriticalSection(&g_csTraceList);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Error message. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        ErrorMessage(
            const std::string &szMessage, 
            UINT nLogLevel
            )
        {
            if(nLogLevel >= g_nLoggingLevel || IsVerbose()) {
                std::cerr << szMessage << std::endl;
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Warnings. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        Warning(
            const std::string &szMessage, 
            UINT nLogLevel
            )
        {
            if(nLogLevel >= g_nLoggingLevel || IsVerbose()) {
                std::cerr << szMessage << std::endl;
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Informs. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        Inform(
            const std::string &szMessage, 
            UINT nLogLevel
            )
        {
            if(IsVerbose() && nLogLevel >= g_nLoggingLevel) {
                std::cout << szMessage << std::endl;
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Informs. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        MandatoryInform(
            const std::string &szMessage
            )
        {
            std::cout << szMessage << std::endl;
            return TRUE;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Traces. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="szMessage">    The message. </param>
        /// <param name="nLogLevel">    The log level. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        Trace(
            const std::string &szMessage, 
            UINT nLogLevel
            )
        {
            if(nLogLevel >= g_nLoggingLevel || IsVerbose()) {
                std::cout << szMessage << std::endl;
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if '' is verbose. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name=""> The. </param>
        ///
        /// <returns>   true if verbose, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL 
        IsVerbose(
            VOID
            )
        {
            return g_bVerbose;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a verbose. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="b">    true to b. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        SetVerbose(
            BOOL b
            )
        {
            g_bVerbose = b; 
#ifdef DEBUG
            BOOL bDebug = TRUE;
#else
            BOOL bDebug = FALSE;
#endif           
            if(g_bVerbose || bDebug) {
                PTask::Runtime::MandatoryInform("%s::%s(%s): PTask verbose mode %s!\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                (g_bVerbose?"TRUE":"FALSE"),
                                                (g_bVerbose?"enabled":"disabled"));
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the physical accelerator count. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name=""> The. </param>
        ///
        /// <returns>   The physical accelerator count. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT
        GetPhysicalAcceleratorCount(
            VOID
            )
        {
            return g_nPhysicalAccelerators;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a physical accelerator count. </summary>
        ///
        /// <remarks>   Crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="nAcceleratorCount">    Number of accelerators. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        SetPhysicalAcceleratorCount(
            UINT nAcceleratorCount
            )
        {
            g_nPhysicalAccelerators = nAcceleratorCount;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Let the programmer control what PTask does when it encounters an internal failure
        ///             that it is unsure of how to handle. In debug mode, the default behavior is to
        ///             exit, to ensure the programmer cannot overlook the problem.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 1/9/2012. </remarks>
        ///
        /// <param name="bExit">    true to exit. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetExitOnRuntimeFailure(
            BOOL bExit
            )
        {
            g_bExitOnRuntimeFailure = bExit;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets exit on runtime failure. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2014. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL
        GetExitOnRuntimeFailure() 
        {
            return g_bExitOnRuntimeFailure; 
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a dump type. </summary>
        ///
        /// <remarks>   Crossbac, 1/9/2012. </remarks>
        ///
        /// <param name="typ">  The typ. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetDumpType(
            DEBUGDUMPTYPE typ
            ) 
        { 
            g_nDefaultDumpType = typ; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a dump stride. </summary>
        ///
        /// <remarks>   Crossbac, 1/9/2012. </remarks>
        ///
        /// <param name="n">    The threshold value to set. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetDumpStride(
            int n
            ) 
        { 
            g_nDefaultDumpStride = n; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets a dump length. </summary>
        ///
        /// <remarks>   Crossbac, 1/9/2012. </remarks>
        ///
        /// <param name="n">    The threshold value to set. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetDumpLength(
            int n
            ) 
        { 
            g_nDefaultDumpLength = n; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a dump type. </summary>
        ///
        /// <remarks>   Crossbac, 1/9/2012. </remarks>
        ///
        /// <param name=""> The. </param>
        ///
        /// <returns>   The dump type. </returns>
        ///-------------------------------------------------------------------------------------------------

        DEBUGDUMPTYPE 
        GetDumpType(
            VOID
            ) 
        { 
            return g_nDefaultDumpType; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a dump stride. </summary>
        ///
        /// <remarks>   Crossbac, 1/9/2012. </remarks>
        ///
        /// <param name=""> The. </param>
        ///
        /// <returns>   The dump stride. </returns>
        ///-------------------------------------------------------------------------------------------------

        int 
        GetDumpStride(
            VOID
            ) 
        { 
            return g_nDefaultDumpStride; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a dump length. </summary>
        ///
        /// <remarks>   Crossbac, 1/9/2012. </remarks>
        ///
        /// <param name=""> The. </param>
        ///
        /// <returns>   The dump length. </returns>
        ///-------------------------------------------------------------------------------------------------

        int 
        GetDumpLength(
            VOID
            ) 
        { 
            return g_nDefaultDumpLength; 
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the meta function for the given port. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <param name="pPort">                    [in,out] If non-null, the port. </param>
        /// <param name="eCanonicalMetaFunction">   The canonical meta function. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        SetPortMetaFunction(
            Port * pPort, 
            METAFUNCTION eCanonicalMetaFunction
            )
        {
            pPort->SetMetaFunction(eCanonicalMetaFunction);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds a task. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pGraph">           [in,out] If non-null, the graph. </param>
        /// <param name="pKernel">          [in,out] If non-null, the kernel. </param>
        /// <param name="uiInputPortCount"> Number of input ports. </param>
        /// <param name="pvInputPorts">     [in,out] If non-null, the pv input ports. </param>
        /// <param name="uiOutputPorts">    The output ports. </param>
        /// <param name="pvOutputPorts">    [in,out] If non-null, the pv output ports. </param>
        /// <param name="lpszTaskName">     [in,out] If non-null, name of the task. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Task*
        AddTask(
            Graph *             pGraph,
            CompiledKernel *	pKernel,
            UINT				uiInputPortCount,
            Port**				pvInputPorts,
            UINT				uiOutputPorts,
            Port**				pvOutputPorts,
            char *				lpszTaskName
            )
        {
            return pGraph->AddTask(pKernel,
                                   uiInputPortCount,
                                   pvInputPorts,
                                   uiOutputPorts,
                                   pvOutputPorts,
                                   lpszTaskName);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Bind a derived port. </summary>
        ///
        /// <remarks>   Crossbac, 12/22/2011. </remarks>
        ///
        /// <param name="pGraph">           [in,out] If non-null, the graph. </param>
        /// <param name="pDescribedPort">   [in,out] If non-null, the described port. </param>
        /// <param name="pDescriberPort">   [in,out] If non-null, the describer port. </param>
        /// <param name="func">             (optional) the func. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        BindDerivedPort(
            Graph * pGraph,
            Port * pDescribedPort, 
            Port* pDescriberPort, 
            DESCRIPTORFUNC func
            )
        {
            pGraph->BindDescriptorPort(pDescribedPort, pDescriberPort, func);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Put the given graph in the running state. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        RunGraph(
            Graph * pGraph
            )
        {
            pGraph->Run();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pushes a datablock into the given channel. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
        /// <param name="pBlock">   [in,out] If non-null, the block. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        Push(
            Channel * pChannel,
            Datablock * pBlock
            )
        {
            pChannel->Push(pBlock);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Pulls the next datablock from the given channel. </summary>
        ///
        /// <remarks>   Crossbac, 1/11/2012. </remarks>
        ///
        /// <param name="pChannel">     [in,out] If non-null, the channel. </param>
        /// <param name="dwTimeout">    (optional) the timeout. </param>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        Datablock * 
        Pull(
            Channel * pChannel,
            DWORD dwTimeout
            )
        {
            return pChannel->Pull(dwTimeout);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check to see if the graph is well-formed. This is not an exhaustive check, but a
        ///             collection of obvious sanity checks. If the bFailOnWarning flag is set, then the
        ///             runtime will exit the process if it finds anything wrong with the graph.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="pGraph">           [in] non-null, the graph. </param>
        /// <param name="bVerbose">         (optional) verbose output?. </param>
        /// <param name="bFailOnWarning">   (optional) fail on warning flag: if set, exit the process
        ///                                 when malformed graph elements are found. </param>
        ///
        /// <returns>   PTRESULT--use PTSUCCESS/PTFAILED macros: PTASK_OK:   the graph is well-formed
        ///             PTASK_ERR_GRAPH_MALFORMED: the graph is malformed in a way that cannot be
        ///                                        tolerated by the runtime. Or the the issue may be
        ///                                        tolerable but the user requested fail on warning.
        ///             PTASK_WARNING_GRAPH_MALFORMED: the graph is malformed in a way that can be
        ///                                        tolerated by the runtime.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT 
        CheckGraphSemantics(
            Graph * pGraph,
            BOOL bVerbose,
            BOOL bFailOnWarning)
        {
            return pGraph->CheckGraphSemantics(bVerbose, bFailOnWarning);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Get the GPU Lynx metric specified for instrumenting GPU kernels.
		/// </summary>
        ///
        /// <remarks>   t-nailaf, 5/21/13. </remarks>
        ///
        /// <param name="metric">    the specified instrumentation metric. </param>
        ///-------------------------------------------------------------------------------------------------

        INSTRUMENTATIONMETRIC 
        GetInstrumentationMetric(
            VOID
            )
        {
            return Runtime::g_InstrumentationMetric;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Control the PTask runtime mode for instrumenting GPU kernels. The 
		///				instrumentation metric specifies one of the default metrics made available by GPU Lynx 
		///				for instrumenting kernels (such as activityFactor, memoryEfficiency, etc).
        ///             </summary>
        ///
        /// <remarks>   t-nailaf, 5/21/13. </remarks>
        ///
        /// <param name="metric"> specifies the metric used to instrument kernels.    </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetInstrumentationMetric(
            INSTRUMENTATIONMETRIC metric
            )
        {
            Runtime::g_InstrumentationMetric = metric;
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Instrument the PTX module, given the PTX file location and the 
        ///             instrumented PTX file location.
        /// </summary>
        ///
        /// <remarks>   t-nailaf, 5/21/13. </remarks>
        ///
        /// <param name="ptxFile"> specifies the path to the original PTX file.    </param>
        /// <param name="instrumentedPTXFile"> specifies the path to the instrumented PTX file.    </param>
        ///-------------------------------------------------------------------------------------------------
        void
        Instrument(
            const char * ptxFile,
            const char *instrumentedPTXFile
            )
        {
#ifndef PTASK_LYNX_INSTRUMENTATION
            UNREFERENCED_PARAMETER(ptxFile);
            UNREFERENCED_PARAMETER(instrumentedPTXFile);
            PTask::Runtime::MandatoryInform("PTask::Runtime::Instrument() call ignored...\n");
            PTask::Runtime::MandatoryInform("Lynx instrumentation compiled out of this PTask build!\n");
#else
            if(MultiGPUEnvironment())
            {
                Runtime::HandleError("Instrumentation cannot be enabled in a multi-GPU environment! Please choose a single GPU for instrumentation.\n");
            }

            std::vector<std::string> kernelsToInstrument;
            std::string instrumentedPTX = lynx::instrument(std::string(ptxFile), (unsigned int)Runtime::g_InstrumentationMetric, &kernelsToInstrument);
            
            if(!instrumentedPTX.empty())
            {
                std::ofstream outfile(instrumentedPTXFile, std::ios::trunc);
                outfile << instrumentedPTX;
                outfile.close();
            }
#endif
        }
    
        void
        CleanupInstrumentation(void)
        {
#ifdef PTASK_LYNX_INSTRUMENTATION
            lynx::cleanup();
#endif
        }

        BOOL
        Instrumented(void)
        {
#ifndef PTASK_LYNX_INSTRUMENTATION
            return FALSE;
#else
            return g_bInstrumented;
#endif
        }

        void 
        SetInstrumented(
            BOOL bInstrumented
            )
        {
#ifndef PTASK_LYNX_INSTRUMENTATION
            PTask::Runtime::MandatoryInform("PTask::Runtime::SetInstrumented(%s) call ignored...\n", 
                                            (bInstrumented?"TRUE":"FALSE"));
            PTask::Runtime::MandatoryInform("Lynx instrumentation compiled out of this PTask build!\n");
#else
            g_bInstrumented = bInstrumented;
#endif
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a handle for the runtime terminate event--handy for any 
        ///             internal services or user code that may need to know about 
        ///             runtime shutdown calls that occur on other threads of control. 
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   The runtime terminate event. </returns>
        ///-------------------------------------------------------------------------------------------------

        HANDLE 
        GetRuntimeTerminateEvent(
            VOID
            )
        {
            if(!g_bPTaskInitialized && g_hRuntimeTerminateEvent == INVALID_HANDLE_VALUE) {
                PTask::Runtime::Warning("XXX: attempt to get runtime terminate event from uninitialized runtime instance!\n");
                return INVALID_HANDLE_VALUE;
            }
            return g_hRuntimeTerminateEvent;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a handle for the runtime initialization complete event--handy for any 
        ///             internal services or user code that may need to know whether
        ///             runtime initialization is complete.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <returns>   The runtime terminate event. </returns>
        ///-------------------------------------------------------------------------------------------------

        HANDLE 
        GetRuntimeInitializedEvent(
            VOID
            )
        {
            if(!PTVALIDSYNCHANDLE(g_hRuntimeInitialized)) {
                PTask::Runtime::HandleError("XXX: attempt to get runtime initialized event from uninitialized runtime instance!\n");
                return INVALID_HANDLE_VALUE;
            }
            return g_hRuntimeInitialized;
        }
               

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a graph. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <param name="lpszGraphName">    (optional) [in,out] If non-null, name of the graph. </param>
        ///
        /// <returns>   The new graph. </returns>
        ///-------------------------------------------------------------------------------------------------

        PTask::Graph * 
        CreateGraph(
            VOID
            )
        {
            return PTask::Graph::CreateGraph(NULL);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a graph. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <param name="lpszGraphName">    (optional) [in,out] If non-null, name of the graph. </param>
        ///
        /// <returns>   The new graph. </returns>
        ///-------------------------------------------------------------------------------------------------

        PTask::Graph * 
        CreateGraph(
            char * lpszGraphName
            )
        {
            return PTask::Graph::CreateGraph(lpszGraphName);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Force GC. </summary>
        ///
        /// <remarks>   crossbac, 6/17/2013. </remarks>
        ///
        /// <param name="bCollectDanglingRefs"> true if the caller thinks he has no 
        ///                                     dangling references and global state
        ///                                     can be cleaned up even if there are 
        ///                                     "live" objects (refcount>0). </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        PTRESULT
        ForceGC(
            BOOL bCollectDanglingRefs
            )
        {
            GarbageCollector::ForceGC();
            if(bCollectDanglingRefs) {
                Scheduler::LockScheduler();
                if(Scheduler::GetLiveGraphCount() == 0) {
                    DeleteTemplates();
                }
                Scheduler::UnlockScheduler();
            }
            return PTASK_OK;
        }

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Collect instrumentation data. </summary>
		///
		/// <remarks>	crossbac, 8/12/2013. </remarks>
		///
		/// <param name="szEventName">	   	[in,out] If non-null, name of the event. </param>
		///
		/// <returns>	. </returns>
		///-------------------------------------------------------------------------------------------------

		double 
		CollectInstrumentationData(
			char * szEventName 
			)
		{
#ifndef ADHOC_STATS
            UNREFERENCED_PARAMETER(szEventName);
			return 0.0;
#else
			if(!Runtime::g_bAdhocInstrumentationEnabled) 
				return 0.0;
			std::string strEventName(szEventName);
			return Instrumenter::CollectDataPoint(strEventName);
#endif
		}

		///-------------------------------------------------------------------------------------------------
		/// <summary>	Collect instrumentation data. </summary>
		///
		/// <remarks>	crossbac, 8/12/2013. </remarks>
		///
		/// <param name="szEventName">	   	[in,out] If non-null, name of the event. </param>
		///
		/// <returns>	. </returns>
		///-------------------------------------------------------------------------------------------------

		double 
		CollectInstrumentationData(
			char * szEventName,
            UINT& uiSamples,
            double& dMin,
            double& dMax
			)
		{
#ifndef ADHOC_STATS
            UNREFERENCED_PARAMETER(dMin);
            UNREFERENCED_PARAMETER(dMax);
            UNREFERENCED_PARAMETER(uiSamples);
            UNREFERENCED_PARAMETER(szEventName);
			return 0.0;
#else
			if(!Runtime::g_bAdhocInstrumentationEnabled) 
				return 0.0;
			std::string strEventName(szEventName);
			return Instrumenter::CollectDataPoint(strEventName, uiSamples, dMin, dMax);
#endif
		}

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Print runtime configuration. </summary>
        ///
        /// <remarks>   crossbac, 9/7/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        PrintRuntimeConfiguration(
            VOID
            )
        {
            PrintRuntimeConfiguration(std::cout);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Print runtime configuration. </summary>
        ///
        /// <remarks>   crossbac, 9/7/2013. </remarks>
        ///
        /// <param name="ios">  [in,out] The ios. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        PrintRuntimeConfiguration(
            std::ostream& ios
            )
        {
            ios << "\nPTask Runtime Config"                                                                       << std::endl;
            ios << "--------------------------------------------------------------------------"                   << std::endl;
            ios << "\tGraphAssignmentPolicy                   : " << GraphAssignmentPolicyString(GetGraphAssignmentPolicy()) << std::endl;
	        ios << "\tTaskThreadPoolPolicy                    : " << ThreadPoolPolicyString(GetTaskThreadPoolPolicy()) << std::endl;
            ios << "\tBlockPoolBlockResizePolicy              : " << BlockPoolResizePolicyString(GetBlockPoolBlockResizePolicy()) << std::endl;
            ios << "\tBlockResizeMemorySpacePolicy            : " << BlockResizeMemspacePolicyString(GetBlockResizeMemorySpacePolicy()) << std::endl;
            ios << "\tDefaultOutputViewMaterializationPolicy  : " << ViewPolicyString(GetDefaultOutputViewMaterializationPolicy()) << std::endl;
            ios << "\tDefaultViewMaterializationPolicy        : " << ViewPolicyString(GetDefaultViewMaterializationPolicy()) << std::endl;
            ios << "\tGCSweepThresholdPercent                 : " << GetGCSweepThresholdPercent()                 << std::endl;
            ios << "\tDEFAULT_DATA_BUFFER_SIZE                : " << DEFAULT_DATA_BUFFER_SIZE                     << std::endl;
            ios << "\tDEFAULT_META_BUFFER_SIZE                : " << DEFAULT_META_BUFFER_SIZE                     << std::endl;
            ios << "\tDEFAULT_TEMPLATE_BUFFER_SIZE            : " << DEFAULT_TEMPLATE_BUFFER_SIZE                 << std::endl;
            ios << "\tAggressiveReleaseMode                   : " << GetAggressiveReleaseMode()                   << std::endl;
            ios << "\tBlockPoolsEnabled                       : " << GetBlockPoolsEnabled()                       << std::endl;
            ios << "\tCriticalPathAllocMode                   : " << GetCriticalPathAllocMode()                   << std::endl;
            ios << "\tRCProfilingEnabled                      : " << GetRCProfilingEnabled()                      << std::endl;
            ios << "\tDBProfilingEnabled                      : " << GetDBProfilingEnabled()                      << std::endl;
            ios << "\tCTProfilingEnabled                      : " << GetCTProfilingEnabled()                      << std::endl;
            ios << "\tTPProfilingEnabled                      : " << GetTPProfilingEnabled()                      << std::endl;
            ios << "\tPBufferProfilingEnabled                 : " << GetPBufferProfilingEnabled()                 << std::endl;
            ios << "\tInvocationCountingEnabled               : " << GetInvocationCountingEnabled()               << std::endl;
            ios << "\tChannelProfilingEnabled                 : " << GetChannelProfilingEnabled()                 << std::endl;
            ios << "\tBlockPoolProfilingEnabled               : " << GetBlockPoolProfilingEnabled()               << std::endl;
            ios << "\tEnableDebugLogging                      : " << GetEnableDebugLogging()                      << std::endl;
            ios << "\tAdhocInstrumentationEnabled             : " << GetAdhocInstrumentationEnabled()             << std::endl;
            ios << "\tPBufferClearOnCreatePolicy              : " << GetPBufferClearOnCreatePolicy()              << std::endl;
            ios << "\tProfilePSDispatch                       : " << GetProfilePSDispatch()                       << std::endl;
            ios << "\tSizeDescriptorPoolSize                  : " << GetSizeDescriptorPoolSize()                  << std::endl;
            ios << "\tGlobalThreadPool                        : " << HasGlobalThreadPool()                        << std::endl;
            ios << "\tGlobalThreadPoolSize                    : " << GetGlobalThreadPoolSize()                    << std::endl;
            ios << "\tPrimeGlobalThreadPool                   : " << GetPrimeGlobalThreadPool()                   << std::endl;
            ios << "\tGlobalThreadPoolGrowable                : " << GetGlobalThreadPoolGrowable()                << std::endl;
            ios << "\tICBlockPoolSize                         : " << GetICBlockPoolSize()                         << std::endl;
            ios << "\tGCBatchSize                             : " << GetGCBatchSize()                             << std::endl;
            ios << "\tDefaultChannelCapacity                  : " << GetDefaultChannelCapacity()	              << std::endl;
            ios << "\tDefaultInitChannelBlockPoolSize         : " << GetDefaultInitChannelBlockPoolSize()         << std::endl;
            ios << "\tDefaultInputChannelBlockPoolSize        : " << GetDefaultInputChannelBlockPoolSize()        << std::endl;
            ios << "\tDefaultBlockPoolGrowIncrement           : " << GetDefaultBlockPoolGrowIncrement()           << std::endl;
            ios << "\tSchedulingMode                          : " << GetSchedulingMode()                          << std::endl;
            ios << "\tSchedulerThreadCount                    : " << GetSchedulerThreadCount()                    << std::endl;
            ios << "\tEagerMetaPortMode                       : " << GetEagerMetaPortMode()                       << std::endl;
            ios << "\tDebugMode                               : " << GetDebugMode()                               << std::endl;
            ios << "\tForceSynchronous                        : " << GetForceSynchronous()                        << std::endl;
            ios << "\tExtremeTraceMode                        : " << GetExtremeTraceMode()                        << std::endl;
            ios << "\tCoherenceProfileMode                    : " << GetCoherenceProfileMode()                    << std::endl;
            ios << "\tTaskProfileMode                         : " << GetTaskProfileMode()                         << std::endl;
            ios << "\tDebugAsynchronyMode                     : " << GetDebugAsynchronyMode()                     << std::endl;
            ios << "\tProfilePlatformBuffers                  : " << GetProfilePlatformBuffers()                  << std::endl;
            ios << "\tPageLockingEnabled                      : " << GetPageLockingEnabled()	                  << std::endl;
            ios << "\tAggressivePageLocking                   : " << GetAggressivePageLocking()                   << std::endl;
            ios << "\tDispatchLoggingEnabled                  : " << GetDispatchLoggingEnabled()	              << std::endl;
            ios << "\tDispatchTracingEnabled                  : " << GetDispatchTracingEnabled()	              << std::endl;
            ios << "\tMaximumConcurrency                      : " << GetMaximumConcurrency()                      << std::endl;
            ios << "\tMaximumHostConcurrency                  : " << GetMaximumHostConcurrency()                  << std::endl;
            ios << "\tTrackDeviceMemory                       : " << GetTrackDeviceMemory()                       << std::endl;
            ios << "\tTaskThreadPoolSize                      : " << GetTaskThreadPoolSize()                      << std::endl;
            ios << "\tSchedulerThreadPerTaskThreshold         : " << GetSchedulerThreadPerTaskThreshold()         << std::endl;
            ios << "\tUseGraphMonitorWatchdog                 : " << GetUseGraphMonitorWatchdog()                 << std::endl;
            ios << "\tCUDAHeapSize                            : " << GetCUDAHeapSize()                            << std::endl;
            ios << "\tThreadPoolPriorityQueues                : " << GetThreadPoolPriorityQueues()                << std::endl;
            ios << "\tDefaultGraphPartitioningMode            : " << GetDefaultGraphPartitioningMode()            << std::endl;
            ios << "\tProvisionBlockPoolsForCapacity          : " << GetProvisionBlockPoolsForCapacity()          << std::endl;
            ios << "\tTaskDispatchLocksIncomingAsyncSources   : " << GetTaskDispatchLocksIncomingAsyncSources()   << std::endl;
            ios << "\tThreadPoolSignalPerThread               : " << GetThreadPoolSignalPerThread()               << std::endl;
            ios << "\tTaskDispatchReadyCheckIncomingAsyncDeps : " << GetTaskDispatchReadyCheckIncomingAsyncDeps() << std::endl;
            ios << "\tTaskDispatchLocklessIncomingDepWait     : " << GetTaskDispatchLocklessIncomingDepWait()     << std::endl;
            ios << "\tIgnoreLocalityThreshold                 : " << GetIgnoreLocalityThreshold()                 << std::endl;
            ios << "\tUseHost                                 : " << GetUseHost()                                 << std::endl;
            ios << "\tUseCUDA                                 : " << GetUseCUDA()                                 << std::endl;
            ios << "\tUseOpenCL                               : " << GetUseOpenCL()                               << std::endl;
            ios << "\tUseDirectX                              : " << GetUseDirectX()	                          << std::endl;
            ios << "\tMinimumDirectXFeatureLevel              : " << GetMinimumDirectXFeatureLevel()              << std::endl;
            ios << "\tUseReferenceDrivers                     : " << GetUseReferenceDrivers()                     << std::endl;
            ios << "\tLoggingLevel                            : " << GetLoggingLevel()                            << std::endl;
            ios << "\tVerbose                                 : " << IsVerbose()                                  << std::endl;
            ios << "--------------------------------------------------------------------------"                   << std::endl << std::endl;
        }
    };
};
