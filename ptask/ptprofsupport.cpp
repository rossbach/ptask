///-------------------------------------------------------------------------------------------------
// file:	ptprofsupport.cpp
//
// summary:	Implements the ptprofsupport class
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------
// task profililing collects per-dispatch statistics about the
// latency of various dispatch-related operations such as lock 
// aqcuisition, input/output binding, time blocked on the ready 
// queue etc.
///-------------------------------------------------------------------------
#ifdef PROFILE_TASKS
static const int TASK_PROFILE_SUPPORT = 1;
#else
static const int TASK_PROFILE_SUPPORT = 0;
#endif

///-------------------------------------------------------------------------
// refcount object profiling tracks allocation and release of reference
// counted objects--this mode is handy for finding objects that are leaked
// or double-freed due to mismanagement of reference-counts in the runtime
// or at user-level.
///-------------------------------------------------------------------------
#ifdef PROFILE_REFCOUNT_OBJECTS
static const int REFCOUNT_PROFILE_SUPPORT = 1;
#else
static const int REFCOUNT_PROFILE_SUPPORT = 0;
#endif

///-------------------------------------------------------------------------
// datablock profiling tracks allocation/release of individual datablocks.
// It is essentially the same functionality as refcount profiling, but is
// restricted to datablocks only, which enables the profiler to track more
// detailed information about the history of objects that are leaked
// or double-freed.
///-------------------------------------------------------------------------
#ifdef PROFILE_DATABLOCKS
static const int DATABLOCK_PROFILE_SUPPORT = 1;
#else
static const int DATABLOCK_PROFILE_SUPPORT = 0;
#endif

///-------------------------------------------------------------------------
// collect frequency statistics and latency information about blocks
// that are required to "migrate", which here means blocks for which
// view materialization at dispatch time requires copy from one accelerator
// memory space to another (excluding the host memory space). 
///-------------------------------------------------------------------------
#ifdef PROFILE_MIGRATION
static const int MIGRATION_PROFILE_SUPPORT = 1;
#else
static const int MIGRATION_PROFILE_SUPPORT = 0;
#endif

///-------------------------------------------------------------------------
// collect allocation and delete statistics/latency data for platform-specific
// buffers. This mode is useful for pinpointing bottlenecks in workloads that
// are caused by allocation on the critical path. 
///-------------------------------------------------------------------------
#ifdef PROFILE_PBUFFERS
static const int PBUFFER_PROFILE_SUPPORT = 1;
#else
static const int PBUFFER_PROFILE_SUPPORT = 0;
#endif

///-------------------------------------------------------------------------
// keep track of the number of dispatches per task and compare that data
// with a user-provided prediction. This mode is useful for debugging iterative
// graph structures, since it will assert/break into the debugger when a
// graph's control flow diverges from the expectation expressed by the user.
///-------------------------------------------------------------------------
#ifdef DISPATCH_COUNT_DIAGNOSTICS
static const int DISPATCH_COUNT_SUPPORT = 1;
#else
static const int DISPATCH_COUNT_SUPPORT = 0;
#endif

///-------------------------------------------------------------------------
// support various diagnostics tools for debugging graphs or understanding
// graph behavior in the presence of faulty or abusive inputs. This mode
// enables the runtime to output warnings when graph structures are not
// well-formed, and enables watchdog monitor threads, hung-graph drawing
// utilities, etc. 
///-------------------------------------------------------------------------
#ifdef GRAPH_DIAGNOSTICS
static const int GRAPH_DIAGNOSTICS_SUPPORT = 1;
#else
static const int GRAPH_DIAGNOSTICS_SUPPORT = 0;
#endif

///-------------------------------------------------------------------------
// profile and report the behavior of channels (average occupancy,
// low and high-watermarks, unplanned block pool expansions, etc.)
///-------------------------------------------------------------------------
#ifdef PROFILE_CHANNELS
static const int CHANNEL_PROFILE_SUPPORT = 1;
#else
static const int CHANNEL_PROFILE_SUPPORT = 0;
#endif

#ifdef ADHOC_STATS
static const int ADHOC_INSTRUMENTATION_SUPPORT = 1;
#else
static const int ADHOC_INSTRUMENTATION_SUPPORT = 0;
#endif

///-------------------------------------------------------------------------
// profile and report the flow of control signals through a graph.
// Generally this is a diagnostics tool for helping understand the dynamic
// behavior of predicated dataflow structures which are notoriously
// difficult to get just right. 
///-------------------------------------------------------------------------
#ifdef PROFILE_CONTROLSIGNALS
static const int SIGNAL_PROFILE_SUPPORT = 1;
#else
static const int SIGNAL_PROFILE_SUPPORT = 0;
#endif

namespace PTask {
    
    namespace Runtime {

        int g_bTPProfilingSupported          = TASK_PROFILE_SUPPORT;
        int g_bRCProfilingSupported          = REFCOUNT_PROFILE_SUPPORT;
        int g_bDBProfilingSupported          = DATABLOCK_PROFILE_SUPPORT;
        int g_bCTProfilingSupported          = MIGRATION_PROFILE_SUPPORT;
        int g_bPBufferProfilingSupported     = PBUFFER_PROFILE_SUPPORT;
        int g_bInvocationCountingSupported   = DISPATCH_COUNT_SUPPORT;
        int g_bBlockPoolProfilingSupported   = GRAPH_DIAGNOSTICS_SUPPORT;
        int g_bChannelProfilingSupported     = CHANNEL_PROFILE_SUPPORT;
        int g_bAdhocInstrumentationSupported = ADHOC_INSTRUMENTATION_SUPPORT;
        int g_bSignalProfilingSupported      = SIGNAL_PROFILE_SUPPORT;

    };
};