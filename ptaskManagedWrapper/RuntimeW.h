///-------------------------------------------------------------------------------------------------
// file:	RuntimeW.h
//
// summary:	Declares the runtime wrapper class
///-------------------------------------------------------------------------------------------------

#pragma once
#pragma unmanaged
#include "ptaskapi.h"
#pragma managed
#include "CompiledKernelW.h"
#include "DatablockW.h"
#include "DataTemplate.h"
#include "GraphW.h"
#include "ChannelW.h"
#include "PortW.h"
#include "TaskW.h"
using namespace System;
using namespace System::Runtime::InteropServices; // For marshalling helpers

namespace Microsoft {
namespace Research {
namespace PTask {

    typedef unsigned long PTASK_RESULT;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Wrapper for native PTask::Runtime. </summary>
    ///
    /// <remarks>   refactored from code by jcurrey by crossbac, 4/13/2012. 
    ///             Follows the C++/CLI IDisposable pattern described at 
    ///             http://msdn.microsoft.com/en-us/library/system.idisposable.aspx
    /// 			</remarks>
    ///-------------------------------------------------------------------------------------------------

    public ref class Runtime
    {
    public:
        enum class THREADPOOLPOLICY {
            TPP_AUTOMATIC,    // let the scheduler decide how many threads to use
            TPP_EXPLICIT,     // the programmer must provide the task pool size
            TPP_THREADPERTASK // always use 1:1 thread:task mapping (original PTask policy) 
        };

		enum class INSTRUMENTATIONMETRIC {
			NONE = 0,
			ACTIVITY_FACTOR = 1,
			MEMORY_EFFICIENCY = 2,
			BRANCH_DIVERGENCE = 3,
			CLOCK_CYCLE_COUNT = 4
		};

        enum class PTASKSUBSYSTEM { 
            PTSYS_TASKS =                 ::PTask::Runtime::PTSYS_TASKS, 
            PTSYS_TASK_MIGRATION =        ::PTask::Runtime::PTSYS_TASK_MIGRATION, 
            PTSYS_PBUFFERS =              ::PTask::Runtime::PTSYS_PBUFFERS, 
            PTSYS_DATABLOCKS =            ::PTask::Runtime::PTSYS_DATABLOCKS, 
            PTSYS_COHERENCE =             ::PTask::Runtime::PTSYS_COHERENCE, 
            PTSYS_CHANNELS =              ::PTask::Runtime::PTSYS_CHANNELS, 
            PTSYS_DISPATCH =              ::PTask::Runtime::PTSYS_DISPATCH, 
            PTSYS_REFCOUNT_OBJECTS =      ::PTask::Runtime::PTSYS_REFCOUNT_OBJECTS, 
            PTSYS_ADHOC_INSTRUMENTATION = ::PTask::Runtime::PTSYS_ADHOC_INSTRUMENTATION, 
        };

        enum class GRAPHASSIGNMENTPOLICY {
            GMP_USER_DEFINED = ::PTask::Runtime::GMP_USER_DEFINED,
            GMP_ROUND_ROBIN =  ::PTask::Runtime::GMP_ROUND_ROBIN
        };

        static GRAPHASSIGNMENTPOLICY GetGraphAssignmentPolicy();
        static void SetGraphAssignmentPolicy(GRAPHASSIGNMENTPOLICY policy);
        static BOOL SubsystemReport(PTASKSUBSYSTEM eSubSystem);
        static PTASK_RESULT Initialize(bool bUseOpenCL);
        static PTASK_RESULT Initialize();
        static int GetDefaultChannelCapacity();
        static void SetDefaultChannelCapacity(int n);
        static int GetSchedulingMode();
        static void SetSchedulingMode(int mode);
        static void SetMaximumConcurrency(int nGPUs);
        static int GetMaximumConcurrency();
        static void SetMaximumHostConcurrency(int nHostAccelerators);
        static int GetMaximumHostConcurrency();
        static bool GetDebugMode();
        static bool GetForceSynchronous();
        static bool GetExtremeTrace();
        static bool GetDispatchLoggingEnabled();
        static bool GetDispatchTracingEnabled();
        static bool GetCoherenceProfileMode();
        static bool GetTaskProfileMode();
        static bool GetPageLockingEnabled();
        static bool GetDebugAsynchronyMode();
        static void SetDebugMode(bool b);
        static void SetForceSynchronous(bool b);
        static void SetExtremeTrace(bool b);
        static void SetDispatchLoggingEnabled(bool b);
        static void SetDispatchTracingEnabled(bool b);
        static void SetCoherenceProfileMode(bool b);
        static void SetTaskProfileMode(bool b);
        static void SetTaskProfileMode(bool bEnable, bool bConcise);
        static void SetPageLockingEnabled(bool b);
        static void SetDebugAsynchronyMode(bool b);
        static void SetPlatformBufferProfileMode(bool b);
        static bool GetPlatformBufferProfileMode();
        static void SetProfilePSDispatch(bool b);
        static bool GetProfilePSDispatch();
        static bool GetUseHost();
        static bool GetUseCUDA(); 
        static bool GetUseOpenCL();
        static bool GetUseDirectX();
        static void SetUseHost(bool b);
        static void SetUseCUDA(bool b); 
        static void SetUseOpenCL(bool b);
        static void SetUseDirectX(bool b);
        static void SetTrackDeviceMemory(bool bTrack);
        static bool GetTrackDeviceMemory();
        static UINT GetTaskThreadPoolSize();
        static void SetTaskThreadPoolSize(UINT uiThreadPoolSize);
        static THREADPOOLPOLICY GetTaskThreadPoolPolicy();
        static void SetTaskThreadPoolPolicy(THREADPOOLPOLICY ePolicy);
		static INSTRUMENTATIONMETRIC GetInstrumentationMetric();
		static void SetInstrumentationMetric(INSTRUMENTATIONMETRIC metric);
        static UINT GetSchedulerThreadPerTaskThreshold();
        static void SetSchedulerThreadPerTaskThreshold(UINT uiMaxTasks);
        static void SetUseGraphMonitorWatchdog(bool bUseWatchdog);
        static bool GetUseGraphMonitorWatchdog();
        static UINT GetCUDAHeapSize();
        static void SetCUDAHeapSize(UINT uiSizeBytes);
        static UINT GetDispatchWatchdogThreshold();
        static void SetDispatchWatchdogThreshold(UINT dwThreshold);
        static void RequireBlockPool(int nDataSize, int nMetaSize, int nTemplateSize, int nBlocks);
        static void RequireBlockPool(int nDataSize, int nMetaSize, int nTemplateSize);
        static void RequireBlockPool(DataTemplate^ pTemplate, int nBlocks);
        static void SetProvisionBlockPoolsForCapacity(bool bProvision);
        static bool GetProvisionBlockPoolsForCapacity();
        static void SetSizeDescriptorPoolSize(UINT uiBlocks);
        static UINT GetSizeDescriptorPoolSize();
        static UINT GetSchedulerThreadCount();
        static void SetSchedulerThreadCount(UINT uiThreads);
        static UINT GetGlobalThreadPoolSize();
        static bool GetPrimeGlobalThreadPool();  
        static bool GetGlobalThreadPoolGrowable();
        static void SetGlobalThreadPoolSize(UINT uiThreads);
        static void SetPrimeGlobalThreadPool(bool b);   
        static void SetGlobalThreadPoolGrowable(bool b);
        static bool GetThreadPoolSortReadyQueues();
        static void SetThreadPoolSortReadyQueues(bool b);
        static bool GetThreadPoolSignalPerThread();
        static void SetThreadPoolSignalPerThread(bool b);
        static void SetExitOnRuntimeFailure(bool b);
        static bool GetExitOnRuntimeFailure();
        static bool GetPerformanceWarningMode();
        static void SetPerformanceWarningMode(bool b);
        static bool GetBlockPoolsEnabled();
        static void SetBlockPoolsEnabled(bool b);    
        static UINT GetPlatformSpecificRuntimeVersion(ACCELERATOR_CLASS eAccClass);
        static UINT GetDefaultInitChannelBlockPoolSize();
        static void SetDefaultInitChannelBlockPoolSize(UINT n);
        static UINT GetDefaultInputChannelBlockPoolSize();
        static void SetDefaultInputChannelBlockPoolSize(UINT n);
        static bool GetVerbose();
        static void SetVerbose(bool b);
        static UINT GetLoggingLevel();
        static void SetLoggingLevel(UINT ui);
        static void SetAggressiveReleaseMode(bool bEnable);
        static bool GetAggressiveReleaseMode();
        static bool GetDBProfilingEnabled();
        static void SetDBProfilingEnabled(bool bEnable);
        static UINT GetGCSweepThresholdPercent();
        static void SetGCSweepThresholdPercent(UINT uiPercent);
        static PTASK_RESULT Terminate();
        static CompiledKernel^ GetCompiledKernel(String^ sourceFile, String^ operation);
        static CompiledKernel^ GetCompiledKernelEx(String^ sourceFile, String^ operation, String^initBinary, String^initEntry, ACCELERATOR_CLASS ePSClass);
		static double CollectInstrumentationData(String^ strEventName);
		static double CollectInstrumentationData(String^ strEventName, UINT% uiSamples, double% dMin, double% dMax);
        static Graph^ CreateGraph();
        static DataTemplate^ GetDataTemplate(
            String^ type, 
            unsigned int uiStride, 
            unsigned int x, 
            unsigned int y, 
            unsigned int z
        );

        static DataTemplate^ GetDataTemplate(
            String^ type, 
            unsigned int uiStride, 
            unsigned int x, 
            unsigned int y, 
            unsigned int z,
            bool bRaw
        );

        static DataTemplate^ GetDataTemplate(
            String^ type, 
            UInt32 uiStride, 
            DataTemplate::PTASK_PARM_TYPE portType
            );

        static Port^ CreatePort(
            Port::PORTTYPE type,
            DataTemplate^ managedTemplate, 
            unsigned int uiId,
            String^ variableBinding,
            unsigned int nKernelParameterIndex,
            unsigned int nInOutParmOutputIndex);

        static Port^ CreatePort(
            Port::PORTTYPE type,
            DataTemplate^ managedTemplate, 
            unsigned int uiId,
            String^ variableBinding,
            unsigned int nKernelParameterIndex);

        // Runtime.CreatePort
        static Port^ CreatePort(
            Port::PORTTYPE type,
            DataTemplate^ managedTemplate, 
            unsigned int uiId,
            String^ variableBinding);

        // Runtime.CreatePort (variableBinding == NULL)
        static Port^ CreatePort(
            Port::PORTTYPE type,
            DataTemplate^ pTemplate, 
            UINT uiId);

        // Runtime.AllocateControlDatablock
        // Allocate a control data block. 
        // Current control codes are defined as static
        // constants in the Datablock wrapper class.
        static Datablock^ AllocateControlDatablock(
            int ctlCode
            );

        // Runtime.AllocateDatablock
        // This variant does not provide the data at the time of allocation.
        // Instead, call GetBufferPointer() on the resulting Datablock, 
        // to get a handle to the buffer to populate.
        static Datablock^ AllocateDatablock(
            int dataBufferSize, 
            int metaBufferSize, 
            int templateBufferSize
            );

        // Runtime.AllocateDatablock
        // This variant provides the data at the time of allocation.
        // Specifying the destination channel triggers the buffer to be shipped directly
        // to GPU memory.
        static Datablock^ AllocateDatablock(
            DataTemplate^ blockTemplate,
            byte* data,
            int cbData,
            Channel^ destChannel // null is allowed, but should be avoided if possible.
        );

        static void FreeDatablock(Datablock^ block);
        static void FreeDatablock(Datablock^ block, bool bReaderContext);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check to see if the graph is well-formed. This is not an exhaustive check, but a
        ///             collection of obvious sanity checks. Fails on warnings, emits verbose output. 
        ///             Use the other CheckGraphSemantics to change those behaviors.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/27/2011. </remarks>
        ///
        /// <param name="pGraph">           [in] non-null, the graph. </param>
        ///                                 when malformed graph elements are found. </param>
        ///
        /// <returns>   PTRESULT--use PTSUCCESS/PTFAILED macros: PTASK_OK:   the graph is well-formed
        ///             PTASK_ERR_GRAPH_MALFORMED: the graph is malformed in a way that cannot be
        ///                                        tolerated by the runtime. 
        ///             PTASK_WARNING_GRAPH_MALFORMED: the graph is malformed in a way that can be
        ///                                        tolerated by the runtime.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        static PTASK_RESULT 
        CheckGraphSemantics(
            Graph^ pGraph
            );

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

        static PTASK_RESULT 
        CheckGraphSemantics(
            Graph^ pGraph,
            bool bVerbose,
            bool bFailOnWarning
            );

        static void
        Instrument(
            String ^ptxFile,
            String ^instrumentedPTXFile
        );

        static void
        CleanupInstrumentation(void);

        static bool
        Runtime::Instrumented(void);

        static void Runtime::SetInstrumented(bool b);

    };

}}}
