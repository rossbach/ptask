///-------------------------------------------------------------------------------------------------
// file:	RuntimeW.cpp
//
// summary:	Implements the runtime wrapper class
///-------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "PTaskManagedWrapper.h"
#include "Utils.h"

namespace Microsoft {
namespace Research {
namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the runtime. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="bUseOpenCL">   true to use open cl. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTASK_RESULT 
    Runtime::Initialize(bool bUseOpenCL)
    {
        ::PTask::Runtime::SetUseOpenCL(bUseOpenCL);
        return ::PTask::Runtime::Initialize();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the runtime. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTASK_RESULT
    Runtime::Initialize()
    {
		::PTask::Runtime::SetUseOpenCL(FALSE);
        return ::PTask::Runtime::Initialize();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Terminate the runtime. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTASK_RESULT 
    Runtime::Terminate()
    {
        return ::PTask::Runtime::Terminate();
    }

    Runtime::GRAPHASSIGNMENTPOLICY Runtime::GetGraphAssignmentPolicy(){ return (Runtime::GRAPHASSIGNMENTPOLICY)::PTask::Runtime::GetGraphAssignmentPolicy();}
    void Runtime::SetGraphAssignmentPolicy(GRAPHASSIGNMENTPOLICY p)   { ::PTask::Runtime::SetGraphAssignmentPolicy((::PTask::Runtime::GRAPHASSIGNMENTPOLICY)p); }
    BOOL Runtime::SubsystemReport(PTASKSUBSYSTEM eSubSystem)          { return ::PTask::Runtime::SubsystemReport((::PTask::Runtime::ptasksubsystem_t) eSubSystem); }
    UINT Runtime::GetGCSweepThresholdPercent()                        { return ::PTask::Runtime::GetGCSweepThresholdPercent(); }            
    void Runtime::SetGCSweepThresholdPercent(UINT uiPercent)          { ::PTask::Runtime::SetGCSweepThresholdPercent(uiPercent); }            
    bool Runtime::GetDBProfilingEnabled()                             { return ::PTask::Runtime::GetDBProfilingEnabled()!=0; } 
    void Runtime::SetDBProfilingEnabled(bool bEnable)                 { ::PTask::Runtime::SetDBProfilingEnabled(bEnable?1:0); }
    bool Runtime::GetAggressiveReleaseMode()                          { return ::PTask::Runtime::GetAggressiveReleaseMode()!=0; }
    void Runtime::SetAggressiveReleaseMode(bool bEnable)              { ::PTask::Runtime::SetAggressiveReleaseMode(bEnable?1:0); }
    bool Runtime::GetVerbose()                                        { return ::PTask::Runtime::IsVerbose()!=0; }
    void Runtime::SetVerbose(bool b)                                  { ::PTask::Runtime::SetVerbose(b?1:0); }
    UINT Runtime::GetLoggingLevel()                                   { return ::PTask::Runtime::GetLoggingLevel(); }
    void Runtime::SetLoggingLevel(UINT ui)                            { ::PTask::Runtime::SetLoggingLevel(ui); }
    UINT Runtime::GetDefaultInputChannelBlockPoolSize()               { return ::PTask::Runtime::GetDefaultInputChannelBlockPoolSize(); } 
    void Runtime::SetDefaultInputChannelBlockPoolSize(UINT n)         { ::PTask::Runtime::SetDefaultInputChannelBlockPoolSize(n); }
    UINT Runtime::GetDefaultInitChannelBlockPoolSize()                { return ::PTask::Runtime::GetDefaultInitChannelBlockPoolSize(); } 
    void Runtime::SetDefaultInitChannelBlockPoolSize(UINT n)          { ::PTask::Runtime::SetDefaultInitChannelBlockPoolSize(n); }
    bool Runtime::GetBlockPoolsEnabled()                              { return ::PTask::Runtime::GetBlockPoolsEnabled()!=0; }
    void Runtime::SetBlockPoolsEnabled(bool b)                        { ::PTask::Runtime::SetBlockPoolsEnabled(b?1:0); }
    bool Runtime::GetPerformanceWarningMode()                         { return ::PTask::Runtime::GetCriticalPathAllocMode()!=0; }
    void Runtime::SetPerformanceWarningMode(bool b)                   { ::PTask::Runtime::SetCriticalPathAllocMode(b?1:0); }
    void Runtime::SetExitOnRuntimeFailure(bool b)                     { ::PTask::Runtime::SetExitOnRuntimeFailure(b?1:0); }
    bool Runtime::GetExitOnRuntimeFailure()                           { return ::PTask::Runtime::GetExitOnRuntimeFailure()!=0; }
    bool Runtime::GetThreadPoolSortReadyQueues()                      { return ::PTask::Runtime::GetThreadPoolPriorityQueues()!=0;}
    void Runtime::SetThreadPoolSortReadyQueues(bool b)                { ::PTask::Runtime::SetThreadPoolPriorityQueues(b?1:0); }
    bool Runtime::GetThreadPoolSignalPerThread()                      { return ::PTask::Runtime::GetThreadPoolSignalPerThread()!=0; }
    void Runtime::SetThreadPoolSignalPerThread(bool b)                { ::PTask::Runtime::SetThreadPoolSignalPerThread(b?1:0); }
    UINT Runtime::GetGlobalThreadPoolSize()                           { return ::PTask::Runtime::GetGlobalThreadPoolSize(); }
    bool Runtime::GetPrimeGlobalThreadPool()                          { return ::PTask::Runtime::GetPrimeGlobalThreadPool()!=0; }
    bool Runtime::GetGlobalThreadPoolGrowable()                       { return ::PTask::Runtime::GetGlobalThreadPoolGrowable()!=0; }
    void Runtime::SetGlobalThreadPoolSize(UINT uiThreads)             { ::PTask::Runtime::SetGlobalThreadPoolSize(uiThreads); }
    void Runtime::SetPrimeGlobalThreadPool(bool b)                    { ::PTask::Runtime::SetPrimeGlobalThreadPool(b?1:0); }
    void Runtime::SetGlobalThreadPoolGrowable(bool b)                 { ::PTask::Runtime::SetGlobalThreadPoolGrowable(b?1:0); }
    UINT Runtime::GetSchedulerThreadCount()                           { return ::PTask::Runtime::GetSchedulerThreadCount(); }
    void Runtime::SetSchedulerThreadCount(UINT uiThreads)             { ::PTask::Runtime::SetSchedulerThreadCount(uiThreads); }
    void Runtime::SetProvisionBlockPoolsForCapacity(bool bProvision)  { ::PTask::Runtime::SetProvisionBlockPoolsForCapacity(bProvision?1:0); }
    bool Runtime::GetProvisionBlockPoolsForCapacity()                 { return ::PTask::Runtime::GetProvisionBlockPoolsForCapacity()!=0; }
    void Runtime::SetSizeDescriptorPoolSize(UINT uiBlocks)            { ::PTask::Runtime::SetSizeDescriptorPoolSize(uiBlocks); }
    UINT Runtime::GetSizeDescriptorPoolSize()                         { return ::PTask::Runtime::GetSizeDescriptorPoolSize(); }
    void Runtime::SetProfilePSDispatch(bool bEnable)                  { ::PTask::Runtime::SetProfilePSDispatch(bEnable?1:0); }
    bool Runtime::GetProfilePSDispatch()                              { return ::PTask::Runtime::GetProfilePSDispatch()?true: false; }
    void Runtime::SetTrackDeviceMemory(bool bTrack)                   { ::PTask::Runtime::SetTrackDeviceMemory(bTrack?1:0); }
    bool Runtime::GetTrackDeviceMemory()                              { return ::PTask::Runtime::GetTrackDeviceMemory()==0?false:true; }
    UINT Runtime::GetTaskThreadPoolSize()                             { return ::PTask::Runtime::GetTaskThreadPoolSize(); }
    void Runtime::SetTaskThreadPoolSize(UINT uiThreadPoolSize)        { ::PTask::Runtime::SetTaskThreadPoolSize(uiThreadPoolSize); }
    UINT Runtime::GetSchedulerThreadPerTaskThreshold()                { return ::PTask::Runtime::GetSchedulerThreadPerTaskThreshold(); }
    void Runtime::SetSchedulerThreadPerTaskThreshold(UINT uiMaxTasks) { ::PTask::Runtime::SetSchedulerThreadPerTaskThreshold(uiMaxTasks); }
    void Runtime::SetUseGraphMonitorWatchdog(bool bUseWatchdog)       { ::PTask::Runtime::SetUseGraphMonitorWatchdog(bUseWatchdog?1:0); }
    bool Runtime::GetUseGraphMonitorWatchdog()                        { return ::PTask::Runtime::GetUseGraphMonitorWatchdog()==0?false:true; }
    UINT Runtime::GetCUDAHeapSize()                                   { return ::PTask::Runtime::GetCUDAHeapSize(); }
    void Runtime::SetCUDAHeapSize(UINT uiSizeBytes)                   { ::PTask::Runtime::SetCUDAHeapSize(uiSizeBytes); }
    UINT Runtime::GetDispatchWatchdogThreshold()                      { return ::PTask::Runtime::GetDispatchWatchdogThreshold(); }
    void Runtime::SetDispatchWatchdogThreshold(UINT dwThreshold)      { ::PTask::Runtime::SetDispatchWatchdogThreshold(dwThreshold); }
    void Runtime::SetTaskThreadPoolPolicy(THREADPOOLPOLICY ePolicy)   { ::PTask::Runtime::SetTaskThreadPoolPolicy((::PTask::Runtime::THREADPOOLPOLICY) ePolicy); }
    Runtime::THREADPOOLPOLICY Runtime::GetTaskThreadPoolPolicy()      { return (THREADPOOLPOLICY)::PTask::Runtime::GetTaskThreadPoolPolicy(); }
	void Runtime::SetInstrumentationMetric(INSTRUMENTATIONMETRIC metric)		  { ::PTask::Runtime::SetInstrumentationMetric((::PTask::Runtime::INSTRUMENTATIONMETRIC) metric); }
	Runtime::INSTRUMENTATIONMETRIC Runtime::GetInstrumentationMetric()			  { return (INSTRUMENTATIONMETRIC)::PTask::Runtime::GetInstrumentationMetric(); }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the platform specific runtime version for a particular accelerator class.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 8/29/2013. </remarks>
    ///
    /// <param name="eAcceleratorClass">   The class. </param>
    ///
    /// <returns>   The platform specific runtime version. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Runtime::GetPlatformSpecificRuntimeVersion(
        __in ACCELERATOR_CLASS eAcceleratorClass
        )     
    { 
        UINT uiPSRuntimeVersion = 0;
        BOOL bSuccess = ::PTask::Runtime::GetPlatformSpecificRuntimeVersion(
                            static_cast<::PTask::ACCELERATOR_CLASS>(eAcceleratorClass),
                            uiPSRuntimeVersion);
        return bSuccess ? uiPSRuntimeVersion : 0;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the default channel capacity. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <returns>   The default channel capacity. </returns>
    ///-------------------------------------------------------------------------------------------------

    int Runtime::GetDefaultChannelCapacity() {
        return ::PTask::Runtime::GetDefaultChannelCapacity();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the default channel capacity. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="n">    The n. </param>
    ///-------------------------------------------------------------------------------------------------

    void Runtime::SetDefaultChannelCapacity(int n)
    {
        ::PTask::Runtime::SetDefaultChannelCapacity(n);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the scheduling mode. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <returns>   The scheduling mode. </returns>
    ///-------------------------------------------------------------------------------------------------

    int Runtime::GetSchedulingMode()
    {
        return ::PTask::Runtime::GetSchedulingMode();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the scheduling mode. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="mode"> The mode. </param>
    ///-------------------------------------------------------------------------------------------------

    void Runtime::SetSchedulingMode(int mode)
    {
        ::PTask::Runtime::SetSchedulingMode(mode);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the maximum concurrency. 0 is unlimited.</summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="nGPUs">    The number of gpus. </param>
    ///-------------------------------------------------------------------------------------------------

    void Runtime::SetMaximumConcurrency(int nGPUs)
    {
        ::PTask::Runtime::SetMaximumConcurrency(nGPUs);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the maximum concurrency. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <returns>   The maximum concurrency. </returns>
    ///-------------------------------------------------------------------------------------------------

    int Runtime::GetMaximumConcurrency()
    {
        return ::PTask::Runtime::GetMaximumConcurrency();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the maximum concurrency. 0 is unlimited.</summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="nGPUs">    The number of gpus. </param>
    ///-------------------------------------------------------------------------------------------------

    void Runtime::SetMaximumHostConcurrency(int n)
    {
        ::PTask::Runtime::SetMaximumHostConcurrency(n);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the maximum concurrency. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <returns>   The maximum concurrency. </returns>
    ///-------------------------------------------------------------------------------------------------

    int Runtime::GetMaximumHostConcurrency()
    {
        return ::PTask::Runtime::GetMaximumHostConcurrency();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   getters and setters for debug mode and dispatch logging enable. </summary>
    ///
    /// <remarks>   crossbac, 6/12/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool Runtime::GetDebugMode() { return ::PTask::Runtime::GetDebugMode() != 0; }
    bool Runtime::GetForceSynchronous() { return ::PTask::Runtime::GetForceSynchronous() != 0; }
    bool Runtime::GetExtremeTrace() { return ::PTask::Runtime::GetExtremeTraceMode() != 0; }
    bool Runtime::GetDispatchLoggingEnabled() { return ::PTask::Runtime::GetDispatchLoggingEnabled() != 0; }
    bool Runtime::GetDispatchTracingEnabled() { return ::PTask::Runtime::GetDispatchTracingEnabled() != 0; }
    void Runtime::SetDebugMode(bool bMode) { ::PTask::Runtime::SetDebugMode(bMode ? 1 : 0); }
    void Runtime::SetForceSynchronous(bool bMode) { ::PTask::Runtime::SetForceSynchronous(bMode ? 1 : 0); }
    void Runtime::SetExtremeTrace(bool bMode) { ::PTask::Runtime::SetExtremeTraceMode(bMode ? 1 : 0); }
    void Runtime::SetDispatchLoggingEnabled(bool bEnabled) { ::PTask::Runtime::SetDispatchLoggingEnabled(bEnabled?1:0); }
    void Runtime::SetDispatchTracingEnabled(bool bEnabled) { ::PTask::Runtime::SetDispatchTracingEnabled(bEnabled?1:0); }
    void Runtime::SetCoherenceProfileMode(bool bEnabled) { ::PTask::Runtime::SetCoherenceProfileMode(bEnabled?1:0); }
    void Runtime::SetTaskProfileMode(bool bEnabled) { ::PTask::Runtime::SetTaskProfileMode(bEnabled?1:0); }
    void Runtime::SetTaskProfileMode(bool bEnabled, bool bConcise) { ::PTask::Runtime::SetTaskProfileMode(bEnabled?1:0, bConcise?1:0); }
    void Runtime::SetPageLockingEnabled(bool bEnabled) { ::PTask::Runtime::SetPageLockingEnabled(bEnabled?1:0); }
    void Runtime::SetDebugAsynchronyMode(bool b) { ::PTask::Runtime::SetDebugAsynchronyMode(b?1:0); }
    bool Runtime::GetCoherenceProfileMode() { return ::PTask::Runtime::GetCoherenceProfileMode() != 0; }
    bool Runtime::GetTaskProfileMode() { return ::PTask::Runtime::GetTaskProfileMode() != 0; }
    bool Runtime::GetPageLockingEnabled() { return ::PTask::Runtime::GetPageLockingEnabled() != 0; }
    bool Runtime::GetDebugAsynchronyMode() { return ::PTask::Runtime::GetDebugAsynchronyMode() != 0; }
    void Runtime::SetPlatformBufferProfileMode(bool b) { ::PTask::Runtime::SetProfilePlatformBuffers(b?1:0); }
    bool Runtime::GetPlatformBufferProfileMode() { return ::PTask::Runtime::GetProfilePlatformBuffers() != 0; }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the use host property. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool Runtime::GetUseHost()
    {
        return ::PTask::Runtime::GetUseHost() != 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the use cuda property. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool Runtime::GetUseCUDA()
    {
        return ::PTask::Runtime::GetUseCUDA() != 0;
    } 

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the use opencl property. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool Runtime::GetUseOpenCL()
    {
        return ::PTask::Runtime::GetUseOpenCL() != 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the use directx property. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool Runtime::GetUseDirectX()
    {
        return ::PTask::Runtime::GetUseDirectX() != 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the use host property. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="b">    true to b. </param>
    ///-------------------------------------------------------------------------------------------------

    void Runtime::SetUseHost(bool b)
    {
        ::PTask::Runtime::SetUseHost(b);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the use cuda property. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="b">    true to b. </param>
    ///-------------------------------------------------------------------------------------------------

    void Runtime::SetUseCUDA(bool b)
    {
        ::PTask::Runtime::SetUseCUDA(b);
    } 

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the use opencl property. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="b">    true to b. </param>
    ///-------------------------------------------------------------------------------------------------

    void Runtime::SetUseOpenCL(bool b)
    {
        ::PTask::Runtime::SetUseOpenCL(b);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the use directx property. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="b">    true to b. </param>
    ///-------------------------------------------------------------------------------------------------

    void Runtime::SetUseDirectX(bool b)
    {
        ::PTask::Runtime::SetUseDirectX(b);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a compiled kernel. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="sourceFile">   [in,out] If non-null, source file. </param>
    /// <param name="operation">    [in,out] If non-null, the operation. </param>
    ///
    /// <returns>   null if it fails, else the compiled kernel. </returns>
    ///-------------------------------------------------------------------------------------------------

    CompiledKernel^ 
    Runtime::GetCompiledKernel(String^ sourceFile, String^ operation)
    {
        // TODO: Check if this leaks, and fix if it does.
        IntPtr p1 = Marshal::StringToHGlobalAnsi(sourceFile);
        char*lpszSourceFile = static_cast<char*>(p1.ToPointer());
        IntPtr p2 = Marshal::StringToHGlobalAnsi(operation);
        char*lpszOperation = static_cast<char*>(p2.ToPointer());

        ::PTask::CompiledKernel* nativeCompiledKernel = 
            ::PTask::Runtime::GetCompiledKernel(lpszSourceFile, lpszOperation);

        CompiledKernel^ managedCompiledKernel = gcnew CompiledKernel(nativeCompiledKernel);
        return managedCompiledKernel;
    }

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Gets compiled kernel with static initializer. </summary>
	///
	/// <remarks>	crossbac, 8/13/2013. </remarks>
	///
	/// <param name="sourceFile">	[in,out] If non-null, source file. </param>
	/// <param name="operation"> 	[in,out] If non-null, the operation. </param>
	/// <param name="initBinary">	[in,out] If non-null, the initialise binary. </param>
	/// <param name="initEntry"> 	[in,out] If non-null, the initialise entry. </param>
	/// <param name="ePSClass">  	The ps class. </param>
	///
	/// <returns>	nullptr if it fails, else the compiled kernel ex. </returns>
	///-------------------------------------------------------------------------------------------------

	CompiledKernel^ 
	Runtime::GetCompiledKernelEx(
		String^ sourceFile, 
		String^ operation, 
		String^initBinary, 
		String^initEntry, 
		ACCELERATOR_CLASS ePSClass
		)
	{
        IntPtr p1 = Marshal::StringToHGlobalAnsi(sourceFile);
        char*lpszSourceFile = static_cast<char*>(p1.ToPointer());
        IntPtr p2 = Marshal::StringToHGlobalAnsi(operation);
        char*lpszOperation = static_cast<char*>(p2.ToPointer());
        IntPtr p3 = Marshal::StringToHGlobalAnsi(initBinary);
        char*lpszInitBinary= static_cast<char*>(p3.ToPointer());
        IntPtr p4 = Marshal::StringToHGlobalAnsi(initEntry);
        char*lpszInitEntry = static_cast<char*>(p4.ToPointer());

        ::PTask::CompiledKernel* nativeCompiledKernel = 
            ::PTask::Runtime::GetCompiledKernelEx(lpszSourceFile, 
											    lpszOperation, 
											    lpszInitBinary, 
											    lpszInitEntry, 
											    (::PTask::ACCELERATOR_CLASS)ePSClass);

        CompiledKernel^ managedCompiledKernel = gcnew CompiledKernel(nativeCompiledKernel);
        return managedCompiledKernel;
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates the graph. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Graph^ 
    Runtime::CreateGraph()
    {
        ::PTask::Graph* nativeGraph = ::PTask::Runtime::CreateGraph();
        Graph^ managedGraph = gcnew Graph(nativeGraph);
        return managedGraph;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a data template. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="type">     [in,out] If non-null, the type. </param>
    /// <param name="uiStride"> The stride. </param>
    /// <param name="x">        The x coordinate. </param>
    /// <param name="y">        The y coordinate. </param>
    /// <param name="z">        The z coordinate. </param>
    ///
    /// <returns>   null if it fails, else the data template. </returns>
    ///-------------------------------------------------------------------------------------------------

    DataTemplate^ 
    Runtime::GetDataTemplate(
        String^ type, 
        UInt32 uiStride, 
        UInt32 x, 
        UInt32 y, 
        UInt32 z
        )
    {
        return Runtime::GetDataTemplate(type, uiStride, x, y, z, false);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a data template. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="type">     [in,out] If non-null, the type. </param>
    /// <param name="uiStride"> The stride. </param>
    /// <param name="x">        The x coordinate. </param>
    /// <param name="y">        The y coordinate. </param>
    /// <param name="z">        The z coordinate. </param>
    /// <param name="bRaw">     true to raw. </param>
    ///
    /// <returns>   null if it fails, else the data template. </returns>
    ///-------------------------------------------------------------------------------------------------

    DataTemplate^ 
    Runtime::GetDataTemplate(
        String^ type, 
        UInt32 uiStride, 
        UInt32 x, 
        UInt32 y, 
        UInt32 z,
        bool bRaw
        )
    {
        // TODO: Check if this leaks, and fix if it does.
        IntPtr p1 = Marshal::StringToHGlobalAnsi(type);
        char*lpszType = static_cast<char*>(p1.ToPointer());

        ::PTask::DatablockTemplate* nativeTemplate =
            ::PTask::Runtime::GetDatablockTemplate(lpszType, uiStride, x, y, z, bRaw, bRaw);

        // TODO: Keep a map of native to managed instances and return existing managed instance
        // if this native instance has already been retrieved. (So that there is a single instance
        // per set of characteristics, at the managed as well as native level).
        DataTemplate^ managedTemplate = gcnew DataTemplate(nativeTemplate);
        return managedTemplate;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a data template. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="type">     [in,out] If non-null, the type. </param>
    /// <param name="uiStride"> The stride. </param>
    /// <param name="portType"> Type of the port. </param>
    ///
    /// <returns>   null if it fails, else the data template. </returns>
    ///-------------------------------------------------------------------------------------------------

	DataTemplate^ 
    Runtime::GetDataTemplate(
        String^ type, 
        UInt32 uiStride, 
		DataTemplate::PTASK_PARM_TYPE portType
        )
    {
		::PTask::PTASK_PARM_TYPE nativeType = static_cast<::PTask::PTASK_PARM_TYPE>(portType);

        // TODO: Check if this leaks, and fix if it does.
        IntPtr p1 = Marshal::StringToHGlobalAnsi(type);
        char*lpszType = static_cast<char*>(p1.ToPointer());

        ::PTask::DatablockTemplate* nativeTemplate =
            ::PTask::Runtime::GetDatablockTemplate(lpszType, uiStride, nativeType);

        // TODO: Keep a map of native to managed instances and return existing managed instance
        // if this native instance has already been retrieved. (So that there is a single instance
        // per set of characteristics, at the managed as well as native level).
        DataTemplate^ managedTemplate = gcnew DataTemplate(nativeTemplate);
        return managedTemplate;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a port. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="type">                     The type. </param>
    /// <param name="managedTemplate">          [in,out] If non-null, the managed template. </param>
    /// <param name="uiId">                     The identifier. </param>
    /// <param name="variableBinding">          [in,out] If non-null, the variable binding. </param>
    /// <param name="nKernelParameterIndex">    Zero-based index of the n kernel parameter. </param>
    /// <param name="nInOutParmOutputIndex">    Zero-based index of the n in out parm output. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port^ 
    Runtime::CreatePort(
        Port::PORTTYPE type,
        DataTemplate^ managedTemplate, 
        unsigned int uiId,
        String^ variableBinding,
		unsigned int nKernelParameterIndex,
		unsigned int nInOutParmOutputIndex)
    {
        ::PTask::PORTTYPE nativeType = static_cast<::PTask::PORTTYPE>(type);

        ::PTask::DatablockTemplate* nativeTemplate = managedTemplate->GetNativeDatablockTemplate();

        // Convert (optional) variable binding from managed to native string.
        char*lpszVariableBinding = NULL;
        if (!String::IsNullOrEmpty(variableBinding))
        {
            // TODO: Check if this leaks, and fix if it does.
            IntPtr p1 = Marshal::StringToHGlobalAnsi(variableBinding);
            lpszVariableBinding = static_cast<char*>(p1.ToPointer());
        }

        ::PTask::Port* nativePort =
            ::PTask::Runtime::CreatePort(nativeType, nativeTemplate, uiId, lpszVariableBinding, nKernelParameterIndex, nInOutParmOutputIndex);

        Port^ managedPort = gcnew Port(nativePort, managedTemplate);
        return managedPort;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a port. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="type">                     The type. </param>
    /// <param name="managedTemplate">          [in,out] If non-null, the managed template. </param>
    /// <param name="uiId">                     The identifier. </param>
    /// <param name="variableBinding">          [in,out] If non-null, the variable binding. </param>
    /// <param name="nKernelParameterIndex">    Zero-based index of the n kernel parameter. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

	Port^ 
    Runtime::CreatePort(
        Port::PORTTYPE type,
        DataTemplate^ managedTemplate, 
        unsigned int uiId,
        String^ variableBinding,
		unsigned int nKernelParameterIndex)
	{
		return CreatePort(type, managedTemplate, uiId, variableBinding, nKernelParameterIndex, static_cast<unsigned int>(-1));
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a port. (variableBinding == NULL)</summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="type">         The type. </param>
    /// <param name="pTemplate">    [in,out] If non-null, the template. </param>
    /// <param name="uiId">         The identifier. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port^ 
    Runtime::CreatePort(
        Port::PORTTYPE type,
        DataTemplate^ pTemplate, 
        UINT uiId)
    {
        return CreatePort(type, pTemplate, uiId, nullptr, static_cast<unsigned int>(-1), static_cast<unsigned int>(-1));
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a port. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="type">             The type. </param>
    /// <param name="managedTemplate">  [in,out] If non-null, the managed template. </param>
    /// <param name="uiId">             The identifier. </param>
    /// <param name="variableBinding">  [in,out] If non-null, the variable binding. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

	Port^ 
    Runtime::CreatePort(
        Port::PORTTYPE type,
        DataTemplate^ managedTemplate, 
        unsigned int uiId,
        String^ variableBinding)
	{
		return CreatePort(type, managedTemplate, uiId, variableBinding, static_cast<unsigned int>(-1), static_cast<unsigned int>(-1));
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate control datablock.This variant allocates a control data block. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="ctlcode">  The ctlcode. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock^ 
    Runtime::AllocateControlDatablock(
        int ctlcode 
        )
    {
        ::PTask::Datablock* nativeBlock = 
            ::PTask::Runtime::AllocateDatablock(
                ctlcode
                );
        // nativeBlock->AddRef();
        Datablock^ managedBlock = gcnew Datablock(nativeBlock);
        return managedBlock;
	}
   
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate datablock. This variant does not provide the data at the time of
    ///             allocation. Instead, call GetBufferPointer() on the resulting Datablock, to get a
    ///             handle to the buffer to populate.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="dataBufferSize">       Size of the data buffer. </param>
    /// <param name="metaBufferSize">       Size of the meta buffer. </param>
    /// <param name="templateBufferSize">   Size of the template buffer. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock^ 
    Runtime::AllocateDatablock(
        int dataBufferSize, 
        int metaBufferSize, 
        int templateBufferSize
        )
    {
        ::PTask::Datablock* nativeBlock = 
            ::PTask::Runtime::AllocateDatablock(
                NULL,
                dataBufferSize,
                metaBufferSize,
                templateBufferSize);
        // nativeBlock->AddRef();
        Datablock^ managedBlock = gcnew Datablock(nativeBlock);
        return managedBlock;
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate datablock. This variant provides the data at the time of allocation.
    ///             Specifying the destination channel triggers the buffer to be shipped directly to
    ///             GPU memory.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="blockTemplate">    [in,out] If non-null, the block template. </param>
    /// <param name="data">             [in,out] If non-null, the data. </param>
    /// <param name="destChannel">      [in,out] If non-null, destination channel. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock^
    Runtime::AllocateDatablock(
        DataTemplate^ blockTemplate,
        byte* data,
        int cbData,
        Channel^ destChannel // null is allowed, but should be avoided if possible.
        )
    {
        // destChannel may be null.
        ::PTask::Channel* nativeDestChannel = NULL;
        if (destChannel != nullptr)
        {
            nativeDestChannel = destChannel->GetNativeChannel();
        }

        ::PTask::Datablock* nativeBlock = 
            ::PTask::Runtime::AllocateDatablock(
                blockTemplate->GetNativeDatablockTemplate(),
                data,
                (UINT)cbData,
                nativeDestChannel,
                0 // TODO: Need to expose buffer access flags? 0 = set automatically.
                );
        // nativeBlock->AddRef();
        Datablock^ managedBlock = gcnew Datablock(nativeBlock);
        return managedBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Free datablock. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="block">    [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Runtime::FreeDatablock(Datablock^ block)
    {
        block->Free();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Free datablock. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="block">    [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Runtime::FreeDatablock(
        __in Datablock^ block,
        bool bReaderContext
        )
    {
        block->Free(bReaderContext);
    }

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

    PTASK_RESULT 
    Runtime::CheckGraphSemantics(
        Graph^ pGraph
        )
    {
        return CheckGraphSemantics(pGraph, true, true);
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

    PTASK_RESULT
    Runtime::CheckGraphSemantics(
        Graph^ pGraph,
        bool bVerbose,
        bool bFailOnWarning
        )
    {
        return (PTASK_RESULT) 
            ::PTask::Runtime::CheckGraphSemantics(pGraph->GetNativeGraph(),
                                                   bVerbose,
                                                   bFailOnWarning);
    }

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Collect instrumentation data. </summary>
	///
	/// <remarks>	crossbac, 8/12/2013. </remarks>
	///
	/// <param name="strEventName">	[in,out] If non-null, name of the event. </param>
	///
	/// <returns>	. </returns>
	///-------------------------------------------------------------------------------------------------

	double 
	Runtime::CollectInstrumentationData(
		String^ strEventName
		)
	{
        IntPtr p1 = Marshal::StringToHGlobalAnsi(strEventName);
        char* szEventName = static_cast<char*>(p1.ToPointer());
		return ::PTask::Runtime::CollectInstrumentationData(szEventName);
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Collect instrumentation data. </summary>
    ///
    /// <remarks>   crossbac, 8/21/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] If non-null, name of the event. </param>
    /// <param name="uiSamples">    [in,out] If non-null, the samples. </param>
    /// <param name="dMin">         [in,out] If non-null, the minimum. </param>
    /// <param name="dMax">         [in,out] If non-null, the maximum. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    double 
    Runtime::CollectInstrumentationData(
        String^ strEventName, 
        UINT% uiSamples, 
        double% dMin, 
        double% dMax
        )
    {
        IntPtr p1 = Marshal::StringToHGlobalAnsi(strEventName);
        char* szEventName = static_cast<char*>(p1.ToPointer());
        UINT _uiSamples;
        double _dMin;
        double _dMax;
		double dResult = ::PTask::Runtime::CollectInstrumentationData(szEventName, _uiSamples, _dMin, _dMax);
        uiSamples = _uiSamples;
        dMin = _dMin;
        dMax = _dMax;
        return dResult;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Instrumenteds this object. </summary>
    ///
    /// <remarks>	crossbac, 8/12/2013. </remarks>
    ///
    /// <returns>	true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool
    Runtime::Instrumented(void)
    {
        return ::PTask::Runtime::Instrumented() != 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Sets an instrumented. </summary>
    ///
    /// <remarks>	crossbac, 8/12/2013. </remarks>
    ///
    /// <param name="b">	true to b. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
	Runtime::SetInstrumented(bool b)
    {
        ::PTask::Runtime::SetInstrumented(b); 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Instruments. </summary>
    ///
    /// <remarks>	crossbac, 8/12/2013. </remarks>
    ///
    /// <param name="ptxFile">			  	[in,out] If non-null, the ptx file. </param>
    /// <param name="instrumentedPTXFile">	[in,out] If non-null, the instrumented ptx file. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Runtime::Instrument(
       String ^ptxFile,
       String ^instrumentedPTXFile)
    {
       
        IntPtr p1 = Marshal::StringToHGlobalAnsi(ptxFile);
        const char* ptx = static_cast<const char*>(p1.ToPointer());
        IntPtr p2 = Marshal::StringToHGlobalAnsi(instrumentedPTXFile);
        const char* instrumentedPTX = static_cast<const char*>(p2.ToPointer());
        ::PTask::Runtime::Instrument(ptx, instrumentedPTX);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Cleanup instrumentation. </summary>
    ///
    /// <remarks>	crossbac, 8/12/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Runtime::CleanupInstrumentation(void)
    {
        ::PTask::Runtime::CleanupInstrumentation();
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
    Runtime::RequireBlockPool(
        int nDataSize, 
        int nMetaSize, 
        int nTemplateSize
        )
    {
        RequireBlockPool(nDataSize, nMetaSize, nTemplateSize, 0);
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
    Runtime::RequireBlockPool(
        int nDataSize, 
        int nMetaSize, 
        int nTemplateSize, 
        int nBlocks
        )
    {
        ::PTask::Runtime::RequireBlockPool(nDataSize, nMetaSize, nTemplateSize, nBlocks);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Require block pool. </summary>
    ///
    /// <remarks>   crossbac, 8/20/2013. </remarks>
    ///
    /// <param name="pTemplate">    [in,out] If non-null, the template. </param>
    /// <param name="nBlocks">      The blocks. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Runtime::RequireBlockPool(
        DataTemplate^ pTemplate, 
        int nBlocks
        )
    {
        ::PTask::Runtime::RequireBlockPool(pTemplate->GetNativeDatablockTemplate(), nBlocks);
    }

}}}
