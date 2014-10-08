///-------------------------------------------------------------------------------------------------
// file:	CompiledKernelW.h
//
// summary:	Declares the compiled kernel class, which wraps the ptask class of the same
//          name.  
///-------------------------------------------------------------------------------------------------

#pragma once
#pragma unmanaged
#include "ptaskapi.h"
#pragma managed
using namespace System;
using namespace System::Runtime::InteropServices; // For marshalling helpers

namespace Microsoft {
namespace Research {
namespace PTask {

    public enum class ACCELERATOR_CLASS {

        /// <summary> Accelerator based on DirectX 11 backend </summary>
        ACCELERATOR_CLASS_DIRECT_X = 0,   

        /// <summary> Accelerator based on OpenCL backend </summary>
        ACCELERATOR_CLASS_OPEN_CL = 1,
        
        /// <summary> Accelerator based on CUDA backend </summary>
        ACCELERATOR_CLASS_CUDA = 2,
        
        /// <summary> Accelerator based on DirectX reference driver. 
        /// 		  This is a software-emulated implemention of DX11
        /// 		  hardware, so should not be used in a deployed environment.
        /// 		  We include it to enable debugging on machines without
        /// 		  proper hardware. See Runtime::SetUseReferenceDrivers()
        /// 		   </summary>
        ACCELERATOR_CLASS_REFERENCE = 3,
        
        /// <summary> Host-based accelerator. Technically, not an accelerator,
        /// 		  but a computation resource. Enables PTasks that run 
        /// 		  on the CPU to become part of a ptask graph. </summary>
        ACCELERATOR_CLASS_HOST = 4,
        
        /// <summary> Unknown accelerator type. Should never be used. </summary>
        ACCELERATOR_CLASS_UNKNOWN = 5

    };

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Compiled kernel wrapper. </summary>
    ///
    /// <remarks>   Follows the C++/CLI IDisposable pattern described at 
    ///             http://msdn.microsoft.com/en-us/library/system.idisposable.aspx
    /// 			jcurrey, maintainer </remarks>
    ///-------------------------------------------------------------------------------------------------

    public ref class CompiledKernel 
    {
    public:
        CompiledKernel(::PTask::CompiledKernel* compiledKernel);
        ~CompiledKernel(); // IDisposable
        !CompiledKernel(); // finalizer
        ::PTask::CompiledKernel* GetNativeCompiledKernel();
    private:
        ::PTask::CompiledKernel* m_nativeCompiledKernel;
        bool                   m_disposed;
    };
}}}