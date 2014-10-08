///-------------------------------------------------------------------------------------------------
// file:	CompiledKernelW.cpp
//
// summary:	Implements the compiled kernel class
///-------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "PTaskManagedWrapper.h"

namespace Microsoft {
namespace Research {
namespace PTask {

    CompiledKernel::CompiledKernel(
        ::PTask::CompiledKernel* compiledKernel
        )
    {
        m_nativeCompiledKernel = compiledKernel;
        m_disposed = false;
    }

    CompiledKernel::~CompiledKernel()
    {
        this->!CompiledKernel();
        m_disposed = true;
    }

    CompiledKernel::!CompiledKernel()
    {
        // Do NOT delete m_nativeCompiledKernel here, since Runtime::Terminate 
        // deletes all instances that it has vended.
        m_nativeCompiledKernel = NULL;
    }

    ::PTask::CompiledKernel*
    CompiledKernel::GetNativeCompiledKernel()
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Task already disposed");

        return m_nativeCompiledKernel;
    }
}}}