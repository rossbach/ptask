///-------------------------------------------------------------------------------------------------
// file:	DataTemplate.h
//
// summary:	Declares the data template class, which wraps the PTask DatablockTemplate class
///-------------------------------------------------------------------------------------------------

#pragma once
#pragma unmanaged
#include "ptaskapi.h"
#pragma managed
#include "CompiledKernelW.h"
#include "DatablockW.h"
#include "DataTemplate.h"
using namespace System;
using namespace System::Runtime::InteropServices; // For marshalling helpers

namespace Microsoft {
namespace Research {
namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Data template. </summary>
    ///
    /// <remarks>   refactored from code by jcurrey by crossbac, 4/13/2012. 
    ///             Follows the C++/CLI IDisposable pattern described at 
    ///             http://msdn.microsoft.com/en-us/library/system.idisposable.aspx
    /// 			</remarks>
    ///-------------------------------------------------------------------------------------------------

    public ref class DataTemplate
    {
    public:
        enum class PTASK_PARM_TYPE {
            PTPARM_INT = 0,
            PTPARM_FLOAT = 1,
            PTPARM_DOUBLE = 2,      
            PTPARM_BYVALSTRUCT = 3,  // structs passed by value
            PTPARM_BYREFSTRUCT = 4,  // structs passed by ref are same as buffer
            PTPARM_BUFFER = 5,       
            PTPARM_NONE = 6
            // etc....
            // TODO: XXXX: FILL IN!
        };
        DataTemplate(::PTask::DatablockTemplate* datablockTemplate);
        ~DataTemplate(); // IDisposable
        !DataTemplate(); // finalizer
        ::PTask::DatablockTemplate* GetNativeDatablockTemplate();
        unsigned int GetDatablockByteCount();
        void SetInitialValue(byte *buffer, unsigned int count, unsigned int records, bool bExplicitlyEmpty);
        byte * GetInitialValue();
        unsigned int GetInitialValueSize();

    private:
        ::PTask::DatablockTemplate* m_nativeDatablockTemplate;
        bool                      m_disposed;
    };

}}}
