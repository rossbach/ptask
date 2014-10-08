///-------------------------------------------------------------------------------------------------
// file:	DatablockW.h
//
// summary:	Declares the datablock wrapper class
///-------------------------------------------------------------------------------------------------

#pragma once
#pragma unmanaged
#include "ptaskapi.h"
#pragma managed
#include "CompiledKernelW.h"
using namespace System;
using namespace System::Runtime::InteropServices; // For marshalling helpers

namespace Microsoft {
namespace Research {
namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Datablock. </summary>
    ///
    /// <remarks>   refactored from code by jcurrey by crossbac, 4/13/2012. 
    ///             Follows the C++/CLI IDisposable pattern described at 
    ///             http://msdn.microsoft.com/en-us/library/system.idisposable.aspx
    /// 			</remarks>
    ///-------------------------------------------------------------------------------------------------

    public ref class Datablock : public IDisposable
    {
    public:

        /// <summary> The datablock carries no control signal </summary>
        static const unsigned __int64 DBCTLC_NONE = 0x00000000; 

        /// <summary>   Code indicating beginning of stream. </summary>
        static const unsigned __int64 DBCTLC_BOF = 0x1;
    
        /// <summary> The datablock with this code is the last in a stream </summary>
        static const unsigned __int64 DBCTLC_EOF = 0x2;

        /// <summary> a block carrying this signal is the first block
        /// 		  in an iteration run. 
        /// 		  </summary>
        static const unsigned __int64 DBCTLC_BEGINITERATION = 0x4;

        /// <summary> a block carrying this signal is the final block
        /// 		  in an iteration run. 
        /// 		  </summary>
        static const unsigned __int64 DBCTLC_ENDITERATION = 0x8;

        /// <summary> A runtime error has occurred, and the datablock carrying 
        /// 		  this code is propagating that fact to downstream tasks.
        /// 		  </summary>
        static const unsigned __int64 DBCTLC_ERR = 0x10;
  
        /// <summary>   Use this code to intentionally create outputs
        /// 			with values that will never pass the predicate. 
        /// 			Think: "myDataSource > /dev/null" . </summary>
        static const unsigned __int64 DBCTLC_DEVNULL = 0x20;

        /// <summary>   Flags that mean use default permissions for datablock access. </summary>
        static const unsigned int PT_ACCESS_DEFAULT = 0x0;

        /// <summary> The datablock or buffer will be read from host code</summary>
        static const unsigned int PT_ACCESS_HOST_READ = 0x1;
    
        /// <summary> The datablock or buffer will be written from host code </summary>
        static const unsigned int PT_ACCESS_HOST_WRITE = 0x2;
    
        /// <summary> The datablock or buffer will be read from accelerator code </summary>
        static const unsigned int PT_ACCESS_ACCELERATOR_READ = 0x4;
    
        /// <summary> The datablock or buffer will be written from accelerator code </summary>
        static const unsigned int PT_ACCESS_ACCELERATOR_WRITE = 0x8;
    
        /// <summary> The datablock or buffer should be bound to constant memory on the accelerator  </summary>
        static const unsigned int PT_ACCESS_IMMUTABLE = 0x10;
    
        /// <summary> The datablock will be accessed at byte-granularity by accelerator code </summary>
        static const unsigned int PT_ACCESS_BYTE_ADDRESSABLE = 0x20;

        static const unsigned int DF_SIZE = 0;
        static const unsigned int DF_EOF = 1;
        static const unsigned int DF_BOF = 2;
		static const unsigned int DF_METADATA_SPLITTER = 3;
        static const unsigned int DF_CONTROL_CODE = 4;
		static const unsigned int DF_BLOCK_UID = 5;
        static const unsigned int DF_DATA_SOURCE = 6;
        static const unsigned int DF_METADATA_SOURCE = 7;
        static const unsigned int DF_TEMPLATE_SOURCE = 8;

        Datablock(::PTask::Datablock* datablock);
        ~Datablock(); // IDisposable
        !Datablock(); // finalizer

        ::PTask::Datablock* GetNativeDatablock();
        void Lock();
        void Unlock();
        void LockForViewSynchronization();
        void UnlockForViewSynchronization();
        byte* GetDataBufferReadable();
        byte* GetDataBufferWriteable();
        UInt32 GetDataBufferSize();
        void GrowDataBuffer(Int32 newSize);
        byte* GetMetaBufferWriteable();
        byte* GetMetaBufferReadable();
        UInt32 GetMetaBufferSize();
        void GrowMetaBuffer(Int32 newSize);
        byte* GetTemplateBuffer();
        UInt32 GetTemplateBufferSize();
        void GrowTemplateBuffer(Int32 newSize);
        bool GrowBuffers(Int32 newDataSize, Int32 newMetaSize, Int32 newTemplateSize);
        void Seal(Int32 rCount, Int32 dataSize, Int32 metaSize, Int32 templateSize);
        bool TestControlSignal(unsigned __int64 luiSignal);
        unsigned __int64 GetControlSignals();
        int GetNumOfRecords();
        void SetBlockControlCode(unsigned __int64 ui);
        unsigned int GetAccessFlags();
        void SetAccessFlags(unsigned int flags);
        void Free();
        void Free(bool bReaderContext);

    private:
        ::PTask::Datablock*    m_nativeDatablock;
        bool                   m_disposed;
        ::PTask::Accelerator * m_pViewSyncAccelerator;
        void                   CheckDisposed();

    };
}}}

