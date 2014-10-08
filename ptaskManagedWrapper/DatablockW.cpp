///-------------------------------------------------------------------------------------------------
// file:	DatablockW.cpp
//
// summary:	Implements the datablock wrapper class
///-------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "PTaskManagedWrapper.h"
#include <msclr\lock.h>

namespace Microsoft {
namespace Research {
namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="datablock">    [in,out] If non-null, the datablock. </param>
    ///-------------------------------------------------------------------------------------------------

    Datablock::Datablock(::PTask::Datablock* datablock)
    {
        m_nativeDatablock = datablock;
        // m_nativeDatablock->AddRef();
        m_pViewSyncAccelerator = NULL;
        m_disposed = false;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Datablock::~Datablock()
    {
        this->!Datablock();
        m_disposed = true;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Finaliser. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Datablock::!Datablock()
    {
        msclr::lock lock(this);
        if(m_nativeDatablock != NULL) {
            ::PTask::Datablock * pBlock = m_nativeDatablock;
            m_nativeDatablock = NULL;
            LONG lNewRefCount = pBlock->Release();
            if(lNewRefCount != 0) {
                ::PTask::Runtime::MandatoryInform("%s::%s() datablock release returned %d\n", 
                                                __FILE__,
                                                __FUNCTION__,
                                                lNewRefCount);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Frees this object. </summary>
    ///
    /// <remarks>   crossbac, 8/29/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void Datablock::Free() {
        CheckDisposed();
        msclr::lock lock(this);
        if(m_nativeDatablock != NULL) {
            ::PTask::Datablock * pBlock = m_nativeDatablock;
            m_nativeDatablock = NULL;
            pBlock->Release();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Frees this object. </summary>
    ///
    /// <remarks>   crossbac, 8/29/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void Datablock::Free(
        __in bool bReaderContext
        ) {
        CheckDisposed();
        msclr::lock lock(this);
        if(m_nativeDatablock != NULL) {
            ::PTask::Datablock * pBlock = m_nativeDatablock;
            m_nativeDatablock = NULL;
            LONG lNewRefCount = pBlock->Release();
            if(Runtime::GetAggressiveReleaseMode()) {
                if(lNewRefCount != 0 && bReaderContext) {
                    pBlock->Lock();
                    UINT uiBytes = static_cast<UINT>(pBlock->InvalidateDeviceViews(false, true));
                    pBlock->Unlock();
                    ::PTask::Runtime::MandatoryInform("%s::%s() datablock release returned %d...forced release of device buffers (%d bytes)\n", 
                                                    __FILE__,
                                                    __FUNCTION__,
                                                    lNewRefCount,
                                                    uiBytes);
                }
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Locks this object. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///-------------------------------------------------------------------------------------------------

    void Datablock::Lock() 
    {
        CheckDisposed();
        m_nativeDatablock->Lock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unlocks this object. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///-------------------------------------------------------------------------------------------------

    void Datablock::Unlock() 
    {
        CheckDisposed();
        m_nativeDatablock->Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Locks this object. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///-------------------------------------------------------------------------------------------------

    void Datablock::LockForViewSynchronization()
    {
        CheckDisposed();
        assert(m_pViewSyncAccelerator == NULL);
        m_pViewSyncAccelerator = m_nativeDatablock->LockForViewSync(TRUE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unlocks this object. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///-------------------------------------------------------------------------------------------------

    void Datablock::UnlockForViewSynchronization() 
    {
        CheckDisposed();
        m_nativeDatablock->UnlockForViewSync(TRUE, m_pViewSyncAccelerator);
        m_pViewSyncAccelerator = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the native datablock. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   null if it fails, else the native datablock. </returns>
    ///-------------------------------------------------------------------------------------------------

    ::PTask::Datablock*
    Datablock::GetNativeDatablock()
    {
        CheckDisposed();
        return m_nativeDatablock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a pointer to the data buffer, with intent to read. We distinguish between
    ///             readable and writeable because a readable copy has different impact on the
    ///             coherence traffic for the datablock. In essence, a writeable copy is expensive,
    ///             so it's worth asking for a readable version if reading is the only use case. If
    ///             the caller actually writes this data, it will note be reflected in other memory
    ///             spaces.
    ///             
    ///             The pointer should only be considered valid while the caller has a lock on the
    ///             datablock.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   null if it fails, else the data buffer readable. </returns>
    ///-------------------------------------------------------------------------------------------------

    byte*
    Datablock::GetDataBufferReadable()
    {
        CheckDisposed();
        return (byte*) m_nativeDatablock->GetDataPointer(FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a pointer to the data buffer, with intent to write. We distinguish between
    ///             readable and writeable because a readable copy has different impact on the
    ///             coherence traffic for the datablock. In essence, a writeable copy is expensive,
    ///             so it's worth asking for a readable version if reading is the only use case. 
    ///             
    ///             The pointer should only be considered valid while the caller has a lock on the
    ///             datablock.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   null if it fails, else the data buffer readable. </returns>
    ///-------------------------------------------------------------------------------------------------
    
    byte*
    Datablock::GetDataBufferWriteable()
    {
        CheckDisposed();
        return (byte*) m_nativeDatablock->GetDataPointer(TRUE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the data buffer size. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   The data buffer size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UInt32
    Datablock::GetDataBufferSize()
    {
        CheckDisposed();
        return m_nativeDatablock->GetDataBufferLogicalSizeBytes();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Grow data buffer. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <param name="newSize">  Size of the new. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Datablock::GrowDataBuffer(Int32 newSize)
    {
        CheckDisposed();
        m_nativeDatablock->GrowDataBuffer(newSize);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Grow buffers. </summary>
    ///
    /// <remarks>   Crossbac, 3/26/2013. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <param name="newDataSize">      Size of the new data. </param>
    /// <param name="newMetaSize">      Size of the new meta. </param>
    /// <param name="newTemplateSize">  Size of the new template. </param>
    ///-------------------------------------------------------------------------------------------------

    bool
    Datablock::GrowBuffers(
        Int32 newDataSize, 
        Int32 newMetaSize, 
        Int32 newTemplateSize
        )
    {
        CheckDisposed();
        return m_nativeDatablock->GrowBuffers(newDataSize, newMetaSize, newTemplateSize)!=0;
    }
    
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a pointer to the meta buffer, with intent to read. We distinguish between
    ///             readable and writeable because a readable copy has different impact on the
    ///             coherence traffic for the datablock. In essence, a writeable copy is expensive,
    ///             so it's worth asking for a readable version if reading is the only use case. If
    ///             the caller actually writes this data, it will note be reflected in other memory
    ///             spaces.
    ///             
    ///             If the caller is requesting a readable copy, we assume the caller is a
    ///             Dandelion consumer, and synthesize the meta data channel if it does not
    ///             already exist.
    ///             
    ///             The pointer should only be considered valid while the caller has a lock on the
    ///             datablock.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   null if it fails, else the data buffer readable. </returns>
    ///-------------------------------------------------------------------------------------------------
    /// 
    
    byte*
    Datablock::GetMetaBufferReadable()
    {
        CheckDisposed();
        m_nativeDatablock->SynthesizeMetadataFromTemplate(NULL);
        return (byte*) m_nativeDatablock->GetMetadataPointer(FALSE, TRUE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a pointer to the meta buffer, with intent to write. We distinguish between
    ///             readable and writeable because a readable copy has different impact on the
    ///             coherence traffic for the datablock. In essence, a writeable copy is expensive,
    ///             so it's worth asking for a readable version if reading is the only use case. 
    ///             
    ///             The pointer should only be considered valid while the caller has a lock on the
    ///             datablock.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   null if it fails, else the data buffer readable. </returns>
    ///-------------------------------------------------------------------------------------------------

    byte*
    Datablock::GetMetaBufferWriteable()
    {
        CheckDisposed();
        return (byte*) m_nativeDatablock->GetMetadataPointer(TRUE, FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the meta buffer size. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   The meta buffer size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UInt32
    Datablock::GetMetaBufferSize()
    {
        CheckDisposed();
        return m_nativeDatablock->GetMetaBufferLogicalSizeBytes();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Grow meta buffer. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <param name="newSize">  Size of the new. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Datablock::GrowMetaBuffer(Int32 newSize)
    {
        CheckDisposed();
        m_nativeDatablock->GrowMetaBuffer(newSize);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the template buffer. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   null if it fails, else the template buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    byte*
    Datablock::GetTemplateBuffer()
    {
        CheckDisposed();
        return (byte*)(m_nativeDatablock->GetTemplatePointer(TRUE, TRUE));
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the template buffer size. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   The template buffer size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UInt32
    Datablock::GetTemplateBufferSize()
    {
        CheckDisposed();
        return m_nativeDatablock->GetTemplateBufferLogicalSizeBytes();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Grow template buffer. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <param name="newSize">  Size of the new. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Datablock::GrowTemplateBuffer(Int32 newSize)
    {
        CheckDisposed();
        m_nativeDatablock->GrowTemplateBuffer(newSize);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Seals. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <param name="rCount">       Number of. </param>
    /// <param name="dataSize">     Size of the data. </param>
    /// <param name="metaSize">     Size of the meta. </param>
    /// <param name="templateSize"> Size of the template. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Datablock::Seal(Int32 rCount, Int32 dataSize, Int32 metaSize, Int32 templateSize)
    {
        CheckDisposed();
        m_nativeDatablock->Lock();
        m_nativeDatablock->Seal(rCount, dataSize, metaSize, templateSize);
        m_nativeDatablock->Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the error code. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   The error code. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned __int64
    Datablock::GetControlSignals()
    {
        CheckDisposed();
        m_nativeDatablock->Lock();
        unsigned __int64 code = m_nativeDatablock->GetControlSignals();
        m_nativeDatablock->Unlock();
        return code;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the error code. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   The error code. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool
    Datablock::TestControlSignal(
        unsigned __int64 luiSignal
        )
    {
        CheckDisposed();
        m_nativeDatablock->Lock();
        ::PTask::CONTROLSIGNAL luiCode = (::PTask::CONTROLSIGNAL) luiSignal;
        bool bResult = (m_nativeDatablock->TestControlSignal(luiCode) != 0);
        m_nativeDatablock->Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the number of records. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   The number of records. </returns>
    ///-------------------------------------------------------------------------------------------------

    int
    Datablock::GetNumOfRecords()
    {
        CheckDisposed();
        m_nativeDatablock->Lock();
        int records = m_nativeDatablock->GetRecordCount();
        m_nativeDatablock->Unlock();
        return records;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a block control code. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <param name="ui">   The user interface. </param>
    ///-------------------------------------------------------------------------------------------------

	void
	Datablock::SetBlockControlCode(
        unsigned __int64 ui
        )
	{
        CheckDisposed();
        m_nativeDatablock->Lock();
        m_nativeDatablock->SetControlSignal(ui);
		m_nativeDatablock->Unlock();
	}

    unsigned int 
    Datablock::GetAccessFlags(
        VOID
        )
    {
        CheckDisposed();
        m_nativeDatablock->Lock();
        unsigned int flags = m_nativeDatablock->GetAccessFlags();
		m_nativeDatablock->Unlock();
        return flags;
    }

    void 
    Datablock::SetAccessFlags(
        unsigned int flags
        )
    {
        CheckDisposed();
        m_nativeDatablock->Lock();
        m_nativeDatablock->SetAccessFlags(flags);
		m_nativeDatablock->Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check disposed. </summary>
    ///
    /// <remarks>   crossbac </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///-------------------------------------------------------------------------------------------------

    void
    Datablock::CheckDisposed(
        void
        ) 
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Datablock already disposed");
    }

}}}
