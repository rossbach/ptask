//--------------------------------------------------------------------------------------
// File: PHBuffer.cpp
// Maintainer: crossbac@microsoft.com
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "PHBuffer.h"
#include "hostaccelerator.h"
#include "datablock.h"
#include <assert.h>

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pParent">  [in,out] If non-null, the parent. </param>
    /// <param name="f">        The f. </param>
    /// <param name="p">        [in,out] If non-null, the p. </param>
    /// <param name="uiUID">    (optional) the uid. </param>
    ///-------------------------------------------------------------------------------------------------

    PHBuffer::PHBuffer(
        Datablock * pParent,
        BUFFERACCESSFLAGS flags, 
        UINT nChannelIndex,
        Accelerator * pAccelerator, 
        Accelerator * pAllocAccelerator,
        UINT uiUID
        ) : PBuffer(pParent, flags, nChannelIndex, pAccelerator, pAllocAccelerator, uiUID)
    {
        m_pBuffer = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    PHBuffer::~PHBuffer(
        VOID
        )
    {
        if(m_pBuffer) {
            if(m_pAllocatingAccelerator != NULL && 
                m_pAllocatingAccelerator->SupportsPinnedHostMemory()) {
                m_pAllocatingAccelerator->FreeHostMemory(m_pBuffer, m_bPhysicalBufferPinned);
            } else if(m_pAccelerator != NULL) {
                assert(!m_bPhysicalBufferPinned);
                m_pAccelerator->FreeHostMemory(m_pBuffer, FALSE);
            } else {
                free(m_pBuffer);
            }
            m_pBuffer = NULL;
        }
        m_mapBindableObjects.clear();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes a device-side buffer that is expected to be bound to immutable device
    ///             resources (i.e. those in constant memory).
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/4/2012. </remarks>
    ///
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="uiBufferSizeBytes">    The buffer size in bytes. </param>
    /// <param name="pExtent">              (optional) [in] If non-null, the initial buffer contents. </param>
    /// <param name="strDebugBufferName">   (optional) [in] If non-null, a name to assign to the
    ///                                     buffer which will be used to label runtime- specific
    ///                                     objects to aid in debugging. Ignored on release builds. </param>
    /// <param name="bByteAddressable">     (optional) true if the buffer should be byte addressable. </param>
    ///
    /// <returns>   PTRESULT (use PTSUCCESS/PTFAILED macros) </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT	
    PHBuffer::CreateImmutableBuffer(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferSizeBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __in char * strDebugBufferName, 
        __in bool bByteAddressable
        )
    {
        UNREFERENCED_PARAMETER(pAsyncContext);
        assert(m_pBuffer == NULL);
        
        // finalize dimensions can return 0. In such a case we must still allocate something, even if
        // it's empty to make it bindable to device-side resources. Hence, we distinguish between alloc
        // size and transfer size.         
        FinalizeDimensions(bByteAddressable, uiBufferSizeBytes);
        UINT uAllocCount = GetAllocationExtentBytes();
        UINT uTransferBytes = GetLogicalExtentBytes();

        m_pBuffer = MemorySpace::AllocateMemoryExtent(HOST_MEMORY_SPACE_ID, uAllocCount, 0);

        assert(m_pBuffer != NULL);
        VOID* pInitialBufferContents = pExtent ? pExtent->lpvAddress : NULL;
        UINT uiInitialContentsSizeBytes = pExtent ? pExtent->uiSizeBytes : 0;
        if(pInitialBufferContents && m_pBuffer != pInitialBufferContents) {

            // If the user gave us a size for the initial data buffer, make sure we don't overrun it or or
            // the newly allocated buffer. If the size parameter was defaulted (0), it means we were
            // supposed to infer the size, so uCount is the best we can do. 

            uTransferBytes = uiInitialContentsSizeBytes ? 
                                min(uiInitialContentsSizeBytes, uTransferBytes) : 
                                uTransferBytes;

            memcpy(m_pBuffer, pInitialBufferContents, uTransferBytes);
            m_bPopulated = TRUE;
        }

        return CreateBindableObjects(strDebugBufferName);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates immutable bindable objects. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="szname">   [in,out] (optional)  If non-null, the szname. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    PHBuffer::CreateBindableObjectsImmutable(
        char * szname
        )
    {
        UNREFERENCED_PARAMETER(szname);
        assert(m_pBuffer);
        BINDABLEOBJECT view;
        view.vptr = m_pBuffer;
        m_mapBindableObjects[BVT_ACCELERATOR_IMMUTABLE] = view;
        return PTASK_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates readable bindable objects. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="szname">   [in,out] (optional)  If non-null, the szname. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    PHBuffer::CreateBindableObjectsReadable(
        char * szname
        )
    {
        UNREFERENCED_PARAMETER(szname);
        assert(m_pBuffer);
        BINDABLEOBJECT view;
        view.vptr = m_pBuffer;
        m_mapBindableObjects[BVT_ACCELERATOR_READABLE] = view;
        return PTASK_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Create writeable bindable objects. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="szname">   [in,out] (optional)  If non-null, the szname. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    PHBuffer::CreateBindableObjectsWriteable(
        char * szname
        )
    {
        UNREFERENCED_PARAMETER(szname);
        assert(m_pBuffer);
        BINDABLEOBJECT view;
        view.vptr = m_pBuffer;
        m_mapBindableObjects[BVT_ACCELERATOR_WRITEABLE] = view;
        return PTASK_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize host view. </summary>
    ///
    /// <remarks>   crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAsyncContext">        [in,out] If non-null, information describing the lpv. </param>
    /// <param name="pExtent">              [in,out] The data. </param>
    /// <param name="bForceSynchronous">    (optional) the elide synchronization. </param>
    /// <param name="bRequestOutstanding">  [in,out] The request outstanding. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT		
    PHBuffer::__populateHostView(
        __in  AsyncContext * pAsyncContext,
        __in  HOSTMEMORYEXTENT * pExtent,
        __in  BOOL bForceSynchronous,
        __out BOOL &bRequestOutstanding
        )
    {
        // there is nothing to be done here--either we have a host buffer or we don't have one. 
        UNREFERENCED_PARAMETER(bForceSynchronous);
        UNREFERENCED_PARAMETER(pExtent);
        UNREFERENCED_PARAMETER(pAsyncContext);
        assert(m_pBuffer != NULL);
        UINT cbBuffer = GetElementCount()*m_vDimensionsFinalized.cbElementStride;
        assert(m_vDimensionsFinalized.AllocationSizeBytes() <= cbBuffer);
        bRequestOutstanding = FALSE;
        return (UINT) cbBuffer;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize mutable accelerator view. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="uiBufferSizeBytes">    If non-null, the data. </param>
    /// <param name="pExtent">              [in,out] The bytes. </param>
    /// <param name="pModule">              [in,out] (optional)  If non-null, the module. </param>
    /// <param name="lpszBinding">          (optional) the binding. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

	UINT		
    PHBuffer::__populateMutableAcceleratorView(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferSizeBytes,
        __in HOSTMEMORYEXTENT * pExtent, 
        __out BOOL&             bOutstanding,
        __in void * pModule,
        __in const char * lpszBinding
        )
    {
        UNREFERENCED_PARAMETER(pModule);
        UNREFERENCED_PARAMETER(lpszBinding);
        UNREFERENCED_PARAMETER(pAsyncContext);
        assert(m_pBuffer);
        UINT cbCopy = 0;
        bOutstanding = FALSE;
        void * pInitialData = pExtent ? pExtent->lpvAddress : NULL;
        UINT uiInitialDataBytes = pExtent ? pExtent->uiSizeBytes : 0;
        if(pInitialData != m_pBuffer) {
            cbCopy = uiInitialDataBytes;
            if(uiBufferSizeBytes != PBUFFER_DEFAULT_SIZE &&  uiBufferSizeBytes < cbCopy) 
                cbCopy = uiBufferSizeBytes;
            memcpy(m_pBuffer, pInitialData, cbCopy);
        }
        return cbCopy;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize immmutable accelerator view. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="uiBufferSizeBytes">    The buffer size in bytes. </param>
    /// <param name="pExtent">              [in,out] If non-null, the data. </param>
    /// <param name="pModule">              [in,out] (optional)  If non-null, the module. </param>
    /// <param name="lpszBinding">          (optional) the binding. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

	UINT		
    PHBuffer::__populateImmutableAcceleratorView(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferSizeBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __out BOOL&             bOutstanding,
        __in void * pModule,
        __in const char * lpszBinding
        )
    {
        UNREFERENCED_PARAMETER(pModule);
        UNREFERENCED_PARAMETER(lpszBinding);
        UNREFERENCED_PARAMETER(pAsyncContext);
        assert(m_pBuffer);
        bOutstanding = FALSE;
        UINT cbCopy = 0;
        void * pInitialData = pExtent ? pExtent->lpvAddress : NULL;
        UINT uiInitialDataBytes = pExtent ? pExtent->uiSizeBytes : 0;
        if(pInitialData != m_pBuffer) {
            cbCopy = uiInitialDataBytes;
            if(uiBufferSizeBytes != PBUFFER_DEFAULT_SIZE &&  uiBufferSizeBytes < cbCopy) 
                cbCopy = uiBufferSizeBytes;
            memcpy(m_pBuffer, pInitialData, cbCopy);
        }
        return cbCopy;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Force synchronize. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void                
    PHBuffer::ForceSynchronize(
        VOID
        )
    {
        assert(FALSE);
        PTask::Runtime::HandleError("%s called! What gives?\n", __FUNCTION__);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes a device-side buffer that is expected to be bound to mutable device
    ///             resources (not in constant memory).
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/4/2012. </remarks>
    ///
    /// <param name="pAsyncContext">        [in,out] (optional)  If non-null, context for the
    ///                                     asynchronous. </param>
    /// <param name="uiBufferSizeBytes">    The buffer size in bytes. </param>
    /// <param name="pExtent">              (optional) [in] If non-null, the initial buffer contents. </param>
    /// <param name="strDebugBufferName">   (optional) [in] If non-null, a name to assign to the
    ///                                     buffer which will be used to label runtime- specific
    ///                                     objects to aid in debugging. Ignored on release builds. </param>
    /// <param name="bByteAddressable">     (optional) true if the buffer should be byte addressable. </param>
    ///
    /// <returns>   PTRESULT (use PTSUCCESS/PTFAILED macros) </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT	
    PHBuffer::CreateMutableBuffer(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferSizeBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __in char * strDebugBufferName, 
        __in bool bByteAddressable                                                    
        ) {

        return InitializeBuffer(pAsyncContext, 
                                uiBufferSizeBytes,
                                pExtent,
                                strDebugBufferName,
                                bByteAddressable);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes a host buffer. Since most of the views created by required overrides
    ///             in PBuffer are meaningless in host memory (e.g. immutability)
    ///             we provide one routine to create buffers, and map all the required overrides to
    ///             it.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/4/2012. </remarks>
    ///
    /// <param name="pAsyncContext">        [in,out] (optional)  If non-null, context for the
    ///                                     asynchronous. </param>
    /// <param name="uiBufferSizeBytes">    The buffer size in bytes. </param>
    /// <param name="pExtent">              (optional) [in] If non-null, the initial buffer contents. </param>
    /// <param name="szname">               (optional) [in] If non-null, a name to assign to the
    ///                                     buffer which will be used to label runtime- specific objects
    ///                                     to aid in debugging. Ignored on release builds. </param>
    /// <param name="bByteAddressable">     (optional) true if the buffer should be byte addressable. </param>
    ///
    /// <returns>   PTRESULT (use PTSUCCESS/PTFAILED macros) </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT	
    PHBuffer::InitializeBuffer(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferSizeBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __in char * szname, 
        __in bool bByteAddressable                                                    
        )
    {
        assert(m_pBuffer == NULL);
        UNREFERENCED_PARAMETER(bByteAddressable);
        UNREFERENCED_PARAMETER(pAsyncContext);
        FinalizeDimensions(bByteAddressable, uiBufferSizeBytes);
        UINT uAllocBytes = GetAllocationExtentBytes();
        UINT uContentBytes = GetLogicalExtentBytes();

        if(m_pAllocatingAccelerator && 
           m_pAllocatingAccelerator->SupportsPinnedHostMemory() &&
           (m_bPinnedBufferRequested || PTask::Runtime::GetAggressivePageLocking())) {
            m_pBuffer = m_pAllocatingAccelerator->AllocatePagelockedHostMemory(uAllocBytes, &m_bPhysicalBufferPinned);
        } else {
            m_pBuffer = MemorySpace::AllocateMemoryExtent(HOST_MEMORY_SPACE_ID, uAllocBytes, 0);
        } 

        assert(m_pBuffer != NULL);
        VOID* pInitData = pExtent ? pExtent->lpvAddress : NULL;
        UINT cbInitData = pExtent ? pExtent->uiSizeBytes : 0;
        if(pInitData) {
            // If the user gave us a size for the initial data buffer, 
            // make sure we don't overrun it or or the newly allocated buffer.
            // If the size parameter was defaulted (0), it means we were
            // supposed to infer the size, so uCount is the best we can do.
            UINT uTransferBytes = cbInitData ? min(cbInitData, uContentBytes) : uContentBytes;
            memcpy(m_pBuffer, pInitData, uTransferBytes);
            m_bPopulated = TRUE;
        }

        return CreateBindableObjects(szname);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Finalize the dimensions of the device buffer that will be created to back this
    ///             PHBuffer. We specialize the host buffer implementation to not
    ///             require the block to be sealed to allocate buffers!
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/4/2012. </remarks>
    ///
    /// <param name="bByteAddressable">     [out] (optional) true if the buffer should be byte
    ///                                     addressable. </param>
    /// <param name="uiBufferSizeBytes">    (optional) the buffer size in bytes. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT        
    PHBuffer::FinalizeDimensions(
        __out bool &bByteAddressable,
        __in UINT uiBufferSizeBytes
        )
    {
        return PBuffer::FinalizeDimensions(bByteAddressable, uiBufferSizeBytes, FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return true if the derived class supports a memset API. </summary>
    ///
    /// <remarks>   crossbac, 8/14/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    PHBuffer::SupportsMemset(
        VOID
        )
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   memset. </summary>
    ///
    /// <remarks>   crossbac, 8/14/2013. </remarks>
    ///
    /// <param name="nValue">           The value. </param>
    /// <param name="szExtentBytes">    The extent in bytes. </param>
    ///
    /// <returns>   the number of bytes set </returns>
    ///-------------------------------------------------------------------------------------------------

    size_t       
    PHBuffer::FillExtent(
        int nValue, 
        size_t szExtentBytes
        )
    {      
        memset(m_pBuffer, nValue, szExtentBytes);
        return szExtentBytes;
    }

};

