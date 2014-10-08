//--------------------------------------------------------------------------------------
// File: PDXBuffer.cpp
// Maintainer: crossbac@microsoft.com
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "PDXBuffer.h"
#include <stdio.h>
#include <crtdbg.h>
#include "ptdxhdr.h"
#include <assert.h>
#include "accelerator.h"
#include "PTaskRuntime.h"
#include "nvtxmacros.h"
#include "AsyncContext.h"
#include "dxaccelerator.h"
#include <iomanip>

using namespace std;

#if defined (DEBUG) && defined (DIRECTXCOMPILESUPPORT)
#define SET_DEBUG_OBJECT_NAME(x,y) \
    if((y)!=NULL) { (x)->SetPrivateData(WKPDID_D3DDebugObjectName, (UINT)(strlen(y)+1), (y)); }
#define SET_DEBUG_OBJECT_NAME_RAW(x,y) \
    (x)->SetPrivateData(WKPDID_D3DDebugObjectName, (UINT)(strlen(y)+1), (y))
#else
#define SET_DEBUG_OBJECT_NAME(x,y) UNREFERENCED_PARAMETER(y)
#define SET_DEBUG_OBJECT_NAME_RAW(x,y) 
#endif

#define PLATFORMDEVICE(x) reinterpret_cast<ID3D11Device*>(x)
#define PLATFORMCONTEXT(x) reinterpret_cast<ID3D11DeviceContext*>(x)
#define PLATFORMBUFFER(x) reinterpret_cast<ID3D11Buffer*>(x)
#define PLATFORMSHRBUFFER(x) reinterpret_cast<IDXGIKeyedMutex*>(x)
#define PLATFORMBUFFERPTR(x) reinterpret_cast<ID3D11Buffer**>(x)
#ifndef SAFE_TYPED_RELEASE
#define SAFE_TYPED_RELEASE(t,p)      { if (p) { (reinterpret_cast<t*>(p))->Release(); (p)=NULL; } }
#endif
#define ACQUIRE_CTX(acc)                             \
        MARKRANGEENTER(L"PDXBuf-acqLock");           \
        acc->Lock();                                 \
        BOOL bFCC = !acc->IsDeviceContextCurrent();  \
        if(bFCC) acc->MakeDeviceContextCurrent();    \
        MARKRANGEEXIT(); 
#define RELEASE_CTX(acc)                             \
        if(bFCC) acc->ReleaseCurrentDeviceContext(); \
        acc->Unlock(); 

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pParent">          [in,out] If non-null, the parent. </param>
    /// <param name="f">                The f. </param>
    /// <param name="nChannelIndex">    Zero-based index of the datablock channel this PBuffer is
    ///                                 backing. </param>
    /// <param name="p">                [in,out] If non-null, the p. </param>
    /// <param name="pAllocator">       [in,out] If non-null, the allocator. </param>
    /// <param name="uiUID">            The uid. </param>
    ///-------------------------------------------------------------------------------------------------

    PDXBuffer::PDXBuffer(
        Datablock * pParent,
        BUFFERACCESSFLAGS f,
        UINT nChannelIndex,
        Accelerator * p, 
        Accelerator * pAllocator,
        UINT uiUID
        ) : PBuffer(pParent, f, nChannelIndex, p, pAllocator, uiUID)
    {
        m_pStageBuffer = NULL;
        m_pOutstandingOpBuffer = NULL;
        m_bHtoDStagePopulated = FALSE;
        m_bDtoHStagePopulated = FALSE;
        m_pOutstandingDtoHTarget = NULL;
        m_pOutstandingHtoDTarget = NULL;
        m_pOutstandingQuery = NULL;
        m_bP2PShareable = FALSE;
        m_pDXGIKeyedMutex = NULL;
        m_pDXGIResource = NULL;
        m_hDXGIHandle = INVALID_HANDLE_VALUE;
        m_bP2PLocked = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    PDXBuffer::~PDXBuffer(
        VOID
        )
    {
        // note that we don't need to release any additional
        // objects for immutable views, since they just use
        // the ID3D11Buffer* in m_pBuffer directly.
        if(m_mapBindableObjects.find(BVT_ACCELERATOR_READABLE) != m_mapBindableObjects.end()) {
            BINDABLEOBJECT view = m_mapBindableObjects[BVT_ACCELERATOR_READABLE];
            SAFE_TYPED_RELEASE(ID3D11ShaderResourceView, view.psrv);
        }
        if(m_mapBindableObjects.find(BVT_ACCELERATOR_WRITEABLE) != m_mapBindableObjects.end()) {
            BINDABLEOBJECT view = m_mapBindableObjects[BVT_ACCELERATOR_WRITEABLE];
            SAFE_TYPED_RELEASE(ID3D11UnorderedAccessView, view.puav);
        }

        if(m_pDXGIKeyedMutex) m_pDXGIKeyedMutex->Release();
        if(m_pDXGIResource) m_pDXGIResource->Release();
        if(m_pStageBuffer) m_pStageBuffer->Release();
        if(m_pOutstandingQuery) m_pOutstandingQuery->Release();
        if(m_pOutstandingDtoHTarget) delete m_pOutstandingDtoHTarget;
        if(m_pBuffer) {
            PLATFORMBUFFER(m_pBuffer)->Release();
            m_pBuffer = NULL;
        }
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
    /// <param name="strDebugObjectName">   (optional) [in] If non-null, a name to assign to the
    ///                                     buffer which will be used to label runtime- specific
    ///                                     objects to aid in debugging. Ignored on release builds. </param>
    /// <param name="bByteAddressable">     (optional) true if the buffer should be byte addressable. </param>
    ///
    /// <returns>   PTRESULT (use PTSUCCESS/PTFAILED macros) </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT	
    PDXBuffer::CreateImmutableBuffer(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferSizeBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __in char * strDebugObjectName, 
        __in bool bByteAddressable
        )
    {
        // we don't do anything special for asynchrony with
        // the DX11 backend yet. 
        assert(m_pBuffer == NULL);
        UNREFERENCED_PARAMETER(pAsyncContext);

        // finalize dimensions can return 0. In such a case we must still allocate something, even if
        // it's empty to make it bindable to device-side resources. Hence, we distinguish between alloc
        // size and transfer size.         
        FinalizeDimensions(bByteAddressable, uiBufferSizeBytes);
        UINT uAllocCount = GetAllocationExtentBytes();
        UINT uTransferBytes = GetLogicalExtentBytes();

        VOID* pInitialBufferContents = pExtent ? pExtent->lpvAddress : NULL;
        UINT uiInitialContentsSizeBytes = pExtent ? pExtent->uiSizeBytes : 0;

        // If the user gave us a size for the initial data buffer, make sure we don't overrun it or or
        // the newly allocated buffer. If the size parameter was defaulted (0), it means we were
        // supposed to infer the size, so uCount is the best we can do. 
        
        UINT uCopyBytes = uiInitialContentsSizeBytes ? 
                            min(uiInitialContentsSizeBytes, uAllocCount) : 
                            uAllocCount;

        // constant buffers must have size % 16 == 0. we don't want to impose that restriction on the
        // caller, so use a temporary buffer that meets the requirements if necesary, doctoring the
        // actual elements to fit the adjusted size. 
        
        UINT pad = (uAllocCount % 16)?(16-(uAllocCount%16)):0;
        BOOL bPadConstBuf = pad > 0;
        VOID * pPaddedBuffer = NULL;
        VOID * pInitPtr = pInitialBufferContents;
        if(bPadConstBuf) {
            pPaddedBuffer = malloc(uAllocCount+pad);
            memset(pPaddedBuffer, 0, uAllocCount+pad);
            memcpy(pPaddedBuffer, pInitialBufferContents, uCopyBytes);
            pInitPtr = pPaddedBuffer;
            uTransferBytes = uCopyBytes + pad;
        }
        m_vDimensionsFinalized.cbElementStride = uAllocCount+pad; 
        m_vDimensionsFinalized.uiXElements = ((uAllocCount+pad)/m_vDimensionsFinalized.cbElementStride);
        m_vDimensionsFinalized.uiYElements = 1;
        m_vDimensionsFinalized.uiZElements = 1;

        // initialize a buffer descriptor for DX API call
        D3D11_BUFFER_DESC desc;
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        desc.MiscFlags = 0;    
        desc.ByteWidth = bPadConstBuf?(uAllocCount+pad):uAllocCount;
        desc.StructureByteStride = desc.ByteWidth;
        ID3D11Buffer *pConstantBuffer = NULL;

        HRESULT hr = S_OK;
        if(pInitialBufferContents != NULL && uTransferBytes != 0) {
            assert(desc.ByteWidth <= uiInitialContentsSizeBytes || bPadConstBuf);
            D3D11_SUBRESOURCE_DATA InitData;
            memset(&InitData, 0, sizeof(D3D11_SUBRESOURCE_DATA));
            InitData.pSysMem = pInitPtr;
            m_pAccelerator->Lock();
            ID3D11Device* pDevice = (ID3D11Device*) m_pAccelerator->GetDevice();
            hr = pDevice->CreateBuffer( &desc, &InitData, &pConstantBuffer );
            m_pAccelerator->Unlock();
        } else {
            m_pAccelerator->Lock();
            ID3D11Device* pDevice = (ID3D11Device*) m_pAccelerator->GetDevice();
            hr = pDevice->CreateBuffer( &desc, NULL, &pConstantBuffer );
            m_pAccelerator->Unlock();
        }
        
        // if we had to allocate something with 16-byte
        // alignment to do the creation, free it. 
        if(pPaddedBuffer) free(pPaddedBuffer);

        if(SUCCEEDED(hr)) {
            m_pBuffer = pConstantBuffer;
            SET_DEBUG_OBJECT_NAME(pConstantBuffer, strDebugObjectName);            
            return CreateBindableObjects(strDebugObjectName);
        }

        // the create failed. sigh.
        PTask::Runtime::HandleError("%s failed.\n", __FUNCTION__);
        m_pBuffer = NULL;
        return PTASK_ERR;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates immutable bindable objects (ID3D11Buffers). </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="szname">   [in] If non-null, the a string used to label the object that can be
    ///                         used for debugging. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    PDXBuffer::CreateBindableObjectsImmutable(
        char * szname
        )
    {
        UNREFERENCED_PARAMETER(szname);
        BINDABLEOBJECT view;
        view.pconst = PLATFORMBUFFER(m_pBuffer);
        assert(m_mapBindableObjects.find(BVT_ACCELERATOR_IMMUTABLE) == m_mapBindableObjects.end());
        m_mapBindableObjects[BVT_ACCELERATOR_IMMUTABLE] = view;
        return S_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates readable bindable objects (ShaderResourceViews). </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="strDebugObjectName">   [in,out] If non-null, name of the debug object. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT
    PDXBuffer::CreateBindableObjectsReadable( 
        char * strDebugObjectName
        )
    {
        HRESULT hr = S_OK;
        D3D11_BUFFER_DESC descBuf;
        D3D11_SHADER_RESOURCE_VIEW_DESC srvdesc;
        ID3D11ShaderResourceView * pReadableView = NULL;

        ZeroMemory( &descBuf, sizeof(descBuf) );
        (PLATFORMBUFFER(m_pBuffer))->GetDesc( &descBuf );
        ZeroMemory( &srvdesc, sizeof(srvdesc) );
        srvdesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
        srvdesc.BufferEx.FirstElement = 0;
        if ( descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS ) { // raw buf
            srvdesc.Format = DXGI_FORMAT_R32_TYPELESS;
            srvdesc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
            srvdesc.BufferEx.NumElements = descBuf.ByteWidth / 4;
        } else if ( descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED ) { // structured buf
            srvdesc.Format = DXGI_FORMAT_UNKNOWN;
            srvdesc.BufferEx.NumElements = descBuf.ByteWidth / descBuf.StructureByteStride;
        } else {
            return NULL;
       }

        m_pAccelerator->Lock();
        ID3D11Device* pDevice = (ID3D11Device*) m_pAccelerator->GetDevice();
        hr = pDevice->CreateShaderResourceView( (ID3D11Resource*)m_pBuffer, &srvdesc, &pReadableView );
        m_pAccelerator->Unlock();
        if(FAILED(hr)) {
            PTask::Runtime::HandleError("%s failed\n", __FUNCTION__);
            return hr;
        }

        SET_DEBUG_OBJECT_NAME(pReadableView, strDebugObjectName);
        BINDABLEOBJECT view;	
        view.psrv = pReadableView;
        assert(m_mapBindableObjects.find(BVT_ACCELERATOR_READABLE) == m_mapBindableObjects.end());
        m_mapBindableObjects[BVT_ACCELERATOR_READABLE] = view;
        return hr;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates writable bindable objects (UnorderedAccessViews). </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="strDebugObjectName">   [in,out] If non-null, name of the debug object. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT
    PDXBuffer::CreateBindableObjectsWriteable(
        char * strDebugObjectName
        )
    {
        HRESULT hr = S_OK;
        D3D11_BUFFER_DESC descBuf;
        D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
        ID3D11UnorderedAccessView* pWriteableView = NULL; 

        ZeroMemory( &desc, sizeof(desc) );
        ZeroMemory( &descBuf, sizeof(descBuf) );
        (PLATFORMBUFFER(m_pBuffer))->GetDesc( &descBuf );
        desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        desc.Buffer.FirstElement = 0;
        if ( descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS ) {
            // This is a Raw Buffer
            desc.Format = DXGI_FORMAT_R32_TYPELESS; // Format must be DXGI_FORMAT_R32_TYPELESS, when creating Raw Unordered Access View
            desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
            desc.Buffer.NumElements = descBuf.ByteWidth / 4; 
        } else if ( descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED ) {
            // This is a Structured Buffer
            desc.Format = DXGI_FORMAT_UNKNOWN;      // Format must be must be DXGI_FORMAT_UNKNOWN, when creating a View of a Structured Buffer
            desc.Buffer.NumElements = descBuf.ByteWidth / descBuf.StructureByteStride; 
        } else {
            return NULL;
        }

        m_pAccelerator->Lock();
        ID3D11Device* pDevice = (ID3D11Device*) m_pAccelerator->GetDevice();
        hr = pDevice->CreateUnorderedAccessView( (ID3D11Resource*)m_pBuffer, &desc, &pWriteableView );
        m_pAccelerator->Unlock();

        if(FAILED(hr)) {
            PTask::Runtime::HandleError("%s failed\n", __FUNCTION__);
            return hr;
        }

        SET_DEBUG_OBJECT_NAME(pWriteableView, strDebugObjectName);
        BINDABLEOBJECT view;
        view.puav = pWriteableView;
        assert(m_mapBindableObjects.find(BVT_ACCELERATOR_WRITEABLE) == m_mapBindableObjects.end());
        m_mapBindableObjects[BVT_ACCELERATOR_WRITEABLE] = view;
        return hr;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize immutable accelerator view. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="nBufferBytes">     The buffer in bytes. </param>
    /// <param name="pExtent">          [in,out] If non-null, the data. </param>
    /// <param name="pVModule">         [in,out] (optional)  If non-null, the module. </param>
    /// <param name="lpszBinding">      (optional) the binding. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT		
    PDXBuffer::__populateImmutableAcceleratorView(
        __in AsyncContext * pAsyncContext,
        __in UINT nBufferBytes,
        __in HOSTMEMORYEXTENT * pExtent, 
        __out BOOL&             bOutstanding,
        __in void * pVModule,
        __in const char * lpszBinding
        )
    { 
        UNREFERENCED_PARAMETER(nBufferBytes);
        UNREFERENCED_PARAMETER(pAsyncContext);
        UNREFERENCED_PARAMETER(lpszBinding);
        UNREFERENCED_PARAMETER(pVModule);
        HRESULT hr = E_FAIL;
        assert(m_eAccessFlags & PT_ACCESS_HOST_WRITE);
        assert(m_eAccessFlags & PT_ACCESS_IMMUTABLE);
        assert(m_pBuffer != NULL);

        void * pData = pExtent ? pExtent->lpvAddress : NULL;
        UINT nDataBytes = pExtent ? pExtent->uiSizeBytes : 0;

        m_pAccelerator->Lock();
        MARKRANGEENTER(L"__populateImmutableAcceleratorView");
        D3D11_MAPPED_SUBRESOURCE MappedResource;
        ID3D11DeviceContext* pd3dImmediateContext = (ID3D11DeviceContext*) m_pAccelerator->GetContext();
        if(SUCCEEDED(hr = pd3dImmediateContext->Map( (ID3D11Resource*)m_pBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource ))) {
            memcpy( MappedResource.pData, pData, nDataBytes );
            pd3dImmediateContext->Unmap( (ID3D11Resource*)m_pBuffer, 0 );
            ID3D11Buffer* ppCB[1] = { PLATFORMBUFFER(m_pBuffer) };
            pd3dImmediateContext->CSSetConstantBuffers( 0, 1, ppCB );
            m_bPopulated = TRUE;
        } 
        m_pAccelerator->Unlock();
        MARKRANGEEXIT();
        bOutstanding = FALSE;
        return SUCCEEDED(hr) ? nDataBytes : 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize mutable accelerator view. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="nBufferBytes">     The buffer in bytes. </param>
    /// <param name="pExtent">          [in,out] If non-null, the data. </param>
    /// <param name="pModule">          [in,out] (optional)  If non-null, the module. </param>
    /// <param name="lpszBinding">      (optional) the binding. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

	UINT		
    PDXBuffer::__populateMutableAcceleratorView(
        __in AsyncContext *     pAsyncContext,
        __in UINT               nBufferBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __out BOOL&             bOutstanding,
        __in void *             pModule,
        __in const char *       lpszBinding
        ) 
    {
        UNREFERENCED_PARAMETER(nBufferBytes);
        UNREFERENCED_PARAMETER(pAsyncContext);
        UNREFERENCED_PARAMETER(lpszBinding);
        UNREFERENCED_PARAMETER(pModule);

        bOutstanding = FALSE;
        HRESULT hr = E_FAIL;
        UINT uiTransferred = 0;
        void * pData = pExtent ? pExtent->lpvAddress : NULL;
        UINT nBytes = pExtent ? pExtent->uiSizeBytes : 0;
        assert(pData != NULL && nBytes != 0);
        assert(m_pBuffer != NULL);
        assert(!(m_eAccessFlags & PT_ACCESS_IMMUTABLE));

        ID3D11DeviceContext* pContext = (ID3D11DeviceContext*) m_pAccelerator->GetContext();

        // if this method is being called, it means that we've created the GPU side
        // buffer, and we are writing to it from CPU-side. If this gets called,
        // we *should* have planned to handle this by creating a staging buffer first.  
        // In cases where we have the contents at buffer creation time, *and* we
        // do not expect to reuse the memory, there is no need to ever call this. 
        assert(m_pStageBuffer != NULL); 
        assert(m_pOutstandingOpBuffer == NULL);
        assert(m_pOutstandingQuery == NULL);

        // if(HasOutstandingOps())
        //    CompleteOutstandingOps();

        m_bPopulated = FALSE;
        m_bHtoDStagePopulated = FALSE;
        m_pAccelerator->Lock();
        MARKRANGEENTER(L"__populateMutableAcceleratorView+MAP");
        if(nBytes && pData) {

            D3D11_BUFFER_DESC desc;
            D3D11_MAPPED_SUBRESOURCE MappedResource; 
            m_pStageBuffer->GetDesc(&desc);
            size_t uiDatasize = min(nBytes, desc.ByteWidth);
            hr = pContext->Map( m_pStageBuffer, 0, D3D11_MAP_WRITE, 0, &MappedResource );
            
            if(SUCCEEDED(hr)) {

                // populate the stage buffer
                void * p = MappedResource.pData;
                memcpy(p, pData, uiDatasize);
                pContext->Unmap( m_pStageBuffer, 0 );                
                m_bHtoDStagePopulated = TRUE;

                // queue a copy from the stage buffer to 
                // the actual buffer for this object. 

                ID3D11Buffer * pPSBuffer = PLATFORMBUFFER(m_pBuffer);
                pContext->CopyResource(pPSBuffer, m_pStageBuffer);
                uiTransferred = (UINT)uiDatasize;
                m_bPopulated = TRUE;

                bOutstanding = FALSE;
                    // m_pAccelerator->SupportsExplicitAsyncOperations();
            } 
        }
        MARKRANGEEXIT();
        m_pAccelerator->Unlock();
        

        if(!SUCCEEDED(hr)) {
            assert(false);
            PTask::Runtime::HandleError("%s failed\n", __FUNCTION__);
        } 

        return uiTransferred;
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
    PDXBuffer::__populateHostView(
        __in  AsyncContext * pAsyncContext,
        __in  HOSTMEMORYEXTENT * pExtent,
        __in  BOOL bForceSynchronous,
        __out BOOL &bRequestOutstanding
        )
    {
        HRESULT hr;
        UINT uiTransferred = 0;
        void * lpvBuffer = pExtent ? pExtent->lpvAddress : NULL;
        UINT cbBuffer = pExtent ? pExtent->uiSizeBytes : 0;
        assert(lpvBuffer != NULL);
        assert(cbBuffer != 0);
        assert(m_pBuffer != NULL);
        bRequestOutstanding = FALSE;

        UNREFERENCED_PARAMETER(bForceSynchronous);
        UNREFERENCED_PARAMETER(pAsyncContext);

        m_bDtoHStagePopulated = FALSE;
        if(m_pStageBuffer == NULL) {
    
            // we really want to keep this off the critical path!
            PTask::Runtime::MandatoryInform("WARNING: %s::%s dynamic alloc stage buffer!\n",
                                            __FILE__,
                                            __FUNCTION__);
            m_pAccelerator->Lock();
            ID3D11Device * pDevice = PLATFORMDEVICE(m_pAccelerator->GetDevice());
            hr = CreateStagingBuffer(pDevice);
            m_pAccelerator->Unlock();

            if(!SUCCEEDED(hr) || m_pStageBuffer == NULL) {
                // no staging buffer...not good: panic.
                assert(false);
                PTask::Runtime::HandleError("%s: failed to create stage buffer!\n", __FUNCTION__);
                return (UINT) uiTransferred;            
            }
        }

        assert(m_pStageBuffer != NULL);
        MARKRANGEENTER(L"__populateHostView-CopyRes");
        m_pAccelerator->Lock();
        ID3D11DeviceContext* pContext = (ID3D11DeviceContext*) m_pAccelerator->GetContext();
        pContext->CopyResource( m_pStageBuffer, (ID3D11Resource*)m_pBuffer );
        m_bDtoHStagePopulated = TRUE;
        MARKRANGEEXIT();

        D3D11_BUFFER_DESC dPrimaryBuffer;
        D3D11_BUFFER_DESC dStageBuffer;
        (PLATFORMBUFFER(m_pBuffer))->GetDesc(&dPrimaryBuffer);
        (PLATFORMBUFFER(m_pStageBuffer))->GetDesc(&dStageBuffer);        
            
        D3D11_MAPPED_SUBRESOURCE MappedResource; 
        MARKRANGEENTER(L"__populateHostView+MAP");
        hr = pContext->Map( m_pStageBuffer, 0, D3D11_MAP_READ, 0, &MappedResource );
        size_t datasize = cbBuffer;
        if(SUCCEEDED(hr)) {
            void * p = MappedResource.pData;
            assert(cbBuffer >= datasize);
            memcpy(lpvBuffer, p, min(cbBuffer, datasize));
            pContext->Unmap( m_pStageBuffer, 0 );
            uiTransferred = static_cast<UINT>(datasize);
        } else {
            PTask::Runtime::HandleError("%s:%s failed (res=%8X)\n", 
                                        __FUNCTION__,
                                        "ID3D11Buffer::Map()",
                                        hr);
        }

        bRequestOutstanding = FALSE; // m_pAccelerator->SupportsExplicitAsyncOperations();
        m_pAccelerator->Unlock();
        MARKRANGEEXIT();
        return (UINT) uiTransferred;
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
    /// <param name="cbBufferSize">         Size of the buffer. </param>
    /// <param name="pExtent">              (optional) [in] If non-null, the initial buffer contents. </param>
    /// <param name="strDebugObjectName">   (optional) [in] If non-null, a name to assign to the
    ///                                     buffer which will be used to label runtime- specific
    ///                                     objects to aid in debugging. Ignored on release builds. </param>
    /// <param name="bIsByteAddressable">   (optional) true if the buffer should be byte addressable. </param>
    ///
    /// <returns>   PTRESULT (use PTSUCCESS/PTFAILED macros) </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT	
    PDXBuffer::CreateMutableBuffer(
        __in AsyncContext * pAsyncContext,
        __in UINT cbBufferSize,
        __in HOSTMEMORYEXTENT * pExtent,
        __in char * strDebugObjectName, 
        __in bool bIsByteAddressable                                                    
        )
    {
        assert(m_pBuffer == NULL);
        UNREFERENCED_PARAMETER(pAsyncContext);

        // finalize dimensions can return 0. In such a case we must still allocate something, even if
        // it's empty to make it bindable to device-side resources. Hence, we distinguish between alloc
        // size and transfer size.         
        
        FinalizeDimensions(bIsByteAddressable, cbBufferSize);
        UINT uAllocCount = GetAllocationExtentBytes();
        UINT uTransferBytes = GetLogicalExtentBytes();

        // create a descriptor for this buffer. Getting all the bind and misc
        // flags right is a non-trivial undertaking, since not all combinations
        // are valid, and even amongst the valid combinations, there can be
        // a significant performance price for not choosing the most appropriate one.
        // Considerations:
        //  - if the buffer is going to be written by the GPU -> BIND_UNORDERED_ACCESS
        //  - if the buffer is going to be read by the GPU -> BIND_SHADER_RESOURCE
        //  - if we need strided access (rather than byte-granularity), the structured
        //    buffer flags make things faster. 
        //  - if the buffer backs a block that is also backed on another GPU memory
        //    space we can get (ostensibly) better transfer performance by using
        //    the D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX flags.
        //  - These choices impact which USAGE flags are valid choices
        // 
        // In sum, we need to know every way we might use this buffer in the future--
        // which is a difficult prediction to make.

        HRESULT hr = S_OK;
        ID3D11Buffer *pMutableBuffer = NULL;
        D3D11_BUFFER_DESC desc;
        ZeroMemory( &desc, sizeof(desc) );

        // set the allocation size and the graphics pipeline bind flags. 
        // since we are using DirectCompute only, we really only care
        // about being able to bind Unordered access views and shader
        // resource views, derived from the R/W permissions required 
        // for the parent datablock object. 
        
        desc.ByteWidth = uAllocCount;                               
        desc.BindFlags |= (m_eAccessFlags & PT_ACCESS_ACCELERATOR_READ) ? D3D11_BIND_SHADER_RESOURCE : 0;
        desc.BindFlags |= (m_eAccessFlags & PT_ACCESS_ACCELERATOR_WRITE) ? D3D11_BIND_UNORDERED_ACCESS : 0;
        desc.StructureByteStride = bIsByteAddressable ? 1 : m_vDimensionsFinalized.cbElementStride;

        // set the aptly named "MiscFlags" field. There are a few we care about here. 
        // * If we need byte-aligned access, we have say so explicitly, otherwise, we must
        //   declare the stride we expect (!!). 
        // * If we want to be able to use P2P APIs to copy this to other devices,
        //   we need to set these flags, and the recommended idiom is depends on 
        //   the DirectX feature level. Our choices here also restrict our choices
        //   about CPU access as well (shareable buffers cannot be accessed by
        //   CPU--nonsense, considering the DX API has to copy the blasted thing
        //   through system memory using the CPU for 99.9999% of all the cards it
        //   supports). 
        // Detecting whether to set the shared flag must take into account 
        // the resources available in the system and whether or not the 
        // graph structure actually enables a cross-GPU opportunity here.   
        //
        // Also note: from:
        //  "Starting with Windows 8, we recommend that you enable resource data sharing between two or more 
        //   Direct3D devices by using a combination of the D3D11_RESOURCE_MISC_SHARED_NTHANDLE and 
        //   D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX flags instead."
        // we should implement this idiom, but as this writing all our 11.1 devices enumerate
        // as 11.0, so its extra work. so leave a warning for the future that detects this. 

        BOOL bMultiGPUEnv = PTask::Runtime::MultiGPUEnvironment();
        BOOL bExplicitShareHint = (m_eAccessFlags & PT_ACCESS_SHARED_HINT);
        BOOL bExplicitShareSupport = m_pAccelerator->SupportsDeviceToDeviceTransfer(NULL);
        BOOL bCreateShared = bMultiGPUEnv && bExplicitShareHint && bExplicitShareSupport;

        // deal with the usage flags. 
        // New strategy here is to always use default, and create additional staging buffers 
        // for cases where we may need to do data transfers in the future.
        desc.Usage = D3D11_USAGE_DEFAULT;

        // now deal with misc flags. 
        desc.MiscFlags = 0;
        desc.MiscFlags |= bIsByteAddressable ? D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS : D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.MiscFlags |= bCreateShared ? D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX : 0;

        D3D11_SUBRESOURCE_DATA InitData;
        VOID* pInitialBufferContents = pExtent ? pExtent->lpvAddress : NULL;
        UINT cbInitialBufferContents = pExtent ? pExtent->uiSizeBytes : 0;
        BOOL bPooledBlock = (m_eAccessFlags & PT_ACCESS_POOLED_HINT);
        BOOL bValidInitValue = (pInitialBufferContents && uTransferBytes != 0);

        // we want a H->D staging buffer if the device needs to read the block and
        // 1. no initial value is supplied OR
        // 2. the block is pooled
        // 3. the block is a likely transit point between multiple GPUs
        // 4. host write permissions were explicitly requested 
        // ** an exception is if host permissions were requested but the block is not pooled
        //    and we do have an initial value, we should assume this is a single-use resource,
        //    so we can discard it. 
        
        BOOL bGPURead = (m_eAccessFlags & PT_ACCESS_ACCELERATOR_READ);
        BOOL bExplicitHostWritePerms = (m_eAccessFlags & PT_ACCESS_HOST_WRITE);
        BOOL bP2PTransitPoint = (bMultiGPUEnv && bExplicitShareHint);
        BOOL bSingleUseBlock = (!bGPURead || bValidInitValue) && !bPooledBlock && !bP2PTransitPoint;
        BOOL bCreateHtoDStageBuffer = bP2PTransitPoint || bPooledBlock || (bExplicitHostWritePerms && !bSingleUseBlock);

        // we want a D to H stage buffer if the block is on an exposed output channel
        // or if it is at a transit point between GPUs. The exposed output channel should
        // be detectable from the host read permissions. 
        
        BOOL bCreateDtoHStageBuffer = (m_eAccessFlags & PT_ACCESS_HOST_READ) || bP2PTransitPoint || (bMultiGPUEnv && bPooledBlock);

        D3D11_SUBRESOURCE_DATA * pInitData = NULL;

        if(bValidInitValue) {

            if(!(cbInitialBufferContents >= desc.ByteWidth || cbInitialBufferContents == 0)) {
                assert(cbInitialBufferContents >= desc.ByteWidth || cbInitialBufferContents == 0);
                PTask::Runtime::HandleError("%s: Attempt to initialize device side buffer, " 
                                            "datasize mismatch\n",
                                            __FUNCTION__);
                hr = E_FAIL;
                return 0; 
            } 
            InitData.pSysMem = pInitialBufferContents;
            pInitData = &InitData;
        }

        MARKRANGEENTER(L"CreateMutable::CreateBuffer");
        m_pAccelerator->Lock();
        ID3D11Device* pDevice = (ID3D11Device*) m_pAccelerator->GetDevice();
        hr = pDevice->CreateBuffer( &desc, pInitData, &pMutableBuffer );
        m_pAccelerator->Unlock();
        MARKRANGEEXIT();

        if(SUCCEEDED(hr)) {

            m_pBuffer = pMutableBuffer;
            if(bCreateShared) {

                HRESULT hrKM = pMutableBuffer->QueryInterface(__uuidof(IDXGIKeyedMutex), (void**)&m_pDXGIKeyedMutex);
                HRESULT hrIR = pMutableBuffer->QueryInterface(__uuidof(IDXGIResource), reinterpret_cast<void **>(&m_pDXGIResource));
                HRESULT hrSH = m_pDXGIResource->GetSharedHandle(&m_hDXGIHandle);
                
                assert(SUCCEEDED(hrKM) && m_pDXGIKeyedMutex);
                assert(SUCCEEDED(hrIR) && m_pDXGIResource);
                assert(SUCCEEDED(hrSH) && (m_hDXGIHandle != INVALID_HANDLE_VALUE));
                if(!(SUCCEEDED(hrKM) && m_pDXGIKeyedMutex) ||
                   !(SUCCEEDED(hrIR) && m_pDXGIResource) ||
                   !(SUCCEEDED(hrSH) && (m_hDXGIHandle != INVALID_HANDLE_VALUE))) {

                    PTask::Runtime::HandleError("%s: Creating sharable objects failed\n", __FUNCTION__);
                    return PTASK_ERR;
                }

                m_bP2PShareable = TRUE;
            }

            m_bPopulated = bValidInitValue;
            SET_DEBUG_OBJECT_NAME(pMutableBuffer, strDebugObjectName);

            if(SUCCEEDED(hr) && (bCreateDtoHStageBuffer || bCreateHtoDStageBuffer)) {
                m_pAccelerator->Lock();
                ID3D11Device* pDevice = (ID3D11Device*) m_pAccelerator->GetDevice();
                hr = CreateStagingBuffer(pDevice);
                m_pAccelerator->Unlock();
            }
        }

        if(SUCCEEDED(hr)) {           
            return CreateBindableObjects(strDebugObjectName);
        } else {
            assert(false);
            PTask::Runtime::HandleError("%s: InitializeMutableBuffer failed\n", __FUNCTION__);
            return PTASK_ERR;
        }
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Force synchronize. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void                
    PDXBuffer::ForceSynchronize(
        VOID
        ) 
    {
        // cjr: 3/11/14: DX APIs now support query for outstanding work.
        // If this is being called, it means its time for someone to dig
        // in and use them to implement a better DX AsyncContext.
        // the error call below should force the app to exit: hopefully that's
        // sufficiently attention-getting!
        PTask::Runtime::HandleError("DX Flush called...this shouldn't happen!\n");

        m_pAccelerator->Lock();
        ID3D11DeviceContext* pContext = (ID3D11DeviceContext*) m_pAccelerator->GetContext();
        pContext->Flush();
        m_pAccelerator->Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a staging buffer for host to device transfers. </summary>
    ///
    /// <remarks>   Crossbac, 3/11/2014. </remarks>
    ///
    /// <param name="pDevice">  [in,out] If non-null, the device. </param>
    /// <param name="nBytes">   The bytes. </param>
    /// <param name="pData">    [in,out] If non-null, the data. </param>
    ///
    /// <returns>   The new hto d stage buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    PDXBuffer::CreateStagingBuffer(
        __in ID3D11Device * pDevice
        )
    {
        ID3D11Buffer * resultbuf = NULL;
        D3D11_BUFFER_DESC desc;
        ZeroMemory( &desc, sizeof(desc) );
        (PLATFORMBUFFER(m_pBuffer))->GetDesc( &desc );
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE | D3D11_CPU_ACCESS_READ;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.MiscFlags = 0;

        MARKRANGEENTER(L"CreateStagingBuffer");
        assert(m_pStageBuffer == NULL);
        m_pAccelerator->Lock();
        m_bHtoDStagePopulated = FALSE;
        HRESULT hr = pDevice->CreateBuffer(&desc, NULL, &resultbuf);

        if (SUCCEEDED(hr)) {

            SET_DEBUG_OBJECT_NAME_RAW(resultbuf, "CreateStageBuffer-DebugBuf");
            m_pStageBuffer = resultbuf;
            m_pAccelerator->Unlock();
            MARKRANGEEXIT();
            return hr;

        } else {

            assert(false);
            PTask::Runtime::HandleError("%s failed\n", __FUNCTION__);
            m_pAccelerator->Unlock();
            MARKRANGEEXIT();
            return hr;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Complete any outstanding ops. </summary>
    ///
    /// <remarks>   Crossbac, 3/12/2014. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    PDXBuffer::CompleteOutstandingOps(
        VOID
        )
    {
        if(m_pOutstandingOpBuffer != NULL) {

            BOOL bResult = FALSE;
            ID3D11DeviceContext* pContext = (ID3D11DeviceContext*) m_pAccelerator->GetContext();

            if(m_pOutstandingQuery) {
                MARKRANGEENTER(L"WaitQuery");
                BOOL bOutstanding = TRUE;
                while(bOutstanding) {
                    // m_pAccelerator->Lock();
                    bOutstanding = pContext->GetData( m_pOutstandingQuery, NULL, 0, 0 ) == S_FALSE;
                    // m_pAccelerator->Unlock();
                }
                // spin until event is finished
                m_pOutstandingQuery->Release();
                m_pOutstandingQuery = NULL;
                MARKRANGEEXIT();
            }

            if(m_pOutstandingDtoHTarget != NULL) {

                // outstanding D to H operation.
                MARKRANGEENTER(L"CompleteOutstandingOps(D->H)");
                assert(m_pOutstandingDtoHTarget != NULL);
                D3D11_MAPPED_SUBRESOURCE MappedResource; 
                D3D11_BUFFER_DESC desc;
                ZeroMemory(&desc, sizeof(desc));

                PLATFORMBUFFER(m_pStageBuffer)->GetDesc(&desc);
                m_pAccelerator->Lock();
                HRESULT hr = pContext->Map( m_pStageBuffer, 0, D3D11_MAP_READ, 0, &MappedResource );
                if(SUCCEEDED(hr)) {
                    size_t datasize = min(desc.ByteWidth, m_pOutstandingDtoHTarget->uiSizeBytes);
                    void * p = MappedResource.pData;
                    memcpy(m_pOutstandingDtoHTarget->lpvAddress, p, datasize);
                    pContext->Unmap( m_pStageBuffer, 0 );
                    m_pAccelerator->Unlock();
                    m_pOutstandingOpBuffer = NULL;
                    if(m_pOutstandingDtoHTarget) {
                        delete m_pOutstandingDtoHTarget;
                        m_pOutstandingDtoHTarget = NULL;
                    }
                    bResult = TRUE;
                } else {
                    PTask::Runtime::HandleError("%s:%s failed (res=%8X)\n", 
                                                __FUNCTION__,
                                                "ID3D11Buffer::Map()",
                                                hr);
                    bResult = FALSE;
                }
                MARKRANGEEXIT();

            } else if(m_pOutstandingHtoDTarget) {

                // we already populated an H to D staging buffer previously
                MARKRANGEENTER(L"CompleteOutstandingOps(H->D)");                
                if(m_pOutstandingOpBuffer == m_pStageBuffer &&
                   m_bHtoDStagePopulated) { // &&
                   //m_pOutstandingHtoDTarget &&
                   //m_pOutstandingHtoDTarget->lpvAddress != NULL) {

                    // we already populated the stage buffer, so we
                    // can just queue a copy without doing any additional
                    // map/unmap 
                    ID3D11DeviceContext* pContext = (ID3D11DeviceContext*) m_pAccelerator->GetContext();
                    m_pAccelerator->Lock();
                    pContext->CopyResource( (ID3D11Resource*)m_pBuffer, m_pStageBuffer );
                    m_pAccelerator->Unlock();
                    m_pOutstandingOpBuffer = NULL;
                    m_bPopulated = TRUE;
                    if(m_pOutstandingHtoDTarget) { 
                        delete m_pOutstandingHtoDTarget;
                        m_pOutstandingHtoDTarget = NULL;     
                    }
                    bResult = TRUE;
                } else {
                    assert(FALSE);
                }
                MARKRANGEEXIT();
            }
            return bResult;
        }

        return TRUE;
    }
        
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check for any outstanding ops. </summary>
    ///
    /// <remarks>   Crossbac, 3/12/2014. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    PDXBuffer::HasOutstandingOps(
        VOID
        )
    {
        BOOL bResult = FALSE;
        if(m_pOutstandingOpBuffer != NULL) {
            bResult = TRUE;
            ID3D11DeviceContext* pContext = (ID3D11DeviceContext*) m_pAccelerator->GetContext();
            if(m_pOutstandingQuery) {
                bResult = pContext->GetData( m_pOutstandingQuery, NULL, 0, 0 ) == S_FALSE;
                if(!bResult) {
                    m_pOutstandingQuery->Release();
                    m_pOutstandingQuery = NULL;
                    m_pOutstandingOpBuffer = NULL;                    
                }
            }
            return bResult;
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Device to device transfer. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pDstBuffer">       [in,out] If non-null, the accelerator. </param>
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    PDXBuffer::DeviceToDeviceTransfer(
        __inout PBuffer *       pDstBuffer,
        __in    AsyncContext *  pAsyncContext
        ) 
    {
        assert(m_pBuffer != NULL);
        assert(pAsyncContext != NULL);

        if(ContextRequiresSync(pAsyncContext, OT_MEMCPY_SOURCE))
            WaitOutstandingAsyncOperations(pAsyncContext, OT_MEMCPY_SOURCE);
        if(pDstBuffer->ContextRequiresSync(pAsyncContext, OT_MEMCPY_TARGET))
            pDstBuffer->WaitOutstandingAsyncOperations(pAsyncContext, OT_MEMCPY_TARGET);

        // first attempt the transfer in the CUDA device context of the destination accelerator, which
        // should typically be the one associated with the async context. If that fails, attempt it
        // from the CUDA context of the source. If that also fails, fail the function call and let the
        // calling layer handle the error by routing the transfer through host memory. 
        
        DXAccelerator * pSrcAcc = (DXAccelerator*) m_pAccelerator;
        DXAccelerator * pDestAcc = (DXAccelerator*) pDstBuffer->GetAccelerator();
        DXAccelerator * pAsyncAcc = (DXAccelerator*) pAsyncContext->GetDeviceContext();
        DXAccelerator * pRemoteAcc = pAsyncAcc == pSrcAcc ? pDestAcc : pSrcAcc;
        UNREFERENCED_PARAMETER(pRemoteAcc);

        assert(pDestAcc != NULL);
        assert(pSrcAcc != NULL);
        assert(pDestAcc->LockIsHeld());
        assert(pSrcAcc->LockIsHeld());
        assert(pAsyncAcc == pSrcAcc || pAsyncAcc == pDestAcc);

        void * pDstDBuffer = pDstBuffer->GetBuffer();
        size_t nBytes = pDstBuffer->GetLogicalExtentBytes();
        BOOL bXferSucceeded = FALSE;
        pDstBuffer->MarkDirty(TRUE);

        D3D11_BUFFER_DESC dDstBufferDesc;
        D3D11_BUFFER_DESC dSrcBufferDesc;
        ID3D11Buffer * pPSDestBuffer = PLATFORMBUFFER(pDstDBuffer);
        ID3D11Buffer * pPSSrcBuffer = PLATFORMBUFFER(m_pBuffer);
        pPSDestBuffer->GetDesc(&dDstBufferDesc);
        pPSSrcBuffer->GetDesc(&dSrcBufferDesc);

        ID3D11Device * pDevice = PLATFORMDEVICE(pAsyncAcc->GetDevice());
        ID3D11DeviceContext * pContext = PLATFORMCONTEXT(pAsyncAcc->GetContext());
        D3D11_BUFFER_DESC* pRemoteBufferDesc = (pAsyncAcc == pSrcAcc) ? &dDstBufferDesc : &dSrcBufferDesc;
        PBuffer * pRemotePBuffer = (pAsyncAcc == pSrcAcc) ? pDstBuffer : this;

        if(pRemoteBufferDesc->MiscFlags & D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX) {

            // we can transfer from the source to the dest based
            // on the usage with which the dst buffer was created.

            UINT64 uiAcqKey = 0;
            UINT64 uiRelKey = uiAcqKey;
            ACQUIRE_CTX(pAsyncAcc);
            PDXBuffer * pDXRemoteBuffer = reinterpret_cast<PDXBuffer*>(pRemotePBuffer);
            ID3D11Buffer * pSharedBufferPointer = NULL;

            // make sure the destination buffer is initialized into
            // a shareable state (has handles/mutex objects) 
            
            assert(pRemoteAcc->SupportsDeviceToDeviceTransfer(pAsyncAcc));
            assert(pDXRemoteBuffer->m_hDXGIHandle != INVALID_HANDLE_VALUE);
            assert(pDXRemoteBuffer->m_pDXGIKeyedMutex != NULL);

            // get a pointer to the buffer we're interested in based
            // on its handle (the ID3D11 buffer pointer in the dst buffer
            // object is valid only in the destination device context!)
            
            HRESULT hrOpen = pDevice->OpenSharedResource(pDXRemoteBuffer->m_hDXGIHandle, 
                                                         __uuidof(ID3D11Buffer), 
                                                         reinterpret_cast<void **>(&pSharedBufferPointer));

            if(SUCCEEDED(hrOpen) && pSharedBufferPointer != NULL) {

                // acquire a lock on the destination resource...
                // be sure to examine the return value because it encodes 
                // DWORD mutex wait return values as well, which the succeeded 
                // macro will fail to catch. 
            
                pDXRemoteBuffer->PlatformSpecificAcquireSync(uiAcqKey);
                pContext->CopyResource(pSharedBufferPointer, pPSSrcBuffer);
                bXferSucceeded = TRUE; 
                pDXRemoteBuffer->PlatformSpecificReleaseSync(uiRelKey);

            } else {
                
                // ugh...
                PTask::Runtime::HandleError("%s::%s(%d): OpenSharedResource failed with HR=%16X\n",
                                            __FILE__,
                                            __FUNCTION__,
                                            __LINE__,
                                            hrOpen);
            }


            RELEASE_CTX(pSrcAcc);            
        }

        m_bPopulated = bXferSucceeded;
        BOOL bResult = nBytes > 0 && bXferSucceeded;
        if(!bResult) {
            PTask::Runtime::MandatoryInform("XXXX\n");
            PTask::Runtime::MandatoryInform("XXXX\n");
            PTask::Runtime::MandatoryInform("DeviceToDeviceTransfer FAILED!\n");
            PTask::Runtime::MandatoryInform("XXXX\n");
            PTask::Runtime::MandatoryInform("XXXX\n");
        }
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Acquires the synchronise. </summary>
    ///
    /// <remarks>   Crossbac, 3/14/2014. </remarks>
    ///
    /// <param name="uiAcquireKey"> The acquire key. </param>
    ///-------------------------------------------------------------------------------------------------

    PDXBuffer *
    PDXBuffer::PlatformSpecificAcquireSync(
        __in UINT64 uiAcquireKey
        )
    {
        if(m_pDXGIKeyedMutex != NULL) {
            assert(m_bP2PShareable);
            HRESULT hrSync = m_pDXGIKeyedMutex->AcquireSync(uiAcquireKey, INFINITE);
            if(hrSync != S_OK) {
                PTask::Runtime::HandleError("%s::%s(%d) AquireSync failed with hr = %16X\n",
                                            __FILE__,
                                            __FUNCTION__,
                                            __LINE__,
                                            hrSync);
            }
            m_bP2PLocked = hrSync == S_OK;
            return m_bP2PLocked ? this : NULL;
        }        
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the synchronise. </summary>
    ///
    /// <remarks>   Crossbac, 3/14/2014. </remarks>
    ///
    /// <param name="uiReleaseKey"> The release key. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    PDXBuffer::PlatformSpecificReleaseSync(
        __in UINT64 uiReleaseKey
        )
    {
        if(m_pDXGIKeyedMutex != NULL) {
            assert(m_bP2PShareable);
            assert(m_bP2PLocked);
            m_pDXGIKeyedMutex->ReleaseSync(uiReleaseKey);
            m_bP2PLocked = FALSE;
        }
    }
};
