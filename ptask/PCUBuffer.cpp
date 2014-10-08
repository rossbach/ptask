//--------------------------------------------------------------------------------------
// File: PCUBuffer.cpp
// Maintainer: crossbac@microsoft.com
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#ifdef CUDA_SUPPORT

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "PCUBuffer.h"
#include "AsyncContext.h"
#include "AsyncDependence.h"
#include "accelerator.h"
#include "cuaccelerator.h"
#include "datablock.h"
#include "MemorySpace.h"
#include "graph.h"
#include "Scheduler.h"
#include "PBufferProfiler.h"
#include "extremetrace.h"
#include "DatablockProfiler.h"
#include "datablocktemplate.h"
#include "ptgc.h"
#include "port.h"
#include "nvtxmacros.h"
#include <assert.h>

#define PTASSERT(x) assert(x)

#ifdef PROFILE_PBUFFERS
#define pbpptr(x)     (reinterpret_cast<PBufferProfiler*>(x))
#define pbpinit(x)    (((x)!=NULL)&&(pbpptr(x))->m_bAllocProfilerInit)
#define pbpenabled(x) (Runtime::GetProfilePlatformBuffers()&&pbpinit(x))
#define pbptick(x)    ((pbpptr(x))->m_pAllocationTimer->elapsed(false))
#define approfile_enter(x)                                                      \
    double dStart_##x = 0.0;                                                    \
    if(pbpenabled(m_pProfiler)) {                                               \
        dStart_##x = pbptick(m_pProfiler); }                                    
#define approfile_exit(x,y)                                                     \
    double dL_##x = 0.0;                                                        \
    if(pbpenabled(m_pProfiler)) {                                               \
        dL_##x = pbptick(m_pProfiler)-dStart_##x;                               \
        pbpptr(m_pProfiler)->Record(y, m_pPSAcc->GetAcceleratorId(), dL_##x); }
#else
#define approfile_enter(x)  
#define approfile_exit(x,y) 
#endif

#define ERRBUFSZ 2048
#define PLATFORMBUFFER(x) reinterpret_cast<CUdeviceptr>(x)
#define PLATFORMBUFFERPTR(x) reinterpret_cast<CUdeviceptr*>(x)
#define PLATFORMACCELERATOR(x) reinterpret_cast<CUAccelerator*>(x)
#define ACQUIRE_CTX(acc)                            \
        MARKRANGEENTER(L"PCUBuf-acqLock");          \
        acc->Lock();                                \
        BOOL bFCC = !acc->IsDeviceContextCurrent(); \
        if(bFCC) acc->MakeDeviceContextCurrent();   \
        MARKRANGEEXIT(); 
#define RELEASE_CTX(acc)                            \
        if(bFCC) acc->ReleaseCurrentDeviceContext();\
        acc->Unlock(); 
#define RE(str) PTask::Runtime::HandleError(str)
#define _V(stmt)                                                            \
    do{ if(CUDA_SUCCESS != (res = (stmt))) {                                \
        char sz[ERRBUFSZ];                                                  \
        sprintf_s(sz, ERRBUFSZ, #stmt " failed with %d at %s::%s line %d",  \
                                  res,                                      \
                                  __FILE__,                                 \
                                  __FUNCTION__,                             \
                                  __LINE__);                                \
        RE(sz);                                                             \
        RELEASE_CTX(m_pPSAcc);                                              \
        return PTASK_ERR; } } while(0)
#define _VV(ctxt, stmt)                                                     \
    do { if(CUDA_SUCCESS != (res = (stmt))) {                               \
        char sz[ERRBUFSZ];                                                  \
        sprintf_s(sz, ERRBUFSZ, #stmt " failed with %d at %s::%s line %d",  \
                                  res,                                      \
                                  __FILE__,                                 \
                                  __FUNCTION__,                             \
                                  __LINE__);                                \
        RE(sz);                                                             \
        RELEASE_CTX(ctxt);                                                  \
        return; } } while(0)                                                         
#define _VI(stmt,v)                                                         \
  do {                                                                      \
    if(CUDA_SUCCESS != (res = (stmt))) {                                    \
        char sz[ERRBUFSZ];                                                  \
        sprintf_s(sz, ERRBUFSZ, #stmt " failed with %d at %s::%s line %d",  \
                                  res,                                      \
                                  __FILE__,                                 \
                                  __FUNCTION__,                             \
                                  __LINE__);                                \
        RE(sz);                                                             \
        RELEASE_CTX(m_pPSAcc);                                              \
        return v; } } while(0)
#define WARN_IF_SYNCHRONOUS(x)                                                        \
    do { if((x)&&PTask::Runtime::GetDebugAsynchronyMode()) {                          \
        char sz[ERRBUFSZ];                                                            \
        sprintf_s(sz, ERRBUFSZ, "Synchronous XFER at %s line %d", __FILE__, __LINE__);\
        PTask::Runtime::Warning(sz);                                                  \
    } } while(0)
#define WARN_IF_EMPTY(x)                                                                 \
    do { if(x) {                                                                         \
        char sz[ERRBUFSZ];                                                               \
        sprintf_s(sz, ERRBUFSZ, "Empty materialization: %s line %d", __FILE__, __LINE__);\
        PTask::Runtime::Inform(sz);                                                      \
    } } while(0)
#define WARN(x)                                                               \
    do {                                                                      \
        char sz[ERRBUFSZ];                                                    \
        sprintf_s(sz, ERRBUFSZ, "%s: %s line %d", (x), __FILE__, __LINE__);   \
        PTask::Runtime::Warning(sz);                                          \
    } while(0)

#ifdef CHECK_CRITICAL_PATH_ALLOC
#define check_critical_path_alloc(uAllocCount)                                     \
    if(Runtime::GetCriticalPathAllocMode() && Scheduler::GetRunningGraphCount()) { \
	    PTask::Runtime::MandatoryInform("cuMemAlloc(%d)!\n", uAllocCount);         \
    }
#else
#define check_critical_path_alloc(uAllocCOunt)                              
#endif


namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pParent">          [in,out] If non-null, the parent. </param>
    /// <param name="f">                The f. </param>
    /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
    /// <param name="p">                [in,out] If non-null, the p. </param>
    /// <param name="pAlloc">           [in,out] If non-null, the allocate. </param>
    /// <param name="uiUID">            (optional) the uid. </param>
    ///-------------------------------------------------------------------------------------------------

    PCUBuffer::PCUBuffer(
        Datablock * pParent,
        BUFFERACCESSFLAGS f,
        UINT nChannelIndex,
        Accelerator * p, 
        Accelerator * pAlloc,
        UINT uiUID
        ) : PBuffer(pParent, f, nChannelIndex, p, pAlloc, uiUID)
    {
        m_pPSAcc = PLATFORMACCELERATOR(p);
        m_pPageLockedBuffer = NULL;
        m_bPageLockedBufferOwned = FALSE; 
        m_bDeviceBufferOwned = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    PCUBuffer::~PCUBuffer(
        VOID
        )
    {
        // note that we don't need to release any additional
        // objects for immutable views, since they just use
        // the ID3D11Buffer* in m_pBuffer directly.
        if(m_pBuffer) {
            assert(m_pPSAcc != NULL);
            ACQUIRE_CTX(m_pPSAcc);
            if(m_pPageLockedBuffer) {
                if(m_bPageLockedBufferOwned) {
                    m_pPSAcc->FreeHostMemory(m_pPageLockedBuffer, TRUE);
                } else {
                    assert(false && "cuMemHostRegister not being used for page-locking in PTask!");
                    CUresult res = cuMemHostUnregister(m_pPageLockedBuffer);
                    trace3("cuMemHostUnregister(%16llX)->res=%d\n", m_pPageLockedBuffer, res);
                    if(m_bDeviceBufferOwned) {
                        res = cuMemFree(PLATFORMBUFFER(m_pBuffer));
                        trace3("cuMemFree(%16llX)->res=%d\n", m_pBuffer, res);
                        m_pPSAcc->RecordDeallocation(m_pBuffer);
                    }
                }
            } else {
                assert(m_bDeviceBufferOwned);
                CUresult res = cuMemFree(PLATFORMBUFFER(m_pBuffer)); UNREFERENCED_PARAMETER(res);
                trace3("cuMemFree(%16llX)->res=%d\n", m_pBuffer, res);
                PTASSERT(res == CUDA_SUCCESS);
                m_pPSAcc->RecordDeallocation(m_pBuffer);
            }
            RELEASE_CTX(m_pPSAcc);
            m_pBuffer = NULL;
        }
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
    PCUBuffer::__populateHostView(
        __in  AsyncContext * pAsyncContext,
        __in  HOSTMEMORYEXTENT * pExtent,
        __in  BOOL bForceSynchronous,
        __out BOOL &bRequestOutstanding
        )
    {
        #pragma warning(disable:4127)
        CUresult res;
        assert(pAsyncContext != NULL);
        assert(m_pBuffer != NULL);
        assert(pExtent != NULL);
        assert(pExtent->lpvAddress != NULL);

        //bForceSynchronous = TRUE;

        MARKRANGEENTER(L"__populateHostView");
        UINT cbSrcBuffer = GetLogicalExtentBytes();

        if(pExtent->uiSizeBytes == 0  || 
           cbSrcBuffer == 0   || 
           pExtent->lpvAddress == NULL  ||
           m_pBuffer == NULL) {
            // can't copy to/from an empty or non-existent buffer!
            // WARN_IF_EMPTY(TRUE);
            MARKRANGEEXIT();
            return 0;
        }

        CUstream hStream = GetStream(pAsyncContext);

        ACQUIRE_CTX(m_pPSAcc);

        // Zero-length buffers are legitimate because graphs may compute results with empty output
        // (e.g. a join where no rows matched in either relation). We need PTask graphs (and device
        // code) to be robust to empty datablocks consequently. Skip the actual memcpy call for zero-
        // length buffers, but note that we go to the hassle of setting the last access stream
        // regardless of whether the transfer size is non-zero because we want the Datablock buffer map
        // coherence state machine to continue to work even for degenerate buffers. 

        UINT cbTransferBytes = min(pExtent->uiSizeBytes, cbSrcBuffer);

        // Perform the transfer. On the first attempt, use Async APIs. These can fail if host buffers
        // were not allocated through CUDA API (and we are using a pre-4.0 CUDA backend), but we can
        // make PTask robust to this by assuming host buffers were not allocated through CUDA on a
        // failure, and retrying with the synchronous API version. We complain loudly though to make
        // sure the programmer finds out about the lost parallelism and fixes it if possible. 

        CUstream hXferStream = bForceSynchronous ? NULL : hStream;
        trace5("cuMemcpyDtoHAsync(%16llX, %16llX, %d, %u)\n", 
               pExtent->lpvAddress, 
               m_pBuffer, 
               cbTransferBytes, 
               hXferStream);

        BOOL bAsynchronousCall = TRUE;
        if(PTask::Runtime::GetDisableDeviceToHostXfer()) {
            res = CUDA_SUCCESS;
        } else {
            res = cuMemcpyDtoHAsync(pExtent->lpvAddress, 
                                    PLATFORMBUFFER(m_pBuffer), 
                                    cbTransferBytes, 
                                    hXferStream);
        }

        if(CUDA_SUCCESS != res && (hXferStream != NULL)) {

            WARN_IF_SYNCHRONOUS(TRUE);
            trace4("cuMemcpyDtoH(%16llX, %16llX, %d)\n", pExtent->lpvAddress, m_pBuffer, cbTransferBytes);
            _VI(cuMemcpyDtoH(pExtent->lpvAddress, PLATFORMBUFFER(m_pBuffer), cbTransferBytes), 0);
            PTask::Runtime::Warning("Synchronous API used where async failed in PCUBuffer::__populateHostView!");
            bAsynchronousCall = FALSE;
        }

        BOOL bSuccess = (res == CUDA_SUCCESS);
        if(bSuccess && bForceSynchronous && bAsynchronousCall) {
            trace("cuStreamSynchronize()\n");
            cuStreamSynchronize(hXferStream);
        }

        bRequestOutstanding = (bAsynchronousCall && bSuccess && !bForceSynchronous);

        RELEASE_CTX(m_pPSAcc);
        MARKRANGEEXIT();

        return bSuccess ? cbTransferBytes : 0;
        #pragma warning(default:4127)
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if a device-side view of this data can be materialized
    /// 			using memset APIs rather than memcpy APIs. </summary>
    ///
    /// <remarks>   crossbac, 7/10/2012. </remarks>
    ///
    /// <param name="uiBufferBytes">        The buffer in bytes. </param>
    /// <param name="pInitialData">         [in,out] If non-null, the data. </param>
    /// <param name="uiInitialDataBytes">   The bytes. </param>
    ///
    /// <returns>   true if device view memsettable, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    PCUBuffer::IsDeviceViewMemsettable(
        __in UINT uiBufferBytes,
        __in void * pInitialData,
        __in UINT uiInitialDataBytes
        )
    {
        // todo: analyze initial data to
        // see if it's uniform...
        UNREFERENCED_PARAMETER(pInitialData); 
        if(uiInitialDataBytes == sizeof(UINT) && (uiBufferBytes % sizeof(UINT) == 0))
            return TRUE;
        if(uiInitialDataBytes == sizeof(BYTE))
            return TRUE;
        if(m_pParent) {
            DatablockTemplate * pTemplate = m_pParent->GetTemplate();
            if(pTemplate) {
                BOOL bMemsettable = pTemplate->IsInitialValueMemsettable();
                if(bMemsettable) {
                    // make sure we aren't neglecting some actual
                    // data that conflicts with what the template thinks
                    // is the value used to create the buffer. 
                    UINT uiStride = pTemplate->GetInitialValueMemsetStride();
                    if(uiStride == sizeof(UINT)) {
                        UINT uiRefValue = *((UINT*) pTemplate->GetInitialValue());
                        UINT uiElements = uiInitialDataBytes/uiStride;
                        UINT * pElements = (UINT*) pInitialData;
                        for(UINT i=0; i<uiElements; i++) {
                            if(pElements[i] != uiRefValue) {
                                bMemsettable = FALSE;
                                break;
                            }
                        }
                    } else {
                        assert(uiStride == sizeof(BYTE));
                        BYTE ucRefValue = *((BYTE*)pTemplate->GetInitialValue());
                        BYTE * pElements = (BYTE*)pInitialData;
                        for(UINT i=0; i<uiInitialDataBytes; i++) {
                            if(pElements[i] != ucRefValue) {
                                bMemsettable = FALSE;
                                break;
                            }
                        }
                    }
                }
                return bMemsettable;
            }
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a device memset stride. </summary>
    ///
    /// <remarks>   crossbac, 7/10/2012. </remarks>
    ///
    /// <param name="uiBufferBytes">        The buffer in bytes. </param>
    /// <param name="pInitialData">         [in,out] If non-null, the data. </param>
    /// <param name="uiInitialDataBytes">   The bytes. </param>
    ///
    /// <returns>   The device memset stride. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    PCUBuffer::GetDeviceMemsetStride(
        __in UINT uiBufferBytes,
        __in UINT uiInitialDataBytes
        )
    {
        if(uiInitialDataBytes == sizeof(UINT) && (uiBufferBytes % sizeof(UINT) == 0))
            return sizeof(UINT);
        if(uiInitialDataBytes == sizeof(BYTE))
            return sizeof(BYTE);
        assert(m_pParent != NULL);        
        DatablockTemplate * pTemplate = m_pParent->GetTemplate();
        assert(pTemplate != NULL);
        return pTemplate->GetInitialValueMemsetStride();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a device memset count. </summary>
    ///
    /// <remarks>   crossbac, 7/10/2012. </remarks>
    ///
    /// <param name="uiBufferBytes">        The buffer in bytes. </param>
    /// <param name="pInitialData">         [in,out] If non-null, the data. </param>
    /// <param name="uiInitialDataBytes">   The bytes. </param>
    ///
    /// <returns>   The device memset count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    PCUBuffer::GetDeviceMemsetCount(
        __in UINT uiBufferBytes,
        __in UINT uiInitialDataBytes
        )
    {
        if(uiInitialDataBytes == sizeof(UINT) && (uiBufferBytes % sizeof(UINT) == 0))
            return uiBufferBytes / sizeof(UINT);
        if(uiInitialDataBytes == sizeof(BYTE))
            return uiBufferBytes;
        assert(m_pParent != NULL);        
        DatablockTemplate * pTemplate = m_pParent->GetTemplate();
        assert(pTemplate != NULL);
        UINT uiStride = pTemplate->GetInitialValueMemsetStride();
        assert(uiBufferBytes % uiStride == 0);
        return uiBufferBytes / uiStride;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a device memset count. </summary>
    ///
    /// <remarks>   crossbac, 7/10/2012. </remarks>
    ///
    /// <param name="pInitialValue">    [in,out] The buffer in bytes. </param>
    ///
    /// <returns>   The device memset count. </returns>
    ///-------------------------------------------------------------------------------------------------

    void *
    PCUBuffer::GetDeviceMemsetValue(
        __in void * pInitialValue
        )
    {
        if(pInitialValue != NULL)
            return pInitialValue;
        assert(m_pParent != NULL);        
        DatablockTemplate * pTemplate = m_pParent->GetTemplate();
        assert(pTemplate != NULL);
        return const_cast<void*>(pTemplate->GetInitialValue());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize mutable accelerator view. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="uiBufferBytes">    If non-null, the data. </param>
    /// <param name="pExtent">          [in,out] The bytes. </param>
    /// <param name="pModule">          [in,out] (optional)  If non-null, the module. </param>
    /// <param name="lpszBinding">      (optional) the binding. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

	UINT		
    PCUBuffer::__populateMutableAcceleratorView(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __out BOOL& bRequestOutstanding,
        __in void * pModule,
        __in const char * lpszBinding
        )
    {
        #pragma warning(disable:4127)
        CUresult res;
        assert(m_pBuffer != NULL);
        UNREFERENCED_PARAMETER(pModule);
        UNREFERENCED_PARAMETER(lpszBinding);
        assert(!(m_eAccessFlags & PT_ACCESS_IMMUTABLE));
        assert(pAsyncContext != NULL);
        
        bRequestOutstanding = FALSE;
        CUstream hStream = GetStream(pAsyncContext);
        UINT uiLogicalSizeBytes = GetLogicalExtentBytes();
        UINT uiRequestedBufferBytes = (uiBufferBytes == PBUFFER_DEFAULT_SIZE) ? 
                                        uiLogicalSizeBytes :
                                        uiBufferBytes;
        UINT uiPBufferSizeBytes = min(uiLogicalSizeBytes, uiRequestedBufferBytes);

        if(pExtent == NULL ||
           pExtent->lpvAddress == NULL || 
           uiPBufferSizeBytes == 0 ||
           m_pBuffer == NULL) {
            // empty or non existent buffer parameter.
            // WARN_IF_EMPTY(TRUE);
            return 0;
        }

        // Figure out our options for negotiating the transfer. In some cases, a device-side view
        // can be materialized using memset (common for derived block such as size-descriptors and
        // initial value blocks), which is faster than a PCI transfer. Whether or not we can perform
        // either operation asynchronously (which we also prefer) depends on whether we have a
        // stream object and the transfer method we choose. If we can memset, the presence of a stream
        // is sufficient: if we must do a PCI transfer, we also require that the source be page-locked.
        // An additional wrinkle is that we want to be able to fall back to synchrony if the asynchronous 
        // attempt fails, which complicates the control flow somewhat.

        void * pInitialData = pExtent ? pExtent->lpvAddress : NULL;
        UINT uiInitialDataBytes = pExtent ? pExtent->uiSizeBytes : NULL;
        BOOL bPageLockedSource = pExtent ? pExtent->bPinned : FALSE;
        BOOL bMemsettable = IsDeviceViewMemsettable(uiPBufferSizeBytes, pInitialData, uiInitialDataBytes);
        BOOL bAsynchronous = bPageLockedSource;
        BOOL bSynchronous = !bAsynchronous; // redundant, convenient.
        BOOL bSynchronousFallback = FALSE;

        ACQUIRE_CTX(m_pPSAcc);
        WARN_IF_SYNCHRONOUS(bSynchronous);
        m_bPopulated = FALSE;

        if(bMemsettable) {

            // materialize the view using memset

            UINT uiStride = GetDeviceMemsetStride(uiPBufferSizeBytes, uiInitialDataBytes);            
            UINT uiElements = GetDeviceMemsetCount(uiPBufferSizeBytes, uiInitialDataBytes);            
            VOID * pValue = GetDeviceMemsetValue(pInitialData);
            assert(uiStride != 0);
            assert(uiElements != 0);
            assert(pValue != NULL);
            UINT * puiValue = reinterpret_cast<UINT*>(pValue);
            UINT uiValue = *puiValue;
            BYTE * pucValue = reinterpret_cast<BYTE*>(pValue);
            BYTE ucValue = *pucValue;

            if(bAsynchronous || TRUE) {

                // use asynchronous memset

                if(uiStride == sizeof(UINT)) {

                    res = cuMemsetD32Async(PLATFORMBUFFER(m_pBuffer), uiValue, uiElements, hStream);
                    trace6("cuMemsetD32Async(%16llX, %d, %d, stream:%16llX)->res=%d\n", 
                           m_pBuffer, 
                           uiValue, 
                           uiElements, 
                           hStream,
                           res);
                    m_bPopulated = (res == CUDA_SUCCESS);
                    bSynchronousFallback = !m_bPopulated;

                } else if(uiStride == 1) {

                    res = cuMemsetD8Async(PLATFORMBUFFER(m_pBuffer), ucValue, uiElements, hStream);
                    trace6("cuMemsetD32Async(%16llX, %d, %d, stream:%16llX)->res=%d\n", 
                           m_pBuffer, 
                           ucValue, 
                           uiElements, 
                           hStream,
                           res);
                    m_bPopulated = (res == CUDA_SUCCESS);
                    bSynchronousFallback = !m_bPopulated;
                } 
            }

            bRequestOutstanding = m_bPopulated;
            if(!m_bPopulated) {

                // use *synchronous* memset either because the context forces it, 
                // or because our asynchronous attempt failed.
                WARN_IF_SYNCHRONOUS(TRUE);

                if(uiStride == sizeof(UINT)) {

                    trace4("cuMemsetD32(%16llX, %d, %d)\n", m_pBuffer, uiValue, uiElements);
                    _VI(cuMemsetD32(PLATFORMBUFFER(m_pBuffer), uiValue, uiElements), 0);

                } else if(uiStride == sizeof(BYTE)) {

                    trace4("cuMemsetD8(%16llX, %d, %d)\n", m_pBuffer, ucValue, uiElements);
                    _VI(cuMemsetD8(PLATFORMBUFFER(m_pBuffer), ucValue, uiElements), 0);

                }

                m_bPopulated = TRUE;
            }

        } else {

            // perform a host -> device transfer

            if(bAsynchronous || TRUE) {        

                // use asynchronous transfer from host to device
                res = cuMemcpyHtoDAsync(PLATFORMBUFFER(m_pBuffer), pInitialData, uiPBufferSizeBytes, hStream);
                trace6("cuMemcpyHtoDAsync(%16llX, %16llX, %d, stream:%16llX)->res=%d\n", 
                       m_pBuffer, 
                       pInitialData,
                       uiPBufferSizeBytes, 
                       hStream,
                       res);
                m_bPopulated = res == CUDA_SUCCESS;
                bSynchronousFallback = !m_bPopulated;
            }

            bRequestOutstanding = m_bPopulated;
            if(!m_bPopulated) {
                WARN_IF_SYNCHRONOUS(TRUE);
                trace4("cuMemcpyHtoD(%16llX, %16llX, %d)\n", m_pBuffer, pInitialData, uiPBufferSizeBytes);
                _VI(cuMemcpyHtoD(PLATFORMBUFFER(m_pBuffer), pInitialData, uiPBufferSizeBytes), 0);
                m_bPopulated = TRUE;                
            }

        }
        
        bRequestOutstanding &= (bAsynchronous && m_bPopulated && !bSynchronousFallback);

        RELEASE_CTX(m_pPSAcc);

        return (m_bPopulated) ? uiPBufferSizeBytes : 0;
        #pragma warning(default:4127)
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize immutable accelerator view. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="uiBufferBytes">    The buffer in bytes. </param>
    /// <param name="pExtent">          [in,out] If non-null, the extent. </param>
    /// <param name="pVModule">         [in,out] (optional)  If non-null, the module. </param>
    /// <param name="lpszBinding">      (optional) the binding. </param>
    ///
    /// <returns>   . </returns>
    ///
    /// ### <param name="pData">    [in,out] If non-null, the data. </param>
    ///
    /// ### <param name="nBytes">   The bytes. </param>
    ///-------------------------------------------------------------------------------------------------

    UINT		
    PCUBuffer::__populateImmutableAcceleratorView(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __out BOOL& bRequestOutstanding,
        __in void * pVModule,
        __in const char * lpszBinding
        )
    {
        // is this being used?
        UNREFERENCED_PARAMETER(uiBufferBytes);
        assert(false);

        #pragma warning(disable:4127)
        CUresult res;
        size_t uiBufSize = 0;
        bRequestOutstanding = FALSE;
        assert(pAsyncContext != NULL);
        assert(m_pBuffer != NULL);
        assert(m_eAccessFlags & PT_ACCESS_HOST_WRITE);
        assert(m_eAccessFlags & PT_ACCESS_IMMUTABLE);
        CUmodule pModule = (CUmodule) pVModule;
        CUstream hStream = GetStream(pAsyncContext);
        BOOL bAsynchronous = (hStream != NULL);
        WARN_IF_SYNCHRONOUS(!bAsynchronous);

        ACQUIRE_CTX(m_pPSAcc);
        trace("cuModuleGetGlobal()\n");
        _VI(cuModuleGetGlobal(PLATFORMBUFFERPTR(&m_pBuffer), &uiBufSize, pModule, lpszBinding), 0);
        trace5("cuMemcpyHtoDAsync(%16llX, %16llX, %d, stream=%16llX)\n", m_pBuffer, pExtent->lpvAddress, pExtent->uiSizeBytes, hStream);
        _VI(cuMemcpyHtoDAsync(PLATFORMBUFFER(m_pBuffer), pExtent->lpvAddress, pExtent->uiSizeBytes, hStream), 0);
        m_bPopulated = TRUE;
        bRequestOutstanding = (bAsynchronous && m_bPopulated);
        RELEASE_CTX(m_pPSAcc);

        return pExtent->uiSizeBytes;
        #pragma warning(default:4127)
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Device memcpy. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pDstBuffer">       [in,out] If non-null, the accelerator. </param>
    /// <param name="pSrcBuffer">       [in,out] If non-null, buffer for source data. </param>
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    PCUBuffer::Copy(
        __inout PBuffer *       pDstBuffer,
        __inout PBuffer *       pSrcBuffer,
        __in    AsyncContext *  pAsyncContext
        ) 
    {
        return Copy(pDstBuffer, 
                    pSrcBuffer, 
                    pAsyncContext,
                    min(pDstBuffer->GetLogicalExtentBytes(),
                        pSrcBuffer->GetLogicalExtentBytes()));
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Device memcpy. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pDstBuffer">       [in,out] If non-null, the accelerator. </param>
    /// <param name="pSrcBuffer">       [in,out] If non-null, buffer for source data. </param>
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    PCUBuffer::Copy(
        __inout PBuffer *       pDstBuffer,
        __inout PBuffer *       pSrcBuffer,
        __in    AsyncContext *  pAsyncContext,
        __in    UINT            uiCopyBytes
        ) 
    {
        #pragma warning(disable:4127)
        CUresult res;
        assert(m_pBuffer != NULL);
        assert(pAsyncContext != NULL);
        CUstream hStream = GetStream(pAsyncContext);
        BOOL bAsynchronous = TRUE;

        ACQUIRE_CTX(m_pPSAcc);

        if(pDstBuffer->ContextRequiresSync(pAsyncContext, OT_MEMCPY_TARGET))
            pDstBuffer->WaitOutstandingAsyncOperations(pAsyncContext, OT_MEMCPY_TARGET);
        if(pSrcBuffer->ContextRequiresSync(pAsyncContext, OT_MEMCPY_SOURCE))
            pSrcBuffer->WaitOutstandingAsyncOperations(pAsyncContext, OT_MEMCPY_SOURCE);

        CUAccelerator * pSrcAcc = (CUAccelerator*) pSrcBuffer->GetAccelerator();
        CUAccelerator * pDestAcc = (CUAccelerator*) pDstBuffer->GetAccelerator();
        CUAccelerator * pCtxAcc = (CUAccelerator*) (pAsyncContext ? pAsyncContext->GetDeviceContext() : NULL);
		UNREFERENCED_PARAMETER(pSrcAcc);
		UNREFERENCED_PARAMETER(pDestAcc);
		UNREFERENCED_PARAMETER(pCtxAcc);
        assert(pDestAcc == pDestAcc);
        assert(pDestAcc->LockIsHeld() || pDestAcc->IsHost());
        assert(pSrcAcc->LockIsHeld() || pSrcAcc->IsHost());
        assert(pCtxAcc->LockIsHeld());
        assert(pCtxAcc == NULL || pCtxAcc == pSrcAcc);
        void * pSrcDBuffer = pSrcBuffer->GetBuffer();
        void * pDstDBuffer = pDstBuffer->GetBuffer();
        size_t nSrcBytes = pDstBuffer->GetLogicalExtentBytes();
        size_t nDstBytes = pSrcBuffer->GetLogicalExtentBytes();
        size_t nMaxBytes = min(nSrcBytes, nDstBytes);
        size_t nBytes = (uiCopyBytes != 0) ? min(uiCopyBytes, nMaxBytes) : nMaxBytes;

        res = cuMemcpyDtoDAsync(PLATFORMBUFFER(pDstDBuffer), 
                                PLATFORMBUFFER(pSrcDBuffer), 
                                nBytes,
                                hStream);
        trace2("cuMemcpyDtoDAsync() -> res = %d\n", res);
        if(res != CUDA_SUCCESS) {
            bAsynchronous = FALSE;
            WARN_IF_SYNCHRONOUS(TRUE);
            _VI(cuMemcpyDtoD(PLATFORMBUFFER(pDstDBuffer), 
                             PLATFORMBUFFER(pSrcDBuffer), 
                             nBytes), FALSE);
            
        }

        pDstBuffer->MarkDirty(TRUE);
        BOOL bSuccess = (res == CUDA_SUCCESS);
        if(bSuccess && bAsynchronous) {
            pAsyncContext->Lock();
            SyncPoint * pSP = pAsyncContext->CreateSyncPoint(NULL);
            pDstBuffer->AddOutstandingDependence(pAsyncContext, OT_MEMCPY_TARGET, pSP);
            pSrcBuffer->AddOutstandingDependence(pAsyncContext, OT_MEMCPY_SOURCE, pSP);
            pAsyncContext->Unlock();
        }

        RELEASE_CTX(m_pPSAcc);

        return bSuccess;
        #pragma warning(default:4127)
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
    PCUBuffer::DeviceToDeviceTransfer(
        __inout PBuffer *       pDstBuffer,
        __in    AsyncContext *  pAsyncContext
        ) 
    {
        #pragma warning(disable:4127)
        CUresult res;
        assert(m_pBuffer != NULL);
        assert(pAsyncContext != NULL);
        CUstream hStream = GetStream(pAsyncContext);
        BOOL bAsynchronous = (hStream != NULL);
        WARN_IF_SYNCHRONOUS(!bAsynchronous);

        if(ContextRequiresSync(pAsyncContext, OT_MEMCPY_SOURCE))
            WaitOutstandingAsyncOperations(pAsyncContext, OT_MEMCPY_SOURCE);
        if(pDstBuffer->ContextRequiresSync(pAsyncContext, OT_MEMCPY_TARGET))
            pDstBuffer->WaitOutstandingAsyncOperations(pAsyncContext, OT_MEMCPY_TARGET);

        // first attempt the transfer in the CUDA device context of the destination accelerator, which
        // should typically be the one associated with the async context. If that fails, attempt it
        // from the CUDA context of the source. If that also fails, fail the function call and let the
        // calling layer handle the error by routing the transfer through host memory. 
        
        CUAccelerator * pSrcAcc = (CUAccelerator*) m_pAccelerator;
        CUAccelerator * pDestAcc = (CUAccelerator*) pDstBuffer->GetAccelerator();
        assert(pDestAcc != NULL);
        assert(pDestAcc->LockIsHeld());
        assert(pSrcAcc->LockIsHeld());
        void * pDstDBuffer = pDstBuffer->GetBuffer();
        size_t nBytes = pDstBuffer->GetLogicalExtentBytes();
        BOOL bXferSucceeded = FALSE;
        pDstBuffer->MarkDirty(TRUE);

        if(!bXferSucceeded) {

            // attempt the tranfer from the dst perspective using P2P
            ACQUIRE_CTX(pDestAcc);
            assert(pDestAcc->SupportsDeviceToDeviceTransfer(pSrcAcc));
            CUcontext dstContext = (CUcontext) pDestAcc->GetContext();        
            CUcontext srcContext = (CUcontext) pSrcAcc->GetContext();        
            res = cuMemcpyPeerAsync(PLATFORMBUFFER(pDstDBuffer),
                                    dstContext,
                                    PLATFORMBUFFER(m_pBuffer),
                                    srcContext,
                                    nBytes,
                                    hStream);
            trace5("cuMemcpyPeerAsync(dstCtx=%16llX, hStream=%16llX, cnt=%d)->res=%d\n",
                    dstContext,
                    hStream,
                    nBytes,
                    res);
            bXferSucceeded = (res == CUDA_SUCCESS);
            if(res != CUDA_SUCCESS) {
                WARN("WARNING...cuMemcpyPeerAsync failed!");
            }

            if(bXferSucceeded && bAsynchronous) {
                pAsyncContext->Lock();
                SyncPoint * pSP = pAsyncContext->CreateSyncPoint(NULL);
                pDstBuffer->AddOutstandingDependence(pAsyncContext, OT_MEMCPY_TARGET, pSP);
                AddOutstandingDependence(pAsyncContext, OT_MEMCPY_SOURCE, pSP);
                pAsyncContext->Unlock();
            }
            RELEASE_CTX(pDestAcc);
        }

        if(!bXferSucceeded) {
            // attempt P2P from src perspective
            ACQUIRE_CTX(m_pPSAcc);
            assert(pSrcAcc->SupportsDeviceToDeviceTransfer(pDestAcc));
            CUcontext dstContext = (CUcontext) pDestAcc->GetContext();
            CUcontext srcContext = (CUcontext) pSrcAcc->GetContext();
            res = cuMemcpyPeerAsync(PLATFORMBUFFER(pDstDBuffer),
                                    dstContext,
                                    PLATFORMBUFFER(m_pBuffer),
                                    srcContext,
                                    nBytes,
                                    hStream);
            trace5("cuMemcpyPeerAsync(dstCtx=%16llX, hStream=%16llX, cnt=%d)->res=%d\n",
                    dstContext,
                    hStream,
                    nBytes,
                    res);
            bXferSucceeded = (res == CUDA_SUCCESS);
            if(res != CUDA_SUCCESS) {
                WARN("WARNING...cuMemcpyPeerAsync failed!");
            }
        
            if(bXferSucceeded && bAsynchronous) {
                pAsyncContext->Lock();
                SyncPoint * pSP = pAsyncContext->CreateSyncPoint(NULL);
                pDstBuffer->AddOutstandingDependence(pAsyncContext, OT_MEMCPY_TARGET, pSP);
                AddOutstandingDependence(pAsyncContext, OT_MEMCPY_SOURCE, pSP);
                pAsyncContext->Unlock();
            }
            RELEASE_CTX(m_pPSAcc);
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
        #pragma warning(default:4127)
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
    PCUBuffer::CreateMutableBuffer(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferSizeBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __in char * strDebugBufferName, 
        __in bool bByteAddressable                                                    
        )
    {
        #pragma warning(disable:4127)
        CUresult res;
        CUdeviceptr pDeviceBuffer = NULL;

        // finalize dimensions can return 0. In such a case we must still allocate something, even if
        // it's empty to make it bindable to device-side resources. Hence, we distinguish between alloc
        // size and transfer size. 
        assert(m_pBuffer == NULL);
        assert(!IsDimensionsFinalized());
        FinalizeDimensions(bByteAddressable, uiBufferSizeBytes);
        UINT uAllocCount = GetAllocationExtentBytes();
        assert(uAllocCount > 0);

        ACQUIRE_CTX(m_pPSAcc);

        VOID* pInitialBufferContents = pExtent ? pExtent->lpvAddress : NULL;        

        // we are not creating a buffer that requires a page-locked host view. This had better 
        // because we know that no host-side view of the buffer is ever required, meaning we will
        // never attempt an asynchronous transfer between the host and device for this buffer. 
        // In such a case, it is sufficient to call cuMemAlloc--we hope for this to be the common
        // case, since allocating too much page-locked memory can cause serious performance degradation,
        // and finding the the threshold at which that occurs is non-trivial, since it depends
        // on system-wide behavior. 

        approfile_enter(CreateMutableBuffer);
        //check_critical_path_alloc(uAllocCount);
        if(Runtime::GetCriticalPathAllocMode() && 
            Scheduler::GetRunningGraphCount() &&
            (Scheduler::GetRunningGraphCount() == Scheduler::GetLiveGraphCount())) { 
	        PTask::Runtime::MandatoryInform("cuMemAlloc(%d)!\n", uAllocCount);         
        }

        // PTask::Runtime::MandatoryInform("cuMemAlloc(%d)!\n", uAllocCount);      
        res = cuMemAlloc(&pDeviceBuffer, uAllocCount);
        if(CUDA_SUCCESS != res) {

            // we really should handle this by unwinding, forcing a GC, and then retrying...however,
            // generally when this happens it really means there is an abusive structure in the graph, and
            // forcing a GC would just make it harder to diagnose. for now, make a bunch of noise, pause
            // the graph, and *then* declare an unrecoverable error condition. 
            
            PTask::Runtime::MandatoryInform("%s::%s(line %d): cuMemAlloc(%d) failed, res=%d...checking GC status,"
                                            " and retrying after forced gc sweep for uiMemSpace=%d...\n", 
                                            __FILE__,
                                            __FUNCTION__,
                                            __LINE__,
                                            uAllocCount,
                                            m_pPSAcc->GetMemorySpaceId(),
                                            res);

            // start releasing device memory more aggressively...
            PTask::Runtime::SetAggressiveReleaseMode(TRUE);

            // report on the climate if verbose
            if(PTask::Runtime::IsVerbose()) 
                GarbageCollector::Report();

            // force a GC sweep for just the target memspace
            DWORD dwStartGC = GetTickCount();
            GarbageCollector::ForceGC(m_pPSAcc->GetMemorySpaceId());
            DWORD dwEndGC = GetTickCount();

            // and then try again...        
            res = cuMemAlloc(&pDeviceBuffer, uAllocCount);
            if(CUDA_SUCCESS != res) {

                // if we still fail, force release for all sticky device buffers
                // that have been marked releasable on a predicate and try again.
                Port::ForceStickyDeviceBufferRelease();
                res = cuMemAlloc(&pDeviceBuffer, uAllocCount);
            }
   
            if(CUDA_SUCCESS != res) {

                m_pPSAcc->DumpAllocationStatistics(std::cout);
                Scheduler::PauseAndReportGraphStates();
                DatablockProfiler::Report(std::cerr);
                PTask::Runtime::HandleError("%s::%s(line %d): cuMemAlloc(%d) failed, res=%d\n", 
                                            __FILE__,
                                            __FUNCTION__,
                                            __LINE__,
                                            uAllocCount,
                                            res);

                RELEASE_CTX(m_pPSAcc);
                return PTASK_ERR;
            } else {
                PTask::Runtime::MandatoryInform("%s::%s(line %d): forced GC (%d ms) recovery successful!\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                __LINE__,
                                                dwEndGC-dwStartGC);
            }
        }
        approfile_exit(CreateMutableBuffer, uAllocCount);

        trace4("cuMemAlloc(dev:%d, %d)->%16llX\n", 
                m_pPSAcc->GetAcceleratorId(), 
                uAllocCount, 
                pDeviceBuffer);

        m_bDeviceBufferOwned = TRUE;
        m_pBuffer = (void*) pDeviceBuffer;
        m_pPSAcc->RecordAllocation(m_pBuffer, FALSE, uAllocCount);

        if(pInitialBufferContents) {

            BOOL bRequestsOutstanding = FALSE;
            __populateMutableAcceleratorView(pAsyncContext, 
                                             uAllocCount,
                                             pExtent,
                                             bRequestsOutstanding,
                                             NULL,
                                             strDebugBufferName);
            if(bRequestsOutstanding) {
                AddOutstandingDependence(pAsyncContext, OT_MEMCPY_TARGET);
            }
        } else {
            if(PTask::Runtime::GetPBufferClearOnCreatePolicy()) {
                cuMemsetD8(pDeviceBuffer, 0, uAllocCount);
            }
        }

        RELEASE_CTX(m_pPSAcc);

        return CreateBindableObjects(strDebugBufferName);
        #pragma warning(default:4127)
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return true if the derived class supports a memset API. </summary>
    ///
    /// <remarks>   crossbac, 8/14/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    PCUBuffer::SupportsMemset(
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
    PCUBuffer::FillExtent(
        __in int nValue, 
        __in size_t szExtentBytes
        )
    {
        size_t szFilled = 0;
        CUdeviceptr pDeviceBuffer = reinterpret_cast<CUdeviceptr>(m_pBuffer);
        size_t szExtentSizeBytes = GetAllocationExtentBytes();
        size_t szRequestExtent = szExtentBytes ? szExtentBytes : szExtentSizeBytes;
        size_t szSetExtent = min((size_t)szExtentSizeBytes, szRequestExtent);
        ACQUIRE_CTX(m_pPSAcc);
        CUresult res = cuMemsetD8(pDeviceBuffer, (unsigned char)nValue, szSetExtent);
        szFilled = res == CUDA_SUCCESS ? szSetExtent : 0;
        RELEASE_CTX(m_pPSAcc);
        return szFilled;
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
    PCUBuffer::CreateImmutableBuffer(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferSizeBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __in char * strDebugBufferName, 
        __in bool bByteAddressable
        )
    {
        // because we don't actually perform a transfer here (yet), 
        // there is no need to use the async context object!
        UNREFERENCED_PARAMETER(pAsyncContext);

        // finalize dimensions can return 0. In such a case we must still allocate something, even if
        // it's empty to make it bindable to device-side resources. Hence, we distinguish between alloc
        // size and transfer size. 
        
        FinalizeDimensions(bByteAddressable, uiBufferSizeBytes);
        UINT uAllocCount = GetAllocationExtentBytes();
        UNREFERENCED_PARAMETER(uAllocCount);
        UINT uXferCount = GetLogicalExtentBytes();

        // there is nothing for us to do here, since we don't allocate a CUdeviceptr for constants--we
        // get a buffer by calling cuModuleGetGlobal, which requires us to know who the consumer is.
        // When allocating constant datablocks, we don't necessarily know that yet, so we are forced to
        // buffer a copy of the user's init data (if it is given). 
        
        VOID* pInitialBufferContents = pExtent ? pExtent->lpvAddress : NULL;
        UINT uiInitialContentsSizeBytes = pExtent ? pExtent->uiSizeBytes : 0;
        if(pInitialBufferContents) {

            // If the user gave us a size for the initial data buffer, make sure we don't overrun it or or
            // the newly allocated buffer. If the size parameter was defaulted (0), it means we were
            // supposed to infer the size, so uCount is the best we can do. However, (see above for
            // details) the best we can do with the initial data is force it into the host buffer. 
            // Since this call might well be providing the host memory space buffer as initial contents
            // we may have no work to do at all. 

            PBuffer * pPSBuffer = m_pParent->GetPlatformBuffer(HOST_MEMORY_SPACE_ID, m_nChannelIndex, m_pAccelerator);
            void * pHostBuffer = pPSBuffer->GetBuffer();
            assert(pHostBuffer != NULL);

            if(pHostBuffer != pInitialBufferContents) {
                UINT uCopyBytes = uiInitialContentsSizeBytes ? 
                                    min(uiInitialContentsSizeBytes, uXferCount) : 
                                    uXferCount;
                memcpy(pHostBuffer, pInitialBufferContents, uCopyBytes);   
            }
        }

        return CreateBindableObjects(strDebugBufferName);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates immutable bindable objects. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="szname">   [in] If non-null, the a string used to label the object that can be
    ///                         used for debugging. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    PCUBuffer::CreateBindableObjectsImmutable(
        char * szname
        )
    {
        UNREFERENCED_PARAMETER(szname);
        PBuffer * pPSBuffer = m_pParent->GetPlatformBuffer(HOST_MEMORY_SPACE_ID, m_nChannelIndex, m_pAccelerator);
        void * pHostBuffer = pPSBuffer->GetBuffer();
        BINDABLEOBJECT view;
        view.vptr = pHostBuffer;
        m_mapBindableObjects[BVT_ACCELERATOR_IMMUTABLE] = view;
        return S_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates readable bindable objects. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="szname">   [in] If non-null, the a string used to label the object that can be
    ///                         used for debugging. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    PCUBuffer::CreateBindableObjectsReadable( 
        char * szname
        )
    {
        UNREFERENCED_PARAMETER(szname);
        BINDABLEOBJECT view;
        view.vptr = m_pBuffer;
        m_mapBindableObjects[BVT_ACCELERATOR_READABLE] = view;
        return PTASK_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates writable bindable objects. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="szname">   [in] If non-null, the a string used to label the object that can be
    ///                         used for debugging. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    PCUBuffer::CreateBindableObjectsWriteable(
        char * szname
        )
    {
        UNREFERENCED_PARAMETER(szname);
        BINDABLEOBJECT view;
        view.vptr = m_pBuffer;
        m_mapBindableObjects[BVT_ACCELERATOR_WRITEABLE] = view;
        return PTASK_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Force synchronize. When returning a pointer to datablock data to the user it is
    ///             critical that any outstanding operations on that datablock be complete. This
    ///             method syncs any outstanding operations. The caller should have the containing
    ///             datablock and the accelerator locked already!
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    PCUBuffer::ForceSynchronize(
        VOID
        )
    {
        #pragma warning(disable:4127)
        assert(FALSE && "Don't call this!");
        //RetireDependenceFrontier(m_vOutstandingReads, NULL);
        //RetireDependenceFrontier(m_vOutstandingWrites, NULL);
        std::set<CUstream> vSyncedStreams;
        while(m_vRetired.size() != 0) {
            CUresult res;
            AsyncDependence * pDep = m_vRetired.back(); 
            m_vRetired.pop_back();
            CUstream hStream = (CUstream) pDep->GetPlatformContextObject();
            BOOL bSyncRequired = vSyncedStreams.find(hStream) == vSyncedStreams.end();
            if(bSyncRequired) {
                pDep->Lock();
                AsyncContext * pAsyncContext = pDep->GetContext();
                Accelerator * pDeviceCtxt = pAsyncContext->GetDeviceContext();
                ACQUIRE_CTX(pDeviceCtxt);
                _VV(pDeviceCtxt, cuStreamSynchronize(hStream));
                trace2("cuStreamSynchronize(%16llX)\n", hStream);
                RELEASE_CTX(pDeviceCtxt);
                pDep->Unlock();
                vSyncedStreams.insert(hStream);
            }
            pDep->Release();
        }
        ClearDependences(NULL);
        #pragma warning(default:4127)
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a stream. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    ///
    /// <returns>   The stream. </returns>
    ///-------------------------------------------------------------------------------------------------

    CUstream 
    PCUBuffer::GetStream(
        AsyncContext * pAsyncContext
        )
    {
        return (pAsyncContext != NULL) ? 
            (CUstream) pAsyncContext->GetPlatformContextObject() : 
            NULL;
    }

};

#endif // CUDA_SUPPORT