//--------------------------------------------------------------------------------------
// File: PCLBuffer.cpp
// Maintainer: crossbac@microsoft.com
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#ifdef OPENCL_SUPPORT

#include "PCLBuffer.h"
#include <stdio.h>
#include <crtdbg.h>
#include <assert.h>
#include "claccelerator.h"
#include "datablock.h"
#include "PTaskRuntime.h"
#include "oclhdr.h"
#include "MemorySpace.h"

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
    /// <param name="pAllocator">       [in,out] If non-null, the allocator. </param>
    /// <param name="uiUID">            (optional) the uid. </param>
    ///-------------------------------------------------------------------------------------------------

    PCLBuffer::PCLBuffer(
        Datablock * pParent,
        BUFFERACCESSFLAGS f, 
        UINT nChannelIndex,
        Accelerator * p, 
        Accelerator * pAllocator,
        UINT uiUID
        ) : PBuffer(pParent, f, nChannelIndex, p, pAllocator, uiUID)
    {
        m_pBuffer = NULL;
        m_bPopulated = FALSE;
        m_bScalarBinding = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    PCLBuffer::~PCLBuffer(
        VOID
        )
    {
        if(m_pBuffer) {
            clReleaseMemObject((cl_mem) m_pBuffer);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize immutable accelerator view. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="uiBufferSizeBytes">    If non-null, the data. </param>
    /// <param name="pInitData">            [in,out] The bytes. </param>
    /// <param name="pVModule">             [in,out] If non-null, the v module. </param>
    /// <param name="lpszBinding">          (optional) the binding. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT		
    PCLBuffer::__populateImmutableAcceleratorView(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferSizeBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __out BOOL&             bOutstanding,
        __in void * pVModule,
        __in const char * lpszBinding
        )
    {
        UNREFERENCED_PARAMETER(uiBufferSizeBytes);
        UNREFERENCED_PARAMETER(pAsyncContext);
        UNREFERENCED_PARAMETER(pVModule);
        UNREFERENCED_PARAMETER(lpszBinding);
        assert(pExtent != NULL);                  // populate with what?
        assert(pExtent->lpvAddress != NULL);      // need some bytes!
        assert(pExtent->uiSizeBytes != 0);        // need some bytes!

        bOutstanding = FALSE;
        if(m_pParent->IsScalarParameter() || m_pParent->IsScalarParameter()) {

            // there is no device-side work to do here, since we don't allocate a CUdeviceptr for
            // constants. if we actually have init data, we are forced to ensure that the host buffers a
            // copy of the user's init data (if it is given). 
                         
            PBuffer * pPSBuffer = m_pParent->GetPlatformBuffer(HOST_MEMORY_SPACE_ID, m_nChannelIndex);
            void * pHostBuffer = pPSBuffer->GetBuffer();
            assert(pHostBuffer != NULL);
            if(pHostBuffer != pExtent->lpvAddress) {
                assert((pPSBuffer->GetElementCount() * pPSBuffer->GetElementStride()) >= pExtent->uiSizeBytes);
                memcpy(pHostBuffer, pExtent->lpvAddress, pExtent->uiSizeBytes);
            }
                
            // we require m_pBuffer to stay null, but set flags to let us
            // defer freshness tests and materialization to the host PBuffer.
            m_bPopulated = TRUE;
            m_bScalarBinding = TRUE;
            m_pBuffer = NULL; 

        } else {

            // this buffer is going to be bound to a constant buffer, whose binding we will get at dispatch
            // time by calling clEnqueueWriteBuffer. Populate it since we actually have the data to
            // materialize in constant memory on the device. 

            cl_int resTransfer = CL_SUCCESS;

            m_pAccelerator->Lock();
            assert(m_pBuffer != NULL);

            cl_command_queue cqCommandQueue = ((CLAccelerator*)m_pAccelerator)->GetQueue();
            resTransfer = clEnqueueWriteBuffer(cqCommandQueue, 
                                                (cl_mem) m_pBuffer, 
                                                CL_FALSE, 0, 
                                                pExtent->uiSizeBytes, 
                                                pExtent->lpvAddress, 
                                                0, NULL, NULL);

            m_bPopulated = (resTransfer == CL_SUCCESS);
            m_pAccelerator->Unlock();

            if(m_pBuffer == NULL || resTransfer != CL_SUCCESS) {
                PTask::Runtime::HandleError("%s: OpenCL API failure (%s, res=%d)\n",
                                            __FUNCTION__,
                                            "clEnqueueWriteBuffer",
                                            resTransfer);
                return PTASK_ERR;
            }
        }

        return PTASK_OK;
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
    ///
    ///-------------------------------------------------------------------------------------------------

    PTRESULT	
    PCLBuffer::CreateImmutableBuffer(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferSizeBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __in char * strDebugBufferName, 
        __in bool bByteAddressable
        )
    {
        UNREFERENCED_PARAMETER(bByteAddressable);
        HRESULT hr = S_OK;

        assert(m_pBuffer == NULL && !m_bScalarBinding);

        // finalize dimensions can return 0. In such a case we must still allocate something, even if
        // it's empty to make it bindable to device-side resources. Hence, we distinguish between alloc
        // size and transfer size. 
        FinalizeDimensions(bByteAddressable, uiBufferSizeBytes);
        UINT uAllocCount = GetAllocationExtentBytes();

        // check whether the parent datablock is a scalar parameter to a kernel function call, or a
        // byref buffer/global constant buffer. The two cases must be handled differently, since
        // scalars require no explicit creation of device-side buffers. 
         
        if(m_pParent->IsScalarParameter() || m_pParent->IsScalarParameter()) {

            if(pExtent == NULL || pExtent->lpvAddress == NULL) {
                // there is no work to do here, since we don't allocate a deviceptr for
                // constants and if we don't have init data, we can't force creation of the
                // device-side backing buffer.
                return hr;
            }

        } else {

            // this buffer is going to be bound to a constant buffer, whose binding we will get at dispatch
            // time by calling clCreateBuffer. So create the buffer, and populate it if we actually have
            // the data to materialize in constant memory on the device. 

            cl_int resBufferCreate = CL_SUCCESS;

            m_pAccelerator->Lock();
            CLAccelerator * pAccelerator = (CLAccelerator*) m_pAccelerator;
            cl_context pContext = (cl_context) pAccelerator->GetContext();
            m_pBuffer = clCreateBuffer(pContext, CL_MEM_READ_ONLY, uAllocCount, NULL, &resBufferCreate);
            m_pAccelerator->Unlock();

            // at a minimum, the buffer create must have succeeded. if we tried to do the transer, then
            // resTransfer will indicate how it went, but is initialized to CL_SUCCESS so the test of it
            // will pass if no transfer attempt was made. 
            
            if(m_pBuffer == NULL || resBufferCreate != CL_SUCCESS) {
                PTask::Runtime::HandleError("%s: OpenCL API failure (%s, res=%d)\n",
                                            __FUNCTION__,
                                            "clCreateBuffer",
                                            resBufferCreate);
                return E_FAIL;
            }
        }

        VOID * pInitialBufferContents = pExtent ? pExtent->lpvAddress : NULL;
        if(pInitialBufferContents) {
            
            UINT uXfer;
            BOOL bOutstandingAsyncOps = FALSE;
            uXfer = __populateImmutableAcceleratorView(
                        pAsyncContext, 
                        uiBufferSizeBytes,
                        pExtent,
                        bOutstandingAsyncOps,
                        NULL,
                        strDebugBufferName);

            assert(!bOutstandingAsyncOps);
            if(uXfer == 0) {

                PTask::Runtime::HandleError("%s: xfer failure (%s, res=%d)\n",
                                            __FUNCTION__,
                                            "__populateImmutableAcceleratorView",
                                            uXfer);
                return E_FAIL;
            }
        }

        hr = CreateBindableObjects(strDebugBufferName);
        return hr;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a bindable objects immutable. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="szname">   [in] If non-null, the a string used to label the object that can be
    ///                         used for debugging. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    PCLBuffer::CreateBindableObjectsImmutable(
        char * szname
        )
    {
        // because we get a const buffer pointer
        // from the module at bind time, there is nothing
        // to do here for immutable buffers.
        UNREFERENCED_PARAMETER(szname);
        BINDABLEOBJECT view;
        view.vptr = m_pBuffer;
        m_mapBindableObjects[BVT_ACCELERATOR_IMMUTABLE] = view;
        return PTASK_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a bindable objects readable. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="szname">   [in] If non-null, the a string used to label the object that can be
    ///                         used for debugging. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    PCLBuffer::CreateBindableObjectsReadable( 
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
    /// <summary>   Creates a bindable objects writeable. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="szname">   [in] If non-null, the a string used to label the object that can be
    ///                         used for debugging. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    PCLBuffer::CreateBindableObjectsWriteable(
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
    /// <summary>   Materialize mutable accelerator view. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="uiBufferSizeBytes">    If non-null, the data. </param>
    /// <param name="pExtent">              [in,out] The bytes. </param>
    /// <param name="bOutstandingOps">      [in,out] The outstanding ops. </param>
    /// <param name="pModule">              [in,out] (optional)  If non-null, the module. </param>
    /// <param name="lpszBinding">          (optional) the binding. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

	UINT		
    PCLBuffer::__populateMutableAcceleratorView(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferSizeBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __out BOOL& bOutstandingOps,
        __in void * pModule,
        __in const char * lpszBinding
        )
    {
        UNREFERENCED_PARAMETER(uiBufferSizeBytes);
        UNREFERENCED_PARAMETER(pAsyncContext);
        UNREFERENCED_PARAMETER(pModule);
        UNREFERENCED_PARAMETER(lpszBinding);
        assert(m_eAccessFlags & PT_ACCESS_HOST_WRITE);
        assert(!(m_eAccessFlags & PT_ACCESS_IMMUTABLE));
        assert(m_pBuffer != NULL);
        assert(pExtent != NULL);
        assert(pExtent->lpvAddress != NULL);
        assert(pExtent->uiSizeBytes != 0);

        cl_int res;
        bOutstandingOps = FALSE;
        m_pAccelerator->Lock();
        cl_command_queue pQueue = (cl_command_queue) ((CLAccelerator*)m_pAccelerator)->GetQueue();
        cl_mem pBuffer = (cl_mem) m_pBuffer;
        res = clEnqueueWriteBuffer(pQueue, pBuffer, CL_FALSE, 0, pExtent->uiSizeBytes, pExtent->lpvAddress, 0, NULL, NULL);
        if (res != CL_SUCCESS) {
            PTask::Runtime::HandleError("%s: Error in clEnqueueWriteBuffer (res=%d)\n",
                                        __FUNCTION__,
                                        res);
            m_pAccelerator->Unlock();
            return 0;
        } 

        m_bPopulated = TRUE;
        m_pAccelerator->Unlock();
        return res == CL_SUCCESS ? (UINT)pExtent->uiSizeBytes : 0;
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
    PCLBuffer::__populateHostView(
        __in  AsyncContext * pAsyncContext,
        __in  HOSTMEMORYEXTENT * pExtent,
        __in  BOOL bForceSynchronous,
        __out BOOL &bRequestOutstanding
        )
    {
        assert(m_pBuffer != NULL);
        assert(pExtent != NULL);
        assert(pExtent->lpvAddress != NULL);
        assert(pExtent->uiSizeBytes != 0);

        bRequestOutstanding = FALSE;
        m_pAccelerator->Lock();
        CLAccelerator * pAccelerator = (CLAccelerator*) m_pAccelerator;
        cl_command_queue pQueue = (cl_command_queue) pAccelerator->GetQueue();
        cl_mem pBuffer = (cl_mem) m_pBuffer;

        // ------------------------------------------------------------------------
        // no relative ordering guarantees are given between kernel launches
        // on a command queue, and memory object manipulations. (other than that
        // they start in order). In particular, if we are interested in reading
        // memory produced on the accelerator (the common case for this function)
        // there is no guarantee that the kernel producing that data has finished
        // when the buffer read occurs, unless we either:
        // a) explicitly wait for the outstanding kernel using an event shared
        //    between the reader and the launch code
        // b) wait for *all* activity on this command queue to complete. 
        // Of course, a would be more performant. For now, do b). 
        // ------------------------------------------------------------------------
        cl_int ciErr;
        clFinish(pQueue);
        UNREFERENCED_PARAMETER(bForceSynchronous); // currently we always block (CL_TRUE below)
        UNREFERENCED_PARAMETER(pAsyncContext); // currently we always block (CL_TRUE below)
        ciErr = clEnqueueReadBuffer(pQueue, pBuffer, CL_TRUE, 0, pExtent->uiSizeBytes, pExtent->lpvAddress, 0, NULL, NULL);
        if(CL_SUCCESS != ciErr) {
            PTask::Runtime::HandleError("%s: Error in clEnqueueReadBuffer (res=%d)\n", __FUNCTION__, ciErr);
            m_pAccelerator->Unlock();
            return 0;
        }
        m_pAccelerator->Unlock();
        return (UINT) pExtent->uiSizeBytes;
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
    PCLBuffer::CreateMutableBuffer(
        __in AsyncContext * pAsyncContext,
        __in UINT uiBufferSizeBytes,
        __in HOSTMEMORYEXTENT * pExtent,
        __in char * strDebugBufferName, 
        __in bool bByteAddressable                                                    
        )
    {
        UNREFERENCED_PARAMETER(pAsyncContext);

        // finalize dimensions can return 0. In such a case we must still allocate something, even if
        // it's empty to make it bindable to device-side resources. Hence, we distinguish between alloc
        // size and transfer size. 
        assert(m_pBuffer == NULL);
        FinalizeDimensions(bByteAddressable, uiBufferSizeBytes);
        UINT uAllocCount = GetAllocationExtentBytes();
        UINT uXferCount = GetLogicalExtentBytes();

        cl_int ciErr1;
        CLAccelerator * pAccelerator = (CLAccelerator*) m_pAccelerator;
        cl_context pContext = (cl_context) pAccelerator->GetContext();
        cl_mem pEndpoint = clCreateBuffer(pContext, CL_MEM_READ_ONLY, uAllocCount, NULL, &ciErr1);
        if(ciErr1 != CL_SUCCESS) {
            PTask::Runtime::HandleError("%s: clCreateBuffer(%d bytes) failed (res=%d)\n", 
                                        __FUNCTION__,
                                        uAllocCount,
                                        ciErr1);
            m_pAccelerator->Unlock();
            return PTASK_ERR;
        }
        m_pBuffer = pEndpoint;

        VOID * pInitialBufferContents = pExtent ? pExtent->lpvAddress : NULL;
        UINT uiInitialContentsSizeBytes = pExtent ? pExtent->uiSizeBytes : 0;        
        if(pInitialBufferContents != NULL) {

            // If the user gave us a size for the initial data buffer, make sure we don't overrun it or or
            // the newly allocated buffer. If the size parameter was defaulted (0), it means we were
            // supposed to infer the size, so uCount is the best we can do. 
            
            UINT uTransferBytes = uiInitialContentsSizeBytes ? 
                                        min(uiInitialContentsSizeBytes, uXferCount) : 
                                        uXferCount;

            cl_command_queue cqCommandQueue = pAccelerator->GetQueue();
            ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, 
                                          pEndpoint, 
                                          CL_FALSE, 
                                          0, 
                                          uTransferBytes, 
                                          pInitialBufferContents, 
                                          0, NULL, NULL);

            if(ciErr1 != CL_SUCCESS) {
                PTask::Runtime::HandleError("%s: clEnqueueWriteBuffer(%d bytes) failed (res=%d)\n", 
                                            __FUNCTION__,
                                            uTransferBytes,
                                            ciErr1);
                m_pAccelerator->Unlock();
                return PTASK_ERR;
            }
            m_bPopulated = TRUE;
        }

        return CreateBindableObjects(strDebugBufferName);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Force synchronize. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void                
    PCLBuffer::ForceSynchronize(
        VOID
        ) 
    {
        // never necessary at the moment: we are always synchronous!
    }
};

#endif