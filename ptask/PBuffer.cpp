//--------------------------------------------------------------------------------------
// File: PBuffer.cpp
// Maintainer: crossbac@microsoft.com
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "PBuffer.h"
#include "PBufferProfiler.h"
#include "accelerator.h"
#include "PTaskRuntime.h"
#include "datablock.h"
#include "Tracer.h"
#include "task.h"
#include "AsyncContext.h"
#include "AsyncDependence.h"
#include "datablocktemplate.h"
#include "nvtxmacros.h"
#include <assert.h>

#ifdef DEBUG
#define COMPLAIN(x) PTask::Runtime::MandatoryInform(x)
#define CHECK_DEPENDENCE_INVARIANTS() CheckDependenceInvariants()
#define VERIFY_SYNC_DEPENDENCES(r, w, deps) {                     \
    BOOL bAllResolved = FALSE;                                    \
    if(!r && !w) {                                                \
        bAllResolved = TRUE;                                      \
        if(deps) {                                                \
            std::set<AsyncDependence*>::iterator si;              \
            for(si=deps->begin(); si!=deps->end(); si++) {        \
                AsyncDependence* pDep = (*si);                    \
                pDep->Lock();                                     \
                bAllResolved &= !pDep->QueryOutstandingFlag();    \
                pDep->Unlock();                                   \
    } } }                                                         \
    assert(r || w || bAllResolved); }

#define COLLECT_SYNC_DEPENDENCES(sync, rf, wf, deps) {            \
    if(sync && deps) {                                            \
        deque<AsyncDependence*>* pFrontier = wf.size() ?          \
            &wf : &rf;                                            \
        std::deque<AsyncDependence*>::iterator si;                \
        for(si=pFrontier->begin(); si!=pFrontier->end(); si++) {  \
            if(deps->find(*si) != deps->end())                    \
                COMPLAIN("duplicate deps for PBuffer!\n");        \
            deps->insert(*si);                                    \
        } } }
#else
#define CHECK_DEPENDENCE_INVARIANTS()
#define VERIFY_SYNC_DEPENDENCES(rf, wf, deps)
#define COLLECT_SYNC_DEPENDENCES(sync, rf, wf, deps)
#endif
#ifdef PROFILE_PBUFFERS
#define create_pbuffer_profiler(x)      { (x) = new PBufferProfiler(); }
#define destroy_pbuffer_profiler(x)     { if(x!=NULL) { delete ((PBufferProfiler*)x); x=NULL; } }
#else
#define create_pbuffer_profiler(x)      { (x) = NULL; }
#define destroy_pbuffer_profiler(x)
#endif

using namespace std;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pParentDatablock">         [in,out] If non-null, the parent datablock. </param>
    /// <param name="bufferAccessFlags">        The buffer access flags. </param>
    /// <param name="nChannelIndex">            Zero-based index of the datablock channel this
    ///                                         PBuffer is backing. </param>
    /// <param name="pAccelerator">             (optional) [in,out] If non-null, the accelerator. </param>
    /// <param name="pAllocatingAccelerator">   (optional) [in,out] If non-null, the allocating
    ///                                         accelerator. </param>
    /// <param name="uiUniqueIdentifier">       (optional) unique identifier. </param>
    ///-------------------------------------------------------------------------------------------------

    PBuffer::PBuffer(
        Datablock * pParentDatablock,
        BUFFERACCESSFLAGS bufferAccessFlags, 
        UINT nChannelIndex,
        Accelerator * pAccelerator, 
        Accelerator * pAllocatingAccelerator,
        UINT uiUniqueIdentifier
        )
    {
        assert(pParentDatablock != NULL);
        assert(nChannelIndex >= 0 && nChannelIndex < NUM_DATABLOCK_CHANNELS);
        m_nChannelIndex = nChannelIndex;
        m_pParent = pParentDatablock;
        m_pAccelerator = pAccelerator;
        m_uiId = uiUniqueIdentifier;
        m_eAccessFlags = bufferAccessFlags;
        m_pAllocatingAccelerator = pAllocatingAccelerator;
        m_bDimensionsFinalized = FALSE;
        m_bPopulated = FALSE;
        m_pBuffer = NULL;
        m_bIsLogicallyEmpty = FALSE;
        m_nAlignment = DEFAULT_ALIGNMENT;
        m_bRequestedStrideValid = FALSE;
        m_bRequestedElementsValid = FALSE;
        m_bPhysicalBufferPinned = FALSE; 
        m_bPinnedBufferRequested = FALSE;
        m_bDirty = TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    PBuffer::~PBuffer(
        VOID
        )
    {
        ClearDependences(NULL);
        // all other tear-down is specific to 
        // inheriting classes. 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Mark dirty. </summary>
    ///
    /// <remarks>   crossbac, 4/30/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    PBuffer::MarkDirty(
        BOOL bDirty
        )
    {
        m_bDirty = bDirty;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is dirty. </summary>
    ///
    /// <remarks>   crossbac, 4/30/2013. </remarks>
    ///
    /// <returns>   true if dirty, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    PBuffer::IsDirty(
        VOID
        )
    {
        return m_bDirty;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a pointer to the object that can be bound to kernel 
    /// 			variables (or globals) that corresponds to the type of resource
    /// 			to which the object will be bound. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="bvtype">   The bvtype. </param>
    ///
    /// <returns>   The view. </returns>
    ///-------------------------------------------------------------------------------------------------

    BINDABLEOBJECT
    PBuffer::GetBindableObject(
        BINDINGTYPE bvtype
        ) {
        BINDABLEOBJECT result;
        result.vptr = NULL;
        std::map<BINDINGTYPE, BINDABLEOBJECT>::iterator mi;
        mi = m_mapBindableObjects.find(bvtype);
        if(m_mapBindableObjects.end() == mi) {
            assert(m_mapBindableObjects.end() != mi && "request for uninstantiated bindable object type!");
            PTask::Runtime::HandleError("%s: request for uninstantiated type", __FUNCTION__);
            return result;
        }
        return mi->second;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes a new PBuffer. Post-condition: IsInitialized returns true if the
    ///             initialization succeeded.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pAsyncContext">            [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="uiBufferSizeBytes">        The buffer size in bytes. </param>
    /// <param name="pInitialBufferContents">   (optional) [in] If non-null, the initial buffer
    ///                                         contents. </param>
    /// <param name="strDebugBufferName">       (optional) [in] If non-null, a name to assign to the
    ///                                         buffer which will be used to label runtime- specific
    ///                                         objects to aid in debugging. Ignored on release
    ///                                         builds. </param>
    /// <param name="bIsByteAddressable">       (optional) true if the resulting PBuffer must be byte
    ///                                         addressable by the device. </param>
    /// <param name="bPageLock">                true to lock, false to unlock the page. </param>
    ///
    /// <returns>   PTRESULT--use PTSUCCESS/PTFAILED macros to determine success or failure. </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT			
    PBuffer::Initialize(
        AsyncContext * pAsyncContext,
        UINT uiBufferSizeBytes,
        HOSTMEMORYEXTENT * pInitialBufferContents, 
        char * strDebugBufferName, 
        bool bIsByteAddressable,
        bool bPageLock
        ) 
    {
        m_bPinnedBufferRequested = bPageLock;
        if(pInitialBufferContents != NULL && pInitialBufferContents->lpvAddress != NULL) 
            DumpInitData(pInitialBufferContents->lpvAddress);

        // fix up byte addressability flags and set the template.
        // if the parent has no template, it had 
        // better be sealed so that we can actually
        // compute meaningful buffer sizes.
        DatablockTemplate * pTemplate = m_pParent->GetTemplate();
        bIsByteAddressable |= (pTemplate == NULL);
        // assert(pTemplate || m_pParent->IsSealed());

        // if the access flags indicate immutability,
        // create device buffers in constant memory,
        // if the accelerator has support for it.
        if(m_eAccessFlags & PT_ACCESS_IMMUTABLE) {

            // create device buffers in const/shared/group mem.
            return CreateImmutableBuffer(pAsyncContext,
                                         uiBufferSizeBytes,                                         
                                         pInitialBufferContents, 
                                         strDebugBufferName, 
                                         bIsByteAddressable);

        } else {

            // create device buffers in global memory
            return CreateMutableBuffer(pAsyncContext,
                                       uiBufferSizeBytes,
                                       pInitialBufferContents, 
                                       strDebugBufferName, 
                                       bIsByteAddressable);
        }
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
    PBuffer::Copy(
        __inout PBuffer *       pDstBuffer,
        __inout PBuffer *       pSrcBuffer,
        __in    AsyncContext *  pAsyncContext,
        __in    UINT            uiCopyBytes
        )
    {
        UNREFERENCED_PARAMETER(pDstBuffer);
        UNREFERENCED_PARAMETER(pSrcBuffer);
        UNREFERENCED_PARAMETER(pAsyncContext);
        UNREFERENCED_PARAMETER(uiCopyBytes);
        assert(FALSE);
        return FALSE;
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
    PBuffer::Copy(
        __inout PBuffer *       pDstBuffer,
        __inout PBuffer *       pSrcBuffer,
        __in    AsyncContext *  pAsyncContext
        )
    {
        UNREFERENCED_PARAMETER(pDstBuffer);
        UNREFERENCED_PARAMETER(pSrcBuffer);
        UNREFERENCED_PARAMETER(pAsyncContext);
        assert(FALSE);
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize a host view in the given buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pAsyncContext">            [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="lpvBuffer">                [in,out] If non-null, buffer for lpv data. </param>
    /// <param name="bForceSynchronization">    true to force synchronous transfer. </param>
    ///
    /// <returns>   number of bytes copied/transferred. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT				
    PBuffer::PopulateHostViewSynchronous(
        __in AsyncContext * pAsyncContext,
        __in HOSTMEMORYEXTENT * lpvBuffer
        )
    {
        m_bDirty = TRUE;
        assert(IsDimensionsFinalized());
        assert(!(m_eAccessFlags & PT_ACCESS_IMMUTABLE));
        if(ContextRequiresSync(pAsyncContext, OT_MEMCPY_SOURCE)) 
            WaitOutstandingAsyncOperations(pAsyncContext, OT_MEMCPY_SOURCE);
        BOOL bRequestOutstanding = FALSE;
        // we have no object on which to install a dependence, so we must 
        // conservatively force this call to be synchronous.
        BOOL bResult = __populateHostView(pAsyncContext, lpvBuffer, TRUE, bRequestOutstanding);
        assert(!bRequestOutstanding);
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize a host view in the given buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pAsyncContext">            [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="lpvBuffer">                [in,out] If non-null, buffer for lpv data. </param>
    /// <param name="bForceSynchronization">    true to force synchronous transfer. </param>
    ///
    /// <returns>   number of bytes copied/transferred. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT				
    PBuffer::PopulateHostView(
        __in AsyncContext * pAsyncContext,
        __in PBuffer * pHostBuffer,
        __in BOOL bForceSynchronization
        )
    {
        m_bDirty = TRUE;
        assert(IsDimensionsFinalized());
        assert(!(m_eAccessFlags & PT_ACCESS_IMMUTABLE));

        if(bForceSynchronization) {
            if(ContextRequiresSync(NULL, OT_MEMCPY_SOURCE))
                WaitOutstandingAsyncOperations(NULL, OT_MEMCPY_SOURCE);
            if(pHostBuffer->ContextRequiresSync(NULL, OT_MEMCPY_TARGET)) 
                pHostBuffer->WaitOutstandingAsyncOperations(NULL, OT_MEMCPY_TARGET);
        } else {
            if(ContextRequiresSync(pAsyncContext, OT_MEMCPY_SOURCE)) 
                WaitOutstandingAsyncOperations(pAsyncContext, OT_MEMCPY_SOURCE);
            if(pHostBuffer->ContextRequiresSync(pAsyncContext, OT_MEMCPY_TARGET)) 
                pHostBuffer->WaitOutstandingAsyncOperations(pAsyncContext, OT_MEMCPY_TARGET);
        }

        HOSTMEMORYEXTENT extent(pHostBuffer->GetBuffer(),
                                pHostBuffer->GetLogicalExtentBytes(),
                                pHostBuffer->IsPhysicalBufferPinned());
        BOOL bRequestOutstanding = FALSE;
        UINT uiResult = __populateHostView(pAsyncContext, &extent, bForceSynchronization, bRequestOutstanding);
        
        if(bRequestOutstanding) {
            assert(!bForceSynchronization);
            assert(pAsyncContext != NULL);
            assert(pAsyncContext->SupportsExplicitAsyncOperations());
            MARKRANGEENTER(L"add-deps");
            pAsyncContext->Lock();
            MARKRANGEENTER(L"add-deps(lck)");
            SyncPoint * pSP = pAsyncContext->CreateSyncPoint(this);            
            pHostBuffer->AddOutstandingDependence(pAsyncContext, OT_MEMCPY_TARGET, pSP);
            AddOutstandingDependence(pAsyncContext, OT_MEMCPY_SOURCE, pSP);
            MARKRANGEEXIT();
            pAsyncContext->Unlock();
            MARKRANGEEXIT();
        }

        return uiResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize accelerator view. Call only on PBuffers which have buffers allocated!
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, the stream. </param>
    /// <param name="pExtent">          [in,out] If non-null, the data. </param>
    /// <param name="pModule">          [in,out] (optional)  If non-null, the module. </param>
    /// <param name="lpszBinding">      (optional) the binding. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT				
    PBuffer::PopulateAcceleratorView(
        __in AsyncContext * pAsyncContext,
        __in PBuffer * pBuffer,
        __in void * pModule, 
        __in const char * lpszBinding
        )
    {
        m_bDirty = TRUE;
        HOSTMEMORYEXTENT extent(pBuffer->GetBuffer(),
                                pBuffer->GetLogicalExtentBytes(),
                                pBuffer->IsPhysicalBufferPinned());
        return PopulateAcceleratorView(pAsyncContext, pBuffer, &extent, pModule, lpszBinding);
     }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize accelerator view. Call only on PBuffers which have buffers allocated!
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pAsyncContext">        [in,out] If non-null, the stream. </param>
    /// <param name="pHostSourceBuffer">    [in,out] If non-null, buffer for host source data. </param>
    /// <param name="pExtent">              [in,out] If non-null, the data. </param>
    /// <param name="pModule">              [in,out] (optional)  If non-null, the module. </param>
    /// <param name="lpszBinding">          (optional) the binding. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT				
    PBuffer::PopulateAcceleratorView(
        __in AsyncContext * pAsyncContext,
        __in PBuffer * pHostSourceBuffer,
        __in HOSTMEMORYEXTENT * pExtent,
        __in void * pModule, 
        __in const char * lpszBinding
        )
    {
        m_bDirty = TRUE;
        assert(m_pBuffer != NULL);
        assert(IsDimensionsFinalized());

        if(ContextRequiresSync(pAsyncContext, OT_MEMCPY_TARGET))
            WaitOutstandingAsyncOperations(pAsyncContext, OT_MEMCPY_TARGET);
        if(pHostSourceBuffer && pHostSourceBuffer->ContextRequiresSync(pAsyncContext, OT_MEMCPY_SOURCE))
            pHostSourceBuffer->WaitOutstandingAsyncOperations(pAsyncContext, OT_MEMCPY_SOURCE);

        assert(GetAllocationExtentBytes() != 0);
        UINT uiBufferSizeBytes = GetLogicalExtentBytes();
        if(uiBufferSizeBytes == 0 && m_pParent != NULL) {
            // the parent isn't actually the best place to get this
            // information--the parent block may be a pooled block whose 
            // dimensions do not indicate the logical size of the buffer
            // we need to create. 
            uiBufferSizeBytes = m_vDimensionsFinalized.AllocationSizeBytes();
        }
        assert(uiBufferSizeBytes != 0);

        UINT uiResult = 0;
        BOOL bOustandingAsyncOps = FALSE;
        if(m_eAccessFlags & PT_ACCESS_IMMUTABLE) {

            // device specialized (constant) memory
            uiResult =  __populateImmutableAcceleratorView(pAsyncContext,
                                                           uiBufferSizeBytes,
                                                           pExtent,
                                                           bOustandingAsyncOps,
                                                           pModule, 
                                                           lpszBinding);

        } else {
        
            // device global memory
            uiResult = __populateMutableAcceleratorView(pAsyncContext, 
                                                        uiBufferSizeBytes,
                                                        pExtent,
                                                        bOustandingAsyncOps,
                                                        pModule, 
                                                        lpszBinding);
        }

        if(bOustandingAsyncOps) {
            assert(pAsyncContext != NULL);
            assert(pAsyncContext->SupportsExplicitAsyncOperations());
            pAsyncContext->Lock();
            SyncPoint * pSyncPoint = pAsyncContext->CreateSyncPoint(this);
            AddOutstandingDependence(pAsyncContext, OT_MEMCPY_TARGET, pSyncPoint);
            if(pHostSourceBuffer) 
                pHostSourceBuffer->AddOutstandingDependence(pAsyncContext, OT_MEMCPY_SOURCE, pSyncPoint);
            pAsyncContext->Unlock();
        }

        return uiResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Create all necessary bindable objects for the buffer based on it's access flags.
    ///             In some platforms, notably directX, view objects exist for different types of
    ///             access. Read access for the GPU requires a shader resource view, write access
    ///             requires an unordered access view, etc. Other platforms (e.g. CUDA) treat all
    ///             device-visible buffers as a single array of flat memory. This function calls into
    ///             it the PBuffer's specializing type to create any views needed for the underlying
    ///             platform. platform-specific work is encapsulated in CreateMutable and
    ///             CreateImmutable.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/4/2012. </remarks>
    ///
    /// <param name="szname">   [in] If non-null, the a string used to label the object that can be
    ///                         used for debugging. </param>
    ///
    /// <returns>   PTRESULT. </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    PBuffer::CreateBindableObjects( 
        char * szname
        )
    {
        PTRESULT hrGPUConsume = PTASK_OK;
        PTRESULT hrGPUProduce = PTASK_OK;
        PTRESULT hrImmutable = PTASK_OK;
        if(m_eAccessFlags & PT_ACCESS_ACCELERATOR_READ && !(m_eAccessFlags & PT_ACCESS_IMMUTABLE)) 
            hrGPUConsume = CreateBindableObjectsReadable(szname);
        if(m_eAccessFlags & PT_ACCESS_ACCELERATOR_WRITE) 
            hrGPUProduce = CreateBindableObjectsWriteable(szname);
        if(m_eAccessFlags & PT_ACCESS_IMMUTABLE)
            hrImmutable = CreateBindableObjectsImmutable(szname);
        if(PTFAILED(hrGPUConsume))
            return hrGPUConsume;
        if(PTFAILED(hrGPUProduce))
            return hrGPUProduce;
        if(PTFAILED(hrImmutable))
            return hrImmutable;
        return PTASK_OK;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a stride. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The stride. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    PBuffer::GetElementStride(
        VOID
        )
    {
        assert(m_bDimensionsFinalized);
        return m_vDimensionsFinalized.cbElementStride;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a stride. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The stride. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    PBuffer::GetPitch(
        VOID
        )
    {
        assert(m_bDimensionsFinalized);
        return m_vDimensionsFinalized.cbPitch;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the elements. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The elements. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    PBuffer::GetElementCount(
        VOID
        )
    {
        assert(m_bDimensionsFinalized);
        return m_vDimensionsFinalized.TotalElements();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the size in bytes. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <returns>   The size bytes. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT                
    PBuffer::GetAllocationExtentBytes(
        VOID
        )
    {
        assert(m_nAlignment != 0);
        assert(m_bDimensionsFinalized);
        UINT uFDCount = m_vDimensionsFinalized.AllocationSizeBytes();
        UINT uAllocCount = (uFDCount == 0) ? EMPTY_BUFFER_ALLOC_SIZE : uFDCount;
        UINT uAlignedCount = ((uAllocCount % m_nAlignment) == 0) ? uAllocCount : (uAllocCount+m_nAlignment)/m_nAlignment;
        return uAlignedCount;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the logical size of the buffer in bytes. This may differ from the allocation
    ///             size due to things such as alignment, or the need to allocate non-zero size
    ///             buffers to back buffers that are logically empty.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <returns>   The size bytes. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    PBuffer::GetLogicalExtentBytes(
        VOID
        )
    {
        UINT uiSizeBytes = 0;
        assert(m_bDimensionsFinalized);
        if(m_pParent) {
            m_pParent->Lock();
            uiSizeBytes = m_pParent->GetChannelLogicalSizeBytes(m_nChannelIndex);
            m_pParent->Unlock();
        } else {
            uiSizeBytes = m_vDimensionsFinalized.AllocationSizeBytes();
        }
        return uiSizeBytes;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the template. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the template. </returns>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate*
    PBuffer::GetTemplate(
        VOID
        ) 
    {
        return m_pParent->GetTemplate();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the accelerator used to create this PBuffer </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the accelerator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator *       
    PBuffer::GetAccelerator(
        VOID
        ) 
    {
        // return the accelerator used to create
        // this buffer. 
        return m_pAccelerator;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Debug dump. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="ssOut">        [in,out] If non-null, the ss out. </param>
    /// <param name="pcsOutLock">   [in,out] If non-null, the pcs out lock. </param>
    /// <param name="szTaskLabel">  [in,out] (optional) the t. </param>
    /// <param name="szPortLabel">  [in,out] (optional) the dump start element. </param>
    ///-------------------------------------------------------------------------------------------------

    void                
    PBuffer::DebugDump(
        __in std::ostream* ssOut,
        __in CRITICAL_SECTION* pcsOutLock,
        __in char * szTaskLabel,
        __in char * szPortLabel
        )
    {
        ostream * ss = ssOut ? ssOut : &std::cerr;
        if(pcsOutLock) EnterCriticalSection(pcsOutLock);
        (*ss) << (szTaskLabel?szTaskLabel:"") << ":"
              << (szPortLabel?szPortLabel:"") << ":"
              << endl;
        UINT uiDumpStartIndex = 0;
        DebugDump(PTask::Runtime::GetDumpType(), 
                  uiDumpStartIndex, 
                  PTask::Runtime::GetDumpLength(),
                  PTask::Runtime::GetDumpStride());
        if(pcsOutLock) LeaveCriticalSection(pcsOutLock);   
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Debug dump. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="dtyp">                 The dtyp. </param>
    /// <param name="nDumpStartElement">    The dump start element. </param>
    /// <param name="nDumpEndElement">      The dump end element. </param>
    /// <param name="nStride">              The stride. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    PBuffer::DebugDump(
        DEBUGDUMPTYPE dtyp,
        UINT nDumpStartElement, 
        UINT nDumpEndElement,
        UINT nStride
        ) 
    {        
        m_pAccelerator->Lock();
        UINT cbBuffer = m_vDimensionsFinalized.AllocationSizeBytes();
        HOSTMEMORYEXTENT hme(malloc(cbBuffer), cbBuffer, FALSE);
        void * pData = hme.lpvAddress;
        UINT cbTransferred = PopulateHostViewSynchronous(NULL, &hme);
        if(cbTransferred != cbBuffer) {
            printf("pbuffer at %8X, accelerator-ptr at %8X, DtoHxfer FAILED!\n", this, GetBuffer());
        } else {
            UINT nLoopStart;
            UINT nLoopEnd;
            BYTE * p = (BYTE*) pData;
            float * pf = (float*) pData;
            int * pi = (int*) pData;
            printf("pbuffer at %8X, accelerator-bufptr at %8X:\n", this, GetBuffer());
            switch(dtyp) {
            case dt_float:
                nLoopStart = nDumpStartElement;
                nLoopEnd = nDumpEndElement == 0 ? cbBuffer/sizeof(float) : nDumpEndElement;
                for(UINT i=nLoopStart; i<nLoopEnd*nStride; i+=nStride) {
//                    printf("%.2f ", pf[i]);
                    printf("%5.5f ", pf[i]);
                    if(i%8 == 7)
                        printf("\n");
                }
                break;
            case dt_int:
                nLoopStart = nDumpStartElement;
                nLoopEnd = nDumpEndElement == 0 ? cbBuffer/sizeof(int) : nDumpEndElement;
                for(UINT i=nLoopStart; i<nLoopEnd*nStride; i+=nStride) {
                    printf("%d ", pi[i]);
                    if(i%8 == 7)
                        printf("\n");
                }
                break;
            default:
            case dt_raw:
                nLoopStart = nDumpStartElement;
                nLoopEnd = nDumpEndElement == 0 ? cbBuffer/sizeof(BYTE) : nDumpEndElement;
                for(UINT i=nLoopStart; i<nLoopEnd*nStride; i+=nStride) {
                    printf("%2X ", p[i]);
                    if(i%16 == 15)
                        printf("\n");
                }
                break;
            }
            printf("\n");
        }
        m_pAccelerator->Unlock();
        free(hme.lpvAddress);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the access flags. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The access flags. </returns>
    ///-------------------------------------------------------------------------------------------------

    BUFFERACCESSFLAGS	
    PBuffer::GetAccessFlags(
        VOID
        ) 
    { 
        return m_eAccessFlags; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the access flags. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="f">    The f. </param>
    ///-------------------------------------------------------------------------------------------------

    void                
    PBuffer::SetAccessFlags(
        BUFFERACCESSFLAGS f
        ) 
    { 
        m_eAccessFlags = f; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a uid. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The uid. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT				
    PBuffer::GetUID(
        VOID
        ) 
    { 
        return m_uiId; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a uid. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="ui">   The user interface. </param>
    ///-------------------------------------------------------------------------------------------------

    void				
    PBuffer::SetUID(
        UINT ui
        ) 
    { 
        m_uiId=ui; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Platform specific finalize dimension. </summary>
    ///
    /// <remarks>   crossbac, 7/10/2012. </remarks>
    ///
    /// <param name="uiStride">         The stride. </param>
    /// <param name="uiElementCount">   Number of elements. </param>
    /// <param name="uiDimension">      The dimension. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    PBuffer::PlatformSpecificFinalizeDimension(
        __in UINT uiStride,
        __in UINT uiElementCount,
        __in UINT uiDimension
        )
    {
        // given an element count and stride, along with dimension specifier
        // return the most efficient number of elements to allocate for the 
        // underlying architecture. This base class implementation just returns the
        // number provided: subclasses should override to suit. 
        UNREFERENCED_PARAMETER(uiStride);
        UNREFERENCED_PARAMETER(uiDimension);
        return uiElementCount;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Finalize the dimension of the required buffer. 
    /// 			Return the number of bytes needed to create the buffer. </summary>
    ///
    /// <remarks>   Crossbac, 1/4/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    PBuffer::FinalizeDimensions(
        __out bool &bByteAddressable,
        __in UINT uiBufferSizeBytes
        )
    {
        // default implementation requires parent to be sealed, host 
        // sub-class (and others that need to work with buffers for 
        // blocks before the dimensions are final) can override this. 
        return FinalizeDimensions(bByteAddressable, uiBufferSizeBytes, TRUE); 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Finalize the dimensions of the device buffer that will be created to back this
    ///             PBuffer: the bool bRequireSealedParent indicates whether or not the parent datablock
    ///             must be sealed for the block to have its dimensions finalized: for platform buffers
    ///             in any space other than the host, we have to finalize because there will be traffic
    ///             between memory spaces that requires a known, immutable allocation size. In the
    ///             host space however, to let the programmer fill a buffer and subsequently seal it,
    ///             we must be willing to allocate buffers to back an unsealed data block. Exposing a
    ///             parameter for this requirement makes it easier to override FinalizeDimensinons for
    ///             the host-specific pbuffer class. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac </remarks>
    ///
    /// <param name="bByteAddressable">     [out] (optional) true if the buffer should be byte
    ///                                     addressable. </param>
    /// <param name="uiBufferSizeBytes">    (optional) the buffer size in bytes. </param>
    /// <param name="bRequireSealedParent"> The require sealed parent. </param>
    ///
    /// <returns>   bytes allocated. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT        
    PBuffer::FinalizeDimensions(
        __out bool &bByteAddressable,
        __in  UINT uiBufferSizeBytes,
        __in  BOOL bRequireSealedParent
        )
    {
        assert(!IsDimensionsFinalized());
        DatablockTemplate * pTemplate = GetTemplate();

        // first initialize all the requested dimensions. If they are already set, we can 
        // leave them as is, but if they are not initialized, we must derive them from data
        // in the parent datablock, the template, or the override size bytes parameter 
        // in this function call
        
        if(pTemplate != NULL) {

            // if there is a template, we will almost always want to just take the dimesions
            // specified by the template. Exceptions are cases where the template has a variable
            // size, when we have a non-defaulted size parameter in the function call, 
            // or if the parent explicitly can tell us the size parameter in the template should
            // be overridden. This is only something that can happen for the data channel at the
            // moment, FWIW.
            
            BOOL bVariableTemplate = pTemplate->DescribesRecordStream();
            BOOL bOverrideTemplateSizeBytes = 
                (m_nChannelIndex == DBDATA_IDX) &&     // data channel only!
                ((uiBufferSizeBytes != PBUFFER_DEFAULT_SIZE) ||          // non-defaulted size parameter
                  m_pParent->IsTemplateSizeOverridden(m_nChannelIndex)); // explicit size override

            m_vDimensionsRequested.cbElementStride = 
                m_bRequestedStrideValid ? 
                    m_vDimensionsRequested.cbElementStride : 
                    pTemplate->GetStride(m_nChannelIndex); 
            UINT uiStrideRequested = m_vDimensionsRequested.cbElementStride;
            if(m_nChannelIndex != DBDATA_IDX) 
                m_vDimensionsRequested.cbElementStride = 1; // the stride doesn't apply to other channels!
            m_bRequestedStrideValid = TRUE;

            if(!m_bRequestedElementsValid) {

                // the requested elements have not been set, get them from the template,
                // unless we need to override the template size, as described above.

                if(bOverrideTemplateSizeBytes || bVariableTemplate) {

                    // take the override if it's not defaulted, otherwise
                    // ask the parent explicitly for its element size.
                    UINT uiParentSizeBytes = m_pParent->GetChannelAllocationSizeBytes(m_nChannelIndex);
					UINT uiParentElems = (uiParentSizeBytes / uiStrideRequested) + ((uiParentSizeBytes%uiStrideRequested)?1:0);
                    UINT uiElements = uiBufferSizeBytes == 0 ? 
                        uiParentElems :
                        uiBufferSizeBytes;
                    m_vDimensionsRequested.uiXElements = uiElements;
                    m_vDimensionsRequested.uiYElements = 1;
                    m_vDimensionsRequested.uiZElements = 1;
                    m_vDimensionsRequested.cbPitch = uiElements*uiStrideRequested;

                } else {

                    m_vDimensionsRequested.uiXElements = pTemplate->GetXElementCount(m_nChannelIndex);
                    m_vDimensionsRequested.uiYElements = pTemplate->GetYElementCount(m_nChannelIndex);
                    m_vDimensionsRequested.uiZElements = pTemplate->GetZElementCount(m_nChannelIndex);
                    m_vDimensionsRequested.cbPitch = pTemplate->GetPitch(m_nChannelIndex);
                
                }
                m_bRequestedElementsValid = TRUE;
                bByteAddressable = bByteAddressable;
            }

        } else {

            // no template. ask the parent directly for element counts and stride
            UINT uiDefaultStride = 1;
            UINT uiStrideRequested = 1;
            if(bRequireSealedParent || m_pParent->IsSealed()) {
                assert(m_pParent->IsSealed()); 
                uiDefaultStride = m_pParent->GetStride();
                uiStrideRequested = m_vDimensionsRequested.cbElementStride;
                uiStrideRequested = m_bRequestedStrideValid ? uiStrideRequested : uiDefaultStride;
                m_bRequestedStrideValid = TRUE;

                if(!m_bRequestedElementsValid) {
                    m_vDimensionsRequested.Initialize(m_pParent->GetXElementCount(),
                                                      m_pParent->GetYElementCount(),
                                                      m_pParent->GetZElementCount(),
                                                      uiStrideRequested);
                }
            } else {
                // the parent is not sealed and isn't required to be. 
                // alloc based on pre-seal requested sizes
                uiStrideRequested = m_bRequestedStrideValid ? m_vDimensionsRequested.cbElementStride : 1;
                m_bRequestedStrideValid = TRUE;

                if(!m_bRequestedElementsValid) {
                    m_vDimensionsRequested.Initialize(m_pParent->GetDataBufferLogicalSizeBytes(), 1, 1, uiStrideRequested);
                }
            }
            m_bRequestedElementsValid = TRUE;
            bByteAddressable = true;
        }

        // we know the requested stride and number of elements, either because they
        // were already specified on entry or because we just derived them above. 
        // consequently, finalizing them is a matter of choosing a copied size compatible
        // with the architecture, which is a process deferred to classes that specialize PBuffer.

        assert(m_bRequestedElementsValid);
        assert(m_bRequestedStrideValid);
        assert(!m_bDimensionsFinalized);

        UINT uiStride = m_vDimensionsFinalized.cbElementStride = m_vDimensionsRequested.cbElementStride;
        m_vDimensionsFinalized.cbPitch = m_vDimensionsRequested.cbPitch;
        m_vDimensionsFinalized.uiXElements = PlatformSpecificFinalizeDimension(uiStride, m_vDimensionsRequested.uiXElements, XDIM);
        m_vDimensionsFinalized.uiYElements = PlatformSpecificFinalizeDimension(uiStride, m_vDimensionsRequested.uiYElements, YDIM);
        m_vDimensionsFinalized.uiZElements = PlatformSpecificFinalizeDimension(uiStride, m_vDimensionsRequested.uiZElements, ZDIM);
        m_bDimensionsFinalized = TRUE;

        UINT uCount = GetElementCount();
        UINT uPitchedBytes = m_vDimensionsFinalized.AllocationSizeBytes();
        UINT uBufferBytes = max(uCount * uiStride, uPitchedBytes);
        assert(uBufferBytes >= GetLogicalExtentBytes());
        assert(uPitchedBytes >= GetLogicalExtentBytes());
        UINT uiAllocCount = GetAllocationExtentBytes();
        m_bIsLogicallyEmpty = (uBufferBytes == 0 && uiAllocCount != 0);
        return uBufferBytes;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has its platform-specific buffer created (device-side
    /// 			in the common case). </summary>
    ///
    /// <remarks>   Crossbac, 1/4/2012. </remarks>
    ///
    /// <returns>   true if bindable objects, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    PBuffer::IsPlatformBufferInitialized(
        VOID
        )
    {
        return m_pBuffer != NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has its platform-specific buffer populated (device-side
    /// 			in the common case). </summary>
    ///
    /// <remarks>   Crossbac, 1/4/2012. </remarks>
    ///
    /// <returns>   true if bindable objects, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    PBuffer::IsPlatformBufferPopulated(
        VOID
        )
    {
        return m_bPopulated;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has its platform-specific bindable objects created. </summary>
    ///
    /// <remarks>   Crossbac, 1/4/2012. </remarks>
    ///
    /// <returns>   true if bindable objects, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    PBuffer::IsBindable(
        VOID
        )
    {
        return m_mapBindableObjects.size() != 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the buffer object. </summary>
    ///
    /// <remarks>   Crossbac, 1/4/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    void * 
    PBuffer::GetBuffer(
        VOID
        )
    {
        return m_pBuffer;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' is scalar parameter. </summary>
    ///
    /// <remarks>   Crossbac, 1/4/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if scalar parameter, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    PBuffer::IsScalarParameter(
        VOID
        ) 
    {
        assert(m_pParent != NULL);
        DatablockTemplate * pTemplate = m_pParent->GetTemplate();
        if(pTemplate != NULL) {
            return pTemplate->DescribesScalarParameter();
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a parameter type. </summary>
    ///
    /// <remarks>   Crossbac, 1/4/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The parameter type. </returns>
    ///-------------------------------------------------------------------------------------------------

    PTASK_PARM_TYPE
    PBuffer::GetParameterType(
        VOID
        ) 
    {
        assert(m_pParent != NULL);
        DatablockTemplate * pTemplate = m_pParent->GetTemplate();
        if(pTemplate != NULL) {
            return pTemplate->GetParameterBaseType();
        }
        return PTPARM_NONE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has had its dimensions finalized. </summary>
    ///
    /// <remarks>   Crossbac, 1/4/2012. </remarks>
    ///
    /// <returns>   true if dimensions finalized, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    PBuffer::IsDimensionsFinalized(
        VOID
        )
    {
        return m_bDimensionsFinalized;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the accelerator that allocated this buffer. Returns NULL, unless an
    ///             accelerator other than m_pAccelerator allocated the buffer (e.g. when a CUDA
    ///             accelerator has AllocatePagelockedHostMemory() called to allocate host-memory in a PBuffer).
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the accelerator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator *       
    PBuffer::GetAllocatingAccelerator(
        VOID
        )
    {
        return m_pAllocatingAccelerator;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the parent of this item. </summary>
    ///
    /// <remarks>   Crossbac, 1/5/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the parent. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock*  
    PBuffer::GetParent(
        VOID
        )
    {
        return m_pParent;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Clears the dependences. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    PBuffer::ClearDependences(
        __in AsyncContext * pAsyncContext
        ) 
    {
        MARKRANGEENTER(L"ClearDependences");
        RetireDependenceFrontier(m_vReadFrontier, m_vOutstandingReadAccelerators, pAsyncContext);
        RetireDependenceFrontier(m_vWriteFrontier, m_vOutstandingWriteAccelerators, pAsyncContext);
        ReleaseRetiredDependences(pAsyncContext);
        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Clears the dependences. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    PBuffer::ReleaseRetiredDependences(
        __in AsyncContext * pAsyncContext
        ) 
    {
        BOOL bSuccess = TRUE;
        if(m_vRetired.size() > 0) {
            CHECK_DEPENDENCE_INVARIANTS();
            std::deque<AsyncDependence*>::iterator di;
            std::deque<AsyncDependence*> newRetiredList;
            for(di=m_vRetired.begin(); di!=m_vRetired.end(); di++) {
                AsyncDependence * pDep = *di;
                if(pAsyncContext == NULL || pDep->GetContext() == pAsyncContext) {
                    pDep->Release();
                } else {
                    newRetiredList.push_back(pDep);
                }
            }
            m_vRetired.clear();
            m_vRetired.assign(newRetiredList.begin(), newRetiredList.end());
        }
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   We rely heavily on asynchrony, but we release blocks when we are done
    ///             queueing dispatches that use them. Consequently, it is entirely probable that
    ///             we wind up attempting to free Datablocks or return them to their block pools
    ///             before the operations we've queued on them have actually completed.
    ///             With block pools we rely on leaving the outstanding dependences queued
    ///             on the buffers in the datablock. However, for blocks that actually get
    ///             deleted, we need to be sure that any dangling operations have actually
    ///             completed on the GPU. This method is for precisely that--it should only
    ///             be called from the release/dtor code of Datablock. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/25/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    PBuffer::SynchronousWaitOutstandingOperations(
        VOID
        )
    {
        BOOL bSuccess = TRUE;
        std::deque<AsyncDependence*>::iterator di;
        for(di=m_vReadFrontier.begin(); di!=m_vReadFrontier.end(); di++) {
            AsyncDependence * pDep = *di;
            bSuccess &= pDep->SynchronousExclusiveWait();
            pDep->Release();
        }
        for(di=m_vWriteFrontier.begin(); di!=m_vWriteFrontier.end(); di++) {
            AsyncDependence * pDep = *di;
            bSuccess &= pDep->SynchronousExclusiveWait();
            pDep->Release();
        }
        m_vReadFrontier.clear();
        m_vWriteFrontier.clear();
        m_vOutstandingReadAccelerators.clear();
        m_vOutstandingWriteAccelerators.clear();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if there are outstanding dependences that would
    ///             need to resolve before an operation of the given type
    ///             can occur. </summary>
    ///
    /// <remarks>   crossbac, 6/25/2013. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   true if outstanding dependence, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    PBuffer::HasConflictingOutstandingDependences(
        __in ASYNCHRONOUS_OPTYPE eOpType
        )
    {
        return m_vWriteFrontier.size() || (ASYNCOP_ISWRITE(eOpType) && m_vReadFrontier.size());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Lockless wait outstanding: without acquiring any locks attempt to perform a
    ///             synchronous wait for any outstanding async dependences on this buffer that
    ///             conflict with an operation of the given type. This is an experimental API,
    ///             enable/disable with PTask::Runtime::*etTaskDispatchLocklessIncomingDepWait(),
    ///             attempting to leverage the fact that CUDA apis for waiting on events (which
    ///             appear to be thread-safe and decoupled from a particular device context)
    ///             to minimize serialization associated with outstanding dependences on data
    ///             consumed by tasks that do not require accelerators for any other reason than to
    ///             wait for such operations to complete.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///
    /// <param name="eOpType">  Type of the operation. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    PBuffer::LocklessWaitOutstanding(
        __in ASYNCHRONOUS_OPTYPE eOpType
        )
    {
        assert(PTask::Runtime::GetTaskDispatchLocklessIncomingDepWait());
        if(!HasConflictingOutstandingDependences(eOpType))
            return TRUE;
        MARKRANGEENTER(L"PBuffer::LocklessWaitOutstanding");

        BOOL bSuccess = TRUE;
        std::vector<std::deque<AsyncDependence*>*>::iterator vi;
        std::vector<std::deque<AsyncDependence*>*> vFrontiers;
        std::deque<AsyncDependence*>::iterator di;
        if(m_vWriteFrontier.size()) 
            vFrontiers.push_back(&m_vWriteFrontier);
        if(m_vReadFrontier.size() && ASYNCOP_ISWRITE(eOpType))
            vFrontiers.push_back(&m_vReadFrontier);
        for(vi=vFrontiers.begin(); vi!=vFrontiers.end(); vi++) 
            for(di=(*vi)->begin(); di!=(*vi)->end(); di++) 
                bSuccess &= (*di)->LocklessWaitOutstanding();       
        RetireResolvedFrontierEntries();
        MARKRANGEEXIT();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Wait oustanding dependences. What we have to wait for depends on the type of
    ///             operation we plan to queue. Specifically, if the operation is a read, we can add
    ///             it to the read frontier and wait on the last write. If it is a write, we must
    ///             wait for the most recent reads.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="eOpType">          Type of the operation. </param>
    /// <param name="pDependences">     [in,out] If non-null, the dependences. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    PBuffer::WaitOutstandingAsyncOperations(
        __in AsyncContext *              pAsyncContext,
        __in ASYNCHRONOUS_OPTYPE         eOpType,
        __in std::set<AsyncDependence*>* pDependences,
        __out BOOL *                     pbAlreadyResolved
        )
    {
        UNREFERENCED_PARAMETER(pDependences);
        MARKRANGEENTER(L"WaitOutstandingAsyncOperations");

        // if this context is different from the one on which we have outstanding deps, then we need to
        // wait for all of them to finish Fortunately, since all deps in the queue are for the same
        // context by construction, it suffices to wait for the last one and discard the remaining
        // entries in the queue.       
        
        BOOL bSuccess = TRUE;
        BOOL bReaders = m_vReadFrontier.size() != 0;
        BOOL bWriters = m_vWriteFrontier.size() != 0;
        if(pbAlreadyResolved) *pbAlreadyResolved = FALSE;
        if(!bReaders && !bWriters) {

            // it is actually possible for multiple buffers to share a dependence--if that dependence is
            // resolved through a lockless wait, then then the read/write frontiers can get retired out
            // from under us between the time we check for a sync requirement and the time we actually
            // perform the sync. In debug, we actually collect the frontiers so we can verify that this has
            // happened and assert if not. Otherwise, we have to assume this is what happened.
            
            VERIFY_SYNC_DEPENDENCES(bReaders, bWriters, pDependences);
            if(pbAlreadyResolved) *pbAlreadyResolved = TRUE;
            MARKRANGEEXIT();
            return bSuccess;
        }

        // Runtime::Tracer::LogBufferSyncEvent(this, TRUE, m_pParent, m_pAccelerator->GetAcceleratorId());
        CHECK_DEPENDENCE_INVARIANTS();
        std::vector<std::deque<AsyncDependence*>*> vFrontiers;
        std::vector<std::deque<AsyncDependence*>*>::iterator fi;
        std::deque<AsyncDependence*>::iterator di;

        if(ASYNCOP_ISWRITE(eOpType) && m_vReadFrontier.size())
            vFrontiers.push_back(&m_vReadFrontier);
        if(m_vWriteFrontier.size())
            vFrontiers.push_back(&m_vWriteFrontier);

        std::set<AsyncContext*> vSyncedContexts;
        BOOL bContextSynchronized = FALSE;
        BOOL bSynchronousResolution = FALSE;
        BOOL bSynchronousOpContext = (pAsyncContext == NULL) || 
                                     !pAsyncContext->SupportsExplicitAsyncOperations();

        for(fi=vFrontiers.begin(); fi!=vFrontiers.end(); fi++) {

            std::deque<AsyncDependence*>* pFrontier = *fi;
            for(di=pFrontier->begin(); di!=pFrontier->end(); di++) {

                AsyncDependence * pDep = *di;       
                if(bSynchronousOpContext) {

                    // if we don't have an async context, then we need to conservatively attempt to synchronously
                    // await the completion of whatever outstanding operations we depende on. First attempt to wait
                    // on the individual dependence--if that fails, resort to synchronizing the device context.
                    // Note that we must hold the accelerator lock for the context in the outstanding dependence.
                    // We cannot call the version that acquires locks because this method can be called from
                    // dispatch context. 
                
                    if(pDep->NonblockingQueryOutstanding()) {
                        BOOL bWaitSuccess = pDep->LocklessWaitOutstanding();
                        if(!bWaitSuccess) {
                            AsyncContext * pDepCtxt = pDep->GetContext();
                            assert(pDepCtxt != NULL);
                            assert(pDepCtxt->SupportsExplicitAsyncOperations());
                            pDepCtxt->Lock();
                            bWaitSuccess = pDepCtxt->SynchronizeContext();
                            bContextSynchronized = bWaitSuccess;
                            vSyncedContexts.insert(pDepCtxt);
                            pDepCtxt->Unlock();
                        }
                        bSuccess &= bWaitSuccess;
                    }
                    // we changed the state of some underlying
                    // dependence/event state by either querying it or 
                    // waiting synchronously for it. Consequently, we
                    // need to garbage collect dependences on the frontiers.
                    bSynchronousResolution = TRUE;

                } else {

                    // we have an async context, so we can use fence operations to wait
                    // asynchronously--by calling OrderSubsequentOperationsAfter(), we 
                    // are telling this device context not to dispatch any commands from this
                    // device queue that are enqueued subsequent to this "wait", until the
                    // dependence described by dDep is resolved. We can get pretty far ahead
                    // of the command queues this way.

                    MARKRANGEENTER(L"Lock+order-deps");
                    pAsyncContext->LockAccelerator();
                    bSuccess &= pAsyncContext->OrderSubsequentOperationsAfter(pDep);
                    pAsyncContext->UnlockAccelerator();
                    MARKRANGEEXIT();
                }
            }
        }

        if(bSynchronousResolution) {
            
            // we had to do some synchronous calls to resolve these
            // dependences. Consequently, we know there are out-of-date
            // entries in the frontiers that can be cleaned up, and potentially
            // retired entries that can be cleaned up if we were forced to sync
            // an entire context (rather than a specific event).

            // remove resolved deps from frontiers
            RetireResolvedFrontierEntries(); 

            // remove retired and frontier entries for 
            // contexts we were forced to synchronize            
            if(bContextSynchronized) {
                std::set<AsyncContext*>::iterator si;
                for(si=vSyncedContexts.begin(); si!=vSyncedContexts.end(); si++) {
                    ClearDependences(*si);
                }
            }
        } 

        //Runtime::Tracer::LogBufferSyncEvent(this, FALSE, m_pParent, m_pAccelerator->GetAcceleratorId());
        MARKRANGEEXIT();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Retire any entries in the dependence frontier that 
    ///             have been resolved synchronously, either through 
    ///             context sync or event query. </summary>
    ///
    /// <remarks>   crossbac, 7/2/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    PBuffer::RetireResolvedFrontierEntries(
        VOID
        )
    {
        std::deque<AsyncDependence*> vLiveWriters;
        std::deque<AsyncDependence*> vLiveReaders;
        std::deque<AsyncDependence*>::iterator vi;
        m_vOutstandingWriteAccelerators.clear();
        for(vi=m_vWriteFrontier.begin(); vi!=m_vWriteFrontier.end(); vi++) {
            BOOL bOutstanding = (*vi)->QueryOutstandingFlag();
            std::deque<AsyncDependence*>& vDestQueue =             
                bOutstanding ? vLiveWriters : m_vRetired;
            vDestQueue.push_back(*vi);
            if(bOutstanding) {
                AsyncDependence * pDep = (*vi);
                AsyncContext * pAsyncContext = pDep->GetContext();
                Accelerator * pAccelerator = pAsyncContext->GetDeviceContext();
                m_vOutstandingWriteAccelerators.insert(pAccelerator);
            }
        }
        m_vWriteFrontier.clear();
        m_vWriteFrontier.assign(vLiveWriters.begin(), vLiveWriters.end());
        m_vOutstandingReadAccelerators.clear();
        for(vi=m_vReadFrontier.begin(); vi!=m_vReadFrontier.end(); vi++) {
            BOOL bOutstanding = (*vi)->QueryOutstandingFlag();
            std::deque<AsyncDependence*>& vDestQueue =             
                 bOutstanding ? vLiveReaders : m_vRetired;
            vDestQueue.push_back(*vi);
            if(bOutstanding) {
                AsyncDependence * pDep = (*vi);
                AsyncContext * pAsyncContext = pDep->GetContext();
                Accelerator * pAccelerator = pAsyncContext->GetDeviceContext();
                m_vOutstandingReadAccelerators.insert(pAccelerator);
            }
        }
        m_vReadFrontier.clear();
        m_vReadFrontier.assign(vLiveReaders.begin(), vLiveReaders.end());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Retire dependence frontier. </summary>
    ///
    /// <remarks>   crossbac, 5/1/2013. </remarks>
    ///
    /// <param name="deps">             [in,out] [in,out] If non-null, the deps. </param>
    /// <param name="acclist">          [in,out] [in,out] If non-null, the acclist. </param>
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    PBuffer::RetireDependenceFrontier(
        __inout std::deque<AsyncDependence*>& deps,
        __inout std::set<Accelerator*>& acclist,
        __in AsyncContext * pAsyncContext
        )
    {
        std::deque<AsyncDependence*> dcpy;
        std::deque<AsyncDependence*>::iterator vi;
        if(pAsyncContext == NULL || !pAsyncContext->SupportsExplicitAsyncOperations()) {
            for(vi=m_vRetired.begin(); vi!=m_vRetired.end(); vi++) {
                AsyncDependence * pRetiredDep = *vi;
                pRetiredDep->Release();
            }
            m_vRetired.clear();
            m_vRetired.assign(deps.begin(), deps.end());
        } else {
            std::deque<AsyncDependence*> newFrontier;
            for(vi=deps.begin(); vi!=deps.end(); vi++) {
                AsyncDependence * pRetiredDep = *vi;
                if(pRetiredDep->GetContext() == pAsyncContext) 
                    m_vRetired.push_back(pRetiredDep);
                else
                    newFrontier.push_back(pRetiredDep);
            }
            deps.clear();
            deps.assign(newFrontier.begin(), newFrontier.end());
        }
        deps.clear();
        acclist.clear();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record outstanding dependences. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="pExplictSyncPoint">    The explict synchronise point. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    PBuffer::AddOutstandingDependence(
        __in AsyncContext * pAsyncContext,
        __in ASYNCHRONOUS_OPTYPE eOperationType,
        __in SyncPoint * pExplicitSyncPoint
        )
    {
        BOOL bResult = FALSE;
        assert(pAsyncContext != NULL);
        MARKRANGEENTER(L"AddOutstandingDependence");
        if(pAsyncContext != NULL && !pAsyncContext->GetDeviceContext()->IsHost()) {

            // this assert is trying to make sure we dont create unnecessary
            // dependences between non-conflicting operations: dont add things when
            // the context does not have anything outstanding that would require a 
            // barrier synchronization operation before this one. However, when 
            // we create outgoing dependences en mass at dispatch time, it's not
            // uncommon for us to call this several times in a row for the
            // same sync point. So only assert when the caller is trying to
            // create a sync point implicitly to manage this dependence
            assert(!ContextRequiresSync(pAsyncContext, eOperationType) || 
                   pExplicitSyncPoint != NULL);

            AsyncDependence * pDep = (pExplicitSyncPoint == NULL) ?
                pAsyncContext->CreateDependence(eOperationType) :
                pAsyncContext->CreateDependence(pExplicitSyncPoint, eOperationType);
            pDep->AddRef();

            if(ASYNCOP_ISREAD(eOperationType)) {
                
                // if the operations is a read, it needs to be ordered after any
                // current outstanding writes. However, since it can be concurrent
                // with any outstanding reads, this operation does not impact the
                // write frontier.
                
                m_vReadFrontier.push_back(pDep);
                m_vOutstandingReadAccelerators.insert(pDep->GetContext()->GetDeviceContext());

            } else {

                // if the operation is a write, then it must be ordered after 
                // the old write frontier and any outstanding reads. Consequently,
                // we can retire both frontiers.
                
                assert(ASYNCOP_ISWRITE(eOperationType));
                RetireDependenceFrontier(m_vReadFrontier, m_vOutstandingReadAccelerators, NULL);
                RetireDependenceFrontier(m_vWriteFrontier, m_vOutstandingWriteAccelerators, NULL);
                m_vWriteFrontier.push_back(pDep);
                m_vOutstandingWriteAccelerators.insert(pDep->GetContext()->GetDeviceContext());
            }

            CHECK_DEPENDENCE_INVARIANTS();
            bResult = TRUE;
        }
        MARKRANGEEXIT();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the set of accelerators involved in dependences that must
    ///             be resolved before an operation of the given type can be executed
    ///             without incurring a read-write hazard. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/26/2013. </remarks>
    ///
    /// <param name="eOpType">  Type of the operation. </param>
    ///
    /// <returns>   null if it fails, else the outstanding accelerator dependences. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::set<Accelerator*>*
    PBuffer::GetOutstandingAcceleratorDependences(
        __in ASYNCHRONOUS_OPTYPE eOpType
        )
    {
        if(ASYNCOP_ISREAD(eOpType))
            return &m_vOutstandingWriteAccelerators;
        return m_vWriteFrontier.size() != 0 ? 
            &m_vOutstandingReadAccelerators :
            &m_vOutstandingWriteAccelerators;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this buffer has any outstanding asynchronous ops: this is mostly a debug
    ///             tool for asserting that after a buffer has been involved in a synchronous
    ///             operation, all its outstanding conflicting operations have completed.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/2/2013. </remarks>
    ///
    /// <param name="eOpType">  Type of the operation. </param>
    ///
    /// <returns>   true if outstanding asynchronous ops, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    PBuffer::HasOutstandingAsyncOps(
        __in ASYNCHRONOUS_OPTYPE eOpType
        )
    {
        std::vector<std::deque<AsyncDependence*>*>::iterator vi;
        std::vector<std::deque<AsyncDependence*>*> vFrontiers;
        std::deque<AsyncDependence*>::iterator di;
        std::deque<AsyncDependence*>* pFrontier;
        if(m_vWriteFrontier.size())
            vFrontiers.push_back(&m_vWriteFrontier);
        if(ASYNCOP_ISWRITE(eOpType) && m_vReadFrontier.size()) 
            vFrontiers.push_back(&m_vReadFrontier);
        if(vFrontiers.size() == 0)
            return FALSE;

        UINT uiOutstandingCount = 0;
        for(vi=vFrontiers.begin(); vi!=vFrontiers.end(); vi++) {
            pFrontier = *vi;
            for(di=pFrontier->begin(); di!=pFrontier->end(); di++) {
                uiOutstandingCount += ((*di)->QueryOutstandingFlag() ? 1 : 0);
            }
        }
        return uiOutstandingCount > 0;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if this context is already the only relevant context
    ///             on the dependence frontier for the given operation type. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="eOpType">          Type of the operation. </param>
    ///
    /// <returns>   true if outstanding context, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL  
    PBuffer::IsSingletonOutstandingContext(
        __in AsyncContext * pAsyncContext,
        __in ASYNCHRONOUS_OPTYPE eOpType
        )
    {
        BOOL bWriteSingleton = FALSE;
        BOOL bReadSingleton = FALSE;

        if(ASYNCOP_ISWRITE(eOpType)) {

            // a write must be ordered after all writes and
            // reads on any other context. Since anything on the read frontier
            // is ordered after the write frontier by construction, it suffices
            // if we are the only inhabitants of the read frontier. 

            if(m_vReadFrontier.size()) {
                bReadSingleton = TRUE;
                std::deque<AsyncDependence*>::iterator di;
                for(di=m_vReadFrontier.begin(); di!=m_vReadFrontier.end(); di++) {
                    AsyncDependence * pDep = *di;
                    AsyncContext * pOutstandingContext = pDep->GetContext();
                    if(pOutstandingContext != pAsyncContext) {
                        bReadSingleton = FALSE;
                    }
                }

                // the read frontier, if non-empyt entirely determines whether we must create dependences--
                // either this context is the only thing on it, in which case we can queue new writes without
                // syncing, or there are operations from another context, in which case, we must sync no matter
                // what is in the write frontier. 

                return bReadSingleton;
            }

        }


        // if we got here we are a reader, or a writer but no outstanding reads are in the dependence
        // frontiers. A read need only be ordered after previous writes. check the write frontier, and
        // return TRUE if the last write was performed on this context. 
        assert(ASYNCOP_ISREAD(eOpType) || m_vReadFrontier.size() == 0);
            
        if(m_vWriteFrontier.size() != 0) {

            assert(m_vWriteFrontier.size() == 1);
            AsyncDependence * pDep = m_vWriteFrontier.back();
            AsyncContext * pOutstandingContext = pDep->GetContext();
            if(pOutstandingContext == pAsyncContext ||
                pOutstandingContext->GetPlatformContextObject() == 
                pAsyncContext->GetPlatformContextObject()) {
                bWriteSingleton = TRUE;
            }
            return bWriteSingleton;
        }

        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Context requires synchronise. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    PBuffer::ContextRequiresSync(
        __in AsyncContext * pAsyncContext,
        __in ASYNCHRONOUS_OPTYPE eOperationType,
        __out std::set<AsyncDependence*>* pDependences
        )
    {
        UNREFERENCED_PARAMETER(pDependences);

        // return true if doing asynchronous operations on this buffer in the given async context
        // requires a sync before hand. Essentially this method returns true if there are outstanding
        // operations on any context other than this one. If there are no outstanding ops, or all of
        // them are oun the same context as that being queried, then we can just add more async ops to
        // the outstanding list for that context (since the device must process them in order)
        
        BOOL bOutstandingWrites = m_vWriteFrontier.size() > 0;
        BOOL bOutstandingReads = m_vReadFrontier.size() > 0;
        if(!bOutstandingReads && !bOutstandingWrites)
            return FALSE;
        
        BOOL bSyncRequired = FALSE;
        if(pAsyncContext == NULL || !pAsyncContext->SupportsExplicitAsyncOperations() ||
            !IsSingletonOutstandingContext(pAsyncContext, eOperationType)) {
            
            // if the given context is NULL (or doesn't support async ops because it's
            // e.g. a host accelerator context), it means all proposed operations will be synchronous by
            // default. Hence, we must wait if there is any outstanding read-write conflict.           
            //   OR
            // if the outstanding operations frontier for this buffer has entries (relevant to the
            // op type) that were queued on this context, then we require no synchronization. 
            // check for that case and return accordingly.             

            bSyncRequired = bOutstandingWrites || (bOutstandingReads && ASYNCOP_ISWRITE(eOperationType));
        } 

        COLLECT_SYNC_DEPENDENCES(bSyncRequired, m_vReadFrontier, m_vWriteFrontier, pDependences);
        return bSyncRequired;

    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps an initialise data. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="uiStride">     The stride. </param>
    /// <param name="uiX">          The x coordinate. </param>
    /// <param name="uiY">          The y coordinate. </param>
    /// <param name="uiZ">          The z coordinate. </param>
    /// <param name="pInitData">    [in,out] If non-null, information describing the initialise. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    PBuffer::DumpInitData(
        VOID * pInitData
        )
    {
#ifdef DANDELION_DEBUG
        UINT uiStride   = m_pParent->GetStride();
        UINT uiX        = m_pParent->GetXElementCount();
        UINT uiY        = m_pParent->GetYElementCount();
        UINT uiZ        = m_pParent->GetZElementCount();
        UINT nBytes     = uiX * uiY * uiZ * uiStride;
        if (pInitData == NULL) {
            printf("Creating PBuffer withouth data (with dimensions for %d bytes of data)\n", nBytes);
        } else {
            printf("Creating PBuffer with %d bytes of data\n", nBytes);
            for (UINT i=0; i<nBytes; i++)  {
                printf("%u ", ((char*)pInitData)[i]);
                if (i % 4 == 3)  {
                    printf("; ");
                }
                if (i % 32 == 31) {
                    printf("\n");
                }
            }
            printf("\n");
        }
#else
        UNREFERENCED_PARAMETER(pInitData);
#endif
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is a "logically" empty buffer: in many cases we must
    ///             allocate a non-zero size buffer so that we have a block that can be bound as a
    ///             dispatch parameter, even when the buffer is logically empty. A side-effect of
    ///             this is that we must carefully track the fact that this buffer is actually empty
    ///             so that ports bound as descriptor ports (which infer other parameters such as
    ///             record count automatically) do the right thing when the buffer size is non-zero
    ///             but the logical size is zero.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/11/2012. </remarks>
    ///
    /// <returns>   true if empty buffer, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    PBuffer::IsEmptyBuffer(
        VOID
        )
    {
        return m_bIsLogicallyEmpty;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is physical buffer pinned. </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///
    /// <returns>   true if physical buffer pinned, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    PBuffer::IsPhysicalBufferPinned(
        VOID
        ) 
    { 
        return m_bPhysicalBufferPinned; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return true if the derived class supports a memset API. </summary>
    ///
    /// <remarks>   crossbac, 8/14/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    PBuffer::SupportsMemset(
        VOID
        )
    {
        return FALSE;
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
    PBuffer::FillExtent(
        int nValue, 
        size_t szExtentBytes
        )
    {      
        UNREFERENCED_PARAMETER(nValue);
        UNREFERENCED_PARAMETER(szExtentBytes);
        assert(FALSE);
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check dependence invariants. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    PBuffer::CheckDependenceInvariants(
        VOID
        ) 
    {
#ifdef DEBUG
        //deque<AsyncDependence*>::iterator di;
        //void * pLastCO = NULL;
        //for(di=m_vAsyncDependences.begin(); di!=m_vAsyncDependences.end(); di++) {
        //    AsyncDependence * pDep = *di;
        //    pDep->Lock();
        //    void * pPSCO = pDep->GetPlatformContextObject();
        //    if(pPSCO == NULL) {
        //        // if we have a null context object, it can only
        //        // occur if we are dealing with dependence created in some
        //        // accelerator's default context, meaning the task pointer is null.
        //        AsyncContext * pAsyncContext = pDep->GetContext();
        //        assert(pAsyncContext != NULL);
        //        Task * pTask = pAsyncContext->GetTaskContext();
        //        assert(pTask == NULL);
        //    }
        //    pDep->Unlock();
        //    assert(pLastCO == NULL || pLastCO == pPSCO);
        //    pLastCO = pPSCO;
        // }
#endif
    }

    void * PBuffer::m_pProfiler = NULL;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initialises the allocation profiler. </summary>
    ///
    /// <remarks>   Crossbac, 9/25/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    PBuffer::InitializeProfiler(
        VOID
        )
    {
        if(PTask::Runtime::GetPBufferProfilingEnabled() && m_pProfiler == NULL) {
            create_pbuffer_profiler(m_pProfiler);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deinit allocation profiler. </summary>
    ///
    /// <remarks>   Crossbac, 9/25/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    PBuffer::DeinitializeProfiler(
        VOID
        )
    {
        if(PTask::Runtime::GetPBufferProfilingEnabled() && m_pProfiler != NULL) {
            destroy_pbuffer_profiler(m_pProfiler);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   report platform buffer allocation stats. </summary>
    ///
    /// <remarks>   Crossbac, 9/25/2012. </remarks>
    ///
    /// <param name="ios">  [in,out] The allocate in bytes. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    PBuffer::ProfilerReport(
        std::ostream &ios
        )
    {
        if(Runtime::GetProfilePlatformBuffers() && m_pProfiler != NULL) {
            PBufferProfiler * pProfiler = reinterpret_cast<PBufferProfiler*>(m_pProfiler);
            pProfiler->Report(ios);
        }
    }
    
};
