///-------------------------------------------------------------------------------------------------
// file:	BlockPool.cpp
//
// summary:	Implements the block pool class
///-------------------------------------------------------------------------------------------------

#include "AsyncContext.h"
#include "BlockPool.h"
#include "OutputPort.h"
#include "InputPort.h"
#include "PTaskRuntime.h"
#include "MemorySpace.h"
#include "task.h"
#include "assert.h"
#include "ptgc.h"

#ifdef DEBUG
#define CHECK_POOL_INVARIANTS(pBlock) \
    assert(!Contains(pBlock)); \
    assert(!Contains(NULL));
#define CHECK_BLOCK_REFERENCED(pBlock) \
    pBlock->Lock(); \
    assert(pBlock->RefCount() != 0); \
    pBlock->Unlock();
#define CHECK_POOL_STATES()   CheckPoolStates()
#else
#define CHECK_POOL_INVARIANTS(pBlock)
#define CHECK_BLOCK_REFERENCED(pBlock)
#define CHECK_POOL_STATES()
#endif

using namespace std;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    BlockPool::BlockPool(
        DatablockTemplate * pTemplate,
        BUFFERACCESSFLAGS ePermissions,
        UINT uiPoolSize,
        BlockPoolOwner * pBlockPoolOwner
        ) : Lockable(NULL)
    {
        m_bEnabled = FALSE;
        m_pTemplate = pTemplate;
        m_bPoolHintsSet = FALSE;
        m_nPoolHintPoolSize = 0;
        m_nPoolHintStride = 0;
        m_nPoolHintDataBytes = 0;
        m_nPoolHintMetaBytes = 0;
        m_nPoolHintTemplateBytes = 0;
        m_nMaxPoolSize = uiPoolSize == 0 ? PTask::Runtime::GetICBlockPoolSize() : uiPoolSize;
        m_bPageLockHostViews = FALSE;
        m_bEagerDeviceMaterialize = FALSE;
        m_ePermissions = ePermissions;
        m_bGrowable = FALSE;
        m_bHasInitialValue = pTemplate != NULL && pTemplate->HasInitialValue();
        m_pPoolOwner = pBlockPoolOwner;
        m_uiHighWaterMark = 0;
        m_uiOwnedBlocks = 0;
        m_uiLowWaterMark = MAXDWORD32;
        m_uiGrowIncrement = PTask::Runtime::GetDefaultBlockPoolGrowIncrement();
        if(m_bHasInitialValue) {
            m_vInitialValue.lpvAddress = const_cast<void*>(pTemplate->GetInitialValue());
            m_vInitialValue.bPinned = FALSE;
            m_vInitialValue.uiSizeBytes = pTemplate->GetInitialValueSizeBytes();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    BlockPool::~BlockPool() {
        ReleaseBlocks();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets request page locked. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="bPageLocked">  true to lock, false to unlock the page. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    BlockPool::SetRequestsPageLocked(
        BOOL bPageLocked
        )
    {
        m_bPageLockHostViews = bPageLocked;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets request page locked. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    BlockPool::GetRequestsPageLocked(
        VOID
        )
    {
        return m_bPageLockHostViews;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a growable. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="bGrowable">    true if growable. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    BlockPool::SetGrowable(
        BOOL bGrowable
        )
    {
        m_bGrowable = bGrowable;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is growable. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if growable, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    BlockPool::IsGrowable(
        VOID
        )
    {
        return m_bGrowable;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets grow increment. </summary>
    ///
    /// <remarks>   crossbac, 6/20/2013. </remarks>
    ///
    /// <param name="uiBlockCount"> Number of blocks. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    BlockPool::SetGrowIncrement(
        UINT uiBlockCount
        ) 
    {
        m_uiGrowIncrement = uiBlockCount;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets grow increment. </summary>
    ///
    /// <remarks>   crossbac, 6/20/2013. </remarks>
    ///
    /// <returns>   The grow increment. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    BlockPool::GetGrowIncrement(
        VOID
        )
    {
        return m_uiGrowIncrement;
    }        

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets eager device materialize. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="bEager">   true to eager. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    BlockPool::SetEagerDeviceMaterialize(
        BOOL bEager
        )
    {
        m_bEagerDeviceMaterialize = bEager;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets eager device materialize. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    BlockPool::GetEagerDeviceMaterialize(
        VOID
        )
    {
        return m_bEagerDeviceMaterialize;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets pool size. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="uiSize">   The size. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    BlockPool::SetPoolSize(
        UINT uiSize
        )
    {
        m_nMaxPoolSize = uiSize;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets pool size. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   The pool size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    BlockPool::GetPoolSize(
        VOID
        )
    {
        return m_nMaxPoolSize;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Grows the pool by the given number of blocks. </summary>
    ///
    /// <remarks>   crossbac, 6/20/2013. </remarks>
    ///
    /// <param name="uiBlockCount"> Number of blocks. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    BlockPool::Grow(
        UINT uiBlockCount
        )
    {
        assert(m_bGrowable);
        if(m_bGrowable && uiBlockCount) {

            if(PTask::Runtime::GetProvisionBlockPoolsForCapacity()) {
                PTask::Runtime::MandatoryInform("XXXX: BlockPool: %s empty, growing from %d by %d on critical path!\n", 
                                                m_pPoolOwner->GetPoolOwnerName(),
                                                m_uiOwnedBlocks,
                                                uiBlockCount);
            }
            
            for(UINT ui=0; ui<uiBlockCount; ui++) {
                Datablock * pBlock = AllocateBlockForPool();
                AddNewBlock(pBlock);
            }
        }

    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a destination buffer for a block with an upstream
    /// 			allocator. Succeeds only if the pool happens to have blocks
    /// 			backed by sufficient resources in all channels that are backed. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pAccelerator"> (optional) [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   null if it fails, else the destination buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *		
    BlockPool::GetPooledBlock(
        __in Accelerator * pAccelerator,
        __in UINT uiDataBytes,
        __in UINT uiMetaBytes,
        __in UINT uiTemplateBytes
        )
    {
        if(!PTask::Runtime::GetBlockPoolsEnabled()) return NULL;
        UNREFERENCED_PARAMETER(pAccelerator);
        Datablock * pBlock = NULL;

        Lock();       

        if(!m_bEnabled) {
            Unlock();
            PTask::Runtime::MandatoryInform("XXXX: BlockPool: %s alloc attempt on disabled pool!\n", m_pPoolOwner->GetPoolOwnerName());
            return NULL;

        }

        BOOL bAddNewBlock = FALSE;
        if((m_pBlockPool.size() == 0) && m_bGrowable && m_uiGrowIncrement != 0) {
            Grow(m_uiGrowIncrement);
            bAddNewBlock = TRUE;
        }

        assert(m_pBlockPool.size() || !bAddNewBlock);
        if(m_pBlockPool.size() > 0) {

            if(m_bPoolHintsSet == TRUE) {
            
                assert(uiDataBytes != 0);
                BOOL bDimensionsOK = m_nPoolHintDataBytes >= uiDataBytes &&
                                     m_nPoolHintMetaBytes >= uiMetaBytes &&
                                     m_nPoolHintTemplateBytes >= uiTemplateBytes;
                BOOL bBlockAvailable = m_pBlockPool.size() > 0;
                if(bDimensionsOK && bBlockAvailable) {

                    // if we got a block that is big enough we should try to use it. However, if it doesn't have
                    // buffers in the memory space we need, there isn't really any savings from using it, so don't
                    // bother unless it has extant buffers for the given accelerator. 

                    Datablock * pCandidateBlock = m_pBlockPool.front();
                    pCandidateBlock->Lock();
                    if(pAccelerator == NULL || pCandidateBlock->HasBuffers(pAccelerator)) {
                        assert(pCandidateBlock->GetDataBufferAllocatedSizeBytes() >= uiDataBytes);
                        assert(pCandidateBlock->GetMetaBufferAllocatedSizeBytes() >= uiMetaBytes);
                        assert(pCandidateBlock->GetTemplateBufferAllocatedSizeBytes() >= uiTemplateBytes);
                        m_pBlockPool.pop_front();
                        pBlock = pCandidateBlock;
                    }
                    pCandidateBlock->Unlock();
                    CHECK_POOL_INVARIANTS(pBlock);
                }
            } else {
                pBlock = m_pBlockPool.front();
                m_pBlockPool.pop_front();
                if(m_bHasInitialValue) {
                    pBlock->Lock();
                    if(!pBlock->GetPoolInitialValueValid()) {
                        pBlock->ResetInitialValueForPool(pAccelerator);
                    }
                    pBlock->Unlock();
                }
            }
        }

        m_uiLowWaterMark = min(m_uiLowWaterMark, (UINT)m_pBlockPool.size());
        Unlock();
        if(!pBlock) {
            assert(!m_bGrowable);
        }
        return pBlock;
    }    

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets high water mark. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    BlockPool::GetHighWaterMark(
        VOID
        )
    {
        return m_uiHighWaterMark;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets high water mark. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    BlockPool::GetOwnedBlockCount(
        VOID
        )
    {
        return m_uiOwnedBlocks;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the low water mark. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    BlockPool::GetLowWaterMark(
        VOID
        )
    {
        return m_uiLowWaterMark;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the currently available count. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    BlockPool::GetAvailableBlockCount(
        VOID
        )
    {
        return static_cast<UINT>(m_pBlockPool.size());
    }



    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pool contains block. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    BlockPool::Contains(
        Datablock * pBlock
        )
    {
        BOOL bResult = FALSE;
#ifdef DEBUG
        Lock();
        for(deque<Datablock*>::iterator vi=m_pBlockPool.begin();
            vi!=m_pBlockPool.end();
            vi++) {
            if(*vi == pBlock) {
                bResult = TRUE;
                break;
            }
        }
        Unlock();
#else
        pBlock; // suppress compiler complaints
#endif
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate block. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator">     [in,out] If non-null, the accelerator. </param>
    /// <param name="pAsyncContext">    [in,out] (optional)  If non-null, the async context in which
    ///                                 the block will be used. Logically, this uniquely determined by
    ///                                 the combination of accelerator and task. Concretely, this
    ///                                 typically maps to an object like a cuda stream, a command queue,
    ///                                 or an ID3D11ImmediateContext*. </param>
    /// <param name="bPooled">          true if the allocated block should be added to the block pool
    ///                                 maintained by this port. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *  
    BlockPool::AllocateBlockForPool(
        VOID
        ) 
    {
        if(m_bPoolHintsSet) {
            assert(m_pTemplate != Datablock::m_gSizeDescriptorTemplate);
            return AllocateBlockWithPoolHint(m_nPoolHintDataBytes, 
                                             m_nPoolHintMetaBytes, 
                                             m_nPoolHintTemplateBytes);
        }

        Datablock * pBlock = NULL;
        Lock();      

        if(m_pTemplate == Datablock::m_gSizeDescriptorTemplate) {
           
            pBlock = Datablock::CreateEmptySizeDescriptorBlock();

        } else {

            HOSTMEMORYEXTENT * pExtent = m_bHasInitialValue ? &m_vInitialValue : NULL;

            if(m_bPageLockHostViews) {

                BOOL bCreateDeviceBuffers = m_bEagerDeviceMaterialize;
                pBlock = Datablock::CreateDatablockAsync(m_vAccelerators,
                                                         m_pTemplate,
                                                         pExtent,
                                                         m_ePermissions,
                                                         DBCTLC_NONE,
                                                         bCreateDeviceBuffers,
                                                         m_bEagerDeviceMaterialize,
                                                         m_bPageLockHostViews);

            } else {
        
                pBlock = Datablock::CreateDatablock(NULL,
                                                    m_pTemplate,
                                                    pExtent,
                                                    m_ePermissions,
                                                    DBCTLC_NONE);

                pBlock->Lock();

                if(m_bEagerDeviceMaterialize) {

                    std::set<Accelerator*>::iterator si;
                    for(si=m_vAccelerators.begin(); si!=m_vAccelerators.end(); si++) {
                        Accelerator * pAccelerator = *si;
                        AsyncContext * pAsyncContext = pAccelerator->GetAsyncContext(ASYNCCTXT_XFERHTOD);
                        assert(pAccelerator->LockIsHeld());
                        if(!pBlock->HasBuffers(pAccelerator))
                            pBlock->AllocateBuffers(pAccelerator, pAsyncContext, TRUE);
                    }
                }

                pBlock->Unlock();
            }
        }

        assert(pBlock != NULL);
#ifdef DEBUG
        assert(!Contains(pBlock));
        assert(!Contains(NULL));
#endif
        Unlock();
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate block. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="bFinalized">   [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *  
    BlockPool::AllocateBlockForPoolAsync(
        BOOL &bFinalized
        ) 
    {
        bFinalized = FALSE;
        Datablock * pBlock = NULL;
        Lock();      

        if(m_pTemplate == Datablock::m_gSizeDescriptorTemplate) {
           
            pBlock = Datablock::CreateEmptySizeDescriptorBlock();

        } else {

            HOSTMEMORYEXTENT * pExtent = m_bHasInitialValue ? &m_vInitialValue : NULL;
            BOOL bMaterializeDeviceViewsNow = FALSE;
            BOOL bCreateDeviceBuffers = m_bEagerDeviceMaterialize;
            if(m_bHasInitialValue) {
                bFinalized = !m_bEagerDeviceMaterialize;                                        
            } else {
                bFinalized = TRUE;
            }
            pBlock = Datablock::CreateDatablockAsync(m_vAccelerators,
                                                     m_pTemplate,
                                                     pExtent,
                                                     m_ePermissions,
                                                     DBCTLC_NONE,
                                                     bCreateDeviceBuffers,
                                                     bMaterializeDeviceViewsNow,
                                                     m_bPageLockHostViews);
        }
        assert(pBlock != NULL);
#ifdef DEBUG
        assert(!Contains(pBlock));
        assert(!Contains(NULL));
#endif
        Unlock();
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Finalize a block allocated with the async variant. Basically
    ///             we need to populate any views on this pass.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/30/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    BlockPool::FinalizeBlock(
        __in Datablock * pBlock
        )
    {
        Lock();      

        if(m_bHasInitialValue && m_bEagerDeviceMaterialize) {
            pBlock->Lock();
            std::set<Accelerator*>::iterator si;
            for(si=m_vAccelerators.begin(); si!=m_vAccelerators.end(); si++) {
                Accelerator * pAccelerator = *si;
                AsyncContext * pAsyncContext = pAccelerator->GetAsyncContext(ASYNCCTXT_XFERHTOD);
                assert(pAccelerator->LockIsHeld());
                assert(pBlock->HasBuffers(pAccelerator));
                if(!pAccelerator->IsHost())
                    pBlock->UpdateView(pAccelerator, pAsyncContext, TRUE, BSTATE_SHARED, DBDATA_IDX, DBDATA_IDX);
            }
            pBlock->SetPoolInitialValueValid(TRUE);
            pBlock->Unlock();
        }
        assert(pBlock != NULL);
#ifdef DEBUG
        assert(!Contains(pBlock));
        assert(!Contains(NULL));
#endif
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate block. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] (optional)  If non-null, the async context. </param>
    /// <param name="bPooled">          true to pooled. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *  
    BlockPool::AllocateBlockWithPoolHint(
        __in UINT uiDataBytes,
        __in UINT uiMetaBytes,
        __in UINT uiTemplateBytes
        ) 
    {
        // assert(FALSE && "un-implemented (completely)");
        Datablock * pBlock = NULL;
        Lock();      

        HOSTMEMORYEXTENT * pExtent = m_bHasInitialValue ? &m_vInitialValue : NULL;

        if(m_bPageLockHostViews) {
            PTask::Runtime::Inform("page-locking unimplemented for pooled variable-length blocks.\n");
        } 
        Accelerator * pAccelerator = *(m_vAccelerators.begin());
        AsyncContext * pAsyncContext = pAccelerator->GetAsyncContext(ASYNCCTXT_XFERHTOD);
        pBlock = Datablock::CreateDatablock(pAsyncContext,
                                            m_pTemplate,
                                            uiDataBytes,
                                            uiMetaBytes,
                                            uiTemplateBytes,
                                            m_ePermissions,
                                            DBCTLC_NONE,
                                            pExtent,
                                            0,
                                            FALSE);
        pBlock->Lock();
        if(m_bEagerDeviceMaterialize) {
            std::set<Accelerator*>::iterator si;
            for(si=m_vAccelerators.begin(); si!=m_vAccelerators.end(); si++) {
                Accelerator * pAccelerator = *si;
                pAccelerator->Lock();
                if(!pBlock->HasBuffers(pAccelerator)) {
					pBlock->Seal(1, uiDataBytes, uiMetaBytes, uiTemplateBytes);
                    pBlock->AllocateBuffers(pAccelerator, NULL, TRUE, DBDATA_IDX, 
                                            pAccelerator->IsHost() ? 
                                                DBTEMPLATE_IDX :
                                                DBDATA_IDX);

					pBlock->Unseal();
				}
                pAccelerator->Unlock();
            }
        }
        assert(pBlock != NULL);
        pBlock->Unlock();
        Unlock();
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds to the pool. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    BlockPool::ReturnBlock(
        Datablock * pBlock
        )
    {
        assert(pBlock != NULL);
        if(pBlock == NULL)
            return; 

        pBlock->Lock();
        assert(pBlock->RefCount() == 0);
        assert(pBlock->GetPoolOwner() == m_pPoolOwner);
        assert(!m_bHasInitialValue || pBlock->GetPoolInitialValueValid());
        pBlock->ClearAllControlSignals();
        pBlock->Unlock();
        Lock();
        if(m_bEnabled) {
#ifdef DEBUG
            assert(!Contains(pBlock));
#endif
            m_pBlockPool.push_back(pBlock);
            m_uiHighWaterMark = max(m_uiHighWaterMark, static_cast<UINT>(m_pBlockPool.size()));
#ifdef DEBUG
            assert(!Contains(NULL));
#endif
        } else {
            GarbageCollector::QueueForGC(pBlock);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds to the pool. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    BlockPool::AddNewBlock(
        Datablock * pBlock
        )
    {
        assert(PTask::Runtime::GetBlockPoolsEnabled());
        assert(pBlock != NULL);
        pBlock->Lock();
        pBlock->SetPooledBlock(m_pPoolOwner);
        assert(pBlock->RefCount() == 0);
        pBlock->Unlock();
        Lock();
#ifdef DEBUG
        assert(!Contains(pBlock));
#endif
        m_pBlockPool.push_back(pBlock);
        m_uiHighWaterMark = max(m_uiHighWaterMark, static_cast<UINT>(m_pBlockPool.size()));
        m_uiOwnedBlocks++;
#ifdef DEBUG
        assert(!Contains(NULL));
#endif
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the pooled blocks described by.  </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    BlockPool::ReleaseBlocks(
        VOID
        )
    {
        Lock();
        vector<Datablock*> torelease;
        vector<Datablock*>::iterator vi;
        deque<Datablock*>::iterator di;
        for(di=m_pBlockPool.begin(); di!=m_pBlockPool.end(); di++) {
            // privatize the list so we 
            // we can release the blocks without
            // holding a lock (which would violate
            // lock-ordering for blocks/ports)
            torelease.push_back(*di);
        }
        m_pBlockPool.clear();
        Unlock();
        for(vi=torelease.begin(); vi!=torelease.end(); vi++) {
            Datablock * pBlock = (*vi);
            pBlock->Lock();
            assert(pBlock->RefCount()==0);
            pBlock->SetPooledBlock(NULL);
            pBlock->Unlock();
            GarbageCollector::QueueForGC(pBlock);
        }        
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Locks the target accelerators. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    BlockPool::LockTargetAccelerators(
        VOID
        )
    {
        std::set<Accelerator*>::iterator si;
        for(si=m_vAccelerators.begin(); si!=m_vAccelerators.end(); si++) 
            (*si)->Lock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unlocks the target accelerators. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    BlockPool::UnlockTargetAccelerators(
        VOID
        )
    {
        std::set<Accelerator*>::iterator si;
        for(si=m_vAccelerators.begin(); si!=m_vAccelerators.end(); si++) 
            (*si)->Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate block pool. Attempt to preallocate blocks on this port.
    ///             
    ///             Allocation of data-blocks and platform-specific buffers can be a signficant
    ///             latency expense at dispatch time. We can actually preallocate output datablocks
    ///             and create device- side buffers at graph construction time. For each node in the
    ///             graph, allocate data blocks on any output ports, and create device-specific
    ///             buffers for all accelerators capable of executing the node.
    ///             
    ///             Not all port types can profitably pool blocks. Hence, the superclass
    ///             implementation of this method does nothing.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/15/2012. </remarks>
    ///
    /// <param name="pAccelerators">    [in] If non-null, the accelerators on which views of blocks
    ///                                 allocated in the pool may be required. </param>
    /// <param name="uiPoolSize">       [in] (optional) Size of the pool. If zero/defaulted,
    ///                                     Runtime::GetICBlockPoolSize() will be used to determine the
    ///                                     size of the pool. </param>
    ///
    /// <returns>   True if it succeeds, false if it fails. If a port type doesn't actually implement
    ///             pooling, return false as well.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    BlockPool::AllocateBlockPool(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )
    {     
        if(!PTask::Runtime::GetBlockPoolsEnabled()) {
            PTask::Runtime::Inform("%s::%s: PTask block pools disabled--skipping allocation for pool: %s\n",
                                   __FILE__,
                                   __FUNCTION__,
                                   m_pPoolOwner->GetPoolOwnerName());
            return FALSE;
        }

        BOOL bHostFound = FALSE;
        BOOL bPoolConstructSuccess = TRUE;
        BOOL bUseHintDimensions = m_bPoolHintsSet;
        UINT uiDefaultSize = bUseHintDimensions ? m_nPoolHintPoolSize : PTask::Runtime::GetICBlockPoolSize();
        UINT uiActualPoolSize = (uiPoolSize > 0) ? uiPoolSize : uiDefaultSize;
        
        std::set<Accelerator*>::iterator si;
        std::vector<Accelerator*>::iterator vi;
        for(vi=pAccelerators->begin(); vi!=pAccelerators->end(); vi++) {
            Accelerator * pAccelerator = *vi;
            bHostFound |= pAccelerator->IsHost();
            m_vAccelerators.insert(pAccelerator);
        }
        LockTargetAccelerators();
        assert(m_vAccelerators.size());
        assert(bHostFound || !m_bPageLockHostViews);

        for(UINT i=0; i<uiActualPoolSize; i++) {

            Datablock * pDestBlock = AllocateBlockForPool();

            assert(pDestBlock != NULL);
            if(pDestBlock == NULL) {
                // if we've eaten all the memory, and the assert above hasn't helped 
                // us, it means this is a release build and we've got to try to recover.
                // in that case, just keep plowing going and be sure to return false
                PTask::Runtime::Warning("block pool pre-allocation failed...recovering...");
                bPoolConstructSuccess = FALSE;
                break;
            }
            AddNewBlock(pDestBlock);
        }
        UnlockTargetAccelerators();

        m_bEnabled = bPoolConstructSuccess;
        m_uiLowWaterMark  = static_cast<UINT>(m_pBlockPool.size());
        m_uiHighWaterMark = static_cast<UINT>(m_pBlockPool.size());
        return bPoolConstructSuccess;
    }       

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destroys the block pool. AddRef everything in the bool, set its owner
    ///             to null, and then release it. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    BlockPool::DestroyBlockPool(
        VOID
        )
    {
        Lock();
        ReleaseBlocks();
        m_bEnabled = FALSE;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate block pool. Attempt to preallocate blocks on this port.
    ///             Asynchronous version. Only allocates device-space buffers
    ///             in the first pass. Second pass queues all the copies.
    ///             This function handles only the first pass.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/15/2012. </remarks>
    ///
    /// <param name="pAccelerators">    [in] If non-null, the accelerators on which views of blocks
    ///                                 allocated in the pool may be required. </param>
    /// <param name="uiPoolSize">       [in] (optional) Size of the pool. If zero/defaulted,
    /// 								Runtime::GetICBlockPoolSize() will be used to determine the
    /// 								size of the pool. </param>
    /// 								
    /// <returns>   True if it succeeds, false if it fails. If a port type doesn't actually implement
    ///             pooling, return false as well.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    BlockPool::AllocateBlockPoolAsync(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )
    {
        if(!PTask::Runtime::GetBlockPoolsEnabled()) {
            PTask::Runtime::Inform("%s::%s: PTask block pools disabled--skipping allocation for pool: %s\n",
                                   __FILE__,
                                   __FUNCTION__,
                                   m_pPoolOwner->GetPoolOwnerName());
            return FALSE;
        }

        BOOL bHostFound = FALSE;
        BOOL bPoolConstructSuccess = TRUE;
        BOOL bUseHintDimensions = m_bPoolHintsSet;
        UINT uiDefaultSize = bUseHintDimensions ? m_nPoolHintPoolSize : PTask::Runtime::GetICBlockPoolSize();
        UINT uiActualPoolSize = (uiPoolSize > 0) ? uiPoolSize : uiDefaultSize;
        
        std::set<Accelerator*>::iterator si;
        std::vector<Accelerator*>::iterator vi;
        for(vi=pAccelerators->begin(); vi!=pAccelerators->end(); vi++) {
            Accelerator * pAccelerator = *vi;
            assert(pAccelerator->LockIsHeld());
            bHostFound |= pAccelerator->IsHost();
            m_vAccelerators.insert(pAccelerator);
        }

        assert(m_vAccelerators.size());
        assert(bHostFound || !m_bPageLockHostViews);

        for(UINT i=0; i<uiActualPoolSize; i++) {

            BOOL bFinalized = FALSE;            
            Datablock * pDestBlock = AllocateBlockForPoolAsync(bFinalized);

            assert(pDestBlock != NULL);
            if(pDestBlock == NULL) {
                // if we've eaten all the memory, and the assert above hasn't helped 
                // us, it means this is a release build and we've got to try to recover.
                // in that case, just keep plowing going and be sure to return false
                PTask::Runtime::Warning("block pool pre-allocation failed...recovering...");
                bPoolConstructSuccess = FALSE;
                break;
            }

            if(bFinalized) {
                AddNewBlock(pDestBlock);                
            } else {
                m_vOutstandingBlocks.push_back(pDestBlock);
            }
        }

        return bPoolConstructSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate block pool. Attempt to preallocate blocks on this port.
    ///             Asynchronous version. Only allocates device-space buffers
    ///             in the first pass. Second pass queues all the copies.
    ///             This function handles the second pass.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/15/2012. </remarks>
    ///
    /// <param name="pAccelerators">    [in] If non-null, the accelerators on which views of blocks
    ///                                 allocated in the pool may be required. </param>
    /// <param name="uiPoolSize">       [in] (optional) Size of the pool. If zero/defaulted,
    /// 								Runtime::GetICBlockPoolSize() will be used to determine the
    /// 								size of the pool. </param>
    /// 								
    /// <returns>   True if it succeeds, false if it fails. If a port type doesn't actually implement
    ///             pooling, return false as well.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    BlockPool::FinalizeBlockPoolAsync(
        VOID
        )
    {
        Lock();
        std::vector<Datablock*>::iterator vi;
        for(vi=m_vOutstandingBlocks.begin(); vi!=m_vOutstandingBlocks.end(); vi++) {
            Datablock * pBlock = *vi;
            FinalizeBlock(pBlock);
            AddNewBlock(pBlock);            
        }
        m_vOutstandingBlocks.clear();
        m_bEnabled = TRUE;
        m_uiLowWaterMark = static_cast<UINT>(m_pBlockPool.size());
        Unlock();
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Force block pooling for a port that has an up-stream allocator. In general, when
    ///             we have an upstream allocator (meta) port, the runtime will not create a block
    ///             pool for the corresponding output port. This turns out to put device-side
    ///             allocation on the critical path in some cases, so we provide a way to override
    ///             that behavior and allow a port to create a pool based on some size hints. When
    ///             there is a block available with sufficient space in the pool, the meta port can
    ///             avoid the allocation and draw from the pool.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 9/25/2012. </remarks>
    ///
    /// <param name="nPoolSize">                Size of the block pool. </param>
    /// <param name="nStride">                  The stride. </param>
    /// <param name="nDataBytes">               The data in bytes. </param>
    /// <param name="nMetaBytes">               The meta in bytes. </param>
    /// <param name="nTemplateBytes">           The template in bytes. </param>
    /// <param name="bPageLockHostViews">       True if host buffers for datablocks in this pool
    ///                                         should be allocated from page-locked memory. </param>
    /// <param name="bEagerDeviceMaterialize">  true to eager device materialize. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    BlockPool::ForceBlockPoolHint(
        __in UINT nPoolSize,
        __in UINT nStride,
        __in UINT nDataBytes,
        __in UINT nMetaBytes,
        __in UINT nTemplateBytes,
        __in BOOL bPageLockHostViews,
        __in BOOL bEagerDeviceMaterialize
        )
    {
        Lock();
        m_bPoolHintsSet = TRUE;
        m_nPoolHintPoolSize = nPoolSize;
        m_nPoolHintStride = nStride;
        m_nPoolHintDataBytes = nDataBytes;
        m_nPoolHintMetaBytes = nMetaBytes;
        m_nPoolHintTemplateBytes = nTemplateBytes;   
        m_bEagerDeviceMaterialize = bEagerDeviceMaterialize;
        m_bPageLockHostViews = bPageLockHostViews;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has block pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if block pool, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    BlockPool::HasBlockPool(
        VOID
        )
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the pool is enabled. </summary>
    ///
    /// <remarks>   crossbac, 6/18/2013. </remarks>
    ///
    /// <returns>   true if enabled, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    BlockPool::IsEnabled(
        VOID
        )
    {
        // note that we do not require a lock to check this flag. This is because it is used within the
        // Datablock::Release() method (indirectly) to decide whether to return a block to a pool or
        // delete it. The ReturnToPool method checks the enabled flag as well (with a lock). Because
        // once a pool is disabled it will never be re-enabled, the only possible race on this flag is
        // when a releasing thread attempts to return a block to the pool concurrently with a thread
        // that disables the pool by calling DestroyBlockPool. In such a case, the call to ReturnToPool
        // is responsible for queuing the block for GC. In the common case when there is no race, some
        // work restoring the block to its initial usable state is elided. 
        return m_bEnabled;
    }



#ifdef GRAPH_DIAGNOSTICS

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check that the block pool contain only datablocks with no control signals. </summary>
    ///
    /// <remarks>   Crossbac, 3/2/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    BlockPool::CheckBlockPoolStates(
        VOID
        )
    {
        Lock();
        deque<Datablock*>::iterator di;
        for(di=m_pBlockPool.begin(); di!=m_pBlockPool.end(); di++) {
            Datablock * pBlock = *di;
            pBlock->Lock();
            if(pBlock->HasAnyControlSignal()) {
                cout << this << " block pool contains ctl-signal block: " << pBlock << endl;
                assert(!pBlock->HasAnyControlSignal());
            }
            pBlock->Unlock(); 
        }
        Unlock();
    }
#endif

};
