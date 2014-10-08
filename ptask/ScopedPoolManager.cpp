///-------------------------------------------------------------------------------------------------
// file:	ScopedPoolManager.cpp
//
// summary:	Implements the scoped pool manager class
///-------------------------------------------------------------------------------------------------

#include <stdio.h>
#include <crtdbg.h>
#include "ScopedPoolManager.h"
#include "ptaskutils.h"
#include "PTaskRuntime.h"
#include "Scheduler.h"
#include "graph.h"
#include "ptgc.h"

namespace PTask {


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Default constructor. </summary>
    ///
    /// <remarks>   crossbac, 8/20/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    ScopedPoolManager::ScopedPoolManager(
        Graph * pScopedObject
        ) : Lockable("Scoped Pool Manager")
    {
        m_pGraph = pScopedObject;
        m_bPoolsAllocated = FALSE;
        m_bDestroyed = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   crossbac, 8/20/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    ScopedPoolManager::~ScopedPoolManager(
        void
        )
    {
        DestroyPools();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Require block pool. </summary>
    ///
    /// <remarks>   crossbac, 8/20/2013. </remarks>
    ///
    /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
    /// <param name="nDataSize">        Size of the data. </param>
    /// <param name="nMetaSize">        Size of the meta. </param>
    /// <param name="nTemplateSize">    Size of the template. </param>
    /// <param name="nBlocks">          (Optional) The blocks. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    ScopedPoolManager::RequireBlockPool(
        __in DatablockTemplate * pTemplate,
        __in int                 nDataSize, 
        __in int                 nMetaSize, 
        __in int                 nTemplateSize,
        __in int                 nBlocks
        )
    {
        // if there is no template,
        // this is really an untyped pool
        if(pTemplate == NULL)
            return RequireBlockPool(nDataSize, 
                                    nMetaSize, 
                                    nTemplateSize, 
                                    nBlocks);

        BOOL bResult = TRUE;
        Lock();
        if(!m_bPoolsAllocated) {
            std::map<DatablockTemplate*, ScopedPoolManager::POOLDESCRIPTOR>::iterator mi;
            mi = m_vRequiredPoolsTyped.find(pTemplate);
            if(mi != m_vRequiredPoolsTyped.end()) {
                bResult = FALSE;
                PTask::Runtime::MandatoryInform("%s::%s ignoring request to pool blocks for template:%s on graph %s (already exists)\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                pTemplate->GetTemplateName(),
                                                m_pGraph->GetName());
            } else {
                m_vRequiredPoolsTyped[pTemplate] = std::make_tuple(pTemplate, nDataSize, nMetaSize, nTemplateSize, nBlocks);
                bResult = TRUE;
            }
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Require block pool. </summary>
    ///
    /// <remarks>   crossbac, 8/20/2013. </remarks>
    ///
    /// <param name="nDataSize">        Size of the data. </param>
    /// <param name="nMetaSize">        Size of the meta. </param>
    /// <param name="nTemplateSize">    Size of the template. </param>
    /// <param name="nBlocks">          (Optional) The blocks. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    ScopedPoolManager::RequireBlockPool(
        __in int                 nDataSize, 
        __in int                 nMetaSize, 
        __in int                 nTemplateSize,
        __in int                 nBlocks
        )
    {
        if(nDataSize == 0) {
            PTask::Runtime::MandatoryInform("%s::%s ignoring request to pool 0-size blocks\n",
                                            __FILE__,
                                            __FUNCTION__);
            return FALSE;
        }            

        BOOL bResult = TRUE;
        Lock();
        if(!m_bPoolsAllocated) {
            std::map<int, ScopedPoolManager::POOLDESCRIPTOR>::iterator mi;
            mi = m_vRequiredPoolsUntyped.find(nDataSize);
            if(mi != m_vRequiredPoolsUntyped.end()) {
                bResult = FALSE;
                PTask::Runtime::MandatoryInform("%s::%s ignoring request to pool %d-size blocks (already exists)\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                nDataSize);
            } else {
                m_vRequiredPoolsUntyped[nDataSize] = std::make_tuple(((DatablockTemplate*)NULL), nDataSize, nMetaSize, nTemplateSize, nBlocks);
            }
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Require block pool. </summary>
    ///
    /// <remarks>   crossbac, 8/20/2013. </remarks>
    ///
    /// <param name="pTemplate">    [in,out] If non-null, the template. </param>
    /// <param name="nBlocks">      (Optional) The blocks. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    ScopedPoolManager::RequireBlockPool(
        __in DatablockTemplate * pTemplate,
        __in int                 nBlocks
        )
    {
        return RequireBlockPool(pTemplate, 0, 0, 0, nBlocks);
    }
        
	///-------------------------------------------------------------------------------------------------
	/// <summary>	Allocate global pools. </summary>
	///
	/// <remarks>	crossbac, 8/14/2013. </remarks>
	///-------------------------------------------------------------------------------------------------

	BOOL
	ScopedPoolManager::AllocatePools(
		VOID
		)
	{
        Lock();                 
        if(!m_bPoolsAllocated && (m_vRequiredPoolsTyped.size() || m_vRequiredPoolsUntyped.size())) {

		    std::vector<Accelerator*> oPoolAcc;
		    std::vector<Accelerator*> oPoolPL;
		    std::set<Accelerator*> oLockSet;
		    std::map<ACCELERATOR_CLASS, std::set<Accelerator*>>* ppAccMap;
		    ppAccMap = Scheduler::EnumerateBlockPoolAccelerators();
		    std::map<ACCELERATOR_CLASS, std::set<Accelerator*>>& pAccMap = *ppAccMap;
		    std::set<Accelerator*>::iterator aci;

            if(m_pGraph->GetStrictAffinitizedAccelerator() != NULL) {

                // if the graph has a scoped affinity to a particular accelerator,
                // it suffices to find that accelerator and be done with it. 
                // otherwise, we need to go over all available accelerator objects.
                
                Accelerator * pStrictAcc = m_pGraph->GetStrictAffinitizedAccelerator();
			    oPoolAcc.push_back(pStrictAcc);
			    oPoolPL.push_back(pStrictAcc);
			    oLockSet.insert(pStrictAcc);

            } else {

                // this graph has no strict affinity. this means that creating a scoped
                // block pool must consider all the available accelerator objects to 
                // which tasks may be dynamically bound. typically, this means query for
                // all the accelerators of the given classes for which we support pooling.
                
		        for(aci=pAccMap[ACCELERATOR_CLASS_CUDA].begin(); aci!=pAccMap[ACCELERATOR_CLASS_CUDA].end(); aci++) {
			        Accelerator * pAccelerator = *aci;
			        oPoolAcc.push_back(pAccelerator);
			        oPoolPL.push_back(pAccelerator);
			        oLockSet.insert(pAccelerator);
		        }
            }
		    Accelerator * pHostAcc = *(pAccMap[ACCELERATOR_CLASS_HOST].begin());
		    oPoolPL.push_back(pHostAcc);
		    oLockSet.insert(pHostAcc);

            // lock any accelerators we require
            // to create backing buffers for blocks
		    std::set<Accelerator*>::iterator lsi;
		    std::map<BlockPoolOwner*, std::vector<Accelerator*>*>::iterator pi;
		    for(lsi=oLockSet.begin(); lsi!=oLockSet.end(); lsi++) 
			    (*lsi)->Lock();

            int nDefaultPoolSize = PTask::Runtime::GetDefaultInputChannelBlockPoolSize();
            std::map<DatablockTemplate*, POOLDESCRIPTOR>::iterator di;
            for(di=m_vRequiredPoolsTyped.begin(); di!=m_vRequiredPoolsTyped.end(); di++) {

                // allocate typed block pools
                // typed pools are ones that map to a template
                std::map<DatablockTemplate*, GlobalBlockPool*>::iterator gi;
                gi = m_vTypedBlockPools.find(di->first);
                if(gi == m_vTypedBlockPools.end()) {

                    // the pool does not exist yet...so create it!
                    // we don't yet support pooling for blocks that
                    // have both a template and prescribed channel sizes.
                    // complain about it if the programmer has requested such.
                    
                    int nDataSize = std::get<1>(di->second);
                    int nMetaSize = std::get<2>(di->second);
                    int nTemplateSize = std::get<3>(di->second);
                    int nBlocks = std::get<4>(di->second);

                    if(nDataSize || nMetaSize || nTemplateSize) {

                        // still unsupported--blocks with type and channel size
                        // typically occur on internal channels, and are best managed by
                        // primitive-initiated pooling on the relevant ports/channels.
                        
                        PTask::Runtime::MandatoryInform("%s::%s ignoring channel sizes (%d,%d,%d) on block pool for (%s)--unimplemented!\n",
                                                        __FILE__,
                                                        __FUNCTION__,
                                                        nDataSize,
                                                        nMetaSize,
                                                        nTemplateSize,
                                                        di->first->GetTemplateName());
                    } 

                    GlobalBlockPool * pPool = new GlobalBlockPool(di->first, ACCELERATOR_CLASS_CUDA, PT_ACCESS_DEFAULT);
                    pPool->AllocateBlockPool(&oPoolPL, nBlocks);
                    BlockPoolOwner::RegisterActivePoolOwner(m_pGraph, pPool);
                    m_vTypedBlockPools[di->first] = pPool;

                    PTask::Runtime::Inform("Created scoped pool(%s), block-count: %d\n", 
                                           di->first->GetTemplateName(),
                                           nBlocks?nBlocks:nDefaultPoolSize);


                } else {

                    // we already have a pool that matches this
                    PTask::Runtime::MandatoryInform("%s::%s attempt to allocate scoped block pool(%s)--already exists!\n",
                                                    __FILE__,
                                                    __FUNCTION__,
                                                    di->first->GetTemplateName());
                }
            }

            std::map<int, POOLDESCRIPTOR>::iterator ti;
            for(ti=m_vRequiredPoolsUntyped.begin(); ti!=m_vRequiredPoolsUntyped.end(); ti++) {

                // allocate untyped pools
                // these may or may not have a template,
                // but have data/meta/template sizes 
                // typically, dandelion is the client for these

                std::map<int, GlobalBlockPool*>::iterator gpi = m_vUntypedBlockPools.find(ti->first);
                BUFFERACCESSFLAGS eFlags = PT_ACCESS_DEFAULT;
                ACCELERATOR_CLASS eClass = ACCELERATOR_CLASS_CUDA;
                int nDataSize = std::get<1>(ti->second);
                int nMetaSize = std::get<2>(ti->second);
                int nTemplateSize = std::get<3>(ti->second);
                int nBlocks = std::get<4>(ti->second);

                if(gpi == m_vUntypedBlockPools.end()) {

                    // no such pool yet...create it.
                    GlobalBlockPool * pPool = new GlobalBlockPool(nDataSize, nMetaSize, nTemplateSize, eClass, eFlags);
                    pPool->AllocateBlockPool(&oPoolPL, nBlocks);
                    BlockPoolOwner::RegisterActivePoolOwner(m_pGraph, pPool);
                    m_vUntypedBlockPools[ti->first] = pPool;  
                    PTask::Runtime::Inform("Created scoped pool(%d, %d, %d), block-count: %d\n", 
                                           nDataSize,
                                           nMetaSize,
                                           nTemplateSize,
                                           nBlocks?nBlocks:nDefaultPoolSize);

                } else {

                    // we already have a pool for that data size.
                    // TODO: check for meta size/template size match.
                    // Maybe there is a legitimate need to key on more than
                    // just the data channel size.

                    PTask::Runtime::MandatoryInform("%s::%s attempt to allocate scoped block pool(%d,%d,%d,%d)--already exists!\n",
                                                    __FILE__,
                                                    __FUNCTION__,
                                                    nDataSize,
                                                    nMetaSize,
                                                    nTemplateSize,
                                                    nBlocks);
                }
            }

            // unlock any accelerators we locked 
            // before creating the pools
            for(lsi=oLockSet.begin(); lsi!=oLockSet.end(); lsi++) 
			    (*lsi)->Unlock();
            m_bPoolsAllocated = TRUE;
        }

        Unlock();                   
        return m_bPoolsAllocated;
	}
     
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Find a block pool for the block. If there is no good fit,
    ///             create one if the bCreateIfNotFound flag is set. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 8/30/2013. </remarks>
    ///
    /// <param name="pBlock">               [in,out] If non-null, the block. </param>
    /// <param name="bCreateIfNotFound">    The create if not found. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    ScopedPoolManager::AddBlockToBestFitPool(
        __in Datablock * pBlock,
        __in BOOL bCreateIfNotFound
        )
    {
        BOOL bResult = FALSE;
        assert(pBlock != NULL);
        assert(PTask::Runtime::GetBlockPoolsEnabled());
        if(!PTask::Runtime::GetBlockPoolsEnabled() || pBlock == NULL)
            return bResult;


        pBlock->Lock();
        DatablockTemplate * pTemplate = pBlock->GetTemplate();
        UINT uiDataBufferSizeBytes = pBlock->GetDataBufferAllocatedSizeBytes();
        UINT uiMetaBufferSizeBytes = pBlock->GetMetaBufferAllocatedSizeBytes();
        UINT uiTemplBufferSizeBytes = pBlock->GetTemplateBufferAllocatedSizeBytes();
        pBlock->Unlock();

        Lock();                 
        BOOL bFoundMatch = FALSE;
        GlobalBlockPool * pPool = NULL;
        if(pTemplate) {

            // if the block has a template, require a match
            // on the actual template pointer for now.
            std::map<DatablockTemplate*, GlobalBlockPool*>::iterator mi;
            mi = m_vTypedBlockPools.find(pTemplate);
            if(mi != m_vTypedBlockPools.end()) {
                pPool = mi->second;
                bFoundMatch = TRUE;
            }

        } else {

            // here we require an exact match too--we could try to put it
            // in a "best" fit pool, but since pool allocators also do best
            // fit, it is better if the pool allocator knows how big the actual
            // blocks in the pool are.
            
            std::map<int, GlobalBlockPool*>::iterator mi;
            mi=m_vUntypedBlockPools.find(uiDataBufferSizeBytes);
            if(mi!=m_vUntypedBlockPools.end()) {
                pPool = mi->second;
                bFoundMatch = TRUE;
            }
        }

        if(!bFoundMatch && bCreateIfNotFound) {

            Lock();
            if(pTemplate) {
                m_vRequiredPoolsTyped[pTemplate] = 
                    std::make_tuple(pTemplate, 
                                    uiDataBufferSizeBytes, 
                                    uiMetaBufferSizeBytes, 
                                    uiTemplBufferSizeBytes, 
                                    1);
                pPool = new GlobalBlockPool(pTemplate, ACCELERATOR_CLASS_CUDA, PT_ACCESS_DEFAULT);
                pPool->ConfigureBlockPool();
                m_vTypedBlockPools[pTemplate] = pPool;
            } else {
                m_vRequiredPoolsUntyped[uiDataBufferSizeBytes] = 
                    std::make_tuple(((DatablockTemplate*)NULL), 
                                    uiDataBufferSizeBytes, 
                                    uiMetaBufferSizeBytes, 
                                    uiTemplBufferSizeBytes, 
                                    1);
                pPool = new GlobalBlockPool(uiDataBufferSizeBytes, 
                                      uiMetaBufferSizeBytes,
                                      uiTemplBufferSizeBytes,
                                      ACCELERATOR_CLASS_HOST,
                                      PT_ACCESS_DEFAULT);
                pPool->ConfigureBlockPool();
                m_vUntypedBlockPools[uiDataBufferSizeBytes] = pPool;

            }
            if(pPool != NULL) {
                BlockPoolOwner::RegisterActivePoolOwner(NULL, pPool);
                bFoundMatch = TRUE;
            }
            Unlock();
        }

        if(bFoundMatch && pPool) {

            // we found (or created) block pool to return this
            // block to. We know by construction that the block doesn't
            // know this pool will be its owner--so we need to set ownership too.
            pBlock->Lock();
            pBlock->SetPooledBlock(pPool);
            pBlock->ClearResizeFlags();
            pBlock->Unlock();
            pPool->ReturnToPool(pBlock);
            bResult = TRUE;

        } else {

            // we did not find (or create) a block pool for this block.
            // Consequently, we are forced to just queue it for GC
            GarbageCollector::QueueForGC(pBlock);
            bResult = FALSE;
        }

        Unlock();         
        return bResult;
    }

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Destroys the global pools. </summary>
	///
	/// <remarks>	crossbac, 8/14/2013. </remarks>
	///-------------------------------------------------------------------------------------------------

	BOOL
	ScopedPoolManager::DestroyPools(		
		VOID
		)
	{
        // the pools lock should already be held when this is called.
        // so assert that this is the case. To make the condition survivable
        // in release builds, go ahead and acquire the lock anyway--the lock
        // re-entrant, so it's harmless if the locking discipline was folllowed, 
        // might prevent a race otherwise. 

        Lock(); 

        BOOL bResult = FALSE;
        if(!m_bDestroyed) {

            assert(m_bPoolsAllocated || !(m_vTypedBlockPools.size() || m_vUntypedBlockPools.size()));
            std::map<DatablockTemplate*, GlobalBlockPool*>::iterator di;
            std::map<int, GlobalBlockPool*>::iterator ti;
            for(di=m_vTypedBlockPools.begin(); di!=m_vTypedBlockPools.end(); di++) {
                BlockPoolOwner::RetirePoolOwner(di->second);
                if(di->second) delete di->second;
            }
            for(ti=m_vUntypedBlockPools.begin(); ti!=m_vUntypedBlockPools.end(); ti++) {
                BlockPoolOwner::RetirePoolOwner(ti->second);
                if(ti->second) delete ti->second;
            }

            m_vTypedBlockPools.clear();
            m_vUntypedBlockPools.clear();
            m_bDestroyed = TRUE;
        }

        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Searches for the first matching pool. </summary>
    ///
    /// <remarks>   crossbac, 8/20/2013. </remarks>
    ///
    /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
    /// <param name="uiDataSize">       Size of the data. </param>
    /// <param name="uiMetaSize">       Size of the meta. </param>
    /// <param name="uiTemplateSize">   Size of the template. </param>
    ///
    /// <returns>   null if it fails, else the found matching pool. </returns>
    ///-------------------------------------------------------------------------------------------------

    GlobalBlockPool * 
    ScopedPoolManager::FindMatchingPool(
        __in DatablockTemplate * pTemplate,
        __in UINT                uiDataSize,
        __in UINT                uiMetaSize,
        __in UINT                uiTemplateSize
        )
    {
        UNREFERENCED_PARAMETER(uiMetaSize);
        UNREFERENCED_PARAMETER(uiTemplateSize);
        GlobalBlockPool * pPool = NULL;
        Lock();

        if(pTemplate) {

            // if we have a template, require a match
            // on the actual template pointer for now,
            // don't bother supporting channel sizes yet
            
            if(uiDataSize == 0 || pTemplate == Datablock::m_gSizeDescriptorTemplate) {
                std::map<DatablockTemplate*, GlobalBlockPool*>::iterator mi;
                mi = m_vTypedBlockPools.find(pTemplate);
                if(mi != m_vTypedBlockPools.end()) {
                    pPool = mi->second;
                }
            }

        } else {

            int nBestDelta = INT_MAX;
            std::map<int, GlobalBlockPool*>::iterator mi;
            std::map<int, GlobalBlockPool*>::iterator best = m_vUntypedBlockPools.end();
            for(mi=m_vUntypedBlockPools.begin(); mi!=m_vUntypedBlockPools.end(); mi++) {
                if(mi->first >= (int)uiDataSize) {
                    int nDelta = (int)uiDataSize - mi->first;
                    if(best == m_vUntypedBlockPools.end() || nDelta < nBestDelta) {
                        best = mi;
                        nBestDelta = nDelta;
                    } 
                }
            }
            if(best != m_vUntypedBlockPools.end()) 
                pPool = best->second;
        }

        Unlock();
        return pPool;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Request a pooled block. </summary>
    ///
    /// <remarks>   crossbac, 8/21/2013. </remarks>
    ///
    /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
    /// <param name="uiDataSize">       Size of the data. </param>
    /// <param name="uiMetaSize">       Size of the meta. </param>
    /// <param name="uiTemplateSize">   Size of the template. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    ScopedPoolManager::RequestBlock(
        __in DatablockTemplate * pTemplate,
        __in UINT                uiDataSize,
        __in UINT                uiMetaSize,
        __in UINT                uiTemplateSize
        )
    {
        return AllocateDatablock(pTemplate,
                                 uiDataSize,
                                 uiMetaSize,
                                 uiTemplateSize);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate datablock. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTemplate">        [in,out] If non-null, the datablock template. </param>
    /// <param name="uiDataSize">       Size of the data. </param>
    /// <param name="uiMetaSize">       Size of the meta. </param>
    /// <param name="uiTemplateSize">   Size of the template. </param>
    ///
    /// <returns>   null if it fails, else. </returns>   
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    ScopedPoolManager::AllocateDatablock(
        __in DatablockTemplate * pTemplate,
        __in UINT                uiDataSize,
        __in UINT                uiMetaSize,
        __in UINT                uiTemplateSize
        )
    {            
        GlobalBlockPool * pPool = FindMatchingPool(pTemplate,
                                                   uiDataSize,
                                                   uiMetaSize,
                                                   uiTemplateSize);

		Datablock * pBlock = NULL;
        if(pPool != NULL) {
            if(pTemplate || pPool->IsMatchingRequest(uiDataSize, 
											         uiMetaSize, 
											         uiTemplateSize)) {
			    pBlock = pPool->GetPooledBlock();
            }
		}
        return pBlock;
    }

};
