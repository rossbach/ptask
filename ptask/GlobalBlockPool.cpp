///-------------------------------------------------------------------------------------------------
// file:	GlobalBlockPool.cpp
//
// summary:	Implements the global block pool class
///-------------------------------------------------------------------------------------------------

#include <stdio.h>
#include <crtdbg.h>
#include "GlobalBlockPool.h"
#include "ptaskutils.h"
#include "InputPort.h"
#include "OutputPort.h"
#include "task.h"
#include "GlobalPoolManager.h"
#include "ptgc.h"

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Constructor. </summary>
    ///
    /// <remarks>	Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pTemplate">	[in,out] If non-null, the template. </param>
    ///-------------------------------------------------------------------------------------------------

    GlobalBlockPool::GlobalBlockPool(
        __in DatablockTemplate * pTemplate,
		__in ACCELERATOR_CLASS  eAcceleratorClass,
		__in BUFFERACCESSFLAGS	ePermissions
        ) : Lockable(NULL)
    {
        m_pBlockPool = NULL;
        m_bHasBlockPool = TRUE;
		m_eAcceleratorClass = eAcceleratorClass;
		m_ePermissions = ePermissions;
		m_pTemplate = pTemplate;
		m_nDataBytes = 0;
		m_nMetaBytes = 0;
		m_nTemplateBytes = 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Constructor. </summary>
    ///
    /// <remarks>	Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pTemplate">	[in,out] If non-null, the template. </param>
    ///-------------------------------------------------------------------------------------------------

    GlobalBlockPool::GlobalBlockPool(
        __in UINT nDataBytes,
        __in UINT nMetaBytes,
        __in UINT nTemplateBytes,
		__in ACCELERATOR_CLASS  eAcceleratorClass,
		__in BUFFERACCESSFLAGS	ePermissions
        ) : Lockable(NULL)
    {
        m_pBlockPool = NULL;
        m_bHasBlockPool = TRUE;
		m_eAcceleratorClass = eAcceleratorClass;
		m_ePermissions = ePermissions;
		m_pTemplate = NULL;
		m_nDataBytes = nDataBytes;
		m_nMetaBytes = nMetaBytes;
		m_nTemplateBytes = nTemplateBytes;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    GlobalBlockPool::~GlobalBlockPool() {
        if(m_pBlockPool) {
            m_pBlockPool->DestroyBlockPool();
            delete m_pBlockPool;
            m_pBlockPool = NULL;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has block pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if block pool, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    GlobalBlockPool::HasBlockPool(
        VOID
        )
    {
        return m_bHasBlockPool;
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
    /// <param name="bPageLockHostViews">       (optional) the page lock host views. </param>
    /// <param name="bEagerDeviceMaterialize">  (optional) the eager device materialize. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    GlobalBlockPool::ForceBlockPoolHint(
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
        m_pBlockPool->ForceBlockPoolHint(nPoolSize,
                                         nStride,
                                         nDataBytes,
                                         nMetaBytes,
                                         nTemplateBytes,
                                         bPageLockHostViews,
                                         bEagerDeviceMaterialize);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate block pool. Attempt to preallocate blocks on this port.
    ///             
    ///             Allocation of data-blocks and platform-specific buffers can be a signficant
    ///             latency expense at dispatch time. We can actually preallocate output datablocks
    ///             and create device- side buffers at graph construction time. For each node in the
    ///             graph, allocate data blocks on any output ports, and create device-specific
    ///             buffers for all accelerators capable of executing the node.
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
    GlobalBlockPool::AllocateBlockPool(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )        
    {
        if(!m_bHasBlockPool)
            return FALSE;

        if(!PTask::Runtime::GetBlockPoolsEnabled()) {
            PTask::Runtime::Inform("%s::%s: PTask block pools disabled--skipping allocation for pool: %s\n",
                                   __FILE__,
                                   __FUNCTION__,
                                   "GlobalPool**");
            return FALSE;
        }

        Lock();
        if(ConfigureBlockPool(uiPoolSize)) {
            m_pBlockPool->AllocateBlockPool(pAccelerators, uiPoolSize);
        }
        Unlock();
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Configure the block pool. Do not allocate any blocks for it yet. </summary>
    ///
    /// <remarks>   crossbac, 6/15/2012. </remarks>
    ///
    /// <param name="uiPoolSize">   [in] Size of the pool. If zero/defaulted,
    ///                             Runtime::GetICBlockPoolSize() will be used to determine the size
    ///                             of the pool. </param>
    ///
    /// <returns>   True if it succeeds, false if it fails. If a port type doesn't actually implement
    ///             pooling, return false as well.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    GlobalBlockPool::ConfigureBlockPool(
        __in unsigned int uiPoolSize
        )        
    {
        if(!m_bHasBlockPool)
            return FALSE;

        if(!PTask::Runtime::GetBlockPoolsEnabled()) {
            PTask::Runtime::Inform("%s::%s: PTask block pools disabled--skipping configuration for pool: %s\n",
                                   __FILE__,
                                   __FUNCTION__,
                                   "GlobalPool**");
            return FALSE;
        }

        Lock();
        m_pBlockPool = new BlockPool(m_pTemplate, m_ePermissions, uiPoolSize, this);
        m_pBlockPool->SetEagerDeviceMaterialize(m_eAcceleratorClass != ACCELERATOR_CLASS_HOST);
        m_pBlockPool->SetRequestsPageLocked(m_eAcceleratorClass != ACCELERATOR_CLASS_HOST); 
        m_pBlockPool->SetGrowable(FALSE);
		if(m_pTemplate == NULL) {
			BOOL bPageLockHostViews = m_eAcceleratorClass != ACCELERATOR_CLASS_HOST;
			BOOL bEagerDeviceMaterialize = m_eAcceleratorClass != ACCELERATOR_CLASS_HOST;
			ForceBlockPoolHint(uiPoolSize, 1, m_nDataBytes, m_nMetaBytes, m_nTemplateBytes, bPageLockHostViews, bEagerDeviceMaterialize);
		}
        Unlock();
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets high water mark. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    GlobalBlockPool::GetHighWaterMark(
        VOID
        )
    {
        if(!m_bHasBlockPool || m_pBlockPool == NULL) 
            return 0;
        return m_pBlockPool->GetHighWaterMark();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the total number of blocks owned by the pool. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The total number of blocks owned by the pool (whether they are queued or not). </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    GlobalBlockPool::GetOwnedBlockCount(
        VOID
        )
    {
        if(!m_bHasBlockPool || m_pBlockPool == NULL) 
            return 0;
        return m_pBlockPool->GetOwnedBlockCount();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the low water mark. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    GlobalBlockPool::GetLowWaterMark(
        VOID
        )
    {
        if(m_pBlockPool == NULL) 
            return 0;
        return m_pBlockPool->GetLowWaterMark();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the currently available count. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    GlobalBlockPool::GetAvailableBlockCount(
        VOID
        )
    {
        if(m_pBlockPool == NULL) 
            return 0;
        return m_pBlockPool->GetAvailableBlockCount();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destroys the block pool. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GlobalBlockPool::DestroyBlockPool(
        VOID
        )
    {
        if(m_pBlockPool) {
            m_pBlockPool->DestroyBlockPool();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Queries if a block pool is active. </summary>
    ///
    /// <remarks>   crossbac, 6/18/2013. </remarks>
    ///
    /// <returns>   true if a block pool is active, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    GlobalBlockPool::IsBlockPoolActive(
        VOID
        )
    {
        if(m_pBlockPool) {
            return m_pBlockPool->IsEnabled();
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the owner name. </summary>
    ///
    /// <remarks>   crossbac, 6/18/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the owner name. </returns>
    ///-------------------------------------------------------------------------------------------------

    char *
    GlobalBlockPool::GetPoolOwnerName(
        VOID
        )
    {
        return "GlobalPool";
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
    GlobalBlockPool::AllocateBlockPoolAsync(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )
    {
        if(!m_bHasBlockPool) return FALSE;

        // figure out the pool size. If the user has defaulted the parameter, then take the default
        // input channel pool size. Additionally, if there a constraint on the capacity of this channel,
        // then (assuming the user program doesn't abuse the API) that capacity + 1 is the largest
        // number of blocks that can ever be in flight for this channel, so we can reduce the pool size
        // based on that constraint accordingly. Be sure to warn the user that we have done so though. 
         
        if(uiPoolSize == 0) {
            uiPoolSize = Runtime::GetDefaultInputChannelBlockPoolSize();
        }


        m_pBlockPool = new BlockPool(m_pTemplate, m_ePermissions, uiPoolSize, this);
        m_pBlockPool->SetEagerDeviceMaterialize(FALSE);
        m_pBlockPool->SetRequestsPageLocked(m_eAcceleratorClass != ACCELERATOR_CLASS_HOST); 
        m_pBlockPool->SetGrowable(TRUE);
        BOOL bResult = m_pBlockPool->AllocateBlockPoolAsync(pAccelerators, uiPoolSize);
        Unlock();
        return bResult;
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
    GlobalBlockPool::FinalizeBlockPoolAsync(
        VOID
        )
    {
        if(m_pBlockPool) 
            return m_pBlockPool->FinalizeBlockPoolAsync();
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets pooled block. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the pooled block. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    GlobalBlockPool::GetPooledBlock(
        VOID
        )
    {
        if(!PTask::Runtime::GetBlockPoolsEnabled()) 
            return NULL;

        Datablock * pBlock = NULL;
        Lock();
        if(m_bHasBlockPool) {
            pBlock = m_pBlockPool->GetPooledBlock(NULL, m_nDataBytes, m_nMetaBytes, m_nTemplateBytes);
            if(pBlock != NULL) {
#ifdef DEBUG
                pBlock->Lock();
                assert(pBlock->RefCount() == 0);
                pBlock->Unlock();
#endif
                // pBlock->AddRef();
            } else {
				if(m_pBlockPool->IsGrowable()) {
					PTask::Runtime::Inform("XXXX: GlobalBlockPool:%s empty block pool, returning null: EN=%d, GRW=%d\n", 
										   "GlobalPool", 
										   m_pBlockPool->IsEnabled(), 
										   m_pBlockPool->IsGrowable());
				}
            }
        }
        Unlock();
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return a block to the pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    GlobalBlockPool::AddNewBlock(
        Datablock * pBlock
        )
    {
        Lock();
        if(m_pBlockPool) {
            m_pBlockPool->AddNewBlock(pBlock);            
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return a block to the pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    GlobalBlockPool::ReturnToPool(
        Datablock * pBlock
        )
    {
        BOOL bReturnBlock = TRUE;
        assert(pBlock != NULL);
        if(pBlock == NULL) return;
        pBlock->Lock();
        UINT uiRefCount = pBlock->RefCount();
        BOOL bResized = pBlock->IsResizedBlock();
        BOOL bPooled = pBlock->IsPooled();
        BlockPoolOwner * pOwner = pBlock->GetPoolOwner();
        UNREFERENCED_PARAMETER(uiRefCount); // used in assert
        UNREFERENCED_PARAMETER(pOwner);     // used in assert
        UNREFERENCED_PARAMETER(bPooled);    // used in assert
        assert(bPooled && pOwner == this);
        assert(uiRefCount == 0);
        pBlock->Unlock();

        if(bResized) {
            BLOCKPOOLRESIZEPOLICY ePolicy = 
                PTask::Runtime::GetBlockPoolBlockResizePolicy();
            switch(ePolicy) {
            case BPRSP_EXIT_POOL:
                bReturnBlock = FALSE;
                pBlock->Lock();
                pBlock->SetPooledBlock(NULL);
                pBlock->Unlock();
                GarbageCollector::QueueForGC(pBlock);
                break;
            case BPRSP_REMAIN_IN_POOL:
                bReturnBlock = TRUE;
                break;
            case BPRSP_FIND_EXISTING_POOL:
            case BPRSP_FIND_OR_CREATE_POOL:
                bReturnBlock = FALSE;
                BOOL bCreateIfNotFound = ePolicy == BPRSP_FIND_OR_CREATE_POOL;
                GlobalPoolManager::AddBlockToBestFitPool(pBlock, bCreateIfNotFound);
                break;
            }
        }

        if(bReturnBlock) {
            Lock();
            if(m_pBlockPool) {            
                m_pBlockPool->ReturnBlock(pBlock);
            }
            Unlock();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   gets the pool size. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    UINT
    GlobalBlockPool::GetPoolSize(
        VOID
        )
    {
        UINT uiSize = 0; 
        Lock();
        if(m_pBlockPool) {
            uiSize = m_pBlockPool->GetPoolSize();
        }
        Unlock();    
        return uiSize;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets request page locked. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="bPageLocked">  true to lock, false to unlock the page. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    GlobalBlockPool::SetRequestsPageLocked(
        BOOL bPageLocked
        )
    {
        Lock();
        if(m_pBlockPool) {
            m_pBlockPool->SetRequestsPageLocked(bPageLocked);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets request page locked. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    GlobalBlockPool::GetRequestsPageLocked(
        VOID
        )
    {
        Lock();
        BOOL bResult = FALSE;
        if(m_pBlockPool) {
            bResult = m_pBlockPool->GetRequestsPageLocked();
        }
        Unlock();    
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is matching request. </summary>
    ///
    /// <remarks>   crossbac, 8/14/2013. </remarks>
    ///
    /// <param name="nDataBytes">       The data in bytes. </param>
    /// <param name="nMetaBytes">       The meta in bytes. </param>
    /// <param name="nTemplateBytes">   The template in bytes. </param>
    ///
    /// <returns>   true if matching request, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

	BOOL 
	GlobalBlockPool::IsMatchingRequest(			
		__in UINT nDataBytes,
		__in UINT nMetaBytes,
		__in UINT nTemplateBytes
		)
	{
		return nDataBytes == m_nDataBytes &&
			   nMetaBytes == m_nMetaBytes && 
			   nTemplateBytes == m_nTemplateBytes;
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
    GlobalBlockPool::GetBlockFromPool(
        __in Accelerator * pAccelerator,
        __in UINT uiDataBytes,
        __in UINT uiMetaBytes,
        __in UINT uiTemplateBytes
        )
	{
        UNREFERENCED_PARAMETER(pAccelerator);
		if(IsMatchingRequest(uiDataBytes, uiMetaBytes, uiTemplateBytes)) 
	        return GetPooledBlock();	
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is global pool. </summary>
    ///
    /// <remarks>   crossbac, 8/30/2013. </remarks>
    ///
    /// <returns>   true if global pool, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    GlobalBlockPool::BlockPoolIsGlobal(
        void
        )
    {
        return TRUE;
    }

};
