//--------------------------------------------------------------------------------------
// File: OutputPort.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "AsyncContext.h"
#include "OutputPort.h"
#include "InputPort.h"
#include "PTaskRuntime.h"
#include "MemorySpace.h"
#include "task.h"
#include "ptgc.h"
#include "assert.h"
#include "GlobalPoolManager.h"
#include "graph.h"
#include "signalprofiler.h"

#ifdef DEBUG
#define CHECK_BLOCK_POOL_INVARIANTS(pBlock) \
    assert(!PoolContainsBlock(pBlock)); \
    assert(!PoolContainsBlock(NULL));
#define CHECK_BLOCK_REFERENCED(pBlock) \
    pBlock->Lock(); \
    assert(pBlock->RefCount() != 0); \
    pBlock->Unlock();
#else
#define CHECK_BLOCK_POOL_INVARIANTS(pBlock)
#define CHECK_BLOCK_REFERENCED(pBlock)
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

    OutputPort::OutputPort(
        VOID
        ) 
    {
        m_uiId = NULL;
        m_pTemplate = NULL;
        m_ePortType = OUTPUT_PORT;
        m_pDatablock = NULL;
        m_pInOutProducer = NULL;
        m_pAllocatorPort = NULL;
        m_pControlPort = NULL;
        m_bPortOpen = TRUE;
        m_bInitialPortStateOpen = TRUE;
        m_bPoolHintsSet = FALSE;
        m_nPoolHintPoolSize = 0;
        m_nPoolHintStride = 0;
        m_nPoolHintDataBytes = 0;
        m_nPoolHintMetaBytes = 0;
        m_nPoolHintTemplateBytes = 0;
        m_nMaxPoolSize = 0;
        m_pDescribedPort = NULL;
        m_eDescriptorFunc = DF_METADATA_SOURCE;
        m_bPendingAllocation = FALSE;
        m_pPendingAllocationAccelerator = NULL;
        m_uiPendingAllocationSize = 0;    
        m_bBlockPoolActive = FALSE;
        m_uiPoolHighWaterMark = 0;
        m_uiOwnedBlockCount = 0;
        m_uiLowWaterMark = MAXDWORD32;
        m_bPageLockHostViews = FALSE;
        m_bExplicitMemSpaceTransitionPoint = FALSE;
        m_bExplicitMemSpaceTransitionPointSet = FALSE; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    OutputPort::~OutputPort() {
        ReleasePooledBlocks();
        PTSRELEASE(m_pDatablock);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destroys the block pool. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    OutputPort::DestroyBlockPool(
        VOID
        )
    {
        Lock();
        m_bBlockPoolActive = FALSE;
        ReleasePooledBlocks();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Is the block pool available/active? </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    OutputPort::IsBlockPoolActive(
        VOID
        )
    {
        return m_bBlockPoolActive;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets high water mark. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    OutputPort::GetHighWaterMark(
        VOID
        )
    {
        return m_uiPoolHighWaterMark;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the total number of blocks owned by the pool. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The total number of blocks owned by the pool (whether they are queued or not). </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    OutputPort::GetOwnedBlockCount(
        VOID
        )
    {
        return m_uiOwnedBlockCount;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the low water mark. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    OutputPort::GetLowWaterMark(
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
    OutputPort::GetAvailableBlockCount(
        VOID
        )
    {
        return static_cast<UINT>(m_pBlockPool.size());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' is occupied. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if occupied, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    OutputPort::IsOccupied(
        VOID
        ) 
    {
        BOOL bResult = FALSE;
        Lock();
        if(m_vChannels.size() != 0) {
            for(vector<Channel*>::iterator vi=m_vChannels.begin(); vi!=m_vChannels.end(); vi++) {
                if(!(*vi)->IsReady(CE_SRC)) {
                    bResult = TRUE;
                    break;
                }
            }
        }
        if(bResult && m_pControlPort != NULL) {
            // if this port is gated by another
            // port, it may be the case that we will
            // drop the output, so the port is not
            // really occupied. 
            if(m_pControlPort->GetPortType() != OUTPUT_PORT) {
                // if the controller is an output port, we
                // actually cannot predict whether we will
                // drop the output. If the controller is some
                // kind of input port, we can peek its datablock 
                // and see.
                Datablock * pBlock = m_pControlPort->Peek();
                pBlock->Lock();
                bResult = FALSE;
                if(pBlock->IsControlAndDataBlock()) {
                    // we are only going to produce output
                    // if we actually see an EOF block. 
                    bResult =  pBlock->HasAnyControlSignal();
                }
                pBlock->Unlock();
            }
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the datablock described by.  </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void	
    OutputPort::ReleaseDatablock(
        VOID
        ) 
    {
        Lock();
        if(m_pDatablock != NULL) {
            m_pDatablock->Release();
            m_pDatablock = NULL;
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return NULL.  </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *		
    OutputPort::Pull(
        VOID
        ) 
    {
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Returns NULL. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The VOID to peek. </param>
    ///
    /// <returns>   null if it fails, else the current top-of-stack object. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *		
    OutputPort::Peek(
        VOID
        ) 
    {
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pushes a datablock into this port. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="p">    [in,out] If non-null, the Datablock* to push. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL		
    OutputPort::Push(
        Datablock* p
        ) 
    {
        BOOL bSuccess = FALSE;
        BOOL bSignalTask = FALSE;
        Lock();
        // if this port is the control port gating some
        // other ports, be sure to open them if this
        // an EOF datablock.
        ctlpingress(this, p);
        ctlpdeclegressctr();
        if(m_pGatedPorts.size()) {
            // when an output port is the controller for
            // gated port(s), this indicates a situation
            // where GPU code is making the decision whether
            // to output a block. Since the GPU can only
            // communicate to the runtime through memory
            // transfers in a datablock we have to actually
            // look at what's in this data block.
            p->Lock();
            BOOL bEOF = p->IsEOF();
            if(!bEOF) {
                // examine block contents
                CONTROLSIGNAL * pEOF = (CONTROLSIGNAL*) p->GetDataPointer(FALSE);
                bEOF = (*pEOF != 0);
            }
            p->Unlock();
            if(bEOF) {                
                vector<Port*>::iterator pi;
                for(pi=m_pGatedPorts.begin(); pi!=m_pGatedPorts.end(); pi++) {
                    OutputPort * pPort = (OutputPort*)(*pi);
                    assert(pPort->GetPortType() == OUTPUT_PORT);
                    pPort->SignalGate();
                }
            }
        }
        // the port is always open for an non-gated port.
        // for a gated port, it should only be open if it
        // has a control port gating it. 
        assert(m_bPortOpen || m_pControlPort != NULL);
        if(m_bPortOpen || m_pControlPort == NULL) {
            if(m_vChannels.size() != 0) {
                bSuccess = TRUE;
                // if there are multiple channels downstream on 
                // this port, and the are not all readonly 
                // consumers, then we need to clone the block
                // for some consumers. If we use the
                // block itself for all readers each writer requires a clone. 
                // If all conflicts are writers, we need writer count - 1 clones.
                UINT nReaders = GetDownstreamReadonlyPortCount(p);
                UINT nWriters = GetDownstreamWriterPortCount(p);
                BOOL bRequiresClone = !m_bSuppressClones && ((nReaders && nWriters) || (nWriters > 1));
                UINT nCloneCount = 0;
                if(bRequiresClone) {
                    if(nReaders == 0) 
                        nCloneCount = nWriters - 1;
                    else
                        nCloneCount = nWriters;
                }
                assert(!bRequiresClone || nCloneCount > 0);
                vector<Channel*>::iterator vi;
                int nWriterIndex = 0;
                int nReaderIndex = 0;
                std::map<Channel*, Datablock*> vPushMap;
                for(vector<Channel*>::iterator vi=m_vChannels.begin(); vi!=m_vChannels.end(); vi++) {
                    // we may pass this datablock on to a
                    // consumer who may modify it, or who may 
                    // fail to read it before the next invocation
                    // of this ptask. 
                    Channel * pChannel = (*vi);
                    BOOL bPushClone = FALSE;                    
                    if(bRequiresClone && pChannel->PassesPredicate(CE_SRC, p)) {
                        Port * pDownstreamPort = pChannel->GetBoundPort(CE_DST);
                        BOOL bWriter = TRUE;
                        if(pDownstreamPort != NULL) {
                            // If there is a downstream port, this is an internal
                            // channel, which means the consumer can be a writer only
                            // if this port is an inout port. That can only occur for
                            // INPUT_PORT objects.
                            bWriter = ((pDownstreamPort->GetPortType() == INPUT_PORT) &&
                                       ((((InputPort*)pDownstreamPort)->GetInOutConsumer() != NULL) ||
                                                     (pDownstreamPort->IsDestructive())));
                        }
                        bPushClone = bWriter && ((nReaders == 0 && nWriterIndex) || nReaders);
                        if(bWriter) 
                            nWriterIndex++;
                        else 
                            nReaderIndex++;
                    }
                    if(bPushClone) {
                        p->Lock();
                        AsyncContext * pSrcAsyncContext = NULL;
                        AsyncContext * pDstAsyncContext = NULL;
                        Accelerator * pLastAccelerator = p->GetMostRecentAccelerator();
                        Accelerator * pDestAccelerator = pChannel->GetUnambigousDownstreamMemorySpace();
                        if(pLastAccelerator != NULL &&
                           pLastAccelerator->SupportsExplicitAsyncOperations() &&
                           pDestAccelerator != NULL && 
                           pDestAccelerator->SupportsExplicitAsyncOperations()) {
                           pSrcAsyncContext = pLastAccelerator->GetAsyncContext(ASYNCCTXT_XFERDTOD);
                           pDstAsyncContext = pDestAccelerator->GetAsyncContext(ASYNCCTXT_XFERDTOD);

                        } else if(pLastAccelerator != NULL)  {                        
                            pSrcAsyncContext = pLastAccelerator->GetAsyncContext(ASYNCCTXT_XFERDTOH);
                        }
                        Datablock * pClone = Datablock::Clone(p, pSrcAsyncContext, pDstAsyncContext);
                        p->Unlock();
                        pClone->AddRef();
                        nCloneCount--;
                        vPushMap[pChannel] = pClone;
                    } else {
                        vPushMap[pChannel] = p;
                    }
                }
                std::map<Channel*, Datablock*>::iterator mi;
                for(mi=vPushMap.begin(); mi!=vPushMap.end(); mi++) {
                    // push the clones first
                    Channel * pOutboundChannel = mi->first;
                    Datablock * pOutboundBlock = mi->second;
                    if(pOutboundBlock != p) {
                        
                        ctlpopegress(this, pOutboundBlock);
                        pOutboundChannel->Push(pOutboundBlock);
                        pOutboundBlock->Release();
                    } 
                }
                for(mi=vPushMap.begin(); mi!=vPushMap.end(); mi++) {
                    // push the clones first
                    Channel * pOutboundChannel = mi->first;
                    Datablock * pOutboundBlock = mi->second;
                    if(pOutboundBlock == p) {
                        ctlpopegress(this, pOutboundBlock);
                        pOutboundChannel->Push(pOutboundBlock);
                    } 
                }                
            }
            // if this is a gated port, it is only open because
            // the control port opened it for this invocation.
            // close the port after we produce the output. 
            if(m_pControlPort != NULL) {
                m_bPortOpen = m_bInitialPortStateOpen;
            }
        } else {
            // release the data block!
            // The one subtlety here is that we expect the consuming
            // channel to signal the ptask that a port status change
            // has occurred when the block is pulled. Since 
            // we are just swallowing the output, we have to signal
            // it instead.
            bSignalTask = TRUE;
        }
        Unlock();
        if(bSignalTask) {
            m_pBoundTask->SignalPortStatusChange();
        }
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates this object. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pTemplate">            [in,out] If non-null, the template. </param>
    /// <param name="uiId">                 The identifier. </param>
    /// <param name="lpszVariableBinding">  [in,out] If non-null, the variable binding. </param>
    /// <param name="nParmIdx">             Zero-based index of the n parm. </param>
    /// <param name="nInOutRouteIdx">       Zero-based index of the n in out route. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port*
    OutputPort::Create(
        DatablockTemplate * pTemplate,
        UINT uiId, 
        char * lpszVariableBinding,
        int nParmIdx,
        int nInOutRouteIdx
        )
    {
        OutputPort * pPort = new OutputPort();
        if(SUCCEEDED(pPort->Initialize(pTemplate, uiId, lpszVariableBinding, nParmIdx, nInOutRouteIdx)))
            return pPort;
        delete pPort;
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return the channel index on the target datablock to which
    ///             this port/variable should be bound. Typically this is the
    ///             DATABLOCK_DATA_CHANNEL. However, if this output port is a
    ///             descriptor of another output port, we may want to bind to
    ///             a different buffer in the block at dispatch time.  
    ///             </summary>
    ///
    /// <remarks>   Crossbac, </remarks>
    ///
    /// <param name="pAccelerator"> (optional) [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   null if it fails, destination channel index. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT		
    OutputPort::GetDestinationChannel(
        Accelerator * pAccelerator
        )
    {
        // the destination channel is a static property of the
        // graph. We can get it without needing a lock.
        UNREFERENCED_PARAMETER(pAccelerator);
        if(m_pDescribedPort != NULL) {
            switch(m_eDescriptorFunc) {
            case DF_DATA_SOURCE: return DBDATA_IDX;
            case DF_METADATA_SOURCE: return DBMETADATA_IDX;
            case DF_TEMPLATE_SOURCE: return DBTEMPLATE_IDX;
            }
        }
        return DBDATA_IDX;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a destination buffer. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in] (optional)  If non-null, the accelerator. </param>
    ///
    /// <returns>   null if it fails, else the destination buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *
    OutputPort::GetDestinationBuffer(
        __in Accelerator * pAccelerator
        ) 
    {
        Lock();
        if(!m_pDatablock) {
            
            // If we don't have an output buffer already, try to get one from our block pool. That is, of
            // course, unless there is an upstream meta port that controls block allocations on this port,
            // in which case we really *can't* perform the allocation because there is no guarantee that
            // upstream metaport has a block available. In such a case the block pool had best be empty!           
            
            if(m_pAllocatorPort != NULL) {
                
                // bad things are going on!
                assert(m_pBlockPool.size() == 0 && "XXX: non-empty block pool for meta-port bound OutputPort!");
                assert(m_pAllocatorPort == NULL && "XXX: block pool-allocation for meta-port bound OutputPort!");

            } else if(m_pBlockPool.size() > 0 && PTask::Runtime::GetBlockPoolsEnabled()) {

                // take the block at the head of the queue.
                m_pDatablock = m_pBlockPool.front();
                m_pBlockPool.pop_front();
                if(m_pTemplate->HasInitialValue() && m_vDirtySet.find(m_pDatablock)!=m_vDirtySet.end()) {
                    m_pDatablock->Lock();
                    assert(m_pDatablock->GetPoolInitialValueValid());
                    if(!m_pDatablock->GetPoolInitialValueValid()) {
                        m_pDatablock->ResetInitialValueForPool(pAccelerator);
                    }
                    m_pDatablock->Unlock();
                    m_vDirtySet.erase(m_pDatablock);
                }
                m_uiLowWaterMark = min(m_uiLowWaterMark, (UINT)m_pBlockPool.size());
                m_pDatablock->AddRef();
            }
        }

        if(!m_pDatablock) {

            // If we *still* don't have an output buffer already, we couldn't get one from the block pool.
            // So, allocate one. All the previous caveats still apply. If there is an upstream meta port 
            // that controls block allocations on this port, we *can't* perform the allocation.

            assert(m_pAllocatorPort == NULL && "XXX: auto-allocation for meta-port bound OutputPort!");
            AllocateDestinationBlock(pAccelerator);

        } 

        CHECK_BLOCK_POOL_INVARIANTS(m_pDatablock);
        Unlock();
        return m_pDatablock;
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
    OutputPort::GetPooledDestinationBuffer(
        __in Accelerator * pAccelerator,
        __in UINT uiDataBytes,
        __in UINT uiMetaBytes,
        __in UINT uiTemplateBytes
        )
    {
        UNREFERENCED_PARAMETER(pAccelerator);
        Datablock * pBlock = NULL;

        if(!PTask::Runtime::GetBlockPoolsEnabled()) 
            return NULL;

        Lock();       

        // This method is an optimization for meta-port/output-port pairs. In general, when there is an
        // upstream meta port that controls block allocations on this port, we really *can't* perform
        // the pre-allocation necessary to create a block pool because the block dimensions are
        // determined at runtime. In such a case the block pool is empty, unless the programmer used
        // ForceBlockPoolHint to force the runtime to pool some (conservatively sized) blocks anyway.
        // if the programmer has provided hints that help us over-ride this behavior, use them to see
        // if we can provide a block from the pool. Otherwise return null: the metaport logic has
        // the fall back code required to allocate blocks when there is no pool available. 

        if(m_pAllocatorPort != NULL &&  // there is an allocator for this port
            m_bPoolHintsSet == TRUE &&  // the programmer has provided hints
            m_pDatablock == NULL) {     // no block is already set as the destination
            
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
                if(pCandidateBlock->HasBuffers(pAccelerator)) {
                    assert(pCandidateBlock->GetDataBufferAllocatedSizeBytes() >= uiDataBytes);
                    assert(pCandidateBlock->GetMetaBufferAllocatedSizeBytes() >= uiMetaBytes);
                    assert(pCandidateBlock->GetTemplateBufferAllocatedSizeBytes() >= uiTemplateBytes);
                    m_pDatablock = pCandidateBlock;
                    m_pBlockPool.pop_front();
                    m_uiLowWaterMark = min(m_uiLowWaterMark, (UINT)m_pBlockPool.size());
                    pBlock = pCandidateBlock;
                    pBlock->AddRef();
                }
                pCandidateBlock->Unlock();
                CHECK_BLOCK_POOL_INVARIANTS(m_pDatablock);
            } 
        }

        Unlock();
        return pBlock;
    }    

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a destination buffer. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    OutputPort::SetDestinationBuffer(
        Datablock * pBlock
        ) 
    {
#ifdef EXTREME_DEBUG
        if(pBlock) {
            pBlock->Lock();
            BUFFERACCESSFLAGS eFlags = pBlock->GetAccessFlags();
            assert(eFlags & PT_ACCESS_ACCELERATOR_WRITE);
            pBlock->Unlock();
        }
#endif
        SetDestinationBuffer(pBlock, FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a destination buffer. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pBlock">       [in,out] If non-null, the block. </param>
    /// <param name="bAddToPool">   true to add to pool. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    OutputPort::SetDestinationBuffer(
        __in Datablock * pBlock,
        __in BOOL bAddToPool
        ) 
    {
        Lock();
        if(pBlock != m_pDatablock) {
            if(m_pDatablock) {
                // should force this block back into the pool if it 
                // is a pooled block, otherwise will queue it for GC.
                m_pDatablock->Release();
            }
            m_pDatablock = pBlock;
            if(pBlock != NULL) {
                // if we're given a block here,  set it to the destination 
                // block and add a reference to it. 
                CHECK_BLOCK_REFERENCED(pBlock);
                if(m_pDatablock)
                    m_pDatablock->AddRef();
                if(bAddToPool && m_bBlockPoolActive && PTask::Runtime::GetBlockPoolsEnabled()) {
                    pBlock->Lock();
                    pBlock->SetPooledBlock(this);
                    m_uiOwnedBlockCount++;
                    pBlock->Unlock();
                }
            } 
        }
        CHECK_BLOCK_POOL_INVARIANTS(m_pDatablock);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the destination buffer described by.  </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    OutputPort::ReleaseDestinationBuffer(
        VOID
        ) 
    {
        SetDestinationBuffer(NULL);
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
    OutputPort::PoolContainsBlock(
        Datablock * pBlock
        )
    {
        BOOL bResult = FALSE;
        UNREFERENCED_PARAMETER(pBlock);
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
#endif
        return bResult;
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
    OutputPort::AllocateBlock(
        AsyncContext * pAsyncContext,
        BOOL bPooled
        ) 
    {
        Accelerator * pAccelerator = NULL;
        if(pAsyncContext != NULL) 
            pAccelerator = pAsyncContext->GetDeviceContext();
        return AllocateBlock(pAccelerator, pAsyncContext, bPooled);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets block permissions flags for blocks allocated as "destination blocks"
    ///             on this port. </summary>
    ///
    /// <remarks>   Crossbac, 1/28/2014. </remarks>
    ///
    /// <returns>   The destination block permissions. </returns>
    ///-------------------------------------------------------------------------------------------------

    BUFFERACCESSFLAGS 
    OutputPort::GetAllocationAccessFlags(
        __in BOOL bPooledBlock
        )
    {
        std::vector<Channel*>::iterator ci;
        BUFFERACCESSFLAGS eFlags = (m_pTemplate && m_pTemplate->IsByteAddressable()) ? 
            PT_ACCESS_BYTE_ADDRESSABLE : 
            PT_ACCESS_DEFAULT;

        eFlags |= Runtime::GetDebugMode() ? PT_ACCESS_HOST_READ : 0;
        for(ci=m_vChannels.begin(); ci!=m_vChannels.end(); ci++) {
            switch((*ci)->GetType()) {
            default:  break;
            case CT_GRAPH_OUTPUT: eFlags |= (PT_ACCESS_HOST_READ | PT_ACCESS_ACCELERATOR_WRITE); break;
            case CT_INTERNAL: eFlags |= (PT_ACCESS_ACCELERATOR_READ | PT_ACCESS_ACCELERATOR_WRITE); break;
            }
        }
        if(HasDownstreamHostConsumer())
            eFlags |= (PT_ACCESS_HOST_READ | PT_ACCESS_ACCELERATOR_WRITE);
        if(IsExplicitMemorySpaceTransitionPoint())
            eFlags |= PT_ACCESS_SHARED_HINT;
        if(bPooledBlock)
            eFlags |= PT_ACCESS_POOLED_HINT;
        return eFlags;
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
    OutputPort::AllocateBlock(
        __in Accelerator *  pAccelerator,
        __in AsyncContext * pAsyncContext,
        __in BOOL           bPooled
        ) 
    {
        // We need to examine what the port is connected to to make a good decision about what views to
        // materialize. If the channel is internal, the host may never need to read, but the device
        // needs read and write access, while an output channel might require host read but no device
        // read permissions.
        // -----------------
        // We also must be able to handle the case where the ptask generates variable-length record-
        // stream like data. In such a case, the DatablockTemplate alone does not give us sufficient
        // information to make a reasonable guess. Ideally, we paramaterize the output port with user-
        // defined callback that allows us to estimate the output size based on arbitrary information
        // such as join-predicate selectivity, etc. Since our client may be a managed-language program,
        // this may be impractical. For now, we require the port to be able to provide us with a
        // maximum output size.         

        Datablock * pBlock = NULL;
        Lock();      
        
        // make sure this port is part of a well-formed graph before trying to allocate blocks for it.
        // We can't determine what access flags are needed if we don't have a channel binding. 

        if(m_vChannels.size() == 0) {

            // this is a degenerate structure!
            assert(false);
            std::string strWarning = "Cannot pre-allocate block pool for OutputPort(";
            strWarning += m_lpszVariableBinding;
            strWarning += ") on Task(";
            strWarning += m_pBoundTask->GetTaskName();
            strWarning += ") because no channels are bound to that port!";
            PTask::Runtime::Warning(strWarning);
            Unlock();
            return NULL;
        }

        BUFFERACCESSFLAGS eFlags = GetAllocationAccessFlags(bPooled);

        if(m_vChannels.size() > 0) {

            // check the graph for any matching pooled blocks. if there are none there,
            // check the global pool manager for any matching pooled blocks. If that
            // fails it means we are going to have to allocate a new Datablock the hard way.
            
            pBlock = m_pGraph->RequestPooledBlock(m_pTemplate, 0, 0, 0);            
            pBlock = pBlock ? pBlock : GlobalPoolManager::RequestBlock(m_pTemplate, 0, 0, 0);
            if(pBlock != NULL) {

                bPooled = FALSE;
                pBlock->Lock();
                assert(!m_pTemplate->HasInitialValue() || pBlock->GetPoolInitialValueValid());

            } else {
             
                if(m_pTemplate->HasInitialValue()) {

                    pBlock = Datablock::CreateDestinationBlock(pAccelerator,
                                                               NULL,
                                                               m_pTemplate,
                                                               m_pTemplate->GetDatablockByteCount(),
                                                               eFlags,
                                                               bPooled);
                    pBlock->Lock();
                    pBlock->SetPoolInitialValueValid(TRUE);

                } else {

                    pBlock = Datablock::CreateDatablock(pAsyncContext,
                                                        m_pTemplate,
                                                        NULL,
                                                        eFlags,
                                                        DBCTLC_NONE);
                
                    pBlock->Lock();
                    if(!pBlock->HasBuffers(pAccelerator))
                        pBlock->AllocateBuffers(pAccelerator, pAsyncContext, m_pTemplate->HasInitialValue());
                    if(pAsyncContext != NULL) {
                        Accelerator * pAsyncCtxAccelerator = pAsyncContext->GetDeviceContext();
                        if(pAsyncCtxAccelerator && !pBlock->HasBuffers(pAsyncCtxAccelerator)) {
                            pBlock->AllocateBuffers(pAsyncCtxAccelerator, pAsyncContext, FALSE);
                        }
                    }
                    if(pAccelerator->GetMemorySpaceId() != HOST_MEMORY_SPACE_ID && 
                        pAccelerator->SupportsPinnedHostMemory() && 
                        HasDownstreamHostConsumer()) {
                        // if some downstream host view is likely to be needed, and this 
                        // accelerator requires a special allocator get good performance
                        // for H<->D transfers, preallocate the host buffer as well.
                        pBlock->SetPreferPageLockedHostViews(m_bPageLockHostViews);                
                        pBlock->AllocateBuffers(HOST_MEMORY_SPACE_ID, 
                                                pAsyncContext, 
                                                FALSE);
                    }
                }
            }

            pBlock->Unlock();
        }
        assert(pBlock != NULL);
#ifdef DEBUG
        assert(!PoolContainsBlock(m_pDatablock));
        assert(!PoolContainsBlock(NULL));
#endif
        Unlock();
        if(bPooled && m_bBlockPoolActive && PTask::Runtime::GetBlockPoolsEnabled()) {
            pBlock->Lock();
            pBlock->SetPooledBlock(this);
            pBlock->Unlock();
        }
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate destination block. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] (optional)  If non-null, the accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    OutputPort::AllocateDestinationBlock(
        Accelerator * pAccelerator
        ) 
    {        
        BOOL bAddRefCalled = FALSE;
        Datablock * pBlock = NULL;
        Task * pTask = GetTask();
        AsyncContext * pAsyncContext = pTask->GetOperationAsyncContext(pAccelerator, ASYNCCTXT_XFERHTOD);
        if(pAsyncContext != NULL) {
            pBlock = AllocateBlock(pAsyncContext, TRUE);
            if(pBlock != NULL) {
#ifdef DEBUG
                pBlock->Lock();
                assert(pBlock->RefCount() == 0);
                pBlock->Unlock();
#endif
                pBlock->AddRef();
                bAddRefCalled = TRUE;
            }
        } 
        
        if(pBlock == NULL) {
            pBlock = AllocateBlock(pAccelerator, pAsyncContext, TRUE);
#ifdef DEBUG
            pBlock->Lock();
            assert(pBlock->RefCount() == 0);
            pBlock->Unlock();
#endif
            pBlock->AddRef();
            bAddRefCalled = TRUE;
        }
        assert(pBlock != NULL);
        Lock();
        SetDestinationBuffer(pBlock);
        if(bAddRefCalled) {
#ifdef DEBUG
            pBlock->Lock();
            assert(pBlock->RefCount() > 1);
            pBlock->Unlock();
#endif
            pBlock->Release(); // SetDestinationBuffer will addref this block too
        }
#ifdef DEBUG
        assert(!PoolContainsBlock(m_pDatablock));
        assert(!PoolContainsBlock(NULL));
#endif
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate block. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator">     [in,out] (optional)  If non-null, the async context. </param>
    /// <param name="uiDataBytes">      The data in bytes. </param>
    /// <param name="uiMetaBytes">      The meta in bytes. </param>
    /// <param name="uiTemplateBytes">  The template in bytes. </param>
    /// <param name="bPooled">          true to pooled. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *  
    OutputPort::AllocateBlock(
        __in Accelerator * pAccelerator,
        __in UINT uiDataBytes,
        __in UINT uiMetaBytes,
        __in UINT uiTemplateBytes,
        __in BOOL bPooled
        ) 
    {
        // We need to examine what the port is connected to to make a good decision about what views to
        // materialize. If the channel is internal, the host may never need to read, but the device
        // needs read and write access, while an output channel might require host read but no device
        // read permissions.
        // -----------------
        // We also must be able to handle the case where the ptask generates variable-length record-
        // stream like data. In such a case, the DatablockTemplate alone does not give us sufficient
        // information to make a reasonable guess. Ideally, we paramaterize the output port with user-
        // defined callback that allows us to estimate the output size based on arbitrary information
        // such as join-predicate selectivity, etc. Since our client may be a managed-language program,
        // this may be impractical. For now, we require the port to be able to provide us with a
        // maximum output size. 
        
        Datablock * pBlock = NULL;
        Lock();      

        // make sure this port is part of a well-formed graph before trying to allocate blocks for it.
        // We can't determine what access flags are needed if we don't have a channel binding. 

        if(m_vChannels.size() == 0) {

            // this is a degenerate structure!
            assert(false);
            std::string strWarning = "Cannot pre-allocate block pool for OutputPort(";
            strWarning += m_lpszVariableBinding;
            strWarning += ") on Task(";
            strWarning += m_pBoundTask->GetTaskName();
            strWarning += ") because no channels are bound to that port!";
            PTask::Runtime::Warning(strWarning);
            Unlock();
            return NULL;
        }

        DatablockTemplate * pTemplate = GetTemplate();
        BUFFERACCESSFLAGS eFlags = pTemplate->IsByteAddressable() ? PT_ACCESS_BYTE_ADDRESSABLE : 0;
        if(m_vChannels.size() > 0) {
            std::vector<Channel*>::iterator ci;
            for(ci=m_vChannels.begin(); ci!=m_vChannels.end(); ci++) {
                Channel * pOutboundChannel = (*ci);
                switch(pOutboundChannel->GetType()) {
                case CT_GRAPH_INPUT: 
                    assert(false);
                    break;
                case CT_GRAPH_OUTPUT:
                    eFlags |= (PT_ACCESS_HOST_READ | PT_ACCESS_ACCELERATOR_WRITE);
                    break;
                case CT_INTERNAL:
                    eFlags |= (PT_ACCESS_ACCELERATOR_READ | PT_ACCESS_ACCELERATOR_WRITE);
                    if(PTask::Runtime::GetDebugMode()) {
                        // in debug mode we materialize a host view
                        // on internal channel so that a debugger can
                        // have visibility for intermediate data in a graph. 
                        // Consequently, we need host read permissions.
                        eFlags |= PT_ACCESS_HOST_READ;
                    }
                    break;
                }
            }
            AsyncContext * pAsyncContext = pAccelerator->GetAsyncContext(ASYNCCTXT_XFERHTOD);
            pBlock = Datablock::CreateDatablock(pAsyncContext,
                                                m_pTemplate,
                                                uiDataBytes,
                                                uiMetaBytes,
                                                uiTemplateBytes,
                                                eFlags,
                                                DBCTLC_NONE,
                                                NULL,
                                                0,
                                                FALSE);

            if(!bPooled) 
                pBlock->AddRef();

            pBlock->Lock();
            if(!pBlock->HasBuffers(pAccelerator))
                pBlock->AllocateBuffers(pAccelerator, pAsyncContext, FALSE);
            if(pAsyncContext != NULL) {
                Accelerator * pAsyncCtxAccelerator = pAsyncContext->GetDeviceContext();
                if(pAsyncCtxAccelerator && !pBlock->HasBuffers(pAsyncCtxAccelerator)) {
                    pBlock->AllocateBuffers(pAsyncCtxAccelerator, pAsyncContext, FALSE);
                }
            }
            if(pAccelerator->GetMemorySpaceId() != HOST_MEMORY_SPACE_ID && 
                pAccelerator->SupportsPinnedHostMemory() && 
                HasDownstreamHostConsumer()) {
                // if some downstream host view is likely to be needed, and this 
                // accelerator requires a special allocator get good performance
                // for H<->D transfers, preallocate the host buffer as well.                
                pBlock->AllocateBuffers(HOST_MEMORY_SPACE_ID, 
                                        pAsyncContext, 
                                        FALSE);
            }
            pBlock->Unlock();
        }
        assert(pBlock != NULL);
        Unlock();
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an in out producer. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    OutputPort::SetInOutProducer(
        Port * pPort
        ) 
    {
        Lock();
        m_pInOutProducer = pPort;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' has output channel. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if output channel, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    OutputPort::HasOutputChannel(
        VOID
        ) 
    {
        BOOL bResult = FALSE;
        Lock();
        for(UINT i=0; i<GetChannelCount(); i++) {
            Channel * pChannel = GetChannel(i);
            if(pChannel->GetType() == CT_GRAPH_OUTPUT) {
                bResult = TRUE;
                break;
            }
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has block pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if block pool, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    OutputPort::HasBlockPool(
        VOID
        )
    {
        if(HasAllocatorInputPort() && !m_bPoolHintsSet) {
            
            // if this block has an upstream allocator and the programmer has provided no pool hints, we
            // can't make assumptions about block size because the blocks on this port have variable size.
            // In short, pooling can't work here, so return FALSE because we did not allocate a pool on
            // this port. 
            return FALSE; 
        }

        if(GetInOutProducer() != NULL) {

            // If this port is the out part of an in/out pair, blocks on this port will always be allocated
            // upstream. We could build a block pool but since blocks will never be allocated by this port.
            // Outgoing blocks on this port always come from an upstream port, so a block pool on this port
            // is completely wasted. 
            return FALSE; 
        }    

        return TRUE;    
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds to the pool. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    OutputPort::AddNewBlock(
        Datablock * pBlock
        )
    {
        assert(PTask::Runtime::GetBlockPoolsEnabled());
        assert(pBlock != NULL);
        pBlock->Lock();
        pBlock->SetPooledBlock(this);
        pBlock->ClearAllControlSignals();
        pBlock->Unlock();
        Lock();
#ifdef DEBUG
        assert(!PoolContainsBlock(pBlock));
#endif
        m_pBlockPool.push_back(pBlock);
        m_uiPoolHighWaterMark = max(m_uiPoolHighWaterMark, (UINT)m_pBlockPool.size());
        m_uiOwnedBlockCount++;

#ifdef DEBUG
//        assert(!PoolContainsBlock(m_pDatablock));
        assert(!PoolContainsBlock(NULL));
#endif
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
    OutputPort::ReturnToPool(
        Datablock * pBlock
        )
    {
        assert(pBlock != NULL);
        pBlock->Lock();
        assert(pBlock->RefCount() == 0);
        pBlock->ClearAllControlSignals();
        pBlock->SetApplicationContext(NULL);
        pBlock->Unlock();
        Lock();
        if(m_bBlockPoolActive && PTask::Runtime::GetBlockPoolsEnabled()) {
            #ifdef DEBUG
                assert(!PoolContainsBlock(pBlock));
            #endif
            m_pBlockPool.push_back(pBlock);
            m_vDirtySet.insert(pBlock);
            m_uiPoolHighWaterMark = max(m_uiPoolHighWaterMark, (UINT)m_pBlockPool.size());

            #ifdef DEBUG
            // assert(!PoolContainsBlock(m_pDatablock));
            assert(!PoolContainsBlock(NULL));
            #endif
        } else {
            GarbageCollector::QueueForGC(pBlock);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   gets the pool size. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    UINT
    OutputPort::GetPoolSize(
        VOID
        )
    {        
        UINT uiDefaultSize = PTask::Runtime::GetICBlockPoolSize();
        UINT uiActualPoolSize = (m_nPoolHintPoolSize > 0) ? m_nPoolHintPoolSize : uiDefaultSize;
        return uiActualPoolSize;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the pooled blocks described by.  </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    OutputPort::ReleasePooledBlocks(
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
    /// <summary>   Raises a Gate signal. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    OutputPort::SignalGate(
        VOID
        )
    {
        Lock();
        m_bPortOpen = !this->m_bInitialPortStateOpen;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a control port. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="p">                [in,out] If non-null, the Datablock* to push. </param>
    /// <param name="bInitiallyOpen">   (optional) the initially open. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    OutputPort::SetControlPort(
        Port * p,
        BOOL bInitiallyOpen
        ) 
    { 
        Lock();
        m_pControlPort = p; 
        m_bPortOpen = bInitiallyOpen;
        m_bInitialPortStateOpen = bInitiallyOpen;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a control port. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the control port. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port *          
    OutputPort::GetControlPort(
        VOID    
        ) 
    { 
        return m_pControlPort; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' has control port. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if control port, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    OutputPort::HasControlPort(
        VOID
        ) 
    { 
        return m_pControlPort != NULL; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets an in out producer. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the in out producer. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port *          
    OutputPort::GetInOutProducer(
        VOID
        ) 
    { 
        return m_pInOutProducer; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an allocator port. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="p">    [in,out] If non-null, the Datablock* to push. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    OutputPort::SetAllocatorPort(
        Port * p
        ) 
    { 
        Lock();
        m_pAllocatorPort = p; 
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the allocator port. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the allocator port. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port *          
    OutputPort::GetAllocatorPort(
        VOID
        ) 
    { 
        return m_pAllocatorPort; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' has an allocator input port. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if allocator input port, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    OutputPort::HasAllocatorInputPort(
        VOID
        ) 
    { 
        return m_pAllocatorPort != NULL; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has downstream host consumer. </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///
    /// <param name="vVisitSet">    [in,out] If non-null, set the visit belongs to. </param>
    ///
    /// <returns>   true if downstream host consumer, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    OutputPort::HasDownstreamHostConsumer(
        std::set<OutputPort*> &vVisitSet
        )
    {
        // return the number of downstream ports/channels
        // of this port can that modify the contents
        // of this datablock making it potentiall unsafe
        // to readshare it by pushing the same
        // block into multiple downstream channels.
        BOOL bResult = FALSE;
        if(vVisitSet.find(this) != vVisitSet.end())
            return bResult;

        set<OutputPort*> vRecurseSet;
        vVisitSet.insert(this);

        Lock();
        for(UINT i=0; i<GetChannelCount(); i++) {
            Channel * pChannel = GetChannel(i);
            if((pChannel->GetType() == CT_GRAPH_OUTPUT)) {
                bResult = TRUE;
                break;
            }
            Port * pPort = pChannel->GetBoundPort(CE_DST);
            assert(pPort != NULL); // should only happen on outputs!
            if(pPort == NULL) continue;
            Task * pTask = pPort->GetTask();
            if(pTask->GetAcceleratorClass() == ACCELERATOR_CLASS_HOST) {
                if(!pPort->HasDependentAcceleratorBinding()) {
                    bResult = TRUE;
                    break;
                }
            } else {
                if(pPort->GetPortType() == INPUT_PORT) {
                    InputPort * pIPort = reinterpret_cast<InputPort*>(pPort);
                    OutputPort * pOPort = reinterpret_cast<OutputPort*>(pIPort->GetInOutConsumer());
                    if(pOPort != NULL && pOPort != this) 
                        vRecurseSet.insert(pOPort);
                }
            }
        }
        Unlock();
        if(!bResult) {
            set<OutputPort*>::iterator si;
            for(si=vRecurseSet.begin(); si!=vRecurseSet.end(); si++) {
                if((*si)->HasDownstreamHostConsumer(vVisitSet)) {
                    bResult = TRUE;
                    break;
                }
            }
        }
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has downstream host consumer. </summary>
    ///
    /// <remarks>   Crossbac, 7/12/2012. </remarks>
    ///
    /// <returns>   true if downstream host consumer, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    OutputPort::HasDownstreamHostConsumer(
        VOID
        )
    {
        std::set<OutputPort*> vVisitSet;
        return HasDownstreamHostConsumer(vVisitSet);
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' has downstream writer ports. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if downstream writer ports, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    OutputPort::HasDownstreamWriterPorts(
        VOID
        ) 
    {
        // return true if anything downstream 
        // of this port can modify the contents
        // of this datablock making it potentiall unsafe
        // to readshare it by pushing the same
        // block into multiple downstream channels.
        return GetDownstreamWriterPortCount(NULL) > 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' has downstream readonly ports. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if downstream readonly ports, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    OutputPort::HasDownstreamReadonlyPorts(
        VOID
        )
    {
        // return true if anything downstream 
        // of this port consumes the block read-only
        return GetDownstreamReadonlyPortCount(NULL) > 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the downstream writer port count. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, a block that would be pushed into the
    /// 						channel. This allows us to check whether we are dealing with
    /// 						a predicated channel that would release the block. </param>      
    /// 						
    /// <returns>   The downstream writer port count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT            
    OutputPort::GetDownstreamWriterPortCount(
        Datablock * pBlock
        ) 
    {
        // return the number of downstream ports/channels
        // of this port can that modify the contents
        // of this datablock making it potentiall unsafe
        // to readshare it by pushing the same
        // block into multiple downstream channels.
        Lock();
        UINT nResult = 0;
        for(UINT i=0; i<GetChannelCount(); i++) {
            Channel * pChannel = GetChannel(i);
            if(pBlock && !pChannel->PassesPredicate(CE_SRC, pBlock))
                continue;
            if((pChannel->GetType() == CT_GRAPH_OUTPUT)) {
                if(pChannel->GetPredicationType(CE_SRC) == PTask::CGATEFN_DEVNULL) {
                    // if this a dev-null channel, there can be no downstream writer.
                    continue;
                }
                if(pBlock == NULL || pChannel->PassesPredicate(CE_SRC, pBlock)) {
                    // if we don't have a block to test, we have assume there is a writer.
                    // if the predicate passes, there is a potential writer (the user)
                    // if the predicate fails there is no possibility of a reader or writer!
                    nResult++;
                }
                continue;
            }
            Port * pPort = pChannel->GetBoundPort(CE_DST);
            if(pPort == NULL) continue;
            if(pPort->IsInOutParameter() || pPort->IsDestructive()) {
                nResult++;
                continue;
            }
        }
        Unlock();
        return nResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the propagated control code. We override this in output port
    ///             because an output port may also be a scope terminus, which means
    ///             that propagated control signals need to include any that are also
    ///             mandated by that role. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The propagated control code. </returns>
    ///-------------------------------------------------------------------------------------------------

    CONTROLSIGNAL            
    OutputPort::GetPropagatedControlSignals(
        VOID
        )
    {
        assert(LockIsHeld());
        CONTROLSIGNAL luiResult = m_luiPropagatedControlCode; 
        if(IsScopeTerminus()) {
            luiResult |= m_luiScopeTerminalSignal;
        }
        return luiResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the downstream readonly port count. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, a block that would be pushed into the
    /// 						channel. This allows us to check whether we are dealing with
    /// 						a predicated channel that would release the block. </param>      
    /// 						
    /// <returns>   The downstream readonly port count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    OutputPort::GetDownstreamReadonlyPortCount(
        Datablock * pBlock
        )
    {
        // return true if anything downstream 
        // of this port consumes the block read-only
        Lock();
        UINT nResult = 0;
        for(UINT i=0; i<GetChannelCount(); i++) {
            Channel * pChannel = GetChannel(i);
            if((pChannel->GetType() == CT_GRAPH_OUTPUT)) {
                continue;
            }
            Port * pPort = pChannel->GetBoundPort(CE_DST);
            if(pPort == NULL) continue;
            if(pPort->IsInOutParameter() || pPort->IsDestructive()) {
                continue;
            }
            if(pBlock && !pChannel->PassesPredicate(CE_SRC, pBlock))
                continue;
            // if there is a channel that is not connected
            // to either an inout port or is an output, and the block
            // will pass predication (or we have no block and must conservatively assume
            // it will pass), then we have a readonly consumer. 
            nResult++;
        }
        Unlock();
        return nResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the suppress clones property, which allows a user to suppress output cloning
    ///             for blocks on ports with multiple (R/W conflicting) downstream consumers, if the
    ///             programmer happens to know something about the structure of the graph that the
    ///             runtime cannot (or does not detect) and that makes it safe to do so.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 2/29/2012. </remarks>
    ///
    /// <param name="bSuppressClones">  true to suppress clones. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    OutputPort::SetSuppressClones(
        BOOL bSuppressClones
        )
    {
        Lock();
        m_bSuppressClones = bSuppressClones;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the suppress clones property, which allows a user to suppress output cloning
    ///             for blocks on ports with multiple (R/W conflicting) downstream consumers, if the
    ///             programmer happens to know something about the structure of the graph that the
    ///             runtime cannot (or does not detect) and that makes it safe to do so.  Note that
    ///             we do not require a lock to query this property because it is assumed this method
    ///             is used only during graph construction and is not used while a graph is running.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 2/29/2012. </remarks>
    ///
    /// <returns>   the value of the suppress clones property. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    OutputPort::GetSuppressClones(
        VOID
        )
    {
        return m_bSuppressClones;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets request page locked. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="bPageLocked">  true to lock, false to unlock the page. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    OutputPort::SetRequestsPageLocked(
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
    OutputPort::GetRequestsPageLocked(
        VOID
        )
    {
        return m_bPageLockHostViews;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is block pool candidate. </summary>
    ///
    /// <remarks>   crossbac, 4/30/2013. </remarks>
    ///
    /// <returns>   true if block pool candidate, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    OutputPort::IsBlockPoolCandidate(
        VOID
        )
    {
        if(HasAllocatorInputPort() && !m_bPoolHintsSet) {
            
            // if this block has an upstream allocator and the programmer has provided no pool hints, we
            // can't make assumptions about block size because the blocks on this port have variable size.
            // In short, pooling can't work here, so return FALSE because we did not allocate a pool on
            // this port. 
            return FALSE; 
        }

        if(GetInOutProducer() != NULL) {

            // If this port is the out part of an in/out pair, blocks on this port will always be allocated
            // upstream. We could build a block pool but since blocks will never be allocated by this port.
            // Outgoing blocks on this port always come from an upstream port, so a block pool on this port
            // is completely wasted. 
            return FALSE; 
        }

        if(m_pBoundTask->GetAcceleratorClass() != ACCELERATOR_CLASS_HOST || HasDependentAcceleratorBinding()) {
            return TRUE;
        }

        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Searches for the maximal downstream path. </summary>
    ///
    /// <remarks>   crossbac, 6/21/2013. </remarks>
    ///
    /// <param name="vTasksVisited">    [in,out] [in,out] If non-null, the tasks visited. </param>
    /// <param name="vPath">            [in,out] [in,out] If non-null, full pathname of the file. </param>
    ///
    /// <returns>   The found maximal downstream capacity. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    OutputPort::FindMaximalDownstreamCapacity(
        __inout std::set<Task*>& vTasksVisited,
        __inout std::vector<Channel*>& vPath
        )
    {
        UINT uiMaxCapacity = 0;
        Channel * pMaximalChannel = NULL;
        std::vector<Channel*>::iterator vi;
        std::vector<Channel*> vMaximalOutboundPath;

        // first we need to check for complex cycles. if the task
        // to which this port is bound is already on the path, terminate
        if(vTasksVisited.find(m_pBoundTask) != vTasksVisited.end())
            return 0;

        vTasksVisited.insert(m_pBoundTask);
        for(vi=m_vChannels.begin(); vi!=m_vChannels.end(); vi++) {

            // traverse all the channels recursively visiting outbound in/out pairs to find the maximal
            // channel capacity exiting from this port. Ideally we provision block pools such that we can
            // always meet this capacity without allocating blocks dynamically. 

            Channel * pOutboundChannel = *vi;
            UINT uiChannelCapacity = pOutboundChannel->GetCapacity();
            InputPort * pDownstreamPort = reinterpret_cast<InputPort*>(pOutboundChannel->GetBoundPort(CE_DST));

            if(pDownstreamPort != NULL && pDownstreamPort->GetInOutConsumer()) {

                // we have an outbound channel connected to an in/out consumer pair. 
                // the downstream capacity on this channel is the channel capacity plus
                // the maximal downstream capacity of the out party in the inout pair.
                // watch out for cycles. The downstream consumer can this task, so
                // only recurse on the out member of the pair if it is not "this".

                std::vector<Channel*> vOutboundPath;
                OutputPort * pDSOutPort = reinterpret_cast<OutputPort*>(pDownstreamPort->GetInOutConsumer());
                if(pDSOutPort == this) {

                    // this channel connects to an in/out pair, but the pair involves this output port and the
                    // current channel is a cycle. We assert FALSE here because this method is used to configure
                    // block pools for output ports, and we should never build a block pool for an output port that
                    // has an inout producer because no blocks will ever be allocated at the output port. For
                    // form's sake, we subsequently include the cycle's capacity in the computation so we truly can
                    // return the maximal downstream capacity. 

                    // assert(pDSOutPort != this);
                    if(uiChannelCapacity > uiMaxCapacity) {
                        uiMaxCapacity = uiChannelCapacity;
                        pMaximalChannel = pOutboundChannel;
                        vMaximalOutboundPath.clear();
                    }

                } else {
                    UINT uiDownstreamCapacity = pDSOutPort->FindMaximalDownstreamCapacity(vTasksVisited, vOutboundPath);
                    if(uiChannelCapacity + uiDownstreamCapacity > uiMaxCapacity) {
                        vMaximalOutboundPath.clear();
                        uiMaxCapacity = uiChannelCapacity + uiDownstreamCapacity;
                        vMaximalOutboundPath.assign(vOutboundPath.begin(), vOutboundPath.end());
                        pMaximalChannel = pOutboundChannel;
                    }
                }

            } else {

                // this is a simple output channel: either it is connected to an input port with no in/out
                // semantics or it is an exposed output channel. Regardless, the maximal contribution of this
                // channel to the outbound capacity is the capacity of the current channel because the will be
                // released by any consumers. 
                
                if(uiChannelCapacity > uiMaxCapacity) {
                    uiMaxCapacity = uiChannelCapacity;
                    pMaximalChannel = pOutboundChannel;
                    vMaximalOutboundPath.clear();
                }
            }
        }

        if(pMaximalChannel) {
            vPath.push_back(pMaximalChannel);           
            for(vi=vMaximalOutboundPath.begin(); vi!=vMaximalOutboundPath.end(); vi++)
                vPath.push_back(*vi);
        }
        return uiMaxCapacity;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this port is an explicit memory space transition point. 
    ///             We return true only when we know for certain that this task 
    ///             executes on one GPU and at least one downstream tasks definitely
    ///             needs a view of our outputs on another GPU. In general we can only
    ///             tell this with high precision when there is task affinity involved.
    ///             We use this to set the sharing hint on the access flags for blocks
    ///             allocated, which in turn allows some back ends to better optimize GPU-side
    ///             buffer allocation and data transfer. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/13/2014. </remarks>
    ///
    /// <returns>   true if explicit memory space transition point, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    OutputPort::IsExplicitMemorySpaceTransitionPoint(
        VOID
        )
    {
        Lock();
        if(!m_bExplicitMemSpaceTransitionPointSet) {
            m_bExplicitMemSpaceTransitionPoint = FALSE;
            Accelerator * pAffinitizedAccelerator = m_pBoundTask->GetAffinitizedAcceleratorHint();
            if(pAffinitizedAccelerator != NULL)  {
                std::vector<Channel*>::iterator chi;
                for(chi=m_vChannels.begin(); chi!=m_vChannels.end(); chi++) {
                    Channel * pChannel = *chi;
                    InputPort * pIPort = reinterpret_cast<InputPort*>(pChannel->GetBoundPort(CE_DST));
                    Task * pDownstreamTask = pIPort ? pIPort->GetTask() : NULL;
                    if(pDownstreamTask!=NULL) {
                        Accelerator * pDSAffinitizedAccelerator = pDownstreamTask->GetAffinitizedAcceleratorHint();
                        if(pDSAffinitizedAccelerator != NULL && 
                           pDSAffinitizedAccelerator != pAffinitizedAccelerator &&
                           (pDSAffinitizedAccelerator->GetClass() == ACCELERATOR_CLASS_DIRECT_X ||
                            pAffinitizedAccelerator->GetClass() == ACCELERATOR_CLASS_HOST)) {
                            m_bExplicitMemSpaceTransitionPoint = TRUE;
                            break;
                        }
                    }
                }
            }
            m_bExplicitMemSpaceTransitionPointSet = TRUE;
        }
        Unlock();
        return m_bExplicitMemSpaceTransitionPoint;
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
    OutputPort::AllocateBlockPool(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )
    {
        if(!IsBlockPoolCandidate())
            return FALSE;

        if(!PTask::Runtime::GetBlockPoolsEnabled()) {
            PTask::Runtime::Inform("%s::%s: PTask block pools disabled--skipping allocation for pool on: %s\n",
                                   __FILE__,
                                   __FUNCTION__,
                                   m_lpszVariableBinding);
            return FALSE;
        }

        // this is a port for which we can actually build a usable pool. go through all the
        // accelerators which might require a view of this block and allocate platform-specific
        // (PBuffers) for the block for each accelerator.
        // 
        // FIXME: TODO:
        // ------------
        // If this port is the upstream producer on an internal channel, we can also go look at what's
        // downstream to figure out if there is any need to allocate additional buffers. For example,
        // if this is a DX task and the downstream consumer is a host-task with a dependent
        // accelerators on the destination port, we know that we are going to have to migrate any views
        // of the block through host memory (so we might as well preallocate there too).
        // 
        // Additionally, we want to dynamically adjust to the potential channel occupancy implications
        // of the downstream graphs structures. More concretely, if this port's outbound channel is
        // attached to tasks with in/out semantics, the blocks allocated here may traverse multiple
        // hops before being released, which complicates the task of provisioning to avoid allocation
        // on the critical path. Traverse the downstream graph, find the maximum downstream path and
        // channel capacities on that path, and provision such that we can serve blocks from the block
        // pool until all channels on the maximal path are full. 
        
        BOOL bPoolConstructSuccess = TRUE;
        BOOL bUseHintDimensions = m_bPoolHintsSet && m_pAllocatorPort != NULL;
        UINT uiDefaultSize = bUseHintDimensions ? m_nPoolHintPoolSize : PTask::Runtime::GetICBlockPoolSize();
        UINT uiActualPoolSize = (uiPoolSize > 0) ? uiPoolSize : uiDefaultSize;
        
        if(Runtime::GetProvisionBlockPoolsForCapacity()) {

            std::vector<Channel*> vPath;
            std::set<Task*> vTasksVisited;
            UINT uiMaximalDownstreamCapacity = FindMaximalDownstreamCapacity(vTasksVisited, vPath);
            if(uiMaximalDownstreamCapacity > uiActualPoolSize) {
                PTask::Runtime::MandatoryInform("Growing block pool on %s from %d to %d based on downstream capacity!\n",
                                                m_lpszVariableBinding,
                                                uiActualPoolSize,
                                                uiMaximalDownstreamCapacity);
                uiActualPoolSize = uiMaximalDownstreamCapacity;
            }
        }
        
        BOOL bTransitionPoint = IsExplicitMemorySpaceTransitionPoint();
        BOOL bPopulateViews = m_pTemplate->HasInitialValue();  // dont' populate any buffers unnecessarily!!!

        for(UINT i=0; i<uiActualPoolSize; i++) {

            Datablock * pDestBlock = NULL;
            std::vector<Accelerator*>::iterator vi;
            for(vi=pAccelerators->begin(); 
                vi!=pAccelerators->end() && bPoolConstructSuccess; 
                vi++) {
            
                // check to see that this acclerator actually
                // needs a buffer created, in light of task affinity 
                // and graph structure--if this task has explicit
                // affinity, we only want buffers on the affinitized acc.
                Accelerator * pAccelerator = *vi;
                AsyncContext * pAsyncContext = NULL;
                if(!IsBlockPoolViewAccelerator(pAccelerator))
                    continue;



                pAsyncContext = m_pBoundTask->GetOperationAsyncContext(pAccelerator, ASYNCCTXT_XFERHTOD);

                if(!pDestBlock && bPopulateViews && !bUseHintDimensions) {

                    pDestBlock = Datablock::CreateDestinationBlock(pAccelerator,
                                                                   NULL,
                                                                   m_pTemplate,
                                                                   m_pTemplate->GetDatablockByteCount(),
                                                                   GetAllocationAccessFlags(TRUE),
                                                                   TRUE);
                    pDestBlock->Lock();
                    pDestBlock->SetPoolInitialValueValid(TRUE);
                }

                if(!pDestBlock) {

                    // we don't have a block yet. Allocate it based on the template if we
                    // have no upstream allocator port, and based on the hint dimensions
                    // if there is an upstream allocator and the programmer has forced block pooling.
                    pDestBlock = bUseHintDimensions ? 
                        AllocateBlock(pAccelerator, 
                                      m_nPoolHintDataBytes, 
                                      m_nPoolHintMetaBytes, 
                                      m_nPoolHintTemplateBytes,
                                      TRUE) :
                        AllocateBlock(pAccelerator, pAsyncContext, TRUE);
                    assert(pDestBlock != NULL);
                    if(pDestBlock == NULL) {
                        // if we've eaten all the memory, and the assert above hasn't helped 
                        // us, it means this is a release build and we've got to try to recover.
                        // in that case, just keep plowing going and be sure to return false
                        PTask::Runtime::Warning("block pool pre-allocation failed...recovering...");
                        bPoolConstructSuccess = FALSE;
                        break;
                    }
                    pDestBlock->Lock();
                }

                if(pAccelerator->GetMemorySpaceId() != HOST_MEMORY_SPACE_ID) {

                    if(!pDestBlock->HasBuffers(pAccelerator))
                        pDestBlock->AllocateBuffers(pAccelerator, pAsyncContext, bPopulateViews);
                    if((pAccelerator->SupportsPinnedHostMemory() && HasDownstreamHostConsumer()) || bTransitionPoint) {
                        // if some downstream host view is likely to be needed, and this 
                        // accelerator requires a special allocator get good performance
                        // for H<->D transfers, preallocate the host buffer as well.
                        if(!pDestBlock->HasBuffers(HOST_MEMORY_SPACE_ID)) {
                            pDestBlock->AllocateBuffers(HOST_MEMORY_SPACE_ID, 
                                                        pAsyncContext, 
                                                        bPopulateViews); 
                        }
                    }
                } 
            }

            if(pDestBlock != NULL) {

                if(bPopulateViews) {
                    // make sure coherence state is right for all
                    // buffers with initial values configured.
                    pDestBlock->ForceExistingViewsValid();
                }
                pDestBlock->Unlock();
                AddNewBlock(pDestBlock);                
            }
        }

        m_bBlockPoolActive = bPoolConstructSuccess;
        m_uiLowWaterMark = (UINT) m_pBlockPool.size();
        return bPoolConstructSuccess;
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
    OutputPort::AllocateBlockPoolAsync(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )
    {
        return AllocateBlockPool(pAccelerators, uiPoolSize);
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
    OutputPort::FinalizeBlockPoolAsync(
        VOID
        )
    {
        m_uiLowWaterMark = static_cast<UINT>(m_pBlockPool.size());
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check type-specific semantics. Return true if all the structures are initialized
    ///             for this port in a way that is consistent with a well-formed graph. Called by
    ///             CheckSemantics()
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="pos">      [in,out] output string stream. </param>
    /// <param name="pGraph">   [in,out] non-null, the graph. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    OutputPort::CheckTypeSpecificSemantics(
        std::ostream * pos,
        PTask::Graph * pGraph
        )
    {
        UNREFERENCED_PARAMETER(pGraph);
        BOOL bResult = TRUE;
        std::ostream& os = *pos;
        if(m_vChannels.size() == 0) {
            bResult = FALSE;
            os << this 
                << "not bound to any output channels. "
                << "It will be tough to retrieve your output this way!"
                << endl;
        }
        vector<Channel*>::iterator vi;
        for(vi=m_vChannels.begin(); vi!=m_vChannels.end(); vi++) {
            // this port had better be attached to an output
            // channel or an internal channel.
            Channel * pChannel = *vi;
            if(pChannel->GetType() == CT_GRAPH_INPUT) {
                bResult = FALSE;
                os << this << "bound to an input channel!" << endl;
            }
            CHANNELPREDICATE ePredicate = pChannel->GetPredicationType(CE_SRC);
            if(ePredicate != CGATEFN_NONE && ePredicate != CGATEFN_DEVNULL) {
                // This output port is connected to a predicated channel.
                // All canonical predication types require control information
                // on the data block, which means this port had better have
                // a control propagation source. If the predicate type is set
                // to user-defined, warn the user. 
                if(m_pControlPropagationSource == NULL) {
                    if(pChannel->GetControlPropagationSource() == NULL) {
                    switch(ePredicate) {
                        case CGATEFN_USER_DEFINED:
                            os << this << " is connected to "
                                << pChannel << " which is configured with user-defined predication " 
                                << "but has no control propagation source. User-defined predicates "
                                << "do not necessarily require control information, does yours?"
                                << endl;
                            break;
                        default:
                            bResult = FALSE;
                            os << this << " is connected to "
                                << pChannel << " which is configured with channel predication type " 
                                << ePredicate
                                << ", but has no control propagation source. The predicate will "
                                << "never change state. Is this your intent?"
                                << endl;
                            break;
                        }
                    }
                }
            }
        }
        // check for places in the graph where the structure risks incurring
        // datablock cloning overheads, and warn the user about it. 
        UINT nReaders = GetDownstreamReadonlyPortCount(NULL);
        UINT nWriters = GetDownstreamWriterPortCount(NULL);
        BOOL bRequiresClone = !m_bSuppressClones && ((nReaders && nWriters) || (nWriters > 1));
        if(bRequiresClone) {
            os  << this 
                << " has " << nReaders 
                << " downstream readers and " << nWriters
                << " downstream writers. This structure can force the runtime to clone " << std::endl
                << "datablocks, depending on predication state of the connecting channels. " << std::endl
                << "Is this structure intentional?" << std::endl;
        }

        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Force block pooling for a port that has an up-stream allocator. 
    /// 			In general, when we have an upstream allocator (meta) port, the runtime
    /// 			will not create a block pool for the corresponding output port. This
    /// 			turns out to put device-side allocation on the critical path in some
    /// 			cases, so we provide a way to override that behavior and allow a port
    /// 			to create a pool based on some size hints. When there is a block
    /// 			available with sufficient space in the pool, the meta port can avoid
    /// 			the allocation and draw from the pool. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 9/25/2012. </remarks>
    ///
    /// <param name="nPoolSize">        Size of the block pool. </param>
    /// <param name="nStride">          The stride. </param>
    /// <param name="nDataBytes">       The data in bytes. </param>
    /// <param name="nMetaBytes">       The meta in bytes. </param>
    /// <param name="nTemplateBytes">   The template in bytes. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    OutputPort::ForceBlockPoolHint(
        __in UINT nPoolSize,
        __in UINT nStride,
        __in UINT nDataBytes,
        __in UINT nMetaBytes,
        __in UINT nTemplateBytes,
        __in BOOL bPageLockHostViews,
        __in BOOL bEagerDeviceMaterialize
        )
    {
        UNREFERENCED_PARAMETER(bPageLockHostViews);
        UNREFERENCED_PARAMETER(bEagerDeviceMaterialize);
        Lock();
        m_bPoolHintsSet = TRUE;
        m_nPoolHintPoolSize = nPoolSize;
        m_nPoolHintStride = nStride;
        m_nPoolHintDataBytes = nDataBytes;
        m_nPoolHintMetaBytes = nMetaBytes;
        m_nPoolHintTemplateBytes = nTemplateBytes;   
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind descriptor port. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    /// <param name="func">     The func. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    OutputPort::BindDescriptorPort(
        Port * pPort,
        DESCRIPTORFUNC func
        ) 
    {
        Lock();
        DEFERREDPORTDESC * desc = new DEFERREDPORTDESC();
        desc->func = func;
        desc->pPort = pPort;
        m_vDescriptorPorts.push_back(desc);
        Unlock();
        OutputPort * pOPort = reinterpret_cast<OutputPort*>(pPort);
        pOPort->BindDescribedPort(this, func);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind described port. </summary>
    ///
    /// <remarks>   Crossbac, 2/14/2013. </remarks>
    ///
    /// <param name="eFunc">    [in,out] If non-null, the function. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    OutputPort::BindDescribedPort(
        __in Port * pPort,
        __in DESCRIPTORFUNC eFunc
        )
    {
        // first, force the template dims in the described port's 
        // meta/template channel to match the template dims for
        // the data channel of the describing port. If we don't do this
        // things go wrong at allocation time because the buffer allocator
        // typically wants to look at the template for the parent block,
        // which comes from the described port. The simplest way to
        // work around this is force a match. This implementation is a bit
        // hacky: we should formalize this. 
        DatablockTemplate * pDescribedPortTemplate = pPort->GetTemplate();
        DatablockTemplate * pDescriberTemplate = GetTemplate();
        UINT uiSrcChannelIdx = DBDATA_IDX;
        UINT uiDstChannelIdx = 
            eFunc == DF_METADATA_SOURCE ? 
                DBMETADATA_IDX : 
                DBTEMPLATE_IDX;
        BUFFERDIMENSIONS vSrcDims(pDescriberTemplate->GetBufferDimensions(uiSrcChannelIdx));
        pDescribedPortTemplate->SetBufferDimensions(vSrcDims, uiDstChannelIdx);

        Lock();
        m_pDescribedPort = pPort;
        m_eDescriptorFunc = eFunc;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets this port to be the scope terminus for a subgraph. Generally speaking, this
    ///             means that it is responsible for popping the control signal context on outbound
    ///             datablocks. Less generally speaking, since the control signal stack is not fully
    ///             used yet, this means the port is responsible for setting specified control signal
    ///             on outbound blocks (without overwriting other existing control signals). The
    ///             default super-class implementation of this method fails because only output ports
    ///             can terminate a scope in a well-formed graph.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <param name="luiSignal">    true to trigger. </param>
    /// <param name="bTerminus">    true to terminus. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    OutputPort::SetScopeTerminus(
        __in CONTROLSIGNAL luiSignal, 
        __in BOOL bTerminus
        )
    {
        Lock();
        if(bTerminus) {
            m_bScopeTerminus = bTerminus;
            m_luiScopeTerminalSignal |= luiSignal;
        } else {
            m_luiScopeTerminalSignal &= ~luiSignal;
            m_bScopeTerminus = HASSIGNAL(m_luiScopeTerminalSignal);
        }
        Unlock();
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is the scope terminus for a subgraph. If it is, 
    ///             it is responsible for appending a control signal to outbound blocks. 
    ///              </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <returns>   true if scope terminus port, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    OutputPort::IsScopeTerminus(
        VOID
        )
    {
        return m_bScopeTerminus && 
            HASSIGNAL(m_luiScopeTerminalSignal);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is the scope terminus for a subgraph. If it is, 
    ///             it is responsible for appending a control signal to outbound blocks. 
    ///              </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <returns>   true if scope terminus port, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    OutputPort::IsScopeTerminus(
        CONTROLSIGNAL luiControlSignal
        )
    {
        return m_bScopeTerminus && 
            TESTSIGNAL(m_luiScopeTerminalSignal, luiControlSignal);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the port is a descriptor port. </summary>
    ///
    /// <remarks>   Crossbac, 2/14/2013. </remarks>
    /// 
    /// <returns>   true if descriptor port, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    OutputPort::IsDescriptorPort(
        VOID
        ) 
    {
        return m_pDescribedPort != NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets descriptor port for the given descriptor function (if there is one). </summary>
    ///
    /// <remarks>   Crossbac, 2/15/2013. </remarks>
    ///
    /// <param name="eFunc">    The function. </param>
    ///
    /// <returns>   null if it fails, else the descriptor port. </returns>
    ///-------------------------------------------------------------------------------------------------

    OutputPort * 
    OutputPort::GetDescriptorPort(
        DESCRIPTORFUNC eFunc
        )
    {
        // this is a static property (set during Graph::Finalize()) that is only
        // queried at dispatch time. No lock is necessary consequently. 
        if(m_vDescriptorPorts.size() == 0)
            return NULL;
        vector<DEFERREDPORTDESC*>::iterator vi;
        for(vi=m_vDescriptorPorts.begin(); vi!= m_vDescriptorPorts.end(); vi++) {
            DEFERREDPORTDESC * pDesc = *vi;
            if(pDesc->func == eFunc)
                return reinterpret_cast<OutputPort*>(pDesc->pPort);
        }
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets pending allocation. </summary>
    ///
    /// <remarks>   Crossbac, 2/15/2013. </remarks>
    ///
    /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
    /// <param name="uiSizeBytes">          The size in bytes. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    OutputPort::SetPendingAllocationSize(
        __in Accelerator * pDispatchAccelerator, 
        __in UINT          uiSizeBytes
        )
    {
        // this had better be called only if this port
        // is the descriptor port for another port, and we have
        // no outstanding pending collaborative allocations!        
        assert(m_pDescribedPort != NULL);
        assert(m_eDescriptorFunc == DF_METADATA_SOURCE || m_eDescriptorFunc == DF_TEMPLATE_SOURCE);
        assert(!m_bPendingAllocation);

        Lock();
        m_bPendingAllocation = TRUE;
        m_uiPendingAllocationSize = uiSizeBytes;
        m_pPendingAllocationAccelerator = pDispatchAccelerator;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets pending the allocation size. Should only be called if this port is a
    ///             descriptor port for another output port, and both ports have meta-port allocators
    ///             that determine the buffer sizes required at dispatch. In such a case, we defer
    ///             the block allocation and binding until all sizes are available. This method
    ///             returns the pending size, which should have been stashed in a call to
    ///             SetPendingAllocationSize.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 2/15/2013. </remarks>
    ///
    /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    UINT
    OutputPort::GetPendingAllocationSize(
        __in Accelerator * pDispatchAccelerator
        )
    {
        assert(m_bPendingAllocation);
        assert(m_eDescriptorFunc == DF_METADATA_SOURCE || m_eDescriptorFunc == DF_TEMPLATE_SOURCE);
        assert(m_pPendingAllocationAccelerator == pDispatchAccelerator);
        UNREFERENCED_PARAMETER(pDispatchAccelerator);

        UINT uiSize = 0;
        Lock();
        // this is silly...either we need to lock all the
        // related output ports and meta ports before doing
        // this, or convince ourselves there is no need for
        // a lock. This lock, for instance, does nothing to
        // ensure correctness, since it's an obvious TOCTOU
        uiSize = m_uiPendingAllocationSize;
        Unlock();
        return uiSize;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Complete pending bindings. </summary>
    ///
    /// <remarks>   Crossbac, 2/14/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    OutputPort::CompletePendingAllocation(
        __in Accelerator * pAccelerator, 
        __in Datablock * pBlock
        )
    {
        // this block provides the buffer required to complete
        // the output binding for this port. figure it out from
        // out descriptor func, and finish the binding.
        UNREFERENCED_PARAMETER(pAccelerator);
        assert(m_pPendingAllocationAccelerator == pAccelerator);
        assert(m_pDescribedPort != NULL);
        assert(m_bPendingAllocation);
        assert(m_uiPendingAllocationSize != 0);
        Lock();
        SetDestinationBuffer(pBlock, FALSE);
        m_bPendingAllocation = FALSE;
        m_uiPendingAllocationSize = 0;
        m_pPendingAllocationAccelerator = NULL;
        Unlock();
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
    OutputPort::GetBlockFromPool(
        __in Accelerator * pAccelerator,
        __in UINT uiDataBytes,
        __in UINT uiMetaBytes,
        __in UINT uiTemplateBytes
        )
	{
		return GetPooledDestinationBuffer(pAccelerator, uiDataBytes, uiMetaBytes, uiTemplateBytes);
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check that the block pool contain only datablocks with no control signals. </summary>
    ///
    /// <remarks>   Crossbac, 3/2/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    OutputPort::CheckBlockPoolStates(
        VOID
        )
    {
#ifdef GRAPH_DIAGNOSTICS
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
#endif
    }

};
