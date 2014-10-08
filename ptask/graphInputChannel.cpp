//--------------------------------------------------------------------------------------
// File: GraphInputChannel.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include <stdio.h>
#include <crtdbg.h>
#include "channel.h"
#include "graphInputChannel.h"
#include "ptaskutils.h"
#include "InputPort.h"
#include "OutputPort.h"
#include "task.h"

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pGraph">                   [in,out] If non-null, the graph. </param>
    /// <param name="pTemplate">                [in,out] If non-null, the template. </param>
    /// <param name="hRuntimeTerminateEvent">   Handle of the terminate. </param>
    /// <param name="hGraphTeardownEvent">      Handle of the stop. </param>
    /// <param name="hGraphStopEvent">          The graph stop event. </param>
    /// <param name="lpszChannelName">          [in,out] (optional)  If non-null, name of the
    ///                                         channel. </param>
    /// <param name="bHasBlockPool">            true if this object has block pool. </param>
    ///-------------------------------------------------------------------------------------------------

    GraphInputChannel::GraphInputChannel(
        __in Graph * pGraph,
        __in DatablockTemplate * pTemplate,
        __in HANDLE hRuntimeTerminateEvent,
        __in HANDLE hGraphTeardownEvent,
        __in HANDLE hGraphStopEvent,
        __in char * lpszChannelName,
        __in BOOL bHasBlockPool
        ) : Channel(pGraph,
                    pTemplate, 
                    hRuntimeTerminateEvent,
                    hGraphTeardownEvent,
                    hGraphStopEvent,
                    lpszChannelName,
                    bHasBlockPool)
    {
        m_type = CT_GRAPH_INPUT;
        m_pBlockPool = NULL;
        m_bHasBlockPool = bHasBlockPool;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    GraphInputChannel::~GraphInputChannel() {
        if(m_pBlockPool) {
            m_pBlockPool->DestroyBlockPool();
            delete m_pBlockPool;
            m_pBlockPool = NULL;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the channel is (or can be) connected to a data source or sink that can be
    ///             streamed. Generally speaking, this is a property of the primitive whose IO
    ///             resources are being exposed by this port; consequently this property must be set
    ///             explicitly by the programmer when graph structures that are stateful are
    ///             constructured. For example, in a sort primitive, the main input can be streamed
    ///             (broken into multiple blocks) only if there is a merge network downstream of the
    ///             node performing the sort. Code that feeds the main input port needs to know this
    ///             to decide whether to grow blocks until all data is present, or two push partial
    ///             input.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   true if the port can stream data, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    GraphInputChannel::CanStream(
        VOID
        )
    {
        Lock();
        // an input channel can stream if
        // its downstream port can stream
        BOOL bResult = FALSE;
        if(m_pDstPort != NULL) {
            bResult = m_pDstPort->CanStream();
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
    GraphInputChannel::HasBlockPool(
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
    GraphInputChannel::ForceBlockPoolHint(
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
    GraphInputChannel::AllocateBlockPool(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )        
    {
        UNREFERENCED_PARAMETER(pAccelerators);
        UNREFERENCED_PARAMETER(uiPoolSize);
        assert(FALSE && "obsolete: use AllocateBlockPoolAsync instead!");
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets high water mark. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    GraphInputChannel::GetHighWaterMark(
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
    GraphInputChannel::GetOwnedBlockCount(
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
    GraphInputChannel::GetLowWaterMark(
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
    GraphInputChannel::GetAvailableBlockCount(
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
    GraphInputChannel::DestroyBlockPool(
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
    GraphInputChannel::IsBlockPoolActive(
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
    GraphInputChannel::GetPoolOwnerName(
        VOID
        )
    {
        return m_lpszName;
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
    GraphInputChannel::AllocateBlockPoolAsync(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )
    {
        if(!m_bHasBlockPool) {

            // The user did not mark this channel for a block pool.
            // This is done (somewhat circuitously) by marking the destination
            // port as having an upstream block pool at port creation time
            return FALSE;       
        }
        
        // Now figure out the pool size. If the user has defaulted the parameter, then take the default
        // input channel pool size. Additionally, if there a constraint on the capacity of this channel,
        // then (assuming the user program doesn't abuse the API) that capacity + 1 is the largest
        // number of blocks that can ever be in flight for this channel, so we can reduce the pool size
        // based on that constraint accordingly. Be sure to warn the user that we have done so though. 
         
        BOOL bDefaultedPoolSize = FALSE;
        if(uiPoolSize == 0) {

            // by choosing a 0 size, the user is telling us to figure it out. Check the destination
            // port for a size parameter. If it too is defaulted, then take the runtime-level
            // default pool size. If the global pool size is defaulted to zero and we cannot
            // get the size from the port, it means we are not supposed to use a pool. 
            // -----------------------------------------------------------------------
            // TODO: FIXME: cjr:
            // There is some ambiguity here in how to interpret a 0-size global default if the 
            // provision block pools for capacity flag is on. It's probably worth clearing this up
            // and supporting a more formal knob for disabling block pools. 
            
            InputPort * pIPort = reinterpret_cast<InputPort*>(GetBoundPort(CE_DST));
            UINT uiPortBlockPoolSize = pIPort == NULL ? 0 : pIPort->GetUpstreamChannelPoolSize();
            uiPoolSize = (uiPortBlockPoolSize == 0) ? Runtime::GetDefaultInputChannelBlockPoolSize() : uiPortBlockPoolSize;
            bDefaultedPoolSize = (uiPortBlockPoolSize == 0);
        }

        if(uiPoolSize == 0 && Runtime::GetDefaultInputChannelBlockPoolSize() == 0) {

            // the channel was marked for a block pool, but the size and the global default size are 0,
            // which would mean creating a 0-size pool. The global 0 default means disable pools currently.

            if(Runtime::GetProvisionBlockPoolsForCapacity()) {

                // Note that by returning here, we are letting the 0-size global default override the provision
                // block pools for capacity flag. Make sure the user knows about this by quacking vociferously.

                PTask::Runtime::MandatoryInform("%s:%s: *not* provisionging pool size(%d) on %s to match capacity(%d)\n"
                                                "    because the global input channel pool size default is 0 (meaning no pools)!\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                uiPoolSize, 
                                                m_lpszName,
                                                m_uiCapacity);
            }

            return FALSE;
        }

        if(Runtime::GetProvisionBlockPoolsForCapacity() && bDefaultedPoolSize) {

            // generally speaking we want to resize the pool to ensure we never have to allocate
            // in line. However, we can get ourselves in trouble that way. If the blocks are large,
            // we can go overboard, and in situations where the blocks are actually shared state,
            // it's a waste to allocate more than one. As a heuristic, if the input port has multiple
            // predicate channels attached, we are dealing with shared state, so don't alter the size.
            
            Port * pPort = GetBoundPort(CE_DST);
            BOOL bMutableState = (pPort->GetChannelCount() && pPort->GetControlChannelCount());

            if(!bMutableState) {

                // it's ok to doctor the pool size
                std::vector<Channel*> vPath;
                std::set<Task*> vTasksVisited;
                UINT uiMaximalDownstreamCapacity = FindMaximalDownstreamCapacity(vTasksVisited, vPath);
                if(uiMaximalDownstreamCapacity > uiPoolSize) {
                    PTask::Runtime::MandatoryInform("Growing block pool on %s from %d to %d based on downstream capacity!\n",
                                                    m_lpszName,
                                                    uiPoolSize,
                                                    uiMaximalDownstreamCapacity);
                    uiPoolSize = uiMaximalDownstreamCapacity;
                }

                uiPoolSize = (uiPoolSize > uiMaximalDownstreamCapacity + 1) ? uiMaximalDownstreamCapacity + 1 : uiPoolSize;
            }
        } 

        Lock();
        BUFFERACCESSFLAGS ePermissions = PT_ACCESS_DEFAULT;
        InputPort * pPort = reinterpret_cast<InputPort*>(GetBoundPort(CE_DST));
        Task * pTask = pPort->GetTask();
        ACCELERATOR_CLASS accClass = pTask->GetAcceleratorClass();
        BOOL bDepPort = pPort->HasDependentAcceleratorBinding();
        BOOL bPageLockViews = accClass == ACCELERATOR_CLASS_CUDA || bDepPort;
        if(bDepPort || accClass != ACCELERATOR_CLASS_HOST) {
            ePermissions = PT_ACCESS_ACCELERATOR_READ;
            OutputPort * pOPort = reinterpret_cast<OutputPort*>(pPort->GetInOutConsumer());
            if(pOPort != NULL) {
                ePermissions |= PT_ACCESS_ACCELERATOR_WRITE;
                if(pOPort->HasDownstreamHostConsumer()) {
                    ePermissions |= PT_ACCESS_HOST_READ;
                }
            }
        }
        ePermissions |= PT_ACCESS_HOST_WRITE;
        ePermissions |= PT_ACCESS_POOLED_HINT;
        m_pBlockPool = new BlockPool(m_pTemplate, ePermissions, uiPoolSize, this);
        m_pBlockPool->SetEagerDeviceMaterialize(TRUE);
        m_pBlockPool->SetRequestsPageLocked(bPageLockViews); // !m_pTemplate->HasInitialValue());
        m_pBlockPool->SetGrowable(pPort->IsUpstreamChannelPoolGrowable());
        m_pBlockPool->SetGrowIncrement(pPort->GetUpstreamChannelPoolGrowIncrement());
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
    GraphInputChannel::FinalizeBlockPoolAsync(
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
    GraphInputChannel::GetPooledBlock(
        VOID
        )
    {
        Datablock * pBlock = NULL;
        Lock();
        if(m_bHasBlockPool) {
            pBlock = m_pBlockPool->GetPooledBlock();
            if(pBlock != NULL) {
#ifdef DEBUG
                pBlock->Lock();
                assert(pBlock->RefCount() == 0);
                pBlock->Unlock();
#endif
                pBlock->AddRef();
            } else {
                PTask::Runtime::Inform("XXXX: GraphInputChannel:%s empty block pool, returning null: EN=%d, GRW=%d\n", 
                                       m_lpszName, 
                                       m_pBlockPool->IsEnabled(), 
                                       m_pBlockPool->IsGrowable());
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
    GraphInputChannel::AddNewBlock(
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
    /// <summary>   Query if this object is global pool. </summary>
    ///
    /// <remarks>   crossbac, 8/30/2013. </remarks>
    ///
    /// <returns>   true if global pool, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    GraphInputChannel::BlockPoolIsGlobal(
        void
        )
    {
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return a block to the pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    GraphInputChannel::ReturnToPool(
        Datablock * pBlock
        )
    {
        Lock();
        if(m_pBlockPool) {
            m_pBlockPool->ReturnBlock(pBlock);
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
    GraphInputChannel::GetPoolSize(
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
    GraphInputChannel::SetRequestsPageLocked(
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
    GraphInputChannel::GetRequestsPageLocked(
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
    /// <summary>   Derives an initial value datablock for this channel based on its template,
    /// 			and pushes that datablock into this channel, blocking until there is capacity
    /// 			for an optional timeout in milliseconds. Default timeout is infinite. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="dwTimeout">    (optional) the timeout in milliseconds. Use 0xFFFFFFFF for no
    ///                             timeout. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
	GraphInputChannel::PushInitializer(
		DWORD dwTimeout
		)
	{
		Datablock * pBlock = NULL;
		Lock();
		if(m_bHasBlockPool && m_pBlockPool) {
			if(m_pBlockPool->GetAvailableBlockCount()) {
				pBlock = m_pBlockPool->GetPooledBlock();
			}
		}
		Unlock();
		if(pBlock) {
			return Push(pBlock, dwTimeout);
		}
		return Channel::PushInitializer(dwTimeout);
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
    GraphInputChannel::GetBlockFromPool(
        __in Accelerator * pAccelerator,
        __in UINT uiDataBytes,
        __in UINT uiMetaBytes,
        __in UINT uiTemplateBytes
        )
	{
		Datablock * pResult = NULL;
		Lock();
		if(m_pBlockPool) {
			pResult = m_pBlockPool->GetPooledBlock(pAccelerator, uiDataBytes, uiMetaBytes, uiTemplateBytes);
		}
		Unlock();
		return pResult;
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
    GraphInputChannel::CheckTypeSpecificSemantics(
        std::ostream * pos,
        PTask::Graph * pGraph
        ) 
    {
        UNREFERENCED_PARAMETER(pGraph);
        BOOL bResult = TRUE;
        std::ostream& os = *pos;
        if(m_pDstPort == NULL) {
            bResult = FALSE;
            os << this << "should be bound to a non-null destination port!" << std::endl;
        }
        if(m_pSrcPort != NULL) {
            bResult = FALSE;
            os << this 
                << "should NOT be bound to a any src port, but is bound to " 
                << m_pSrcPort
                << "!"
                << std::endl;
        }
        return bResult;        
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this channel has downstream writers. An output channel is
    ///             considered a writer because we must conservatively assume consumed
    ///             blocks will be written.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   true if downstream writers, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    GraphInputChannel::HasDownstreamWriters(
        VOID
        )
    {
        assert(m_pDstPort != NULL);
        if(m_pDstPort == NULL) return FALSE;
        return (m_pDstPort->IsInOutParameter() || m_pDstPort->IsDestructive());
    }

};
