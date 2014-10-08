//--------------------------------------------------------------------------------------
// File: StickyPort.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "AsyncContext.h"
#include "StickyPort.h"
#include "assert.h"
#include "signalprofiler.h"
#include <vector>

using namespace std;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    StickyPort::StickyPort(
        VOID
        ) 
    {
        m_uiId = NULL;
        m_pTemplate = NULL;
        m_ePortType = STICKY_PORT;
        m_pStickyDatablock = NULL;
        m_bSticky = TRUE;
        m_bDispatchDimensionsHint = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    StickyPort::~StickyPort() {
        PTSRELEASE(m_pStickyDatablock);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this port is occupied. A StickyPort is occupied if the upstream channel
    ///             has avaiable blocks, or if it still has a copy of it's sticky block. If the
    ///             upstream channel is empty, StickyPorts return the last block returned, so if that
    ///             block is non-null the port is still occupied.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if occupied, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    StickyPort::IsOccupied(
        VOID
        ) 
    {
        BOOL bResult = FALSE;
        Lock();
        if(m_pStickyDatablock != NULL) {
            bResult = TRUE;
        } else {
            if(m_vControlChannels.size()) {
                bResult = m_vControlChannels[0]->IsReady();
            }
            if(!bResult && m_vChannels.size()) {
                bResult = m_vChannels[0]->IsReady();
            }
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pull the next datablock from this port. If the upstream channel has datablocks
    ///             pull from there and return the result. Otherwise, return the last block pulled,
    ///             or null if no such block exists.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *		
    StickyPort::Pull(
        VOID
        ) 
    {
        BOOL bPulled = FALSE;
        Lock();        
        if(m_vControlChannels.size() != 0) {
            assert(!m_bPermanentBlock);            
            if(m_vControlChannels[0]->IsReady()) {
                Datablock * pBlock = m_vControlChannels[0]->Pull();
                if(pBlock) {
                    ctlpingress(this, pBlock);
                    PTSRELEASE(m_pStickyDatablock);
                    m_pStickyDatablock = pBlock;
                }
                bPulled = TRUE;
            }
        }
        if(!bPulled && m_vChannels.size() != 0) {
            assert(!m_bPermanentBlock);            
            if(m_vChannels[0]->IsReady()) {
                Datablock * pBlock = m_vChannels[0]->Pull();
                if(pBlock) {
                    ctlpingress(this, pBlock);
                    PTSRELEASE(m_pStickyDatablock);
                    m_pStickyDatablock = pBlock;
                }
            }
        }
        Unlock();
        ctlpegress(this, m_pStickyDatablock);
        return m_pStickyDatablock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a block to be the permanently sticky block for this port. Obviously, only
    ///             valid for certain kinds of ports (input varieties). Use for blocks that will have
    ///             only one value for the lifetime of the graph, to avoid creating and manageing an
    ///             exposed channel or initializer channel that will only every be used once. Do not
    ///             connect an upstream channel to ports that have been configured with a permanent
    ///             block.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="p">    If non-null, the Datablock* to push. </param>
    ///-------------------------------------------------------------------------------------------------

    void			
    StickyPort::SetPermanentBlock(
        Datablock * pBlock
        )
    {
        Lock();
        assert(m_vChannels.size() == 0);
        assert(m_vControlChannels.size() == 0);
        assert(m_bSticky);
        assert(!m_bTriggerPort);
        assert(m_pStickyDatablock == NULL);
        assert(m_bPermanentBlock == FALSE);
        assert(pBlock != NULL);
        m_bPermanentBlock = TRUE;
        m_pStickyDatablock = pBlock;
        m_pStickyDatablock->AddRef();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Returns the value that will be returned by the next Pull call without blocking
    ///             and without changing the state of the port.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The VOID to peek. </param>
    ///
    /// <returns>   null if it fails, else the current top-of-stack object. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *		
    StickyPort::Peek(
        VOID
        ) 
    {
        BOOL bPeeked = FALSE;
        Datablock * pBlock = NULL;
        Lock();        
        if(m_vControlChannels.size() && m_vControlChannels[0]->IsReady()) {
            pBlock = m_vControlChannels[0]->Peek();
            bPeeked = TRUE;
        } 
        if(!bPeeked && m_vChannels.size() && m_vChannels[0]->IsReady()) {
            pBlock = m_vChannels[0]->Peek();
            bPeeked = TRUE;
        }         
        if(!bPeeked && m_pStickyDatablock) {
            pBlock = m_pStickyDatablock;
        }
        Unlock();
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Push is meaningless for StickyPort. No-op.</summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="p">    [in,out] If non-null, the Datablock* to push. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL		
    StickyPort::Push(
        Datablock* p
        ) 
    {
        assert(FALSE);
        UNREFERENCED_PARAMETER(p);
        return FALSE;
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
    StickyPort::Create(
        DatablockTemplate * pTemplate,
        UINT uiId, 
        char * lpszVariableBinding,
        int nParmIdx,
        int nInOutRouteIdx
        )
    {
        StickyPort * pPort = new StickyPort();
        if(SUCCEEDED(pPort->Initialize(pTemplate, uiId, lpszVariableBinding, nParmIdx, nInOutRouteIdx)))
            return pPort;
        delete pPort;
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the destination buffer. Always return NULL for StickyPort. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] (optional)  If non-null, the accelerator. </param>
    ///
    /// <returns>   null if it fails, else the destination buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *
    StickyPort::GetDestinationBuffer(
        Accelerator * pAccelerator
        ) 
    {
        UNREFERENCED_PARAMETER(pAccelerator);
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the destination buffer. A no-op for StickyPort. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> [in,out] If non-null, the. </param>
    ///-------------------------------------------------------------------------------------------------

    void			
    StickyPort::SetDestinationBuffer(
        Datablock *
        ) 
    {
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
    StickyPort::CheckTypeSpecificSemantics(
        std::ostream * pos,
        PTask::Graph * pGraph
        )
    {
        UNREFERENCED_PARAMETER(pGraph);
        BOOL bResult = TRUE;
        std::ostream& os = *pos;
        if(m_bPermanentBlock && 
           (m_vChannels.size() != 0 ||
            m_vControlChannels.size() != 0)) {
            bResult = FALSE;
            os << this 
                << " is configured as a StickyPort with a permanent sticky block, which"
                << " requires it to be bound with no upstream"
                << " channel connections: this condition does not hold."
                << endl;
        }
        if(m_vChannels.size() == 0 && 
           m_vControlChannels.size() != 0) { 
            bResult = FALSE;
            os << this 
                << "is bound to only to control channels. "
                << "Is this what you intend?"
                << endl;
        }
        if(m_vChannels.size() == 0 && 
           m_vControlChannels.size() == 0 && 
           !m_bPermanentBlock) {
            bResult = FALSE;
            os << this 
                << "not bound to any channels. "
                << "Where will it get its input?"
                << endl;
        }
        vector<Channel*>::iterator vi;
        for(vi=m_vChannels.begin(); vi!=m_vChannels.end(); vi++) {
            // this port had better be attached to an output
            // channel or an internal channel.
            Channel * pChannel = *vi;
            if(pChannel->GetType() == CT_GRAPH_OUTPUT) {
                bResult = FALSE;
                os << this << "bound to an output channel!" << endl;
            }
        }
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
    StickyPort::HasBlockPool(
        VOID
        )
    {
        return FALSE;
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
    StickyPort::ForceBlockPoolHint(
        __in UINT nPoolSize,
        __in UINT nStride,
        __in UINT nDataBytes,
        __in UINT nMetaBytes,
        __in UINT nTemplateBytes,
        __in BOOL bPageLockHostViews,
        __in BOOL bEagerDeviceMaterialize
        )
    {
        UNREFERENCED_PARAMETER(nPoolSize);
        UNREFERENCED_PARAMETER(nStride);
        UNREFERENCED_PARAMETER(nDataBytes);
        UNREFERENCED_PARAMETER(nMetaBytes);
        UNREFERENCED_PARAMETER(nTemplateBytes);
        UNREFERENCED_PARAMETER(bPageLockHostViews);
        UNREFERENCED_PARAMETER(bEagerDeviceMaterialize);
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
    StickyPort::AllocateBlockPool(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )
    {
        UNREFERENCED_PARAMETER(pAccelerators);
        UNREFERENCED_PARAMETER(uiPoolSize);
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destroys the block pool. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    StickyPort::DestroyBlockPool(
        VOID
        )
    {

    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Is the block pool available/active? </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    StickyPort::IsBlockPoolActive(
        VOID
        )
    {
        return FALSE;
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
    StickyPort::AllocateBlockPoolAsync(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )
    {
        UNREFERENCED_PARAMETER(pAccelerators);
        UNREFERENCED_PARAMETER(uiPoolSize);
        return TRUE;
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
    StickyPort::FinalizeBlockPoolAsync(
        VOID
        )
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds to the pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    StickyPort::AddNewBlock(
        Datablock * pBlock
        )
    {
        UNREFERENCED_PARAMETER(pBlock);
        assert(FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds to the pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    StickyPort::ReturnToPool(
        Datablock * pBlock
        )
    {
        UNREFERENCED_PARAMETER(pBlock);
        assert(FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   gets the pool size. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    UINT
    StickyPort::GetPoolSize(
        VOID
        )
    {
        assert(FALSE);
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets request page locked. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="bPageLocked">  true to lock, false to unlock the page. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    StickyPort::SetRequestsPageLocked(
        BOOL bPageLocked
        )
    {
        assert(FALSE);
        UNREFERENCED_PARAMETER(bPageLocked);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets request page locked. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    StickyPort::GetRequestsPageLocked(
        VOID
        )
    {
        return FALSE;
    }

};
