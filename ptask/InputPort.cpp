//--------------------------------------------------------------------------------------
// File: InputPort.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "AsyncContext.h"
#include "InputPort.h"
#include "OutputPort.h"
#include "assert.h"
#include "PTaskRuntime.h"
#include "task.h"
#include "signalprofiler.h"
#include <string>

using namespace std;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    InputPort::InputPort(
        VOID
        ) 
    {
        m_uiId = NULL;
        m_pTemplate = NULL;
        m_ePortType = INPUT_PORT;
        m_pInOutConsumer = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    InputPort::~InputPort() {
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Configure iteration. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="uiIterations"> The iterations. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    InputPort::BeginIterationScope(
        UINT uiIterations
        )
    {
        Lock();
        assert(m_bIterated);
        assert(m_pIterationSource != NULL);
        assert(uiIterations != 0);
        assert(!m_uiIterationUpperBound && "ConfigureIteration clobbering existing iteration!");
        assert(!m_uiIterationIndex && "ConfigureIteration clobbering existing iteration!");
        m_uiIterationUpperBound = uiIterations;
        m_uiIterationIndex = 0;
        if(m_pIterationSource->IsSticky()) {
            m_uiStickyIterationUpperBound = m_uiIterationUpperBound;
        } else {
            m_uiStickyIterationUpperBound = 0;
        }
        Unlock();
        m_pBoundTask->SignalPortStatusChange();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   complete scoped iteration. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="uiIterations"> The iterations. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    InputPort::EndIterationScope(
        VOID
        )
    {
        Lock();
        assert(m_bIterated);
        assert(m_pIterationSource != NULL);
        assert(!m_bActiveIterationScope);
        if(m_uiIterationIndex == m_uiIterationUpperBound) {
            m_bActiveIterationScope = FALSE;
            if(m_pIterationSource->IsSticky()) {
                m_uiIterationUpperBound = m_uiStickyIterationUpperBound;
                m_uiIterationIndex = 0;
            } else {
                m_uiStickyIterationUpperBound = 0;
            }
        } else {
            m_bActiveIterationScope = TRUE;
        }
        Unlock();
        m_pBoundTask->SignalPortStatusChange();
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an iteration source. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    InputPort::SetIterationSource(
        Port * pPort
        )
    {
        Lock();
        assert(m_pIterationSource == NULL); // already configured?
        assert(!m_bIterated);               // already configured?
        m_pIterationSource = pPort;
        m_bIterated = TRUE;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the iteration source. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the iteration source. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port *          
    InputPort::GetIterationSource(
        VOID
        )
    {
        return m_pIterationSource;
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
    InputPort::IsOccupied(
        VOID
        ) 
    {
        BOOL bResult = FALSE;
        Lock();
        if(m_bPermanentBlock) {
            assert(m_pReplayableBlock != NULL);
            bResult = m_pReplayableBlock != NULL;
        } else if(m_bIterated) {
            // if we are an iterated port and we have a replayable
            // block then we are occupied, because thats the next thing
            // we will produce.
            if(m_uiIterationUpperBound != 0 && m_uiIterationIndex < m_uiIterationUpperBound) {
                bResult = (m_pReplayableBlock != NULL);
                if(!bResult) {
                    if(m_vControlChannels.size() > 0) {
                        bResult = m_vControlChannels[0]->IsReady();
                    }
                    if(!bResult) {
                        if(m_vChannels.size() > 0) {
                            bResult = m_vChannels[0]->IsReady();
                        }
                    }
                }
            }
        } else {
            // if we are not iterated, or we are iterated but have no 
            // replayable block yet, then we are ready if there is a block upstream.
            assert(m_vControlChannels.size() <= 1);
            assert(m_vChannels.size() <= 1);
            if(m_vControlChannels.size() > 0) {
                bResult = m_vControlChannels[0]->IsReady();
            }
            if(!bResult) {
                if(m_vChannels.size() > 0) {
                    bResult = m_vChannels[0]->IsReady();
                }
            }
            if(!bResult && m_bSticky) {
                bResult = (m_pReplayableBlock != NULL);
            }
        }
        Unlock();
        return bResult;
    }
    
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pulls the given.  </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *		
    InputPort::Pull(
        VOID
        ) 
    {
        // if there is a block on the control channel
        // prefer it, other wise do a pull on the
        // main channel. 
        Datablock * pBlock = NULL;
        Lock();
        if(m_bIterated) {
            assert(!m_bPermanentBlock);
            if(m_uiIterationUpperBound != 0 && (m_uiIterationIndex < m_uiIterationUpperBound)) {
                // if the iteration upper bound is not configured, do not
                // return any blocks. Wait for a new iteration to begin.
                if(!m_pReplayableBlock) {
                    pBlock = AttemptPull(&m_vControlChannels);
                    if(pBlock == NULL) {
                        pBlock = AttemptPull(&m_vChannels);
                    }
                    assert(pBlock);
                    m_pReplayableBlock = pBlock;
                    // m_pReplayableBlock->AddRef();
                }
                if(m_pReplayableBlock != NULL && m_uiIterationIndex < m_uiIterationUpperBound) {
                    // if this port is scoped under a general iterator, check to see if we have a replayable block
                    // whose replay count is not yet exhausted. If we do, then we want to return that block
                    // regardless of whether we have data available upstream. If this is the last iteration, 
                    // we will release the block, which we do by eliding the AddRef that we normally call to
                    // ensure we keep the block around.
                    pBlock = m_pReplayableBlock;
                    if(++m_uiIterationIndex == m_uiIterationUpperBound) {
                        // no addref, ensuring the consumer's release will delete the block
                        // (assuming no one else is also keeping a reference).
                        m_pReplayableBlock = NULL; 
                        if(m_bActiveIterationScope && m_pIterationSource->IsSticky()) {
                            m_bActiveIterationScope = FALSE;
                            m_uiIterationUpperBound = m_uiStickyIterationUpperBound;
                            m_uiIterationIndex = 0;
                            if(m_uiIterationUpperBound)
                                m_pBoundTask->SignalPortStatusChange();
                        }
                    } else {
                        // add a reference so we still have the block 
                        // after the consumer releases it. 
                        m_pReplayableBlock->AddRef();
                    }
                }
            }
        } else {
            // This is not an iterated port. Check upstream for data. 
            assert(!m_bIterated);
            if(!m_bPermanentBlock) {
                assert(m_vControlChannels.size() <= 1);
                assert(m_vChannels.size() <= 1);
                pBlock = AttemptPull(&m_vControlChannels);
                if(pBlock == NULL) {
                    pBlock = AttemptPull(&m_vChannels);
                } 
    #ifdef GRAPH_DIAGNOSTICS
                // cjr 3/11/13:
                // this logic is dubious at best and destructive at worst.
                // If, in fact, there is a block available on the control channel
                // it should be preferred by definition. 
                //else 
                //{
                //    Datablock * pBlock2 = AttemptPull(&m_vChannels);
                //    if(pBlock && pBlock2) {
                //        Channel * pChannel = m_vChannels[0];
                //        Channel * pControlChannel = m_vControlChannels[0];
                //        if((pChannel->GetType() == PTask::CT_INITIALIZER) ||
                //            (pControlChannel->GetType() == PTask::CT_INITIALIZER)) {
                //            assert(false && "potential race: init channel and non-init channel both ready!");
                //        }
                //    }
                //}
                else 
                {
                    Channel * pChannel = m_vChannels[0];
                    if((pChannel->GetType() == PTask::CT_INITIALIZER)) {
                        CONTROLSIGNAL signal = pChannel->GetPropagatedControlSignals();
                        // UINT uiQueueDepth = 0; // pChannel->GetQueueDepth();
                        if(HASSIGNAL(signal) && pChannel->IsReady(CE_DST)) {
                            PTask::Runtime::Warning("WARNING: init channel and control channel were both ready!\n");
                        }
                    }
                }
    #endif
                ctlpingress(this, pBlock);
            }
            if(m_bSticky) {
                if(m_pReplayableBlock != NULL) {
                    if(pBlock && !m_bPermanentBlock) {
                        m_pReplayableBlock->Release();
                        m_pReplayableBlock = pBlock;
                    } else {
                        pBlock = m_pReplayableBlock;
                    }
                } else {
                    m_pReplayableBlock = pBlock;
                }
                if(m_pReplayableBlock != NULL) {
                    m_pReplayableBlock->AddRef();
                }
            }
        }
        Unlock();
        ctlpegress(this, pBlock);
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Returns the top-of-stack object without removing it. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The VOID to peek. </param>
    ///
    /// <returns>   null if it fails, else the current top-of-stack object. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *		
    InputPort::Peek(
        VOID
        ) 
    {
        Datablock * pBlock = NULL;
        Lock();
        if(m_bIterated) {
            if(m_uiIterationUpperBound != 0 && m_uiIterationIndex < m_uiIterationUpperBound) {
                pBlock = m_pReplayableBlock;
                if(pBlock == NULL) {
                    if(m_vControlChannels.size() != 0) {
                        pBlock = m_vControlChannels[0]->Peek();
                    } 
                    if(pBlock == NULL) {
                        if(m_vChannels.size() != 0) {
                            pBlock = m_vChannels[0]->Peek();
                        }
                    }
                }
            }
        } else {
            if(!m_bPermanentBlock) {
                assert(m_vControlChannels.size() <= 1);
                assert(m_vChannels.size() <= 1);
                if(m_vControlChannels.size() != 0) {
                    pBlock = m_vControlChannels[0]->Peek();
                } 
                if(pBlock == NULL) {
                    if(m_vChannels.size() != 0) {
                        pBlock = m_vChannels[0]->Peek();
                    }
                }
            }
            assert(!m_bPermanentBlock || m_pReplayableBlock);
            if(m_bSticky && !pBlock && m_pReplayableBlock != NULL) {
                pBlock = m_pReplayableBlock;
            }
        }
        Unlock();
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   pushing a block into an input port is meaningless: any pushing for input ports
    ///             should be performed by pushing into an attached channel. This operation always
    ///             fails and reports an error consequently.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the Datablock* to push. </param>
    ///
    /// <returns>   false: it always fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL		
    InputPort::Push(
        __in Datablock* pBlock
        ) 
    {
        assert(FALSE);
        UNREFERENCED_PARAMETER(pBlock);
        PTask::Runtime::HandleError("%s: Attempt to push a block into an InputPort(%s)!\n", 
                                    __FUNCTION__,
                                    m_lpszVariableBinding);

        return FALSE;
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
    InputPort::SetPermanentBlock(
        Datablock * pBlock
        )
    {
        Lock();
        assert(m_vChannels.size() == 0);
        assert(m_vControlChannels.size() == 0);
        assert(m_bSticky);
        assert(!m_bTriggerPort);
        assert(m_pReplayableBlock == NULL);
        assert(m_bPermanentBlock == FALSE);
        assert(pBlock != NULL);
        m_bPermanentBlock = TRUE;
        m_pReplayableBlock = pBlock;
        m_pReplayableBlock->AddRef();
        Unlock();
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
    /// <param name="nInOutRouteIdx">       Zero-based index of the in out route. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port*
    InputPort::Create(
        DatablockTemplate * pTemplate,
        UINT uiId, 
        char * lpszVariableBinding,
        int nParmIdx,
        int nInOutRouteIdx
        )
    {
        InputPort * pPort = new InputPort();
        if(SUCCEEDED(pPort->Initialize(pTemplate, uiId, lpszVariableBinding, nParmIdx, nInOutRouteIdx)))
            return pPort;
        delete pPort;
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a destination buffer. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] (optional)  If non-null, the accelerator. </param>
    ///
    /// <returns>   null if it fails, else the destination buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *
    InputPort::GetDestinationBuffer(
        Accelerator * pAccelerator
        ) 
    {
        UNREFERENCED_PARAMETER(pAccelerator);
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an in out consumer. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    InputPort::SetInOutConsumer(
        Port * pPort
        ) 
    {
        Lock();
        assert(!m_bPermanentBlock);
        m_pInOutConsumer = pPort;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind control channel. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    InputPort::BindControlChannel(
        Channel * pChannel
        ) 
    {
        Lock();
        assert(!m_bPermanentBlock);
        m_vControlChannels.push_back(pChannel);
        pChannel->AddRef();
        assert(m_vControlChannels.size() == 1);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind control channel. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    InputPort::UnbindControlChannel(
        VOID
        ) 
    {
        Lock();
        size_t nChannels = m_vControlChannels.size();
        if(nChannels > 0) {
            assert(nChannels == 1);
            m_vControlChannels[0]->Release();
            m_vControlChannels.clear();
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a destination buffer. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> [in,out] If non-null, the. </param>
    ///-------------------------------------------------------------------------------------------------

    void			
    InputPort::SetDestinationBuffer(
        Datablock *
        ) 
    {
        assert(false);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets an in out consumer. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the in out consumer. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port*           
    InputPort::GetInOutConsumer(
        VOID
        ) 
    { 
        return m_pInOutConsumer; 
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
    InputPort::CheckTypeSpecificSemantics(
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
                << " is configured as a InputPort with a permanent sticky block, which"
                << " requires it to be bound with no upstream"
                << " channel connections: this condition does not hold."
                << endl;
        }
        if(m_vChannels.size() == 0 && m_vControlChannels.size() == 0 && !m_bPermanentBlock) {
            bResult = FALSE;
            os << this << "not bound to any input channels" << endl;
        }
        vector<Channel*>::iterator vi;
        for(vi=m_vChannels.begin(); vi!=m_vChannels.end(); vi++) {
            // this port had better be attached to an input
            // channel or an internal channel.
            Channel * pChannel = *vi;
            if(pChannel->GetType() == CT_GRAPH_OUTPUT) {
                bResult = FALSE;
                os << this << "bound to an output channel!" << endl;
            }
            if(pChannel->GetType() == CT_INTERNAL) {
                Port * pSrc = pChannel->GetBoundPort(CE_SRC);
                Port * pDst = pChannel->GetBoundPort(CE_DST);
                if(pSrc == NULL || pDst == NULL) {
                    bResult = FALSE;
                    os << pChannel << "has missing port bindings!" << endl;
                } else {
                    Task * pSrcTask = pSrc->GetTask();
                    Task * pDstTask = pDst->GetTask();
                    if(pSrcTask == pDstTask) {
                        // this channel is a back-edge. There had better be
                        // a control channel on this port or the internal
                        // channel will never be fed!
                        if(m_vControlChannels.size() == 0) {
                            bResult = FALSE;
                            os  << this 
                                << " is fed by a back-channel: " 
                                << pChannel 
                                << " that has no forward (e.g. initializer channel). "
                                << m_pBoundTask 
                                << " can never enter the ready state!"
                                << endl;
                        }
                    }
                }
            }
        }
        for(vi=m_vControlChannels.begin(); vi!=m_vControlChannels.end(); vi++) {
            // this port had better be attached to an input
            // channel or an internal channel.
            Channel * pChannel = *vi;
            if(pChannel->GetType() == CT_GRAPH_OUTPUT) {
                bResult = FALSE;
                os << this << "bound to an output control channel!" << endl;
            }
            if(pChannel->GetType() == CT_INTERNAL) {
                Port * pSrc = pChannel->GetBoundPort(CE_SRC);
                Port * pDst = pChannel->GetBoundPort(CE_DST);
                if(pSrc == NULL || pDst == NULL) {
                    bResult = FALSE;
                    os << pChannel << "has missing port bindings!" << endl;
                } else {
                    Task * pSrcTask = pSrc->GetTask();
                    Task * pDstTask = pDst->GetTask();
                    if(pSrcTask == pDstTask) {
                        // this channel is a back-edge. There had better be
                        // a control channel on this port or the internal
                        // channel will never be fed!
                        if(m_vChannels.size() == 0) {
                            bResult = FALSE;
                            os  << this 
                                << " is fed by a back-channel: " 
                                << pChannel 
                                << " that has no forward (e.g. initializer channel). "
                                << m_pBoundTask 
                                << " can never enter the ready state!"
                                << endl;
                        }
                    }
                }
            }
        }
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the replayable block. </summary>
    ///
    /// <remarks>   Crossbac, 2/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void            
    InputPort::ReleaseReplayableBlock(
        VOID
        )
    {
        Lock();
        if(m_pReplayableBlock!= NULL) {
            m_pReplayableBlock->Release();
            m_pReplayableBlock = NULL;
        }
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
    InputPort::HasBlockPool(
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
    InputPort::ForceBlockPoolHint(
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
    InputPort::AllocateBlockPool(
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
    InputPort::DestroyBlockPool(
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
    InputPort::IsBlockPoolActive(
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
    InputPort::AllocateBlockPoolAsync(
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
    InputPort::FinalizeBlockPoolAsync(
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
    InputPort::AddNewBlock(
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
    InputPort::ReturnToPool(
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
    InputPort::GetPoolSize(
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
    InputPort::SetRequestsPageLocked(
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
    InputPort::GetRequestsPageLocked(
        VOID
        )
    {
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Find the maximal capacity downstream port/channel path starting at this port.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 1/3/2014. </remarks>
    ///
    /// <param name="vTasksVisited">    [in,out] [in,out] If non-null, the tasks visited. </param>
    /// <param name="vPath">            [in,out] [in,out] If non-null, full pathname of the file. </param>
    ///
    /// <returns>   The found maximal downstream capacity. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    InputPort::FindMaximalDownstreamCapacity(
        __inout std::set<Task*>& vTasksVisited,
        __inout std::vector<Channel*>& vPath
        )
    {
        if(m_pInOutConsumer != NULL) {

            // we have an outbound channel connected to an in/out consumer pair. 
            // the downstream capacity on  this port is the maximal downstream 
            // capacity of the out party in the inout pair. 

            return m_pInOutConsumer->FindMaximalDownstreamCapacity(vTasksVisited, vPath); 
        }

        // blocks arriving at this port have reached their terminus
        // along this path, so the downstream capacity is 0.

        return 0;
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
    InputPort::IsExplicitMemorySpaceTransitionPoint(
        VOID
        )
    {
        // if this port passes blocks through,
        // then we need to know whether the output
        // port is a transition point. Otherwise,
        if(m_pInOutConsumer != NULL) 
            return m_pInOutConsumer->IsExplicitMemorySpaceTransitionPoint();
        return FALSE;
    }


};
