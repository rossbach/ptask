//--------------------------------------------------------------------------------------
// File: InitializerChannel.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "AsyncContext.h"
#include "channel.h"
#include "InitializerChannel.h"
#include "ptaskutils.h"
#include "PTaskRuntime.h"
#include "InputPort.h"
#include "OutputPort.h"
#include "task.h"
#include "signalprofiler.h"
#include <assert.h>

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 2/14/2013. </remarks>
    ///
    /// <param name="pGraph">                   [in,out] If non-null, the template. </param>
    /// <param name="pDatablockTemplate">       [in,out] Handle of the terminate. </param>
    /// <param name="hRuntimeTerminateEvent">   Handle of the stop. </param>
    /// <param name="hGraphTeardownEvt">        The graph teardown event. </param>
    /// <param name="hGraphStopEvent">          The graph stop event. </param>
    /// <param name="lpszChannelName">          [in,out] If non-null, name of the channel. </param>
    /// <param name="bHasBlockPool">            The has block pool. </param>
    ///-------------------------------------------------------------------------------------------------

    InitializerChannel::InitializerChannel(
        __in Graph * pGraph,
        __in DatablockTemplate * pDatablockTemplate, 
        __in HANDLE hRuntimeTerminateEvent,
        __in HANDLE hGraphTeardownEvt, 
        __in HANDLE hGraphStopEvent, 
        __in char * lpszChannelName,
        __in BOOL bHasBlockPool
        ) : Channel(pGraph,
                    pDatablockTemplate, 
                    hRuntimeTerminateEvent, 
                    hGraphTeardownEvt, 
                    hGraphStopEvent, 
                    lpszChannelName,
                    bHasBlockPool)
    {
        m_type = CT_INITIALIZER;
        m_luiPropagatedControlCode = DBCTLC_NONE;
        m_pControlPropagationSource = NULL;
        m_pPeekedControlPropagationSignalSrc = NULL;
        m_bControlBlockPeeked = FALSE;
        m_luiPeekedControlSignal = DBCTLC_NONE;
        m_pBlockPool = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 2/14/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    InitializerChannel::~InitializerChannel() {
        if(m_pBlockPool) {
            m_pBlockPool->DestroyBlockPool();
            delete m_pBlockPool;
            m_pBlockPool = NULL;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destroys the block pool. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    InitializerChannel::DestroyBlockPool(
        VOID
        )
    {
        if(m_pBlockPool) {
            m_pBlockPool->DestroyBlockPool();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Is the block pool available/active? </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    InitializerChannel::IsBlockPoolActive(
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
    InitializerChannel::GetPoolOwnerName(
        VOID
        )
    {
        return m_lpszName;
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
    InitializerChannel::CanStream(
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
    /// <summary>   Return true if the block that is (or would be) produced in demand to a pull call
    ///             passes all/any predicates.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 2/1/2012. </remarks>
    ///
    /// <param name="ppDemandAllocatedBlock">   [out] If non-null, on exit, the demand allocated
    ///                                         block if all predicates are passed. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    InitializerChannel::PassesPredicates(
        Datablock ** ppDemandAllocatedBlock
        )
    {
        BOOL bPasses = FALSE;
        Lock();
        Datablock * pDemandAllocatedBlock = NULL;
        assert((m_vPredicators[CE_DST].eCanonicalPredicate == CGATEFN_NONE ||
                m_vPredicators[CE_SRC].eCanonicalPredicate == CGATEFN_NONE) ||
                m_vPredicators[CE_SRC].lpfnPredicate != NULL);

        if(m_vPredicators[CE_DST].eCanonicalPredicate == CGATEFN_NONE &&
           m_vPredicators[CE_SRC].eCanonicalPredicate == CGATEFN_NONE) {

            // Initializers are always ready to demand allocate
            // blocks in response to a pull. If there is no predicator
            // on this channel, then we can always produce a new block.
            bPasses = TRUE;

        } else {

            if(m_vPredicators[CE_DST].eCanonicalPredicate != CGATEFN_NONE) {
                PREDICATE_FAILURE_ACTION pfa;
                switch(m_vPredicators[CE_DST].eCanonicalPredicate) {
                case CGATEFN_NONE: 
                    assert(FALSE); // should have been handled above!
                    bPasses = TRUE;
                    break;
                case CGATEFN_DEVNULL:
                    bPasses = FALSE;
                    break;
                case CGATEFN_CLOSE_ON_BOF:                
                    bPasses = !TESTSIGNAL(m_luiPropagatedControlCode, DBCTLC_BOF);
                    break;
                case CGATEFN_OPEN_ON_BOF:
                    bPasses = TESTSIGNAL(m_luiPropagatedControlCode, DBCTLC_BOF);
                    break;
                case CGATEFN_CLOSE_ON_EOF:                
                    bPasses = !TESTSIGNAL(m_luiPropagatedControlCode, DBCTLC_EOF);
                    break;
                case CGATEFN_OPEN_ON_EOF:
                    bPasses = TESTSIGNAL(m_luiPropagatedControlCode, DBCTLC_EOF);
                    break;
                case CGATEFN_CLOSE_ON_BEGINITERATION:
                    bPasses = !TESTSIGNAL(m_luiPropagatedControlCode, DBCTLC_BEGINITERATION);
                    break;
                case CGATEFN_OPEN_ON_BEGINITERATION:
                    bPasses = TESTSIGNAL(m_luiPropagatedControlCode, DBCTLC_BEGINITERATION);
                    break;
                case CGATEFN_CLOSE_ON_ENDITERATION:
                    bPasses = !TESTSIGNAL(m_luiPropagatedControlCode, DBCTLC_ENDITERATION);
                    break;
                case CGATEFN_OPEN_ON_ENDITERATION:
                    bPasses = TESTSIGNAL(m_luiPropagatedControlCode, DBCTLC_ENDITERATION);
                    break;
                case CGATEFN_USER_DEFINED: 
                    assert(m_vPredicators[CE_SRC].lpfnPredicate != NULL);
                    pfa = m_vPredicators[CE_SRC].ePredicateFailureAction;
                    pDemandAllocatedBlock = AllocateBlock(FindAsyncContext(CE_DST, ASYNCCTXT_XFERHTOD));
                    bPasses = (m_vPredicators[CE_SRC].lpfnPredicate)(this, pDemandAllocatedBlock, pfa);
                    break;
                default:
                    assert(false);
                    break;
                }
            } else {
                // predicates at the source end need mean we're taking the
                // control signal from another input port on this task. 
                // Such predicates need to be evaluated by 
                // peeking the control signal from the input port that is
                // bound to this one. 
                assert(m_pControlPropagationSource != NULL);
                assert(m_pControlPropagationSource->GetPortType() != OUTPUT_PORT);
                assert(m_pControlPropagationSource->GetTask() == m_pDstPort->GetTask());
                if(m_pPeekedControlPropagationSignalSrc != NULL) {
                    assert(m_bControlBlockPeeked);
                } else {                                        
                    m_pPeekedControlPropagationSignalSrc = m_pControlPropagationSource->Peek();
                    m_bControlBlockPeeked = (m_pPeekedControlPropagationSignalSrc != NULL);
                    if(m_bControlBlockPeeked) {
                        m_pPeekedControlPropagationSignalSrc->AddRef();
                        m_bControlBlockPeeked = TRUE;
                        m_pPeekedControlPropagationSignalSrc->Lock();
                        m_luiPeekedControlSignal = m_pPeekedControlPropagationSignalSrc->GetControlSignals();
                        m_pPeekedControlPropagationSignalSrc->Unlock();
                    }
                }
                if(m_pPeekedControlPropagationSignalSrc == NULL) {
                    bPasses = FALSE; 
                } else {
                    switch(m_vPredicators[CE_SRC].eCanonicalPredicate) {
                    case CGATEFN_NONE: 
                        assert(FALSE); // should have been handled above!
                        bPasses = TRUE;
                        break;
                    case CGATEFN_DEVNULL:
                        bPasses = FALSE;
                        break;
                    case CGATEFN_CLOSE_ON_BOF:                
                        bPasses = !TESTSIGNAL(m_luiPeekedControlSignal, DBCTLC_BOF);
                        break;
                    case CGATEFN_OPEN_ON_BOF:
                        bPasses = TESTSIGNAL(m_luiPeekedControlSignal, DBCTLC_BOF);
                        break;
                    case CGATEFN_CLOSE_ON_EOF:                
                        bPasses = !TESTSIGNAL(m_luiPeekedControlSignal, DBCTLC_EOF);
                        break;
                    case CGATEFN_OPEN_ON_EOF:
                        bPasses = TESTSIGNAL(m_luiPeekedControlSignal, DBCTLC_EOF);
                        break;
                    case CGATEFN_CLOSE_ON_BEGINITERATION:
                        bPasses = !TESTSIGNAL(m_luiPeekedControlSignal, DBCTLC_BEGINITERATION);
                        break;
                    case CGATEFN_OPEN_ON_BEGINITERATION:
                        bPasses = TESTSIGNAL(m_luiPeekedControlSignal, DBCTLC_BEGINITERATION);
                        break;
                    case CGATEFN_CLOSE_ON_ENDITERATION:
                        bPasses = !TESTSIGNAL(m_luiPeekedControlSignal, DBCTLC_ENDITERATION);
                        break;
                    case CGATEFN_OPEN_ON_ENDITERATION:
                        bPasses = TESTSIGNAL(m_luiPeekedControlSignal, DBCTLC_ENDITERATION);
                        break;
                    case CGATEFN_USER_DEFINED: 
                        assert(FALSE);
                        bPasses = FALSE;
                        break;
                    default:
                        assert(false);
                        break;
                    }
                }
            }
        }
        if(pDemandAllocatedBlock != NULL && (!ppDemandAllocatedBlock || !bPasses)) {
            pDemandAllocatedBlock->Release();
            pDemandAllocatedBlock = NULL;
        } 
        if(bPasses && ppDemandAllocatedBlock) {
            *ppDemandAllocatedBlock = (pDemandAllocatedBlock != NULL) ? 
                pDemandAllocatedBlock : AllocateBlock(FindAsyncContext(CE_DST, ASYNCCTXT_XFERHTOD));
        } 
        Unlock();
        return bPasses;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the channel is ready. For initializers, we are always ready, 
    /// 			unless the block we would allocate in response to the next Pull call 
    /// 			would fail to pass predication. The specification of the behavior is that
    /// 			predication acts on a datablock, examining control signals (or arbitrary
    /// 			information if the predicator is user defined): to meet this specification,
    /// 			we could allocate the datablock, mark it with the appropriate control signal,
    /// 			and then apply the predicator. However, it's much simpler to see if predicate
    /// 			would pass if we had a datablock to test it on. This works for canonical predicators
    /// 			only. For user-defined blocks we have to take the brute force approach.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="type"> (optional) the type of the channel endpoint. Note that initializer
    /// 					channels do not have a connectable source end, so if type is
    /// 					not CE_DST, the call fails. </param>
    ///
    /// <returns>   true if ready, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    InitializerChannel::IsReady(
        CHANNELENDPOINTTYPE type
        ) 
    {
        // there is no source end on an
        // initializer channel!
        if(type != CE_DST)
            return FALSE;

        Lock();
        BOOL bResult = FALSE;
        if(m_uiBlockTransitLimit && m_uiBlocksDelivered >= m_uiBlockTransitLimit) {
            bResult = FALSE;
        } else {
            bResult = PassesPredicates(NULL);
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pulls the next datablock from the channel, blocking with an optional timeout in
    ///             milliseconds. Default timeout is infinite.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="dwTimeout">    (optional) the timeout in milliseconds. Use 0xFFFFFFFF for no
    ///                             timeout. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    InitializerChannel::Pull(
        DWORD dwTimeout
        ) 
    {
        UNREFERENCED_PARAMETER(dwTimeout);
        Datablock * pBlock = NULL;

        Lock();
        if(m_uiBlockTransitLimit && (m_uiBlocksDelivered >= m_uiBlockTransitLimit)) {
            // don't pull on non-ready channels!
            assert(FALSE); 
            pBlock = NULL;
        } else {
            if(PassesPredicates(&pBlock)) {
                assert(pBlock != NULL);
                // clear the control code. 
                m_luiPropagatedControlCode = DBCTLC_NONE;
                if(m_bControlBlockPeeked) {
                    assert(m_pControlPropagationSource != NULL);
                    assert(m_vPredicators[CE_SRC].eCanonicalPredicate != CGATEFN_NONE);
                    assert(m_pPeekedControlPropagationSignalSrc != NULL);
                    m_luiPeekedControlSignal = DBCTLC_NONE;
                    if(m_pPeekedControlPropagationSignalSrc != NULL)
                        m_pPeekedControlPropagationSignalSrc->Release();
                    m_pPeekedControlPropagationSignalSrc = NULL;
                    m_bControlBlockPeeked = FALSE;
                }
                if(pBlock != NULL) {
                    m_uiBlocksDelivered++;
                }
                if(m_uiBlockTransitLimit && 
                   m_uiBlocksDelivered < m_uiBlockTransitLimit &&
                   m_luiInitialPropagatedControlCode != DBCTLC_NONE) {
                    // if there is a block transit limit on this initializer
                    // and the programmer has configured an initial contol signal,
                    // we need to put the channel back into its initial state. 
                    m_luiPropagatedControlCode = m_luiInitialPropagatedControlCode;
                }
            }        
        }
        Unlock();
        ctlpegress(this, pBlock);
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Peek always returns NULL for initializers.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name=""> The VOID to peek. </param>
    ///
    /// <returns>   null if it fails, else the currently available datablock object. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    InitializerChannel::Peek(
        VOID
        ) 
    {       
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pushes a datablock into this channel. Meaningless for initializers.
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pBlock">       [in,out] If non-null, the block. </param>
    /// <param name="dwTimeout">    (optional) the timeout in milliseconds. Use 0xFFFFFFFF for no
    ///                             timeout. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    InitializerChannel::Push(
        Datablock* pBlock,
        DWORD dwTimeout
        ) 
    {
        assert(FALSE);
        UNREFERENCED_PARAMETER(pBlock);
        UNREFERENCED_PARAMETER(dwTimeout);
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate a datablock. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *  
    InitializerChannel::AllocateBlock(
        AsyncContext * pAsyncContext
        ) 
    {
        Datablock * pBlock = NULL;
        Lock();
        if(m_pBlockPool) {
            pBlock = m_pBlockPool->GetPooledBlock();     
            if(pBlock != NULL) {
#ifdef DEBUG
                pBlock->Lock();
                assert(pBlock->RefCount() == 0);
                pBlock->Unlock();
#endif
                pBlock->AddRef();
            }
        } 
        if(pBlock == NULL) {
            DatablockTemplate * pTemplate = GetTemplate();
            BUFFERACCESSFLAGS eFlags = pTemplate->IsByteAddressable() ? PT_ACCESS_BYTE_ADDRESSABLE : 0;
            eFlags |= (PT_ACCESS_ACCELERATOR_READ | PT_ACCESS_ACCELERATOR_WRITE);
            pBlock = Datablock::CreateInitialValueBlock(pAsyncContext, pTemplate, eFlags);
        }
        Unlock();
        return pBlock;
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
    InitializerChannel::CheckTypeSpecificSemantics(
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
    /// <summary>   Query if this object has block pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if block pool, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    InitializerChannel::HasBlockPool(
        VOID
        )
    {
        return m_pBlockPool != NULL;
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
    InitializerChannel::ForceBlockPoolHint(
        __in UINT nPoolSize,
        __in UINT nStride,
        __in UINT nDataBytes,
        __in UINT nMetaBytes,
        __in UINT nTemplateBytes,
        __in BOOL bPageLockHostViews,
        __in BOOL bEagerDeviceMaterialize
        )
    {
        m_pBlockPool->ForceBlockPoolHint(nPoolSize,
                                         nStride,
                                         nDataBytes,
                                         nMetaBytes,
                                         nTemplateBytes,
                                         bPageLockHostViews,
                                         bEagerDeviceMaterialize);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is block pool candidate. </summary>
    ///
    /// <remarks>   crossbac, 4/30/2013. </remarks>
    ///
    /// <returns>   true if block pool candidate, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    InitializerChannel::IsBlockPoolCandidate(
        VOID
        )
    {
        InputPort * pPort = reinterpret_cast<InputPort*>(GetBoundPort(CE_DST));
        Task * pTask = pPort->GetTask();
        ACCELERATOR_CLASS accClass = pTask->GetAcceleratorClass();
        if(accClass == ACCELERATOR_CLASS_HOST) {
            if(pPort->HasDependentAcceleratorBinding())
                return TRUE;
            OutputPort * pOPort = reinterpret_cast<OutputPort*>(pPort->GetInOutConsumer());
            if(pOPort != NULL && !pOPort->HasDownstreamHostConsumer()) 
                return TRUE;
            return FALSE;
        } 
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is block pool candidate. </summary>
    ///
    /// <remarks>   crossbac, 4/30/2013. </remarks>
    ///
    /// <returns>   true if block pool candidate, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    InitializerChannel::IsAcceleratorOnlyBlockPoolCandidate(
        VOID
        )
    {
        assert(IsBlockPoolCandidate());
        InputPort * pPort = reinterpret_cast<InputPort*>(GetBoundPort(CE_DST));
        Task * pTask = pPort->GetTask();
        ACCELERATOR_CLASS accClass = pTask->GetAcceleratorClass();
        if(accClass == ACCELERATOR_CLASS_HOST) {
            if(pPort->HasDependentAcceleratorBinding()) {
                OutputPort * pOPort = reinterpret_cast<OutputPort*>(pPort->GetInOutConsumer());
                if(pOPort != NULL && !pOPort->HasDownstreamHostConsumer()) 
                    return TRUE;
            }
        } else {
            OutputPort * pOPort = reinterpret_cast<OutputPort*>(pPort->GetInOutConsumer());
            if(pOPort != NULL && pOPort->HasDownstreamHostConsumer()) {
                return TRUE;
            }
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is block pool candidate. </summary>
    ///
    /// <remarks>   crossbac, 4/30/2013. </remarks>
    ///
    /// <returns>   true if block pool candidate, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    InitializerChannel::IsPagelockedBlockPoolCandidate(
        VOID
        )
    {
        InputPort * pPort = reinterpret_cast<InputPort*>(GetBoundPort(CE_DST));
        Task * pTask = pPort->GetTask();
        ACCELERATOR_CLASS accClass = pTask->GetAcceleratorClass();
        if(pPort->HasDependentAcceleratorBinding() || accClass != ACCELERATOR_CLASS_HOST) {
            OutputPort * pOPort = reinterpret_cast<OutputPort*>(pPort->GetInOutConsumer());
            if(pOPort != NULL) {
                if(pOPort->HasDownstreamHostConsumer()) {
                    return TRUE;
                }
            }
        }
        return FALSE;
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
    InitializerChannel::AllocateBlockPool(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )        
    {
        BOOL bSuccess = AllocateBlockPoolAsync(pAccelerators, uiPoolSize);
        if(bSuccess) {
            bSuccess &= FinalizeBlockPoolAsync();
        }
        return bSuccess;
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
    InitializerChannel::AllocateBlockPoolAsync(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )
    {
        if(!PTask::Runtime::GetBlockPoolsEnabled()) {
            PTask::Runtime::Inform("%s::%s: PTask block pools disabled--skipping allocation for pool on: %s\n",
                                   __FILE__,
                                   __FUNCTION__,
                                   m_lpszName);
            return FALSE;
        }

        // there are some common special cases where we can force the use of a very small pool 
        // that are very much worth handling because they reduce memory pressure. 
        // 1. If this channel is connected to an INPUT_PORT that does not have an in/out
        //    relationship with another port, we know there can only ever be one block in
        //    flight at a time: as soon as this task completes dispatch, the block
        //    should be released, putting it back in the pool. Moreover, the block is
        //    read-only, so in such a case we should be able to allocate once. Period.
        // 2. This block is connected to an INPUT_PORT *with* in/out semantics, but
        //    there is a simple cycle in the graph back from the out port to this is
        //    input port (a very common scenario in dbn). Again, in such a case, we know
        //    we'll only ever need one block.
        // In all other cases, we just take the default pool size for initializer channels. 
        // We *could* expend some additional effort to traverse the graph and find other situations
        // that impose an upper bound on the total number of possible in flight blocks, but
        // for now, just handle the low hanging fruit. 

        uiPoolSize = (uiPoolSize == 0) ? Runtime::GetDefaultInitChannelBlockPoolSize() : uiPoolSize;
        if(uiPoolSize > 1) {

            // we're looking for opportunities to constrain the channel size.
            // if the graph construction code has already forced it below what we
            // will ever use as a minimum value there is no sense doing all this work.
            BOOL bReadonlyInputBinding = FALSE;
            BOOL bInoutCycleBinding = FALSE;
            InputPort * pInputPort = reinterpret_cast<InputPort*>(GetBoundPort(CE_DST));
            OutputPort * pInoutConsumer = reinterpret_cast<OutputPort*>(pInputPort->GetInOutConsumer());
            bReadonlyInputBinding = pInoutConsumer == NULL;

            if(pInoutConsumer) {
                
                // look for a cycle that implies this initializer channel is 
                // producing blocks that carry long-lived shared mutable state: generally speaking, in
                // such cases, there is no sense pre-allocating more than one block for this pool. Detecting
                // when to limit the pool is complicated by a few factors, but a good heuristic is
                // to check whether the port is bound to multiple channels which have predicates
                // with opposite sense. 
                
                Channel * pPrimary = pInputPort->GetChannelCount() ? pInputPort->GetChannel(0) : NULL;
                Channel * pControl = pInputPort->GetControlChannelCount() ? pInputPort->GetControlChannel(0) : NULL;
                bInoutCycleBinding = (pPrimary != NULL && pControl != NULL);
            }

            if((bReadonlyInputBinding || bInoutCycleBinding) && (m_uiBlockTransitLimit == 0)) {
                // only one block can ever be in flight if it comes
                // from this channel. Constrain the pool size. 
                UINT uiNewPoolSize = 1;                
                PTask::Runtime::Inform("Constraining block pool on %s (%d->%d) based on graph structure\n",
                                       m_lpszName, 
                                       uiPoolSize,
                                       uiNewPoolSize);
                uiPoolSize = uiNewPoolSize;
            }
        }

        BOOL bPageLock = FALSE;
        BOOL bPageLockPossible = FALSE;
        BUFFERACCESSFLAGS ePermissions = PT_ACCESS_DEFAULT;
        InputPort * pPort = reinterpret_cast<InputPort*>(GetBoundPort(CE_DST));
        Task * pTask = pPort->GetTask();
        ACCELERATOR_CLASS accClass = pTask->GetAcceleratorClass();

        // is page lock support available in any of the accelerators
        // we care about? If not there is no sense trying to infer whether
        // we can use it profitably based on the graph structure.

        std::vector<Accelerator*>::iterator ti;
        for(ti=pAccelerators->begin(); ti!=pAccelerators->end(); ti++) {
            Accelerator * pAccelerator = *ti;
            bPageLockPossible |= pAccelerator->SupportsPinnedHostMemory();
        }

        // look at the semantics of the bound port and downstream
        // graph to figure out the block permissions we need, and to
        // decide if we can benefit from allocating blocks backed by pinned memory,
        // assuming it is actually supported.

        if(pPort->HasDependentAcceleratorBinding() || accClass != ACCELERATOR_CLASS_HOST) {
            ePermissions = PT_ACCESS_ACCELERATOR_READ;
            OutputPort * pOPort = reinterpret_cast<OutputPort*>(pPort->GetInOutConsumer());
            if(pOPort != NULL) {
                ePermissions |= PT_ACCESS_ACCELERATOR_WRITE;
                if(pOPort->HasDownstreamHostConsumer()) {
                    ePermissions |= PT_ACCESS_HOST_READ;
                    bPageLock |= bPageLockPossible;
                }
            }
        }

        ePermissions |= PT_ACCESS_HOST_WRITE;
        ePermissions |= PT_ACCESS_POOLED_HINT;
        m_pBlockPool = new BlockPool(m_pTemplate, ePermissions, uiPoolSize, this);
        m_pBlockPool->SetEagerDeviceMaterialize(TRUE);
        m_pBlockPool->SetRequestsPageLocked(bPageLock);
        return m_pBlockPool->AllocateBlockPoolAsync(pAccelerators, uiPoolSize);
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
    InitializerChannel::FinalizeBlockPoolAsync(
        VOID
        )
    {
        if(m_pBlockPool) 
            return m_pBlockPool->FinalizeBlockPoolAsync();
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
    InitializerChannel::AddNewBlock(
        Datablock * pBlock
        )
    {
        Lock();
        if(m_pBlockPool)
            m_pBlockPool->AddNewBlock(pBlock);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets high water mark. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    InitializerChannel::GetHighWaterMark(
        VOID
        )
    {
        if(m_pBlockPool == NULL) 
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
    InitializerChannel::GetOwnedBlockCount(
        VOID
        )
    {
        if(m_pBlockPool == NULL) 
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
    InitializerChannel::GetLowWaterMark(
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
    InitializerChannel::GetAvailableBlockCount(
        VOID
        )
    {
        if(m_pBlockPool == NULL) 
            return 0;
        return m_pBlockPool->GetAvailableBlockCount();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return a block to the pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    InitializerChannel::ReturnToPool(
        Datablock * pBlock
        )
    {
        Lock();
        if(m_pBlockPool)
            m_pBlockPool->ReturnBlock(pBlock);
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
    InitializerChannel::BlockPoolIsGlobal(
        void
        )
    {
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   gets the pool size. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    UINT
    InitializerChannel::GetPoolSize(
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
    InitializerChannel::SetRequestsPageLocked(
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
    InitializerChannel::GetRequestsPageLocked(
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
    InitializerChannel::GetBlockFromPool(
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
    InitializerChannel::HasDownstreamWriters(
        VOID
        )
    {
        assert(m_pDstPort != NULL);
        if(m_pDstPort == NULL) return FALSE;
        return (m_pDstPort->IsInOutParameter() || m_pDstPort->IsDestructive());
    }
};
