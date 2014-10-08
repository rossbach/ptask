//--------------------------------------------------------------------------------------
// File: MetaPort.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "AsyncContext.h"
#include "MetaPort.h"
#include "OutputPort.h"
#include "assert.h"
#include "task.h"
#include "PTaskRuntime.h"
#include "MemorySpace.h"
#include "signalprofiler.h"
#include <vector>

using namespace std;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    MetaPort::MetaPort(
        VOID
        ) 
    {
        m_uiId = NULL;
        m_pTemplate = NULL;
        m_ePortType = META_PORT;
        m_pAllocationPort = NULL;
        m_eMetaFunction = MF_ALLOCATION_SIZE; // default
        m_pGeneralIterationBlock = NULL;
        m_nGeneralIterationCount = 0;
        m_nGeneralIterationMax = 0;
        m_pCollaborativeMetaAllocator = NULL;
        m_pCollaborativeTemplateAllocator = NULL;
        m_uiAllocHint = 0;
        m_bForceAllocHint = FALSE;
        m_bDispatchDimensionsHint = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    MetaPort::~MetaPort() {
        if(m_pGeneralIterationBlock) 
            m_pGeneralIterationBlock->Release();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an allocation hint. </summary>
    ///
    /// <remarks>   crossbac, 8/21/2013. </remarks>
    ///
    /// <param name="uiAllocationHint"> The allocation hint. </param>
    /// <param name="bForceAllocHint">  true to force allocate hint. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    MetaPort::SetAllocationHint(
        __in UINT uiAllocationHint,
        __in BOOL bForceAllocHint
        )
    {
        Lock();
        m_uiAllocHint = uiAllocationHint;
        m_bForceAllocHint = bForceAllocHint;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this port has been configured with a statically known allocation size. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 8/21/2013. </remarks>
    ///
    /// <returns>   true if static allocation size, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    MetaPort::IsStaticAllocationSize(
        void
        )
    {
        Lock();
        BOOL bResult = m_uiAllocHint && m_bForceAllocHint;
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets an integer value from a block consumed from this MetaPort. Should not be
    ///             called when the port is unoccupied because it will block on a Pull call. On exit,
    ///             bControlBlock is TRUE if the consumed block carried a control signal;
    ///             uiControlCode will be set accordingly if this is the case. The integer value can
    ///             be used by iteration control or output allocation meta functions.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="bControlBlock">    [out] True on exit if the block pulled to compute the
    ///                                 allocation size carried a control signal. </param>
    /// <param name="luiControlCode">   [out] If the block pulled to compute the allocation size
    ///                                 carried a control signal, the control code from that block. </param>
    ///
    /// <returns>   The integer value at offset 0 in the datablock's data channel. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    MetaPort::GetIntegerValue(
        BOOL &bControlBlock,
        CONTROLSIGNAL &luiControlCode
        )
    {
        UINT cbResult = 0;
        Lock();
        assert(IsOccupied());
        Datablock * pBlock = this->Pull();
        assert(pBlock != NULL);
        Unlock();
        pBlock->Lock();
        luiControlCode = pBlock->GetControlSignals();
        bControlBlock = HASSIGNAL(luiControlCode);
        if(m_bForceAllocHint && m_uiAllocHint != 0) {
            cbResult = m_uiAllocHint;
#ifdef DEBUG
            UINT * pInteger = (UINT*) pBlock->GetDataPointer(FALSE);
            UINT cbCheck = *pInteger;
            assert(cbResult == cbCheck);
#endif
        } else {
            UINT * pInteger = (UINT*) pBlock->GetDataPointer(FALSE);
            cbResult = *pInteger;
        }
        pBlock->Unlock();
        pBlock->Release();
        return cbResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the port is occupied. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <returns>   true if occupied, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    MetaPort::IsOccupied(
        VOID
        ) 
    {
        BOOL bResult = FALSE;
        Lock();
        if(m_bPermanentBlock) {
            assert(m_vChannels.size() == 0);
            assert(m_vControlChannels.size() == 0);
            assert(m_eMetaFunction == MF_ALLOCATION_SIZE);
            assert(m_pReplayableBlock != NULL);
            bResult = m_pReplayableBlock != NULL;
        } else {
            assert(m_vChannels.size() <= 1);
            assert(m_vControlChannels.size() <= 1);
            if(m_pGeneralIterationBlock != NULL) {
                bResult = TRUE;
            } else {
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
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pulls the port. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *		
    MetaPort::Pull(
        VOID
        ) 
    {
        Datablock * pBlock = NULL;
        Lock();
        assert(m_vChannels.size() <= 1 || m_bPermanentBlock);
        assert(m_vControlChannels.size() <= 1 || m_bPermanentBlock);
        if(m_bPermanentBlock) {
            assert(m_pReplayableBlock != NULL);
            assert(m_vChannels.size() == 0);
            assert(m_vControlChannels.size() == 0);
            pBlock = m_pReplayableBlock;
            pBlock->AddRef();
        } else if(m_pGeneralIterationBlock) {
            assert(!m_bPermanentBlock);
            pBlock = m_pGeneralIterationBlock;
        } else {
            if(m_vControlChannels.size() > 0) {
                pBlock = m_vControlChannels[0]->Peek();
                if(pBlock) {
                    pBlock = m_vControlChannels[0]->Pull();
                }
            }
            if(pBlock == NULL) {
                if(m_vChannels.size() > 0) {
                    pBlock = m_vChannels[0]->Peek();
                    if(pBlock) {
                        pBlock = m_vChannels[0]->Pull();
                    }
                }
            }
            if(m_bSticky) {
                if(m_pReplayableBlock != NULL) {
                    if(pBlock) {
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
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Returns the top-of-stack object without removing it. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the current top-of-stack object. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *		
    MetaPort::Peek(
        VOID
        ) 
    {
        Datablock * pBlock = NULL;
        Lock();
        if(m_bPermanentBlock) {
            assert(m_bSticky);
            assert(m_pReplayableBlock != NULL);
            assert(m_vChannels.size() == 0);
            assert(m_vControlChannels.size() == 0);
            pBlock = m_pReplayableBlock;
        } else if(m_pGeneralIterationBlock) {
            pBlock = m_pGeneralIterationBlock;
        } else {
            if(m_vControlChannels.size() && m_vControlChannels[0]->IsReady()) {
                pBlock = m_vControlChannels[0]->Peek();
            }
            if(pBlock == NULL) {
                if(m_vChannels.size() != 0) {
                    pBlock = m_vChannels[0]->Peek();
                }
            }
            if(m_bSticky && !pBlock && m_pReplayableBlock != NULL) {
                pBlock = m_pReplayableBlock;
            }
        }
        Unlock();
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pushes an object into this port. FIXME: todo: this should be removed. It pushes a
    ///             block into the upstream channel for this port, and doesn't seem to care whether
    ///             it uses the control channel or not. For now, just assert in debug mode.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the Datablock* to push. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL		
    MetaPort::Push(
        __in Datablock* pBlock
        ) 
    {
        assert(FALSE);
        UNREFERENCED_PARAMETER(pBlock);
        PTask::Runtime::HandleError("%s: Attempt to push a block into a MetaPort(%s)!\n",
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
    MetaPort::SetPermanentBlock(
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
        assert(m_eMetaFunction == MF_ALLOCATION_SIZE);
        assert(pBlock != NULL);
        m_bPermanentBlock = TRUE;
        m_pReplayableBlock = pBlock;
        m_pReplayableBlock->AddRef();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates this object. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="pTemplate">            [in,out] If non-null, the template. </param>
    /// <param name="uiId">                 The identifier. </param>
    /// <param name="lpszVariableBinding">  [in] If non-null, the variable binding. </param>
    /// <param name="nParmIdx">             Zero-based index of the n parm. </param>
    /// <param name="nInOutRouteIdx">       Zero-based index of the n in out route. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port*
    MetaPort::Create(
        DatablockTemplate * pTemplate,
        UINT uiId, 
        char * lpszVariableBinding,
        int nParmIdx,
        int nInOutRouteIdx
        )
    {
        MetaPort * pPort = new MetaPort();
        if(SUCCEEDED(pPort->Initialize(pTemplate, uiId, lpszVariableBinding, nParmIdx, nInOutRouteIdx)))
            return pPort;
        delete pPort;
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a destination buffer. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="pAccelerator"> (optional) [in] If non-null, an accelerator object to assist
    ///                             creating a datablock if none is available. </param>
    ///
    /// <returns>   null if it fails, else the destination buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *
    MetaPort::GetDestinationBuffer(
        Accelerator * pAccelerator
        ) 
    {
        UNREFERENCED_PARAMETER(pAccelerator);
        assert(FALSE);
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a destination buffer. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    MetaPort::SetDestinationBuffer(
        Datablock * 
        ) 
    {
        assert(FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an allocation port. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    MetaPort::SetAllocationPort(
        Port * pPort
        )
    {
        Lock();
        m_pAllocationPort = pPort;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the allocation port. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the allocation port. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port *          
    MetaPort::GetAllocationPort()
    {
        Port * pResult = NULL;
        pResult = m_pAllocationPort;
        return pResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind control channel. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    MetaPort::BindControlChannel(
        Channel * pChannel
        ) 
    {
        Lock();
        assert(!m_bPermanentBlock);
        if(!m_bPermanentBlock) {
            m_vControlChannels.push_back(pChannel);
            pChannel->AddRef();
            assert(m_vControlChannels.size() == 1);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind control channel. </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    MetaPort::UnbindControlChannel(
        VOID
        ) 
    {
        Lock();
        assert(!m_bPermanentBlock);
        size_t nChannels = m_vControlChannels.size();
        if(nChannels > 0) {
            assert(nChannels == 1);
            m_vControlChannels[0]->Release();
            m_vChannels.clear();
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a meta function. </summary>
    ///
    /// <remarks>   Crossbac, 1/10/2012. </remarks>
    ///
    /// <param name="eMetaFunctionSpecifier">   Information describing the meta function. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    MetaPort::SetMetaFunction(
        METAFUNCTION eMetaFunctionSpecifier
        )
    {
        m_eMetaFunction = eMetaFunctionSpecifier; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Perform the work associated with this port's meta function. For example, if the
    ///             port is an allocator, allocate a block for the downstream output port. If it is
    ///             an iterator, set the iteration count on the Task.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/10/2012. </remarks>
    ///
    /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    MetaPort::PerformMetaFunction(
        Accelerator * pDispatchAccelerator
        )
    {
        switch(m_eMetaFunction) {
        case MF_ALLOCATION_SIZE:
            PerformAllocation(pDispatchAccelerator);
            break;
        case MF_SIMPLE_ITERATOR:
            ConfigureSimpleIteration();
            break;
        case MF_GENERAL_ITERATOR:
            ConfigureGeneralIteration();
            break;
        case MF_NONE:
        case MF_USER_DEFINED:
            assert(FALSE);
            break;
        } 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Perform any post-dispatch work associated with this port's meta function. For
    /// 			example, if the port is an iteration construct, reset the loop bounds and 
    /// 			propagate any control signals associated with the iteration. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/10/2012. </remarks>
    ///
    /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    MetaPort::FinalizeMetaFunction(
        Accelerator * pDispatchAccelerator
        )
    {
        UNREFERENCED_PARAMETER(pDispatchAccelerator);
        switch(m_eMetaFunction) {
        case MF_ALLOCATION_SIZE:
            // nothing to do
            break;
        case MF_SIMPLE_ITERATOR:
            // nothing to do
            break;
        case MF_GENERAL_ITERATOR:
            FinalizeGeneralIteration();
            break;
        case MF_NONE:
        case MF_USER_DEFINED:
            assert(FALSE);
            break;
        } 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an iteration target to the list. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    MetaPort::BindIterationTarget(
        Port * pPort
        )
    {
        Lock();
        assert(!m_bPermanentBlock);
        assert(m_eMetaFunction == MF_GENERAL_ITERATOR);
        assert(pPort->GetPortType() == INPUT_PORT);
        if((m_eMetaFunction == MF_GENERAL_ITERATOR) && (pPort->GetPortType() == INPUT_PORT)) {
            m_vIterationTargets.push_back(pPort);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the meta function. </summary>
    ///
    /// <remarks>   Crossbac, 1/10/2012. </remarks>
    ///
    /// <returns>   The meta function. </returns>
    ///-------------------------------------------------------------------------------------------------

    METAFUNCTION    
    MetaPort::GetMetaFunction(
        VOID
        )
    {
        return m_eMetaFunction;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Searches for collaborating meta ports: if this port is an allocator
    ///             for output ports with descriptor ports, block allocation may have 
    ///             dependences on other meta ports for the bound task. We need to know this
    ///             at dispatch time, but it is a static property of the graph, so
    ///             we pre-compute it as a side-effect of OnGraphComplete(). 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 2/15/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    MetaPort::FindCollaboratingMetaPorts(
        VOID
        )
    {
        // performed before the graph is running: no lock needed
        if(m_eMetaFunction == MF_ALLOCATION_SIZE) {
            // stash pointers to other metaports 
            OutputPort * pOPort = reinterpret_cast<OutputPort*>(m_pAllocationPort);
            m_pCollaborativeMetaAllocator = pOPort->GetDescriptorPort(DF_METADATA_SOURCE);
            m_pCollaborativeTemplateAllocator = pOPort->GetDescriptorPort(DF_TEMPLATE_SOURCE);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the channel allocation size when this meta port is an allocator for an
    ///             output port with descriptor ports (meaning another meta port is responsible for
    ///             computing that allocation size). If this meta port is not involved in such a
    ///             graph structure, return 0.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 2/15/2013. </remarks>
    ///
    /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
    /// <param name="eFunc">                The function. </param>
    /// <param name="ppPortTemplate">       [out] on exit the template for the related collaborative
    ///                                     port, if one is available. These are needed when initial
    ///                                     values are supplied by the template. </param>
    ///
    /// <returns>   The meta buffer allocation size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    MetaPort::GetCollaborativeAllocationSize(
        __in  Accelerator *        pDispatchAccelerator, 
        __in  DESCRIPTORFUNC       eFunc,
        __out DatablockTemplate ** ppPortTemplate
        )
    {
        OutputPort * pCollaborator = NULL;
        *ppPortTemplate = NULL;
        if(eFunc == DF_METADATA_SOURCE && m_pCollaborativeMetaAllocator != NULL) {
            pCollaborator = reinterpret_cast<OutputPort*>(m_pCollaborativeMetaAllocator);
            *ppPortTemplate = pCollaborator->GetTemplate();
            return pCollaborator->GetPendingAllocationSize(pDispatchAccelerator);
        } 
        if(eFunc == DF_TEMPLATE_SOURCE && m_pCollaborativeTemplateAllocator != NULL) {
            pCollaborator = reinterpret_cast<OutputPort*>(m_pCollaborativeTemplateAllocator);
            *ppPortTemplate = pCollaborator->GetTemplate();
            return pCollaborator->GetPendingAllocationSize(pDispatchAccelerator);
        }
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Finalize collaborative allocations. If this port has completed a collaborative
    ///             allocation (where other meta ports determine meta/template channel sizes)
    ///             we need to finish the binding of an output block at those ports. </summary>
    ///
    /// <remarks>   Crossbac, 2/15/2013. </remarks>
    ///
    /// <param name="pDispatchAccelerator"> [in] non-null, the dispatch accelerator. </param>
    /// <param name="pBlock">               [in,out] non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    MetaPort::FinalizeCollaborativeAllocations(
        __in    Accelerator * pDispatchAccelerator,
        __inout Datablock *   pBlock
        )
    {
        OutputPort * pCollaborator = NULL;
        if(m_pCollaborativeMetaAllocator != NULL) {
            pCollaborator = reinterpret_cast<OutputPort*>(m_pCollaborativeMetaAllocator);
            pCollaborator->CompletePendingAllocation(pDispatchAccelerator, pBlock);
        } 
        if(m_pCollaborativeTemplateAllocator != NULL) {
            pCollaborator = reinterpret_cast<OutputPort*>(m_pCollaborativeTemplateAllocator);
            pCollaborator->CompletePendingAllocation(pDispatchAccelerator, pBlock);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Perform allocation.  In this case, a datablock on a metaport provides an integer-
    ///             valued allocation size for another output port on the ptask. Hence, this function
    ///             looks at all metaports, and performs output datablock allocation as needed.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/10/2012. </remarks>
    ///
    /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    ///-------------------------------------------------------------------------------------------------

    void                    
    MetaPort::PerformAllocation(
        Accelerator * pDispatchAccelerator
        )
    {
        assert(!IsFormalParameter());
        OutputPort * pAllocationPort = (OutputPort*) GetAllocationPort();
        assert(pAllocationPort != NULL);
        assert(pAllocationPort->GetPortType() == OUTPUT_PORT);
        Task * pTask = GetTask();
        AsyncContext * pAsyncContext = pTask->GetOperationAsyncContext(pDispatchAccelerator, ASYNCCTXT_TASK);
            
        // figure out what size we will be allocating. The call pPort->GetIntegerValue will pull the
        // input block from the port and interpret its contents as a single 4-byte integer specifying
        // the size block to allocate. Since the metaport can also carry control information, we need
        // to be sure to propagate that information if it exists. 

        BOOL bPropagateControl;
        CONTROLSIGNAL luiBlockControlCode;
        UINT uiDestElements = GetIntegerValue(bPropagateControl, luiBlockControlCode);
        if(bPropagateControl) {

            // if this block is marked with control data, then 
            // we need to open any output ports that
            // are gated by this input port.
            SignalGatedPorts();
            PropagateControlSignal(luiBlockControlCode);
        }

        Datablock * pDestBlock = NULL;
        UINT cbDestStride = 1;
        BOOL bCheckPooledBlocks = TRUE;        
        DatablockTemplate * pDestTemplate = pAllocationPort->GetTemplate();
        DatablockTemplate * pMetaTemplate = NULL;
        DatablockTemplate * pTemplateTemplate = NULL;

        if(pDestTemplate != NULL) {
            cbDestStride = pDestTemplate->GetStride();
        }

        UINT uiDestBytes = cbDestStride * uiDestElements;
        UINT uiMetaBytes = 0;       // common case, may be updated below for described ports
        UINT uiTemplateBytes = 0;   // common case, may be updated below for described ports

        if(pAllocationPort->IsDescriptorPort()) {

            // if this port is a descriptor port for another output port, the described port is responsible
            // for allocating the actual block. So the descriptor just stashes the value and returns so the
            // actual block is allocated later on a different  port. We guarantee this always works by
            // visiting allocator ports in an order such that descriptor ports always precide described
            // ports at Bind-for-dispatch time. 
            // 
            pAllocationPort->SetPendingAllocationSize(pDispatchAccelerator, uiDestBytes);
            return;

        } else {
            
            // if we are allocating for a port with descriptor ports which have
            // metaport allocators, other output ports will have stashed the channel sizes 
            // we need to complete the allocation. Retrieve them.
            
            uiMetaBytes = GetCollaborativeAllocationSize(pDispatchAccelerator, DF_METADATA_SOURCE, &pMetaTemplate);
            uiTemplateBytes = GetCollaborativeAllocationSize(pDispatchAccelerator, DF_TEMPLATE_SOURCE, &pTemplateTemplate);
        }

        BOOL bCollaborativeAllocation = (uiMetaBytes != 0 || uiTemplateBytes != 0);

        if(bCheckPooledBlocks) {

            // if the output port has block pool hints it may have a pooled block
            // that we can actually use. If this call returns a non-null block,
            // it is already set as the destination buffer on return. 
            
            pDestBlock = pAllocationPort->GetPooledDestinationBuffer(pDispatchAccelerator, 
                                                                     uiDestBytes, 
                                                                     uiMetaBytes, 
                                                                     uiTemplateBytes);

        }
        if(pDestBlock != NULL) {

            // we were able to allocate from the output port's block pool despite the 
            // fact that we are trying to allocate something with a dynamic size. This means
            // the block came from a pool forced by the programmer, and the block is conservatively
            // sized: consequently, we need to seal it to the dimensions we actually want here. 
            pDestBlock->Lock();
            pDestBlock->Seal(uiDestElements, uiDestBytes, uiMetaBytes, uiTemplateBytes);
            pDestBlock->Unlock();

        } else {

            if(bCollaborativeAllocation) {

                // we are allocating multiple channels based on allocation data received on multiple meta
                // ports. Make sure the target we create has all the required buffers. The call will seal the
                // block. the use of a collaborative port indicates that different channels in the block are
                // written by different ports. In particular, the contract for such blocks is that there is an
                // entry per logical record in the meta channel (which is used to deserialize the records when
                // the block is consumed). Consequently, the number of records is not the number of objects in
                // the data channel (which may be of variable length and typically is), but the number of
                // objects in the meta-data channel.                 

                UINT uiRecordCount = pMetaTemplate == NULL ? 1 : (uiMetaBytes / pMetaTemplate->GetStride());
                
                pDestBlock = Datablock::CreateDestinationBlock(pDispatchAccelerator,
                                                               pAsyncContext,
                                                               pDestTemplate, 
                                                               pMetaTemplate,
                                                               pTemplateTemplate,
                                                               uiRecordCount,
                                                               uiDestBytes,
                                                               uiMetaBytes,
                                                               uiTemplateBytes,
                                                               PT_ACCESS_ACCELERATOR_WRITE,
                                                               FALSE);

            } else {

                // this block is being allocated for an output port that
                // has no descriptor ports, and requires only a data channel
                // buffer: use a simple allocator, no seal call is required. 
                
                pDestBlock = Datablock::CreateDestinationBlock(pDispatchAccelerator,
                                                               pAsyncContext,
                                                               pDestTemplate, 
                                                               uiDestElements,
                                                               PT_ACCESS_ACCELERATOR_WRITE,
                                                               FALSE);
            }

            // set this block as the target for the 
            pAllocationPort->SetDestinationBuffer(pDestBlock, FALSE);
            pDestBlock->Release();
        }

        if(bCollaborativeAllocation) {

            // we have allocating multiple channels based on allocation data
            // received on multiple meta ports. This means that other output ports
            // have a dependency on other channels in the destination block. 
            // FinalizeCollaborativeAllocations ensures that those
            // ports actually have their destination buffers set as well. 
            
            FinalizeCollaborativeAllocations(pDispatchAccelerator, pDestBlock);
        }

    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Configure simple iteration. Simple iteration is distinguished from general
    ///             iteration because it involves iterative invocation of a single PTask node. The
    ///             mechanisms required to build this are so much simpler than those required to
    ///             build general iteration over arbitrary subgraphs that it is worth bothering to
    ///             distinguish the case. Here, the datablock recieved on this port contains an
    ///             integer-valued iteration count, which we set on the task directly. Task::Dispatch
    ///             is responsible for clearing the iteration count after dispatch.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/10/2012. </remarks>
    ///
    /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void                    
    MetaPort::ConfigureSimpleIteration(
        VOID
        )
    {
        // this port should not be a formal parameter on accelerator
        // code, and since this is an iterator, we had better not 
        // be bound to an allocation port. 
        assert(!m_bPermanentBlock);
        assert(!IsFormalParameter());
        assert(((OutputPort*) GetAllocationPort()) == NULL);
            
        // figure out how many iterations. The call pPort->GetIntegerValue will pull the
        // input block from the port and interpret its contents as a single 4-byte integer specifying
        // the size block to allocate. Since the metaport can also carry control information, we need
        // to be sure to propagate that information if it exists. 
        BOOL bPropagateControl;
        CONTROLSIGNAL luiBlockControlCode;
        UINT uiDispatchIterations = GetIntegerValue(bPropagateControl, luiBlockControlCode);
        if(bPropagateControl) {
            // if this block is marked with control data, then 
            // we need to open any output ports that
            // are gated by this input port.
            SignalGatedPorts();
            PropagateControlSignal(luiBlockControlCode);
        }

        Task * pTask = GetTask();
        assert(pTask != NULL);
        pTask->SetDispatchIterationCount(uiDispatchIterations);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Perform general iteration. </summary>
    ///
    /// <remarks>   Crossbac, 1/10/2012. </remarks>
    ///
    /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void                    
    MetaPort::ConfigureGeneralIteration(
        VOID
        )
    {
        // this port should not be a formal parameter on accelerator
        // code, and since this is an iterator, we had better not 
        // be bound to an allocation port. 
        assert(!m_bPermanentBlock);
        assert(!IsFormalParameter());
        assert(((OutputPort*) GetAllocationPort()) == NULL);

        if(m_pGeneralIterationBlock == NULL) {
            
            // initialize iterations. The call pPort->GetIntegerValue will pull the
            // input block from the port and interpret its contents as a single 4-byte integer specifying
            // the loop upper bound. Since the metaport can also carry control information, we need
            // to be sure to propagate that information if it exists. 
            Lock();
            assert(IsOccupied());
            m_pGeneralIterationBlock = Pull();
            assert(m_pGeneralIterationBlock != NULL);
            Unlock();
            m_pGeneralIterationBlock->Lock();
            CONTROLSIGNAL luiControlCode = m_pGeneralIterationBlock->GetControlSignals();
            BOOL bControlBlock = HASSIGNAL(luiControlCode);
            UINT * pInteger = (UINT*) m_pGeneralIterationBlock->GetDataPointer(FALSE);
            m_nGeneralIterationMax = *pInteger;
            if(m_nGeneralIterationMax == 0) {
                assert(FALSE);
                PTask::Runtime::HandleError("%s: "
                                            "PTask cannot handle iteration max of 0 because "
                                            "loop bounds are determined at dispatch time, "
                                            "making 1 the smallest iteration max\n",
                                            __FUNCTION__);
            }
            m_nGeneralIterationCount = 0;
            m_pGeneralIterationBlock->Unlock();

            // printf("%s: iter %d of %d\n", this->m_pBoundTask->GetTaskName(), m_nGeneralIterationCount, m_nGeneralIterationMax);

            if(bControlBlock) {
                // if this block is marked with control data other than iteration,
                // we are in deep trouble
                assert(false);
                SignalGatedPorts();
                PropagateControlSignal(luiControlCode);
            } else {
                PropagateControlSignal(DBCTLC_BEGINITERATION);
                std::vector<Port*>::iterator vi;
                for(vi=m_vIterationTargets.begin(); vi!=m_vIterationTargets.end(); vi++) {
                    Port * pPort = *vi;
                    if(this->IsSticky()) 
                        pPort->GetTask()->SignalPortStatusChange();
                    else
                        pPort->BeginIterationScope(m_nGeneralIterationMax);
                }
            }
        }

        if(m_pGeneralIterationBlock) {
            m_nGeneralIterationCount++;
            // printf("%s: iter %d of %d\n", this->m_pBoundTask->GetTaskName(), m_nGeneralIterationCount, m_nGeneralIterationMax);
            if(m_nGeneralIterationCount >= m_nGeneralIterationMax) {
                // this is the last invocation. 
                PropagateControlSignal(DBCTLC_ENDITERATION);
                m_pGeneralIterationBlock->Release();
                m_pGeneralIterationBlock = 0;
                m_nGeneralIterationCount = 0;
                m_nGeneralIterationMax = 0;
                std::vector<Port*>::iterator vi;
                for(vi=m_vIterationTargets.begin(); vi!=m_vIterationTargets.end(); vi++) {
                    Port * pPort = *vi;
                    pPort->EndIterationScope();
                }
            }
        } 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Configure iteration targets. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void            
    MetaPort::ConfigureIterationTargets(
        Datablock * pBlock
        )
    {
        // cjr: it is not necessary to lock the port here
        // because we are only touching this port's static structures.
        // Lock();
        assert(!m_bPermanentBlock);
        if(m_eMetaFunction == MF_GENERAL_ITERATOR) {
            pBlock->Lock();
            UINT * pInteger = (UINT*) pBlock->GetDataPointer(FALSE);
            UINT uiIterationMax = *pInteger;
            pBlock->Unlock();
            std::vector<Port*>::iterator vi;
            for(vi=m_vIterationTargets.begin(); vi!=m_vIterationTargets.end(); vi++) {
                Port * pPort = *vi;
                pPort->BeginIterationScope(uiIterationMax);
            }
        }
        //Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Post dispatch update hook for general iteration. </summary>
    ///
    /// <remarks>   Crossbac, 1/10/2012. </remarks>
    ///
    /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void                    
    MetaPort::FinalizeGeneralIteration(
        VOID
        )
    {
        // this port should not be a formal parameter on accelerator
        // code, and since this is an iterator, we had better not 
        // be bound to an allocation port. 
        assert(!m_bPermanentBlock);
        assert(!IsFormalParameter());
        assert(((OutputPort*) GetAllocationPort()) == NULL);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has block pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if block pool, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    MetaPort::HasBlockPool(
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
    MetaPort::ForceBlockPoolHint(
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
    MetaPort::AllocateBlockPool(
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
    MetaPort::DestroyBlockPool(
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
    MetaPort::IsBlockPoolActive(
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
    MetaPort::AllocateBlockPoolAsync(
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
    MetaPort::FinalizeBlockPoolAsync(
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
    MetaPort::AddNewBlock(
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
    MetaPort::ReturnToPool(
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
    MetaPort::GetPoolSize(
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
    MetaPort::SetRequestsPageLocked(
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
    MetaPort::GetRequestsPageLocked(
        VOID
        )
    {
        return FALSE;
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
    MetaPort::CheckTypeSpecificSemantics(
        std::ostream * pos,
        PTask::Graph * pGraph
        )
    {
        UNREFERENCED_PARAMETER(pGraph);
        BOOL bResult = TRUE;
        std::ostream& os = *pos;

        if(m_bPermanentBlock && 
           (m_eMetaFunction != MF_ALLOCATION_SIZE ||
           m_vChannels.size() != 0 ||
           m_vControlChannels.size() != 0)) {
            bResult = FALSE;
            os << this 
                << " is configured as a MetaPort with a permanent sticky block, which"
                << " requires it to be bound as an allocator port, with no upstream"
                << " channel connections: at least one of these conditions does not hold."
                << endl;
        }

        // first check that this port has all the parts it requires
        // to perform the meta-function being asked of it. 
        switch(m_eMetaFunction) {
        case MF_ALLOCATION_SIZE: 
            if(m_pAllocationPort == NULL) {
                bResult = FALSE;
                os << this 
                    << " is configured as an allocator MetaPort, but"
                    << " is not bound to an allocation output port."
                    << " What port should it be allocating for?"
                    << endl;
            }            
            break;
        case MF_GENERAL_ITERATOR:
            if(m_pControlPropagationPorts.size() == 0) {
                bResult = FALSE;
                os << this 
                    << " is configured as an iterator MetaPort, but"
                    << " is not bound to any control propagation ports."
                    << " The end iteration signal will have no effect."
                    << " did you mean to do this?"
                    << endl;
            }                        
            break;
        case MF_SIMPLE_ITERATOR:
            if(m_pAllocationPort != NULL) {
                bResult = FALSE;
                os << this 
                    << " is configured as an iterator MetaPort, but"
                    << " is also bound to an allocation output port."
                    << " No blocks will be pushed into the allocation port."
                    << endl;
            }                        
            break;
        case MF_NONE:
        case MF_USER_DEFINED:
            bResult = FALSE;
            os << this 
                << " is configured with a MetaPort meta-function that"
                << " is not yet implemented."
                << endl;
            break;
        } 

        // check that we have some channels 
        if(m_vChannels.size() == 0 && !m_bPermanentBlock) {
            bResult = FALSE;
            os << this 
                << "not bound to any channels. "
                << "Where will it get its input?"
                << endl;
        }

        vector<Channel*>::iterator vi;
        for(vi=m_vChannels.begin(); vi!=m_vChannels.end(); vi++) {
            // this port had better be not be attached directly to an output
            Channel * pChannel = *vi;
            if(pChannel->GetType() == CT_GRAPH_OUTPUT) {
                bResult = FALSE;
                os << this << "bound to an output channel!" << endl;
            }
        }
        return bResult;
    }

};
