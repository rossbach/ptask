//--------------------------------------------------------------------------------------
// File: Port.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "port.h"
#include "OutputPort.h"
#include "InputPort.h"
#include "task.h"
#include "assert.h"
#include "PTaskRuntime.h"
#include "graph.h"
#include "signalprofiler.h"
#include <string>

using namespace std;

namespace PTask {

    /// <summary>   The port m v releaseable sticky ports. </summary>
    std::map<Graph*, std::set<Port*>>  Port::m_vReleaseableStickyPorts;

    /// <summary>   The lock for releasable sticky ports. </summary>
    CRITICAL_SECTION                   Port::m_csReleasableStickyPorts;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    Port::Port(
        VOID
        ) : Lockable(NULL)
    {
        m_uiId = NULL;
        m_pGraph = NULL;
        m_pBoundTask = NULL;
        m_lpszVariableBinding = NULL;
        m_uiOriginalIndex = UNBOUND_PORT;
        m_uiBoundPortIndex = UNBOUND_PORT;
        m_pControlPropagationSource = NULL;
        m_pTemplate = NULL;
        m_luiPropagatedControlCode = DBCTLC_NONE;
        m_bSticky = FALSE;
        m_bDestructive = FALSE;
        m_pReplayableBlock = NULL;
        m_bMarshallable = TRUE;
        m_bCanStream = TRUE;
        m_uiStickyIterationUpperBound = 0;
        m_uiIterationUpperBound = 0;
        m_uiIterationIndex = 0;
        m_bIterated = FALSE;
        m_pIterationSource = NULL;
        m_bActiveIterationScope = FALSE;
        m_bSuppressClones = FALSE;
        m_bTriggerPort = FALSE;
        m_bScopeTerminus = FALSE;
        m_luiScopeTerminalSignal = DBCTLC_NONE;
        m_luiStickyReleaseSignal = DBCTLC_NONE;
        m_bStickyReleaseSignalConfigured = FALSE;      
        m_bPermanentBlock = FALSE;
        m_eGeometryDimBinding = GD_NONE;
        m_nDependentAcceleratorBinding = -1;
        m_pMandatoryAccelerator = NULL;
        m_accDependentAcceleratorClass = ACCELERATOR_CLASS_UNKNOWN;
        m_bHasUpstreamChannelPool = FALSE;
        m_bUpstreamChannelPoolGrowable = FALSE;
        m_uiUpstreamChannelPoolSize = 0;
        m_uiUpstreamChannelPoolGrowIncrement = 0;
        m_luiInitialPropagatedControlCode = DBCTLC_NONE;
        m_bDispatchDimensionsHint = TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    Port::~Port(
        VOID
        ) 
    {
        Lock();
        PTSRELEASE(m_pReplayableBlock);
        vector<Channel*>::iterator vi;
        for(vi=m_vChannels.begin(); vi!=m_vChannels.end(); vi++) {
            (*vi)->Drain();
            PTSRELEASE(*vi);
        }
        for(vi=m_vControlChannels.begin(); vi!=m_vControlChannels.end(); vi++) {
            (*vi)->Drain();
            PTSRELEASE(*vi);
        }
        if(m_pTemplate) {
            m_pTemplate->Release();
        }
        m_vChannels.clear();
        m_vControlChannels.clear();
        if(m_lpszVariableBinding)
            delete [] m_lpszVariableBinding;
        if(m_pBoundTask) 
            UnbindTask();
        std::vector<DEFERREDPORTDESC*>::iterator vdi;
        for(vdi=m_vDescriptorPorts.begin(); vdi!=m_vDescriptorPorts.end(); vdi++) {
            DEFERREDPORTDESC* pDesc = (*vdi);
            delete pDesc;
        }
        std::vector<DEFERREDCHANNELDESC*>::iterator dci;
        UnbindDeferredChannels();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes this object. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTemplate">                [in] If non-null, the datablock template. </param>
    /// <param name="uiId">                     An identifier for the port (programmer-supplied). </param>
    /// <param name="lpszBinding">              [in] If non-null, the variable binding. </param>
    /// <param name="nFormalParameterIndex">    Zero-based index of the n formal parameter. </param>
    /// <param name="nInOutRoutingIndex">       Zero-based index of the n in out routing. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT			
    Port::Initialize(
        DatablockTemplate * pTemplate, 
        UINT uiId,
        char * lpszBinding,
        int nFormalParameterIndex,
        int nInOutRoutingIndex
        )
    {
        Lock();
        if(lpszBinding) {
            size_t szlen = strlen(lpszBinding);
            m_lpszVariableBinding = new char[szlen+1];
            strcpy_s(m_lpszVariableBinding, szlen+1, lpszBinding);
        } 
        m_pTemplate = pTemplate;
        if(m_pTemplate != NULL) {
            m_pTemplate->AddRef();
        }
        m_uiId = uiId;
        m_nFormalParameterIndex = nFormalParameterIndex;
        m_nInOutRoutingIndex = nInOutRoutingIndex;
        Unlock();
        return S_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the port is occupied. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> none</param>
    ///
    /// <returns>   true if occupied, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Port::IsOccupied(
        VOID
        ) 
    {
        // only output ports can have multiple channels
        BOOL bResult = FALSE;
        Lock();
        size_t nChannels = m_vChannels.size();
        assert(nChannels == 0 || nChannels == 1);
        if(nChannels != 0) {
            Channel * pChannel = m_vChannels[0];
            bResult = pChannel->IsReady();
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind channel. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pChannel"> [in] non-null, the channel. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::BindChannel(
        Channel * pChannel
        ) 
    {
        Lock();
        assert(!m_bPermanentBlock);
        m_vChannels.push_back(pChannel);
        pChannel->AddRef();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind task. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pTask">        [in] non-null, the task. </param>
    /// <param name="uiPortIndex">  Zero-based index of the user interface port. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::BindTask(
        Task * pTask,
        UINT uiPortIndex
        ) 
    {
        Lock();
        if(m_pBoundTask != NULL)
            m_pBoundTask->Release();
        m_pBoundTask = pTask;
        m_pGraph = pTask->GetGraph();
        m_pBoundTask->AddRef();
        m_uiBoundPortIndex = uiPortIndex;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind channel. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="idx">  The index. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::UnbindChannel(
        int idx
        ) 
    {
        Lock();
        size_t nChannels = m_vChannels.size();
        assert(idx < nChannels);
        if((size_t) idx < nChannels) {
            m_vChannels[idx]->Release();
            m_vChannels.erase(m_vChannels.begin()+idx);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind task. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::UnbindTask(
        VOID
        ) 
    {
        Lock();
        if(m_pBoundTask != NULL)
            m_pBoundTask->Release();
        m_pBoundTask = NULL;
        m_pGraph = NULL;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pulls a datablock from this port. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *		
    Port::Pull() {
        // output ports are special in this regard
        assert(!m_bSticky);
        assert(!m_bPermanentBlock);
        Datablock * pBlock = NULL;
        Lock();
        assert(m_vChannels.size() <= 1);
        assert(m_vControlChannels.size() == 0);
        if(m_vChannels.size() != 0) {
            assert(m_vChannels[0]->GetType() != CT_INITIALIZER);
            pBlock = m_vChannels[0]->Pull();
        }
        Unlock();
        ctlpegress(this, pBlock);
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Helper, for channels that may have selectable inputs. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <param name="pChannels">    [in,out] If non-null, the channels. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Port::AttemptPull(
        std::vector<Channel*>* pChannels
        )
    {
        assert(LockIsHeld());
        Datablock * pBlock = NULL;
        if(pChannels->size() != 0) {
            Channel * pChannel = (*pChannels)[0];
            if(pChannel->GetType() == CT_INITIALIZER && pChannel->IsReady(CE_DST)) {
                pBlock = pChannel->Pull();
            } else {
                pBlock = pChannel->Peek();
                if(pBlock != NULL) {
                    pBlock = pChannel->Pull();                
                }
            }
            //if(pBlock != NULL) {
            //    std::cout << this << " pulled from " << pChannel << std::endl;
            //}
        }
        ctlpcondegress(pBlock, this, pBlock);
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Returns the datablock that would be returned by the next call to pull(), but without
    ///     removing it.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the current top-of-stack object. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *		
    Port::Peek() {
        Lock();
        Datablock * pBlock = NULL;
        assert(m_vChannels.size() <= 1);
        if(m_vChannels.size() != 0) {
            pBlock = m_vChannels[0]->Peek();
        }
        Unlock();
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pushes a datablock into this port. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the Datablock* to push. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL		
    Port::Push(
        __in Datablock* pBlock
        ) 
    {
        assert(FALSE);
        UNREFERENCED_PARAMETER(pBlock);
        PTask::Runtime::HandleError("%s called: this method is meaningless for the superclass!",
                                    __FUNCTION__);
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a channel. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="idx">  The index. </param>
    ///
    /// <returns>   null if it fails, else the channel. </returns>
    ///-------------------------------------------------------------------------------------------------

    Channel *
    Port::GetChannel(
        UINT idx
        )
    {
        return GetChannel(&m_vChannels, idx);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a control channel. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="idx">  The index. </param>
    ///
    /// <returns>   null if it fails, else the channel. </returns>
    ///-------------------------------------------------------------------------------------------------

    Channel *
    Port::GetControlChannel(
        UINT idx
        )
    {
        return GetChannel(&m_vControlChannels, idx);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this port is formal parameter for its bound task. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if formal parameter, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::IsFormalParameter(
        VOID
        )
    { 
        BOOL bResult = FALSE;
        Lock();
        if(GetPortType() != META_PORT) {
            bResult = m_nFormalParameterIndex != PT_DEFAULT_KERNEL_PARM_IDX; 
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this port is a scalar parameter for its bound task. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if scalar parameter, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::IsScalarParameter(
        VOID
        )
    { 
        switch(GetParameterType()) {
        case PTPARM_INT: return TRUE;
        case PTPARM_FLOAT: return TRUE;
        case PTPARM_BYVALSTRUCT: return TRUE;
        default: return FALSE;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind control channel. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pChannel"> [in] non-null, the channel. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::BindControlChannel(
        Channel * pChannel
        ) 
    {
        // subclasses must implement
        UNREFERENCED_PARAMETER(pChannel);
        assert(FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind control channel. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::UnbindControlChannel(
        VOID
        ) 
    {
        // subclasses must implement
        assert(FALSE);
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
    Port::BindDescriptorPort(
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
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is bound as descriptor port. </summary>
    ///
    /// <remarks>   crossbac, 4/19/2012. </remarks>
    ///
    /// <returns>   true if descriptor port, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::IsDescriptorPort(
        VOID
        )
    {
        Lock();
        BOOL bResult = m_vDescriptorPorts.size() > 0;
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind descriptor ports. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::UnbindDescriptorPorts(
        VOID
        ) 
    {
        Lock();
        size_t nPorts = m_vDescriptorPorts.size();
        if(nPorts > 0) {
            m_vDescriptorPorts.clear();
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a deferred channels. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the deferred channels. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::vector<DEFERREDCHANNELDESC*>*
    Port::GetDeferredChannels(
        VOID
        )
    {
        return &m_vDeferredChannels;            
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind deferred channel. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pChannel"> [in] non-null, the channel. </param>
    /// <param name="func">     The func. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::BindDeferredChannel(
        Channel * pChannel,
        DESCRIPTORFUNC func
        ) 
    {
        Lock();
        DEFERREDCHANNELDESC * desc = new DEFERREDCHANNELDESC();
        pChannel->AddRef();
        desc->pChannel = pChannel;
        desc->func = func;
        m_vDeferredChannels.push_back(desc);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind deferred channels. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::UnbindDeferredChannels(
        VOID
        ) 
    {
        Lock();
        std::vector<DEFERREDCHANNELDESC*>::iterator vi;
        for(vi=m_vDeferredChannels.begin(); vi!=m_vDeferredChannels.end(); vi++) {
            (*vi)->pChannel->Release();
            delete (*vi);
        }
        m_vDeferredChannels.clear();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a gated port. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="p">    [in] non-null, a the gated port. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::AddGatedPort(
        Port * p
        )
    {
        Lock();
        m_pGatedPorts.push_back(p);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a control propagation port. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="p">    [in] non-null, a the control propagation port. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::AddControlPropagationPort(
        Port * p
        )
    {
        Lock();
        m_pControlPropagationPorts.push_back(p);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a control propagation channel. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="p">    [in] non-null, a the control propagation port. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::AddControlPropagationChannel(
        Channel * pChannel
        )
    {
        Lock();
        m_pControlPropagationChannels.insert(pChannel);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets all gated ports. 
    /// 			You must lock the port to make this call.</summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the gated ports. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::vector<Port*>*  
    Port::GetGatedPorts(
        VOID
        )
    {
        assert(LockIsHeld());
        return &m_pGatedPorts;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the port has gated ports. 
    /// 			You must lock the port to make this call.</summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if gated ports, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Port::HasGatedPorts(
        VOID
        )
    {
        assert(LockIsHeld());
        BOOL bResult = m_pGatedPorts.size() > 0;
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Raises a Gated ports signal. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::SignalGatedPorts(
        VOID
        )
    {
        Lock();
        // if this block is marked EOF, then 
        // we need to open any output ports that
        // are gated by this input port.        
        if(m_pGatedPorts.size()) {
            vector<Port*>::iterator vi;
            for(vi = m_pGatedPorts.begin(); vi!=m_pGatedPorts.end(); vi++) {
                OutputPort * pGatedPort = (OutputPort*)(*vi);
                pGatedPort->SignalGate();
            }
        }        
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an initial value for propagated control codes. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="uiCode">   The code. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::SetInitialPropagatedControlSignal(
        CONTROLSIGNAL uiCode
        ) 
    {
        Lock();        
        m_luiInitialPropagatedControlCode = uiCode; // stash so we can reset!
        SetPropagatedControlSignal(uiCode);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a propagated control code. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="uiCode">   The code. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::SetPropagatedControlSignal(
        CONTROLSIGNAL luiCode
        ) 
    {
        Lock();
        assert(GetPortType() == OUTPUT_PORT);
        // printf("setting uiCtlCode on port %s = %d\n", this->GetVariableBinding(), uiCode);
        if(m_pControlPropagationSource != NULL) {
            m_luiPropagatedControlCode = (m_luiPropagatedControlCode | luiCode);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resets the port. </summary>
    ///
    /// <remarks>   Crossbac, 5/2/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::Reset(
        VOID
        )
    {
        if(m_luiInitialPropagatedControlCode) {
            m_luiPropagatedControlCode = m_luiInitialPropagatedControlCode;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a propagated control code. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="uiCode">   The code. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::ClearPropagatedControlSignal(
        CONTROLSIGNAL luiCode
        ) 
    {
        Lock();
        assert(GetPortType() == OUTPUT_PORT);
        // printf("setting uiCtlCode on port %s = %d\n", this->GetVariableBinding(), uiCode);
        if(m_pControlPropagationSource != NULL)
            m_luiPropagatedControlCode = (m_luiPropagatedControlCode & (~luiCode));
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a propagated control code. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="uiCode">   The code. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::ClearAllPropagatedControlSignals(
        VOID
        ) 
    {
        Lock();
        assert(GetPortType() == OUTPUT_PORT);
        // printf("setting uiCtlCode on port %s = %d\n", this->GetVariableBinding(), uiCode);
        m_luiPropagatedControlCode = DBCTLC_NONE;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Propagate control code. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="uiCode">   The code. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::PropagateControlSignal(
        CONTROLSIGNAL luiCode
        ) 
    {
        Lock();
        assert(GetPortType() != OUTPUT_PORT);
        vector<Port*>::iterator vi;
        set<Channel*>::iterator ri;
        for(vi = m_pControlPropagationPorts.begin(); 
            vi != m_pControlPropagationPorts.end();  
            vi++) {
            (*vi)->SetPropagatedControlSignal(luiCode);
        }
        for(ri = m_pControlPropagationChannels.begin(); 
            ri != m_pControlPropagationChannels.end();  
            ri++) {
            (*ri)->Lock();
        }
        for(ri = m_pControlPropagationChannels.begin(); 
            ri != m_pControlPropagationChannels.end();  
            ri++) {
            (*ri)->SetPropagatedControlSignal(luiCode);
        }
        for(ri = m_pControlPropagationChannels.begin(); 
            ri != m_pControlPropagationChannels.end();  
            ri++) {
            (*ri)->Unlock();
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the port's parameter type. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The parameter type. </returns>
    ///-------------------------------------------------------------------------------------------------

    PTASK_PARM_TYPE 
    Port::GetParameterType(
        VOID
        ) 
    { 
        return m_pTemplate ? m_pTemplate->GetParameterBaseType() : PTPARM_NONE; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the sticky property of the port. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="bSticky">  true to sticky. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::SetSticky(
        BOOL bSticky
        )
    {
        Lock();
        switch(GetPortType()) {
        case INPUT_PORT:
        case STICKY_PORT:
        case META_PORT:
            m_bSticky = bSticky;
            break;
        default:
            assert(false && "only sticky ports, meta ports, and input ports can be sticky");
            break;
        }
        // std::cout << this << " is sticky. " << std::endl;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a control signal which allows a sticky port (either of class "StickyPort",
    ///             or another class with the sticky property set) to release its sticky datablock
    ///             safely. Being able to release such blocks without cleaning up the entire graph
    ///             is an important memory optimization for some Dandelion workloads running
    ///             at scale. </summary>
    ///
    /// <remarks>   crossbac, 9/10/2013. </remarks>
    ///
    /// <param name="luiControlSignal"> The lui control signal. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::SetStickyReleaseSignal(
        CONTROLSIGNAL luiControlSignal
        )
    {
        Lock();
        m_luiStickyReleaseSignal = luiControlSignal;
        m_bStickyReleaseSignalConfigured = TRUE;        
        Unlock();
        EnterCriticalSection(&m_csReleasableStickyPorts);
        m_vReleaseableStickyPorts[m_pGraph].insert(this);
        LeaveCriticalSection(&m_csReleasableStickyPorts);
        
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has a sticky release signal configured. </summary>
    ///
    /// <remarks>   crossbac, 9/10/2013. </remarks>
    ///
    /// <returns>   true if sticky release signal, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::HasStickyReleaseSignal(
        void
        )
    {
        return m_bStickyReleaseSignalConfigured && m_luiStickyReleaseSignal != DBCTLC_NONE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the port is sticky. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if sticky, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::IsSticky(
        VOID
        )
    {
        return m_bSticky;
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
    Port::SetPermanentBlock(
        Datablock * p
        )
    {
        // if the subclass hasn't overridden this member, 
        // then it is being called on a port type for which
        // the API is not meaningful.
        UNREFERENCED_PARAMETER(p);
        assert(FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Mark the port destructive (or non-destructive). </summary>
    ///
    /// <remarks>   Crossbac, 5/8/2012. </remarks>
    ///
    /// <param name="bDestructive"> true to destructive. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::SetDestructive(
        BOOL bDestructive
        )
    {
        m_bDestructive = bDestructive;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this port is destructive. </summary>
    ///
    /// <remarks>   Crossbac, 5/8/2012. </remarks>
    ///
    /// <returns>   true if destructive, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Port::IsDestructive(
        VOID
        )
    {
        return m_bDestructive;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Query if this port works with Marshallable output. Unmarshallable data is any that
    ///     contains pointers to dynamically allocated device-side buffers, which are (by
    ///     construction) invisible to ptask. For example, a hash-table cannot be migrated because
    ///     pointers will be invalid on another device, and no facility exists to marshal the
    ///     hashtable by chasing the pointers and flattening the data structure. If PTask does not
    ///     know that a datablock is unmarshallable, migration will cause havoc.
    /// </summary>
    ///
    /// <remarks>   No lock is required for this call because marshallability
    /// 			should not be a dynamic property.
    /// 			Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   true if marshallable output, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::IsMarshallable(
        VOID
        )
    {
        return m_bMarshallable;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Marks this port as producing output that is marshallable. Unmarshallable data is any that
    ///     contains pointers to dynamically allocated device-side buffers, which are (by
    ///     construction) invisible to ptask. For example, a hash-table cannot be migrated because
    ///     pointers will be invalid on another device, and no facility exists to marshal the
    ///     hashtable by chasing the pointers and flattening the data structure. If PTask does not
    ///     know that a datablock is unmarshallable, migration will cause havoc.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="bOutputIsMarshallable">    true if port is marshallable. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::SetMarshallable(
        BOOL bIsMarshallable
        )
    {
        Lock();
        m_bMarshallable = bIsMarshallable;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the port is (or can be) connected to a data source or sink that can be
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
    Port::CanStream(
        VOID
        )
    {
        return m_bCanStream;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets whether the port is (or can be) connected to a data source or sink that can be
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
    /// <param name="bIsMarshallable">  true if port is marshallable. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::SetCanStream(
        BOOL bCanStream
        )
    {
        Lock();
        m_bCanStream = bCanStream;
        Unlock();
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the datablock template describing this port. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the template. </returns>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate * 
    Port::GetTemplate(
        VOID
        ) 
    { 
        return m_pTemplate; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the name of the ptask variable to which this port is bound. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the variable binding. </returns>
    ///-------------------------------------------------------------------------------------------------

    const char *	
    Port::GetVariableBinding(
        VOID
        ) 
    { 
        return m_lpszVariableBinding; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the index of this port in the array it was in when passed 
    ///             into AddTask. </summary>
    ///
    /// <remarks>   jcurrey, 5/6/2013. </remarks>
    ///
    /// <param name="index">   The original index. </returns>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::SetOriginalIndex(UINT index)
    {
        m_uiOriginalIndex = index;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the index of this port in the array it was in when passed into AddTask. </summary>
    ///
    /// <remarks>   jcurrey, 5/6/2013. </remarks>
    ///
    /// <returns>   The original index. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    Port::GetOriginalIndex()
    {
        return m_uiOriginalIndex;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the formal parameter index of this port. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The formal parameter index. </returns>
    ///-------------------------------------------------------------------------------------------------

    int             
    Port::GetFormalParameterIndex(
        VOID
        ) 
    { 
        return m_nFormalParameterIndex; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this port is part of an in/out parameter pair. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if in out parameter, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::IsInOutParameter(
        VOID
        ) 
    { 
        return m_nInOutRoutingIndex != PT_DEFAULT_INOUT_ROUTING_IDX; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the in out routing index for an in/out port pair. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The in out routing index. </returns>
    ///-------------------------------------------------------------------------------------------------

    int              
    Port::GetInOutRoutingIndex(
        VOID
        ) 
    { 
        return m_nInOutRoutingIndex; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a control propagation source for this port. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pControlPropagationSourcePort">    [in] non-null, a the source port. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::SetControlPropagationSource(
        Port * pControlPropagationSourcePort
        )
    { 
        m_pControlPropagationSource = pControlPropagationSourcePort; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the control propagation source for this port </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the control propagation source. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port*           
    Port::GetControlPropagationSource(
        VOID
        ) 
    { 
        return m_pControlPropagationSource; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the propagated control code. You must hold the port lock. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The propagated control code. </returns>
    ///-------------------------------------------------------------------------------------------------

    CONTROLSIGNAL          
    Port::GetPropagatedControlSignals(
        VOID
        ) 
    { 
        assert(LockIsHeld());
        return m_luiPropagatedControlCode; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the uid. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The uid. </returns>
    ///-------------------------------------------------------------------------------------------------

    const UINT			
    Port::GetUID(
        VOID
        ) 
    { 
        return m_uiId; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the task to which this port is bound </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the task. </returns>
    ///-------------------------------------------------------------------------------------------------

    Task *	
    Port::GetTask(
        VOID
        ) 
    { 
        return m_pBoundTask; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a port type. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <returns>   The port type. </returns>
    ///-------------------------------------------------------------------------------------------------

    PORTTYPE		
    Port::GetPortType(
        VOID
        ) 
    { 
        // no lock required--this can never change
        return m_ePortType; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a channel count. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <returns>   The channel count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT             
    Port::GetChannelCount(
        VOID
        ) 
    { 
        return GetChannelCount(&m_vChannels);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the control channel count. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <returns>   The channel count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT             
    Port::GetControlChannelCount(
        VOID
        ) 
    { 
        return GetChannelCount(&m_vControlChannels);    
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check semantics. Return true if all the structures are initialized for this port
    ///             in a way that is consistent with a well-formed graph.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="pGraph">   [in,out] non-null, the graph. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::CheckSemantics(
        std::ostream * pos,
        Graph * pGraph
        )
    {
        std::ostream& strWarning = *pos;
        BOOL bResult = TRUE;
        Lock();
        if(!m_bMarshallable) {
            // if the port has unmarshallable data, then the task
            // it bound too should also be marked in such a way.
            // if the port is an input and in/out unmarshallable, both ports 
            // need to marked as such.
            Task * pTask = GetTask();
            if(!pTask->ProducesUnmigratableData()) {
                bResult = FALSE;
                strWarning << this << " marked unmarshallable but bound task " 
                           << pTask << " not marked as producing unmigratable data"
                           << endl;
            }
        }
        if(m_bSticky) {
            // constant ports and input ports can have the sticky flag
            PORTTYPE portType = GetPortType();
            if(portType != STICKY_PORT && portType != INPUT_PORT && portType != META_PORT) {
                bResult = FALSE;
                strWarning << this << " marked sticky, which is not possible for ports of its type." << endl;
            }
        }

        if(m_nInOutRoutingIndex != PT_DEFAULT_INOUT_ROUTING_IDX) {
            // if we have an inout routing index other than -1, it means this port is part of an inout
            // pair. It had better be a port type that can produce mutable blocks, and it had better have
            // an inout consumer that also knows about it. Since inout is implemented by subclasses
            // we have to defer all but the type check to CheckTypeSpecificSemantics() calls.
            if(m_ePortType != INPUT_PORT && m_ePortType != INITIALIZER_PORT) {
                bResult = FALSE;
                strWarning << this << " marked in/out, which is not possible for ports of its type." << endl;
                
            }             
        }

        if(m_pBoundTask == NULL) {
            // ports are required to be bound to tasks.
            bResult = FALSE;
            std::stringstream strWarning;
            strWarning << this << " not bound to a task!" << endl;
        }

        if(m_pControlPropagationPorts.size() != 0) {
            // datablocks received on this port with control signals set 
            // must cause blocks on all output ports in this list to receive the
            // same control signal. This is only meaningful on input type ports.
            if(m_ePortType == OUTPUT_PORT) {
                // an output port cannot be a control propagation source
                bResult = FALSE;
                strWarning << this << " marked as a control propagation source, which is not possible for output ports." << endl;
            }
            vector<Port*>::iterator vi;
            for(vi=m_pControlPropagationPorts.begin(); vi!=m_pControlPropagationPorts.end(); vi++) {
                Port * pDest = *vi;
                OutputPort * pOutputPort = dynamic_cast<OutputPort*>(pDest);
                if(pOutputPort == NULL) {
                    // an output port cannot be a control propagation source
                    bResult = FALSE;
                    strWarning << this << " marked as a control propagation source, with a destination that is not an output port." << endl;
                }
                if(this != pDest->GetControlPropagationSource()) {
                    // our destination doesn't think we are the source!
                    bResult = FALSE;
                    strWarning << this << " marked as a control propagation source, with a destination that is marked with a different source." << endl;
                }
            }
        }

        if(m_pGatedPorts.size() > 0) {
            // control signals on this port gate other outputs on the same task. 
            // make sure that those ports think we are the control source.
            vector<Port*>::iterator vi;
            for(vi=m_pGatedPorts.begin(); vi!=m_pGatedPorts.end(); vi++) {
                Port * pDest = *vi;                
                OutputPort * pOutputPort = dynamic_cast<OutputPort*>(pDest);
                if(this != pOutputPort->GetControlPort()) {
                    // our gated port doesn't think we are the control source!
                    bResult = FALSE;
                    strWarning << this << " marked with gated port which is not marked with it as the control source." << endl;
                }
            }
        }

        if(m_vControlChannels.size() > 0) {
            // this port is a switch port, where the control channel 
            // is the preferred channel. This port had better be an input port
            // or meta port, and the structure is only meaningful if there is also 
            // a normal port. 
            if(m_ePortType != INPUT_PORT && m_ePortType != META_PORT) {
                // only input and meta ports can be switch ports
                bResult = FALSE;
                strWarning << this << " marked as a switch port, which is not possible for ports of its type." << endl;
            }
            if(m_vChannels.size() == 0) {
                bResult = FALSE;
                strWarning << this << " has a control channel but no normal channels, which will not work correctly to form a switch port." << endl;
            }
        }

        if(m_vDeferredChannels.size()) {
            // if this port has deferred channels it means it defers its input
            // to another channel source, making this port the describer and the
            // deferred port the described channel. We can only have one channel
            // to which we defer, and it had better be planning to feed us input. 
            if(m_ePortType == OUTPUT_PORT) {
                bResult = FALSE;
                strWarning << this 
                            << " has a deferred channel which is not meaningful for output ports." 
                            << endl;
            }
            std::vector<DEFERREDCHANNELDESC*>::iterator vi;
            for(vi=m_vDeferredChannels.begin(); vi!=m_vDeferredChannels.end(); vi++) {
                DEFERREDCHANNELDESC* pDesc = *vi;
                Channel * pChannel = pDesc->pChannel;
                Port * pDeferredPort = pChannel->GetBoundPort(CE_DST);
                if(pDeferredPort == this) {
                    bResult = FALSE;
                    strWarning << this << " has a deferred channel " 
                                << pChannel << " which is bound to itself. What did you intend?" 
                                << endl;
                }
                size_t nDescPortSize = pDeferredPort->m_vDescriptorPorts.size();
                switch(nDescPortSize) {
                case 0:
                    bResult = FALSE;
                    strWarning << this << " defers channel " 
                                << pChannel << " is bound to " 
                                << pDeferredPort << ", which does not have any descriptor ports!"
                                << endl;
                    break;
                case 1:                     
                    if(pDeferredPort->m_vDescriptorPorts[0]->pPort != this) {
                        bResult = FALSE;
                        strWarning << this << " defers channel " 
                                    << pChannel << " which is bound as a descriptor for " 
                                    << pDeferredPort->m_vDescriptorPorts[0]->pPort
                                    << ":  it should be a descriptor for " 
                                    << this
                                    << endl;
                    }
                    break;
                default:
                    bResult = FALSE;
                    strWarning << this << " defers channel " 
                                << pChannel << " is bound to " 
                                << pDeferredPort << ", which has "
                                << nDescPortSize << " descriptor ports (should be only 1)!"
                                << endl;
                    break;
                }
            }
        }

        if(m_vDescriptorPorts.size()) {
            // other ports are deferreing to us. 
            // we can have any number of ports deferring to us.
            if(m_ePortType == OUTPUT_PORT) {
                bResult = FALSE;
                strWarning << this 
                            << " is bound as a descriptor port(s) which is not meaningful for output ports." 
                            << endl;
            }        
            std::vector<DEFERREDPORTDESC*>::iterator vi; 
            for(vi=m_vDescriptorPorts.begin(); vi!=m_vDescriptorPorts.end(); vi++) {
                DEFERREDPORTDESC* pDesc = (*vi);
                Port * pDescPort = pDesc->pPort;
                if(pDescPort->GetPortType() == OUTPUT_PORT || pDescPort->GetPortType() == INITIALIZER_PORT) {
                    bResult = FALSE;
                    strWarning << this << " is bound as a descriptor port of (" 
                               << pDescPort << ") whose type is not meaningful in that role." 
                               << endl;
                }
                // assert that the deferred channel list actually contains a channel whose
                // destination is this port: moreover, there should only be one such entry.
                DEFERREDCHANNELDESC* pCorrespondingDesc = NULL;
                std::vector<DEFERREDCHANNELDESC*>::iterator di;
                std::vector<DEFERREDCHANNELDESC*>* pChannels = pDescPort->GetDeferredChannels();
                for(di=pChannels->begin(); di!=pChannels->end(); di++) {                
                    DEFERREDCHANNELDESC* pCDesc = *di;
                    if(pCDesc->pChannel == NULL) {
                        assert(pCDesc->pChannel != NULL);
                    } else {
                        if(pCDesc->pChannel->GetBoundPort(CE_DST) == this) {
                            if(pCorrespondingDesc != NULL) {
                                bResult = FALSE;
                                strWarning << this << " is a descriptor port of (" 
                                           << pDescPort << ") but is bound multiple times in that role." 
                                           << endl;
                            } else {
                                pCorrespondingDesc = pCDesc;
                            }
                        }
                    }
                    if(pCDesc == NULL) {
                        bResult = FALSE;
                        strWarning << this << " is a descriptor port of (" 
                                    << pDescPort << ") but the corresponding deferred channel is absent at that port!" 
                                    << endl;
                    }
                }
            }
        }

        // Check any properties that require access to subclass
        // data structures to check.
        bResult &= CheckTypeSpecificSemantics(pos, pGraph);

        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a meta function. </summary>
    ///
    /// <remarks>   Crossbac, 1/10/2012. </remarks>
    ///
    /// <param name="eMetaFunctionSpecifier">   Information describing the meta function. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::SetMetaFunction(
        METAFUNCTION eMetaFunctionSpecifier
        )
    {
        assert(FALSE);
        PTask::Runtime::HandleError("Port::SetMetaFunction called--"
                                    "only subclass implementations should be called!");
        eMetaFunctionSpecifier = eMetaFunctionSpecifier; // supress compiler warning
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the meta function. </summary>
    ///
    /// <remarks>   Crossbac, 1/10/2012. </remarks>
    ///
    /// <returns>   The meta function. </returns>
    ///-------------------------------------------------------------------------------------------------

    METAFUNCTION    
    Port::GetMetaFunction(
        VOID
        )
    {
        assert(FALSE);
        PTask::Runtime::HandleError("Port::GetMetaFunction called--"
                                    "only subclass implementations should be called!");
        return MF_NONE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the channel bound to this port at the given index. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pChannelList">     [in,out] If non-null, list of channels. </param>
    /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
    ///
    /// <returns>   null if it fails, else the channel. </returns>
    ///-------------------------------------------------------------------------------------------------

    Channel *		
    Port::GetChannel(
        vector<Channel*>* pChannelList, 
        UINT nChannelIndex
        )
    {
        Channel * pChannel = NULL;
        Lock();
        assert(nChannelIndex < m_vChannels.size());
        if((size_t) nChannelIndex < m_vChannels.size()) {
            pChannel = (*pChannelList)[nChannelIndex];
        }
        Unlock();
        return pChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a channel count. </summary>
    ///
    /// <remarks>   Crossbac, 2/2/2012. </remarks>
    ///
    /// <param name="pChannelList"> [in,out] If non-null, list of channels. </param>
    ///
    /// <returns>   The channel count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT            
    Port::GetChannelCount(
        vector<Channel*>* pChannelList
        )
    {
        assert(pChannelList != NULL);
        return (UINT) pChannelList->size();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Configure scoped iteration. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="uiIterations"> The iterations. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::BeginIterationScope(
        UINT uiIterations
        )
    {
        UNREFERENCED_PARAMETER(uiIterations);
        assert(FALSE && "Port::BeginIterationScope called: subclasses must implement!");
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   end scoped iteration. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="uiIterations"> The iterations. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::EndIterationScope(
        VOID
        )
    {
        assert(FALSE && "Port::EndIterationScope called: subclasses must implement!");
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an iteration source. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::SetIterationSource(
        Port * pPort
        )
    {
        UNREFERENCED_PARAMETER(pPort);
        assert(FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the iteration source. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the iteration source. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port *          
    Port::GetIterationSource(
        VOID
        )
    {
        assert(FALSE);
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an iteration target to the list. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::BindIterationTarget(
        Port * pPort
        )
    {
        UNREFERENCED_PARAMETER(pPort);
        assert(FALSE && "subclasses must implement this!");
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the suppress clones property, which allows a user to suppress output cloning
    ///             for blocks on ports with multiple (R/W conflicting) downstream consumers, if the
    ///             programmer happens to know something about the structure of the graph that the
    ///             runtime cannot (or does not detect) and that makes it safe to do so. The base-class
    ///             implementation asserts because this is really a sub-class specific method, but
    ///             then goes on to do the right thing to set the property. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 2/29/2012. </remarks>
    ///
    /// <param name="bSuppressClones">  true to suppress clones. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::SetSuppressClones(
        BOOL bSuppressClones
        )
    {
        assert(FALSE && "Base-class implementation Port::SetSuppressClones called");
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
    ///             The base-class implementation asserts because this is really a sub-class specific
    ///             method, but then goes on to do the right thing to get the property.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 2/29/2012. </remarks>
    ///
    /// <returns>   the value of the suppress clones property. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::GetSuppressClones(
        VOID
        )
    {
        assert(FALSE && "Base-class implementation Port::GetSuppressClones called");
        return m_bSuppressClones;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind this port to a particular dimension for geometry estimation. </summary>
    ///
    /// <remarks>   crossbac, 5/1/2012. </remarks>
    ///
    /// <param name="eGeoDimension">    The geo dimension. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::BindToEstimatorDimension(
        GEOMETRYESTIMATORDIMENSION eGeoDimension
        )
    {
        m_eGeometryDimBinding = eGeoDimension;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the estimator dimension binding. </summary>
    ///
    /// <remarks>   crossbac, 5/1/2012. </remarks>
    ///
    /// <returns>   The estimator dimension binding. </returns>
    ///-------------------------------------------------------------------------------------------------

    GEOMETRYESTIMATORDIMENSION 
    Port::GetEstimatorDimensionBinding(
        VOID
        )
    {
        return m_eGeometryDimBinding;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind dependent accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 5/16/2012. </remarks>
    ///
    /// <param name="accClass"> The acc class. </param>
    /// <param name="nIndex">   Zero-based index of the accelerator in the dependent accelerator
    ///                         list. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::BindDependentAccelerator(
        ACCELERATOR_CLASS accClass, 
        int nIndex
        )
    {
        Lock();
        m_accDependentAcceleratorClass = accClass;
        m_nDependentAcceleratorBinding = nIndex;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind dependent accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 5/16/2012. </remarks>
    ///
    /// <param name="accClass"> The acc class. </param>
    /// <param name="nIndex">   Zero-based index of the accelerator in the dependent accelerator
    ///                         list. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL           
    Port::SetDependentAffinity(
        Accelerator* pAccelerator, 
        AFFINITYTYPE affinityType
        )
    {
        BOOL bSuccess = FALSE;
        Lock();
        if(affinityType == AFFINITYTYPE_NONE) {
            if(m_vAffinities.find(pAccelerator) != m_vAffinities.end()) {
                m_vAffinities.erase(pAccelerator);
                bSuccess = TRUE;
            }
        } else {
            bSuccess = TRUE;
            m_vAffinities[pAccelerator] = affinityType;
            if(affinityType == AFFINITYTYPE_MANDATORY) {
                m_pMandatoryAccelerator = pAccelerator;
            }
        }        
        Unlock();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets task-accelerator affinity. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="vAccelerators">    [in,out] non-null, the accelerators. </param>
    /// <param name="pvAffinityTypes">  [in,out] List of types of affinities. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Port::SetDependentAffinity(
        std::vector<Accelerator*> &vAccelerators, 
        std::vector<AFFINITYTYPE> &pvAffinityTypes
        )
    {
        BOOL bDependentBinding = HasDependentAcceleratorBinding();
        assert(bDependentBinding == TRUE);
        if(!bDependentBinding) return FALSE;

        Lock();
        BOOL bSuccess = TRUE;
        assert(vAccelerators.size() == pvAffinityTypes.size());
        std::vector<Accelerator*>::iterator ai = vAccelerators.begin();
        std::vector<AFFINITYTYPE>::iterator afi = pvAffinityTypes.begin();
        while(ai != vAccelerators.end() && afi != pvAffinityTypes.end()) {
            Accelerator * pAccelerator = *ai;
            AFFINITYTYPE affinityType = *afi;
            if(pAccelerator->GetClass() != m_accDependentAcceleratorClass) {
                assert(FALSE);
                PTask::Runtime::Warning("attempt to affinitize port->accelerator with wrong class!");
                bSuccess = FALSE;
            } else {
                bSuccess &= SetDependentAffinity(pAccelerator, affinityType);
            }
            ai++;
            afi++;
        }
        Unlock();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has dependent accelerator binding. </summary>
    ///
    /// <remarks>   Crossbac, 5/16/2012. </remarks>
    ///
    /// <returns>   true if dependent accelerator binding, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL              
    Port::HasDependentAcceleratorBinding(
        VOID
        )
    {
        return m_accDependentAcceleratorClass != ACCELERATOR_CLASS_UNKNOWN &&
               m_nDependentAcceleratorBinding != -1;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a dependent accelerator class. </summary>
    ///
    /// <remarks>   Crossbac, 5/16/2012. </remarks>
    ///
    /// <param name="nIndex">   Zero-based index of the accelerator in the dependent accelerator
    ///                         list. </param>
    ///
    /// <returns>   The dependent accelerator class. </returns>
    ///-------------------------------------------------------------------------------------------------

    ACCELERATOR_CLASS 
    Port::GetDependentAcceleratorClass(
        int nIndex
        )
    {
        UNREFERENCED_PARAMETER(nIndex);
        return m_accDependentAcceleratorClass;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the dependent accelerator index. </summary>
    ///
    /// <remarks>   Crossbac, 5/16/2012. </remarks>
    ///
    /// <returns>   The dependent accelerator index. </returns>
    ///-------------------------------------------------------------------------------------------------

    int               
    Port::GetDependentAcceleratorIndex(
        VOID
        )
    {
        return m_nDependentAcceleratorBinding;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a trigger port. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    /// <param name="bTrigger"> true to trigger. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::SetTriggerPort(
        __in Graph * pGraph, 
        __in BOOL bTrigger
        )
    {
        m_pGraph = pGraph;
        m_bTriggerPort = bTrigger;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is trigger port. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <returns>   true if trigger port, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::IsTriggerPort(
        VOID
        )
    {
        return m_bTriggerPort && m_pGraph != NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Handle trigger. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <param name="luiCode">  The code. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Port::HandleTriggers(
        CONTROLSIGNAL luiCode
        )
    {
        assert(m_bTriggerPort);
        assert(m_pGraph != NULL);
        m_pGraph->ExecuteTriggers(this, luiCode);
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
    Port::SetScopeTerminus(
        __in CONTROLSIGNAL luiSignal, 
        __in BOOL bTerminus
        )
    {
        assert(FALSE); 
        UNREFERENCED_PARAMETER(luiSignal);
        UNREFERENCED_PARAMETER(bTerminus);
        return FALSE;
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
    Port::IsScopeTerminus(
        VOID
        )
    {
        return FALSE;
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
    Port::IsScopeTerminus(
        CONTROLSIGNAL luiControlSignal
        )
    {
        UNREFERENCED_PARAMETER(luiControlSignal);
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
    /// 								Runtime::GetICBlockPoolSize() will be used to determine the
    /// 								size of the pool. </param>
    ///
    /// <returns>   True if it succeeds, false if it fails. If a port type doesn't actually implement
    ///             pooling, return false as well.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::AllocateBlockPool(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )
    {
        UNREFERENCED_PARAMETER(pAccelerators);
        UNREFERENCED_PARAMETER(uiPoolSize);
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destroys the block pool. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Port::DestroyBlockPool(
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
    Port::IsBlockPoolActive(
        VOID
        )
    {
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
    Port::GetPoolOwnerName(
        VOID
        )
    {
        return m_lpszVariableBinding;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets mandatory dependent accelerator if one has been specified. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the mandatory dependent accelerator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Port::GetMandatoryDependentAccelerator(
        VOID
        )
    {
        return this->m_pMandatoryAccelerator;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the dependent affinities map. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the dependent affinities. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::map<Accelerator*, AFFINITYTYPE> * 
    Port::GetDependentAffinities(
        VOID
        )
    {
        // achtung!
        return &m_vAffinities;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets upstream channel pool size. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="uiPoolSize">       Size of the pool. </param>
    /// <param name="bGrowable">        The growable. </param>
    /// <param name="uiGrowIncrement">  The grow increment. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Port::SetUpstreamChannelPool(
        __in UINT uiPoolSize, 
        __in BOOL bGrowable,
        __in UINT uiGrowIncrement
        )
    {
        m_bHasUpstreamChannelPool = TRUE;
        m_bUpstreamChannelPoolGrowable = bGrowable;
        m_uiUpstreamChannelPoolSize = uiPoolSize;
        m_uiUpstreamChannelPoolGrowIncrement = uiGrowIncrement;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has upstream channel pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if upstream channel pool, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Port::HasUpstreamChannelPool(
        VOID
        )
    {
        return m_bHasUpstreamChannelPool;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has upstream channel pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if upstream channel pool, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Port::IsUpstreamChannelPoolGrowable(
        VOID
        )
    {
        return m_bUpstreamChannelPoolGrowable;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets upstream channel pool size. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   The upstream channel pool size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Port::GetUpstreamChannelPoolSize(
        VOID
        )
    {
        return m_uiUpstreamChannelPoolSize;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets upstream channel pool size. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   The upstream channel pool size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Port::GetUpstreamChannelPoolGrowIncrement(
        VOID
        )
    {
        return m_bHasUpstreamChannelPool && m_bUpstreamChannelPoolGrowable ? m_uiUpstreamChannelPoolGrowIncrement : 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets high water mark. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Port::GetHighWaterMark(
        VOID
        )
    {
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the total number of blocks owned by the pool. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The total number of blocks owned by the pool (whether they are queued or not). </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Port::GetOwnedBlockCount(
        VOID
        )
    {
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the low water mark. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Port::GetLowWaterMark(
        VOID
        )
    {
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the currently available count. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2013. </remarks>
    ///
    /// <returns>   The high water mark. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Port::GetAvailableBlockCount(
        VOID
        )
    {
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Locks any bound channels. For some operations it is necessary
    ///             to enforce a lock ordering discipline between channels and ports.
    ///             For example, naively, pushing into a channel locks the channel
    ///             and then finds the attached port, and locks it, while the dispatch
    ///             ready checker encounters ports first and traverses them to get to
    ///             channels, which naively encourages the opposite order and admits
    ///             the possibility of deadlock. Consequently, we require a channel->port
    ///             ordering when both locks are required. This utility allows a port
    ///             to lock it attached channels before acquiring its own lock when necessary.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/21/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Port::AssembleChannelLockSet(
        VOID
        )
    {
        // add all the channels to a set and then
        // traverse the set to lock them all. This
        // gives us a lock order. 
        Lock();
        std::vector<Channel*>::iterator vi;
        for(vi=m_vChannels.begin(); vi!=m_vChannels.end(); vi++) 
            m_vChannelLockSet.insert(*vi);
        for(vi=m_vControlChannels.begin(); vi!=m_vControlChannels.end(); vi++) 
            m_vChannelLockSet.insert(*vi);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Locks any bound channels. For some operations it is necessary
    ///             to enforce a lock ordering discipline between channels and ports.
    ///             For example, naively, pushing into a channel locks the channel
    ///             and then finds the attached port, and locks it, while the dispatch
    ///             ready checker encounters ports first and traverses them to get to
    ///             channels, which naively encourages the opposite order and admits
    ///             the possibility of deadlock. Consequently, we require a channel->port
    ///             ordering when both locks are required. This utility allows a port
    ///             to lock it attached channels before acquiring its own lock when necessary.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/21/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Port::LockBoundChannels(
        VOID
        )
    {
        std::set<Channel*>::iterator si;
        for(si=m_vChannelLockSet.begin(); si!=m_vChannelLockSet.end(); si++) {
            (*si)->Lock();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unlocks any bound channels. For some operations it is necessary
    ///             to enforce a lock ordering discipline between channels and ports.
    ///             For example, naively, pushing into a channel locks the channel
    ///             and then finds the attached port, and locks it, while the dispatch
    ///             ready checker encounters ports first and traverses them to get to
    ///             channels, which naively encourages the opposite order and admits
    ///             the possibility of deadlock. Consequently, we require a channel->port
    ///             ordering when both locks are required. This utility allows a port
    ///             to lock it attached channels before acquiring its own lock when necessary.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/21/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Port::UnlockBoundChannels(
        VOID
        )
    {
        std::set<Channel*>::iterator si;
        for(si=m_vChannelLockSet.begin(); si!=m_vChannelLockSet.end(); si++) {
            (*si)->Unlock();
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
    Port::GetBlockFromPool(
        __in Accelerator * pAccelerator,
        __in UINT uiDataBytes,
        __in UINT uiMetaBytes,
        __in UINT uiTemplateBytes
        )
	{
		UNREFERENCED_PARAMETER(pAccelerator);
		UNREFERENCED_PARAMETER(uiDataBytes);
		UNREFERENCED_PARAMETER(uiMetaBytes);
		UNREFERENCED_PARAMETER(uiTemplateBytes);
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
    Port::BlockPoolIsGlobal(
        void
        )
    {
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   port toString operator. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="os">   [in,out] The operating system. </param>
    /// <param name="port"> The port. </param>
    ///
    /// <returns>   The shifted result. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::ostream& operator <<(std::ostream &os, Port * pCPort) {
        if(pCPort == NULL) {
            os << "port:null";
            return os;
        }
        Port * pPort = const_cast<Port*>(pCPort);
        std::string strType;
        switch(pPort->GetPortType()) {
        case INPUT_PORT: strType="INPUT"; break;
        case OUTPUT_PORT: strType="OUTPUT"; break;
        case STICKY_PORT: strType="STICKY"; break;
        case META_PORT: strType="META"; break;
        case INITIALIZER_PORT: strType="INIT"; break;
        }
        const char * lpszTask = pPort->GetTask() ? pPort->GetTask()->GetTaskName() : "unbound";
        const char * lpszPortName = pPort->GetVariableBinding() ? pPort->GetVariableBinding() : "anonymous";
        os << strType << "(t=" << lpszTask << ", v=" << lpszPortName << ")";
        return os;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   check for non-zero control codes. </summary>
    ///
    /// <remarks>   Crossbac, 3/2/2012. </remarks>
    ///
    /// <param name="vPortMap"> [in,out] [in,out] If non-null, the port map. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Port::CheckControlCodes(
        VOID
        )
    {
#ifdef GRAPH_DIAGNOSTICS
        if(HASSIGNAL(m_luiPropagatedControlCode)) {
            std::cout 
                << m_pBoundTask << ":" 
                << this << "ctlcode: " 
                << std::hex << m_luiPropagatedControlCode << std::dec
                << std::endl;
        }
        if(m_pReplayableBlock) {
            m_pReplayableBlock->Lock();
            CONTROLSIGNAL luiCode = m_pReplayableBlock->GetControlSignals();
            std::cout 
                << m_pBoundTask << ":" 
                << this << "->m_pReplayableBlock.ctlcode: " 
                << std::hex << luiCode << std::dec
                << std::endl;            
            m_pReplayableBlock->Unlock();
        }
        if(m_ePortType == OUTPUT_PORT) {
            Datablock * pOPortBlock = ((OutputPort*)this)->m_pDatablock;
            if(pOPortBlock != NULL) {                
                pOPortBlock->Lock();
                CONTROLSIGNAL luiCode = pOPortBlock->GetControlSignals();
                if(HASSIGNAL(luiCode)) {
                    std::cout 
                        << m_pBoundTask << ":" 
                        << this << "->m_pDatablock.ctlcode: " 
                        << std::hex << luiCode << std::dec
                        << std::endl;            
                }
                pOPortBlock->Unlock();
            }
        }
#endif
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Force sticky port release. </summary>
    ///
    /// <remarks>   crossbac, 9/10/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Port::ForceStickyDeviceBufferRelease(
        void
        )
    {
        std::set<Port*>::iterator si;
        std::map<Graph*, std::set<Port*>>::iterator mi;
        EnterCriticalSection(&m_csReleasableStickyPorts);
        UINT uiCandidatePorts = 0;
        for(mi=m_vReleaseableStickyPorts.begin(); mi!=m_vReleaseableStickyPorts.end(); mi++) 
            for(si=mi->second.begin(); si!=mi->second.end(); si++) 
                uiCandidatePorts++;
        PTask::Runtime::MandatoryInform("%s::%s: %d candidates for forcible release...\n",
                                        __FILE__,
                                        __FUNCTION__,
                                        uiCandidatePorts);
        for(mi=m_vReleaseableStickyPorts.begin(); mi!=m_vReleaseableStickyPorts.end(); mi++) {
            for(si=mi->second.begin(); si!=mi->second.end(); si++) {
                Datablock * pBlock = (*si)->m_pReplayableBlock;
                if(pBlock != NULL) {
                    pBlock->Lock();
                    std::map<UINT, size_t> vBufferSizes;
                    if(pBlock->GetInstantiatedBufferSizes(vBufferSizes)) {
                        std::map<UINT, size_t>::iterator bsi;
                        for(bsi=vBufferSizes.begin(); bsi!=vBufferSizes.end(); bsi++) {
                            PTask::Runtime::MandatoryInform("%s::%s(%s--sticky): force release memspace_%d, %d bytes reclaimed...\n",
                                                            __FILE__,
                                                            __FUNCTION__,
                                                            (*si)->m_lpszVariableBinding,
                                                            bsi->first,
                                                            (UINT)bsi->second);
                            pBlock->ReleasePhysicalBuffers(bsi->first);
                        }
                    }
                    pBlock->Unlock();
                }
            }
        }
        LeaveCriticalSection(&m_csReleasableStickyPorts);

    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the static data structures for the port class. </summary>
    ///
    /// <remarks>   crossbac, 9/10/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Port::InitializeGlobal(
        void
        )
    {
        InitializeCriticalSection(&m_csReleasableStickyPorts);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the static data structures for the port class. </summary>
    ///
    /// <remarks>   crossbac, 9/10/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Port::DestroyGlobal(
        void
        )
    {
        DeleteCriticalSection(&m_csReleasableStickyPorts);
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
    Port::FindMaximalDownstreamCapacity(
        __inout std::set<Task*>& vTasksVisited,
        __inout std::vector<Channel*>& vPath
        )
    {
        UNREFERENCED_PARAMETER(vPath);
        UNREFERENCED_PARAMETER(vTasksVisited);

        // only initializer ports, input ports and output ports can participate in complex paths that
        // contribute to an aggregate path capacity. These sub-classes override this method, so if
        // we actually get called here, it's a fishy situation. 

        assert(m_ePortType != INPUT_PORT && 
               m_ePortType != INITIALIZER_PORT &&
               m_ePortType != OUTPUT_PORT);

        // this port is the terminus of any path
        // because things that arrive here are immutable
        // and not passed to another port.

        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is dispatch dimensions hint. </summary>
    ///
    /// <remarks>   Crossbac, 1/16/2014. </remarks>
    ///
    /// <returns>   true if dispatch dimensions hint, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::IsDispatchDimensionsHint(
        VOID
        )
    {
        return m_bDispatchDimensionsHint;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets dispatch dimensions hint. </summary>
    ///
    /// <remarks>   Crossbac, 1/16/2014. </remarks>
    ///
    /// <param name="bIsHintSource">    true if this object is hint source. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Port::SetDispatchDimensionsHint(
        BOOL bIsHintSource
        )
    {
        m_bDispatchDimensionsHint = bIsHintSource;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pAccelerator' is a block pool view accelerator:
    ///             this means that blocks in the pool should eagerly materialize
    ///             buffers/initval views for blocks when they are allocated
    ///             at graph finalization. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/13/2014. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   true if there is no reason to filter this accelerator from
    ///             the list of view materializations for a block pool. Generally,
    ///             only explicit affinity can impact this result.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Port::IsBlockPoolViewAccelerator(
        __in Accelerator* pAccelerator
        )
    {
        Accelerator * pAffinitizedAccelerator = m_pBoundTask->GetAffinitizedAcceleratorHint();
        assert(pAccelerator != NULL);
        if(pAccelerator == NULL)
            return FALSE;
        if(pAffinitizedAccelerator == NULL)
            return TRUE; 

        BOOL bTransitionPoint = IsExplicitMemorySpaceTransitionPoint();
        switch(PTask::Runtime::GetDatablockAffinitiyPolicy()) {

        case DAP_IGNORE_AFFINITY: 
            if(pAccelerator != pAffinitizedAccelerator) {
                PTask::Runtime::MandatoryInform(
                    "Ignoring task affinity for Datablock pool on OutputPort of task %s!\n", 
                    m_pBoundTask->GetTaskName());
            }               
            return TRUE;

        case DAP_BOUND_TASK_AFFINITY:

            // Skip accelerators other than the one affinitized with the bound task.
            // *UNLESS* this block is known to be a transition point in the graph
            // from one GPU to another GPU: this means downstream task will
            // need these buffers regardless of whether this task does or not.
            return (pAccelerator == pAffinitizedAccelerator || bTransitionPoint);

        case DAP_TRANSITIVE_AFFINITY: 
            PTask::Runtime::HandleError("DAP_TRANSITIVE_AFFINITY not implemented. Exiting!\n");
            return TRUE;

        default: 
            PTask::Runtime::HandleError("Unknown Datablock affinity policy %d. Exiting!\n",
                                         PTask::Runtime::GetDatablockAffinitiyPolicy());
            return TRUE;
        }
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
    Port::IsExplicitMemorySpaceTransitionPoint(
        VOID
        )
    {
        // subclasses override
        return FALSE; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the super-set of all "control signals of interest" for this graph object.  
    ///             A control signal is "of interest" if the behavior of this object is is predicated
    ///             in some way by the presence or absence of a given signal. This function returns
    ///             the bit-wise OR of all such signals.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/7/2014. </remarks>
    ///
    /// <returns>   The bitwise OR of all found control signals of interest. </returns>
    ///-------------------------------------------------------------------------------------------------

    CONTROLSIGNAL 
    Port::GetControlSignalsOfInterest(
        VOID
        )
    {
        CONTROLSIGNAL luiSignals = DBCTLC_NONE;
        std::vector<Channel*>::iterator ci;
        for(ci=m_vChannels.begin(); ci!=m_vChannels.end(); ci++) 
            luiSignals |= (*ci)->GetControlSignalsOfInterest();
        luiSignals |= m_luiInitialPropagatedControlCode;
        luiSignals |= m_luiScopeTerminalSignal;
        luiSignals |= m_luiStickyReleaseSignal;
        return luiSignals;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this port is bound to an (effectively) read-only input. </summary>
    ///
    /// <remarks>   crossbac, 7/8/2014. </remarks>
    ///
    /// <returns>   true if constant semantics, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::IsConstantSemantics(
        VOID
        )
    {
        switch(m_ePortType) {
        case PORTTYPE::INITIALIZER_PORT: return IsSticky() && !IsInOutParameter();
        case PORTTYPE::INPUT_PORT:       return IsSticky() && !IsInOutParameter();
        case PORTTYPE::META_PORT:        return TRUE;
        case PORTTYPE::OUTPUT_PORT:      return FALSE;
        case PORTTYPE::STICKY_PORT:      return TRUE;
        }
        return FALSE;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is input parameter. </summary>
    ///
    /// <remarks>   crossbac, 7/8/2014. </remarks>
    ///
    /// <returns>   true if input parameter, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::IsInputParameter(
        VOID
        )
    {
        switch(m_ePortType) {
        case PORTTYPE::INITIALIZER_PORT: return TRUE;
        case PORTTYPE::INPUT_PORT:       return TRUE;
        case PORTTYPE::META_PORT:        return FALSE;
        case PORTTYPE::OUTPUT_PORT:      return FALSE;
        case PORTTYPE::STICKY_PORT:      return TRUE;
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is input parameter. </summary>
    ///
    /// <remarks>   crossbac, 7/8/2014. </remarks>
    ///
    /// <returns>   true if input parameter, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::IsInitializerParameter(
        VOID
        )
    {
        switch(m_ePortType) {
        case PORTTYPE::INITIALIZER_PORT: return TRUE;
        case PORTTYPE::INPUT_PORT:       return FALSE;
        case PORTTYPE::META_PORT:        return FALSE;
        case PORTTYPE::OUTPUT_PORT:      return FALSE;
        case PORTTYPE::STICKY_PORT:      return FALSE;
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is output parameter. </summary>
    ///
    /// <remarks>   crossbac, 7/8/2014. </remarks>
    ///
    /// <returns>   true if output parameter, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Port::IsOutputParameter(
        VOID
        )
    {
        switch(m_ePortType) {
        case PORTTYPE::INITIALIZER_PORT: return IsInOutParameter();
        case PORTTYPE::INPUT_PORT:       return IsInOutParameter();
        case PORTTYPE::META_PORT:        return FALSE;
        case PORTTYPE::OUTPUT_PORT:      return TRUE;
        case PORTTYPE::STICKY_PORT:      return FALSE;
        }
        return FALSE;
    }

};
