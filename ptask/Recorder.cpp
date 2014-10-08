//--------------------------------------------------------------------------------------
// File: Recorder.cpp
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#ifdef XMLSUPPORT

#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif
#include "Recorder.h"
#include "graph.h"
#include "task.h"
#include <assert.h>

using namespace std;
//using namespace PTask::Runtime;

namespace PTask {

    RecordedAction::RecordedAction(
        RECORDEDACTIONTYPE type, 
        std::string name        
        ) 
    {
        m_type = type;
        m_name = name;
    }
       
    const char * RecordedAction::GetName() { return m_name.c_str(); }


    // BindControlPort

    BindControlPort::BindControlPort()
        : RecordedAction(BINDCONTROLPORT, std::string("BindControlPort")) { }

    BindControlPort::BindControlPort(
            Port * pController,
            Port * pGatedPort,
            BOOL bInitiallyOpen
            )
            : RecordedAction(BINDCONTROLPORT, std::string("BindControlPort")),
            m_controllerPortUID(pController->GetUID()), 
            m_gatedPortUID(pGatedPort->GetUID()), 
            m_initiallyOpen(bInitiallyOpen) { }

    void
    BindControlPort::Write(XMLWriter * writer)
    {
        writer->WriteElement("ControllerPort", m_controllerPortUID);
        writer->WriteElement("GatedPort", m_gatedPortUID);
        writer->WriteElement("InitiallyOpen", m_initiallyOpen);
    }

    void
    BindControlPort::Read(XMLReader * reader)
    {
        m_controllerPortUID = reader->ReadUINTElement("ControllerPort");
        m_gatedPortUID = reader->ReadUINTElement("GatedPort");
        m_initiallyOpen = reader->ReadBooleanElement("InitiallyOpen");
    }

    void
    BindControlPort::Replay(XMLReader * reader)
    {
        Port * pControllerPort = reader->GetPort(m_controllerPortUID);
        Port * pGatedPort = reader->GetPort(m_gatedPortUID);
        Graph * pGraph = reader->GetGraph();
        pGraph->BindControlPort(pControllerPort, pGatedPort, m_initiallyOpen);
    }

    // BindControlPropagationChannel

    BindControlPropagationChannel::BindControlPropagationChannel()
        : RecordedAction(BINDCONTROLPROPAGATIONCHANNEL, std::string("BindControlPropagationChannel")) { }

    BindControlPropagationChannel::BindControlPropagationChannel(
            Port * pInputPort, 
            Channel * pControlledChannel
            )
            : RecordedAction(BINDCONTROLPROPAGATIONCHANNEL, std::string("BindControlPropagationChannel")),
            m_inputPortUID(pInputPort->GetUID()), 
            m_controlledChannelName(pControlledChannel->GetName()) { }

    void
    BindControlPropagationChannel::Write(XMLWriter * writer)
    {
        writer->WriteElement("InputPort", m_inputPortUID);
        writer->WriteElement("ControlledChannel", m_controlledChannelName.c_str());
    }

    void
    BindControlPropagationChannel::Read(XMLReader * reader)
    {
        m_inputPortUID = reader->ReadUINTElement("InputPort");
        reader->ReadStringElement("ControlledChannel", m_controlledChannelName);        
    }

    void
    BindControlPropagationChannel::Replay(XMLReader * reader)
    {
        Port * pInputPort = reader->GetPort(m_inputPortUID);
        Graph * pGraph = reader->GetGraph();
        Channel * pControlledChannel = pGraph->GetChannel(const_cast<char*>(m_controlledChannelName.c_str()));
        pGraph->BindControlPropagationChannel(pInputPort, pControlledChannel);
    }

    BindControlPropagationPort::BindControlPropagationPort()
        : RecordedAction(BINDCONTROLPROPAGATIONPORT, std::string("BindControlPropagationPort")) { }

    BindControlPropagationPort::BindControlPropagationPort(
            Port * pInputPort, 
            Port * pOutputPort
            )
            : RecordedAction(BINDCONTROLPROPAGATIONPORT, std::string("BindControlPropagationPort")),
            m_inputPortUID(pInputPort->GetUID()), 
            m_outputPortUID(pOutputPort->GetUID()) { }

    void
    BindControlPropagationPort::Write(XMLWriter * writer)
    {
        writer->WriteElement("InputPort", m_inputPortUID);
        writer->WriteElement("OutputPort", m_outputPortUID);
    }

    void
    BindControlPropagationPort::Read(XMLReader * reader)
    {
        m_inputPortUID = reader->ReadUINTElement("InputPort");
        m_outputPortUID = reader->ReadUINTElement("OutputPort");
    }

    void
    BindControlPropagationPort::Replay(XMLReader * reader)
    {
        Port * pInputPort = reader->GetPort(m_inputPortUID);
        Port * pOutputPort = reader->GetPort(m_outputPortUID);
        Graph * pGraph = reader->GetGraph();
        pGraph->BindControlPropagationPort(pInputPort, pOutputPort);
    }

    // BindDescriptorPort

    BindDescriptorPort::BindDescriptorPort()
        : RecordedAction(BINDDESCRIPTORPORT, std::string("BindDescriptorPort")) { }

    BindDescriptorPort::BindDescriptorPort(
            Port * pDescribedPort, 
            Port * pDescriberPort,
            DESCRIPTORFUNC func
            )
            : RecordedAction(BINDDESCRIPTORPORT, std::string("BindDescriptorPort")),
            m_describedPortUID(pDescribedPort->GetUID()), 
            m_describerPortUID(pDescriberPort->GetUID()), 
            m_func(func) { }

    void
    BindDescriptorPort::Write(XMLWriter * writer)
    {
        writer->WriteElement("DescribedPort", m_describedPortUID);
        writer->WriteElement("DescriberPort", m_describerPortUID);
        writer->WriteElement("DescriptorFunction", (int)m_func);
    }

    void
    BindDescriptorPort::Read(XMLReader * reader)
    {
        int func;
        m_describedPortUID = reader->ReadUINTElement("DescribedPort");
        m_describerPortUID = reader->ReadUINTElement("DescriberPort");
        func = reader->ReadIntegerElement("DescriptorFunction");
        m_func = (DESCRIPTORFUNC)func;
    }

    void
    BindDescriptorPort::Replay(XMLReader * reader)
    {
        Port * pDescribedPort = reader->GetPort(m_describedPortUID);
        Port * pDescriberPort = reader->GetPort(m_describerPortUID);
        Graph * pGraph = reader->GetGraph();
        pGraph->BindDescriptorPort(pDescribedPort, pDescriberPort);
    }

    // BindIterationScope

    BindIterationScope::BindIterationScope()
        : RecordedAction(BINDITERATIONSCOPE, std::string("BindIterationScope")) { }

    BindIterationScope::BindIterationScope(
            Port * pMetaPort, 
            Port * pScopedPort
            )
            : RecordedAction(BINDITERATIONSCOPE, std::string("BindIterationScope")),
            m_metaPortUID(pMetaPort->GetUID()), 
            m_scopedPortUID(pScopedPort->GetUID()) { }

    void
    BindIterationScope::Write(XMLWriter * writer)
    {
        writer->WriteElement("MetaPort", m_metaPortUID);
        writer->WriteElement("ScopedPort", m_scopedPortUID);
    }

    void
    BindIterationScope::Read(XMLReader * reader)
    {
        m_metaPortUID = reader->ReadUINTElement("MetaPort");
        m_scopedPortUID = reader->ReadUINTElement("ScopedPort");
    }

    void
    BindIterationScope::Replay(XMLReader * reader)
    {
        Port * pMetaPort = reader->GetPort(m_metaPortUID);
        Port * pScopedPort = reader->GetPort(m_scopedPortUID);
        Graph * pGraph = reader->GetGraph();
        pGraph->BindIterationScope(pMetaPort, pScopedPort);
    }

    // SetBlockAndGridSize

    SetBlockAndGridSize::SetBlockAndGridSize()
        : RecordedAction(SETBLOCKANDGRIDSIZE, std::string("SetBlockAndGridSize")) { }

    SetBlockAndGridSize::SetBlockAndGridSize(
            Task * task,
            PTASKDIM3 grid,
            PTASKDIM3 block)
            : RecordedAction(SETBLOCKANDGRIDSIZE, std::string("SetBlockAndGridSize")),
            m_taskName(task->GetTaskName()), m_grid(grid), m_block(block) { }

    void
    SetBlockAndGridSize::Write(XMLWriter * writer)
    {
        writer->WriteElement("TaskName", m_taskName.c_str());
        writer->WriteElement("BlockXDim", m_block.x);
        writer->WriteElement("BlockYDim", m_block.y);
        writer->WriteElement("BlockZDim", m_block.z);
        writer->WriteElement("GridXDim", m_grid.x);
        writer->WriteElement("GridYDim", m_grid.y);
        writer->WriteElement("GridZDim", m_grid.z);
    }

    void
    SetBlockAndGridSize::Read(XMLReader * reader)
    {
        reader->ReadStringElement("TaskName", m_taskName);
        m_block.x = reader->ReadIntegerElement("BlockXDim");
        m_block.y = reader->ReadIntegerElement("BlockYDim");
        m_block.z = reader->ReadIntegerElement("BlockZDim");
        m_grid.x = reader->ReadIntegerElement("GridXDim");
        m_grid.y = reader->ReadIntegerElement("GridYDim");
        m_grid.z = reader->ReadIntegerElement("GridZDim");
    }

    void
    SetBlockAndGridSize::Replay(XMLReader * reader)
    {
        Graph * pGraph = reader->GetGraph();
        Task * task = pGraph->GetTask((char*)m_taskName.c_str());
        task->SetBlockAndGridSize(m_grid, m_block);
    }

    // SetComputeGeometry

    SetComputeGeometry::SetComputeGeometry()
        : RecordedAction(SETCOMPUTEGEOMETRY, std::string("SetComputeGeometry")) { }

    SetComputeGeometry::SetComputeGeometry(
            Task * task,
            int tgx,
            int tgy,
            int tgz)
            : RecordedAction(SETCOMPUTEGEOMETRY, std::string("SetComputeGeometry")), 
            m_taskName(task->GetTaskName()), m_tgx(tgx), m_tgy(tgy), m_tgz(tgz) { }

    void
    SetComputeGeometry::Write(XMLWriter * writer)
    {
        writer->WriteElement("TaskName", m_taskName.c_str());
        writer->WriteElement("PreferredXDim", m_tgx);
        writer->WriteElement("PreferredYDim", m_tgy);
        writer->WriteElement("PreferredZDim", m_tgz);
    }

    void
    SetComputeGeometry::Read(XMLReader * reader)
    {
        reader->ReadStringElement("TaskName", m_taskName);
        m_tgx = reader->ReadIntegerElement("PreferredXDim");
        m_tgy = reader->ReadIntegerElement("PreferredYDim");
        m_tgz = reader->ReadIntegerElement("PreferredZDim");
    }

    void
    SetComputeGeometry::Replay(XMLReader * reader)
    {
        Graph * pGraph = reader->GetGraph();
        Task * task = pGraph->GetTask((char*)m_taskName.c_str());
        task->SetComputeGeometry(m_tgx, m_tgy, m_tgz);
    }

    // SetPredicationType

    SetPredicationType::SetPredicationType()
        : RecordedAction(SETPREDICATIONTYPE, std::string("SetPredicationType")) { }

    SetPredicationType::SetPredicationType(
            Channel * pChannel,
            CHANNELENDPOINTTYPE eEndpoint, 
            CHANNELPREDICATE eCanonicalPredicator
            )
            : RecordedAction(SETPREDICATIONTYPE, std::string("SetPredicationType")), 
            m_channelName(pChannel->GetName()), 
            m_endpointType(eEndpoint), m_canonicalPredicate(eCanonicalPredicator) { }

    void
    SetPredicationType::Write(XMLWriter * writer)
    {
        writer->WriteElement("ChannelName", m_channelName.c_str());
        writer->WriteElement("EndpointType", m_endpointType);
        writer->WriteElement("CanonicalPredicate", m_canonicalPredicate);
    }

    void
    SetPredicationType::Read(XMLReader * reader)
    {
        reader->ReadStringElement("ChannelName", m_channelName);
        m_endpointType = reader->ReadIntegerElement("EndpointType");
        m_canonicalPredicate = reader->ReadIntegerElement("CanonicalPredicate");
    }

    void
    SetPredicationType::Replay(XMLReader * reader)
    {
        Graph * pGraph = reader->GetGraph();
        Channel * pChannel = pGraph->GetChannel((char*)m_channelName.c_str());
        pChannel->SetPredicationType(
            (CHANNELENDPOINTTYPE)m_endpointType, 
            (CHANNELPREDICATE)m_canonicalPredicate);
    }

    // static
    Recorder *
    Recorder::s_pInstance = nullptr;  

    // static
    void
    Recorder::Record(RecordedAction * action)
    {
        Recorder::Instance()->RecordAction(action);
    }

    void
    Recorder::RecordAction(RecordedAction * action)
    {
        m_vRecordedActions.push_back(action);
    }

   std::vector<RecordedAction *>*
   Recorder::GetRecordedActions()
   {
       return &m_vRecordedActions;
   }

   RecordedAction *
   Recorder::CreateAction(const char * actionName) 
   {
       RecordedAction * action = nullptr;

       // TODO : Make this automatically extensible, via registration of action sub-types.
       if (!strcmp("BindControlPort", actionName))
       {
           action = new BindControlPort();
       }
       else if (!strcmp("BindControlPropagationChannel", actionName))
       {
           action = new BindControlPropagationChannel();
       }
       else if (!strcmp("BindControlPropagationPort", actionName))
       {
           action = new BindControlPropagationPort();
       }
       else if (!strcmp("BindDescriptorPort", actionName))
       {
           action = new BindDescriptorPort();
       }
       else if (!strcmp("BindIterationScope", actionName))
       {
           action = new BindIterationScope();
       }
       else if (!strcmp("SetBlockAndGridSize", actionName))
       {
            action = new SetBlockAndGridSize();
       }
       else if (!strcmp("SetComputeGeometry", actionName))
       {
            action = new SetComputeGeometry();
       }
       else if (!strcmp("SetPredicationType", actionName))
       {
            action = new SetPredicationType();
       }
       else
        {
            assert(false);
       }
       return action;
   }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor: CJR: preserving Jon's implementation, although it is unclear
    ///             why he chose to declare an empty constructor... </summary>
    ///
    /// <remarks>   jcurrey, not sure when, moved from header by crossbac, 7/22/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Recorder::Recorder(
        VOID
        )
    {
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/22/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Recorder::~Recorder(
        VOID
        ) 
    {
        // CJR: cleaning up leaked recorded action objects...
        // CJR: is this thread-safe?
        std::vector<RecordedAction*>::iterator vi;
        for(vi=m_vRecordedActions.begin(); vi!=m_vRecordedActions.end(); vi++) {
            delete *vi;
        }
        m_vRecordedActions.clear();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Copy constructor. </summary>
    ///
    /// <remarks>   jcurrey, not sure when, moved from header by crossbac, 7/22/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Recorder::Recorder(Recorder const&) { }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Assignment operator. </summary>
    ///
    /// <remarks>   jcurrey, not sure when, moved from header by crossbac, 7/22/2013. </remarks>
    ///
    /// <returns>   A shallow copy of this object. </returns>
    ///-------------------------------------------------------------------------------------------------

    Recorder& Recorder::operator =(Recorder const&) { return *this; }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   static initializer to make creation of singleton object explicit (and
    ///             more importantly, to provide a booked for it's explicit deletion on exit). 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/22/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Recorder::Initialize(
        VOID
        )
    {
        // TODO: Use a lock if keep this as a singleton.
        // cjr: yes. this is super-unsafe. please fix, Jon.
        if (!s_pInstance)   
            s_pInstance = new Recorder;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   static de-initializer to make deletion of singleton object explicit. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/22/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------
    
    void 
    Recorder::Destroy(
        VOID
        )
    {
        if(s_pInstance) {
            delete s_pInstance;
            s_pInstance = nullptr;
        }
    }

    
    // static
    Recorder *
    Recorder::Instance()
    {
        Initialize();
        return s_pInstance;
    }


};
#endif