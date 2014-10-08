//--------------------------------------------------------------------------------------
// File: XMLWriter.cpp
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#ifdef XMLSUPPORT

#include "XMLWriter.h"
#include "graph.h"
#include "task.h"
#include "channel.h"
#include "port.h"
#include "outputport.h"
#include "primitive_types.h"
#include "CompiledKernel.h"
#include "graphInputChannel.h"
#include "graphOutputChannel.h"
#include "internalChannel.h"
#include "Recorder.h"

#include "cutask.h"
#include "dxtask.h"
#include "hosttask.h"
#include "cltask.h"

#pragma warning(disable : 4127)  // conditional expression is constant

using namespace std;
//using namespace PTask::Runtime;

namespace PTask {

    const wchar_t *
    XMLWriter::ToWChar(const char * str)
    {
        #pragma warning(push)
        #pragma warning(disable:4996)
        const size_t cSize = strlen(str)+1;
        wchar_t* strW = new wchar_t[cSize];
        mbstowcs(strW, str, cSize);
        return strW;
        #pragma warning(pop)
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   jcurrey, 5/5/2013. </remarks>
    ///
    /// <param name="filename">   The name of the file to write XML to. </param>
    ///-------------------------------------------------------------------------------------------------

    XMLWriter::XMLWriter(
        const char * filename
        )
    {
        HRESULT hr = S_OK;

        const WCHAR* filenameW = ToWChar(filename);
        
        //Open writeable output stream
        if (S_OK != (hr = SHCreateStreamOnFile(filenameW, STGM_CREATE | STGM_WRITE, &m_pOutFileStream)))
        {
            wprintf(L"Error creating XMLWriter for filename %s, error is %08.8lx\n", filenameW, hr);
                throw new XMLWriterException();
        }
        delete(filenameW);

        if (S_OK != (hr = CreateXmlWriter(__uuidof(IXmlWriter), (void**) &m_pWriter, NULL)))
        {
            wprintf(L"Error creating xml writer, error is %08.8lx\n", hr);
            throw new XMLWriterException();
        }

        if (S_OK != (hr = m_pWriter->SetOutput(m_pOutFileStream)))
        {
            wprintf(L"Error, Method: SetOutput, error is %08.8lx\n", hr);
            throw new XMLWriterException();
        }

        if (S_OK != (hr = m_pWriter->SetProperty(XmlWriterProperty_Indent, TRUE)))
        {
            wprintf(L"Error, Method: SetProperty XmlWriterProperty_Indent, error is %08.8lx\n", hr);
            throw new XMLWriterException();
        }

        if (S_OK != (hr = m_pWriter->WriteStartDocument(XmlStandalone_Omit)))
        {
            wprintf(L"Error, Method: WriteStartDocument, error is %08.8lx\n", hr);
            throw new XMLWriterException();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   jcurrey, 5/5/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    XMLWriter::~XMLWriter()
    {
        if(m_pOutFileStream) { m_pOutFileStream->Release(); m_pOutFileStream = NULL; } 
        if(m_pWriter) { m_pWriter->Release(); m_pWriter = NULL; } 

    }

    void AddTemplates(
        std::map<UINT, Port*>* pPortMap,
        std::set<DatablockTemplate*>* pTemplateSet
        )
    {
        std::map<UINT, Port*>::iterator pi;
        for(pi=pPortMap->begin(); pi!=pPortMap->end(); pi++) {
            Port* pPort = pi->second;
            DatablockTemplate * t = pPort->GetTemplate();
            if (t != nullptr)
            {
                pTemplateSet->insert(t);
            }
        }
    }

    void
    XMLWriter::WriteGraph(Graph * pGraph)
    {
        this->WriteElementStartTag("Graph");

        // Build sets of DatablockTemplates and CompiledKernels referenced in graph.
        std::set<DatablockTemplate*> templateSet;
        std::set<CompiledKernel*> kernelSet;
        std::map<std::string, Task*>::iterator taskIter;
        for(taskIter=pGraph->m_vTasks.begin(); taskIter!=pGraph->m_vTasks.end() ; ++taskIter) {
            Task * pTask = taskIter->second;

            AddTemplates(pTask->GetInputPortMap(), &templateSet);
            AddTemplates(pTask->GetConstantPortMap(), &templateSet);
            AddTemplates(pTask->GetOutputPortMap(), &templateSet);
            AddTemplates(pTask->GetMetaPortMap(), &templateSet);

            kernelSet.insert(pTask->GetCompiledKernel());
        }

        // Assign an (arbitrary) ID to each CompiledKernel, used to associate each Task with the 
        // appropriate kernel.
        //
        // (SourceFile + Operation cannot be assumed to uniquely identify a CompiledKernel instance. 
        // They won't e.g. in the case where multiple DX11 kernels are created using the same 
        // SourceFile and Operation, but different thread group dimensions.)
        std::vector<CompiledKernel*> kernelArray;
        std::set<CompiledKernel*>::iterator kernelSetIter;
        for(kernelSetIter=kernelSet.begin(); kernelSetIter!=kernelSet.end() ; ++kernelSetIter) {
            CompiledKernel * pKernel = *kernelSetIter;
            kernelArray.push_back(pKernel);
        }

        // Serialize DatablockTemplates.
        this->WriteComment("  DatablockTemplates  ");
        this->WriteElementStartTag("DatablockTemplates");
        UINT numTemplates = (UINT)templateSet.size();
        this->WriteElement("Count", numTemplates);
        std::set<DatablockTemplate*>::iterator templateIter;
        for(templateIter=templateSet.begin(); templateIter!=templateSet.end() ; ++templateIter) {
            DatablockTemplate * pTemplate = *templateIter;
            this->WriteDatablockTemplate(pTemplate);
        }
        this->WriteElementEndTag(); // </DatablockTemplates>

        // Serialize CompiledKernels.
        this->WriteComment("  CompiledKernels  ");
        this->WriteComment("  (KernelID is internal to this serialization. Used to associate each Task with its CompiledKernel)  ");
        this->WriteElementStartTag("CompiledKernels");
        UINT numKernels = (UINT)kernelArray.size();
        this->WriteElement("Count", numKernels);
        for(UINT i=0; i<numKernels; i++) {
            CompiledKernel* pKernel = kernelArray[i];
            this->WriteCompiledKernel(pKernel, i);
        }
        this->WriteElementEndTag(); // </CompiledKernels>

        // Serialize Tasks.
        this->WriteComment("  Tasks  ");
        this->WriteElementStartTag("Tasks");
        UINT numTasks = (UINT)pGraph->m_vTasks.size();
        this->WriteElement("Count", numTasks);
        for(taskIter=pGraph->m_vTasks.begin(); taskIter!=pGraph->m_vTasks.end() ; ++taskIter) {
            Task * pTask = taskIter->second;
            // Look up the kernel ID of the CompiledKernel associated with this Task.
            CompiledKernel * pKernel = pTask->GetCompiledKernel();
            int kernelID = -1;
            for(int i=0; i<kernelArray.size(); i++)
            {
                if (kernelArray[i] == pKernel)
                {
                    // Sanity check: Kernel array was assigned from a set, 
                    // so each kernel should appear only once.
                    assert(-1 == kernelID);
                    kernelID = i;
                }
            }
            assert(-1 != kernelID);
            this->WriteTask(pTask, kernelID);
        }
        this->WriteElementEndTag(); // </Tasks>

        // Serialize Channels.
        this->WriteComment("  Channels  ");
        this->WriteComment("  (Ports are referenced by their UniqueID)  ");
        this->WriteElementStartTag("Channels");
        UINT numChannels = (UINT)pGraph->m_vChannels.size();
        this->WriteElement("Count", numChannels);
        map<std::string, Channel*>::iterator channelIter;
        for(channelIter=pGraph->m_vChannels.begin(); channelIter!=pGraph->m_vChannels.end() ; ++channelIter) {
            Channel * pChannel = channelIter->second;
            this->WriteChannel(pChannel);
        }
        this->WriteElementEndTag(); // </Channels>

        // Serialize recorded actions.
        this->WriteElementStartTag("Actions");
        std::vector<RecordedAction*>* recordedActions = 
            Recorder::Instance()->GetRecordedActions();
        int numActions = static_cast<int>(recordedActions->size());
        this->WriteElement("Count", numActions);
        std::vector<RecordedAction*>::iterator vi;
        for(vi = recordedActions->begin(); vi != recordedActions->end(); vi++)
        {
            RecordedAction * action = *vi;
            this->WriteElementStartTag("Action");
            this->WriteElement("ActionName", action->GetName());
            action->Write(this);
            this->WriteElementEndTag(); // </Action>
        }
        this->WriteElementEndTag(); // </Actions>

        this->WriteElementEndTag(); // </Graph>
        this->WriteEndDocument();
    }

    void
    XMLWriter::WriteElementStartTag(const char * elementName)
    {
        const WCHAR * elementNameW = ToWChar(elementName);
        m_pWriter->WriteStartElement(NULL, elementNameW, NULL);
        delete elementNameW;
    }

    void
    XMLWriter::WriteElementText(const char * text)
    {
        const WCHAR * textW = ToWChar(text);
        m_pWriter->WriteString(textW);
        delete textW;
    }

    void
    XMLWriter::WriteElementEndTag()
    {
        m_pWriter->WriteFullEndElement();
    }

    void
    XMLWriter::WriteComment(const char * comment)
    {
        const WCHAR * commentW = ToWChar(comment);
        m_pWriter->WriteComment(commentW);
        delete commentW;
    }

    // WriteEndDocument closes any open elements or attributes
    void
    XMLWriter::WriteEndDocument()
    {
        HRESULT hr = S_OK;
        if (S_OK != (hr = m_pWriter->WriteEndDocument()))
        {
            wprintf(L"Error, Method: WriteEndDocument, error is %08.8lx\n", hr);
            throw new XMLWriterException();
        }
        if (S_OK != (hr = m_pWriter->Flush()))
        {
            wprintf(L"Error, Method: Flush, error is %08.8lx\n", hr);
            throw new XMLWriterException();
        }
    }

    void
    XMLWriter::WriteElement(const char * elementName, const char * text)
    {
        this->WriteElementStartTag(elementName);
        this->WriteElementText(text);
        this->WriteElementEndTag();
    }

    void
    XMLWriter::WriteElement(const char * elementName, int elementValue)
    {
        const int BUFSIZE = 100;
        char valueString[BUFSIZE];
        sprintf_s(valueString, BUFSIZE, "%d", elementValue);
        this->WriteElementStartTag(elementName);
        this->WriteElementText(valueString);
        this->WriteElementEndTag();
    }

    void
    XMLWriter::WriteElement(const char * elementName, unsigned int elementValue)
    {
        const int BUFSIZE = 100;
        char valueString[BUFSIZE];
        sprintf_s(valueString, BUFSIZE, "%u", elementValue);
        this->WriteElementStartTag(elementName);
        this->WriteElementText(valueString);
        this->WriteElementEndTag();
    }

    void
    XMLWriter::WriteElement(const char * elementName, bool elementValue)
    {
        this->WriteElementStartTag(elementName);
        this->WriteElementText(elementValue ? "true" : "false");
        this->WriteElementEndTag();
    }

    void
    XMLWriter::WriteCompiledKernel(CompiledKernel * pCompiledKernel, int kernelID)
    {
        this->WriteElementStartTag("CompiledKernel");
        this->WriteElement("KernelID", kernelID);
        this->WriteElement("SourceFile", pCompiledKernel->GetSourceFile());
        this->WriteElement("Operation", pCompiledKernel->GetOperation());

        /* TODO JC: Add support for thread group geometry. Only used for DirectX.
        this->WriteElement("ThreadGroupX", pCompiledKernel->);
        this->WriteElement("ThreadGroupY", pCompiledKernel->);
        this->WriteElement("ThreadGroupZ", pCompiledKernel->); */

        this->WriteElementEndTag(); // </CompiledKernel>
    }

    void
    XMLWriter::WriteDatablockTemplate(DatablockTemplate * pTemplate)
    {
        this->WriteElementStartTag("DatablockTemplate");
        this->WriteElement("Name", pTemplate->GetTemplateName());
        this->WriteElement("Stride", pTemplate->GetStride());
        this->WriteElement("ParamType", (int)(pTemplate->GetParameterBaseType()));
        this->WriteElement("ElementsX", pTemplate->GetXElementCount());
        this->WriteElement("ElementsY", pTemplate->GetYElementCount());
        this->WriteElement("ElementsZ", pTemplate->GetZElementCount());
        this->WriteElement("IsRecordStream", (pTemplate->DescribesRecordStream() == TRUE));
        this->WriteElement("IsByteAddressable", pTemplate->IsByteAddressable());

        if (pTemplate->GetPitch() != 
            (pTemplate->GetStride() * pTemplate->GetBufferDimensions(DBDATA_IDX).uiXElements))
        {
            printf("TODO : Add support for Pitch");
            throw new XMLWriterException();
        }

        if (pTemplate->GetPitch() != 
            (pTemplate->GetStride() * pTemplate->GetBufferDimensions(DBDATA_IDX).uiXElements))
        {
            printf("TODO : Add support for Pitch");
            throw new XMLWriterException();
        }

        /* JC TODO Handle below params to various ctors ... or at least assert if detect use:

        __in BUFFERDIMENSIONS * pBufferDims, 
        __in unsigned int       uiNumBufferDims, 
            */

        this->WriteElementEndTag(); // </DatablockTemplate>
    }

    void
    XMLWriter::WriteTask(Task * pTask, int kernelID)
    {
        this->WriteElementStartTag("Task");

        this->WriteElement("Name", pTask->GetTaskName());
        this->WriteElement("KernelID", kernelID);
        this->WriteElement("AcceleratorClass", (int)(pTask->GetAcceleratorClass()));

        this->WriteElementStartTag("InputPorts");
        UINT inputPortCount = (UINT) // casting size_t to UINT. Hopefully don't have more than 2^32 ports.
            (UINT) pTask->GetInputPortMap()->size() +
            (UINT) pTask->GetConstantPortMap()->size() +
            (UINT) pTask->GetMetaPortMap()->size();
        this->WriteElement("Count", inputPortCount);
        // Could put all input ports into a map, keyed on OriginalIndex
        // and output in index order. Not bothering for now, as would
        // only affect readability of the XML to a human. OriginalIndex
        // will ensure they are put in the right order when deserialized.
        this->WritePorts(pTask->GetInputPortMap());
        this->WritePorts(pTask->GetConstantPortMap());
        this->WritePorts(pTask->GetMetaPortMap());
        this->WriteElementEndTag(); // </InputPorts>

        this->WriteElementStartTag("OutputPorts");
        UINT outputPortCount = (UINT) // casting size_t to UINT. Hopefully don't have more than 2^32 ports.
            pTask->GetOutputPortMap()->size();
        this->WriteElement("Count", outputPortCount);
        this->WritePorts(pTask->GetOutputPortMap());
        this->WriteElementEndTag(); // </OutputPorts>

        this->WriteElementEndTag(); // </Task>
    }

    void
    XMLWriter::WritePorts(std::map<UINT, Port*>* pPorts)
    {
        std::map<UINT, Port*>::iterator portIter;
        for(portIter=pPorts->begin(); portIter!=pPorts->end() ; ++portIter) {
            // UINT key = portIter->first;
            // this->WriteElement("Key", key);
            Port * pPort = portIter->second;
            this->WritePort(pPort);
        }
    }

    void
    XMLWriter::WritePort(Port * pPort)
    {
        this->WriteElementStartTag("Port");
        this->WriteElement("UniqueID", pPort->GetUID());
        this->WriteElement("OriginalIndex", pPort->GetOriginalIndex());

        // TODO JC Map type to/from string for readability in XML
        this->WriteElement("Type", (int)(pPort->GetPortType()));

        assert(nullptr != pPort->GetTemplate()->GetTemplateName());
        this->WriteElement("DatablockTemplate", pPort->GetTemplate()->GetTemplateName());
   
        // JC need to make VariableBinding nullable?
        assert(nullptr != pPort->GetVariableBinding());
        this->WriteElement("VariableBinding", pPort->GetVariableBinding());
        this->WriteElement("ParameterIndex", pPort->GetFormalParameterIndex()); // JC FormalParameterIndex same as KernelParameterIndex? //JC int v. uint?
        this->WriteElement("InOutIndex", pPort->GetInOutRoutingIndex()); // JC InOutROutingIndex same as InOutIndex // JC int v. uint?

//        this->WriteControlPropagationInfo(pPort);

        this->WriteElementEndTag(); // </Port>
    }

/*
    void
    XMLWriter::WriteControlPropagationInfo(Port * pPort)
    {
        this->WriteElementStartTag("ControlPropagationPorts");
        UINT portCount = (UINT) // casting size_t to UINT.
            (pPort->m_pControlPropagationPorts.size());
        this->WriteElement("Count", portCount);
        if (portCount > 0)
        {
            std::vector<Port*>::iterator vi;
            for(vi = pPort->m_pControlPropagationPorts.begin(); 
                vi != pPort->m_pControlPropagationPorts.end();
                vi++)
            {
                this->WriteElement("UniqueID", (*vi)->GetUID());
            }
        }
        this->WriteElementEndTag(); // ControlPropagationPorts

        int controlPropagationSourceUID = -1;
        Port * controlPropagationSource = pPort->m_pControlPropagationSource;
        if (nullptr != controlPropagationSource)
        {
            controlPropagationSourceUID = (int)(controlPropagationSource->GetUID()); // casting UINT to int
        }
        this->WriteElement("ControlPropagationSource", controlPropagationSourceUID);

        this->WriteElementStartTag("ControlPropagationChannels");
        UINT channelCount = (UINT) // casting size_t to UINT.
            (pPort->m_pControlPropagationChannels.size());
        this->WriteElement("Count", channelCount);
        if (channelCount > 0)
        {
            std::set<Channel*>::iterator si;
            for(si = pPort->m_pControlPropagationChannels.begin(); 
                si != pPort->m_pControlPropagationChannels.end();
                si++)
            {
                this->WriteElement("Name", (*si)->GetName());
            }
        }
        this->WriteElementEndTag(); // ControlPropagationChannels

        this->WriteElementStartTag("GatedPorts");
        portCount = (UINT) // casting size_t to UINT.
            (pPort->m_pGatedPorts.size());
        this->WriteElement("Count", portCount);
        if (portCount > 0)
        {
            std::vector<Port*>::iterator vi;
            for(vi = pPort->m_pGatedPorts.begin(); 
                vi != pPort->m_pGatedPorts.end();
                vi++)
            {
                this->WriteElement("UniqueID", (*vi)->GetUID());
            }
        }
        this->WriteElementEndTag(); // GatedPorts

        // Output port has a control port if BindControlPort was called on it.
        if (PORTTYPE::OUTPUT_PORT == pPort->GetPortType())
        {
            OutputPort * pOutputPort = dynamic_cast<OutputPort *>(pPort);
            int controlPortUID = -1;
            Port * controlPort = pOutputPort->m_pControlPort;
            if (nullptr != controlPort)
            {
                controlPortUID = (int)(controlPort->GetUID()); // casting UINT to int
            }
            this->WriteElementStartTag("OutputPortControlPort");
            this->WriteElement("UniqueID", controlPortUID);
            this->WriteElement("InitialPortStateOpen", pOutputPort->m_bInitialPortStateOpen);
            this->WriteElementEndTag(); // OutputPortControlPort
        }

        this->WriteElementStartTag("DescriptorPorts");
        portCount = (UINT) // casting size_t to UINT.
            (pPort->m_vDescriptorPorts.size());
        this->WriteElement("Count", portCount);
        if (portCount > 0)
        {
            std::vector<DEFERREDPORTDESC*>::iterator vi;
            for(vi = pPort->m_vDescriptorPorts.begin(); 
                vi != pPort->m_vDescriptorPorts.end();
                vi++)
            {
                DEFERREDPORTDESC* descriptorRecord = *vi;
                this->WriteElement("UniqueID", descriptorRecord->pPort->GetUID());
                this->WriteElement("DescriptorFunction", (int)(descriptorRecord->func));
            }
        }
        this->WriteElementEndTag(); // DescriptorPorts

        this->WriteElementStartTag("DeferredChannels");
        channelCount = (UINT) // casting size_t to UINT.
            (pPort->m_vDeferredChannels.size());
        this->WriteElement("Count", channelCount);
        if (channelCount > 0)
        {
            std::vector<DEFERREDCHANNELDESC*>::iterator vi;
            for(vi = pPort->m_vDeferredChannels.begin(); 
                vi != pPort->m_vDeferredChannels.end();
                vi++)
            {
                DEFERREDCHANNELDESC* channelDescriptor = *vi;
                this->WriteElement("Name", channelDescriptor->pChannel->GetName());
                this->WriteElement("DescriptorFunction", (int)(channelDescriptor->func));
            }
        }
        this->WriteElementEndTag(); // DeferredChannels
    }
*/
    void
    XMLWriter::WriteChannel(Channel * pChannel)
    {
        this->WriteElementStartTag("Channel");

        this->WriteElement("Name", pChannel->GetName());

        // TODO JC Map type to/from string for readability in XML
        this->WriteElement("Type", (int)(pChannel->GetType()));

        // Output -1 for <SourcePort> if there isn't one.
        int sourcePortID = -1; // HACK : Using int to hold a UINT. Should not be a problem in practice.
        Port * pSourcePort = pChannel->GetBoundPort(CE_SRC);
        if (nullptr != pSourcePort)
        {
            sourcePortID = (int)pSourcePort->GetUID(); 
        }
            this->WriteElement("SourcePort", sourcePortID);

        // TODO JC Add support for multi-channels and their multiple output ports.
        assert(CT_MULTI != pChannel->GetType());

        // Output -1 for <DestinationPort> if there isn't one.
        int destinationPortID = -1; // HACK : Using int to hold a UINT. Should not be a problem in practice.
        Port * pDestinationPort = pChannel->GetBoundPort(CE_DST);
        if (nullptr != pDestinationPort)
        {
            destinationPortID = pDestinationPort->GetUID();
        }
        this->WriteElement("DestinationPort", destinationPortID);
/*
        int controlPropagationSourceUID = -1;
        Port * controlPropagationSource = pChannel->m_pControlPropagationSource;
        if (nullptr != controlPropagationSource)
        {
            controlPropagationSourceUID = (int)(controlPropagationSource->GetUID()); // casting UINT to int
        }
        this->WriteElement("ControlPropagationSource", controlPropagationSourceUID);

        this->WriteElementStartTag("EndpointPredication");
        this->WriteChannelEndpointPredication(pChannel, CE_SRC);
        this->WriteChannelEndpointPredication(pChannel, CE_DST);
        this->WriteElementEndTag(); // </EndpointPredication>
*/
        this->WriteElementEndTag(); // </Channel>
    }
/*
    void
    XMLWriter::WriteChannelEndpointPredication(Channel * pChannel, CHANNELENDPOINTTYPE eEndpoint)
    {
        this->WriteElement("EndpointType", (int)(pChannel->m_vPredicators[eEndpoint].eEndpoint));
        this->WriteElement("CanonicalPredicate", (int)(pChannel->m_vPredicators[eEndpoint].eCanonicalPredicate));
        if (nullptr != pChannel->m_vPredicators[eEndpoint].lpfnPredicate)
        {
            printf("User-defined predicate functions currently not supported.\n");
            throw new XMLWriterException();
        }
        this->WriteElement("FailureAction", (int)(pChannel->m_vPredicators[eEndpoint].ePredicateFailureAction));
    }
*/
};

#endif