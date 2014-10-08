//--------------------------------------------------------------------------------------
// File: XMLReader.cpp
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#ifdef XMLSUPPORT

#include "XMLReader.h"
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

#define _V(x) {                                                                                 \
    if(!SUCCEEDED(hr = (x))) {                                                                  \
        PTask::Runtime::HandleError("%s::%s: XMLReader error at line %d (%s -> hr=%08.8lx)\n",  \
                                    __FILE__,  __FUNCTION__, __LINE__, #x, hr); } }               

using namespace std;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Copy a char string to a newly allocated wchar string. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. Re-writing original code from jcurrey, which
    ///             had some difficult-to-find leaks and failures to use vector delete
    ///             that were killing the ptask regression tests.
    ///             </remarks>
    ///
    /// <param name="pString">  the string to copy. </param>
    ///
    /// <returns>   null if it fails, else a wchar_t*. </returns>
    ///-------------------------------------------------------------------------------------------------

    const wchar_t *
    XMLReader::AllocWideStringCopy(
        const char * pString
        ) 
    { 
        size_t szConverted = 0;
        const size_t bufSize = strlen(pString)+1;
        wchar_t* strW = new wchar_t[bufSize];
        m_wAllocs.insert(strW);
        mbstowcs_s(&szConverted, strW, bufSize, pString, bufSize); // count must include \0 terminator
        return strW;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate string copy. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. Re-writing original code from jcurrey, which
    ///             had some difficult-to-find leaks and failures to use vector delete
    ///             that were killing the ptask regression tests.
    ///             </remarks>
    ///
    /// <param name="pwString"> The password string. </param>
    ///
    /// <returns>   null if it fails, else a char*. </returns>
    ///-------------------------------------------------------------------------------------------------

    const char *
    XMLReader::AllocStringCopy(
        LPCWSTR pwString
        )
    {
        size_t bufSize = wcslen(pwString)+1;
        char *str = new char[bufSize];
        m_cAllocs.insert(str);
        size_t   charsConverted;
        wcstombs_s(&charsConverted, str, bufSize, pwString, bufSize - 1 ); // count must exclude \0 terminator
        return str;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Free a wide-character string. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <param name="str">  The. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    XMLReader::FreeWideString(
        const wchar_t * str
        )
    {
        assert(m_wAllocs.find(str) != m_wAllocs.end());
        m_wAllocs.erase(str);
        delete [] str;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Free string. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <param name="str">  The. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    XMLReader::FreeString(
        const char * str
        )
    {
        assert(m_cAllocs.find(str) != m_cAllocs.end());
        m_cAllocs.erase(str);
        delete [] str;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   jcurrey, 5/8/2013. </remarks>
    ///
    /// <param name="filename">   The name of the file to read XML from. </param>
    ///-------------------------------------------------------------------------------------------------


    XMLReader::XMLReader(
        const char * filename
        )
    {
        HRESULT hr = S_OK;        
        m_pGraph = NULL;
        m_pInFileStream = NULL;
        m_pReader = NULL;

        const WCHAR* filenameW = AllocWideStringCopy(filename);        
        _V(SHCreateStreamOnFile(filenameW, STGM_READ, &m_pInFileStream)); // Open read-only input stream        
        _V(CreateXmlReader(__uuidof(IXmlReader), (void**) &m_pReader, NULL));    
        m_pReader->SetProperty(XmlReaderProperty_DtdProcessing, DtdProcessing_Prohibit);    
        _V(m_pReader->SetInput(m_pInFileStream));
        FreeWideString(filenameW);  
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   jcurrey, 5/8/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    XMLReader::~XMLReader()
    {
        #pragma warning(disable:4127)
        if(m_pInFileStream) { pInFileStream->Release(); pInFileStream = NULL; } 
        if(m_pReader) { m_pReader->Release(); m_pReader = NULL; } 
        #pragma warning(default:4127)

        // if necessary, go clean up any irresponsibly
        // handled allocations used to back various reads
        assert(m_wAllocs.size() == 0);
        assert(m_cAllocs.size() == 0);
        std::set<const char*>::iterator ci;
        std::set<const wchar_t*>::iterator wi;
        for(ci=m_cAllocs.begin(); ci!=m_cAllocs.end(); ci++)             
            delete [] (*ci);
        for(wi=m_wAllocs.begin(); wi!=m_wAllocs.end(); wi++)             
            delete [] (*wi);

        m_wAllocs.clear(); 
        m_cAllocs.clear();
        m_templateMap.clear();
        m_kernelMap.clear();
        m_portMap.clear();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the graph. </summary>
    ///
    /// <remarks>   jcurrey </remarks>
    ///
    /// <returns>   null if it fails, else the graph. </returns>
    ///-------------------------------------------------------------------------------------------------

    Graph *
    XMLReader::GetGraph()
    {
        return m_pGraph;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a port. </summary>
    ///
    /// <remarks>   jcurrey </remarks>
    ///
    /// <param name="portUID">  The port UID. </param>
    ///
    /// <returns>   null if it fails, else the port. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port *
    XMLReader::GetPort(UINT portUID)
    {
        return m_portMap[portUID];
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads the templates. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <param name="parameter1">   The first parameter. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    XMLReader::ReadTemplates(
        VOID
        )
    {
        DatablockTemplate * pTemplate = NULL;
        ReadElementStartTag("DatablockTemplates");
        int numTemplates = ReadIntegerElement("Count");
        for (int i=0; i<numTemplates; i++) {
            pTemplate = ReadDatablockTemplate();
            if(pTemplate == NULL) return FALSE;
            m_templateMap[pTemplate->GetTemplateName()] = pTemplate;
        }
        ReadElementEndTag("DatablockTemplates");
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads the kernels. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <param name="parameter1">   The first parameter. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    XMLReader::ReadKernels(
        VOID
        )
    {
        int kernelID = 0;
        CompiledKernel * pKernel = NULL;
        ReadElementStartTag("CompiledKernels");
        int nKernels = ReadIntegerElement("Count");
        for (int i=0; i<nKernels; i++) {
            pKernel = ReadCompiledKernel(kernelID);
            if(pKernel == NULL) 
                return FALSE;
            m_kernelMap[kernelID] = pKernel;
        }
        ReadElementEndTag("CompiledKernels");
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads the tasks. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <param name="parameter1">   The first parameter. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    XMLReader::ReadTasks(
        VOID
        )
    {
        ReadElementStartTag("Tasks");
        int nTasks = ReadIntegerElement("Count");
        for(int i=0; i<nTasks; i++) {
            if(ReadTask() == NULL)
                return FALSE;
        }
        ReadElementEndTag("Tasks");
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads the channels. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <param name="parameter1">   The first parameter. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    XMLReader::ReadChannels(
        VOID
        )
    {
        if(!ReadElementStartTag("Channels"))
            return FALSE;
        int nChannels = ReadIntegerElement("Count");
        for (int i=0; i<nChannels; i++) 
            ReadChannel();
        return ReadElementEndTag("Channels");
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads the actions. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <param name="parameter1">   The first parameter. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    XMLReader::ReadActions(
        VOID
        )
    {
        ReadElementStartTag("Actions");
        int nActions = ReadIntegerElement("Count");
        Recorder * recorder = Recorder::Instance();
        for (int i=0; i<nActions; i++) {
            
            ReadElementStartTag("Action");
            const char * lpszActionName = ReadTextElement("ActionName");
            RecordedAction * action = recorder->CreateAction(lpszActionName);
            FreeString(lpszActionName);
            action->Read(this);
            action->Replay(this);            
            delete action;                  // cjr: fix 1/17/2014
            ReadElementEndTag("Action");
        }
        ReadElementEndTag("Actions");
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads a string element an returns it through an std::string so that other classes
    ///             that instantiate themselves through the reader cannot wind up with pointers to
    ///             temporary arrays returned as part of the internal read process (which was
    ///             happening rather a lot in the previous implementation).
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <param name="lpszElementName">  Name of the element. </param>
    /// <param name="szElementValue">   [in,out] The element value. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    XMLReader::ReadStringElement(
        const char * lpszElementName,
        std::string& szElementValue
        )
    {
        const char * lpszValue = ReadTextElement(lpszElementName);
        szElementValue = lpszValue;
        FreeString(lpszValue);
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads a graph. </summary>
    ///
    /// <remarks>   jcurrey, cjr rewrite 1/17/14 to deal with memory leaks. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    XMLReader::ReadGraph(
        Graph * pGraph
        )
    {
        m_pGraph = pGraph;
        m_templateMap.clear();
        m_kernelMap.clear();
        m_portMap.clear();

        ReadNextNode(XmlNodeType_XmlDeclaration);
        ReadElementStartTag("Graph");
        
        if(!ReadTemplates() ||
           !ReadKernels() ||
           !ReadTasks() ||
           !ReadChannels() ||
           !ReadActions())
            return FALSE;

        ReadElementEndTag("Graph");
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads next node. </summary>
    ///
    /// <remarks>   jcurrey original. </remarks>
    ///
    /// <exception cref="XMLReaderException">   Thrown when an XML Reader error condition occurs. </exception>
    ///
    /// <param name="requiredType"> Type of the required. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    XMLReader::ReadNextNode(
        XmlNodeType requiredType
        )
    {
        HRESULT hr = S_OK;
        XmlNodeType nodeType;

        do {
            _V(m_pReader->Read(&nodeType));
        } while(XmlNodeType_Comment == nodeType ||
                XmlNodeType_Whitespace == nodeType);

        return nodeType == requiredType;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads element start tag. </summary>
    ///
    /// <remarks>   jcurrey originally </remarks>
    ///
    /// <param name="lpszElementName">  Name of the element. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    XMLReader::ReadElementStartTag(
        const char * lpszElementName
        )
    {
       #pragma warning(push)
       #pragma warning(disable:4706)
        HRESULT hr = S_OK;
        const WCHAR* actualElementNameW = NULL;
        const WCHAR * lpszwElementNameW = AllocWideStringCopy(lpszElementName);
        ReadNextNode(XmlNodeType_Element);
        _V(m_pReader->GetLocalName(&actualElementNameW, NULL));
        BOOL bResult = (wcscmp(lpszwElementNameW, actualElementNameW)) == 0;
        FreeWideString(lpszwElementNameW);
        return bResult;
        #pragma warning(pop)        
    }

    BOOL
    XMLReader::ReadElementText(
        const char *& text
        )
    {
        // Next node should be of text type.
        if(!ReadNextNode(XmlNodeType_Text))
            return FALSE;

        // We do not delete textW: IXMLReader retains ownership of the string 
        // returned by GetValue, as noted in its MSDN documentation: 
        // http://msdn.microsoft.com/en-us/library/windows/desktop/ms752870(v=vs.85).aspx)
        LPCWSTR textW;
        m_pReader->GetValue(&textW, nullptr);
        text = AllocStringCopy(textW);
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads element end tag. </summary>
    ///
    /// <remarks>   jcurrey originally. </remarks>
    ///
    /// <param name="lpszElementName">  Name of the required element. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    XMLReader::ReadElementEndTag(
        const char * lpszElementName
        )
    {
        #pragma warning(push)
        #pragma warning(disable:4706)
        const WCHAR * requiredElementNameW = AllocWideStringCopy(lpszElementName);
        HRESULT hr = S_OK;

        if(!ReadNextNode(XmlNodeType_EndElement)) {
            FreeWideString(requiredElementNameW);
            return FALSE;
        }

        const WCHAR* actualElementNameW;
        _V(m_pReader->GetLocalName(&actualElementNameW, NULL));
        BOOL bResult = (wcscmp(requiredElementNameW, actualElementNameW)) == 0;
        FreeWideString(requiredElementNameW);
        return bResult;
        #pragma warning(pop)
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads  atext element. </summary>
    ///
    /// <remarks>   jcurrey originally </remarks>
    ///
    /// <param name="elementName">  Name of the element. </param>
    ///
    /// <returns>   null if it fails, else the text element. </returns>
    ///-------------------------------------------------------------------------------------------------

    const char * 
    XMLReader::ReadTextElement(
        const char * elementName
        )
    {
        const char * szValue = NULL;
        if(!ReadElementStartTag(elementName))
            return NULL;
        if(!ReadElementText(szValue))
            return NULL;
        ReadElementEndTag(elementName);
        return szValue;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads integer element. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <param name="elementName">  Name of the element. </param>
    ///
    /// <returns>   The integer element. </returns>
    ///-------------------------------------------------------------------------------------------------

    int
    XMLReader::ReadIntegerElement(
        const char * elementName
        )
    {
        int nValue = 0;
        const char * valueString;
        if(!ReadElementStartTag(elementName))
            return 0;
        if(!ReadElementText(valueString))
            return 0;
        nValue = atoi(valueString);
        FreeString(valueString);
        ReadElementEndTag(elementName);
        return nValue;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads uint element. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <param name="elementName">  Name of the element. </param>
    ///
    /// <returns>   The uint element. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    XMLReader::ReadUINTElement(
        const char * elementName
        )
    {
        UINT nValue;
        const char * valueString;
        if(!ReadElementStartTag(elementName) ||
           !ReadElementText(valueString))
           return 0;
        nValue = (UINT)atoi(valueString); 
        FreeString(valueString);
        ReadElementEndTag(elementName);
        return nValue;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads boolean element. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <exception cref="XMLReaderException">   Thrown when an XML Reader error condition occurs. </exception>
    ///
    /// <param name="elementName">  Name of the element. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool
    XMLReader::ReadBooleanElement(
        const char * elementName
        )
    {
        bool elementValue = false;
        const char * valueString;
        if(!ReadElementStartTag(elementName))
            return false;
        if(!ReadElementText(valueString))
            return false;
        if(!_stricmp(valueString, "true"))
            elementValue = true;
        else if (!_stricmp(valueString, "false"))
            elementValue = false;
        else
            throw new XMLReaderException();        
        FreeString(valueString);
        ReadElementEndTag(elementName);
        return elementValue;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads datablock template. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <returns>   null if it fails, else the datablock template. </returns>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate * 
    XMLReader::ReadDatablockTemplate(
        VOID
        )
    {
        DatablockTemplate * pTemplate = NULL;
        if(!ReadElementStartTag("DatablockTemplate"))
            return NULL;

        const char * name;
        UINT stride;
        int pt;
        UINT elementsX;
        UINT elementsY;
        UINT elementsZ;
        bool isRecordStream;
        bool isByteAddressable;

        name = ReadTextElement("Name");
        stride = ReadUINTElement("Stride");
        pt = ReadIntegerElement("ParamType");
        elementsX = ReadUINTElement("ElementsX");
        elementsY = ReadUINTElement("ElementsY");
        elementsZ = ReadUINTElement("ElementsZ");
        isRecordStream = ReadBooleanElement("IsRecordStream");
        isByteAddressable = ReadBooleanElement("IsByteAddressable");

        PTASK_PARM_TYPE paramType = (PTASK_PARM_TYPE)pt;
        if (PTPARM_NONE == paramType) {
            pTemplate = PTask::Runtime::GetDatablockTemplate(
                (char*)name, stride, elementsX, elementsY, elementsZ,
                isRecordStream, isByteAddressable);
        } else {
            pTemplate = PTask::Runtime::GetDatablockTemplate(
                (char*)name, stride, paramType);
        }

        ReadElementEndTag("DatablockTemplate");
        FreeString(name);
        return pTemplate;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads a compiled kernel. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <exception cref="XMLReaderException">   Thrown when an XML Reader error condition occurs. </exception>
    ///
    /// <param name="pKernel">  [in,out] [in,out] If non-null, the kernel. </param>
    /// <param name="kernelID"> [in,out] Identifier for the kernel. </param>
    ///-------------------------------------------------------------------------------------------------

    CompiledKernel * 
    XMLReader::ReadCompiledKernel(
        int & kernelID
        )
    {
        CompiledKernel * pKernel = NULL;
        if(!ReadElementStartTag("CompiledKernel"))
            return NULL;

        const int COBBUFSIZE = 4906;
        char szCompilerOutputBuffer[COBBUFSIZE];
        kernelID = ReadIntegerElement("KernelID");
        const char * sourceFile = ReadTextElement("SourceFile");
        const char * operation = ReadTextElement("Operation");

        pKernel = PTask::Runtime::GetCompiledKernel((char *)sourceFile, 
                                                    (char *)operation, 
                                                    szCompilerOutputBuffer, 
                                                    COBBUFSIZE);
        
        if (nullptr == pKernel) {
            printf("Failed to compile %s\\%s\n", sourceFile, operation);
            printf("Compiler output:\n\n%s\n\n", szCompilerOutputBuffer);
            return NULL;
        }

        ReadElementEndTag("CompiledKernel");
        FreeString(sourceFile); 
        FreeString(operation);  
        return pKernel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads the task. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <returns>   null if it fails, else the task. </returns>
    ///-------------------------------------------------------------------------------------------------

    Task * 
    XMLReader::ReadTask(
        VOID
        )
    {
        if(!ReadElementStartTag("Task"))
            return NULL;

        const char * name;
        int kernelID;
        int acceleratorClass;
        int numInPorts;
/*        BOOL geometryExplicit;
        UINT preferredXDim;
        UINT preferredYDim;
        UINT preferredZDim;
        BOOL threadBlockSizesExplicit;
        int blockXDim;
        int blockYDim;
        int blockZDim;
        int gridXDim;
        int gridYDim;
        int gridZDim;*/
        Port ** inPorts;
        int numOutPorts;
        Port ** outPorts;

        name = ReadTextElement("Name");
        kernelID = ReadIntegerElement("KernelID");
        acceleratorClass = ReadIntegerElement("AcceleratorClass");
/*
        this->ReadElement("GeometryExplicit", geometryExplicit);
        this->ReadElement("PreferredXDim", preferredXDim);
        this->ReadElement("PreferredYDim", preferredYDim);
        this->ReadElement("PreferredZDim", preferredZDim);

        if (ACCELERATOR_CLASS_CUDA == acceleratorClass ||
            ACCELERATOR_CLASS_HOST == acceleratorClass)
        {
            this->ReadElement("ThreadBlockSizesExplicit", threadBlockSizesExplicit);
            this->ReadElement("BlockXDim", blockXDim);
            this->ReadElement("BlockYDim", blockYDim);
            this->ReadElement("BlockZDim", blockZDim);
            this->ReadElement("GridXDim", gridXDim);
            this->ReadElement("GridYDim", gridYDim);
            this->ReadElement("GridZDim", gridZDim);
        }
*/
        if(!ReadElementStartTag("InputPorts"))
            return NULL;
        numInPorts = ReadIntegerElement("Count");
        inPorts = new Port*[numInPorts];
        for (int i=0; i<numInPorts; i++) {
            Port * pPort = ReadPort();
            inPorts[pPort->GetOriginalIndex()] = pPort;
        }
        if(!ReadElementEndTag("InputPorts"))
            return NULL;

        if(!ReadElementStartTag("OutputPorts"))
            return NULL;
        numOutPorts = ReadIntegerElement("Count");
        outPorts = new Port*[numOutPorts];
        for (int i=0; i<numOutPorts; i++) {
            Port * pPort = ReadPort();
            outPorts[pPort->GetOriginalIndex()] = pPort;
        }
        if(!ReadElementEndTag("OutputPorts"))
            return NULL;

        CompiledKernel * pKernel = m_kernelMap[kernelID];
        Task * pTask = m_pGraph->AddTask(
            pKernel, numInPorts, inPorts, numOutPorts, outPorts, (char*)name); 
        assert(pTask);
        UNREFERENCED_PARAMETER(pTask);

        assert(acceleratorClass == pTask->GetAcceleratorClass());
/*        switch(acceleratorClass)
        {
        case ACCELERATOR_CLASS_CUDA:
            {
                CUTask * task = dynamic_cast<CUTask *>(pTask);
                task->m_bGeometryExplicit = geometryExplicit;
                task->m_nPreferredXDim = preferredXDim;
                task->m_nPreferredYDim = preferredYDim;
                task->m_nPreferredZDim = preferredZDim;

                task->m_bThreadBlockSizesExplicit = threadBlockSizesExplicit;
                task->m_pThreadBlockSize.x = blockXDim;
                task->m_pThreadBlockSize.y = blockYDim;
                task->m_pThreadBlockSize.z = blockZDim;
                task->m_pGridSize.x = gridXDim;
                task->m_pGridSize.y = gridYDim;
                task->m_pGridSize.z = gridZDim;
                break;
            }
        case ACCELERATOR_CLASS_DIRECT_X:
            {
                DXTask * task = dynamic_cast<DXTask *>(pTask);
                task->m_bGeometryExplicit = geometryExplicit;
                task->m_nPreferredXDim = preferredXDim;
                task->m_nPreferredYDim = preferredYDim;
                task->m_nPreferredZDim = preferredZDim;
                break;
            }
        case ACCELERATOR_CLASS_OPEN_CL:
            {
                CLTask * task = dynamic_cast<CLTask *>(pTask);
                task->m_bGeometryExplicit = geometryExplicit;
                task->m_nPreferredXDim = preferredXDim;
                task->m_nPreferredYDim = preferredYDim;
                task->m_nPreferredZDim = preferredZDim;
                break;
            }
        case ACCELERATOR_CLASS_HOST:
            {
                HostTask * task = dynamic_cast<HostTask *>(pTask);
                task->m_bGeometryExplicit = geometryExplicit;
                task->m_nPreferredXDim = preferredXDim;
                task->m_nPreferredYDim = preferredYDim;
                task->m_nPreferredZDim = preferredZDim;

                task->m_bThreadBlockSizesExplicit = threadBlockSizesExplicit;
                task->m_pThreadBlockSize.x = blockXDim;
                task->m_pThreadBlockSize.y = blockYDim;
                task->m_pThreadBlockSize.z = blockZDim;
                task->m_pGridSize.x = gridXDim;
                task->m_pGridSize.y = gridYDim;
                task->m_pGridSize.z = gridZDim;
                break;
            }
        case ACCELERATOR_CLASS_REFERENCE:
        case ACCELERATOR_CLASS_UNKNOWN:
        default:
            {
                printf("Attempt to deserialize Task of unsupported platform type %d\n", 
                    pTask->GetAcceleratorClass());
                throw new XMLReaderException();
            }
        }
*/
        delete [] inPorts;
        delete [] outPorts;
        FreeString(name); // cjr: PLEASE!
        if(!ReadElementEndTag("Task"))
            return NULL;
        return pTask;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads the port. </summary>
    ///
    /// <remarks>   Crossbac, 1/17/2014. </remarks>
    ///
    /// <returns>   null if it fails, else the port. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port * 
    XMLReader::ReadPort(
        VOID
        )
    {
        if(!ReadElementStartTag("Port"))
            return NULL;

        UINT originalIndex;
        int type;
        const char * datablockTemplate;
        UINT uniqueID;
        const char * variableBinding;
        int parameterIndex;
        int inOutIndex;

        uniqueID = ReadUINTElement("UniqueID");
        originalIndex = ReadUINTElement("OriginalIndex");
        type = ReadIntegerElement("Type");
        datablockTemplate = ReadTextElement("DatablockTemplate");
        variableBinding = ReadTextElement("VariableBinding");
        parameterIndex = ReadIntegerElement("ParameterIndex");
        inOutIndex = ReadIntegerElement("InOutIndex");

        std::string templateName(datablockTemplate);
        DatablockTemplate * pTemplate = m_templateMap[templateName];

        Port * pPort = PTask::Runtime::CreatePort((PTask::PORTTYPE)type, pTemplate, uniqueID, 
                                                  (char*)variableBinding, parameterIndex, inOutIndex);

        pPort->SetOriginalIndex(originalIndex);
        if(!ReadElementEndTag("Port"))
            return NULL;
        m_portMap[pPort->GetUID()] = pPort;
        FreeString(datablockTemplate); // cjr: blast!
        FreeString(variableBinding);   // cjr: avast ye!
        return pPort;
    }
/*
    void
    XMLReader::ReadControlPropagationInfo(Port * currentPort)
    {
        this->ReadElementStartTag("ControlPropagationPorts");
        UINT numPorts;
        this->ReadElement("Count", numPorts);
        if (numPorts > 0)
        {
            std::vector<UINT>* portUIDs = new std::vector<UINT>;
            for (int i=0; i<numPorts; i++)
            {
                UINT uid;
                this->ReadElement("UniqueID", uid);
                portUIDs->push_back(uid);
            }
            m_controlPropagationPortsMap[currentPort->GetUID()] = portUIDs;
        }
        this->ReadElementEndTag("ControlPropagationPorts");

        int sourceUID;
        this->ReadElement("ControlPropagationSource", sourceUID);
        if (-1 != sourceUID)
        {
            m_controlPropagationSourceMap[currentPort->GetUID()] = sourceUID;
        }

        this->ReadElementStartTag("ControlPropagationChannels");
        UINT numChannels;
        this->ReadElement("Count", numChannels);
        if (numChannels > 0)
        {
            std::vector<std::string*>* channelNames = new std::vector<std::string*>;
            for (int i=0; i<numChannels; i++)
            {
                const char * name;
                this->ReadElement("Name", name);
                channelNames->push_back(new std::string(name));
                delete [] name;
            }
            m_controlPropagationChannelsMap[currentPort->GetUID()] = channelNames;
        }
        this->ReadElementEndTag("ControlPropagationChannels");

        this->ReadElementStartTag("GatedPorts");
        this->ReadElement("Count", numPorts);
        if (numPorts > 0)
        {
            std::vector<UINT>* portUIDs = new std::vector<UINT>;
            for (int i=0; i<numPorts; i++)
            {
                UINT uid;
                this->ReadElement("UniqueID", uid);
                portUIDs->push_back(uid);
            }
            m_gatedPortsMap[currentPort->GetUID()] = portUIDs;
        }
        this->ReadElementEndTag("GatedPorts");

        // Output port has a control port if BindControlPort was called on it.
        if (PORTTYPE::OUTPUT_PORT == currentPort->GetPortType())
        {
            this->ReadElementStartTag("OutputPortControlPort");
            int controlPortUID;
            BOOL initialPortStateOpen;
            this->ReadElement("UniqueID", controlPortUID);
            this->ReadElement("InitialPortStateOpen", initialPortStateOpen);
            if (-1 != controlPortUID)
            {
                std::pair<int, BOOL>* controlPortInfo = 
                    new std::pair<int, BOOL>(controlPortUID, initialPortStateOpen);
                m_outputPortControlPortMap[currentPort->GetUID()] = controlPortInfo;
            }
            this->ReadElementEndTag("OutputPortControlPort");
        }

        this->ReadElementStartTag("DescriptorPorts");
        this->ReadElement("Count", numPorts);
        if (numPorts > 0)
        {
            std::vector<pair<int,int>*>* descriptorRecords = new std::vector<std::pair<int,int>*>();
            for (int i=0; i<numPorts; i++)
            {
                int descriptorPortUID;
                int func;
                this->ReadElement("UniqueID", descriptorPortUID);
                this->ReadElement("DescriptorFunction", func);
                descriptorRecords->push_back(new std::pair<int,int>(descriptorPortUID, func));
            }
            m_descriptorPortsMap[currentPort->GetUID()] = descriptorRecords;
        }
        this->ReadElementEndTag("DescriptorPorts");

        this->ReadElementStartTag("DeferredChannels");
        this->ReadElement("Count", numChannels);
        if (numChannels > 0)
        {
            std::vector<std::pair<string*,int>*>* deferredChannels = new std::vector<std::pair<string*,int>*>();
            for (int i=0; i<numChannels; i++)
            {
                const char * channelName;
                int func;
                this->ReadElement("Name", channelName);
                this->ReadElement("DescriptorFunction", func);
                string * channelNameString = new string(channelName);
                deferredChannels->push_back(new std::pair<string*,int>(channelNameString, func));
            }
            m_deferredChannelsMap[currentPort->GetUID()] = deferredChannels;
        }
        this->ReadElementEndTag("DeferredChannels");
    }
*/
    Channel * 
    XMLReader::ReadChannel(
        VOID
        )
    {
        if(!ReadElementStartTag("Channel"))
            return NULL;

        const char * name;
        int typeID;
        UINT sourcePortID;
        UINT destinationPortID;

        name = ReadTextElement("Name");
        typeID = ReadIntegerElement("Type");
        sourcePortID = ReadUINTElement("SourcePort");
        destinationPortID =  ReadUINTElement("DestinationPort");
        if(!ReadElementEndTag("Channel"))
            return NULL;

        // Don't create channels which will be automatically re-created by
        // replay of actions whith created them in the original graph.
        if (!strncmp("BindDescriptorPort_Channel_", name, 
            strlen("BindDescriptorPort_Channel_"))) {
            FreeString(name);
            return NULL;
        }

        Channel * pChannel = NULL;
        PTask::CHANNELTYPE channelType = (PTask::CHANNELTYPE)typeID;
        switch(channelType) {
        case CT_GRAPH_INPUT:
            pChannel = m_pGraph->AddInputChannel(m_portMap[destinationPortID],(char*)name);
            break;
        case CT_GRAPH_OUTPUT:
            pChannel = m_pGraph->AddOutputChannel(m_portMap[sourcePortID],(char*)name);
            break;
        case CT_INTERNAL:
            pChannel = m_pGraph->AddInternalChannel(m_portMap[sourcePortID], m_portMap[destinationPortID],(char*)name);
            break;
        case CT_INITIALIZER:
        case CT_MULTI:
            printf("Unimplemented Channel type %d\n", typeID);
            throw new XMLReaderException();
            break;
        default:
            printf("Unknown Channel type %d\n", typeID);
            throw new XMLReaderException();
            break;
        }
        FreeString(name);
        return pChannel;
    }
};
#endif