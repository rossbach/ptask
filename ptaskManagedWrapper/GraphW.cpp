///-------------------------------------------------------------------------------------------------
// file:	GraphW.cpp
//
// summary:	Implements the graph wrapper class
///-------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "PTaskManagedWrapper.h"
#include "Utils.h"

namespace Microsoft {
namespace Research {
namespace PTask { 

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="nativeGraph">  [in,out] If non-null, the native graph. </param>
    ///-------------------------------------------------------------------------------------------------

    Graph::Graph(
        ::PTask::Graph* nativeGraph
        )
    {
        m_nativeGraph = nativeGraph;
        m_nativeGraph->SetManagedObject();
        m_disposed = false;
        m_bRunning = false;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Graph::~Graph()
    {
        this->!Graph();
        m_disposed = true;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Finaliser. </summary>
    ///
    /// <remarks>   crossbac, 9/8/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Graph::!Graph()
    {
        bool bLeakGraph = false;
        if(bLeakGraph) {            
            printf("WARNING: %s::%s--experimental leak of PTask graph!\n",
                   __FILE__,
                   __FUNCTION__
                   );
        } else {
            ::PTask::Graph::DestroyGraph(m_nativeGraph, TRUE);
        }
        m_nativeGraph = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the native graph. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the native graph. </returns>
    ///-------------------------------------------------------------------------------------------------

    ::PTask::Graph* 
    Graph::GetNativeGraph(
        void
        )
    {
        CheckDisposed();
        return m_nativeGraph;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Runs this object. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::Run(
        void
        )
    {
        CheckDisposed();
        m_nativeGraph->Run(false);
        m_bRunning = true;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Runs. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="bSingleThreaded">  true if single threaded. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::Run(
        bool bSingleThreaded
        )
    {
        CheckDisposed();
        m_nativeGraph->Run(bSingleThreaded);
        m_bRunning = true;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is running. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <returns>   true if running, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool
    Graph::IsRunning()
    {
        CheckDisposed();
        return m_bRunning;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Stops this object. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::Stop()
    {
        CheckDisposed();
        if(m_nativeGraph)
            m_nativeGraph->Stop();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Teardowns this object. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::Teardown()
    {
        CheckDisposed();
        if(m_nativeGraph)
            m_nativeGraph->Teardown();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Stops this object. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::Stop(bool bSync)
    {
        UNREFERENCED_PARAMETER(bSync);
        CheckDisposed();
        if(m_nativeGraph)
            m_nativeGraph->Stop();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Teardowns this object. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::Teardown(bool bSync)
    {
        UNREFERENCED_PARAMETER(bSync);
        CheckDisposed();
        if(m_nativeGraph)
            m_nativeGraph->Teardown();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a task. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="kernel">           [in,out] If non-null, the kernel. </param>
    /// <param name="inputPortCount">   Number of input ports. </param>
    /// <param name="inputPorts">       [in,out] If non-null, the input ports. </param>
    /// <param name="outputPortCount">  Number of output ports. </param>
    /// <param name="outputPorts">      [in,out] If non-null, the output ports. </param>
    /// <param name="taskName">         [in,out] If non-null, name of the task. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Task^ 
    Graph::AddTask(
        CompiledKernel^ kernel,
        UInt32          inputPortCount,
        array<Port^>^   inputPorts,
        UInt32          outputPortCount,
        array<Port^>^   outputPorts,
        String^         taskName
        )
    {
        CheckDisposed();
        char* lpszTaskName = Microsoft::Research::PTask::Utils::MarshalString(taskName);

        // Build native port arrays from managed ones.
        ::PTask::Port** nativeInputPorts = new ::PTask::Port*[inputPortCount];
        for(unsigned int i=0; i<inputPortCount; i++) {
            nativeInputPorts[i] = inputPorts[i]->GetNativePort();
            inputPorts[i]->SetBoundToTask();
        }
        ::PTask::Port** nativeOutputPorts = new ::PTask::Port*[outputPortCount];
        for(unsigned int i=0; i<outputPortCount; i++) {
            nativeOutputPorts[i] = outputPorts[i]->GetNativePort();
            outputPorts[i]->SetBoundToTask();
        }

        ::PTask::Task* nativeTask = m_nativeGraph->AddTask(
            kernel->GetNativeCompiledKernel(),
            inputPortCount,
            nativeInputPorts,
            outputPortCount,
            nativeOutputPorts,
            lpszTaskName
            );

        Task^ managedTask = gcnew Task(nativeTask);
        delete [] nativeInputPorts;
        delete [] nativeOutputPorts;
        Utils::FreeUnmanagedString(lpszTaskName);
        return managedTask;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a task. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="kernel">           [in,out] If non-null, the kernel. </param>
    /// <param name="inputPortCount">   Number of input ports. </param>
    /// <param name="inputPorts">       [in,out] If non-null, the input ports. </param>
    /// <param name="outputPortCount">  Number of output ports. </param>
    /// <param name="outputPorts">      [in,out] If non-null, the output ports. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Task^ 
    Graph::AddTask(
        CompiledKernel^ kernel,
        UInt32          inputPortCount,
        array<Port^>^   inputPorts,
        UInt32          outputPortCount,
        array<Port^>^   outputPorts
        )
    {
        return AddTask(kernel, 
                       inputPortCount, 
                       inputPorts, 
                       outputPortCount, 
                       outputPorts, 
                       nullptr);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an input channel to 'channelName'. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="port">         [in,out] If non-null, the port. </param>
    /// <param name="channelName">  [in,out] If non-null, name of the channel. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    GraphInputChannel^
    Graph::AddInputChannel(
        Port^ port, 
        String^ channelName
        )
    {
        CheckDisposed();
        char* lpszChannelName = Utils::MarshalString(channelName);
        ::PTask::GraphInputChannel* pChannel = 
            m_nativeGraph->AddInputChannel(port->GetNativePort(), lpszChannelName);
        GraphInputChannel^ managedChannel = gcnew GraphInputChannel(pChannel, channelName);
        Utils::FreeUnmanagedString(lpszChannelName);
        return managedChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an input channel. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="port"> [in,out] If non-null, the port. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    GraphInputChannel^
    Graph::AddInputChannel(
        Port^ port
        )
    {
        CheckDisposed();       
        ::PTask::GraphInputChannel* pChannel = 
            m_nativeGraph->AddInputChannel(port->GetNativePort());
        GraphInputChannel^ managedChannel = gcnew GraphInputChannel(pChannel, nullptr);
        return managedChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an input channel. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="port">             [in,out] If non-null, the port. </param>
    /// <param name="channelName">      [in,out] If non-null, name of the channel. </param>
    /// <param name="switchChannel">    true to switch channel. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    GraphInputChannel^ 
    Graph::AddInputChannel(
        Port^ port, 
        String^ channelName, 
        bool switchChannel
        )
    {
        CheckDisposed();
        char* lpszChannelName = Utils::MarshalString(channelName);
        ::PTask::GraphInputChannel* nativeChannel = 
            m_nativeGraph->AddInputChannel(port->GetNativePort(), lpszChannelName, switchChannel);
        GraphInputChannel^ managedChannel = gcnew GraphInputChannel(nativeChannel, channelName);
        Utils::FreeUnmanagedString(lpszChannelName);
        return managedChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an input channel. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="port">             [in,out] If non-null, the port. </param>
    /// <param name="channelName">      [in,out] If non-null, name of the channel. </param>
    /// <param name="switchChannel">    true to switch channel. </param>
    /// <param name="pTriggerPort">     [in,out] If non-null, the trigger port. </param>
    /// <param name="uiTriggerCode">    The trigger code. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    GraphInputChannel^ 
    Graph::AddInputChannel(
        Port^ port, 
        String^ channelName, 
        bool switchChannel,
        Port^ pTriggerPort,
        unsigned __int64 luiTriggerCode
        )
    {
        CheckDisposed();
        char* lpszChannelName = Utils::MarshalString(channelName);
        pTriggerPort->GetNativePort()->SetTriggerPort(m_nativeGraph, 1);
        ::PTask::GraphInputChannel* nativeChannel = 
            m_nativeGraph->AddInputChannel(port->GetNativePort(), 
                                           lpszChannelName, 
                                           switchChannel,
                                           pTriggerPort->GetNativePort(),
                                           luiTriggerCode);

        GraphInputChannel^ managedChannel = gcnew GraphInputChannel(nativeChannel, channelName);
        Utils::FreeUnmanagedString(lpszChannelName);
        return managedChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an output channel to 'channelName'. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="port">         [in,out] If non-null, the port. </param>
    /// <param name="channelName">  [in,out] If non-null, name of the channel. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    GraphOutputChannel^
    Graph::AddOutputChannel(
        Port^ port, 
        String^ channelName
        )
    {
        CheckDisposed();
        char* lpszChannelName = Utils::MarshalString(channelName);
        ::PTask::GraphOutputChannel* nativeChannel = 
            m_nativeGraph->AddOutputChannel(port->GetNativePort(), lpszChannelName);
        GraphOutputChannel^ managedChannel = gcnew GraphOutputChannel(nativeChannel, channelName);
        Utils::FreeUnmanagedString(lpszChannelName);
        return managedChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an output channel to 'channelName'. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="port">             If non-null, the port. </param>
    /// <param name="channelName">      If non-null, name of the channel. </param>
    /// <param name="triggerTask">      [in,out] If non-null, the trigger task. </param>
    /// <param name="uiTriggerCode">    The trigger code. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    GraphOutputChannel^
    Graph::AddOutputChannel(
        Port^ port, 
        String^ channelName,
        Task^ triggerTask,
        unsigned __int64 luiTriggerCode
        )
    {
        CheckDisposed();
        char* lpszChannelName = Utils::MarshalString(channelName);
        ::PTask::GraphOutputChannel* nativeChannel;
        nativeChannel = m_nativeGraph->AddOutputChannel(port->GetNativePort(), 
                                                        lpszChannelName,
                                                        triggerTask->GetNativeTask(),
                                                        luiTriggerCode);
        GraphOutputChannel^ managedChannel = gcnew GraphOutputChannel(nativeChannel, channelName);
        Utils::FreeUnmanagedString(lpszChannelName);
        return managedChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an output channel. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="port"> [in,out] If non-null, the port. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    GraphOutputChannel^
    Graph::AddOutputChannel(
        Port^ port
        )
    {
        return AddOutputChannel(port, nullptr);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an internal channel. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="src">          [in,out] If non-null, source for the. </param>
    /// <param name="dst">          [in,out] If non-null, destination for the. </param>
    /// <param name="channelName">  [in,out] If non-null, name of the channel. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    InternalChannel^
    Graph::AddInternalChannel(
        Port^ src, 
        Port^ dst, 
        String^ channelName
        )
    {
        CheckDisposed();
        char* lpszChannelName = Utils::MarshalString(channelName);
        ::PTask::InternalChannel* nativeChannel = 
            m_nativeGraph->AddInternalChannel(src->GetNativePort(), dst->GetNativePort(), lpszChannelName);
        InternalChannel^ managedChannel = gcnew InternalChannel(nativeChannel, channelName);
        return managedChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an internal channel to 'pDst'. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pSrc"> [in,out] If non-null, source for the. </param>
    /// <param name="pDst"> [in,out] If non-null, destination for the. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    InternalChannel^
    Graph::AddInternalChannel(
        Port^ pSrc, 
        Port^ pDst
        )
    {
        return AddInternalChannel(pSrc, pDst, nullptr);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a multi channel. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="dstPorts">         [in,out] If non-null, destination ports. </param>
    /// <param name="channelName">      [in,out] If non-null, name of the channel. </param>
    /// <param name="switchChannel">    true to switch channel. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    MultiChannel^
    Graph::AddMultiChannel(
        array<Port^>^ dstPorts, 
        String^ channelName, 
        bool switchChannel
        )
    {
        CheckDisposed();
        char* lpszChannelName = Utils::MarshalString(channelName);
		int nDstPorts = dstPorts->Length;
        ::PTask::Port **ppDst = new ::PTask::Port*[nDstPorts];
        for (int i=0; i<nDstPorts; i++) {
            ppDst[i] = dstPorts[i]->GetNativePort();
            dstPorts[i]->SetBoundToTask();
        }
        ::PTask::MultiChannel* nativeChannel = 
            m_nativeGraph->AddMultiChannel(ppDst, nDstPorts, lpszChannelName, switchChannel);
        MultiChannel^ managedChannel = gcnew MultiChannel(nativeChannel, channelName);
        Utils::FreeUnmanagedString(lpszChannelName);
        return managedChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a multi channel. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="dstPorts">         [in,out] If non-null, destination ports. </param>
    /// <param name="channelName">      [in,out] If non-null, name of the channel. </param>
    /// <param name="switchChannel">    true to switch channel. </param>
    /// <param name="triggerPort">      [in,out] If non-null, the trigger port. </param>
    /// <param name="uiTriggerCode">    The trigger code. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    MultiChannel^
    Graph::AddMultiChannel(
        array<Port^>^ dstPorts, 
        String^ channelName, 
        bool switchChannel,
        Port^ triggerPort,
        unsigned __int64 luiTriggerCode
        )
    {
        CheckDisposed();        
        char* lpszChannelName = Utils::MarshalString(channelName);
		int nDstPorts = dstPorts->Length;
        ::PTask::Port **ppDst = new ::PTask::Port*[nDstPorts];
        for (int i=0; i<nDstPorts; i++) {
            ppDst[i] = dstPorts[i]->GetNativePort();
            dstPorts[i]->SetBoundToTask();
        }
        triggerPort->GetNativePort()->SetTriggerPort(m_nativeGraph, 1);
        ::PTask::MultiChannel* nativeChannel = 
            m_nativeGraph->AddMultiChannel(ppDst, 
                                           nDstPorts, 
                                           lpszChannelName, 
                                           switchChannel,
                                           triggerPort->GetNativePort(),
                                           luiTriggerCode
                                           );

        MultiChannel^ managedChannel = gcnew MultiChannel(nativeChannel, channelName);
        Utils::FreeUnmanagedString(lpszChannelName);
        return managedChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an initializer channel. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="port"> [in,out] If non-null, the port. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Channel^
    Graph::AddInitializerChannel(
        Port^ port
        )
    {
        return AddInitializerChannel(port, nullptr, false);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an initializer channel. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="port">             [in,out] If non-null, the port. </param>
    /// <param name="channelName">      [in,out] If non-null, name of the channel. </param>
    /// <param name="switchChannel">    true to switch channel. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Channel^ 
    Graph::AddInitializerChannel(
        Port^ port, 
        String^ channelName, 
        bool switchChannel
        )
    {
        CheckDisposed();
        char* lpszChannelName = Utils::MarshalString(channelName);
        ::PTask::Channel* nativeChannel = 
            m_nativeGraph->AddInitializerChannel(port->GetNativePort(), lpszChannelName, switchChannel);
        Channel^ managedChannel = gcnew Channel(nativeChannel, channelName, Initializer, true);
        Utils::FreeUnmanagedString(lpszChannelName);
        return managedChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an initializer channel. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="port">             [in,out] If non-null, the port. </param>
    /// <param name="channelName">      [in,out] If non-null, name of the channel. </param>
    /// <param name="switchChannel">    true to switch channel. </param>
    /// <param name="pTriggerPort">     [in,out] If non-null, the trigger port. </param>
    /// <param name="uiTriggerCode">    The trigger code. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Channel^ 
    Graph::AddInitializerChannel(
        Port^ port, 
        String^ channelName, 
        bool switchChannel,
        Port^ pTriggerPort,
        unsigned __int64 luiTriggerCode
        )
    {
        CheckDisposed();
        char* lpszChannelName = Utils::MarshalString(channelName);
        pTriggerPort->GetNativePort()->SetTriggerPort(m_nativeGraph, 1);
        ::PTask::Channel* nativeChannel = 
            m_nativeGraph->AddInitializerChannel(port->GetNativePort(), 
                                           lpszChannelName, 
                                           switchChannel,
                                           pTriggerPort->GetNativePort(),
                                           luiTriggerCode);

        Channel^ managedChannel = gcnew Channel(nativeChannel, channelName, Initializer, true);
        Utils::FreeUnmanagedString(lpszChannelName);
        return managedChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind descriptor port. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pDescribedPort">   [in,out] If non-null, the described port. </param>
    /// <param name="pDescriberPort">   [in,out] If non-null, the describer port. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::BindDescriptorPort(
        Port ^pDescribedPort, 
        Port ^pDescriberPort
        )
    {
        CheckDisposed();        
        m_nativeGraph->BindDescriptorPort(pDescribedPort->GetNativePort(), 
                                          pDescriberPort->GetNativePort(), 
                                          ::PTask::DF_SIZE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind descriptor port. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pDescribedPort">   [in,out] If non-null, the described port. </param>
    /// <param name="pDescriberPort">   [in,out] If non-null, the describer port. </param>
    /// <param name="uiDescFun">        The description fun. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::BindDescriptorPort(
        Port ^pDescribedPort, 
        Port ^pDescriberPort, 
        UINT uiDescFun
        )
    {
        CheckDisposed();        
        ::PTask::DESCRIPTORFUNC func = static_cast<::PTask::DESCRIPTORFUNC>(uiDescFun);
        m_nativeGraph->BindDescriptorPort(pDescribedPort->GetNativePort(), 
                                          pDescriberPort->GetNativePort(),
                                          func);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind control port. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <param name="pController">      [in,out] If non-null, the controller. </param>
    /// <param name="pGatedPort">       [in,out] If non-null, the gated port. </param>
    /// <param name="bInitiallyOpen">   true to initially open. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::BindControlPort( 
        Port ^pController, 
        Port ^pGatedPort, 
        bool bInitiallyOpen
        )
    {
        CheckDisposed();
        m_nativeGraph->BindControlPort(pController->GetNativePort(), 
                                       pGatedPort->GetNativePort(), 
                                       bInitiallyOpen);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind control propagation port. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pInputPort">   [in,out] If non-null, the input port. </param>
    /// <param name="pOutputPort">  [in,out] If non-null, the output port. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Graph::BindControlPropagationPort(
        Port ^pInputPort, 
        Port ^pOutputPort
        )
    {
        CheckDisposed();
        m_nativeGraph->BindControlPropagationPort(pInputPort->GetNativePort(), 
                                                  pOutputPort->GetNativePort());
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Writes a dot file. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="filename"> [in,out] If non-null, filename of the file. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::WriteDOTFile(
        String^ filename
        )
    {
        return WriteDOTFile(filename, true, false);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Writes a dot file. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="filename">         [in,out] If non-null, filename of the file. </param>
    /// <param name="drawPorts">        true to draw ports. </param>
    /// <param name="presentationMode"> true to enable presentation mode, false to disable it. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::WriteDOTFile(
        String^ filename,
        bool drawPorts,
        bool presentationMode
        )
    {
        CheckDisposed();
        char * lpszFileName = Utils::MarshalString(filename);
        m_nativeGraph->WriteDOTFile(lpszFileName, 
                                    (drawPorts?1:0),
                                    (presentationMode?1:0));
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check if the object is already disposed and throw if appropriate. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///-------------------------------------------------------------------------------------------------

    void
    Graph::CheckDisposed(
        void
        ) 
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Graph already disposed");
    }



}}}
