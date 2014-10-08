///-------------------------------------------------------------------------------------------------
// file:	GraphW.h
//
// summary:	Declares the graph wrapper class
///-------------------------------------------------------------------------------------------------

#pragma once
#pragma unmanaged
#include "ptaskapi.h"
#pragma managed
#include "CompiledKernelW.h"
#include "DatablockW.h"
#include "DataTemplate.h"
using namespace System;
using namespace System::Runtime::InteropServices; 

namespace Microsoft {
namespace Research {
namespace PTask {

    ref class Port;
    ref class Task;
    ref class Channel;
    ref class GraphInputChannel;
    ref class GraphOutputChannel;
    ref class InternalChannel;
    ref class MultiChannel;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Graph. </summary>
    ///
    /// <remarks>   refactored from code by jcurrey by crossbac, 4/13/2012. 
    ///             Follows the C++/CLI IDisposable pattern described at 
    ///             http://msdn.microsoft.com/en-us/library/system.idisposable.aspx
    /// 			</remarks>
    ///-------------------------------------------------------------------------------------------------

    public ref class Graph
    {
    public:
        Graph(::PTask::Graph* nativeGraph);
        ~Graph(); // IDisposable
        !Graph(); // finalizer

        ::PTask::Graph* GetNativeGraph();

        void Run();
        void Run(bool bSingleThreaded);
        void Stop();
        void Teardown();
        void Stop(bool bSync);
        void Teardown(bool bSync);
        Task^ AddTask(
            CompiledKernel^ kernel,
            UInt32          inputPortCount,
            array<Port^>^   inputPorts,
            UInt32          outputPortCount,
            array<Port^>^   outputPorts,
            String^         taskName
            );
        Task^ AddTask(
            CompiledKernel^ kernel,
            UInt32          inputPortCount,
            array<Port^>^   inputPorts,
            UInt32          outputPortCount,
            array<Port^>^   outputPorts
            );

        GraphInputChannel^ AddInputChannel(Port^ port, String^ channelName);
        GraphInputChannel^ AddInputChannel(Port^ port);
        GraphInputChannel^ AddInputChannel(Port^ port, String^ channelName, bool switchChannel); 
        GraphInputChannel^ AddInputChannel(Port^ port, String^ channelName, bool switchChannel, Port^ triggerPort, unsigned __int64 luiTriggerCode); 
        GraphOutputChannel^ AddOutputChannel(Port^ port, String^ channelName, Task^ triggerTask, unsigned __int64 luiTriggerCode);
        GraphOutputChannel^ AddOutputChannel(Port^ port, String^ channelName);
        GraphOutputChannel^ AddOutputChannel(Port^ port);
        InternalChannel^ AddInternalChannel(Port^ src, Port^ dst, String^ channelName);
        InternalChannel^ AddInternalChannel(Port^ src, Port^ dst);
        MultiChannel^ AddMultiChannel(array<Port^>^ dstPorts, String^ channelName, bool switchChannel);
        MultiChannel^ AddMultiChannel(array<Port^>^ dstPorts, String^ channelName, bool switchChannel, Port^ triggerPort, unsigned __int64 luiTriggerCode); 
        Channel^ AddInitializerChannel(Port^ port);
        Channel^ AddInitializerChannel(Port^ port, String^ channelName, bool switchChannel); 
        Channel^ AddInitializerChannel(Port^ port, String^ channelName, bool switchChannel, Port^ triggerPort, unsigned __int64 luiTriggerCode); 

        void BindDescriptorPort(Port ^pDescribedPort, Port ^pDescriberPort);
        void BindDescriptorPort(Port ^pDescribedPort, Port ^pDescriberPort, UINT uiDescFun);
        void BindControlPort( Port ^pController, Port ^pGatedPort, bool bInitiallyOpen);
        void BindControlPropagationPort(Port ^pInputPort, Port ^pOutputPort);

        bool IsRunning();
        void WriteDOTFile(String^ filename);         
        void WriteDOTFile(String^ filename, bool drawPorts, bool presentationMode);         

    private:
        ::PTask::Graph* m_nativeGraph;
        bool          m_disposed;
        bool          m_bRunning;

        void CheckDisposed();
    };
}}}
