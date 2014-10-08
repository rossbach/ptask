///-------------------------------------------------------------------------------------------------
// file:	ChannelW.h
//
// summary:	Declares the channel wrapper classes
///-------------------------------------------------------------------------------------------------

#pragma once
#pragma unmanaged
#include "ptaskapi.h"
#pragma managed
#include "CompiledKernelW.h"
#include "DatablockW.h"
#include "DataTemplate.h"
#include "GraphW.h"
using namespace System;
using namespace System::Runtime::InteropServices; // For marshalling helpers

namespace Microsoft {
namespace Research {
namespace PTask {

    public interface class IEventHandler
    {
    public:
	    void Event(Channel^ channel);
    };

    enum ChannelType {

        /// <summary> Input-only channel. The destination endpoint 
        /// 		  can be bound to a port but the source is 
        /// 		  expected to be fed by user code. 
        /// 		  </summary>
        Input, // --> PTask::CT_GRAPH_INPUT,

        /// <summary> Ouput-only channel. The source endpoint 
        /// 		  can be bound to a port but the destinatoin is 
        /// 		  expected to be drained by user code. 
        /// 		  </summary>
        Output, // --> CT_GRAPH_OUTPUT,

        /// <summary> Internal channel. Both ends bound to ports. </summary>
        Internal, // CT_INTERNAL,

        /// <summary>  channel with single input, multiple outputs </summary>
        Multi, // CT_MULTI,

        /// <summary>   channel that can produce data in response to a pull,
        /// 			without requiring an upstream data source. 
        /// 			</summary>
        Initializer, // CT_INITIALIZER

        Coalesced

    };

    enum ChannelEventSemantics {
        OnChannelEntryAttempt = 0,      // call the handler before the native push call
        OnChannelEntrySuccess = 1,      // call the handler after the push call, if it succeeds
        OnChannelExitAttempt = 2,       // call before calling push
        OnChannelExitSuccess = 3,       // call if the pull succeeds
        NumEventSemanticsTypes = 4      // for array decls: valid total event types
    };

    enum EventHandlerInstanceSemantics {
        DeferToDefault,                 // defer to the global hanlder if both are defined
        OverrideDefault,                // replace the global handler if both are defined
        PrependDefault,                 // call before global handler if both are defined
        PostpendDefault,                // call after the global handler if both are defined
    };

    ref class InstanceHandlerAttrs {
    public:
        IEventHandler^ lpfnHandler;
        ChannelEventSemantics eEventSemantics;
        EventHandlerInstanceSemantics eInstanceSemantics;
    };

    public ref class Channel
    {
    public:

        static const int CT_GRAPH_INPUT = static_cast<const int>(CHANNELTYPE::CT_GRAPH_INPUT);
        static const int CT_GRAPH_OUTPUT = static_cast<const int>(CHANNELTYPE::CT_GRAPH_OUTPUT);
        static const int CT_INTERNAL = static_cast<const int>(CHANNELTYPE::CT_INTERNAL);
        static const int CT_MULTI = static_cast<const int>(CHANNELTYPE::CT_MULTI);
        static const int CT_INITIALIZER = static_cast<const int>(CHANNELTYPE::CT_INITIALIZER);
        static const int CGATEFN_NONE = 0;
        static const int CGATEFN_CLOSE_ON_EOF = 1;
        static const int CGATEFN_OPEN_ON_EOF = 2;
        static const int CGATEFN_OPEN_ON_BEGINITERATION = 3;
        static const int CGATEFN_CLOSE_ON_BEGINITERATION = 4;
        static const int CGATEFN_OPEN_ON_ENDITERATION = 5;
        static const int CGATEFN_CLOSE_ON_ENDITERATION = 6;
        static const int CGATEFN_DEVNULL = 7;
        static const int CGATEFN_CLOSE_ON_BOF = 8;
        static const int CGATEFN_OPEN_ON_BOF = 9;
        static const int CGATEFN_USER_DEFINED = 10;            

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   crossbac, 8/15/2013. </remarks>
        ///
        /// <param name="channel">  [in,out] If non-null, the channel. </param>
        /// <param name="strName">  [in,out] If non-null, the name. </param>
        /// <param name="eType">    The type. </param>
        /// <param name="bIsInput"> The is input. </param>
        ///-------------------------------------------------------------------------------------------------

        Channel(
            __in ::PTask::Channel* channel,
            __in String^ strName,
            __in ChannelType eType,
            __in bool bIsInput
            );

        virtual ~Channel(); // IDisposable
        !Channel(); // finalizer

        virtual ::PTask::Channel* GetNativeChannel();
        virtual DataTemplate^ GetTemplate();
        virtual int GetChannelPredicationType();
        virtual void SetChannelPredicationType(int n);
        virtual int GetViewMaterializationPolicy();
        virtual void SetViewMaterializationPolicy(int n);
        virtual int GetCapacity();
        virtual void SetCapacity(int n);
        virtual int GetQueueOccupancy();
        virtual bool Push(Datablock^ block);
        virtual bool PushInitializer();
        virtual Datablock^ Pull();
        virtual Datablock^ Pull(DWORD dwTimeout);
        virtual bool CanStream();

        virtual property String^ Name { String^ get() { return GetName(); } }        
        virtual property bool Immutable { bool get() { return IsImmutable(); } }
        virtual property ChannelType Type { ChannelType get() { return GetChannelType(); } }
        virtual property bool IsInput { bool get() { return IsInputChannel(); } }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   default handlers, called on push/pull. </summary>
        ///
        /// <remarks>   crossbac, 8/15/2013. </remarks>
        ///
        /// <param name="lpfnHandler">      [in,out] If non-null, the lpfn handler. </param>
        /// <param name="eEventSemantics">  The event semantics. </param>
        ///-------------------------------------------------------------------------------------------------

        static void 
        SetDefaultEventHandler(
            __in IEventHandler^ lpfnHandler, 
            __in ChannelEventSemantics eEventSemantics
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Setter for push/pull event handlers specific to this channel, called when the set
        ///             event. If none is available, we defer to the runtime default handler, if defined.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 8/15/2013. </remarks>
        ///
        /// <param name="eventHandler">     [in,out] If non-null, the event handler. </param>
        /// <param name="eEventSemantics">  The event semantics. </param>
        ///-------------------------------------------------------------------------------------------------

        void 
        SetEventHandler(
            __in IEventHandler^ eventHandler, 
            __in ChannelEventSemantics eEventSemantics,
            __in EventHandlerInstanceSemantics eInstanceSemantics
            );

    protected:

        virtual void OnChannelEvent(ChannelEventSemantics eEvent);
        virtual array<IEventHandler^>^ GetEventHandlers(ChannelEventSemantics eEvent);
        virtual String^ GetName();
        virtual bool IsImmutable();
        virtual ChannelType GetChannelType(); 
        virtual bool IsInputChannel(); 

        void            CheckDisposed();
        bool            m_disposed;
        String^         m_name;
        bool            m_bImmutable;
        bool            m_bImmutableValid;
        ChannelType     m_eChannelType;
        bool            m_bChannelTypeValid;
        bool            m_bIsInput;
        bool            m_bIsInputValid;

    private:


        ::PTask::Channel* m_nativeChannel;
        array<InstanceHandlerAttrs^>^ m_pEventHandlers;
        static array<IEventHandler^>^ m_pDefaultEventHandlers = nullptr;
        static Object^ m_pSyncRoot = gcnew Object();
    };

    // Wrapper for native PTask::MultiChannel.
    public ref class MultiChannel : public Channel
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   crossbac, 8/15/2013. </remarks>
        ///
        /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
        /// <param name="strName">  [in,out] If non-null, the name. </param>
        ///-------------------------------------------------------------------------------------------------

        MultiChannel(
            __in ::PTask::MultiChannel* pChannel,
            __in String^ strName
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   crossbac, 8/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~MultiChannel() {}
    };
        
    // Wrapper for native PTask::GraphInputChannel.
    public ref class GraphInputChannel : public Channel
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   crossbac, 8/15/2013. </remarks>
        ///
        /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
        /// <param name="strName">  [in,out] If non-null, the name. </param>
        ///-------------------------------------------------------------------------------------------------

        GraphInputChannel(
            __in ::PTask::GraphInputChannel* pChannel, 
            __in String^ strName
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   crossbac, 8/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~GraphInputChannel() {}
    };

    // Wrapper for native PTask::GraphOutputChannel.
    public ref class GraphOutputChannel : public Channel
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   crossbac, 8/15/2013. </remarks>
        ///
        /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
        /// <param name="strName">  [in,out] If non-null, the name. </param>
        ///-------------------------------------------------------------------------------------------------

        GraphOutputChannel(
            __in ::PTask::GraphOutputChannel* pChannel,
            __in String^ strName
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   crossbac, 8/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~GraphOutputChannel() {}
    };

    // Wrapper for native PTask::InternalChannel.
    public ref class InternalChannel : public Channel
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   crossbac, 8/15/2013. </remarks>
        ///
        /// <param name="internalChannel">  [in,out] If non-null, the internal channel. </param>
        /// <param name="strName">          [in,out] If non-null, the name. </param>
        ///-------------------------------------------------------------------------------------------------

        InternalChannel(
            __in ::PTask::InternalChannel* internalChannel,
            __in String ^ strName
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   crossbac, 8/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~InternalChannel() {}
    };

    public ref class CoalescedChannel : public Channel {
    public:
        CoalescedChannel(
            __in array<Channel^>^ vChildren,
            __in String^ strName
            );
        virtual ~CoalescedChannel() {}
        virtual ::PTask::Channel* GetNativeChannel() override;
        virtual DataTemplate^ GetTemplate() override;
        virtual int GetChannelPredicationType() override;
        virtual void SetChannelPredicationType(int n) override; 
        virtual int GetViewMaterializationPolicy() override;
        virtual void SetViewMaterializationPolicy(int n) override;
        virtual int GetCapacity() override; 
        virtual void SetCapacity(int n) override;
        virtual int GetQueueOccupancy() override;
        virtual bool Push(Datablock^ block) override;
        virtual bool PushInitializer() override;
        virtual Datablock^ Pull() override;
        virtual Datablock^ Pull(DWORD dwTimeout) override;
        virtual bool CanStream() override;
    protected:
        virtual bool IsImmutable() override;

    protected:
        void OnMeaninglessOperation();
        array<Channel^>^ m_children;
    };

}}}
