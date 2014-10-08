///-------------------------------------------------------------------------------------------------
// file:	ChannelW.cpp
//
// summary:	Implements the channel wrapper classes
///-------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "PTaskManagedWrapper.h"
#include "Utils.h"
#include <msclr\lock.h>

namespace Microsoft {
namespace Research {
namespace PTask {


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

    Channel::Channel(
        __in ::PTask::Channel* channel,
        __in String^ strName,
        __in ChannelType eType,
        __in bool bIsInput
        )
    {
        m_nativeChannel = channel;
        m_disposed = false;
        m_name = strName;
        m_bImmutable = false;
        m_bImmutableValid = false;
        m_bChannelTypeValid = true;
        m_bIsInput = bIsInput;
        m_bIsInputValid = true;
        m_eChannelType = eType;
        m_pEventHandlers = gcnew array<InstanceHandlerAttrs^>(NumEventSemanticsTypes);
        m_pEventHandlers[OnChannelEntryAttempt] = nullptr;
        m_pEventHandlers[OnChannelEntrySuccess] = nullptr;
        m_pEventHandlers[OnChannelExitAttempt ] = nullptr;
        m_pEventHandlers[OnChannelExitSuccess ] = nullptr;
    }

    Channel::~Channel()
    {
        this->!Channel();
        m_disposed = true;
    }

    Channel::!Channel()
    {
        // Do NOT delete m_nativeChannel here, since native Channel instances can only be obtained
        // from a call to (native) Graph::AddNNNChannel methods, and the Graph will delete all Channels
        // that it owns when Graph::Teardown is called.
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the name. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   nullptr if it fails, else the name. </returns>
    ///-------------------------------------------------------------------------------------------------

    String^ 
    Channel::GetName(
        void
        )
    {
        CheckDisposed();
        return m_name;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is immutable. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   true if immutable, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool 
    Channel::IsImmutable(
        void
        )
    {
        CheckDisposed();
        msclr::lock lock(this);
        if(!m_bImmutableValid) {
            m_bImmutable = 
                m_bIsInput &&
                m_nativeChannel->HasDownstreamWriters() == 0; 
            m_bImmutableValid = true;
        }
        return m_bImmutable;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets channel type. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   The channel type. </returns>
    ///-------------------------------------------------------------------------------------------------

    ChannelType 
    Channel::GetChannelType(
        void
        )
    {
        CheckDisposed();
        msclr::lock lock(this);
        if(!m_bChannelTypeValid) {
            m_bChannelTypeValid = true;
            switch(GetNativeChannel()->GetType()) {
            case ::PTask::CHANNELTYPE::CT_GRAPH_INPUT:  m_eChannelType = Input; m_bIsInput = true; break;
            case ::PTask::CHANNELTYPE::CT_GRAPH_OUTPUT: m_eChannelType = Output; m_bIsInput = false; break;
            case ::PTask::CHANNELTYPE::CT_INTERNAL:     m_eChannelType = Internal; m_bIsInput = false; break;
            case ::PTask::CHANNELTYPE::CT_INITIALIZER:  m_eChannelType = Initializer; m_bIsInput = true; break;
            case ::PTask::CHANNELTYPE::CT_MULTI:        m_eChannelType = Multi; m_bIsInput = true; break;
            default: m_bChannelTypeValid = false;
            }
        }
        return m_eChannelType;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is input channel. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   true if input channel, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool 
    Channel::IsInputChannel(
        void
        )
    {
        CheckDisposed();
        return m_bIsInput;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check disposed. </summary>
    ///
    /// <remarks>   crossbac </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///-------------------------------------------------------------------------------------------------

    void
    Channel::CheckDisposed(
        void
        ) 
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Channel already disposed");
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets native channel. </summary>
    ///
    /// <remarks>   crossbac </remarks>
    ///
    /// <returns>   null if it fails, else the native channel. </returns>
    ///-------------------------------------------------------------------------------------------------

    ::PTask::Channel*
    Channel::GetNativeChannel()
    {
        CheckDisposed();
        return m_nativeChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the template. </summary>
    ///
    /// <returns>   nullptr if it fails, else the template. </returns>
    ///-------------------------------------------------------------------------------------------------

    DataTemplate^
    Channel::GetTemplate() {
        CheckDisposed();
        ::PTask::DatablockTemplate* nativeTemplate = GetNativeChannel()->GetTemplate();
        return gcnew DataTemplate(nativeTemplate);        
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets channel predication type. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   The channel predication type. </returns>
    ///-------------------------------------------------------------------------------------------------

    int 
    Channel::GetChannelPredicationType() {
        CheckDisposed();
        return GetNativeChannel()->GetPredicationType(::PTask::CE_SRC);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets channel predication type. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="n">    The int to process. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::SetChannelPredicationType( 
        int n
        )
    {
        CheckDisposed();
        GetNativeChannel()->SetPredicationType(::PTask::CE_SRC, (::PTask::CHANNELPREDICATE)n);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets view materialization policy. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   The view materialization policy. </returns>
    ///-------------------------------------------------------------------------------------------------

    int 
    Channel::GetViewMaterializationPolicy() {
        CheckDisposed();
        return (int)GetNativeChannel()->GetViewMaterializationPolicy();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets view materialization policy. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="n">    The int to process. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::SetViewMaterializationPolicy( 
        int n
        )
    {
        CheckDisposed();
        GetNativeChannel()->SetViewMaterializationPolicy((::PTask::VIEWMATERIALIZATIONPOLICY)n);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the push attempt action. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::OnChannelEvent(
        ChannelEventSemantics eChannelEvent
        )
    {
        CheckDisposed();
        array<IEventHandler^>^ pHandlers = GetEventHandlers(eChannelEvent);
        if(pHandlers != nullptr) {
            for(int i=0; i<pHandlers->Length; i++) {
                IEventHandler^ lpfnHandler = pHandlers[i];
                lpfnHandler->Event(this);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets event handlers. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="eEvent">   The event. </param>
    ///
    /// <returns>   nullptr if it fails, else the event handlers. </returns>
    ///-------------------------------------------------------------------------------------------------

    array<IEventHandler^>^ 
    Channel::GetEventHandlers(
        ChannelEventSemantics eEvent
        )
    {
        msclr::lock globallock(m_pSyncRoot);
        msclr::lock instancelock(this);
        array<IEventHandler^>^ vEvents = nullptr;
        InstanceHandlerAttrs^ vInstanceAttrs = m_pEventHandlers[eEvent];
        IEventHandler^ lpfnDefault = 
            (m_pDefaultEventHandlers == nullptr) ? nullptr :
             m_pDefaultEventHandlers[eEvent];
        if(vInstanceAttrs == nullptr && lpfnDefault == nullptr) 
            return vEvents;
        if(lpfnDefault == nullptr) {
            // no global default. if there is a per-instance
            // event handler defined, return it.
            if(vInstanceAttrs == nullptr || vInstanceAttrs->lpfnHandler == nullptr) 
                return vEvents;
            vEvents = gcnew array<IEventHandler^>(1);
            vEvents[0] = vInstanceAttrs->lpfnHandler;
            return vEvents;
        } else {
            // there is a global handler. if there is also
            // a per instance handler, order the two appropriately,
            // otherwise return only the global handler.
            if(vInstanceAttrs == nullptr) { 
                vEvents = gcnew array<IEventHandler^>(1);
                vEvents[0] = lpfnDefault;
                return vEvents;
            }

            vEvents = gcnew array<IEventHandler^>(2);
            IEventHandler^ lpfnInstance = vInstanceAttrs->lpfnHandler;
            switch(vInstanceAttrs->eInstanceSemantics) {
            case DeferToDefault : vEvents[0] = lpfnDefault; vEvents[1] = nullptr; return vEvents;
            case OverrideDefault: vEvents[0] = lpfnInstance; vEvents[1] = nullptr; return vEvents;
            case PrependDefault : vEvents[0] = lpfnInstance; vEvents[1] = lpfnDefault; return vEvents;
            case PostpendDefault: vEvents[0] = lpfnDefault; vEvents[1] = lpfnInstance; return vEvents;
            }            
        }
        return nullptr;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pushes an object onto this stack. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="block">    [in,out] If non-null, the block to push. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool
    Channel::Push(Datablock^ block)
    {
        CheckDisposed();
        OnChannelEvent(OnChannelEntryAttempt);
        ::PTask::Datablock* nativeBlock = block->GetNativeDatablock();
        bool bSuccess = (GetNativeChannel()->Push(nativeBlock)!=0);
        if(bSuccess) OnChannelEvent(OnChannelEntrySuccess);
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pushes the initializer. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool
    Channel::PushInitializer()
    {
        CheckDisposed();
        OnChannelEvent(OnChannelEntryAttempt);
        bool bSuccess = (GetNativeChannel()->PushInitializer()!=0);
        if(bSuccess) OnChannelEvent(OnChannelEntrySuccess);
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pulls the next block from the graph. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="dwTimeout">    The timeout. </param>
    ///
    /// <returns>   nullptr if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock^
    Channel::Pull(
        __in DWORD dwTimeout
        )
    {
        CheckDisposed();
        OnChannelEvent(OnChannelExitAttempt);
        ::PTask::Datablock* nativeBlock = GetNativeChannel()->Pull(dwTimeout);
        if(nativeBlock != NULL) {
            OnChannelEvent(OnChannelExitSuccess);
            return gcnew Datablock(nativeBlock);
        }
        return nullptr;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pulls the given dwTimeout. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="dwTimeout">    The timeout. </param>
    ///
    /// <returns>   nullptr if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock^
    Channel::Pull()
    {
        return Pull(INFINITE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Determine if we can stream. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   true if we can stream, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool 
    Channel::CanStream(
        void
        )
    {
        CheckDisposed();
        return GetNativeChannel()->CanStream() != 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the capacity. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   The capacity. </returns>
    ///-------------------------------------------------------------------------------------------------

    int 
    Channel::GetCapacity(
        void
        )
    {
        CheckDisposed();
        GetNativeChannel()->Lock();
        int nBlocks = (int)GetNativeChannel()->GetCapacity();
        GetNativeChannel()->Unlock();
        return nBlocks;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a capacity. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="n">    The int to process. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::SetCapacity(
        int n
        )
    {
        CheckDisposed();
        GetNativeChannel()->Lock();
        GetNativeChannel()->SetCapacity((UINT)n);
        GetNativeChannel()->Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets queue occupancy. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   The queue occupancy. </returns>
    ///-------------------------------------------------------------------------------------------------

    int 
    Channel::GetQueueOccupancy(
        void
        )
    {        
        // note that the return value can be stale
        // because we do not provide a lock API at the managed interface,
        // so we are forced to release the lock before returning the 
        // occupancy. Use only as an estimate/hint!
        CheckDisposed();
        GetNativeChannel()->Lock();
        int nBlocks = (int)GetNativeChannel()->GetQueueDepth();
        GetNativeChannel()->Unlock();
        return nBlocks;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Handler, called when the set event. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="lpfnHandler">          [in,out] If non-null, the lpfn handler. </param>
    /// <param name="eHandlerSemantics">    The handler semantics. </param>
    /// <param name="eInstanceSemantics">   The instance semantics. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Channel::SetEventHandler(
        __in IEventHandler^ lpfnHandler,
        __in ChannelEventSemantics eHandlerSemantics,
        __in EventHandlerInstanceSemantics eInstanceSemantics
        )
    {
        msclr::lock lChannel(this);
        InstanceHandlerAttrs^ attrs = gcnew InstanceHandlerAttrs();
        attrs->lpfnHandler = lpfnHandler;
        attrs->eEventSemantics = eHandlerSemantics;
        attrs->eInstanceSemantics = eInstanceSemantics;
        m_pEventHandlers[eHandlerSemantics] = attrs;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   default handlers, called on push/pull. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="lpfnHandler">      [in,out] If non-null, the lpfn handler. </param>
    /// <param name="eEventSemantics">  The event semantics. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::SetDefaultEventHandler(
        __in IEventHandler^ lpfnHandler, 
        __in ChannelEventSemantics eEventSemantics
        )
    {
        msclr::lock lHandler(m_pSyncRoot);
        if(m_pDefaultEventHandlers == nullptr) 
            m_pDefaultEventHandlers = gcnew array<IEventHandler^>(NumEventSemanticsTypes);
        m_pDefaultEventHandlers[eEventSemantics] = lpfnHandler;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="graphInputChannel">    [in,out] If non-null, the graph input channel. </param>
    /// <param name="strName">              [in,out] If non-null, the name. </param>
    ///-------------------------------------------------------------------------------------------------

    GraphInputChannel::GraphInputChannel(
        __in ::PTask::GraphInputChannel* graphInputChannel,
        __in String^ strName
        ) : Channel(graphInputChannel, strName, Input, true) {}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
    /// <param name="strName">  [in,out] If non-null, the name. </param>
    ///-------------------------------------------------------------------------------------------------

    GraphOutputChannel::GraphOutputChannel(
        __in ::PTask::GraphOutputChannel* pChannel,
        __in String^ strName
        ) : Channel(pChannel, strName, Output, false) {}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
    /// <param name="strName">  [in,out] If non-null, the name. </param>
    ///-------------------------------------------------------------------------------------------------

    InternalChannel::InternalChannel(
        __in ::PTask::InternalChannel* pChannel,
        __in String^ strName
        ) : Channel(pChannel, strName, Internal, false) {}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
    /// <param name="strName">  [in,out] If non-null, the name. </param>
    ///-------------------------------------------------------------------------------------------------

    MultiChannel::MultiChannel(
        __in ::PTask::MultiChannel* pChannel,
        __in String^ strName 
        ) : Channel(pChannel, strName, Multi, true) { }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="vChildren">    [in,out] If non-null, the children. </param>
    /// <param name="strName">      [in,out] If non-null, the name. </param>
    ///-------------------------------------------------------------------------------------------------

    CoalescedChannel::CoalescedChannel(
        __in array<Channel^>^ vChildren,
        __in String^ strName
        ) : Channel(nullptr, strName, Coalesced, true) 
    {
        m_children = vChildren;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets native channel. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the native channel. </returns>
    ///-------------------------------------------------------------------------------------------------

    ::PTask::Channel* 
    CoalescedChannel::GetNativeChannel(
        void
        ) 
    { 
        throw gcnew System::Exception("Attempt to use native channel member of a coalesced channel!");
        return nullptr;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Throw an exception because the member call is a meaningless operation 
    ///             for this channel class. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    CoalescedChannel::OnMeaninglessOperation(
        void
        )
    {
        throw gcnew System::Exception("Attempt to use native channel member of a coalesced channel!");
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pushes an object onto this stack. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <param name="block">    [in,out] If non-null, the block to push. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool 
    CoalescedChannel::Push(
        __in Datablock^ block
        ) 
    { 
        bool bSuccess = true;
        for(int i=0; i<m_children->Length; i++) {
            bSuccess |= m_children[i]->Push(block);
        }
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pushes the initializer. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool 
    CoalescedChannel::PushInitializer(
        void
        ) 
    { 
        bool bSuccess = true;
        for(int i=0; i<m_children->Length; i++) {
            bSuccess |= m_children[i]->PushInitializer();
        }
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Determine if we can stream. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   true if we can stream, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool        
    CoalescedChannel::CanStream(
        void
        ) 
    {
        bool bSuccess = true;
        for(int i=0; i<m_children->Length; i++) {
            bSuccess |= m_children[i]->CanStream();
        }
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is immutable. </summary>
    ///
    /// <remarks>   crossbac, 8/15/2013. </remarks>
    ///
    /// <returns>   true if immutable, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool        
    CoalescedChannel::IsImmutable(
        void
        ) 
    {
        msclr::lock lock(this);
        if(!m_bImmutableValid || true) {
            m_bImmutable = true;
            for(int i=0; i<m_children->Length; i++) {
                m_bImmutable |= m_children[i]->Immutable;
            }
        }
        return m_bImmutable;
    }

    // actions that don't make sense for a coalesced channel.
    DataTemplate^   CoalescedChannel::GetTemplate() { OnMeaninglessOperation(); return nullptr; }
    int CoalescedChannel::GetChannelPredicationType() { OnMeaninglessOperation(); return 0;}
    void  CoalescedChannel::SetChannelPredicationType(int n) { UNREFERENCED_PARAMETER(n); OnMeaninglessOperation(); } 
    int CoalescedChannel::GetViewMaterializationPolicy() { OnMeaninglessOperation(); return 0; }
    void CoalescedChannel::SetViewMaterializationPolicy(int n) { UNREFERENCED_PARAMETER(n); OnMeaninglessOperation(); }
    int CoalescedChannel::GetCapacity() { OnMeaninglessOperation(); return 0; } 
    void CoalescedChannel::SetCapacity(int n) { UNREFERENCED_PARAMETER(n); OnMeaninglessOperation(); }
    int CoalescedChannel::GetQueueOccupancy() { OnMeaninglessOperation(); return 0; }
    Datablock^ CoalescedChannel::Pull() { OnMeaninglessOperation(); return nullptr; }
    Datablock^ CoalescedChannel::Pull(DWORD dwTimeout) { UNREFERENCED_PARAMETER(dwTimeout); OnMeaninglessOperation(); return nullptr; }


}}}
