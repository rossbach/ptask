//--------------------------------------------------------------------------------------
// File: multichannel.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include <stdio.h>
#include <crtdbg.h>
#include "multichannel.h"
#include "ptaskutils.h"
#include "port.h"
#include "task.h"
#include <assert.h>
#include "PTaskRuntime.h"
#include "Scheduler.h"

using namespace std;

#ifdef CHECK_CRITICAL_PATH_ALLOC
#define check_critical_path_alloc()                                                \
    if(Runtime::GetCriticalPathAllocMode() && Scheduler::GetRunningGraphCount()) { \
	    PTask::Runtime::MandatoryInform("%s::%s(%s) crit-path alloc: CLONE!\n",    \
                                        __FILE__,                                  \
                                        __FUNCTION__);                             \
    }
#else
#define check_critical_path_alloc()
#endif

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pGraph">                   [in,out] If non-null, the graph. </param>
    /// <param name="pTemplate">                [in,out] If non-null, the template. </param>
    /// <param name="hRuntimeTerminateEvent">   Handle of the terminate event. </param>
    /// <param name="hGraphTeardownEvent">      Handle of the stop event. </param>
    /// <param name="hGraphStopEvent">          The graph stop event. </param>
    /// <param name="lpszChannelName">          [in] If non-null, name of the channel. </param>
    /// <param name="bHasBlockPool">            The has block pool. </param>
    ///-------------------------------------------------------------------------------------------------

    MultiChannel::MultiChannel(
        __in Graph * pGraph,
        __in DatablockTemplate * pTemplate,
        __in HANDLE hRuntimeTerminateEvent,
        __in HANDLE hGraphTeardownEvent,
        __in HANDLE hGraphStopEvent,
        __in char * lpszChannelName,
        __in BOOL bHasBlockPool
        ) : Channel(pGraph,
                    pTemplate, 
                    hRuntimeTerminateEvent,
                    hGraphTeardownEvent,
                    hGraphStopEvent,
                    lpszChannelName,
                    bHasBlockPool)
    {
        m_type = CT_MULTI;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    MultiChannel::~MultiChannel() {
        Lock();
        assert(m_uiRefCount == 0);
        map<UINT, Channel*>::iterator ci;
        for(ci=m_pChannelMap.begin(); ci!=m_pChannelMap.end(); ci++) {
            Channel * pChannel = ci->second;
            pChannel->Release();
        }
        Unlock();
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
    MultiChannel::CanStream(
        VOID
        )
    {
        Lock();
        BOOL bResult = TRUE;
        std::map<UINT, Channel*>::iterator mi;
        for(mi=m_pChannelMap.begin(); mi!=m_pChannelMap.end(); mi++) {
            Channel * pChannel = mi->second;
            bResult &= pChannel->CanStream();
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Coalesce channel. </summary>
    ///
    /// <remarks>   Crossbac, 1/20/2012. </remarks>
    ///
    /// <param name="pChannel"> [in,out] If non-null, the channel. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    MultiChannel::CoalesceChannel(
        Channel * pChannel
        )
    {
        Lock();
        size_t nChannel = m_pChannelMap.size();
        m_pChannelMap[(UINT)nChannel] = pChannel;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the channel is ready. This is an operation that can have a different
    ///             result depending on what end of the channel we are talking about. At the source
    ///             end, the channel is ready if it has capacity, while at the destination end the
    ///             channel is ready if there are blocks available to consume.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="type"> (optional) the type of the channel endpoint. </param>
    ///
    /// <returns>   true if ready, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    MultiChannel::IsReady(
        CHANNELENDPOINTTYPE type
        ) 
    {
        Lock();
        BOOL bReady = TRUE;
        map<UINT, Channel*>::iterator ci;
        switch(type) {
        case CE_SRC:
            for(ci=m_pChannelMap.begin(); ci!=m_pChannelMap.end(); ci++) {
                Channel * pChannel = ci->second;
                if(!pChannel->IsReady()) {
                    bReady = FALSE;
                    break;
                }
            }
            break;
        case CE_DST:
            bReady = GetQueueDepth() > 0;
            break;
        }
        Unlock();
        return bReady;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Not meaningful for multi-channels. assert false
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
    MultiChannel::Pull(
        DWORD dwTimeout
        ) 
    {
        // never pull from a multi-channel!
        dwTimeout;
        assert(FALSE);
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Not meaningful for multi-channels. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name=""> The VOID to peek. </param>
    ///
    /// <returns>   null if it fails, else the currently available datablock object. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    MultiChannel::Peek(
        VOID
        ) 
    {
        assert(false);
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the downstream writer port count. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The downstream writer port count. </returns>
    ///-------------------------------------------------------------------------------------------------
   
    UINT            
    MultiChannel::GetDownstreamWriterPortCount(
        VOID
        ) 
    {
        // return the number of downstream ports/channels
        // of this port can that modify the contents
        // of this datablock making it potentiall unsafe
        // to readshare it by pushing the same
        // block into multiple downstream channels.
        Lock();
        UINT nResult = 0;
        map<UINT, Channel*>::iterator ci;
        for(ci=m_pChannelMap.begin(); ci!=m_pChannelMap.end(); ci++) {
            Channel * pChannel = ci->second;
            if(pChannel->HasDownstreamWriters()) {
                nResult++;
            }
        }
        Unlock();
        return nResult;
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
    MultiChannel::HasDownstreamWriters(
        VOID
        )
    {
        return GetDownstreamWriterPortCount() > 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the downstream readonly port count. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The downstream readonly port count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    MultiChannel::GetDownstreamReadonlyPortCount(
        VOID
        )
    {
        // return true if anything downstream 
        // of this port consumes the block read-only
        Lock();
        UINT nResult = 0;
        map<UINT, Channel*>::iterator ci;
        for(ci=m_pChannelMap.begin(); ci!=m_pChannelMap.end(); ci++) {
            Channel * pChannel = ci->second;
            if((pChannel->GetType() == CT_GRAPH_OUTPUT)) {
                continue;
            }
            Port * pPort = pChannel->GetBoundPort(CE_DST);
            if(pPort == NULL) continue;
            if(pPort->IsInOutParameter() || pPort->IsDestructive()) {
                continue;
            }
            // if there is a channel that is not connected
            // to either an inout port or is an output
            // that we have a readonly consumer. 
            nResult++;
        }
        Unlock();
        return nResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pushes a datablock into this channel, blocking until there is capacity
    /// 			for an optional timeout in milliseconds. Default timeout is infinite. 
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
    MultiChannel::Push(
        Datablock* pBlock,
        DWORD dwTimeout
        ) 
    {
        // first, check that this block passes any channel predication--no sense waiting for space if
        // we aren't going to push the block. We don't need to hold the channel lock to check the
        // predicate. 
        dwTimeout;  // suppress compiler warning
        BOOL bSuccess = FALSE;
        assert(pBlock != NULL);
        if(!PassesPredicate(CE_SRC, pBlock)) {
            // this block is being gated 
            return TRUE;
        }
        if(m_pChannelMap.size() != 0) {
            bSuccess = TRUE;
            // if there are multiple channels downstream on 
            // this port, and the are not all readonly 
            // consumers, then we need to clone the block
            // for some consumers. Of course, we can use the
            // block itself for one input.
            UINT nReaders = GetDownstreamReadonlyPortCount();
            UINT nWriters = GetDownstreamWriterPortCount();
            BOOL bRequiresClone = (nReaders && nWriters) || (nWriters > 1);
            UINT nCloneCount = bRequiresClone ? (nReaders + nWriters - 1) : 0;
            assert(!bRequiresClone || nCloneCount > 0);
            map<UINT, Channel*>::iterator ci;
            for(ci=m_pChannelMap.begin(); ci!=m_pChannelMap.end(); ci++) {
                Channel * pChannel = ci->second;
                // we may pass this datablock on to a
                // consumer who may modify it, or who may 
                // fail to read it before the next invocation
                // of this ptask. 
                if(bRequiresClone && nCloneCount) {
                    pBlock->Lock();
                    check_critical_path_alloc();
                    Datablock * pClone = Datablock::Clone(pBlock, NULL, NULL);
                    pBlock->Unlock();
                    pClone->AddRef();
                    nCloneCount--;
                    assert(nCloneCount >= 0);
                    if(pChannel->GetType() == CT_GRAPH_OUTPUT) {
                        bSuccess &= pChannel->Push(pClone);
                        continue;
                    } 
                    Port * pDSPort = pChannel->GetBoundPort(CE_DST);
                    assert(pDSPort != NULL);
                    if(pDSPort->IsInOutParameter()) {
                        bSuccess &= pChannel->Push(pClone);
                        continue;
                    }
                    pClone->Release();
                }
                bSuccess &= pChannel->Push(pBlock);
            }
        }
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind a port to this channel. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pPort">    [in] non-null, the port to bind. </param>
    /// <param name="type">     (optional) the type of the channel. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    MultiChannel::BindPort(
        Port * pPort, 
        CHANNELENDPOINTTYPE type) 
    {
        Lock();
        switch(type) {
        case CE_SRC: m_pSrcPort = pPort; break;
        case CE_DST: assert(false); break;
        default: assert(false); break;
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind a port from this channel. Return the recently unbound port if the
    ///             operation succeeds.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="type"> (optional) the type of the channel. </param>
    ///
    /// <returns>   null if it fails, else the unbound port. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port * 
    MultiChannel::UnbindPort(
        CHANNELENDPOINTTYPE type
        ) 
    {
        Port * pResult = NULL;
        Lock();
        switch(type) {
        case CE_SRC: pResult = m_pSrcPort; m_pSrcPort = NULL; break;
        case CE_DST: assert(false); break;
        default: assert(false); return NULL;
        }
        Unlock();
        return pResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the port bound at the specified endpoint of the channel. No lock is
    ///             required because the graph structure is assumed to be static once it is running.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="type"> (optional) the type of the channel. </param>
    ///
    /// <returns>   null if it fails, else the bound port. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port * 
    MultiChannel::GetBoundPort(
        CHANNELENDPOINTTYPE type
        ) 
    {
        switch(type) {
        case CE_SRC: return m_pSrcPort;
        case CE_DST: return NULL;
        default: assert(false); return NULL;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Drains the channel be releasing this channel's references to all blocks in the
    ///             queue. The queue is cleared on exit.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    MultiChannel::Drain(
        VOID
        )
    {
        Lock();
        map<UINT, Channel*>::iterator ci;
        for(ci=m_pChannelMap.begin(); ci!=m_pChannelMap.end(); ci++) {
            Channel * pChannel = ci->second;
            pChannel->Drain();
        }
        Unlock();
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
    MultiChannel::CheckTypeSpecificSemantics(
        std::ostream * pos,
        PTask::Graph * pGraph
        ) 
    {
        // no additional checks at present. 
        UNREFERENCED_PARAMETER(pGraph);
        UNREFERENCED_PARAMETER(pos);
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the coalesced channel map. </summary>
    ///
    /// <remarks>   crossbac, 4/18/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the coalesced channel map. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::map<UINT, Channel*>* 
    MultiChannel::GetCoalescedChannelMap(
        VOID
        )
    {
        return &this->m_pChannelMap;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the current queue depth. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The queue depth. </returns>
    ///-------------------------------------------------------------------------------------------------

    size_t 
    MultiChannel::GetQueueDepth(
        VOID
        )
    {
        assert(LockIsHeld());
        size_t szMaxOverCoalescedChannels = 0;
        map<UINT, Channel*>::iterator mi;
        for(mi=m_pChannelMap.begin(); mi!=m_pChannelMap.end(); mi++) 
            mi->second->Lock();
        for(mi=m_pChannelMap.begin(); mi!=m_pChannelMap.end(); mi++) 
            szMaxOverCoalescedChannels = max(szMaxOverCoalescedChannels, mi->second->GetQueueDepth());
        for(mi=m_pChannelMap.begin(); mi!=m_pChannelMap.end(); mi++) 
            mi->second->Unlock();
        return szMaxOverCoalescedChannels;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Sets the capacity of the channel, which is the maximum number of datablocks it can queue
    ///     before subsequent calls to push will block.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="nCapacity">    The capacity. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    MultiChannel::SetCapacity(
        UINT nCapacity
        )
    {
        Lock();
        map<UINT, Channel*>::iterator mi;
        for(mi=m_pChannelMap.begin(); mi!=m_pChannelMap.end(); mi++) 
            mi->second->Lock();
        for(mi=m_pChannelMap.begin(); mi!=m_pChannelMap.end(); mi++) 
            mi->second->SetCapacity(nCapacity);
        for(mi=m_pChannelMap.begin(); mi!=m_pChannelMap.end(); mi++) 
            mi->second->Unlock();
        m_uiCapacity = nCapacity;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the capacity. </summary>
    ///
    /// <remarks>   Crossbac, 7/10/2013. </remarks>
    ///
    /// <returns>   The capacity. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    MultiChannel::GetCapacity(
        VOID
        )
    {
        Lock();
        UINT uiMinCapacity = MAXDWORD32;
        map<UINT, Channel*>::iterator mi;
        for(mi=m_pChannelMap.begin(); mi!=m_pChannelMap.end(); mi++) 
            mi->second->Lock();
        for(mi=m_pChannelMap.begin(); mi!=m_pChannelMap.end(); mi++) 
            uiMinCapacity = min(mi->second->GetCapacity(), uiMinCapacity);
        for(mi=m_pChannelMap.begin(); mi!=m_pChannelMap.end(); mi++) 
            mi->second->Unlock();
        assert(m_uiCapacity == uiMinCapacity);
        Unlock();
        return uiMinCapacity;
    }

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Populate a set of tasks that are bound to this channel as consumers. Because a
	/// 			channel may be an output channel or a multi-channel, the range of cardinality of
	/// 			this result is [0..n]. Return the number of such tasks. Note that we cache the
	/// 			result of this call: computing it requires a transitive closure over paths that
	/// 			can include multi-channels and in/out routing, which in turn means traversing the
	/// 			graph recursively. Since the result of this traversal cannot change, and the
	/// 			traversal requires locking parts of the graph, we prefer to avoid repeating work
	/// 			to recompute the same result.
	/// 			</summary>
	///
	/// <remarks>	Crossbac, 10/2/2012. </remarks>
	///
	/// <param name="pvTasks">	[in,out] non-null, the tasks. </param>
	///
	/// <returns>	The number of downstream consuming tasks. </returns>
	///-------------------------------------------------------------------------------------------------

	UINT 
	MultiChannel::GetDownstreamTasks(
		__inout  std::set<Task*>* pvTasks
		)
	{
		// if we have already computed this result,
		// then return it, since it cannot change. 
		if(m_pvDownstreamTasks != NULL) {
			pvTasks->insert(m_pvDownstreamTasks->begin(), m_pvDownstreamTasks->end());
			return static_cast<UINT>(pvTasks->size());
		}

		m_pvDownstreamTasks = new std::set<Task*>();
		map<UINT, Channel*>::iterator ci;
        for(ci=m_pChannelMap.begin(); ci!=m_pChannelMap.end(); ci++) {
            Channel * pChannel = ci->second;
            if((pChannel->GetType() == CT_GRAPH_OUTPUT)) {
                continue;
            }
            Port * pPort = pChannel->GetBoundPort(CE_DST);
            if(pPort == NULL) continue;
			pChannel->GetDownstreamTasks(m_pvDownstreamTasks);
        }
		pvTasks->insert(m_pvDownstreamTasks->begin(), m_pvDownstreamTasks->end());
		return static_cast<UINT>(pvTasks->size());
	}


	///-------------------------------------------------------------------------------------------------
	/// <summary>	Gets memory spaces downstream of this channel that either *must* consume data
	/// 			that flows through this channel, or *may* consume it. The list is non-trivial
	/// 			because of different channel types and predication. For example, an output
	/// 			channel has no downstream consumers, while a multi-channel can have any number.
	/// 			Enumerating consumers is complicated by the following additional factors:
	/// 			
	/// 			1) The presence of channel predicates can ensure dynamically that a particular
	/// 			bound task never actually consumes a block flowing through it.
	/// 			
	/// 			2) If the channel is bound to In/out ports, then we need to analyze paths of
	/// 			length greater than 1. In fact, we need the transitive closure.
	/// 			
	/// 			3) A task's accelerator class may enable it to be bound to several different
	/// 			accelerators, meaning the list of potential consumers can be greater than 1 even
	/// 			if the channel binding structure is trivial.
	/// 			
	/// 			Note that we cache the result of this call: computing it requires a transitive
	/// 			closure over paths that can include multi-channels and in/out routing, which in
	/// 			turn means traversing the graph recursively. Since the result of this traversal
	/// 			cannot change, and the traversal requires locking parts of the graph, we prefer
	/// 			to avoid repeating work to recompute the same result.
	/// 			</summary>
	///
	/// <remarks>	Crossbac, 10/2/2012. </remarks>
	///
	/// <param name="pvMandatoryAccelerators">	[in,out] non-null, the mandatory accelerators. </param>
	/// <param name="pvPotentialAccelerators">	[in,out] non-null, the potential accelerators. </param>
	///
	/// <returns>	The downstream memory spaces. </returns>
	///-------------------------------------------------------------------------------------------------

	BOOL
	MultiChannel::EnumerateDownstreamMemorySpaces(
		__inout	 std::set<Accelerator*>* pvMandatoryAccelerators,
		__inout  std::set<Accelerator*>* pvPotentialAccelerators
		)
	{
		if(m_pvMandatoryDownstreamAccelerators != NULL) {
			assert(m_pvPotentialDownstreamAccelerators != NULL);
			if(m_pvMandatoryDownstreamAccelerators != pvMandatoryAccelerators) {
				pvMandatoryAccelerators->insert(m_pvMandatoryDownstreamAccelerators->begin(), 
												m_pvMandatoryDownstreamAccelerators->end());
				pvPotentialAccelerators->insert(m_pvPotentialDownstreamAccelerators->begin(),
												m_pvPotentialDownstreamAccelerators->end());
			}
			return TRUE;
		}

		assert(m_pvMandatoryDownstreamAccelerators == NULL);
		assert(m_pvPotentialDownstreamAccelerators == NULL);
		m_pvMandatoryDownstreamAccelerators = new std::set<Accelerator*>();
		m_pvPotentialDownstreamAccelerators = new std::set<Accelerator*>();
		map<UINT, Channel*>::iterator ci;
        for(ci=m_pChannelMap.begin(); ci!=m_pChannelMap.end(); ci++) {
            Channel * pChannel = ci->second;
            if((pChannel->GetType() == CT_GRAPH_OUTPUT)) {
                continue;
            }
            Port * pPort = pChannel->GetBoundPort(CE_DST);
            if(pPort == NULL) continue;
			pChannel->EnumerateDownstreamMemorySpaces(m_pvMandatoryDownstreamAccelerators, 
													  m_pvPotentialDownstreamAccelerators);
        }
		pvMandatoryAccelerators->insert(m_pvMandatoryDownstreamAccelerators->begin(), 
										m_pvMandatoryDownstreamAccelerators->end());
		pvPotentialAccelerators->insert(m_pvPotentialDownstreamAccelerators->begin(),
										m_pvPotentialDownstreamAccelerators->end());
		return (pvMandatoryAccelerators->size() != 0) ||
			   (pvPotentialAccelerators->size() != 0);
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Find the maximal capacity downstream port/channel path starting at this channel.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 1/3/2014. </remarks>
    ///
    /// <param name="vTasksVisited">    [in,out] [in,out] If non-null, the tasks visited. </param>
    /// <param name="vPath">            [in,out] list of channels along the maximal path. </param>
    ///
    /// <returns>   The found maximal downstream capacity. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    MultiChannel::FindMaximalDownstreamCapacity(
        __inout std::set<Task*>& vTasksVisited,
        __inout std::vector<Channel*>& vPath
        )
    {
        UINT uiMaxCapacity = 0;
        Channel * pMaximalChannel = NULL;
        std::vector<Channel*>::iterator vi;
        std::vector<Channel*> vMaximalOutboundPath;

		map<UINT, Channel*>::iterator ci;
        for(ci=m_pChannelMap.begin(); ci!=m_pChannelMap.end(); ci++) {

            // traverse all the coalesced channels recursively 
            // visiting outbound in/out pairs to find the
            // maximal channel capacity for this channel. 

            Channel * pOutboundChannel = ci->second;
            UINT uiChannelCapacity = pOutboundChannel->GetCapacity();
            Port * pDownstreamPort = pOutboundChannel->GetBoundPort(CE_DST);

            if(pDownstreamPort != NULL) {

                std::vector<Channel*> vOutboundPath;
                UINT uiDownstreamCapacity = pDownstreamPort->FindMaximalDownstreamCapacity(vTasksVisited, vOutboundPath);
                if(uiChannelCapacity + uiDownstreamCapacity > uiMaxCapacity) {
                    vMaximalOutboundPath.clear();
                    uiMaxCapacity = uiChannelCapacity + uiDownstreamCapacity;
                    vMaximalOutboundPath.assign(vOutboundPath.begin(), vOutboundPath.end());
                    pMaximalChannel = pOutboundChannel;
                }

            } else {

                // this channel is not bound to a destination port. For a
                // multi-channel (which is really an input construct) this
                // is an error in graph structure. 

                assert(pDownstreamPort != NULL);
                PTask::Runtime::HandleError("%s::%s: unconnected multi-channel component channel %s.%s!\n",
                                            __FILE__,
                                            __FUNCTION__,
                                            m_lpszName,
                                            pOutboundChannel->GetName());
            }
        }

        if(pMaximalChannel) {
            vPath.push_back(pMaximalChannel);           
            for(vi=vMaximalOutboundPath.begin(); vi!=vMaximalOutboundPath.end(); vi++)
                vPath.push_back(*vi);
        }
        return uiMaxCapacity;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this channel has any non trivial predicates. </summary>
    ///
    /// <remarks>   crossbac, 7/3/2014. </remarks>
    ///
    /// <returns>   true if non trivial predicate, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    MultiChannel::HasNonTrivialPredicate(
        VOID
        )
    {
        map<UINT, Channel*>::iterator ci;
        for(ci=m_pChannelMap.begin(); ci!=m_pChannelMap.end(); ci++) 
            if(ci->second->HasNonTrivialPredicate())
                return TRUE;
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
    MultiChannel::GetControlSignalsOfInterest(
        VOID
        )
    {
        CONTROLSIGNAL luiSignals = DBCTLC_NONE;
        map<UINT, Channel*>::iterator ci;
        for(ci=m_pChannelMap.begin(); ci!=m_pChannelMap.end(); ci++) 
            luiSignals |= ci->second->GetControlSignalsOfInterest();
        return luiSignals;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this multi-channel has an exposed component channel. </summary>
    ///
    /// <remarks>   crossbac, 7/7/2014. </remarks>
    ///
    /// <returns>   true if exposed component channel, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    MultiChannel::HasExposedComponentChannel(
        VOID
        )
    {
        MultiChannel * pMChannel = NULL;
        map<UINT, Channel*>::iterator ci;
        for(ci=m_pChannelMap.begin(); ci!=m_pChannelMap.end(); ci++) {
            CHANNELTYPE eType = ci->second->GetType();
            switch(eType) {
            case CHANNELTYPE::CT_INTERNAL: continue;
            case CHANNELTYPE::CT_GRAPH_INPUT: return TRUE;
            case CHANNELTYPE::CT_GRAPH_OUTPUT: return TRUE;
            case CHANNELTYPE::CT_INITIALIZER: if((ci->second)->HasNonTrivialPredicate()) return TRUE; continue;
            case CHANNELTYPE::CT_MULTI: 
                pMChannel = dynamic_cast<MultiChannel*>(ci->second);
                if(pMChannel && pMChannel->HasExposedComponentChannel())
                    return TRUE;
                continue;
            }
        }
        return FALSE;
    }



};
