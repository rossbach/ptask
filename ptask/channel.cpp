//--------------------------------------------------------------------------------------
// File: channel.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "ptaskutils.h"
#include "PTaskRuntime.h"
#include "channel.h"
#include "ChannelProfiler.h"
#include "AsyncContext.h"
#include "port.h"
#include "OutputPort.h"
#include "InputPort.h"
#include "MetaPort.h"
#include "task.h"
#include "graph.h"
#include "Scheduler.h"
#include "shrperft.h"
#include "instrumenter.h"
#include "Recorder.h"
#include "signalprofiler.h"
#include <assert.h>
#include <vector>
#include <algorithm>
using namespace PTask::Runtime;

#ifdef PROFILE_CHANNELS
#define init_channel_stats(x) { \
    x = (PTask::Runtime::GetChannelProfilingEnabled()) ? \
        new ChannelProfiler(this) : NULL; }
#define merge_channel_stats(x) { \
    if(PTask::Runtime::GetChannelProfilingEnabled() && x) \
        x->MergeInstanceStatistics(); }
#define check_channel_invariants()  CheckQueueInvariants()
#else
#define init_channel_stats(x)
#define merge_channel_stats(x) 
#define check_channel_invariants()  
#endif

namespace PTask {

    static const int MAX_CHANNEL_EVENT_NAME = 256;
    static const int MAX_CHANNEL_EVENT_= MAX_PATH;
    ULONG Channel::m_nKernelObjUniquifier = 0;
    ULONG Channel::m_bRandSeeded = FALSE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pGraph">                   [in,out] The graph (set only if the trigger channel
    ///                                         flag is on). </param>
    /// <param name="pTemplate">                [in,out] If non-null, the template. </param>
    /// <param name="hRuntimeTerminateEvent">   This channel's copy of the handle to the runtime
    ///                                         terminate event. Blocking calls must wait on this
    ///                                         event as well as whatever other wait object is
    ///                                         semantically meaningful for the channel to ensure
    ///                                         that Task dispatch threads can unblock when the graph
    ///                                         gets torn down. </param>
    /// <param name="hGraphTeardownEvent">      This channel's copy of the handle to the graph
    ///                                         teardown event. Some blocking calls must wait on this
    ///                                         event as well as whatever other wait object is
    ///                                         semantically meaningful for the channel to ensure
    ///                                         that Task dispatch threads can unblock when the graph
    ///                                         gets stopped. Currently, it is OK to leave threads
    ///                                         blocked on a push/pull call if the graph is
    ///                                         *stopping* or stopped, since the user can start the
    ///                                         graph again. We only want to unblock calls when the
    ///                                         caller is actually tearing the graph apart. </param>
    /// <param name="hGraphStopEvent">          This channel's copy of the handle to the graph stop
    ///                                         event. Some blocking calls must wait on this event as
    ///                                         well as whatever other wait object is semantically
    ///                                         meaningful for the channel to ensure that Task
    ///                                         dispatch threads can unblock when the graph gets
    ///                                         stopped. Currently, it is OK to leave threads blocked
    ///                                         on a push/pull call if the graph is *stopping* or
    ///                                         stopped, since the user can start the graph again. We
    ///                                         only want to unblock calls when the caller is
    ///                                         actually tearing the graph apart. </param>
    /// <param name="lpszChannelName">          [in] If non-null, name of the channel. </param>
    /// <param name="bHasBlockPool">            The has block pool. </param>
    ///-------------------------------------------------------------------------------------------------

    Channel::Channel(
        __in Graph * pGraph,
        __in DatablockTemplate * pTemplate,
        __in HANDLE hRuntimeTerminateEvent,
        __in HANDLE hGraphTeardownEvent,
        __in HANDLE hGraphStopEvent,
        __in char * lpszChannelName,
        __in BOOL bHasBlockPool
        ) :
        ReferenceCounted(lpszChannelName)
    {
        UNREFERENCED_PARAMETER(bHasBlockPool);
        WCHAR lpwszEmptyEvtName[MAX_CHANNEL_EVENT_NAME];
        WCHAR lpwszCapacityEvtName[MAX_CHANNEL_EVENT_NAME];
        WCHAR lpwszAvailableEvtName[MAX_CHANNEL_EVENT_NAME];
        if(!InterlockedExchange(&m_bRandSeeded, TRUE)) {
            srand((UINT) GetTickCount());
        }
        if(lpszChannelName) {
            size_t cvt = 0;
            WCHAR lpswzName[MAX_CHANNEL_EVENT_NAME];
            mbstowcs_s(&cvt, lpswzName, MAX_CHANNEL_EVENT_NAME, lpszChannelName, MAX_CHANNEL_EVENT_NAME);
            wsprintf(lpwszEmptyEvtName, L"%s.Channel.Empty.%d", lpswzName, ptaskutils::nextuid());
            wsprintf(lpwszCapacityEvtName, L"%s.Channel.HasCapacity.%d", lpswzName, ptaskutils::nextuid());
            wsprintf(lpwszAvailableEvtName, L"%s.Channel.Available.%d", lpswzName, ptaskutils::nextuid());
            m_lpszName = new char[strlen(lpszChannelName)+1];
            strcpy_s(m_lpszName, strlen(lpszChannelName)+1, lpszChannelName);
            m_bUserSpecifiedName = TRUE;
        } else {
            wsprintf(lpwszEmptyEvtName, L"PTask.Channel.Empty.%d", ptaskutils::nextuid());
            wsprintf(lpwszCapacityEvtName, L"PTask.Channel.HasCapacity.%d", ptaskutils::nextuid());
            wsprintf(lpwszAvailableEvtName, L"PTask.Channel.Available.%d", ptaskutils::nextuid());
            m_lpszName = CreateUniqueName();
            m_bUserSpecifiedName = FALSE;
        }
        m_pTemplate = pTemplate;
        if(m_pTemplate != NULL) {
            m_pTemplate->AddRef();
        }

        m_uiCapacity                        = Runtime::GetDefaultChannelCapacity();	
        m_hEmpty                            = CreateEvent(NULL, TRUE, TRUE, lpwszEmptyEvtName);
        m_hHasCapacity                      = CreateEvent(NULL, TRUE, TRUE, lpwszCapacityEvtName);
        m_hAvailable                        = CreateEvent(NULL, TRUE, FALSE, lpwszAvailableEvtName);
        m_hRuntimeTerminateEvent            = hRuntimeTerminateEvent;
        m_hGraphStopEvent                   = hGraphStopEvent;
        m_hGraphTeardownEvent               = hGraphTeardownEvent;
        m_bEmpty                            = TRUE;
        m_bHasCapacity                      = TRUE;
        m_bAvailable                        = FALSE;
        m_uiRefCount                        = 0;
        m_dwTimeout                         = DEFAULT_CHANNEL_TIMEOUT;
        m_pSrcPort                          = NULL;
        m_pDstPort                          = NULL;
        m_viewMaterializationPolicy         = PTask::Runtime::GetDefaultViewMaterializationPolicy();
        m_luiPropagatedControlCode          = DBCTLC_NONE;
        m_luiInitialPropagatedControlCode   = DBCTLC_NONE;
        m_pControlPropagationSource         = NULL;
		m_pvDownstreamTasks                 = NULL;
		m_pvMandatoryDownstreamAccelerators = NULL;
		m_pvPotentialDownstreamAccelerators = NULL;
        m_bIsTriggerChannel                 = FALSE;
        m_bWantMostRecentView               = FALSE;
        m_pGraph                            = pGraph;
        m_draw                              = true;
        m_uiBlockTransitLimit               = 0;
        m_uiBlocksDelivered                 = 0;
        m_vPredicators[CE_SRC].eEndpoint    = CE_SRC;
        m_vPredicators[CE_DST].eEndpoint    = CE_DST;
        m_vPredicators[CE_SRC].ePredicateFailureAction = PFA_RELEASE_BLOCK;
        m_vPredicators[CE_DST].ePredicateFailureAction = PFA_FAIL_OPERATION;
        m_uiMaxOccupancy                    = 0;
        m_uiCumulativeOccupancy             = 0;
        m_bHasBlockPool                     = bHasBlockPool;
        init_channel_stats(m_pChannelProfile);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Channel::~Channel() {

        Lock();
        assert(m_uiRefCount == 0);      
        merge_channel_stats(m_pChannelProfile);
        while(m_q.size()) {
            Datablock * pBlock = m_q.back();
            std::cout << this << "draining unconsumed " << pBlock << std::endl;
            m_q.pop_back();
            pBlock->Release();
        }
        if(m_pTemplate) {
            m_pTemplate->Release();
        }
        Unlock();
        if(m_hEmpty) CloseHandle(m_hEmpty);
        if(m_hHasCapacity) CloseHandle(m_hHasCapacity);
        if(m_hAvailable) CloseHandle(m_hAvailable);
        if(m_lpszName) delete [] m_lpszName;
		if(m_pvDownstreamTasks) delete m_pvDownstreamTasks;
		if(m_pvMandatoryDownstreamAccelerators) delete m_pvMandatoryDownstreamAccelerators;
		if(m_pvPotentialDownstreamAccelerators) delete m_pvPotentialDownstreamAccelerators;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check queue invariants--given a queue of blocks and a channel, do some simple
    ///             checks to detect conditions that can lead to incorrect/unexpected results. For
    ///             example, multiple entries of the same object is fine, but only if there are no
    ///             channel predicates (because a control signal change on a block will affect
    ///             multiple queue entries, and therefore has the potential to cause predicates to
    ///             change state at the wrong times). If a channel is a simple cycle with an inout
    ///             port pair, assert that the channel can never have more than one entry.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/27/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::CheckQueueInvariants(
        VOID
        )
    {
        assert(LockIsHeld());
        if(m_q.size()) {

            // make sure there are no duplicate entries in the queue
            // when there are predicates configured for the channel
            
            if((m_vPredicators[CE_DST].eCanonicalPredicate != CGATEFN_NONE) ||
                (m_vPredicators[CE_SRC].eCanonicalPredicate != CGATEFN_NONE)) {

                std::deque<Datablock*>::iterator qi;
                std::set<Datablock*> vUniqueBlocks;
                for(qi=m_q.begin(); qi!=m_q.end(); qi++) 
                    vUniqueBlocks.insert(*qi);

                if(vUniqueBlocks.size() != m_q.size()) {

                    // There are repeated entries in the queue for the same datablock instance. Generally, this is not
                    // a problem--reusing the same block is common, and can lead very naturally to a queue that has
                    // repeated entries for the same datablock object. However, in the presence of predication,
                    // this is a serious problem because it will be impossible to disambiguate which control signal
                    // changes to that block correspond with which entry in the queue for the block. When we
                    // encounter that situation, complain! 
                        
                    Runtime::MandatoryInform("XXXX\n Predicated Channel:%s: repeated entries of the same block instance!\n",
                                             m_lpszName);
                }
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has a user specified name, return false if the 
    ///             runtime generated one on demand for it. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2014. </remarks>
    ///
    /// <returns>   true if user specified name, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Channel::HasUserSpecifiedName(
        VOID
        )
    {
        return m_bUserSpecifiedName;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a unique name. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    char * 
    Channel::CreateUniqueName(
        VOID
        )
    {
        char *guidStr = 0x00;
        GUID *pguid = 0x00;
        pguid = new GUID; 
        ::CoCreateGuid(pguid);
        ::UuidToStringA(pguid, ((RPC_CSTR*) &guidStr));
        int guidlen = (int) strlen(guidStr);
        char * lpszChannelName = new char[2*(guidlen+1)];
        memset(lpszChannelName, 0, 2*(guidlen+1));
        strcpy_s(lpszChannelName, guidlen+1, guidStr);
        ::RpcStringFreeA((RPC_CSTR*)&guidStr);
        delete pguid; 
        return lpszChannelName;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the queue capacity. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="nCapacity">    The capacity. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Channel::SetCapacity(
        unsigned int nCapacity
        ) 
    {
        // TODO:
        // XXXX:
        // probably need to do something about draining
        // in the case where new capacity is less than
        // the old capacity? 
        Lock();
        m_uiCapacity = nCapacity;
        if(m_uiCapacity > m_q.size()) {
            m_bHasCapacity = TRUE;
            SetEvent(m_hHasCapacity);
        } else {
            ResetEvent(m_hHasCapacity);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the queue capacity. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="nCapacity">    The capacity. </param>
    ///-------------------------------------------------------------------------------------------------

    UINT
    Channel::GetCapacity(
        VOID
        ) 
    {
        return m_uiCapacity;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this channel has reached any transit limits. </summary>
    ///
    /// <remarks>   crossbac, 6/27/2013. </remarks>
    ///
    /// <returns>   true if transit limit reached, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Channel::IsTransitLimitReached(
        VOID
        )
    {
        Lock();
        BOOL bTransitLimitReached = m_uiBlockTransitLimit && (m_uiBlocksDelivered >= m_uiBlockTransitLimit);
        Unlock();
        return bTransitLimitReached;
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
    Channel::IsReady(
        CHANNELENDPOINTTYPE type
        ) 
    {
        if(IsTransitLimitReached())
            return FALSE;

        Lock();

        // the endpoint type determines how we answer the question. If the question is whether the
        // channel is ready at the DST end, we care whether there available data in the channel, or
        // m_q.size > 0 and the block at the end of queue would pass any predication functions. If the
        // question is about the source end, we are asking whether the queue is full. Note that we
        // don't care about the predicate at the source end because we can still push into it, even if
        // the predicate won't pass. 

        BOOL bValue = FALSE;
        if(type == CE_SRC) {

            bValue = (m_q.size() < m_uiCapacity);
            if(!bValue) {

                // if the queue is full there is still a case where the channel can be ready: an internal
                // channel that creates a single cycle on a task can be full but ready at both ends because
                // executing the task requires dequeuing from the input, which in turn creates space in the
                // queue. So if the queue has reached capacity check for this condition. 

                if(m_pSrcPort != NULL && 
                   m_pDstPort != NULL &&
                   m_pSrcPort->GetTask() == m_pDstPort->GetTask()) {

                    // this channel is a simple cycle, so if the source end is ready, 
                    // then the destination end is ready despite the fact that the queue is full. 
                    bValue = IsReady(CE_DST);
                }
            }

        } else {

            // We're dealing with the CE_DST (destination) end of the queue. If there is something in the
            // queue, and it passes any predication, the channel is ready. 
            
            if(m_q.size() > 0) {

                bValue = TRUE;
                if(m_vPredicators[CE_DST].eCanonicalPredicate != CGATEFN_NONE) {

                    // Result here depends on what is supposed to happen when the predicate fails. If the failure
                    // action is release we need to examine all blocks on the queue to see if a non-null block will
                    // make it through. If the failure action is to fail the operation, we only care about the head
                    // of the queue. 
                    
                    if(m_vPredicators[CE_DST].ePredicateFailureAction == PFA_RELEASE_BLOCK) {
                        
                        std::deque<Datablock*>::reverse_iterator ri = m_q.rbegin();
                        while(ri != m_q.rend()) {
                            Datablock * pBlock = *ri;
                            if(PassesPredicate(CE_DST, pBlock)) {
                                bValue = TRUE;
                                break;
                            }
                            ri++;
                        }

                    } else if(m_vPredicators[CE_DST].ePredicateFailureAction == PFA_FAIL_OPERATION) {
                        
                        Datablock * pBlock = m_q.back();
                        bValue = PassesPredicate(CE_DST, pBlock);
                    }
                }
            }
        }
        Unlock();
        return bValue;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resets the channel. </summary>
    ///
    /// <remarks>   Crossbac, 5/2/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Channel::Reset(
        VOID
        )
    {
        Lock();
        Drain(); 
        if(m_luiInitialPropagatedControlCode) {
            m_luiPropagatedControlCode = m_luiInitialPropagatedControlCode;
        }
        if(m_uiBlockTransitLimit) {
            m_uiBlocksDelivered = 0;
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets block transit limit. Limits this channel to 
    ///             delivering a specified limit before closing. Can be cleared by 
    ///             calling graph::Reset.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/2/2013. </remarks>
    ///
    /// <param name="uiBlockTransitLimit">  The block transit limit. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::SetBlockTransitLimit(
        UINT uiBlockTransitLimit
        )
    {
        Lock();
        m_uiBlockTransitLimit = uiBlockTransitLimit;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the block transit limit. Limits this channel to 
    ///             delivering a specified limit before closing. Can be cleared by 
    ///             calling graph::Reset.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/2/2013. </remarks>
    ///
    /// <param name="uiBlockTransitLimit">  The block transit limit. </param>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Channel::GetBlockTransitLimit(
        VOID
        )
    {
        Lock();
        UINT uiLimit = m_uiBlockTransitLimit;
        Unlock();
        return uiLimit;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pulls the next datablock from the channel, blocking with an optional timeout in
    ///             milliseconds. Default timeout is infinite.
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
    Channel::Pull(
        DWORD dwTimeout
        ) 
    {
        Datablock * pBlock = NULL;
        if(m_uiBlockTransitLimit && (m_uiBlocksDelivered >= m_uiBlockTransitLimit)) {
            return NULL;
        }

        BOOL bPulled = FALSE;
        BOOL bReadyStateTransition = FALSE;

        while(pBlock == NULL) {

            Lock();
            UINT uiOldQueueSize = static_cast<UINT>(m_q.size());
            BOOL bAvailable = uiOldQueueSize != 0;
            if(bAvailable) SetEvent(m_hAvailable);
            Unlock();

            if(!bAvailable) {
                HANDLE vWaitHandles[] = { 
                    m_hAvailable,
                    m_hRuntimeTerminateEvent,
                    m_hGraphTeardownEvent 
                };
                DWORD dwWaitHandles = sizeof(vWaitHandles) / sizeof(HANDLE);
                DWORD dwWait = WaitForMultipleObjects(dwWaitHandles, vWaitHandles, FALSE, dwTimeout);
                switch(dwWait) {
                case WAIT_OBJECT_0 + 0: break;
                case WAIT_TIMEOUT:      return NULL;
                case WAIT_OBJECT_0 + 1: return NULL;
                case WAIT_OBJECT_0 + 2: return NULL;
                default:                return NULL;
                }
            }

            Lock();
            if(m_q.size()) {

                uiOldQueueSize = static_cast<UINT>(m_q.size());

                if(m_vPredicators[CE_DST].eCanonicalPredicate == CGATEFN_NONE) {

                    // there is no predication on this channel, just take the q head.                
                    pBlock = m_q.back();
                    m_q.pop_back();
                    bPulled = TRUE;

                } else {

                    // Result here depends on what is supposed to happen when the
                    // predicate fails. If the failure action is release we need to
                    // examine all blocks on the queue to see if a non-null block
                    // will make it through. If the failure action is to fail the
                    // operation, we only care about the head of the queue. 

                    if(m_vPredicators[CE_DST].ePredicateFailureAction == PFA_RELEASE_BLOCK) {

                        while(m_q.size()) {
                            Datablock * pCandidate = m_q.back();
                            m_q.pop_back();
                            bPulled = TRUE;
                            if(PassesPredicate(CE_DST, pCandidate)) {
                                pBlock = pCandidate;
                                break;
                            } else {
                                pCandidate->Release();
                            }
                        }

                    } else if(m_vPredicators[CE_DST].ePredicateFailureAction == PFA_FAIL_OPERATION) {

                        Datablock * pCandidate = m_q.back();
                        if(PassesPredicate(CE_DST, pCandidate)) {
                            pBlock = pCandidate;
                            m_q.pop_back();
                            bPulled = TRUE;
                        } else {
                            // we're going to spin. at least
                            // back off for a bit.
                            PTask::Runtime::MandatoryInform("warning...spinning on channel predicate!\n");
                            Sleep(50);
                        }
                    }
                }

                if(pBlock) {
                    ClearAllPropagatedControlSignals();
                    ConfigureChannelTriggers(pBlock);
                    m_uiCumulativeOccupancy += uiOldQueueSize;
                    m_uiBlocksDelivered++;
                }

                if(m_q.size() == 0) {
                    SetEvent(m_hEmpty);
                    ResetEvent(m_hAvailable); 
                    m_bEmpty = TRUE;
                    m_bAvailable = FALSE;
                }
                if(m_q.size() < m_uiCapacity) {
                    SetEvent(m_hHasCapacity);
                    m_bHasCapacity = TRUE;
                }

                // if we transitioned from full to empty then
                // upstream graph objects probably want to update
                // their view of their own ready state. wait until
                // we release the lock though to inform them. 
                bReadyStateTransition = m_q.size() < m_uiCapacity &&
                                        uiOldQueueSize == m_uiCapacity;
            }
            Unlock();

            if(bPulled && bReadyStateTransition && m_pSrcPort) {
                Task * pTask = m_pSrcPort->GetTask();
                if(pTask != NULL) {
                    pTask->SignalPortStatusChange();			
                }
            }
        }

        ctlpegress(this, pBlock);
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Returns the first available datablock on the channel without removing it. Return
    ///             null if there is no datablock currently queued on the channel.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name=""> The VOID to peek. </param>
    ///
    /// <returns>   null if it fails, else the currently available datablock object. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Channel::Peek(
        VOID
        ) 
    {
        Datablock * pBlock = NULL;
        if(m_uiBlockTransitLimit && (m_uiBlocksDelivered >= m_uiBlockTransitLimit)) {
            return NULL;
        }
        Lock();
        if(m_q.size()) {

            if(m_vPredicators[CE_DST].eCanonicalPredicate == CGATEFN_NONE) {

                // we only care about the head of the queue because
                // there is no predication in force for this channel.
                pBlock = m_q.back();

            } else {

                // Result here depends on what is supposed to happen when the
                // predicate fails. If the failure action is release we need to
                // examine all blocks on the queue to see if a non-null block
                // will make it through. If the failure action is to fail the
                // operation, we only care about the head of the queue. 
                if(m_vPredicators[CE_DST].ePredicateFailureAction == PFA_RELEASE_BLOCK) {
                    std::deque<Datablock*>::reverse_iterator ri = m_q.rbegin();
                    while(ri != m_q.rend()) {
                        Datablock * pCandidate = *ri;
                        if(PassesPredicate(CE_DST, pCandidate)) {
                            pBlock = pCandidate;
                            break;
                        }
                        ri++;
                    }
                } else if(m_vPredicators[CE_DST].ePredicateFailureAction == PFA_FAIL_OPERATION) {
                    Datablock * pCandidate = m_q.back();
                    if(PassesPredicate(CE_DST, pCandidate))
                        pBlock = pCandidate;
                }
            }
        }
        Unlock();
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Derives an initial value datablock for this channel based on its template,
    /// 			and pushes that datablock into this channel, blocking until there is capacity
    /// 			for an optional timeout in milliseconds. Default timeout is infinite. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="dwTimeout">    (optional) the timeout in milliseconds. Use 0xFFFFFFFF for no
    ///                             timeout. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Channel::PushInitializer(
        DWORD dwTimeout
        )
    {
        DatablockTemplate * pTemplate = GetTemplate();
        assert(!m_bWantMostRecentView);
        assert(pTemplate != NULL && "Channel::PushInitializer called on channel with no template!");
        if(pTemplate == NULL)
            return FALSE;
        BUFFERACCESSFLAGS eFlags = PT_ACCESS_DEFAULT;
        if(m_pDstPort != NULL) {
            if(m_pDstPort->IsInOutParameter()) 
                eFlags = PT_ACCESS_HOST_WRITE |
                         PT_ACCESS_ACCELERATOR_READ | 
                         PT_ACCESS_ACCELERATOR_WRITE;
        }
        Datablock * pBlock = Datablock::CreateInitialValueBlock(NULL, pTemplate, PT_ACCESS_DEFAULT);
#ifdef DEBUG
        pBlock->Lock();
        assert(pBlock->RefCount() == 1);
        pBlock->Unlock();
#endif
        BOOL bPushSuccess = Push(pBlock, dwTimeout);
        pBlock->Release();
        return bPushSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Handle trigger ports. If the given port has global triggers associated with it,
    ///             check the block for control codes and defer execution of those global triggers to
    ///             the port object.
    ///             </summary>
    ///
    /// <remarks>   channel lock must be held.
    ///             
    ///             crossbac, 6/19/2012.
    ///             </remarks>
    ///
    /// <param name="pBlock">   [in] non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::ConfigurePortTriggers(
        __in Datablock * pBlock
        )
    {
        if(m_pDstPort && m_pDstPort->IsTriggerPort()) {
            pBlock->Lock();
            CONTROLSIGNAL luiCode = pBlock->GetControlSignals();
            pBlock->Unlock();
            if(HASSIGNAL(luiCode)) {
#ifdef DEBUG
                if(m_q.size() > 1 && pBlock != m_q.front()) {
                    PTask::Runtime::HandleError("%s::%s: HANDLING SIGNAL(%d) OUT OF ORDER!\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                (UINT) luiCode);
                }
#endif
                m_pDstPort->HandleTriggers(luiCode);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a trigger channel. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    /// <param name="bTrigger"> true to trigger. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Channel::SetTriggerChannel(
        __in Graph * pGraph, 
        __in BOOL bTrigger
        )
    {
        m_pGraph = pGraph;
        m_bIsTriggerChannel = bTrigger;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is trigger port. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <returns>   true if trigger port, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    Channel::IsTriggerChannel(
        VOID
        )
    {
        return m_bIsTriggerChannel && m_pGraph != NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Handle any triggers on the channel itself. If this channel has global
    ///             triggers associated with it, check the block for control codes and defer
    ///             execution of those global triggers to the channel object.
    ///             </summary>
    ///
    /// <remarks>   channel lock must be held.
    ///             
    ///             crossbac, 6/19/2012.
    ///             </remarks>
    ///
    /// <param name="pBlock">   [in] non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::ConfigureChannelTriggers(
        __in Datablock * pBlock
        )
    {
        assert(LockIsHeld());
        if(IsTriggerChannel()) {
            pBlock->Lock();
            CONTROLSIGNAL luiCode = pBlock->GetControlSignals();
            pBlock->Unlock();
            if(HASSIGNAL(luiCode)) {
                HandleTriggers(luiCode);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Handle trigger. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <param name="uiCode">   The code. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Channel::HandleTriggers(
        CONTROLSIGNAL luiCode
        )
    {
        assert(m_bIsTriggerChannel);
        assert(m_pGraph != NULL);
        m_pGraph->ExecuteTriggers(this, luiCode);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Handle any iteration configured for the destination port. If the given port has a
    ///             meta-port with iteration as the meta function, some of that work must be
    ///             performed as part of pushing the block into this channel. Delegate that work to
    ///             the downstream meta port object.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/19/2012.
    ///             </remarks>
    ///
    /// <param name="pDstPort"> [in] non-null, destination port. </param>
    /// <param name="pBlock">   [in] non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::ConfigureDownstreamIterationTargets(
        __in Datablock * pBlock
        )
    {
        // if the destination port is a sticky metaport, we want to configure any of its iteration
        // targets in case the metaport is actually down stream of the iteration target. 

        if(m_pDstPort && m_pDstPort->GetPortType() == META_PORT) {
            MetaPort * pMetaPort = dynamic_cast<MetaPort*>(m_pDstPort);
            METAFUNCTION mf = pMetaPort->GetMetaFunction();
            if(mf == MF_GENERAL_ITERATOR || mf == MF_SIMPLE_ITERATOR) {
                // CJR: 2/28/13:
                // The assert below is antiquated. It is totally possible to use predication
                // structures on the channels involved in an iteration structure to ensure that
                // queued blocks on incoming channels are not accepted until the iteration completes
                // ----------------------------------------------------------------------------------
                ////// this is essentially a hack that will only work if an iteration count is pushed into an empty
                ////// queue. If the queue is non-empty, configuring iteration may trample inflight iterations! 
                ////// assert(m_q.size() == 1); // otherwise, this is the wrong iteration state to use!
                pMetaPort->ConfigureIterationTargets(pBlock);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Attempt to get an asynchronous context object for the task attached at the
    ///             specified endpoint. If multiple accelerators can execute the attached task, we
    ///             cannot make any assumptions about what contexts will be bound to it. Conversely,
    ///             if the task has a single platform type, and a single device exists in the system
    ///             that can execute it, we can figure out what AsyncContext will be used before it
    ///             is bound.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/11/2012. </remarks>
    ///
    /// <param name="eEndpoint">            To which end of the channel is the sought after context
    ///                                     attached? </param>
    /// <param name="eAsyncContextType">    Type of the asynchronous context. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    AsyncContext * 
    Channel::FindAsyncContext(
        __in CHANNELENDPOINTTYPE eEndpoint,
        __in ASYNCCONTEXTTYPE eAsyncContextType
        )
    {
        // if there is only one target accelerator in the system for this ptask's accelerator class,
        // then we know how to get the async context that will be used by the task even if the
        // scheduler has not yet made a dispatch decision. Note that we can very quickly
        // elide traversing the graph for this step if we have a multi-GPU environment because
        // we cannot be sure which GPU will be used.
        
        AsyncContext * pAsyncContext = NULL;
        if(!Runtime::MultiGPUEnvironment()) {

            // look at the port and the base task class.
            // we can infer the async context if the task is not a
            // host task, or if it is and the port has a dependent assignment.            
            Port * pPort = GetBoundPort(eEndpoint);
            Task * pTask = pPort->GetTask();
            if(pTask) {
                ACCELERATOR_CLASS accTaskClass = pTask->GetAcceleratorClass();
                ACCELERATOR_CLASS accPortClass = pPort->GetDependentAcceleratorClass(0);
                ACCELERATOR_CLASS accClass = accPortClass == ACCELERATOR_CLASS_UNKNOWN ? accTaskClass : accPortClass;
                if(accClass != ACCELERATOR_CLASS_HOST) {
                    Accelerator * pAccelerator = NULL;
                    std::set<Accelerator*> vAccelerators;
                    Scheduler::FindEnabledCapableAccelerators(accClass, vAccelerators);
                    assert(vAccelerators.size() <= 1);
                    assert(eAsyncContextType == ASYNCCTXT_XFERHTOD);
                    if(vAccelerators.size() == 1) {
                        pAccelerator = *(vAccelerators.begin());
                        pAsyncContext = pTask->GetOperationAsyncContext(pAccelerator, eAsyncContextType);
                    }
                }
            }
        }
        return pAsyncContext;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   If the destination port has deferred channels, derive blocks based on the
    /// 			descriptor functions, and push them into those channels. </summary>
    ///
    /// <remarks>   channel lock must be held.
    /// 			
    /// 			crossbac, 6/19/2012. </remarks>
    ///
    /// <param name="pDstPort"> [in] non-null, destination port. </param>
    /// <param name="pBlock">   [in] non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Channel::PushDescriptorBlocks(
        __in Datablock * pBlock
        )
    {
        // note that the channel lock should *not* be held for this call.
        // check to see if we have any deferred channels. If we do it means we need to push some
        // derived values into those channels, where the value is another datablock derived from the
        // one being pushed into this channel. 

        if(m_pDstPort == NULL) return;
        std::vector<DEFERREDCHANNELDESC*>* pDeferredChannels = m_pDstPort->GetDeferredChannels();
        if(pDeferredChannels != NULL && pDeferredChannels->size() > 0) {

            // since the control code of the block won't change as we examine 
            // outgoing channels, we might as well grab it at the top of the loop:
            // it requires a lock, and many iterations below may require it.
            pBlock->Lock();
            CONTROLSIGNAL luiControlCode = pBlock->GetControlSignals();
            UINT uiDBUID = pBlock->GetDBUID();
            pBlock->Unlock();

            // cache pointers to derived value blocks to avoid allocating
            // multiple copies of the same info, when one block will suffice.
            std::map<DESCRIPTORFUNC, Datablock*> cache;
            std::map<DESCRIPTORFUNC, Datablock*>::iterator mi;
            Datablock * pDerivedBlock = NULL;  
            CONTROLSIGNAL luiDerivedControlInformation = DBCTLC_NONE;

            for(std::vector<DEFERREDCHANNELDESC*>::iterator si=pDeferredChannels->begin();
                si != pDeferredChannels->end(); 
                si++) {
                DEFERREDCHANNELDESC * pDesc = *si;
                Channel * pChannel = pDesc->pChannel;

                // first check the cache to see if we have already created a
                // descriptor block that matches the function for this channel.
                // If we have one, we can just push it into the channel and carry on. 
                mi=cache.find(pDesc->func);
                if(mi!=cache.end()) {
                    pChannel->Push(mi->second);
                    continue;
                }
                
                // we didn't have a cached block so we are going to have to allocate
                // one and fill it with the descriptor data according to the function. 
                // we'll need an AsyncContext for creating the block.
                AsyncContext * pAsyncContext = pChannel->FindAsyncContext(CE_DST, ASYNCCTXT_XFERHTOD);
                switch(pDesc->func) {
                case DF_SIZE:
                    // a size descriptor block should always have the dimensions of an integer. 
                    // using the destination channel template is not a good idea, since the block
                    // dimensions will be finalized based on it if it is present.
                    // pDerivedBlock = Datablock::CreateSizeDescriptorBlock(pAsyncContext, pBlock, pChannel->GetTemplate());                   
                    pDerivedBlock = Datablock::CreateSizeDescriptorBlock(pAsyncContext, pBlock);
                    pChannel->Push(pDerivedBlock);
                    cache[DF_SIZE] = pDerivedBlock;
                    break;
                case DF_EOF:
                    luiDerivedControlInformation = TESTSIGNAL(luiControlCode, DBCTLC_EOF);
                    pDerivedBlock = Datablock::CreateControlInformationBlock(pAsyncContext, luiDerivedControlInformation);
                    pChannel->Push(pDerivedBlock);
                    cache[DF_EOF] = pDerivedBlock;
                    break;
                case DF_BOF:
                    luiDerivedControlInformation = TESTSIGNAL(luiControlCode, DBCTLC_BOF);
                    pDerivedBlock = Datablock::CreateControlInformationBlock(pAsyncContext, luiDerivedControlInformation);
                    pChannel->Push(pDerivedBlock);
                    cache[DF_BOF] = pDerivedBlock;
                    break;
                case DF_CONTROL_CODE:
                    luiDerivedControlInformation = luiControlCode;
                    pDerivedBlock = Datablock::CreateControlInformationBlock(pAsyncContext, luiDerivedControlInformation);
                    pChannel->Push(pDerivedBlock);
                    cache[DF_CONTROL_CODE] = pDerivedBlock;
                    break;
		        case DF_BLOCK_UID:
                    pDerivedBlock = Datablock::CreateUniqueIdentifierBlock(NULL, uiDBUID);
                    pChannel->Push(pDerivedBlock);
                    cache[DF_BLOCK_UID] = pDerivedBlock;
                    break;
                case DF_DATA_DIMENSIONS:
                    pDerivedBlock = Datablock::CreateBufferDimensionsDescriptorBlock(pAsyncContext, pBlock, pChannel->GetTemplate());
                    pChannel->Push(pDerivedBlock);
                    cache[DF_DATA_DIMENSIONS] = pDerivedBlock;
                    break;
		        case DF_METADATA_SPLITTER:
                    // XXXXX: TODO: FIXME:
                    // this case is more involved. the simplest implementation just gets the host buffer for the
                    // meta-data channel of this block, and creates a new datablock calling a constructor that
                    // accepts an initial value pointer, which will cause the meta-data to be instantiated in the
                    // data channel of the new block. The AsyncContext and channel objects can be used to ensure
                    // that if a device-side view of the block can be materialized eagerly, it will be. TODO--Jon? 
                    // XXXXX: TODO: FIXME:
                    //
                    // Saving this for Jon Currey to implement.
                    assert(FALSE && "meta-data splitting unimplemented");
                    cache[DF_METADATA_SPLITTER] = pDerivedBlock;
                    break;
                default:
                    assert(false);
                    break;
                }
            }

            // if we have cached any descriptor blocks, we need to release them
            // since we are still holding references to any blocks we created.
            for(mi=cache.begin();mi!=cache.end();mi++)
                mi->second->Release();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Signal any downstream consumers of this channel that something interesting has
    ///             happened (e.g. if tasks should check whether all inputs are now available!)
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/19/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::SignalDownstreamConsumers(
        VOID
        )
    {
        if(m_pDstPort != NULL) {
            Task * pBoundTask = m_pDstPort->GetTask();
            if(pBoundTask != NULL) 
                pBoundTask->SignalPortStatusChange();
        }
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
    Channel::Push(
        Datablock* pBlock,
        DWORD dwTimeout
        ) 
    {
        recordFirstPush();
        ctlpingress(this, pBlock);
        assert(pBlock != NULL);

        // if this channel has an explicit limit on the number of
        // blocks that can flow through it, and that limit has been
        // exceeded, then there is nothing to do. Return failure.

        if(m_uiBlockTransitLimit && (m_uiBlocksDelivered >= m_uiBlockTransitLimit)) {
            return FALSE;
        }

        // check that this block passes any channel predication--no sense waiting for space if we
        // aren't going to push the block. We don't need to hold the channel lock to check the
        // predicate. 

        if(!PassesPredicate(CE_SRC, pBlock)) {

            // this block is being gated. If the failure action is to fail the operation, return an
            // appropriate success code. If the failure action is to release, no explicit action is
            // required, because what we're really doing is skipping the AddRef that would occur if the
            // Push succeeded. When the caller eventually releases the block it will get cleaned up if
            // appropriate.  
                      
            return (m_vPredicators[CE_SRC].ePredicateFailureAction != PFA_FAIL_OPERATION);
        }            

        BOOL bPushed = FALSE;
        BOOL bReadyStateTransition = FALSE;

        while(!bPushed) {

            Lock();
            BOOL bCapacity = (m_q.size() < m_uiCapacity) || m_bWantMostRecentView;
            Unlock();

            // if the channel is full, we cannot accept the datablock. 
            // block on the capacity event, making sure to catch runtime
            // and graph-level teardown events as well. 

            if(!bCapacity) {

                // note that we do not sleep holding the lock so it is safe to return
                // failure if we are awakened by any event other than the capacity event.
                
                HANDLE vWaitHandles[] = { m_hHasCapacity, m_hRuntimeTerminateEvent, m_hGraphTeardownEvent };
                DWORD dwWaitHandles = sizeof(vWaitHandles)/sizeof(HANDLE);
                DWORD dwWait = WaitForMultipleObjects(dwWaitHandles, vWaitHandles, FALSE, dwTimeout);
                switch(dwWait) {
                case WAIT_TIMEOUT:      return FALSE;     // user-configured timeout
                case WAIT_OBJECT_0 + 0: break;            // we have capacity
                case WAIT_OBJECT_0 + 1: return FALSE;     // runtime is shutting down
                case WAIT_OBJECT_0 + 2: return FALSE;     // the graph is tearing down
                default:                return FALSE;     // an error occurred.
                }
            }

            // we've been woken up because ostensibly
            // there is space in the queue. but we've
            // been woken without the lock, so no guarantees.
            Lock();

            if(m_q.size() < m_uiCapacity || m_bWantMostRecentView) {
                
                pBlock->AddRef();
                size_t uiOldQueueSize = m_q.size();
                if(m_bWantMostRecentView) {
                    while(m_q.size()) {
                        Datablock * pDrainBlock = m_q.back();
                        m_q.pop_back();
                        pDrainBlock->Release();
                    }
                }

                m_q.push_front(pBlock);
                m_bEmpty = FALSE;
                m_bAvailable = TRUE;
                ResetEvent(m_hEmpty);
                SetEvent(m_hAvailable);
                if(m_q.size() >= m_uiCapacity) {
                    ResetEvent(m_hHasCapacity);
                    m_bHasCapacity = FALSE;
                }

                // we succeeded in queueing the new block. set the pushed flag (return value)
                // and decide whether we need to signal any downstream consumers. We only want
                // to signal if this channel has made a transition from empty to non-empty
                // which impacts the ready state of the downstream port. In other words, the
                // queue size must have changed from 0 to 1. Released the lock before acting on this.
                
                m_uiMaxOccupancy = max(m_uiMaxOccupancy, (UINT)m_q.size());
                bReadyStateTransition = uiOldQueueSize == 0;
                bPushed = TRUE;
            }
            check_channel_invariants();
            Unlock();
        }

        if(bPushed && pBlock) {

            // we succeeded in queueing the new block, so make sure all the proper side-effects occur such
            // as signaling downstream objects. Note that we wait until we have released the channel lock
            // before doing this because these operations do not affect the channel state but will require
            // locking for downstream objects. 

            ConfigurePortTriggers(pBlock);
            ConfigureDownstreamIterationTargets(pBlock);
            PushDescriptorBlocks(pBlock);
            if(bReadyStateTransition) {
                SignalDownstreamConsumers();                    
            }
        }

        return bPushed;
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
    Channel::BindPort(
        Port * pPort, 
        CHANNELENDPOINTTYPE type
        ) 
    {
        Lock();
        switch(type) {
        case CE_SRC: m_pSrcPort = pPort; break;
        case CE_DST: m_pDstPort = pPort; break;
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
    Channel::UnbindPort(
        CHANNELENDPOINTTYPE type
        ) 
    {
        Port * pResult = NULL;
        Lock();
        switch(type) {
        case CE_SRC: pResult = m_pSrcPort; m_pSrcPort = NULL; break;
        case CE_DST: pResult = m_pDstPort; m_pDstPort = NULL; break;
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
    Channel::GetBoundPort(
        CHANNELENDPOINTTYPE type
        ) 
    {
        switch(type) {
        case CE_SRC: return m_pSrcPort;
        case CE_DST: return m_pDstPort;
        default: assert(false); return NULL;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the queue depth. Lock must be held. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <returns>   The queue depth. </returns>
    ///-------------------------------------------------------------------------------------------------

    size_t 
    Channel::GetQueueDepth(
        VOID
        ) 
    { 
        assert(LockIsHeld());
        return m_q.size(); 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Drains the channel be releasing this channel's references to all blocks in the
    ///             queue. The queue is cleared on exit.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Channel::Drain(
        VOID
        )
    {
        Lock();
        if(m_q.size() != 0) {
            std::deque<Datablock*>::iterator di;
            for(di=m_q.begin(); di!=m_q.end(); di++) {
                (*di)->Release();
            }
            m_q.clear();
            SetEvent(m_hEmpty);
            ResetEvent(m_hAvailable);
            SetEvent(m_hHasCapacity);
            m_bHasCapacity = TRUE;
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the channel predicator function. Lock not required because we assume this is
    ///             set at creation, rather than after the graph has entered the running state.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="eEndpoint">    To which end of the channel does the predicate apply? If CE_SRC
    ///                             is used, pushes have no effect when the predicate does not hold. If
    ///                             CE_DST is used, pulls have no effect, but the upstream producer can
    ///                             still queue data in the channel. </param>
    ///
    /// <returns>   The channel predicator for the given endpoint. </returns>
    ///-------------------------------------------------------------------------------------------------

    LPFNCHANNELPREDICATE 
    Channel::GetPredicator(
        CHANNELENDPOINTTYPE eEndpoint
        )
    {
        assert(eEndpoint - CE_SRC <= CE_DST);
        return m_vPredicators[eEndpoint].lpfnPredicate;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the canonical channel predication type. We do not require that the lock be
    ///             held for this call because the graph structure is assumed to be static. If this
    ///             changes, it will no longer be safe to call this without the channel lock.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="eEndpoint">    To which end of the channel does the predicate apply? If CE_SRC
    ///                             is used, pushes have no effect when the predicate does not hold. If
    ///                             CE_DST is used, pulls have no effect, but the upstream producer can
    ///                             still queue data in the channel. </param>
    ///
    /// <returns>   The canonical predicator type. </returns>
    ///-------------------------------------------------------------------------------------------------
    
    CHANNELPREDICATE 
    Channel::GetPredicationType(
        CHANNELENDPOINTTYPE eEndpoint
        )
    {
        return m_vPredicators[eEndpoint].eCanonicalPredicate;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a channel predicator. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="eEndpoint">    To which end of the channel does the predicate apply? If CE_SRC
    ///                             is used, pushes have no effect when the predicate does not hold. If
    ///                             CE_DST is used, pulls have no effect, but the upstream producer can
    ///                             still queue data in the channel. </param>
    /// <param name="lpfn">         A function pointer to a channel predication function. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::SetPredicator(
        CHANNELENDPOINTTYPE eEndpoint, 
        LPFNCHANNELPREDICATE lpfn
        ) 
    {
        Lock();
        m_vPredicators[eEndpoint].lpfnPredicate = lpfn;
        if(lpfn != NULL) {
            m_vPredicators[eEndpoint].eCanonicalPredicate = CGATEFN_USER_DEFINED;
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a predication type. </summary>
    ///
    /// <remarks>   Crossbac, 2/1/2012. </remarks>
    ///
    /// <param name="eEndpoint">            To which end of the channel does the predicate apply? If
    ///                                     CE_SRC is used, pushes have no effect when the predicate
    ///                                     does not hold. If CE_DST is used, pulls have no effect,
    ///                                     but the upstream producer can still queue data in the
    ///                                     channel. </param>
    /// <param name="eCanonicalPredicator"> The canonical predicator. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::SetPredicationType(
        CHANNELENDPOINTTYPE eEndpoint, 
        CHANNELPREDICATE eCanonicalPredicator
        )
    {
        Lock();
        // note that it is only meaningful to predicate an initializer at the source
        // end when there is a control propagation binding on this channel to another
        // input port bound to the same task, whose control signal needs to be propagated
        // early to have the desired effect. Other source-end bindings for initializers
        // are not meaningful. 
        assert(!(eEndpoint == CE_SRC && this->GetType() == CT_INITIALIZER) || 
            eCanonicalPredicator != CGATEFN_USER_DEFINED);
        m_vPredicators[eEndpoint].eCanonicalPredicate = eCanonicalPredicator;
        RECORDACTION(SetPredicationType, this, eEndpoint, eCanonicalPredicator);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a predication type. </summary>
    ///
    /// <remarks>   Crossbac, 2/1/2012. </remarks>
    ///
    /// <param name="eEndpoint">            To which end of the channel does the predicate apply? If
    ///                                     CE_SRC is used, pushes have no effect when the predicate
    ///                                     does not hold. If CE_DST is used, pulls have no effect,
    ///                                     but the upstream producer can still queue data in the
    ///                                     channel. </param>
    /// <param name="eCanonicalPredicator"> The canonical predicator. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::SetPredicationType(
        CHANNELENDPOINTTYPE eEndpoint, 
        CHANNELPREDICATE eCanonicalPredicator,
        PREDICATE_FAILURE_ACTION pfa
        )
    {
        Lock();
        // note that it is not usually meaningful to predicate an initializer at the source end because
        // source predicates are generally evaluated on a Pull. Consequently, it *only* meaningful to
        // predicate an initializer at the source end when there is a control propagation binding on
        // this channel to another input port bound to the same task, whose control signal needs to be
        // propagated early to have the desired effect. 
        assert(!(eEndpoint == CE_SRC && this->GetType() == CT_INITIALIZER) || 
            eCanonicalPredicator != CGATEFN_USER_DEFINED);       
        m_vPredicators[eEndpoint].eCanonicalPredicate = eCanonicalPredicator;
        m_vPredicators[eEndpoint].ePredicateFailureAction = pfa;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if the channel is predicated. We do not require that the lock be held
    ///             for this call because the graph structure is assumed to be static. If this
    ///             changes, it will no longer be safe to call this without the channel lock.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="eEndpoint">    The endpoint (src, or dst) </param>
    ///
    /// <returns>   The canonical predicator type. </returns>
    ///-------------------------------------------------------------------------------------------------
    
    BOOL 
    Channel::IsPredicated(
        CHANNELENDPOINTTYPE eEndpoint
        )
    {
        return m_vPredicators[eEndpoint].eCanonicalPredicate != CGATEFN_NONE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a view materialization policy. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="policy">   The policy. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Channel::SetViewMaterializationPolicy(
        VIEWMATERIALIZATIONPOLICY policy
        )
    {
        // For now, only support changing the policy on Graph Output type channels.
        Lock();
        if (m_type == CT_GRAPH_OUTPUT) {
            m_viewMaterializationPolicy = policy;
        } else {
            assert(false && "Setting view materialization policy is currently only supported on graph output channels");
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the view materialization policy. We do not require that the lock be
    ///             held for this call because the graph structure is assumed to be static. If this
    ///             changes, it will no longer be safe to call this without the channel lock.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <returns>   The view materialization policy. </returns>
    ///-------------------------------------------------------------------------------------------------

    VIEWMATERIALIZATIONPOLICY
    Channel::GetViewMaterializationPolicy()
    {
        return m_viewMaterializationPolicy;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if the given signal passes the predicate.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="eCanonicalPredicate">  The predicate. </param>
    /// <param name="luiSignal">            the control code. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL    
    Channel::SignalPassesCanonicalPredicate(
        CHANNELPREDICATE eCanonicalPredicate, 
        CONTROLSIGNAL luiCtlCode
        )
    {
        BOOL bPasses = TRUE;
        switch(eCanonicalPredicate) {
        case CGATEFN_NONE:                      bPasses = TRUE; break;
        case CGATEFN_DEVNULL:                   bPasses = FALSE; break;
        case CGATEFN_CLOSE_ON_EOF:              bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_EOF); break;
        case CGATEFN_OPEN_ON_EOF:               bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_EOF); break;
        case CGATEFN_CLOSE_ON_BOF:              bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_BOF); break;
        case CGATEFN_OPEN_ON_BOF:               bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_BOF); break;
        case CGATEFN_CLOSE_ON_BEGINITERATION:   bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_BEGINITERATION); break;
        case CGATEFN_OPEN_ON_BEGINITERATION:    bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_BEGINITERATION); break;
        case CGATEFN_CLOSE_ON_ENDITERATION:     bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_ENDITERATION); break;
        case CGATEFN_OPEN_ON_ENDITERATION:      bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_ENDITERATION); break;
        case CGATEFN_USER_DEFINED:              
            assert(FALSE && "user defined predicates are non-canonical! WRONG FUNCTION!");
            break;
        }
        return bPasses;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if the given block passes the predicate. The block passes if either
    ///             1) the channel predicate holds for that block, or 2) no channel predicate is in
    ///             force.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="eEndpoint">    The endpoint. </param>
    /// <param name="pBlock">       [in,out] If non-null, the block. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL                    
    Channel::PassesPredicate(
        CHANNELENDPOINTTYPE eEndpoint,
        Datablock * pBlock
        )
    {
        // common case, avoid locking the block.
        if(m_vPredicators[eEndpoint].eCanonicalPredicate == CGATEFN_NONE) 
            return TRUE;
        if(m_vPredicators[eEndpoint].eCanonicalPredicate == CGATEFN_DEVNULL) 
            return FALSE;

        pBlock->Lock();
        CONTROLSIGNAL luiCtlCode = pBlock->GetControlSignals();
        pBlock->Unlock();

        BOOL bPasses = TRUE;
        Lock();
        if(luiCtlCode == DBCTLC_NONE && m_luiPropagatedControlCode != DBCTLC_NONE) {
            // if the block isn't carrying a code, see if the propagated
            // code on the channel will pass--(we don't want to mark the block
            // and clear the code until the pull occurs).
            luiCtlCode = m_luiPropagatedControlCode;            
        }
        PREDICATE_FAILURE_ACTION pfa;
        switch(m_vPredicators[eEndpoint].eCanonicalPredicate) {
        case CGATEFN_NONE:                      bPasses = TRUE; break;
        case CGATEFN_DEVNULL:                   bPasses = FALSE; break;
        case CGATEFN_CLOSE_ON_EOF:              bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_EOF); break;
        case CGATEFN_OPEN_ON_EOF:               bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_EOF); break;
        case CGATEFN_CLOSE_ON_BOF:              bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_BOF); break;
        case CGATEFN_OPEN_ON_BOF:               bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_BOF); break;
        case CGATEFN_CLOSE_ON_BEGINITERATION:   bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_BEGINITERATION); break;
        case CGATEFN_OPEN_ON_BEGINITERATION:    bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_BEGINITERATION); break;
        case CGATEFN_CLOSE_ON_ENDITERATION:     bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_ENDITERATION); break;
        case CGATEFN_OPEN_ON_ENDITERATION:      bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_ENDITERATION); break;
        case CGATEFN_USER_DEFINED:              
            assert(m_vPredicators[eEndpoint].lpfnPredicate != NULL);
            pfa = m_vPredicators[CE_SRC].ePredicateFailureAction;
            bPasses = (m_vPredicators[eEndpoint].lpfnPredicate)(this, pBlock, pfa);
            break;
        }
        Unlock();
        return bPasses;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the type of this channel. We do not require that the lock be
    ///             held for this call: channel type is read-only.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The channel type. </returns>
    ///-------------------------------------------------------------------------------------------------

    CHANNELTYPE 
    Channel::GetType(
        VOID
        ) 
    { 
        return m_type; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the timeout for this channel, which will be used for Push/Pull calls when no
    ///             timeout parameter is specified. Default is infinite. Lock not required because
    ///             we assume this is set at creation, rather than after the graph has entered the
    ///             running state.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="dw">   The timeout. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::SetTimeout(
        DWORD dw
        ) 
    { 
        m_dwTimeout = dw; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the datablock template associated with this port. Lock not required because
    ///             we assume this is set at creation, rather than after the graph has entered the
    ///             running state.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the template. </returns>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate * 
    Channel::GetTemplate(
        VOID
        ) 
    { 
        return m_pTemplate; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Controls whether this channel will be drawn by graph rendering tools drawing the
    ///             current graph. Lock not required because draw must be called while the graph is
    ///             not running.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    ///-------------------------------------------------------------------------------------------------

    void 
    Channel::SetNoDraw(
        VOID
        ) 
    { 
        m_draw=false; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Queries if we should draw this channel when rendering the graph. Lock not
    ///             required because draw must be called while the graph is not running.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if the channel should be drawn. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool 
    Channel::ShouldDraw(
        VOID
        ) 
    { 
        return m_draw; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the user-provided channel name. Lock not required because we assume this is
    ///             set at creation, rather than after the graph has entered the running state.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the channel name. </returns>
    ///-------------------------------------------------------------------------------------------------

    char * 
    Channel::GetName(
        VOID
        ) 
    { 
        return m_lpszName; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets want most recent view. Consumers of data on this channel want only the most
    ///             recent value pushed into it. If this member is true, on a push, any previously
    ///             queued blocks will be drained and released. To be used with caution, as it can
    ///             upset the balance of blocks required to keep a pipeline from stalling: typically
    ///             channels with this property set should either be connected to ports with the
    ///             sticky property set, or should be external (exposed channels e.g. outputs)
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/14/2013. </remarks>
    ///
    /// <param name="bWantMostRecentView">  true to want most recent view. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Channel::SetWantMostRecentView(
        BOOL bWantMostRecentView
        )
    {
        Lock();
        if(m_pGraph != NULL) {
            if(m_pGraph->IsRunning()) {
                assert(FALSE);
                PTask::Runtime::Warning("attempt to change channel semantics while graph in run state. "
                                        "Request is being ignored...");
            } else {
                m_bWantMostRecentView = bWantMostRecentView;
            }
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Consumers of data on this channel want only the most recent value pushed into it.
    ///             If this member is true, on a push, any previously queued blocks will be drained
    ///             and released. To be used with caution, as it can upset the balance of blocks
    ///             required to keep a pipeline from stalling: typically channels with this property
    ///             set should either be connected to ports with the sticky property set, or should
    ///             be external (exposed channels e.g. outputs)
    ///             
    ///             Gets want most recent view.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/14/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL           
    Channel::GetWantMostRecentView(
        VOID
        ) 
    {
        assert(m_pGraph == NULL || !m_pGraph->IsRunning());
        return m_bWantMostRecentView;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an initial value for propagated control codes. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="uiCode">   The code. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Channel::SetInitialPropagatedControlSignal(
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
    Channel::SetPropagatedControlSignal(
        CONTROLSIGNAL uiCode
        ) 
    {
        Lock();
        if(m_luiPropagatedControlCode != DBCTLC_NONE && uiCode != DBCTLC_NONE) {
            if(m_vPredicators[CE_DST].eCanonicalPredicate == PTask::CGATEFN_OPEN_ON_ENDITERATION &&
                m_luiPropagatedControlCode == DBCTLC_ENDITERATION) {
                std::cout 
                    << "WARNING! " 
                    << this << ": overwriting unconsumed control code " 
                    << std::hex << m_luiPropagatedControlCode << " with " 
                    << uiCode <<  std::dec << std::endl;
            }
        }
        m_luiPropagatedControlCode = (m_luiPropagatedControlCode | uiCode);
        Task * pBoundTask = m_pDstPort->GetTask();
        if(pBoundTask != NULL) {
            pBoundTask->SignalPortStatusChange();
        }
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
    Channel::ClearPropagatedControlSignal(
        CONTROLSIGNAL luiCode
        ) 
    {
        Lock();
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
    Channel::ClearAllPropagatedControlSignals(
        VOID
        ) 
    {
        Lock();
        m_luiPropagatedControlCode = DBCTLC_NONE;
        Unlock();
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a control propagation source for this port. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pControlPropagationSourcePort">    [in] non-null, a the source port. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    Channel::SetControlPropagationSource(
        Port * pControlPropagationSourcePort
        )
    { 
        m_pControlPropagationSource = pControlPropagationSourcePort; 
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
    Channel::GetPropagatedControlSignals(
        VOID
        ) 
    { 
        return m_luiPropagatedControlCode; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the control propagation source for this port </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the control propagation source. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port*           
    Channel::GetControlPropagationSource(
        VOID
        ) 
    { 
        return m_pControlPropagationSource; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a trigger port. </summary>
    ///
    /// <remarks>   crossbac, 5/23/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the trigger port. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port * 
    Channel::GetTriggerPort(
        VOID
        )
    {
        Port * pTPort = GetBoundPort(CE_DST);
        if(pTPort != NULL && pTPort->IsTriggerPort())
            return pTPort;
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   channel toString operator. This is a debug utility. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="os">   [in,out] The output stream. </param>
    /// <param name="port"> The port. </param>
    ///
    /// <returns>   The shifted result. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::ostream& operator <<(std::ostream &os, Channel * pChannel) { 
        std::string strType;
        std::string strSrc("");
        std::string strDst("");
        std::string strName(pChannel->GetName());
        std::string strSrcTask;
        std::string strDstTask;
        Port * pSrcPort = NULL;
        Port * pDstPort = NULL;
        switch(pChannel->GetType()) {
        case CT_GRAPH_INPUT: 
            strType = "INP-CH";
            pDstPort = pChannel->GetBoundPort(CE_DST);
            strDst = pDstPort->GetVariableBinding();
            strDstTask = pDstPort->GetTask()->GetTaskName();
            strDst += ("(" + strDstTask + ")");
            break;
        case CT_GRAPH_OUTPUT: 
            strType = "OUT_CH"; 
            pSrcPort = pChannel->GetBoundPort(CE_SRC);
            strSrc = pSrcPort->GetVariableBinding();
            strSrcTask = pSrcPort->GetTask()->GetTaskName();
            strSrc += ("(" + strSrcTask + ")");
            break;
        case CT_INTERNAL: 
            strType = "INT-CH"; 
            pDstPort = pChannel->GetBoundPort(CE_DST);
            strDst = pDstPort->GetVariableBinding();
            strDstTask = pDstPort->GetTask()->GetTaskName();
            pSrcPort = pChannel->GetBoundPort(CE_SRC);
            strSrc = pSrcPort->GetVariableBinding();
            strSrcTask = pSrcPort->GetTask()->GetTaskName();
            strDst += ("(" + strDstTask + ")");
            strSrc += ("(" + strSrcTask + ")");
            break;
        case CT_MULTI:
            strType = "MULTI-CH"; 
            pDstPort = pChannel->GetBoundPort(CE_DST);
            strDst = pDstPort->GetVariableBinding();
            strDstTask = pDstPort->GetTask()->GetTaskName();
            strDst += ("(" + strDstTask + ")");
            pSrcPort = pChannel->GetBoundPort(CE_SRC);
            if(pSrcPort) {
                strSrc = pSrcPort->GetVariableBinding();
                strSrcTask = pSrcPort->GetTask()->GetTaskName();
                strSrc += ("(" + strSrcTask + ")");
            }
            break;
        case CT_INITIALIZER:
            strType = "INIT-CH"; 
            pDstPort = pChannel->GetBoundPort(CE_DST);
            strDst = pDstPort->GetVariableBinding();
            strDstTask = pDstPort->GetTask()->GetTaskName();
            strDst += ("(" + strDstTask + ")");
            break;
        }
        os << strType << "(src=" << strSrc << ", dst=" << strDst << ", name=" << strName << ")";
        return os;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check semantics. Return true if all the structures are initialized for this
    ///             channel in a way that is consistent with a well-formed graph.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="pos">      [in,out] If non-null, the position. </param>
    /// <param name="pGraph">   [in,out] non-null, the graph. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    static BOOL g_bDevNullWarningShown = FALSE;

    BOOL            
    Channel::CheckSemantics(
        std::ostream * pos,
        Graph * pGraph
        )
    {
        std::ostream& os = *pos;
        BOOL bResult = TRUE;
        Lock();

        // if there are any predicators on this 
        // channel, make sure there is a control
        // propagation source. 
        for(int i=CE_SRC; i<=CE_DST; i++) {
            if(m_vPredicators[i].eCanonicalPredicate != CGATEFN_NONE) {
                if((m_vPredicators[i].eCanonicalPredicate != CGATEFN_DEVNULL) && 
                   (m_pControlPropagationSource == NULL)) {

                    // look upstream. If there is a port attached to this channel
                    // with a control propagation source, we're good. If this port
                    // is the out part of an inout pair, we should check that as well. 
                    BOOL bFoundControlSource = FALSE;
                    if(m_pSrcPort != NULL) {
                        bFoundControlSource = (m_pSrcPort->GetControlPropagationSource() != NULL);
                        if(!bFoundControlSource) {
                            if(m_pSrcPort->GetPortType() == OUTPUT_PORT) {
                                OutputPort * pOPort = dynamic_cast<OutputPort*>(m_pSrcPort);
                                Port * pIOProducer = pOPort->GetInOutProducer();
                                if(pIOProducer != NULL) {
                                    bFoundControlSource = pIOProducer->GetControlPropagationSource() != NULL;
                                }
                            }
                        }
                    }

                    if(!bFoundControlSource) {

                        // if there is no control source, but the programmer has provided a meaningful
                        // initial value, then this can be legitimate: it represents a channel that the 
                        // programmer wants to open exactly once. If there is no initial control signal
                        // then this is 99% sure to be an error because nothing can ever cause the state
                        // of the channel to change!

                        if(!HASSIGNAL(m_luiPropagatedControlCode)) {
                            bResult = FALSE;
                            os << this 
                               << " is predicated at its "
                               << (i==CE_SRC?"source":"destination")
                               << " end, but has no control propagation source."
                               << " Consequently its predication state will never change. Is this intentional?"
                               << std::endl;
                        }
                    }
                }
            }
            if(m_vPredicators[i].eCanonicalPredicate == CGATEFN_DEVNULL &&
                m_pControlPropagationSource != NULL) {
                // bResult = FALSE;
                if(!g_bDevNullWarningShown) {
                    g_bDevNullWarningShown = TRUE;
                    os << this 
                        << " has a /dev/null predicator at its "
                        << (i==CE_SRC?"source":"destination")
                        << " end, *and* has a control propagation source." 
                        << std::endl
                        << " The control propagation cannot cause its state to change. Is this structure intentional?" 
                        << std::endl
                        << " Additional warnings of this form will be suppressed..." 
                        << std::endl;
                }
            }
        }

        if(m_bWantMostRecentView) {
            // if the "want most recent view" property is set, 
            // we want to be sure that the port bound on
            // the destination end has the sticky property set. 
            Port * pDstPort = GetBoundPort(CE_DST);
            if(pDstPort != NULL) {
                if(!pDstPort->IsSticky()) {
                    os << this 
                        << " has the WantMostRecentView property set, but the downstream port "
                        << pDstPort
                        << " does not have the sticky property set."
                        << " Is this intentional?"
                        << std::endl;
                }
            }
        }

        // Check for obvious type collisions between
        // templates on this channel and connected ports.
        // Dimension mismatch often indicates a erroneous connection.
        {
            Port * pDstPort = GetBoundPort(CE_DST);
            Port * pSrcPort = GetBoundPort(CE_SRC);
            DatablockTemplate * pDstTemplate = pDstPort ? pDstPort->GetTemplate() : NULL;
            DatablockTemplate * pSrcTemplate = pSrcPort ? pSrcPort->GetTemplate() : NULL;

            if(pDstTemplate != NULL && pSrcTemplate != NULL) {

                // if we have non-null templates at both the source and destination ends of this
                // channel, don't bother checking the template on the channel, since it won't
                // ever be used for anything even if it exists. 

                if(pDstTemplate != pSrcTemplate) {

                    BUFFERDIMENSIONS oDstDims = pDstTemplate->GetBufferDimensions();
                    BUFFERDIMENSIONS oSrcDims = pSrcTemplate->GetBufferDimensions();
                    BOOL bExactMatch = oDstDims.IsExactMatch(oSrcDims);
                    BOOL bAllocateMatch = bExactMatch || oDstDims.IsAllocationSizeMatch(oSrcDims);
                    BOOL bDstIsRecordTemplate = pDstTemplate->DescribesRecordStream();
                    BOOL bSrcIsRecordTemplate = pSrcTemplate->DescribesRecordStream();
                    BOOL bRecordMatch = bDstIsRecordTemplate == bSrcIsRecordTemplate;

                    // the case we are most interested in catching is the one where the source and destination
                    // ports have unambiguous dimensions that do not match under any applicable definition of
                    // 'match'. Since ports are allowed to have non-null templates, and even non-null templates may
                    // have the record stream flag set (meaning they describe variable-length data, and the
                    // dimensions are just a hint), it is not always the case that lack of an exact match on
                    // BUFFERDIMENSIONS data structure indicates a miswiring. However, if the templates do not
                    // describe var-length data, and do not match in allocation size, there is almost certain to be
                    // some suffering in the programmer's future, because even if this is not a miswired connection,
                    // the runtime is going to have a hard time doing the right thing wrt memory management
                    // somewhere downstream. 
                    
                    BOOL bDefinitelyOK = bExactMatch || bRecordMatch && bDstIsRecordTemplate;
                    BOOL bMaybeOK = !bDefinitelyOK && bAllocateMatch;

                    if(!bDefinitelyOK) {
                        os << this 
                            << " may connect ports with mismatched templates. "
                            << std::endl << "\t" 
                            << pSrcPort
                            << (bSrcIsRecordTemplate?"V[":"[")
                            << oSrcDims.uiXElements << "x." 
                            << oSrcDims.uiYElements << "y." 
                            << oSrcDims.uiZElements << "z." 
                            << oSrcDims.cbElementStride << "e." 
                            << oSrcDims.cbPitch << "p] <<====>>  "
                            << std::endl << "\t"
                            << pDstPort
                            << (bDstIsRecordTemplate?"V[":"[")
                            << oDstDims.uiXElements << "x." 
                            << oDstDims.uiYElements << "y." 
                            << oDstDims.uiZElements << "z." 
                            << oDstDims.cbElementStride << "e." 
                            << oDstDims.cbPitch << "p]" 
                            << std::endl;
                        if(bMaybeOK) {
                            os << "\tThe allocation byte sizes match--this may be OK."
                               << std::endl;
                        } else {
                            os << "\tIs this intentional? It is very likely to cause heartache."
                               << std::endl;
                        }
                    }

                    // don't cause fail-stop unless there is
                    // definitely a problem. if the allocation size
                    // matches, just badger the programmer, don't fail outright. 
                    bResult &= (bDefinitelyOK || bMaybeOK);
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
	Channel::GetDownstreamTasks(
		__inout  std::set<Task*>* pvTasks
		)
	{
		// if we have already computed this result,
		// then return it, since it cannot change. 
		if(m_pvDownstreamTasks != NULL) {
			pvTasks->insert(m_pvDownstreamTasks->begin(), m_pvDownstreamTasks->end());
			return static_cast<UINT>(m_pvDownstreamTasks->size());
		}

		if((!CheckInvariantCondition(m_type != CT_GRAPH_OUTPUT, "missing override for Channel::GetDownstreamTasks")) ||
			(!CheckInvariantCondition(m_type != CT_MULTI, "missing override for Channel::GetDownstreamTasks")))
			return 0;
        CHANNELENDPOINTTYPE eEndpoint = CE_DST;
        Port * pPort = GetBoundPort(eEndpoint);
		if(!CheckInvariantCondition(pPort != NULL, "null binding on channel requiring non-null CE_DST"))
			return 0;
        Task * pTask = pPort->GetTask();
		if(!CheckInvariantCondition(pTask != NULL, "null port->task binding on non-null port"))
			return 0;

		UINT nTaskCount = 0;
		m_pvDownstreamTasks = new std::set<Task*>();
		m_pvDownstreamTasks->insert(pTask);
		pvTasks->insert(pTask);

		// if the port that binds this channel to its task is not an in/out port,
		// then we are done. If it is an inout port, we need to recurse, being sure
		// to check for cycles in the graph. 
		if(pPort->IsInOutParameter()) {
			UINT i = 0;
			OutputPort * pInOutConsumer = (OutputPort*)((InputPort*)pPort)->GetInOutConsumer(); 
			for(i=0; i<pInOutConsumer->GetChannelCount(); i++) {
				Channel * pChannel = pInOutConsumer->GetChannel(i);
				nTaskCount += pChannel->GetDownstreamTasks(m_pvDownstreamTasks);
			}
			pvTasks->insert(m_pvDownstreamTasks->begin(), m_pvDownstreamTasks->end());
		}
		return static_cast<UINT>(m_pvDownstreamTasks->size());
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
	Channel::EnumerateDownstreamMemorySpaces(
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

		if((!CheckInvariantCondition(m_type != CT_GRAPH_OUTPUT, "missing override for Channel::EnumerateDownstreamMemorySpaces")) ||
			(!CheckInvariantCondition(m_type != CT_MULTI, "missing override for Channel::EnumerateDownstreamMemorySpaces")))
			return FALSE;
        CHANNELENDPOINTTYPE eEndpoint = CE_DST;
        Port * pPort = GetBoundPort(eEndpoint);
		if(!CheckInvariantCondition(pPort != NULL, "null binding on channel requiring non-null CE_DST"))
			return FALSE;
        Task * pTask = pPort->GetTask();
		if(!CheckInvariantCondition(pTask != NULL, "null port->task binding on non-null port"))
			return FALSE;

		// at this point, we have a task and a port. Get a list of all accelerators capable of running
		// this task, where each accelerator is a potential consumer of data flowing through this
		// channel if the bound port type is such that the data will have to be moved to the memory
		// space of the accelerator in order to complete the binding. Note also that we must respect
		// mandatory affinity as specified for the task if the programmer has provided it. Note that
		// sticky ports are a candidate depending on how the accelerator class binds scalars. 

		PORTTYPE ePortType = pPort->GetPortType();
		BOOL bDeviceMemorySpaceReq = (pTask->GetAcceleratorClass() != ACCELERATOR_CLASS_HOST) && 
									 ((ePortType == INPUT_PORT || ePortType == OUTPUT_PORT) ||
									  (ePortType == STICKY_PORT && pTask->GetAcceleratorClass() != ACCELERATOR_CLASS_CUDA));

		if(bDeviceMemorySpaceReq) {
			Accelerator * pMandatory = pTask->GetMandatoryAccelerator();
			if(pMandatory) {
				m_pvMandatoryDownstreamAccelerators->insert(pMandatory);
				m_pvPotentialDownstreamAccelerators->insert(pMandatory);
			} else {
				std::set<Accelerator*> vAccelerators;
				Scheduler::FindEnabledCapableAccelerators(pTask->GetAcceleratorClass(), vAccelerators);
				if(vAccelerators.size() == 1) {
					m_pvMandatoryDownstreamAccelerators->insert(*(vAccelerators.begin()));
					m_pvPotentialDownstreamAccelerators->insert(*(vAccelerators.begin()));
				} else {
					m_pvPotentialDownstreamAccelerators->insert(vAccelerators.begin(), vAccelerators.end());
				}
			}

			// if the port is an in out parameter, we need to recurse. 
			if(pPort->IsInOutParameter()) {
				UINT i = 0;
				OutputPort * pInOutConsumer = (OutputPort*)((InputPort*)pPort)->GetInOutConsumer(); 
				for(i=0; i<pInOutConsumer->GetChannelCount(); i++) {
					Channel * pChannel = pInOutConsumer->GetChannel(i);
					pChannel->EnumerateDownstreamMemorySpaces(m_pvMandatoryDownstreamAccelerators,
															  m_pvPotentialDownstreamAccelerators);
				}
			}
		}
		pvMandatoryAccelerators->insert(m_pvMandatoryDownstreamAccelerators->begin(), 
										m_pvMandatoryDownstreamAccelerators->end());
		pvPotentialAccelerators->insert(m_pvPotentialDownstreamAccelerators->begin(),
										m_pvPotentialDownstreamAccelerators->end());
		return (pvMandatoryAccelerators->size() != 0) ||
			   (pvPotentialAccelerators->size() != 0);
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
	/// <param name="pvMandatoryAccelerators">	If non-null, the mandatory accelerators. </param>
	/// <param name="pvPotentialAccelerators">	If non-null, the potential accelerators. </param>
	///
	/// <returns>	The downstream memory spaces. </returns>
	///-------------------------------------------------------------------------------------------------

	BOOL
	Channel::GetDownstreamMemorySpaces(
		__inout	 std::set<Accelerator*>** ppvMandatoryAccelerators,
		__inout  std::set<Accelerator*>** ppvPotentialAccelerators
		)
	{
		if(m_pvMandatoryDownstreamAccelerators != NULL) {
			assert(m_pvPotentialDownstreamAccelerators != NULL);
			*ppvMandatoryAccelerators = m_pvMandatoryDownstreamAccelerators;
			*ppvPotentialAccelerators = m_pvPotentialDownstreamAccelerators;
			return TRUE;
		}

		assert(m_pvMandatoryDownstreamAccelerators == NULL);
		assert(m_pvPotentialDownstreamAccelerators == NULL);
		std::set<Accelerator*>* pvMandatoryDownstreamAccelerators = new std::set<Accelerator*>();
		std::set<Accelerator*>* pvPotentialDownstreamAccelerators = new std::set<Accelerator*>();
		BOOL bRes = EnumerateDownstreamMemorySpaces(pvMandatoryDownstreamAccelerators, pvPotentialDownstreamAccelerators);
		delete pvMandatoryDownstreamAccelerators;
		delete pvPotentialDownstreamAccelerators;
		*ppvMandatoryAccelerators = m_pvMandatoryDownstreamAccelerators;
		*ppvPotentialAccelerators = m_pvPotentialDownstreamAccelerators;
		return bRes;
	} 

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the unambigous downstream memory space if there is one. </summary>
    ///
    /// <remarks>   crossbac, 7/8/2013. </remarks>
    ///
    /// <returns>   null if the downstream memory space for any blocks pushed into this
    ///             channel cannot be determined unambiguously at the time of the call. 
    ///             If such can be determined, return the accelerator object associated with
    ///             that memory space. 
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Channel::GetUnambigousDownstreamMemorySpace(
        VOID
        )
    {
        std::set<Accelerator*>* ppvMandatoryAccelerators = NULL;
		std::set<Accelerator*>* ppvPotentialAccelerators = NULL;
        GetDownstreamMemorySpaces(&ppvMandatoryAccelerators, &ppvPotentialAccelerators);
        if(ppvMandatoryAccelerators && ppvMandatoryAccelerators->size() == 1)
            return *(ppvMandatoryAccelerators->begin());
        if(ppvPotentialAccelerators && ppvPotentialAccelerators->size() == 1)
            return *(ppvPotentialAccelerators->begin());        
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets cumulative block transit. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///
    /// <returns>   The cumulative block transit. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Channel::GetCumulativeBlockTransit(
        VOID
        )
    {
        return m_uiBlocksDelivered;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets maximum occupancy. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///
    /// <returns>   The maximum occupancy. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Channel::GetMaxOccupancy(
        VOID
        )
    {
        return m_uiMaxOccupancy;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets cumulative occupancy. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///
    /// <returns>   The cumulative occupancy. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Channel::GetCumulativeOccupancy(
        VOID
        )
    {
        return m_uiMaxOccupancy;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets occupancy samples. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///
    /// <returns>   The occupancy samples. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Channel::GetOccupancySamples(
        VOID
        )
    {
        return m_uiBlocksDelivered;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is pool owner. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///
    /// <returns>   true if pool owner, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Channel::IsPoolOwner(
        VOID
        )
    {
        return m_bHasBlockPool;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the graph. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the graph. </returns>
    ///-------------------------------------------------------------------------------------------------

    Graph * 
    Channel::GetGraph(
        VOID
        )
    {
        return m_pGraph;
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
    Channel::FindMaximalDownstreamCapacity(
        __inout std::set<Task*>& vTasksVisited,
        __inout std::vector<Channel*>& vPath
        )
    {
        vPath.push_back(this);
        UINT uiDownstreamPortCapacity = 0;
        Port * pDstPort = GetBoundPort(CE_DST);
        if(pDstPort != NULL) {
            uiDownstreamPortCapacity += pDstPort->FindMaximalDownstreamCapacity(vTasksVisited, vPath);
        }
        return uiDownstreamPortCapacity + m_uiCapacity;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this channel has any non trivial predicates. </summary>
    ///
    /// <remarks>   crossbac, 7/3/2014. </remarks>
    ///
    /// <returns>   true if non trivial predicate, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Channel::HasNonTrivialPredicate(
        VOID
        )
    {
        CHANNELPREDICATE eSrcPredicate = GetPredicationType(CE_SRC);
        CHANNELPREDICATE eDstPredicate = GetPredicationType(CE_DST);
        BOOL bSrcTrivial = eSrcPredicate == CGATEFN_NONE || eSrcPredicate == CGATEFN_DEVNULL;
        BOOL bDstTrivial = eDstPredicate == CGATEFN_NONE || eDstPredicate == CGATEFN_DEVNULL;
        return !(bSrcTrivial && bDstTrivial);
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
    Channel::GetControlSignalsOfInterest(
        VOID
        )
    {
        CONTROLSIGNAL luiSignals = DBCTLC_NONE;
        std::vector<Channel*>::iterator ci;
        CHANNELPREDICATE eSrcPredicate = GetPredicationType(CE_SRC);
        CHANNELPREDICATE eDstPredicate = GetPredicationType(CE_DST);
        luiSignals |= GetControlSignalsOfInterest(eSrcPredicate);
        luiSignals |= GetControlSignalsOfInterest(eDstPredicate);
        luiSignals |= m_luiInitialPropagatedControlCode;
        return luiSignals;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets control signals of interest. </summary>
    ///
    /// <remarks>   crossbac, 7/7/2014. </remarks>
    ///
    /// <param name="ePredicate">   The predicate. </param>
    ///
    /// <returns>   The control signals of interest. </returns>
    ///-------------------------------------------------------------------------------------------------

    CONTROLSIGNAL
    Channel::GetControlSignalsOfInterest(
        __in CHANNELPREDICATE ePredicate
        )
    {
        switch(ePredicate) {
        case CGATEFN_NONE:                     return DBCTLC_NONE;
        case CGATEFN_CLOSE_ON_EOF:             return DBCTLC_EOF;
        case CGATEFN_OPEN_ON_EOF:              return DBCTLC_EOF;
        case CGATEFN_OPEN_ON_BEGINITERATION:   return DBCTLC_BEGINITERATION;
        case CGATEFN_CLOSE_ON_BEGINITERATION:  return DBCTLC_BEGINITERATION;
        case CGATEFN_OPEN_ON_ENDITERATION:     return DBCTLC_ENDITERATION;
        case CGATEFN_CLOSE_ON_ENDITERATION:    return DBCTLC_ENDITERATION;
        case CGATEFN_DEVNULL:                  return DBCTLC_NONE;
        case CGATEFN_CLOSE_ON_BOF:             return DBCTLC_BOF;
        case CGATEFN_OPEN_ON_BOF:              return DBCTLC_BOF;
        case CGATEFN_USER_DEFINED:             assert(FALSE); return DBCTLC_NONE;
        default:                               assert(FALSE); return DBCTLC_NONE;
        }
    }

};
