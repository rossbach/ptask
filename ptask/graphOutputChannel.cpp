//--------------------------------------------------------------------------------------
// File: GraphOutputChannel.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "channel.h"
#include "datablock.h"
#include "GraphOutputChannel.h"
#include "port.h"
#include "ptaskutils.h"

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 2/6/2012. </remarks>
    ///
    /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
    /// <param name="hTerminate">       Handle of the terminate. </param>
    /// <param name="hStop">            Handle of the stop. </param>
    /// <param name="lpszChannelName">  [in,out] (optional)  If non-null, name of the channel. </param>
    ///-------------------------------------------------------------------------------------------------

    GraphOutputChannel::GraphOutputChannel(
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
        m_type = CT_GRAPH_OUTPUT;
        m_viewMaterializationPolicy = PTask::Runtime::GetDefaultOutputViewMaterializationPolicy();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 2/6/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    GraphOutputChannel::~GraphOutputChannel() {
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
    GraphOutputChannel::CanStream(
        VOID
        )
    {
        Lock();
        // an output channel can stream if
        // its upstream port can stream
        BOOL bResult = FALSE;
        if(m_pDstPort != NULL) {
            bResult = m_pSrcPort->CanStream();
        }
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
	GraphOutputChannel::GetDownstreamTasks(
		__inout  std::set<Task*>* pvTasks
		)
	{
		UNREFERENCED_PARAMETER(pvTasks);
		return 0;
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
	GraphOutputChannel::EnumerateDownstreamMemorySpaces(
		__inout	 std::set<Accelerator*>* pvMandatoryAccelerators,
		__inout  std::set<Accelerator*>* pvPotentialAccelerators
		)
	{
		UNREFERENCED_PARAMETER(pvMandatoryAccelerators);
		UNREFERENCED_PARAMETER(pvPotentialAccelerators);
		return TRUE;
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
    GraphOutputChannel::HasDownstreamWriters(
        VOID
        )
    {
        // generally, must assume an exposed channel 
        // must be written. There is one exception: if the channel has a 
        // devnull predicate, it can never produce a block. 
        return m_vPredicators[CE_SRC].eCanonicalPredicate != CGATEFN_DEVNULL &&
               m_vPredicators[CE_DST].eCanonicalPredicate != CGATEFN_DEVNULL;
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
    GraphOutputChannel::CheckTypeSpecificSemantics(
        std::ostream * pos,
        PTask::Graph * pGraph
        ) 
    {
        UNREFERENCED_PARAMETER(pGraph);
        BOOL bResult = TRUE;
        std::ostream& os = *pos;
        if(m_pSrcPort == NULL) {
            bResult = FALSE;
            os << this << "should be bound to a non-null source port!" << std::endl;
        }        
        if(m_pDstPort != NULL) {
            bResult = FALSE;
            os << this 
                << "should NOT be bound to a any destination port, but is bound to " 
                << m_pDstPort
                << "!"
                << std::endl;
        }
        return bResult;
    }
};
