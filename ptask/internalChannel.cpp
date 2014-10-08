//--------------------------------------------------------------------------------------
// File: InternalChannel.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "ptaskutils.h"
#include "task.h"
#include "datablock.h"
#include "port.h"
#include "datablocktemplate.h"
#include "internalchannel.h"

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 2/6/2012. </remarks>
    ///
    /// <param name="pTemplate">                [in,out] If non-null, the template. </param>
    /// <param name="hRuntimeTerminateEvent">   Handle of the terminate. </param>
    /// <param name="hGraphTeardownEvent">      Handle of the stop. </param>
    /// <param name="hGraphStopEvent">          The graph stop event. </param>
    /// <param name="lpszChannelName">          [in,out] (optional)  If non-null, name of the
    ///                                         channel. </param>
    /// <param name="bHasBlockPool">            The has block pool. </param>
    ///-------------------------------------------------------------------------------------------------

    InternalChannel::InternalChannel(
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
        m_type = CT_INTERNAL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 2/6/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    InternalChannel::~InternalChannel() {
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
    InternalChannel::CanStream(
        VOID
        )
    {
        Lock();
        // an internal channel can stream if
        // its downstream port can stream. 
        // the upstream port doesn't really matter
        BOOL bResult = FALSE;
        if(m_pDstPort != NULL) {
            bResult = m_pSrcPort->CanStream();
        }
        Unlock();
        return bResult;
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
    InternalChannel::HasDownstreamWriters(
        VOID
        )
    {
        assert(m_pDstPort != NULL);
        if(m_pDstPort == NULL) return FALSE;
        return (m_pDstPort->IsInOutParameter() || m_pDstPort->IsDestructive());
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
    InternalChannel::CheckTypeSpecificSemantics(
        std::ostream * pos,
        PTask::Graph * pGraph
        ) 
    {
        UNREFERENCED_PARAMETER(pGraph);
        BOOL bResult = TRUE;
        std::ostream& os = *pos;
        if(m_pDstPort == NULL) {
            bResult = FALSE;
            os << this << "should be bound to a non-null destination port!" << std::endl;
        }
        if(m_pSrcPort == NULL) {
            bResult = FALSE;
            os << this << "should be bound to a non-null src port!" << std::endl;
        }
        return bResult;
    }
};
