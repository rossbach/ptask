///-------------------------------------------------------------------------------------------------
// file:	HostAsyncContext.cpp
//
// summary:	Implements the host asynchronous context class
///-------------------------------------------------------------------------------------------------

#include "primitive_types.h"
#include <iostream>
#include <assert.h>
#include "accelerator.h"
#include "task.h"
#include "HostAsyncContext.h"
#include "SyncPoint.h"
using namespace std;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 6/18/2012. </remarks>
    ///
    /// <param name="pDeviceContext">       [in] non-null, context for the device. </param>
    /// <param name="pTaskContext">         [in] non-null, context for the task. </param>
    /// <param name="eAsyncContextType">    Type of the asynchronous context. </param>
    ///-------------------------------------------------------------------------------------------------

    HostAsyncContext::HostAsyncContext(
        __in Accelerator * pDeviceContext,
        __in Task * pTaskContext,
        __in ASYNCCONTEXTTYPE eAsyncContextType
        ) : AsyncContext(pDeviceContext, 
                         pTaskContext,
                         eAsyncContextType) {}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    HostAsyncContext::~HostAsyncContext() {
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes this object. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostAsyncContext::Initialize(
        VOID
        )
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Platform specific create synchronization point. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    SyncPoint *
    HostAsyncContext::PlatformSpecificCreateSyncPoint(
        void * pPSSyncObject
        )
    {
        assert(FALSE);
        UNREFERENCED_PARAMETER(pPSSyncObject);
        SyncPoint * pSyncPoint = NULL;
        return pSyncPoint;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Platform specific destroy synchronization point. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    HostAsyncContext::PlatformSpecificDestroySynchronizationPoint(
        __in SyncPoint * pSyncPoint
        )
    {
        assert(FALSE);
        assert(pSyncPoint != NULL);
        UNREFERENCED_PARAMETER(pSyncPoint);
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Wait for dependence. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
    /// <param name="pDependentContext">    [in,out] If non-null, context for the dependent. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostAsyncContext::PlatformSpecificInsertFence(
        __in SyncPoint * pSyncPoint
        )
    {
        // we don't have a stream for host commands, so 
        // waiting for this sync point means finding the 
        // async context of the sync point and forcing the
        // async context to drain its stream.
        assert(FALSE && "HostAsyncContext::PlatformSpecificInsertFence called...converting synchronous wait!\n");
        return PlatformSpecificSynchronousWait(pSyncPoint);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Wait for dependence. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
    /// <param name="pDependentContext">    [in,out] If non-null, context for the dependent. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostAsyncContext::PlatformSpecificSynchronousWait(
        __in SyncPoint * pSyncPoint
        )
    {
        // we don't have a stream for host commands, so 
        // waiting for this sync point means finding the 
        // async context of the sync point and forcing the
        // async context to drain its stream.
        assert(FALSE && "HostAsyncContext::PlatformSpecificSynchronousWait called!\n");
        AsyncContext * pWaitedForContext = pSyncPoint->GetAsyncContext();
        if(pWaitedForContext->GetDeviceContext()->GetClass() != ACCELERATOR_CLASS_HOST) {
            return pWaitedForContext->SynchronousWait(pSyncPoint);
        } else {
            return TRUE;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   platform specific synchronize context. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostAsyncContext::PlatformSpecificSynchronizeContext(
        VOID
        )
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the platform context object. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the platform context object. </returns>
    ///-------------------------------------------------------------------------------------------------

    void *
    HostAsyncContext::GetPlatformContextObject(
        VOID
        )
    {
        return (void*) NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Wait for dependence. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostAsyncContext::PlatformSpecificQueryOutstanding(
        __in SyncPoint * pSyncPoint
        )
    {
        assert(FALSE);
        UNREFERENCED_PARAMETER(pSyncPoint);
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Platform specific nonblocking check whether the event remains outstanding. </summary>
    ///
    /// <remarks>   crossbac, 7/2/2013. </remarks>
    ///
    /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    HostAsyncContext::PlatformSpecificNonblockingQueryOutstanding(
        __inout SyncPoint * pSyncPoint
        )
    {
        assert(FALSE);
        UNREFERENCED_PARAMETER(pSyncPoint);
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Wait for dependence synchronously without locking the async context
    ///             or underlying accelerator: this simplifies lock acquisition for such
    ///             waits, but at the expense of leaving live dependences that are
    ///             actually resolved.  </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostAsyncContext::PlatformSpecificLocklessSynchronousWait(
        __in SyncPoint * pSyncPoint 
        )
    {
        assert(FALSE);
        UNREFERENCED_PARAMETER(pSyncPoint);
        return FALSE;
    }

};
