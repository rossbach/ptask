///-------------------------------------------------------------------------------------------------
// file:	CLAsyncContext.cpp
//
// summary:	Implements the OpenCL asynchronous context class
///-------------------------------------------------------------------------------------------------

#ifdef OPENCL_SUPPORT

#include "primitive_types.h"
#include "ptaskutils.h"
#include "PTaskRuntime.h"
#include <iostream>
#include "CLAsyncContext.h"
#include "CLTask.h"
#include "datablock.h"
#include "PCUBuffer.h"
#include "InputPort.h"
#include "OutputPort.h"
#include "InitializerPort.h"
#include "MetaPort.h"
#include "Scheduler.h"
#include "SyncPoint.h"
#include <vector>
#include <assert.h>
using namespace std;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. 
    ///             
    ///             FIXME: TODO:
    ///             -------------------
    ///             OpenCL supports events and command queues such that we can implement fine grain
    ///             dependences exactly as they are implemented for the cuda backend. Currently there
    ///             just isn't enough demand for the OpenCL backend to justify prioritizing that
    ///             development effort. Hence, all OpenCL calls are currently synchronous, and the
    ///             platform-specific work of managing dependences and waiting for them to resove can
    ///             be completely elided.
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pDeviceContext">   [in,out] If non-null, context for the device. </param>
    /// <param name="pTaskContext">     [in,out] If non-null, context for the task. </param>
    ///-------------------------------------------------------------------------------------------------

    CLAsyncContext::CLAsyncContext(
        __in Accelerator * pDeviceContext,
        __in Task * pTaskContext,
        __in ASYNCCONTEXTTYPE eAsyncContextType
        ) : AsyncContext(pDeviceContext, 
                         pTaskContext, 
                         eAsyncContextType) { }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    CLAsyncContext::~CLAsyncContext() {
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes this object. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CLAsyncContext::Initialize(
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
    CLAsyncContext::PlatformSpecificCreateSyncPoint(
        void * pPSSyncObject
        )
    {
        SyncPoint * pSyncPoint = NULL;

        m_pDeviceContext->Lock();
        m_pDeviceContext->MakeDeviceContextCurrent();
        Lock();

        pSyncPoint = new SyncPoint(this,
                                   m_pDeviceContext->GetContext(),
                                   (void*) NULL,
                                   pPSSyncObject);

        Unlock();
        m_pDeviceContext->ReleaseCurrentDeviceContext();
        m_pDeviceContext->Unlock();

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
    CLAsyncContext::PlatformSpecificDestroySynchronizationPoint(
        __in SyncPoint * pSyncPoint
        )
    {
        // nothing to do here. We let DirectX
        // manage asynchrony currently. 
        UNREFERENCED_PARAMETER(pSyncPoint);
        assert(pSyncPoint != NULL);
        return TRUE;
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
    CLAsyncContext::PlatformSpecificInsertFence(
        __in SyncPoint * pSyncPoint
        )
    {
        BOOL bSuccess = TRUE;
        // cjr: 6/18/12: FIXME: TODO
        // --------------------------
        // we really shouldn't need to do anything, because we are using synchronous CL APIs everywhere
        // we use OpenCL at the moment. This needs to be addressed. OpenCL supports events and command
        // queues such that we can implement fine grain dependences exactly as they are implemented for
        // the cuda backend. Currently there just isn't enough demand for the OpenCL backend to justify
        // prioritizing that development effort. Hence, all OpenCL calls are currently synchronous, and
        // the platform-specific work of managing dependences and waiting for them to resove can. 
        assert(false && "FIXME: cross-platform async wait needs to be fixed in CLAsyncContext!");
        UNREFERENCED_PARAMETER(pSyncPoint);
        return bSuccess;
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
    CLAsyncContext::PlatformSpecificSynchronousWait(
        __in SyncPoint * pSyncPoint
        )
    {
        BOOL bSuccess = TRUE;
        // cjr: 6/18/12: FIXME: TODO
        // --------------------------
        // we really shouldn't need to do anything, because we are using synchronous CL APIs everywhere
        // we use OpenCL at the moment. This needs to be addressed. OpenCL supports events and command
        // queues such that we can implement fine grain dependences exactly as they are implemented for
        // the cuda backend. Currently there just isn't enough demand for the OpenCL backend to justify
        // prioritizing that development effort. Hence, all OpenCL calls are currently synchronous, and
        // the platform-specific work of managing dependences and waiting for them to resove can. 
        assert(false && "FIXME: cross-platform async wait needs to be fixed in CLAsyncContext!");
        UNREFERENCED_PARAMETER(pSyncPoint);
        return bSuccess;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the platform context object. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the platform context object. </returns>
    ///-------------------------------------------------------------------------------------------------

    void *
    CLAsyncContext::GetPlatformContextObject(
        VOID
        )
    {
        return (void*) m_pDeviceContext->GetContext();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   platform specific synchronize context. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CLAsyncContext::PlatformSpecificSynchronizeContext(
        VOID
        )
    {
        return TRUE;
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
    CLAsyncContext::PlatformSpecificQueryOutstanding(
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
    CLAsyncContext::PlatformSpecificNonblockingQueryOutstanding(
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
    CLAsyncContext::PlatformSpecificLocklessSynchronousWait(
        __in SyncPoint * pSyncPoint 
        )
    {
        assert(FALSE);
        UNREFERENCED_PARAMETER(pSyncPoint);
        return FALSE;
    }

};

#endif