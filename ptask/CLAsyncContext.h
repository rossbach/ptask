///-------------------------------------------------------------------------------------------------
// file:	CLAsyncContext.h
//
// summary:	Declares the OpenCL asynchronous context class
///-------------------------------------------------------------------------------------------------

#ifndef __CL_ASYNC_CONTEXT_H__
#define __CL_ASYNC_CONTEXT_H__
#ifdef OPENCL_SUPPORT

#include "primitive_types.h"
#include "accelerator.h"
#include "claccelerator.h"
#include "task.h"
#include "channel.h"
#include "hrperft.h"
#include "AsyncContext.h"
#include "AsyncDependence.h"
#include <map>
#include <vector>
#include <list>

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   OpenCL asynchronous context. </summary>
    ///
    /// <remarks>   crossbac, 6/18/2012.
    ///             
    ///             FIXME: TODO:
    ///             -------------------
    ///             OpenCL supports events and command queues such that we can implement fine grain
    ///             dependences exactly as they are implemented for the cuda backend. Currently there
    ///             just isn't enough demand for the OpenCL backend to justify prioritizing that
    ///             development effort. Hence, all OpenCL calls are currently synchronous, and the
    ///             platform-specific work of managing dependences and waiting for them to resove can
    ///             be completely elided.
    ///             
    ///             </remarks>
    ///-------------------------------------------------------------------------------------------------

    class CLAsyncContext : public AsyncContext {
    public: 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pDeviceContext">       [in,out] If non-null, context for the device. </param>
        /// <param name="pTaskContext">         [in,out] If non-null, context for the task. </param>
        /// <param name="eAsyncContextType">    Type of the asynchronous context. </param>
        ///-------------------------------------------------------------------------------------------------

        CLAsyncContext(
            __in Accelerator * pDeviceContext,
            __in Task * pTaskContext,
            __in ASYNCCONTEXTTYPE eAsyncContextType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~CLAsyncContext();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Initializes this object. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL Initialize();

    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Platform specific create synchronization point. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   null if it fails, else. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual SyncPoint *
        PlatformSpecificCreateSyncPoint(
            void * pPSSyncObject
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Platform specific destroy synchronization point. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL
        PlatformSpecificDestroySynchronizationPoint(
            __in SyncPoint * pSyncPoint
            );        

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Determines if we can platform specific synchronize context. </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        PlatformSpecificSynchronizeContext(
            VOID
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Wait for dependence asynchronously. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        PlatformSpecificInsertFence(
            __in SyncPoint * pSyncPoint 
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Wait for dependence synchronously. </summary>
        ///
        /// <remarks>   Crossbac, 5/24/2012. </remarks>
        ///
        /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        PlatformSpecificSynchronousWait(
            __in SyncPoint * pSyncPoint 
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Determines if the sync point is resolved (and marks it if so). </summary>
        ///
        /// <remarks>   crossbac, 4/29/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL 
        PlatformSpecificQueryOutstanding(
            __inout SyncPoint * pSyncPoint
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Platform specific nonblocking check whether the event remains outstanding. </summary>
        ///
        /// <remarks>   crossbac, 7/2/2013. </remarks>
        ///
        /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL
        PlatformSpecificNonblockingQueryOutstanding(
            __inout SyncPoint * pSyncPoint
            );

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

        virtual BOOL 
        PlatformSpecificLocklessSynchronousWait(
            __in SyncPoint * pSyncPoint 
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the platform context object. </summary>
        ///
        /// <remarks>   Crossbac, 5/25/2012. </remarks>
        ///
        /// <returns>   null if it fails, else the platform context object. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual void *
        GetPlatformContextObject();

    };

};
#endif
#endif