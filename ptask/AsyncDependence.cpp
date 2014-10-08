///-------------------------------------------------------------------------------------------------
// file:	AsyncDependence.cpp
//
// summary:	Implements the asynchronous dependence class
///-------------------------------------------------------------------------------------------------

#include <stdio.h>
#include <iostream>
#include <crtdbg.h>
#include <assert.h>
#include "accelerator.h"
#include "Task.h"
#include "AsyncContext.h"
#include "AsyncDependence.h"
#include "SyncPoint.h"
#include "nvtxmacros.h"
#include <deque>
using namespace std;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the outstanding
    ///                                 asynchronous operations. </param>
    /// <param name="pSyncPoint">       [in,out] If non-null, the sync point on which to depend. </param>
    ///-------------------------------------------------------------------------------------------------

	AsyncDependence::AsyncDependence(
        __in AsyncContext * pAsyncContext,
        __in SyncPoint * pSyncPoint,
        __in ASYNCHRONOUS_OPTYPE eOperationType
        ) : ReferenceCounted()
    {
        m_pAsyncContext = pAsyncContext;
        m_pAsyncContext->AddRef();
        m_pSyncPoint = pSyncPoint;
        m_pSyncPoint->AddRef();
        m_eOperationType = eOperationType;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

	AsyncDependence::~AsyncDependence() {
        Lock();
        if(m_pSyncPoint != NULL) {            
            m_pSyncPoint->Release();
            m_pSyncPoint = NULL;
        }
        m_pAsyncContext->Release();
        m_pAsyncContext = NULL;
        Unlock();
    }        


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the context. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the context. </returns>
    ///-------------------------------------------------------------------------------------------------

    AsyncContext * 
    AsyncDependence::GetContext(
        VOID
        )
    {
        // no lock required: once this field is
        // is written, it should never change!
        return m_pAsyncContext;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the platform context object. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the platform context object. </returns>
    ///-------------------------------------------------------------------------------------------------

    void * 
    AsyncDependence::GetPlatformContextObject(
        void
        )
    {
        // no lock required: once this field of the contained syncpoint object is written, it should
        // never change. The syncpoint itself can change state, so checking state or waiting requires a
        // lock. But getting the context object should return the same value regardless of the
        // syncpoint state. 
        return m_pSyncPoint->GetPlatformContextObject();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the platform wait object. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the platform wait object. </returns>
    ///-------------------------------------------------------------------------------------------------

    void * 
    AsyncDependence::GetPlatformWaitObject(
        void
        )
    {
        // no lock required: once this field of the contained syncpoint object is written, it should
        // never change. The syncpoint itself can change state, so checking state or waiting requires a
        // lock. But getting the wait object itself should return the same value regardless of the
        // syncpoint state. 
        return m_pSyncPoint->GetPlatformWaitObject();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the synchronise point. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the synchronise point. </returns>
    ///-------------------------------------------------------------------------------------------------

    SyncPoint * 
    AsyncDependence::GetSyncPoint(
        VOID
        )
    {
        return m_pSyncPoint;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets operation type. </summary>
    ///
    /// <remarks>   crossbac, 5/1/2013. </remarks>
    ///
    /// <returns>   The operation type. </returns>
    ///-------------------------------------------------------------------------------------------------

    ASYNCHRONOUS_OPTYPE
    AsyncDependence::GetOperationType(
        VOID
        )
    {
        return m_eOperationType;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is outstanding. </summary>
    ///
    /// <remarks>   crossbac, 6/25/2013. </remarks>
    ///
    /// <returns>   true if outstanding, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    AsyncDependence::IsOutstanding(
        VOID
        )
    {
        BOOL bOutstanding = FALSE;
        assert(LockIsHeld());
        Lock();  // for safety sake
        m_pSyncPoint->Lock();
        bOutstanding = m_pSyncPoint->QueryOutstanding(m_pAsyncContext);
        m_pSyncPoint->Unlock();
        Unlock();
        return bOutstanding;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Blocking wait complete. Acquires locks--not to be called from
    ///             dispatch context!</summary>
    ///
    /// <remarks>   crossbac, 6/25/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    AsyncDependence::SynchronousExclusiveWait(
        VOID
        )
    {
        return AsyncContext::SynchronousWait(this);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Lockless wait outstanding: without acquiring any locks attempt to perform a
    ///             synchronous wait for any outstanding async dependences on this buffer that
    ///             conflict with an operation of the given type. This is an experimental API,
    ///             enable/disable with PTask::Runtime::*etTaskDispatchLocklessIncomingDepWait(),
    ///             attempting to leverage the fact that CUDA apis for waiting on events (which
    ///             appear to be thread-safe and decoupled from a particular device context)
    ///             to minimize serialization associated with outstanding dependences on data
    ///             consumed by tasks that do not require accelerators for any other reason than to
    ///             wait for such operations to complete.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    AsyncDependence::LocklessWaitOutstanding(
        VOID
        )
    {
        // lockless doesn't mean *NO* locks. It means no accelerator or context
        // locks. We still lock the dependence and sync points because we are going
        // to change their state if the wait succeeds.

        Lock();
        BOOL bWaitSuccess = TRUE;
        AsyncContext * pDepContext = m_pSyncPoint->GetAsyncContext();
        assert(pDepContext != NULL);
        assert(pDepContext->SupportsExplicitAsyncOperations());
        if(pDepContext != NULL && pDepContext->SupportsExplicitAsyncOperations()) {
            m_pSyncPoint->Lock();
            if(m_pSyncPoint->NonblockingQueryOutstanding()) {
                MARKRANGEENTER(L"LocklessWaitOutstanding--REALWAIT");
                bWaitSuccess = pDepContext->LocklessWaitOutstanding(this);
                MARKRANGEEXIT();
            }
            m_pSyncPoint->Unlock();
        }
        Unlock();
        return bWaitSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Determines if the dependence is outstanding without acquiring device
    ///             and context locks required to react to resolution if we detect it. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/2/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    AsyncDependence::NonblockingQueryOutstanding(
        VOID
        )
    {
        BOOL bOutstanding = FALSE;
        if(m_pSyncPoint->QueryOutstandingFlag()) {
            Lock();
            m_pSyncPoint->Lock();
            bOutstanding = m_pSyncPoint->NonblockingQueryOutstanding();
            m_pSyncPoint->Unlock();
            Unlock();
        }
        return bOutstanding;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Determines if the sync point this dependence encapsulates has been
    ///             marked resolved or not. The transition from outstanding to resolved
    ///             is monotonic, so we can make this check without a lock, provided
    ///             that only a FALSE return value is considered actionable.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/2/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    AsyncDependence::QueryOutstandingFlag(
        VOID
        )
    {
        return m_pSyncPoint->QueryOutstandingFlag();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Blocking wait complete. Acquires locks--not to be called from
    ///             dispatch context!</summary>
    ///
    /// <remarks>   crossbac, 6/25/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    AsyncDependence::__SynchronousWaitLocksHeld(
        VOID
        )
    {
        if(!m_pSyncPoint->QueryOutstandingFlag())
            return TRUE;
        assert(m_pAsyncContext->LockIsHeld());
        assert(m_pAsyncContext->GetDeviceContext()->LockIsHeld());
        assert(m_pAsyncContext->SupportsExplicitAsyncOperations());
        assert(m_pAsyncContext == m_pSyncPoint->GetAsyncContext());
        Lock();
        BOOL bSuccess = m_pAsyncContext->__SynchronousWaitLocksHeld(m_pSyncPoint);
        Unlock();
        return bSuccess;
    }

};
