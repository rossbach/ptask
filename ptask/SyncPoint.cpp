///-------------------------------------------------------------------------------------------------
// file:	SyncPoint.cpp
//
// summary:	Implements the synchronise point class
///-------------------------------------------------------------------------------------------------

#include "primitive_types.h"
#include <iostream>
#include <assert.h>
#include "SyncPoint.h"
#include "AsyncContext.h"
#include "accelerator.h"

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="_pAsyncContext">               [in] If non-null, context for the asynchronous. </param>
    /// <param name="_pPlatformAsyncContextObject"> [in] non-null, the platform-specific asynchronous
    ///                                             context object. E.g. the stream in CUDA, the
    ///                                             ID3D11ImmediateContext object in DirectX and so
    ///                                             on. </param>
    /// <param name="_pPlatformAsyncWaitObject">    [in] non-null, a platform-specific asynchronous
    ///                                             wait object. E.g. a windows event or a cuda event
    ///                                             object, etc. </param>
    ///-------------------------------------------------------------------------------------------------

    SyncPoint::SyncPoint(
        __in AsyncContext *  _pAsyncContext,
        __in void *          _pPlatformAsyncContextObject,
        __in void *          _pPlatformAsyncWaitObject,
        __in void *          _pPlatformParentSyncObject   
        ) : ReferenceCounted()
    {
        m_bOutstanding = TRUE;
        m_bContextSynchronized = FALSE;
        m_bStatusQueried = FALSE;
        m_pPlatformAsyncContextObject = _pPlatformAsyncContextObject;
        m_pPlatformAsyncWaitObject = _pPlatformAsyncWaitObject;
        m_pPlatformParentSyncObject = _pPlatformParentSyncObject;
        m_pAsyncContext = _pAsyncContext;
        m_pAsyncContext->AddRef();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    SyncPoint::~SyncPoint(
        void
        ) 
    {
        Lock();
        assert(m_uiRefCount == 0);
        assert(!m_bOutstanding || !NonblockingQueryOutstanding());
        Unlock();
        m_pAsyncContext->DestroySyncPoint(this);
        m_pAsyncContext->Release();
        m_pAsyncContext = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Release by decrementing the refcount. We override the implementation inherited
    ///             from ReferenceCounted so that we can figure out if the outstanding list
    ///             for the containing async context can be garbage collected. If the refcount
    ///             goes from 2 to 1, that *should* mean that its async context holds the only
    ///             reference, and therefor we can retire it. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    //LONG
    //SyncPoint::Release() {
    //    LONG privateCount = 0;
    //    Lock();
    //    if(m_uiRefCount == 1) {
    //        privateCount = 0;
    //        m_uiRefCount = 0;
    //    } else {
    //        privateCount = InterlockedDecrement(&m_uiRefCount);
    //        assert(m_uiRefCount >= 0);
    //    }
    //    Unlock();
    //    if(privateCount == 0) 
    //        delete this;
    //    return privateCount;
    //}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets asynchronous context. </summary>
    ///
    /// <remarks>   crossbac, 5/1/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the asynchronous context. </returns>
    ///-------------------------------------------------------------------------------------------------

    AsyncContext * 
    SyncPoint::GetAsyncContext(
        VOID
        )
    {
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
    SyncPoint::GetPlatformContextObject(
        void
        )
    {
        return m_pPlatformAsyncContextObject;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the platform wait object. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the platform wait object. </returns>
    ///-------------------------------------------------------------------------------------------------

    void * 
    SyncPoint::GetPlatformWaitObject(
        void
        )
    {
        return m_pPlatformAsyncWaitObject;
    }    

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the platform wait object. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the platform wait object. </returns>
    ///-------------------------------------------------------------------------------------------------

    void * 
    SyncPoint::GetPlatformParentObject(
        void
        )
    {
        return m_pPlatformParentSyncObject;
    }    


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is outstanding. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    Context for the outstanding asynchronous operations. </param>
    ///
    /// <returns>   true if outstanding, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    SyncPoint::QueryOutstanding(
        __in AsyncContext * pAsyncContext
        )
    {
        assert(LockIsHeld());
        if(m_bOutstanding) {

            AsyncContext * pCheckContext = (pAsyncContext == NULL) ? m_pAsyncContext : pAsyncContext;
            Accelerator * pCheckAccelerator = (pAsyncContext == NULL) ? 
                    m_pAsyncContext->GetDeviceContext() : 
                    pAsyncContext->GetDeviceContext();
            
            pCheckAccelerator->Lock();
            pCheckContext->Lock();
            if(!pCheckContext->PlatformSpecificQueryOutstanding(this)) {
                m_bOutstanding = FALSE;
                m_bStatusQueried = TRUE;
            }
            pCheckContext->Unlock();
            pCheckAccelerator->Unlock();
        }
        return m_bOutstanding;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this sync point represents outstanding work or work that has been
    ///             completed without blocking to acquire the locks needed to update async context
    ///             and accelerator state when a state change on this sync point is detected.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   true if outstanding, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    SyncPoint::NonblockingQueryOutstanding(
        VOID
        )
    {
        assert(LockIsHeld());
        if(m_bOutstanding) {
            assert(m_pAsyncContext != NULL);
            assert(m_pAsyncContext->SupportsExplicitAsyncOperations());
            if(!m_pAsyncContext->PlatformSpecificNonblockingQueryOutstanding(this)) {
                m_bOutstanding = FALSE;
                m_bStatusQueried = TRUE;
            }
        }
        return m_bOutstanding;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this sync point is *definitely* resolved. If this returns false, then
    ///             the sync point represents completed work and no lock is required to check this
    ///             since the transition is monotonic. If it returns TRUE indicating the work is
    ///             still outstanding, that doesn't mean the sync point hasn't resolved. It just
    ///             means the caller should acquire locks and call QueryOutstanding to get a higher
    ///             fidelity answer.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   true if outstanding, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    SyncPoint::QueryOutstandingFlag(
        VOID
        )
    {
        return m_bOutstanding;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Marks this sync point as retired, meaning all the ops preceding it
    /// 			are known to be complete. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    SyncPoint::MarkRetired(
        __in BOOL bContextSynchronized,
        __in BOOL bStatusQueried
        )
    {
        assert(LockIsHeld());
        m_bOutstanding = FALSE;
        m_bStatusQueried = bStatusQueried;
        m_bContextSynchronized = bContextSynchronized;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a string describing this refcount object. Allows subclasses to
    ///             provide overrides that make leaks easier to find when detected by the
    ///             rc profiler. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/9/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the rectangle profile descriptor. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::string
    SyncPoint::GetRCProfileDescriptor(
        VOID
        )
    {
        std::stringstream ss;
        ss  << "SyncPoint(os=" << m_bOutstanding 
            << ", qry=" << m_bStatusQueried
            << ", ctxtsync=" << m_bContextSynchronized
            << ")";
        return ss.str();
    }   
        
};
