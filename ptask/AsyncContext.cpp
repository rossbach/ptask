///-------------------------------------------------------------------------------------------------
// file:	AsyncContext.cpp
//
// summary:	Implements the asynchronous context class
///-------------------------------------------------------------------------------------------------

#include <stdio.h>
#include <iostream>
#include <crtdbg.h>
#include <assert.h>
#include "accelerator.h"
#include "task.h"
#include "SyncPoint.h"
#include "AsyncContext.h"
#include "AsyncDependence.h"
#include "CUTask.h"
#include "nvtxmacros.h"
#include <vector>
#include <set>
#include <algorithm>
#include <deque>
using namespace std;

namespace PTask {

    /// <summary>   true if the conservative GC (which never makes backend API calls)
    ///             should be used to garbage collect dependences on the back of the
    ///             queue whose actual status is unknown. The Conservative version is
    ///             faster when the depth of the queue remains small, because API calls
    ///             to query events can be costly. However, letting the outstanding queue
    ///             get too big eventually introduces a performance cliff, so we generally
    ///             prefer the new GC, which attempts to balance these competing concerns
    ///             using the s_uiGCOutstandingQueryThreshold parameter defined below.
    /// </summary>
    BOOL AsyncContext::s_bUseConservativeGC = FALSE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pDeviceContext">       [in,out] If non-null, context for the device. </param>
    /// <param name="pTaskContext">         [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="eAsyncContextType">    If non-null, the wait object. </param>
    ///-------------------------------------------------------------------------------------------------

    AsyncContext::AsyncContext(
        __in Accelerator * pDeviceContext,
        __in Task * pTaskContext,
        __in ASYNCCONTEXTTYPE eAsyncContextType
        ) : ReferenceCounted()
    {
        m_pTaskContext = pTaskContext;
        m_pDeviceContext = pDeviceContext;      
        m_eAsyncContextType = eAsyncContextType;
        assert(pTaskContext != NULL || m_eAsyncContextType != ASYNCCTXT_TASK);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    AsyncContext::~AsyncContext(
        VOID
        )
    {
        TruncateOutstandingQueue(FALSE);
        assert(m_qOutstanding.size() == 0);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a dependence on the synchronization point. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    AsyncDependence * 
    AsyncContext::CreateDependence(
        __in ASYNCHRONOUS_OPTYPE eOperationType
        )
    {
        Lock();
        AsyncDependence * pDependence = NULL;
        SyncPoint * pSyncPoint = CreateSyncPoint(NULL);
        assert(pSyncPoint != NULL);
        if(pSyncPoint != NULL) {
            pDependence = CreateDependence(pSyncPoint, eOperationType);
        }
        Unlock();
        return pDependence;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a synchronization point. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    SyncPoint * 
    AsyncContext::CreateSyncPoint(
        void * pPSSyncObject
        )
    {
        assert(LockIsHeld());
        MARKRANGEENTER(L"CreateSyncPoint");
        SyncPoint * pSyncPoint = PlatformSpecificCreateSyncPoint(pPSSyncObject);
        if(pSyncPoint != NULL) {
            GarbageCollectOutstandingQueue();
            m_qOutstanding.push_back(pSyncPoint);
            pSyncPoint->AddRef();
        }
        MARKRANGEEXIT();
        return pSyncPoint;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a dependence. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    AsyncDependence *
    AsyncContext::CreateDependence(
        __in SyncPoint * pSyncPoint,
        __in ASYNCHRONOUS_OPTYPE eOperationType
        )
    {
        // create a new dependence for the outstanding async operation.
        // make sure that we are not creating dependences on the default
        // stream (er, context) and that we are not creating dependences for
        // memory operations on execution streams (er, contexts).        
        assert(pSyncPoint != NULL);
        assert(pSyncPoint->m_bOutstanding);
        assert(!ASYNCCTXT_ISDEFAULT(m_eAsyncContextType));
        assert((ASYNCCTXT_ISEXECCTXT(m_eAsyncContextType) && ASYNCOP_ISEXEC(eOperationType)) || 
               (ASYNCCTXT_ISXFERCTXT(m_eAsyncContextType) && ASYNCOP_ISXFER(eOperationType)));

        return new AsyncDependence(this, pSyncPoint, eOperationType);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destroys the synchronise point described by pSyncPoint. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    AsyncContext::DestroySyncPoint(
        __in SyncPoint * pSyncPoint
        )
    {
        return PlatformSpecificDestroySynchronizationPoint(pSyncPoint);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synchronizes the context. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    AsyncContext::SynchronizeContext(
        VOID
        )
    {
        Lock();
        assert(m_pDeviceContext->LockIsHeld());
        BOOL bSuccess = PlatformSpecificSynchronizeContext();
        assert(bSuccess);
        if(bSuccess) {
            TruncateOutstandingQueue(TRUE);
        }
        Unlock();
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
    AsyncContext::OrderSubsequentOperationsAfter(
        __in AsyncDependence * pDependence
        )
    {
        BOOL bSuccess = TRUE;
        Lock();
        pDependence->Lock();

        // if the dependence was issued in the same context 
        // as this one, there is no need to synchronize.
        if(pDependence->GetContext() != this) {
            SyncPoint * pSP = pDependence->GetSyncPoint();
            pSP->Lock();
            // if(pSP->QueryOutstanding(this)) {
                bSuccess = PlatformSpecificInsertFence(pSP);
                assert(bSuccess);
            // } else {
            //     TruncateOutstandingQueueFrom(pSP);
            // }
            pSP->Unlock();
        }

        pDependence->Unlock();
        Unlock();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Wait for dependence asynchronously by inserting a dependence
    ///             in the current context (stream) on the event in the sync point. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    AsyncContext::InsertFence(
        __in SyncPoint * pSyncPoint
        )
    {
        BOOL bSuccess = TRUE;
        Lock();
        pSyncPoint->Lock();
        if(pSyncPoint->QueryOutstanding(this)) {
            bSuccess = PlatformSpecificInsertFence(pSyncPoint);
            assert(bSuccess);
        } else {
            TruncateOutstandingQueueFrom(pSyncPoint);
        }
        pSyncPoint->Unlock();
        Unlock();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synchronous wait for dependence resolution. </summary>
    ///
    /// <remarks>   crossbac, 6/25/2013. </remarks>
    ///
    /// <param name="pDependence">  [in,out] If non-null, the synchronise point. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    AsyncContext::SynchronousWait(
        __in AsyncDependence * pDependence
        )
    {
        pDependence->Lock();
        BOOL bResult = AsyncContext::SynchronousWait(pDependence->GetSyncPoint());
        pDependence->Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synchronous wait for dependence resolution. </summary>
    ///
    /// <remarks>   crossbac, 6/25/2013. </remarks>
    ///
    /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    AsyncContext::SynchronousWait(
        __in SyncPoint * pSyncPoint
        )
    {
        if(__IsSyncPointResolvedNoLock(pSyncPoint))
            return TRUE;

        // the sync point has not been marked as retired yet, so we are going to need some locks to
        // check it. The accelerator and context locks are required because we are going to make some
        // framework calls (e.g. to check or wait on events). 
        AsyncContext * pAsyncContext = pSyncPoint->GetAsyncContext();
        Accelerator * pAccelerator = pAsyncContext->GetDeviceContext();
        pAccelerator->Lock();
        pAsyncContext->Lock();
        BOOL bSuccess = pAsyncContext->__SynchronousWaitLocksHeld(pSyncPoint);
        pAsyncContext->Unlock();
        pAccelerator->Unlock();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sync points, once marked resolved, can never return to the outstanding state.
    ///             Consequently, if a lock-free check of the oustanding flag returns false, there is
    ///             no danger of a race. Conversely, checking if the state is unknown requires
    ///             accelerator and context locks which restrict concurrency and have lock ordering
    ///             disciplines that make it difficult to *always* have these locks when this check
    ///             is required. So a quick check without a lock that can avoid locks when they are
    ///             unnecessary is a handy tool.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/26/2013. </remarks>
    ///
    /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
    ///
    /// <returns>   true if synchronise point resolved no lock, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    AsyncContext::__IsSyncPointResolvedNoLock(
        __in SyncPoint * pSyncPoint
        )
    {
        // sync points are marked not outstanding once they have been found to be completed--checking
        // this flag requires no lock since it's monotonic. AddRef the sync point while we check it to
        // make sure it doesn't get deleted out from under us, but acquire no lock. 

        pSyncPoint->AddRef();
        BOOL bAlreadyResolved = !pSyncPoint->QueryOutstandingFlag();
        pSyncPoint->Release();
        return bAlreadyResolved;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Wait for dependence--synchronously. Because we may have to make backend
    ///             framework calls (e.g. to wait or check CUDA event states) we may require
    ///             a number of fairly coarse locks, including an accelerator lock. When calling
    ///             this from task dispatch context, the caller must acquire all locks up front
    ///             since there are lock ordering displines such as (Accelerator->Datablock) that 
    ///             are there to prevent deadlock for concurrent tasks. 
    ///             
    ///             This version assumes (or rather only asserts) that accelerator locks are held
    ///             already, so it can be called from dispatch context: Task is a friend class
    ///             to enable this while minimizing the potential for abuse.
    ///
    ///             This is a building block for the public version, which first collects locks,
    ///             but which cannot be called from a dispatch context as a result.
    ///              </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pDependence">          [in,out] If non-null, the dependence. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    AsyncContext::__SynchronousWaitLocksHeld(
        __in AsyncDependence * pDependence
        )
    {
        if(__IsSyncPointResolvedNoLock(pDependence->GetSyncPoint()))
            return TRUE;

        pDependence->Lock();
        BOOL bResult = __SynchronousWaitLocksHeld(pDependence->GetSyncPoint());
        pDependence->Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Wait for dependence--synchronously. Because we may have to make backend
    ///             framework calls (e.g. to wait or check CUDA event states) we may require
    ///             a number of fairly coarse locks, including an accelerator lock. When calling
    ///             this from task dispatch context, the caller must acquire all locks up front
    ///             since there are lock ordering displines such as (Accelerator->Datablock) that 
    ///             are there to prevent deadlock for concurrent tasks. 
    ///             
    ///             This version assumes (or rather only asserts) that accelerator locks are held
    ///             already, so it can be called from dispatch context: Task is a friend class
    ///             to enable this while minimizing the potential for abuse.
    ///
    ///             This is a building block for the public version, which first collects locks,
    ///             but which cannot be called from a dispatch context as a result.
    ///              </summary>
    ///              
    /// <remarks>   crossbac, 6/25/2013. </remarks>
    ///
    /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    AsyncContext::__SynchronousWaitLocksHeld(
        __in SyncPoint * pSyncPoint
        )
    {
        if(__IsSyncPointResolvedNoLock(pSyncPoint))
            return TRUE;

        // the sync point has not been marked as retired yet, so we are going to need some locks to
        // check it. The accelerator and context locks are required because we are going to make some
        // framework calls (e.g. to check or wait on events). Assert that we have the accelerator and
        // context locks already--this should be called either from the public function with the same
        // name, or from dispatch context, where the task should have collected all the potentially
        // required up front. 
       
        assert(GetDeviceContext()->LockIsHeld());
        assert(LockIsHeld());

        // Assert that this is being called from the device context that created the sync point. We
        // required this for two reasons:
        // 1. We are going to try to truncate the outstanding queue for this
        //    context if we succeed.
        // 2. The synchronous version of this should *only* be called when there is no
        //    context capable of oustanding operations (in which case we could just
        //    create a dependence instead of syncing). In such cases the only thing
        //    we can do is use the context that created the dependence to synchronize,
        //    so using any other context to make this call violates the contract. 

        AsyncContext * pSyncPointContext = pSyncPoint->GetAsyncContext();
        assert(pSyncPointContext == this); 
        assert(pSyncPointContext != NULL && pSyncPointContext->LockIsHeld());
        assert(pSyncPointContext != NULL && 
               pSyncPointContext->GetDeviceContext()->SupportsExplicitAsyncOperations());

        BOOL bSuccess = FALSE;
        pSyncPoint->Lock();

        if(pSyncPoint->QueryOutstanding(pSyncPointContext)) {

            // attempt to wait only for the operation that induced the creation of the sync point. If this
            // succeeds, we prefer it since it allows us to avoid heavier-weight solutions like
            // synchronizing the stream or synchronizing the accelerator context, both of which are
            // expensive and will likely force us to incure greater latencies that are truly required. 

            bSuccess = pSyncPointContext->PlatformSpecificSynchronousWait(pSyncPoint);
            if(bSuccess) {

                // the operation that induced the creation of this sync point has completed.
                // This means that we can safely mark any previous operations queued on this
                // context as completed without waiting for them.

                pSyncPointContext->TruncateOutstandingQueueFrom(pSyncPoint);

            } else {
                               
                assert(bSuccess && "PlatformSpecificSynchronousWait failure...");
                PTask::Runtime::Inform("__SynchronousWaitLocksHeld: PlatformSpecificSynchronousWait failure...syncing context..\n");
                
                // we failed to sync with just the outstanding operation, so try synchronizing
                // the entire async context before giving up.
                
                bSuccess = pSyncPointContext->PlatformSpecificSynchronizeContext();
                if(!bSuccess) {

                    // we failed to sync just the async context. TODO: sync the entire device!
                    assert(bSuccess && "PlatformSpecificSynchronizeContext failure");
                    PTask::Runtime::MandatoryInform("PlatformSpecificSynchronizeContext failure");

                } else {

                    // we synchronized the AsyncContext, which means that everything that is outstanding
                    // for this context has completed and we can safely mark the sync points resolved.
                    pSyncPointContext->TruncateOutstandingQueue(TRUE);
                }

            } 

        } else {

            // the sync point already resolved. 
            // No need to wait for it. Just return success.
            pSyncPointContext->TruncateOutstandingQueueFrom(pSyncPoint);
            bSuccess = TRUE;
        }
        pSyncPoint->Unlock();

        return bSuccess;
    }



    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pDependence' is dependence resolved. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pDependence">  [in,out] If non-null, the dependence. </param>
    ///
    /// <returns>   true if dependence resolved, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    AsyncContext::QueryDependenceOutstanding(
        __in AsyncDependence * pDependence
        )
    {
        BOOL bOutstanding = FALSE;
        pDependence->Lock();
        SyncPoint * pSyncPoint = pDependence->GetSyncPoint();
        pSyncPoint->Lock();
        if(pSyncPoint->QueryOutstanding(this)) {
            bOutstanding = TRUE;
        }
        pSyncPoint->Unlock();
        pDependence->Unlock();
        return bOutstanding;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Non-blocking check whether the dependence is still outstanding. </summary>
    ///
    /// <remarks>   crossbac, 7/2/2013. </remarks>
    ///
    /// <param name="pDependence">  [in,out] If non-null, the dep. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    AsyncContext::NonblockingQueryOutstanding(
        __inout AsyncDependence * pDependence
        )
    {
        assert(pDependence != NULL);
        BOOL bOutstanding = FALSE;
        pDependence->Lock();
        SyncPoint * pSyncPoint = pDependence->GetSyncPoint();
        pSyncPoint->Lock();
        if(pSyncPoint->QueryOutstandingFlag()) {
            bOutstanding = pSyncPoint->NonblockingQueryOutstanding();
            if(!bOutstanding) {
                pSyncPoint->MarkRetired(FALSE, TRUE);
            }
        }
        pDependence->Unlock();
        pSyncPoint->Unlock();
        return bOutstanding;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synchronous wait for outstanding async op--do not acquire locks required to
    ///             update async and device context state in response to a successful query or wait.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/2/2013. </remarks>
    ///
    /// <param name="pDependence">  [in,out] If non-null, the dep. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    AsyncContext::LocklessWaitOutstanding(
        __inout AsyncDependence * pDependence
        )
    {
        assert(SupportsExplicitAsyncOperations());
        assert(pDependence != NULL);
        pDependence->Lock();
        SyncPoint * pSyncPoint = pDependence->GetSyncPoint();
        pSyncPoint->Lock();
        BOOL bOutstanding = FALSE;
        if(pSyncPoint->QueryOutstandingFlag()) {
            bOutstanding = pSyncPoint->NonblockingQueryOutstanding();
            if(bOutstanding) {
                if(PlatformSpecificLocklessSynchronousWait(pSyncPoint)) {
                    bOutstanding = FALSE;
                }
            }
            if(!bOutstanding) {
                pSyncPoint->MarkRetired(FALSE, TRUE);
            }
        }
        pSyncPoint->Unlock();
        pDependence->Unlock();
        return !bOutstanding;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Truncate queue. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    AsyncContext::TruncateOutstandingQueue(
        __in BOOL bContextSynchronized
        )
    {
        Lock(); 
        while(m_qOutstanding.size() != 0) {            
            SyncPoint * pSP = m_qOutstanding.front();
            m_qOutstanding.pop_front();
            pSP->Lock();
            //if(!bContextSynchronized) {
            //    assert(!pSP->CheckOutstanding(this));
            //}
            pSP->MarkRetired(bContextSynchronized, FALSE);
            pSP->Unlock();
            pSP->Release();
        }
        Unlock();
    } 


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   DEBUG instrumentation for analyzing the composition of outstanding dependences
    ///             on this async context. How many are flagged as resolved, how many are *actually*
    ///             resolved, is the queue monotonic?
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/31/2014. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    AsyncContext::AnalyzeOutstandingQueue(
        VOID
        )
    {
        assert(LockIsHeld());
        if(m_qOutstanding.size() > PTask::Runtime::GetAsyncContextGCQueryThreshold()) { 

            int nIndex = 0;
            int nOutstandingFlag = 0;
            int nResolvedFlag = 0;
            int nOutstandingQuery = 0;
            int nResolvedQuery = 0;
            int nFirstOutstandingFlagIndex = -1;
            int nFirstResolvedFlagIndex = -1;
            int nFirstOutstandingQueryIndex = -1;
            int nFirstResolvedQueryIndex = -1;
            BOOL bNonMonotonic = FALSE;

            std::deque<SyncPoint*>::iterator di;
            for(nIndex=0, di=m_qOutstanding.begin(); di!= m_qOutstanding.end(); di++, nIndex++) {
                SyncPoint * pSP = *di;
                if(pSP->QueryOutstandingFlag()) {
                    if(nOutstandingFlag == 0) {
                        nFirstOutstandingFlagIndex = nIndex;
                    }
                    nOutstandingFlag++;
                } else {
                    if(nResolvedFlag == 0) {
                        nFirstResolvedFlagIndex = nIndex;
                    }
                    nResolvedFlag++;
                }
            }


            for(nIndex=0, di=m_qOutstanding.begin(); di!= m_qOutstanding.end(); di++, nIndex++) {
                SyncPoint * pSP = *di;
                pSP->Lock();
                BOOL bQueryOutstanding = pSP->QueryOutstanding(this);
                pSP->Unlock();
                if(bQueryOutstanding) {
                    if(nOutstandingQuery == 0) {
                        nFirstOutstandingQueryIndex = nIndex;
                    }
                    nOutstandingQuery++;
                } else {
                    if(nResolvedQuery == 0) {
                        nFirstResolvedQueryIndex = nIndex;
                    }
                    nResolvedQuery++;
                    if(nOutstandingQuery > 0)
                        bNonMonotonic = TRUE;
                }
            }

            char * lpszCtxtName = (char*)((m_pTaskContext != NULL) ? 
                                    m_pTaskContext->GetTaskName() : 
                                    m_pDeviceContext->GetDeviceName());

            PTask::Runtime::MandatoryInform("Ctxt-%s GC'ing %d deps, L-%d R-%d (1st:l-%d, r-%d) flags, "
                                            "L-%d R-%d queried (1st:l-%d, r-%d)%s\n", 
                                            lpszCtxtName,
                                            m_qOutstanding.size(), 
                                            nOutstandingFlag,
                                            nResolvedFlag,
                                            nFirstOutstandingFlagIndex,
                                            nFirstResolvedFlagIndex,
                                            nOutstandingQuery,
                                            nResolvedQuery,
                                            nFirstOutstandingQueryIndex,
                                            nFirstResolvedQueryIndex,
                                            (bNonMonotonic ? "--NON-MONOTONIC!!!!":""));
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   garbage collect the outstanding queue. Anything no longer outstanding can be
    ///             removed from the queue. The original version is very conservative about how much
    ///             it actually cleans up--it only checks flags (and thus avoids back-end API calls
    ///             to check event status), which is good for performance until the number of
    ///             outstanding deps piles up. This version attempts to balance these effects by
    ///             making API calls if the number of outstanding deps goes beyond a threshold. This
    ///             version can be reinstated with a static member variable s_bUseConservativeGC. The
    ///             threshold at which to start making API calls is controlled by
    ///             PTask::Runtime::[Get|Set]AsyncContextGCQueryThreshold().
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/31/2014. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    AsyncContext::GarbageCollectOutstandingQueue(
        VOID
        )
    {
        assert(LockIsHeld());
        MARKRANGEENTER(L"AsyncContext-GC");

        if(s_bUseConservativeGC) {

            // use the conservative GC version that makes
            // no API calls to determine the true status 
            // of outstanding deps on this context. 
           
            GarbageCollectOutstandingQueueConservatively();

        } else {

            std::deque<SyncPoint*> vCleanedQueue;
            // AnalyzeOutstandingQueue();  --- instrumentation to dump the state
            //                             --- of outstanding deps to the console.
            // -------------------------------------------------------------------

            // if the outstanding queue is short, behave exactly as the old "conservative" GC method did:
            // just check flags. if things are building up, we need to do something. When the queue is
            // deeper than the threshold, make some API calls to determine the status of outstanding deps.
            // Since more recent deps are more likely to be still active, the ideal approach would involve
            // a binary search in a region toward the front of the queue. For now, just take it from the
            // top and find out if it actually matters. If we find one that *is* resolved, by definition we
            // can retire all sync points deeper in the queue, as they correspond to operations that
            // precede the resolved one, and which are ordered by the driver relative to that SP. 

            BOOL bRetiredSPFound = FALSE;
            BOOL bExplicitQuery = (m_qOutstanding.size() >= PTask::Runtime::GetAsyncContextGCQueryThreshold());
                
            while(m_qOutstanding.size() != 0) {            

                // release our references to any sync points that have already been retired. Note that we do
                // not need to lock the sync-points as we examine them because the transition to retired is
                // monotonic, and we are simply checking to see if any sync points in our queue have been
                // marked retired by other activities. 

                SyncPoint * pSP = m_qOutstanding.front();
                m_qOutstanding.pop_front();

                if(bRetiredSPFound) {

                    // we already found a retired sync point earlier in our traversal of the queue. That means this
                    // one is retired: we don't need to query the runtime to figure that out, so release it,
                    // regardless of whether we started with a beyond threshold queue and made an API call to
                    // figure this out. 
                        
                    pSP->Lock();
                    pSP->MarkRetired(FALSE, FALSE);
                    pSP->Unlock();
                    pSP->Release();

                } else {

                    BOOL bAssumeOutstanding = pSP->QueryOutstandingFlag();
                    BOOL bExplicitOutstanding = bAssumeOutstanding;
                    if(bAssumeOutstanding && bExplicitQuery) {

                        // we want to explicitly query SPs for which we cannot
                        // make a safe inference, since the goal is to truncate the queue. 
                        // however, we may not hold a lock on the sync point, but we do already 
                        // hold a lock on the accelerator, so it should be safe to
                        // acquire the sp.

                        pSP->Lock();
                        bExplicitOutstanding = pSP->NonblockingQueryOutstanding();
                        if(bExplicitOutstanding) 
                            bExplicitOutstanding = pSP->QueryOutstanding(this);
                        pSP->Unlock();

                    }
                    BOOL bOutstanding = bExplicitQuery ? bExplicitOutstanding : bAssumeOutstanding;

                    if(bOutstanding) {
                        
                        // either we know this SP is retired because we queried it, or it's not explicitly marked
                        // retired and we didn't check. So just assume the worst--it's still active, so put it on the
                        // vCleanedQueue list, which is later used to rebuild the queue. 

                        vCleanedQueue.push_back(pSP);

                    } else {

                        // this one is retired...just release it, and remember that 
                        // we found it, since all subsequent sync points in the queue
                        // must also be retired.
                            
                        bRetiredSPFound = TRUE;
                        pSP->Release();
                    }
                }
            }

            // rebuild the outstanding queue--everything *not* in the
            // vCleanedQueue list is known to be retired, so just
            // assign the cleaned queue list to the outstandning queue. 
            
            m_qOutstanding.clear();    // redundant?
            m_qOutstanding.assign(vCleanedQueue.begin(), vCleanedQueue.end());
        }

        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   garbage collect the outstanding queue. Anything no longer outstanding can be
    ///             removed from the queue. This (old, obsolete) version is very conservative about
    ///             how much it actually cleans up--it only checks flags (and thus avoids back-end
    ///             API calls to check event status), which is good for performance until the number
    ///             of outstanding deps piles up. The new version attempts to balance these effects
    ///             by making API calls if the number of outstanding deps goes beyond a threshold.
    ///             This version can be reinstated with a static member variable s_bUseConservativeGC.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    AsyncContext::GarbageCollectOutstandingQueueConservatively(
        VOID
        )
    {
        assert(LockIsHeld());
        std::deque<SyncPoint*> vCleanedQueue;

        while(m_qOutstanding.size() != 0) {            

            // release our references to any sync points
            // that have already been retired. Note that we
            // do not need to lock the sync-points as we examine
            // them because the transition to retired is monotonic,
            // and we are simply checking to see if any sync points in
            // our queue have been marked retired by other activities.

            SyncPoint * pSP = m_qOutstanding.front();
            m_qOutstanding.pop_front();
            if(pSP->QueryOutstandingFlag())
                vCleanedQueue.push_back(pSP);
            else
                pSP->Release();
        }
        m_qOutstanding.clear();
        m_qOutstanding.assign(vCleanedQueue.begin(), vCleanedQueue.end());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Truncate queue. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pSyncPoint">   [in,out] If non-null, the synchronise point. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    AsyncContext::TruncateOutstandingQueueFrom(
        __in SyncPoint * pSyncPoint
        )
    {
        Lock(); 
        std::deque<SyncPoint*>::iterator di;
        di = find(m_qOutstanding.begin(), m_qOutstanding.end(), pSyncPoint);
        if(di != m_qOutstanding.end()) {
            BOOL bReleasedTargetSP = FALSE;
            do {
                SyncPoint * pSP = m_qOutstanding.front();
                m_qOutstanding.pop_front();
                pSP->Lock();
                pSP->MarkRetired(FALSE, FALSE);
                pSP->Unlock();
                bReleasedTargetSP = (pSP == pSyncPoint);
                pSP->Release();

            } while(!bReleasedTargetSP);

        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the device context. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the device context. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    AsyncContext::GetDeviceContext(
        void
        ) 
    {
        return m_pDeviceContext;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the task context. Can be null for the default async context of 
    /// 			an accelerator. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the device context. </returns>
    ///-------------------------------------------------------------------------------------------------

    Task * 
    AsyncContext::GetTaskContext(
        void
        ) 
    {
        return m_pTaskContext;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if this accelerator backing this context encapsulates a backend
    ///             framework that provides explicit APIs for managing outstanding (Asynchronous)
    ///             operations. When this is the case, the corresponding AsyncContext subclass can
    ///             manage outstanding dependences explicitly to increase concurrency and avoid
    ///             syncing with the device. When it is *not* the case, we must synchronize when we
    ///             data to and from this accelerator context and contexts that *do* support an
    ///             explicit async API. For example, CUDA supports the stream and event API to
    ///             explicitly manage dependences and we use this feature heavily to allow task
    ///             dispatch to get far ahead of device- side dispatch. However when data moves
    ///             between CUAccelerators and other accelerator classes, we must use synchronous
    ///             operations or provide a way to wait for outstanding dependences from those
    ///             contexts to resolve. This method is used to tell us whether we can create an
    ///             outstanding dependence after making calls that queue work, or whether we need to
    ///             synchronize.
    ///             
    ///             This method simply calls the method of the same name on the (device context)
    ///             accelerator, and is only provided for convenience.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    AsyncContext::SupportsExplicitAsyncOperations(
        VOID
        )
    {
        assert(m_pDeviceContext != NULL);
        if(m_pDeviceContext == NULL)
            return FALSE;
        return m_pDeviceContext->SupportsExplicitAsyncOperations();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Locks the accelerator. </summary>
    ///
    /// <remarks>   crossbac, 6/26/2013. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    VOID        
    AsyncContext::LockAccelerator(
        VOID
        )
    {
        m_pDeviceContext->Lock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unlocks the accelerator. </summary>
    ///
    /// <remarks>   crossbac, 6/26/2013. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    VOID        
    AsyncContext::UnlockAccelerator(
        VOID
        )
    {
        m_pDeviceContext->Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the type (dedicated purpose) of the asynchronous context. </summary>
    ///
    /// <remarks>   crossbac, 7/8/2013. </remarks>
    ///
    /// <returns>   The asynchronous context type. </returns>
    ///-------------------------------------------------------------------------------------------------

    ASYNCCONTEXTTYPE 
    AsyncContext::GetAsyncContextType(
        VOID
        )
    {
        return m_eAsyncContextType;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notifies the device synchronized. </summary>
    ///
    /// <remarks>   crossbac, 7/8/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    AsyncContext::NotifyDeviceSynchronized(
        VOID
        )
    {
        assert(FALSE);
        PTask::Runtime::HandleError("%s called! Only derived classes should get called!\n",
                                    __FUNCTION__);
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
    AsyncContext::GetRCProfileDescriptor(
        VOID
        )
    {
        std::stringstream ss;
        ss  << "AsyncContext(typ=" << AsyncCtxtTypeToString(m_eAsyncContextType)
            << ", dev=" << m_pDeviceContext->GetDeviceName() 
            << ", oqsiz=" << m_qOutstanding.size()
            << ")";
        return ss.str();
    }

};
