///-------------------------------------------------------------------------------------------------
// file:	CUAsyncContext.cpp
//
// summary:	Implements the cu asynchronous context class
///-------------------------------------------------------------------------------------------------

#ifdef CUDA_SUPPORT

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "datablock.h"
#include "CUAsyncContext.h"
#include "SyncPoint.h"
#include "Task.h"
#include "extremetrace.h"
#include "nvtxmacros.h"
#include <vector>
#include <iostream>
#include <assert.h>

using namespace std;

#define PTASSERT(x) assert(x)

#define ACQUIRE_CTX(acc)                                \
        acc->Lock();                                    \
        BOOL bFCC = !acc->IsDeviceContextCurrent();     \
        if(bFCC) acc->MakeDeviceContextCurrent();        
#define RELEASE_CTX(acc)                                \
        if(bFCC) acc->ReleaseCurrentDeviceContext();    \
        acc->Unlock();
#define ONFAILURE(apicall, code)                                        \
    assert(FALSE);                                                      \
    PTask::Runtime::HandleError("%s::%s--%s failed res=%d (line:%d)\n", \
                                __FILE__,                               \
                                __FUNCTION__,                           \
                                #apicall,                               \
                                code,                                   \
                                __LINE__)

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pDeviceContext">   [in,out] If non-null, context for the device. </param>
    /// <param name="pTaskContext">     [in,out] If non-null, context for the task. </param>
    ///-------------------------------------------------------------------------------------------------

    CUAsyncContext::CUAsyncContext(
        __in Accelerator * pDeviceContext,
        __in Task * pTaskContext,
        __in ASYNCCONTEXTTYPE eAsyncContextType
        ) : AsyncContext(pDeviceContext, 
                         pTaskContext,
                         eAsyncContextType) 
    {
        m_hStream = NULL;
        m_hLastFence = NULL;
        m_hEvent = NULL;
        m_nStreamPriority = 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    CUAsyncContext::~CUAsyncContext() {
        assert(this->m_qOutstanding.size() == 0);
        if(m_hStream != NULL)
            cuStreamDestroy(m_hStream);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes this object. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CUAsyncContext::Initialize(
        VOID
        )
    {
        BOOL bSuccess = TRUE;
        Lock();
        assert(m_hStream == NULL);

        // if the task context is null, then this is either the default
        // async context for the accelerator (the one with stream == 0,
        // under which all async ops without an explicit stream are conducted),
        // or an async context created explicitly for transfer traffic.

        BOOL bNonTaskStream = (m_pTaskContext == NULL) && !ASYNCCTXT_ISDEFAULT(m_eAsyncContextType);
        BOOL bTaskStream = ((m_pTaskContext != NULL) && 
                            (m_pTaskContext->GetAcceleratorClass() == ACCELERATOR_CLASS_CUDA || 
                            (ASYNCCTXT_ISEXECCTXT(m_eAsyncContextType) && m_pTaskContext->DependentBindingsRequirePSObjects())));
        BOOL bStreamCreate = (bNonTaskStream || bTaskStream) && (m_hStream == NULL);

        if(bStreamCreate) {

            // we need to create a stream to back this context. 
            // this is called at init time, so we expect no device
            // context to be current (unless this is a task context).
            // ----------------------------------------------------------
            // cjr: 4/22/14--actually, if user code has already initialized
            // CUDA for other reasons, there can be a current context already.
            // So, there is actually no point in querying it since we only do
            // so to enable an assertion to the contrary. 
            
            //// CUcontext pLastCtx = NULL;
            //// CUresult eQueryRes = cuCtxGetCurrent(&pLastCtx);
            //// if(eQueryRes != CUDA_SUCCESS) {
            ////     PTASSERT(eQueryRes == CUDA_SUCCESS);
            ////     ONFAILURE(cuCtxGetCurrent, eQueryRes);
            //// }
            //// assert(pLastCtx == NULL || m_pTaskContext != NULL);

            CUcontext pDevCtxt = reinterpret_cast<CUcontext>(m_pDeviceContext->GetContext());
            if(m_pTaskContext == NULL) {

                // assert(m_pTaskContext != NULL);
                CUresult ctxRes = cuCtxPushCurrent(pDevCtxt);
                if(ctxRes != CUDA_SUCCESS) {
                    PTASSERT(ctxRes == CUDA_SUCCESS);
                    ONFAILURE(cuCtxPushCurrent, ctxRes);
                }

            } else {

                m_pDeviceContext->MakeDeviceContextCurrent();
            }


            // cjr: using  CU_STREAM_NON_BLOCKING is clearly a bad idea!
            CUresult res = cuStreamCreate(&m_hStream, 0); 
            trace3("cuStreamCreate()->hStream=%16llX, res=%d\n", m_hStream, res);
            bSuccess = (res == CUDA_SUCCESS);
            if(!bSuccess) {
                PTASSERT(res == CUDA_SUCCESS);
                ONFAILURE(cuStreamCreate, res);
            }

            if(m_pTaskContext == NULL) {
                CUcontext pLCtxt = NULL;
                CUresult ctxPopRes = cuCtxPopCurrent(&pLCtxt);
                if(pLCtxt != pDevCtxt) {
                    PTASSERT(ctxPopRes == CUDA_SUCCESS);
                    ONFAILURE(cuCtxPopCurrent, ctxPopRes);
                }
                PTASSERT(pLCtxt == pDevCtxt);
            } else {
                m_pDeviceContext->ReleaseCurrentDeviceContext();
            }


        }

        Unlock();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Platform specific create synchronization point. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    SyncPoint *
    CUAsyncContext::PlatformSpecificCreateSyncPoint(
        void * pPSSyncObject
        )
    {
        SyncPoint * pSyncPoint = NULL;
        Lock();

        ACQUIRE_CTX(m_pDeviceContext);

        BOOL bMultiplexEvents = FALSE;
        
        CUevent hEvent = NULL;
        CUresult res = CUDA_SUCCESS;
        if(bMultiplexEvents) {
            if(m_hEvent != NULL) {
                res = cuEventCreate(&m_hEvent, CU_EVENT_DISABLE_TIMING);
                trace3("cuEventCreate()->hEvent=%16llX, res=%d\n", hEvent, res);
            }
            hEvent = m_hEvent;
        } else {
            res = cuEventCreate(&hEvent, CU_EVENT_DISABLE_TIMING);
            trace3("cuEventCreate()->hEvent=%16llX, res=%d\n", hEvent, res);
        }
         
        if(res != CUDA_SUCCESS) {

            // can't create an event object for this sync-point. 
            // This is most-likely fatal.
            PTASSERT(res == CUDA_SUCCESS);
            ONFAILURE(cuEventCreate, res);

        } else {

            CUresult recordRes = cuEventRecord(hEvent, m_hStream);
            trace4("cuEventRecord()->hEvent=%16llX, m_hStream=%16llX, res=%d\n", hEvent, m_hStream, recordRes);
            if(recordRes != CUDA_SUCCESS) {
                PTASSERT(recordRes == CUDA_SUCCESS);
                ONFAILURE(cuEventRecord, recordRes);
            } else {           
                pSyncPoint = new SyncPoint(this,
                                           (void*) m_hStream,
                                           (void*) hEvent,
                                           pPSSyncObject);
            }
        } 

        RELEASE_CTX(m_pDeviceContext);

        Unlock();
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
    CUAsyncContext::PlatformSpecificDestroySynchronizationPoint(
        __in SyncPoint * pSyncPoint
        )
    {
        assert(pSyncPoint != NULL);
        pSyncPoint->Lock();
        CUevent hEvent = (CUevent) pSyncPoint->GetPlatformWaitObject();
        CUresult res = cuEventDestroy(hEvent);
        trace3("cuEventDestroy()->hEvent=%16llX, res=%d\n", hEvent, res);
        BOOL bResult = (res == CUDA_SUCCESS);
        if(!bResult) {
            PTASSERT(res == CUDA_SUCCESS);
            ONFAILURE(cuEventDestroy, res);
        }
        pSyncPoint->Unlock();
        return bResult;
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
    CUAsyncContext::PlatformSpecificInsertFence(
        __in SyncPoint * pSyncPoint
        )
    {
        BOOL bSuccess = FALSE;
        assert(pSyncPoint != NULL);
        assert(LockIsHeld());
        assert(pSyncPoint->LockIsHeld());
        CUevent hEvent = (CUevent) pSyncPoint->GetPlatformWaitObject();
        assert(hEvent != NULL);

        // if we've already inserted a fence 
        // for this event, don't insert dups!
        if(hEvent == m_hLastFence)
            return TRUE;
        
        ACQUIRE_CTX(m_pDeviceContext);
        CUresult res = cuStreamWaitEvent(m_hStream, hEvent, 0);
        trace4("cuStreamWaitEvent()->hEvent=%16llX, hStream=%16llX, res=%d\n", hEvent, m_hStream, res);
        bSuccess = (res == CUDA_SUCCESS);
        if(bSuccess) {
            m_hLastFence = hEvent;
        } else {
            PTASSERT(res == CUDA_SUCCESS);
            ONFAILURE(cuStreamWaitEvent, res);
        }
        RELEASE_CTX(m_pDeviceContext);

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
    CUAsyncContext::PlatformSpecificSynchronousWait(
        __in SyncPoint * pSyncPoint
        )
    {
        ACQUIRE_CTX(m_pDeviceContext);
        BOOL bSuccess = PlatformSpecificLocklessSynchronousWait(pSyncPoint);
        RELEASE_CTX(m_pDeviceContext);
        return bSuccess;
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
    CUAsyncContext::PlatformSpecificLocklessSynchronousWait(
        __in SyncPoint * pSyncPoint 
        )
    {
        BOOL bSuccess = TRUE;
        assert(pSyncPoint != NULL);
        assert(pSyncPoint->LockIsHeld());

        if(pSyncPoint->QueryOutstandingFlag()) {
        
            CUevent hEvent = (CUevent) pSyncPoint->GetPlatformWaitObject();
            assert(hEvent != NULL);
            if(hEvent != NULL) {

                // query the event before entering a wait state--
                // it may have resolved already. 
                BOOL bOutstanding = TRUE;
                CUresult queryRes = cuEventQuery(hEvent);
                trace3("cuEventQuery()->hEvent=%16llX, res=%d\n", hEvent, queryRes);
                switch(queryRes) {
                case CUDA_SUCCESS:          bSuccess = TRUE; bOutstanding = FALSE; break;
                case CUDA_ERROR_NOT_READY:  bSuccess = FALSE; bOutstanding = TRUE; break;
                default:
                    bSuccess = FALSE;
                    bOutstanding = TRUE;
                    ONFAILURE(cuEventQuery, queryRes);
                }

                if(bOutstanding) {
                    CUresult waitRes = cuEventSynchronize(hEvent);
                    trace3("cuEventSynchronize()->hEvent=%16llX, res=%d\n", hEvent, waitRes);
                    bSuccess = waitRes == CUDA_SUCCESS;
                    bOutstanding = waitRes != CUDA_SUCCESS;
                    if(!bSuccess) {
                        PTASSERT(bSuccess);
                        ONFAILURE(cuEventSynchronize, waitRes);
                    } else {
                        m_hLastFence = hEvent;
                    }
                }
            }
        }

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
    CUAsyncContext::PlatformSpecificQueryOutstanding(
        __in SyncPoint * pSyncPoint
        )
    {
        BOOL bResolved = FALSE;
        assert(pSyncPoint != NULL);
        assert(LockIsHeld());
        assert(pSyncPoint->LockIsHeld());
        CUevent hEvent = (CUevent) pSyncPoint->GetPlatformWaitObject();
        assert(hEvent != NULL);
        
        ACQUIRE_CTX(m_pDeviceContext);
        CUresult res = cuEventQuery(hEvent);
        trace3("cuEventQuery()->hEvent=%16llX, res=%d\n", hEvent, res);
        PTASSERT(res == CUDA_SUCCESS || res == CUDA_ERROR_NOT_READY);
        switch(res) {
        case CUDA_SUCCESS:         bResolved = TRUE; break;
        case CUDA_ERROR_NOT_READY: bResolved = FALSE; break;
        default:
            bResolved = FALSE;
            ONFAILURE(cuEventQuery, res);
            break;

        }
        RELEASE_CTX(m_pDeviceContext);

        return !bResolved;
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
    CUAsyncContext::PlatformSpecificNonblockingQueryOutstanding(
        __inout SyncPoint * pSyncPoint
        )
    {
        BOOL bResolved = FALSE;
        assert(pSyncPoint != NULL);
        assert(pSyncPoint->LockIsHeld());
        assert(pSyncPoint->QueryOutstandingFlag()); // why call this if it's already known?
        CUevent hEvent = (CUevent) pSyncPoint->GetPlatformWaitObject();
        assert(hEvent != NULL);        
        if(hEvent != NULL) {
            CUresult queryRes = cuEventQuery(hEvent);
            trace3("cuEventQuery()->hEvent=%16llX, res=%d\n", hEvent, queryRes);
            switch(queryRes) {
            case CUDA_SUCCESS: bResolved = TRUE; pSyncPoint->MarkRetired(FALSE, TRUE); break;
            case CUDA_ERROR_NOT_READY: bResolved = FALSE; break;
            default:
                bResolved = FALSE;
                ONFAILURE(cuEventQuery, queryRes);
            }
        }
        return !bResolved;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   platform specific synchronize context. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CUAsyncContext::PlatformSpecificSynchronizeContext(
        VOID
        )
    {
        BOOL bSuccess = FALSE;
        assert(LockIsHeld());        
        ACQUIRE_CTX(m_pDeviceContext);
        CUresult res = cuStreamSynchronize(m_hStream);        
        trace3("cuStreamSynchronize()->m_hStream=%16llX -> res=%d\n", m_hStream, res);
        bSuccess = (res == CUDA_SUCCESS);
        if(!bSuccess) {
            PTASSERT(res == CUDA_SUCCESS);
            ONFAILURE(cuStreamSynchronize, res);
        }
        RELEASE_CTX(m_pDeviceContext);
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
    CUAsyncContext::GetPlatformContextObject(
        VOID
        )
    {
        return (void*) m_hStream;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notifies the device synchronized. </summary>
    ///
    /// <remarks>   crossbac, 7/8/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    CUAsyncContext::NotifyDeviceSynchronized(
        VOID
        )
    {
        Lock();
        TruncateOutstandingQueue(TRUE);
        Unlock();
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
    CUAsyncContext::GetRCProfileDescriptor(
        VOID
        )
    {
        std::stringstream ss;
        ss  << "CUAsyncCtxt(typ=" << AsyncCtxtTypeToString(m_eAsyncContextType)
            << ", dev=" << m_pDeviceContext->GetDeviceName() 
            << ", oqsiz=" << m_qOutstanding.size()
            << ")";
        return ss.str();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets stream priority. </summary>
    ///
    /// <remarks>   Crossbac, 3/20/2014. </remarks>
    ///
    /// <returns>   The stream priority. </returns>
    ///-------------------------------------------------------------------------------------------------

    int 
    CUAsyncContext::GetStreamPriority(
        VOID
        )
    {
        return m_nStreamPriority;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets stream priority. </summary>
    ///
    /// <remarks>   Crossbac, 3/20/2014. </remarks>
    ///
    /// <param name="nPriority">    The priority. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CUAsyncContext::SetStreamPriority(
        int nPriority
        )
    {
        // call before creating the stream!
        assert(m_hStream == NULL);
        m_nStreamPriority = nPriority; 
    }

};
#endif