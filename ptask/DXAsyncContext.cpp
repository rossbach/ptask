///-------------------------------------------------------------------------------------------------
// file:	DXAsyncContext.cpp
//
// summary:	Implements the DirextX asynchronous context class
///-------------------------------------------------------------------------------------------------

#include "primitive_types.h"
#include <iostream>
#include <assert.h>
#include "task.h"
#include "accelerator.h"
#include "DXAsyncContext.h"
#include "PDXBuffer.h"
#include "SyncPoint.h"
using namespace std;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///
    /// <param name="pDeviceContext">   [in,out] If non-null, context for the device. </param>
    /// <param name="pTaskContext">     [in,out] If non-null, context for the task. </param>
    ///-------------------------------------------------------------------------------------------------

    DXAsyncContext::DXAsyncContext(
        __in Accelerator * pDeviceContext,
        __in Task * pTaskContext,
        __in ASYNCCONTEXTTYPE eAsyncContextType
        ) : AsyncContext(pDeviceContext, 
                         pTaskContext,
                         eAsyncContextType) 
    {
        m_pDXContext = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 5/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    DXAsyncContext::~DXAsyncContext() {
        if(m_pDXContext != NULL) {
            assert(this->m_qOutstanding.size() == 0);
            m_pDXContext->Release();
            m_pDXContext = NULL;
        } else {
            // looks like it was never used... be sure.
            assert(m_qOutstanding.size() == 0);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes this object. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    DXAsyncContext::Initialize(
        VOID
        )
    {
        BOOL bSuccess = FALSE;
        Lock();
        assert(m_pDXContext == NULL);
        if(m_pDXContext == NULL) {
            m_pDeviceContext->Lock();
            m_pDXContext = (ID3D11DeviceContext*) m_pDeviceContext->GetContext();
            m_pDXContext->AddRef();
            bSuccess = m_pDXContext != NULL;
            assert(bSuccess);
            m_pDeviceContext->Unlock();
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
    DXAsyncContext::PlatformSpecificCreateSyncPoint(
        void * pPSSyncObject
        )
    {
        SyncPoint * pSyncPoint = NULL;
        Lock();
        assert(m_pDXContext != NULL);
        if(m_pDXContext != NULL) {

            m_pDeviceContext->Lock();

            pSyncPoint = new SyncPoint(this,
                                       (void*) m_pDXContext,
                                       (void*) NULL,
                                       pPSSyncObject);

            m_pDeviceContext->Unlock();
        }
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
    DXAsyncContext::PlatformSpecificDestroySynchronizationPoint(
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
    DXAsyncContext::PlatformSpecificInsertFence(
        __in SyncPoint * pSyncPoint
        )
    {
#if 1
        PDXBuffer * pDXBuffer = reinterpret_cast<PDXBuffer*>(pSyncPoint->GetPlatformParentObject());
        if(pDXBuffer != NULL) {
            return pDXBuffer->CompleteOutstandingOps();
        }
        return FALSE;
#else
        assert(pSyncPoint != NULL);
        assert(LockIsHeld());
        assert(pSyncPoint->LockIsHeld());
        ID3D11DeviceContext * pDXContext = (ID3D11DeviceContext*) pSyncPoint->GetPlatformContextObject();
        UINT uiTS = pSyncPoint->GetTimestamp();
        assert(uiTS > m_uiLastSyncTimestamp);
        
        Accelerator * pDeviceContext = 
            pDependentContext == NULL ? 
                m_pDeviceContext :
                pDependentContext->GetDeviceContext();

        pDeviceContext->Lock();
        pDXContext->Flush();
        pDeviceContext->Unlock();
#endif
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
    DXAsyncContext::PlatformSpecificSynchronousWait(
        __in SyncPoint * pSyncPoint
        )
    {
        PDXBuffer * pDXBuffer = reinterpret_cast<PDXBuffer*>(pSyncPoint->GetPlatformParentObject());
        if(pDXBuffer != NULL) {
            return pDXBuffer->CompleteOutstandingOps();
        }
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
    DXAsyncContext::GetPlatformContextObject(
        VOID
        )
    {
        return (void*) m_pDXContext;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   platform specific synchronize context. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    DXAsyncContext::PlatformSpecificSynchronizeContext(
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
    DXAsyncContext::PlatformSpecificQueryOutstanding(
        __in SyncPoint * pSyncPoint
        )
    {
        PDXBuffer * pDXBuffer = reinterpret_cast<PDXBuffer*>(pSyncPoint->GetPlatformParentObject());
        if(pDXBuffer != NULL) {
            return pDXBuffer->HasOutstandingOps();
        }
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
    DXAsyncContext::PlatformSpecificNonblockingQueryOutstanding(
        __inout SyncPoint * pSyncPoint
        )
    {
        assert(pSyncPoint);
        void * lpvParentObject = pSyncPoint->GetPlatformParentObject();
        if(lpvParentObject != NULL) {
            PDXBuffer * pDXBuffer = reinterpret_cast<PDXBuffer*>(lpvParentObject);
            return pDXBuffer->HasOutstandingOps();
            // return !pDXBuffer->CompleteOutstandingOps();
        }
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
    DXAsyncContext::PlatformSpecificLocklessSynchronousWait(
        __in SyncPoint * pSyncPoint 
        )
    {
        assert(pSyncPoint);
        void * lpvParentObject = pSyncPoint->GetPlatformParentObject();
        if(lpvParentObject != NULL) {
            PDXBuffer * pDXBuffer = reinterpret_cast<PDXBuffer*>(lpvParentObject);
            return !pDXBuffer->CompleteOutstandingOps();
        }
        return FALSE;    }


};
