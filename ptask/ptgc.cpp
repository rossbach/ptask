//--------------------------------------------------------------------------------------
// File: ptgc.cpp
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include <stdio.h>
#include <crtdbg.h>
#include "ptaskutils.h"
#include "datablock.h"
#include "ptgc.h"
#include <assert.h>
#include "PTaskRuntime.h"
#include "accelerator.h"
#include <set>
#include "nvtxmacros.h"
using namespace std;

namespace PTask {

    CRITICAL_SECTION GarbageCollector::m_csGlobalGCPtr;
    GarbageCollector * GarbageCollector::g_pGarbageCollector = NULL;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    GarbageCollector::GarbageCollector(
        UINT nGCThreads
        ) : Lockable("PTaskGarbageCollector")
    {
        ptgc_init();
        m_bQuiescent = TRUE;
        m_bAlive = TRUE;
        m_bShutdownInProgress = FALSE;
        m_bShutdownComplete = FALSE;
        m_hRuntimeTerminateEvent = PTask::Runtime::GetRuntimeTerminateEvent();
        m_hGCShutdown = CreateEventA(NULL, TRUE, FALSE, "PTask.Events.GC.Shutdown");
        m_hWorkAvailable = CreateEventA(NULL, FALSE, FALSE, "PTask.Events.GC.WorkAvailable");
        m_hQuiescent = CreateEventA(NULL, TRUE, TRUE, "PTask.Events.GC.Quiescent");
        m_nGCThreads = nGCThreads;
        m_vGCThreads = new HANDLE[nGCThreads];
        for(UINT ui=0; ui<m_nGCThreads; ui++) {
            m_vGCThreads[ui] = CreateThread(NULL, NULL, GarbageCollector::PTaskGCThread, this, NULL, NULL);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    GarbageCollector::~GarbageCollector(
        VOID
        )
    {
        Lock();
        assert(!m_bAlive);
        assert(!m_bShutdownInProgress);
        assert(m_bShutdownComplete); 
        for(UINT ui=0; ui<m_nGCThreads; ui++) {
            CloseHandle(m_vGCThreads[ui]);
            m_vGCThreads[ui] = NULL;
        }
        delete [] m_vGCThreads;
        m_vGCThreads = NULL;
        CloseHandle(m_hGCShutdown);
        CloseHandle(m_hWorkAvailable);
        CloseHandle(m_hQuiescent);
        if(m_vQ.size()) {
            std::deque<Datablock*>::iterator di;
            for(di=m_vQ.begin(); di!=m_vQ.end(); di++) {
                delete *di;
            }
            m_vQ.clear();
        }
        assert(m_vQ.size() == 0);
        Unlock();
        ptgc_deinit();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Shuts down this object and frees any resources it is using. </summary>
    ///
    /// <remarks>   Crossbac, 3/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    GarbageCollector::Shutdown(
        VOID
        )
    {
        Lock();
        assert(m_bAlive);
        assert(!m_bShutdownInProgress);
        assert(!m_bShutdownComplete);
        m_bAlive = FALSE;
        m_bShutdownInProgress = TRUE;
        SetEvent(m_hGCShutdown);
        Unlock();
        WaitForMultipleObjects(m_nGCThreads, m_vGCThreads, TRUE, INFINITE);
        Lock();
        m_bShutdownInProgress = FALSE;
        m_bShutdownComplete = TRUE;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Queue the given datablock for garbage collection. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    GarbageCollector::_QueueForGC(
        Datablock * pBlock
        )
    {
        // enqueue a block for deletion. 
        // might as well do some batching
        // while we are at it. 
        Lock();
        if(m_bAlive) {
            // if we have shutdown (shouldn't happen)
            // just leak the block. It will get cleaned up with
            // the address space. 
            ptgc_lock();
            ptgc_check_double_q(pBlock);
            ptgc_record_q(pBlock);
            ptgc_unlock();
            m_vQ.push_back(pBlock);
            if(m_vQ.size() > (size_t) PTask::Runtime::GetGCBatchSize())
                SetEvent(m_hWorkAvailable);
        } else {
            // the GC is dead. Just
            // delete the block on this thread.
            delete pBlock;
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Queue the given datablock for garbage collection. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    GarbageCollector::QueueForGC(
        Datablock * pBlock
        )
    {
        EnterCriticalSection(&m_csGlobalGCPtr);
        if(g_pGarbageCollector != NULL) {
            g_pGarbageCollector->_QueueForGC(pBlock);
        } else {
            // the gc object got cleaned up.
            // just delete the block.
            delete pBlock;
        }
        LeaveCriticalSection(&m_csGlobalGCPtr);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reports the current state of the queue to the console in some detail. 
    ///             If we are getting tight on memory, this can be a handy tool for checking
    ///             whether more aggressive GC would help the workload. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 9/7/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GarbageCollector::_Report(
        void
        )
    {
        // enqueue a block for deletion. 
        // might as well do some batching
        // while we are at it. 
        Lock();
        std::cout 
            << "GC status: " 
            << (m_bAlive?"ALIVE, q:":"DEAD, q:") 
            << m_vQ.size() 
            << " blocks awaiting deletion."
            << std::endl;
        std::map<UINT, size_t>::iterator mi;
        std::map<UINT, size_t> vPotentialFreeBytes;
        std::deque<Datablock*>::iterator di;
        for(di=m_vQ.begin(); di!=m_vQ.end(); di++) {
            std::map<UINT, size_t>::iterator bi;
            std::map<UINT, size_t> vBlockFreeBytes;
            Datablock * pBlock = (*di); 
            if(pBlock->GetInstantiatedBufferSizes(vBlockFreeBytes)) {
                for(bi=vBlockFreeBytes.begin(); bi!=vBlockFreeBytes.end(); bi++) {
                    if(vPotentialFreeBytes.find(bi->first) == vPotentialFreeBytes.end())
                        vPotentialFreeBytes[bi->first] = 0;
                    vPotentialFreeBytes[bi->first] += bi->second;
                }
            }
        }
        if(vPotentialFreeBytes.size() == 0) {
            std::cout << "no reclaimable buffers awaiting GC" << std::endl;
        } else {
            for(mi=vPotentialFreeBytes.begin(); mi!=vPotentialFreeBytes.end(); mi++) {
                std::cout 
                    << "MEMSPACE_" 
                    << mi->first 
                    << ": "
                    << mi->second
                    << " reclaimable bytes."
                    << std::endl;
            }
        }
        Unlock();   
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reports the current state of the queue to the console in some detail. 
    ///             If we are getting tight on memory, this can be a handy tool for checking
    ///             whether more aggressive GC would help the workload. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 9/7/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GarbageCollector::Report(
        void
        )
    {
        EnterCriticalSection(&m_csGlobalGCPtr);
        if(g_pGarbageCollector != NULL) {
            g_pGarbageCollector->_Report();
        } 
        LeaveCriticalSection(&m_csGlobalGCPtr);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Force a GC sweep that is targeted at a particular memory space. Can be called under
    ///             low-mem conditions by a failing attempt to allocate device memory. Forcing a 
    ///             full GC sweep from that calling context is impractical because a full sweep
    ///             requires locks we cannot acquire without breaking the lock-ordering discipline. 
    ///             However a device-specific allocation context can be assumed to hold a lock on the
    ///             accelerator for which we are allocating, making it safe to sweep the GC queue 
    ///             and free device buffers for that memspace *only* without deleting the parent blocks.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GarbageCollector::_ForceGC(
        __in UINT uiMemSpaceId
        )
    {
        MARKRANGEENTER(L"PTask-Force-GC(memSpaceId)");
        std::set<Datablock*> temp;
        Lock();
        std::deque<Datablock*>::iterator di;
        for(di=m_vQ.begin(); di!=m_vQ.end(); di++) {
            Datablock * pBlock = (*di); 
            pBlock->ReleasePhysicalBuffers(uiMemSpaceId);
        }
        Unlock();
        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Force a GC sweep that is targeted at a particular memory space. Can be called under
    ///             low-mem conditions by a failing attempt to allocate device memory. Forcing a 
    ///             full GC sweep from that calling context is impractical because a full sweep
    ///             requires locks we cannot acquire without breaking the lock-ordering discipline. 
    ///             However a device-specific allocation context can be assumed to hold a lock on the
    ///             accelerator for which we are allocating, making it safe to sweep the GC queue 
    ///             and free device buffers for that memspace *only* without deleting the parent blocks.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GarbageCollector::ForceGC(
        __in UINT uiMemSpaceId
        )
    {
        EnterCriticalSection(&m_csGlobalGCPtr);
        if(g_pGarbageCollector != NULL) {
            g_pGarbageCollector->_ForceGC(uiMemSpaceId);
        } 
        LeaveCriticalSection(&m_csGlobalGCPtr);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Task gc thread. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="p">    The p. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD WINAPI 
    GarbageCollector::PTaskGCThread(
        LPVOID p
        )
    {
        NAMETHREAD(L"PTask-GC");
        GarbageCollector * pGC = (GarbageCollector*) p;
        return pGC->GarbageCollectorThread();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Garbage collector thread proc. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD 
    GarbageCollector::GarbageCollectorThread(
        VOID
        )
    {
        std::set<Datablock*> temp;
        DWORD dwTimeout = 2000;
        BOOL bShutdown = FALSE;
        HANDLE hThread = GetCurrentThread();
        BOOL bPrioSuccess = SetThreadPriority(hThread, THREAD_MODE_BACKGROUND_BEGIN);
        assert(bPrioSuccess);

        HANDLE vStartupEvents[] = { 
            PTask::Runtime::GetRuntimeTerminateEvent(), 
            PTask::Runtime::GetRuntimeInitializedEvent(), 
            m_hWorkAvailable,
            m_hGCShutdown };
        DWORD dwStartWaitHandles = sizeof(vStartupEvents)/sizeof(HANDLE);
        DWORD dwStartup = WaitForMultipleObjects(dwStartWaitHandles, vStartupEvents, FALSE, INFINITE);
        switch(dwStartup) {
        case WAIT_OBJECT_0 + 0: return 0; // terminating! 
        case WAIT_OBJECT_0 + 1: break;    // PTask init complete
        case WAIT_OBJECT_0 + 2: break;    // work available? ok, init must be done...
        case WAIT_OBJECT_0 + 3: return 0; // terminating!
        }
        Accelerator::InitializeTLSContextManagement(PTTR_GC, 0, 1, FALSE);

        // read the m_bAlive variable outside the lock. It's monotonic
        // (we'll never go from !alive -> alive outside the constructor)
        // and the method that sets the alive flag can't return until this
        // thread exits.
        while(m_bAlive && !bShutdown) {
            BOOL bTimeout = FALSE;
            HANDLE vEvents[] = { m_hWorkAvailable, m_hGCShutdown };
            DWORD dw = WaitForMultipleObjects(2, vEvents, FALSE, dwTimeout);
            switch(dw) {
            case WAIT_OBJECT_0: 
                break;
            case WAIT_OBJECT_0 + 1: 
                // if we are shutting down, resume
                // foreground priority so we can unblock
                // the main thread sooner. 
                bShutdown = TRUE; 
                bPrioSuccess = SetThreadPriority(hThread, THREAD_MODE_BACKGROUND_END);
                assert(bPrioSuccess);
                break;
            case WAIT_TIMEOUT:
                bTimeout = TRUE;
                break;
            }

            Lock();
            MARKRANGEENTER(L"PTask-GC-sweep");
            ResetEvent(m_hQuiescent);
            m_bQuiescent = FALSE;
            while(m_vQ.size()) { 
                Datablock * pBlock = m_vQ.front();
                temp.insert(pBlock);
                m_vQ.pop_front();
            }
            Unlock();
            for(std::set<Datablock*>::iterator vi=temp.begin();
                vi!=temp.end();
                vi++)
            {
                Datablock * pBlock = (*vi);
                if(pBlock != NULL) {
                    ptgc_lock();
                    ptgc_check_double_free(pBlock);
                    ptgc_record_free(pBlock);
                    ptgc_unlock();
                    delete pBlock;
                }
            }
            temp.clear();
            Lock();
            m_bQuiescent = TRUE;
            SetEvent(m_hQuiescent);
            Unlock();
            MARKRANGEEXIT();
            if(bShutdown)
                break;
        }

        Accelerator::DeinitializeTLSContextManagement();
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Force GC. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GarbageCollector::ForceGC(
        VOID
        )
    {
        if(g_pGarbageCollector) 
            g_pGarbageCollector->_ForceGC();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Force GC. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GarbageCollector::_ForceGC(
        VOID
        )
    {
        MARKRANGEENTER(L"PTask-Force-GC");
        std::set<Datablock*> temp;
        Lock();
        while(m_vQ.size()) { 
            Datablock * pBlock = m_vQ.front();
            temp.insert(pBlock);
            m_vQ.pop_front();
        }
        Unlock();
        for(std::set<Datablock*>::iterator vi=temp.begin();
            vi!=temp.end();
            vi++)
        {
            Datablock * pBlock = (*vi);
            if(pBlock != NULL) {
                ptgc_lock();
                ptgc_check_double_free(pBlock);
                ptgc_record_free(pBlock);
                ptgc_unlock();
                delete pBlock;
            }
        }
        HANDLE hWaitEvents[] = { m_hGCShutdown, m_hQuiescent };
        DWORD dwWaitEvents = sizeof(hWaitEvents)/sizeof(HANDLE);
        DWORD dwWait = WaitForMultipleObjects(dwWaitEvents, hWaitEvents, FALSE, INFINITE);
        switch(dwWait) {
        case WAIT_OBJECT_0: break;
        case WAIT_OBJECT_0 + 1: 
            Lock();
            if(m_bQuiescent) {                
                ptgc_reset();
            }
            Unlock();
            break;
        default: 
            PTask::Runtime::Warning("failed wait in _ForceGC!");
            break;
        }
        MARKRANGEEXIT();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a GC. </summary>
    ///
    /// <remarks>   Crossbac, 3/18/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GarbageCollector::CreateGC(
        VOID
        ) 
    {        
        assert(g_pGarbageCollector == NULL);
        // never de-init the lock: leak it so we can deal with racy dandelion code
        InitializeCriticalSection(&m_csGlobalGCPtr);
        EnterCriticalSection(&m_csGlobalGCPtr);
        g_pGarbageCollector = new GarbageCollector();
        LeaveCriticalSection(&m_csGlobalGCPtr);        
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destroys the GC. </summary>
    ///
    /// <remarks>   Crossbac, 3/18/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    GarbageCollector::DestroyGC(
        VOID
        ) 
    {
        EnterCriticalSection(&m_csGlobalGCPtr);
        assert(g_pGarbageCollector);
        if(g_pGarbageCollector) {
            g_pGarbageCollector->Shutdown();
            delete g_pGarbageCollector;
            g_pGarbageCollector = NULL;
        }
        LeaveCriticalSection(&m_csGlobalGCPtr);
    }

#ifdef DEBUG
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notifies an allocation. </summary>
    ///
    /// <remarks>   Crossbac, 7/1/2013. </remarks>
    ///
    /// <param name="pNewBlock">    [in,out] If non-null, the new block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    GarbageCollector::NotifyAllocation(
        Datablock * pNewBlock
        )
    {
        g_pGarbageCollector->__NotifyAllocation(pNewBlock);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notifies an allocation. </summary>
    ///
    /// <remarks>   Crossbac, 7/1/2013. </remarks>
    ///
    /// <param name="pNewBlock">    [in,out] If non-null, the new block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    GarbageCollector::__NotifyAllocation(
        Datablock * pNewBlock
        )
    {
        ptgc_lock();
        if(m_vDeleted.find(pNewBlock) != m_vDeleted.end()) 
            m_vDeleted.erase(pNewBlock); 
        ptgc_unlock();
    }
#endif
            

};
