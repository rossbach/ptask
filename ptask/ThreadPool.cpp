///-------------------------------------------------------------------------------------------------
// file:	ThreadPool.cpp
//
// summary:	Implements the thread pool class
///-------------------------------------------------------------------------------------------------

#include <stdio.h>
#include <crtdbg.h>
#include "ThreadPool.h"
#include "ptaskutils.h"
#include "PTaskRuntime.h"
#include <assert.h>
#include <vector>

#ifdef CUDA_SUPPORT
#include <cuda_runtime.h>
#define PRIME_GRAPH_RUNNER_THREAD()		cudaFree(0);
#else
#define PRIME_GRAPH_RUNNER_THREAD()
#endif

namespace PTask {

    ThreadPool * ThreadPool::g_pThreadPool = NULL;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates this object. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///
    /// <param name="ui">   The user interface. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    ThreadPool * 
    ThreadPool::Create(
        __in UINT ui,
        __in BOOL bPrimeThreads,
        __in BOOL bGrowable,
        __in UINT uiGrowIncrement
        ) { 
        if(!g_pThreadPool) {
            g_pThreadPool = new ThreadPool(ui, bPrimeThreads, bGrowable, uiGrowIncrement);
            g_pThreadPool->StartThreads(ui, TRUE);
        }
        return g_pThreadPool;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destroys this object. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    ThreadPool::Destroy(
        void
        )
    {
        if(g_pThreadPool) {
            g_pThreadPool->Lock();
            if(g_pThreadPool) {
                ThreadPool * pPool = g_pThreadPool;
                g_pThreadPool = NULL;
                pPool->Unlock();
                Sleep(100);
                delete pPool;
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="nThreads"> If non-null, the p. </param>
    ///-------------------------------------------------------------------------------------------------

    ThreadPool::ThreadPool(
        __in UINT nThreads,
        __in BOOL bPrimeThreads,
        __in BOOL bGrowable,
        __in UINT uiGrowIncrement
        ) : Lockable("ThreadPool")
    {
        m_uiThreadsAlive = 0;
        m_uiThreads = nThreads;
        m_bPrimeThreads = bPrimeThreads;
        m_bGrowable = bGrowable;
        m_uiGrowIncrement = uiGrowIncrement;
        m_uiTargetSize = m_uiThreads;
        m_hAllThreadsAlive = CreateEvent(NULL, TRUE, FALSE, NULL);
        m_hAllThreadsExited = CreateEvent(NULL, TRUE, TRUE, NULL);
        m_uiAliveWaiters = 0;
        m_uiExitWaiters = 0;
        m_bExiting = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    ThreadPool::~ThreadPool(
        void
        )
    {
        Lock();
        m_bExiting = TRUE;
        if(m_uiAliveWaiters)
            SetEvent(m_hAllThreadsAlive);

        std::set<THREADDESC*> vDelete;
        std::vector<HANDLE> vThreadHandles;
        std::vector<HANDLE>::iterator vi;
        std::set<THREADDESC*>::iterator si;
        std::map<HANDLE, THREADDESC*>::iterator mi;

        for(mi=m_vhThreadDescs.begin(); mi!=m_vhThreadDescs.end(); mi++) {

            THREADDESC* pDesc = mi->second;
            vThreadHandles.push_back(pDesc->hThread); 

            if(m_vZombieThreadDescs.find(pDesc) != m_vZombieThreadDescs.end()) {

                // the challenge here is determining whether we are cleaning
                // up these data structures before or after the thread has
                // already exited. if the desc is on the zombie list, then
                // its thread thread has already left the building. we still need
                // to delete it, but we don't have to try to wake up its thread
                // to get into a state where can free the resource. 
                
                vDelete.insert(pDesc);

            } else {

                // this thread descriptor is not on the zombie list, which
                // means we can't make any assumptions about what state it is in.
                // signal its start event so that if the thread it describes is blocked,
                // it will wake up, see that we have terminated the thread pool,
                // and exit the thread. We mark the descriptor to be deleted by
                // the thread proc exit code, and that we've removed from the live descs.

                pDesc->Lock();
                pDesc->bActive = FALSE;
                pDesc->bTerminate = TRUE;
                pDesc->bRemoveFromPoolOnThreadExit = FALSE;
                pDesc->bDeleteOnThreadExit = TRUE;
                pDesc->Unlock();
                SetEvent(pDesc->hStartEvent);            
            }

        }

        m_vhThreadDescs.clear();
        m_vhAvailable.clear();
        m_vhInFlight.clear();
        Unlock();

        DWORD dwWait = WaitForMultipleObjects((DWORD) vThreadHandles.size(), 
                                              vThreadHandles.data(),
                                              TRUE,
                                              INFINITE); 
        if(dwWait == WAIT_FAILED ||
           (dwWait >= WAIT_ABANDONED_0 && 
           dwWait < WAIT_ABANDONED_0 + vThreadHandles.size())) {
            PTask::Runtime::MandatoryInform("%s::%s abnormal wait result on thread pool exit!\n",
                                            __FILE__,
                                            __FUNCTION__);
        }

        Lock();
        CloseHandle(m_hAllThreadsAlive);
        CloseHandle(m_hAllThreadsExited);
        for(vi=vThreadHandles.begin(); vi!=vThreadHandles.end(); vi++)
            CloseHandle(*vi);
        for(si=m_vZombieThreadDescs.begin(); si!= m_vZombieThreadDescs.end(); si++)
            vDelete.insert(*si);
        for(si=vDelete.begin(); si!= vDelete.end(); si++)
            delete (*si);
        m_vZombieThreadDescs.clear();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets pool size. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   The pool size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    ThreadPool::GetPoolSize(
        void
        )
    {
        return m_uiThreads;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets pool size. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///
    /// <param name="uiThreads">    The threads. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    ThreadPool::SetPoolSize(
        __in UINT uiThreads
        )
    {
        Lock();
        if(uiThreads < m_uiTargetSize) {
            m_uiTargetSize = uiThreads;
        } else if(uiThreads > m_uiTargetSize && m_bGrowable) {
            UINT uiNewThreads = uiThreads - (UINT)m_vhThreadDescs.size();
            m_uiTargetSize = uiThreads;
            StartThreads(uiNewThreads, FALSE);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Request thread. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///
    /// <param name="lpRoutine">    The routine. </param>
    /// <param name="lpParameter">  The parameter. </param>
    /// <param name="bStartThread"> true if the thread can be signaled to start 
    ///                             before returning from this call, false if the
    ///                             caller would prefer to signal it explicitly. </param>
    ///
    /// <returns>   The handle of the thread. </returns>
    ///-------------------------------------------------------------------------------------------------

    HANDLE 
    ThreadPool::RequestThread(
        __in LPTHREAD_START_ROUTINE lpRoutine, 
        __in LPVOID lpParameter,
        __in BOOL bStartThread
        )
    {
        if(g_pThreadPool) 
            return g_pThreadPool->GetThread(lpRoutine, lpParameter, bStartThread);
        return INVALID_HANDLE_VALUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Starts a thread: if a previous call to RequestThread was made with 
    ///             the bStartThread parameter set to false, this API signals the thread
    ///             to begin. Otherwise, the call has no effect (returns FALSE). </summary>
    ///
    /// <remarks>   crossbac, 8/29/2013. </remarks>
    ///
    /// <param name="hThread">  The thread. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    ThreadPool::StartThread(
        __in HANDLE hThread
        )
    {
        if(g_pThreadPool) 
            return g_pThreadPool->SignalThread(hThread);
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Request thread. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///
    /// <param name="lpRoutine">    The routine. </param>
    /// <param name="lpParameter">  The parameter. </param>
    /// <param name="bStartThread"> The start thread. </param>
    ///
    /// <returns>   The handle of the. </returns>
    ///-------------------------------------------------------------------------------------------------

    HANDLE 
    ThreadPool::GetThread(
        __in LPTHREAD_START_ROUTINE lpRoutine, 
        __in LPVOID lpParameter,
        __in BOOL bStartThread
        )
    {
        HANDLE hThread = INVALID_HANDLE_VALUE;
        Lock();
        if(m_vhAvailable.size() == 0 && m_bGrowable) {
            m_uiTargetSize = m_uiThreads+m_uiGrowIncrement;
            Unlock();
            StartThreads(m_uiGrowIncrement, TRUE);
            Lock();
        }
        if(m_vhAvailable.size() != 0) {
            hThread = m_vhAvailable.front();
            THREADDESC * pDesc = m_vhThreadDescs[hThread];
            pDesc->Lock();
            HANDLE hEvent = pDesc->hStartEvent;
            pDesc->lpRoutine = lpRoutine;
            pDesc->lpParameter = lpParameter;
            pDesc->bActive = TRUE;
            pDesc->bRoutineValid = TRUE;
            pDesc->Unlock();
            m_vhAvailable.pop_front();
            m_vhInFlight.insert(hThread);
            if(bStartThread) {
                SetEvent(hEvent);
            } else {
                m_vhWaitingStartSignal.insert(hThread);
            }
        }
        Unlock();
        return hThread;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Starts a thread: if a previous call to RequestThread was made with 
    ///             the bStartThread parameter set to false, this API signals the thread
    ///             to begin. Otherwise, the call has no effect (returns FALSE). </summary>
    ///
    /// <remarks>   crossbac, 8/29/2013. </remarks>
    ///
    /// <param name="hThread">  The thread. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    ThreadPool::SignalThread(
        __in HANDLE hThread
        )
    {
        Lock();
        BOOL bResult = FALSE;
        std::set<HANDLE>::iterator si = m_vhWaitingStartSignal.find(hThread);
        if(si!=m_vhWaitingStartSignal.end()) {
            m_vhWaitingStartSignal.erase(hThread);
            THREADDESC * pDesc = m_vhThreadDescs[hThread];
            HANDLE hEvent = pDesc->hStartEvent;
            SetEvent(hEvent);
            bResult = TRUE;
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Thread pool proc. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pVoidCastGraph">   the graph object, typecast to void* </param>
    ///
    /// <returns>   DWORD: 0 on thread exit. </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD WINAPI 
    ThreadPool::_ThreadPoolProc(
        LPVOID lpvDesc
        )
    {
        THREADDESC * pDesc = reinterpret_cast<THREADDESC*>(lpvDesc);
        ThreadPool * pPool = pDesc->pThreadPool;
        return pPool->ThreadPoolProc(pDesc);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Prime thread. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    ThreadPool::PrimeThread(
        void
        )
    {
        if(m_bPrimeThreads) {
            // we started this thread.
            // make the CUDA context do its thing
			// PRIME_GRAPH_RUNNER_THREAD();
        }
    }
        

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Thread pool proc. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///
    /// <param name="pDesc">    The description. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD 
    ThreadPool::ThreadPoolProc(
        __in THREADDESC * pDesc
        )
    {
        HANDLE hThread = pDesc->hThread;
        HANDLE hStartEvent = pDesc->hStartEvent;
        HANDLE hRuntimeTerminate = PTask::Runtime::GetRuntimeTerminateEvent();
        HANDLE vEvents[] = { hStartEvent, hRuntimeTerminate };
        DWORD dwEvents = sizeof(vEvents)/sizeof(HANDLE);
        DWORD dwResult = 0;
        PrimeThread();
        NotifyThreadAlive(hThread);
        while(!pDesc->bTerminate) {
            BOOL bTerminate = FALSE;
            DWORD dwWait = WaitForMultipleObjects(dwEvents, vEvents, FALSE, INFINITE);
            switch(dwWait) {
            case WAIT_OBJECT_0: 
                if(pDesc->bTerminate) 
                    bTerminate = TRUE;
                break;
            case WAIT_OBJECT_0+1:
            default:
                bTerminate = TRUE;
                break;
            }
            pDesc->Lock();
            pDesc->bTerminate |= bTerminate;
            if(pDesc->bRoutineValid && !pDesc->bTerminate) {
                LPTHREAD_START_ROUTINE lpRoutine = pDesc->lpRoutine;
                LPVOID lpParameter = pDesc->lpParameter;
                pDesc->bActive = TRUE;
                pDesc->Unlock();
                dwResult = (*lpRoutine)(lpParameter);
                pDesc->Lock();
                pDesc->bActive = FALSE;
                pDesc->bRoutineValid = FALSE;
            }
            pDesc->Unlock();
            Lock();
            m_vhInFlight.erase(pDesc->hThread);
            if(!pDesc->bTerminate) 
                m_vhAvailable.push_back(pDesc->hThread);            
            Unlock();
        }
        Lock();
        if(pDesc->bRemoveFromPoolOnThreadExit) {
            m_vhThreadDescs.erase(pDesc->hThread);        
        } else {
            assert(m_vhThreadDescs.find(pDesc) == m_vhThreadDescs.end());
        }
        if(!pDesc->bDeleteOnThreadExit) {
            m_vZombieThreadDescs.insert(pDesc);
        }
        Unlock();
        NotifyThreadExit(hThread);
        if(pDesc->bDeleteOnThreadExit) {
            assert(pDesc->bRemoveFromPoolOnThreadExit);
            CloseHandle(pDesc->hStartEvent);
            if(pDesc->bRemoveFromPoolOnThreadExit)
                CloseHandle(pDesc->hThread);
            delete pDesc;
        } 
        return dwResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Starts the threads. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///
    /// <param name="uiThreads">            The threads. </param>
    /// <param name="bWaitAllThreadsAlive"> The wait all threads alive. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    ThreadPool::StartThreads(
        __in UINT uiThreads,
        __in BOOL bWaitAllThreadsAlive
        )
    {
        Lock();
        if(uiThreads != 0 && m_vhThreadDescs.size() < m_uiTargetSize) 
            ResetEvent(m_hAllThreadsAlive);
        while(m_vhThreadDescs.size() < m_uiTargetSize) {
            for(UINT i=0; i<uiThreads; i++) {
                THREADDESC* pDesc = new THREADDESC(this);
                HANDLE * phThread = &pDesc->hThread;
                *phThread = CreateThread(NULL, 0, _ThreadPoolProc, pDesc, 0, NULL);
                m_vhAvailable.push_back(*phThread);
                m_vhThreadDescs[*phThread] = pDesc;
            }
        }
        m_uiThreads = (UINT)m_vhThreadDescs.size();
        Unlock();
        if(bWaitAllThreadsAlive) 
            WaitThreadsAlive();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets current pool size. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///
    /// <returns>   The current pool size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    ThreadPool::GetCurrentPoolSize(
        void
        )
    {
        Lock();
        UINT uiSizeSnapshot = (UINT)m_vhThreadDescs.size();
        Unlock();
        return uiSizeSnapshot;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets current pool size. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///
    /// <returns>   The current pool size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    ThreadPool::GetTargetPoolSize(
        void
        )
    {       
        return m_uiTargetSize;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets current pool size. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///
    /// <returns>   The current pool size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    ThreadPool::GetGrowIncrement(
        void
        )
    {
        return m_uiGrowIncrement;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets current pool size. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///
    /// <returns>   The current pool size. </returns>
    ///-------------------------------------------------------------------------------------------------

    void
    ThreadPool::SetGrowIncrement(
        UINT uiGrowIncrement
        )
    {
        Lock();
        m_uiGrowIncrement = uiGrowIncrement;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notifies a thread alive. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///
    /// <param name="hThread">  Handle of the thread. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    ThreadPool::NotifyThreadAlive(
        HANDLE hThread
        )
    {
        Lock();
        UNREFERENCED_PARAMETER(hThread);
        ResetEvent(m_hAllThreadsExited);
        InterlockedIncrement(&m_uiThreadsAlive);
        if(m_uiThreadsAlive == m_uiTargetSize) 
            SetEvent(m_hAllThreadsAlive);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Notifies a thread exit. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///
    /// <param name="hThread">  Handle of the thread. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    ThreadPool::NotifyThreadExit(
        HANDLE hThread
        )
    {
        Lock();
        UNREFERENCED_PARAMETER(hThread);
        assert(m_uiThreadsAlive != 0);
        InterlockedDecrement(&m_uiThreadsAlive);
        if(m_uiThreadsAlive == 0) 
            SetEvent(m_hAllThreadsExited);
        if(m_uiThreadsAlive == m_uiTargetSize) 
            SetEvent(m_hAllThreadsAlive);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Wait threads alive. </summary>
    ///
    /// <remarks>   crossbac, 8/22/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    ThreadPool::WaitThreadsAlive(
        void
        )
    {
        Lock();
        HANDLE vHandles[] = {
            m_hAllThreadsAlive,
            Runtime::GetRuntimeTerminateEvent() 
        };
        DWORD dwHandles = sizeof(vHandles)/sizeof(DWORD);
        InterlockedIncrement(&m_uiAliveWaiters);
        Unlock();
        WaitForMultipleObjects(dwHandles, vHandles, FALSE, INFINITE);
    }


};
