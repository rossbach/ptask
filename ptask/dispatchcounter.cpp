///-------------------------------------------------------------------------------------------------
// file:	dispatchcounter.cpp
//
// summary:	Implements the dispatchcounter class
///-------------------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "task.h"
#include "DispatchCounter.h"
#include "hrperft.h"
#include "shrperft.h"
#include <map>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <assert.h>
using namespace std;

namespace PTask {


    /// <summary>   The lock for the task dispatch map. </summary>
    CRITICAL_SECTION            DispatchCounter::m_csDispatchMap;

    /// <summary>   The task dispatch map. </summary>
    std::map<std::string, UINT> DispatchCounter::m_vDispatchMap;

    /// <summary>   true if dispatch counter m b dispatch counting initialized. </summary>
    BOOL                        DispatchCounter::m_bDispatchCountingInitialized = FALSE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initialises the dispatch counting. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    DispatchCounter::Initialize(
        VOID
        )
    {
        if(PTask::Runtime::GetInvocationCountingEnabled()) return;
        if(!m_bDispatchCountingInitialized) return;
        InitializeCriticalSection(&m_csDispatchMap);
        m_bDispatchCountingInitialized = TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deinitialises the dispatch counting. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    DispatchCounter::Deinitialize(
        VOID
        )
    {
        if(PTask::Runtime::GetInvocationCountingEnabled()) return;
        if(!m_bDispatchCountingInitialized) return;
        DeleteCriticalSection(&m_csDispatchMap);
        m_bDispatchCountingInitialized = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps dispatch counts. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    DispatchCounter::Report(
        std::ostream& ss
        )
    {
        if(PTask::Runtime::GetInvocationCountingEnabled()) return;
        if(!m_bDispatchCountingInitialized) return;
        EnterCriticalSection(&m_csDispatchMap);
        map<std::string, UINT>::iterator mi;
        for(mi=m_vDispatchMap.begin(); mi!=m_vDispatchMap.end(); mi++) 
            ss << mi->first.c_str() << ", " << mi->second << std::endl;
        LeaveCriticalSection(&m_csDispatchMap);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps the dispatch counts for every task in the graph. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    DispatchCounter::Verify(
        std::map<std::string, UINT> * pvInvocationCounts
        )
    {
        BOOL bSuccess = TRUE;
        EnterCriticalSection(&m_csDispatchMap);
        map<std::string, UINT>::iterator mi;
        UINT nMismatches = 0;
        for(mi=m_vDispatchMap.begin(); mi!=m_vDispatchMap.end(); mi++) {
            map<std::string, UINT>::iterator pi = pvInvocationCounts->find(mi->first);
            if(pi!=pvInvocationCounts->end()) {
                if(pi->second != mi->second) {
                    bSuccess = FALSE;
                    nMismatches++;
                    std::cout 
                        << mi->first << ": expected " 
                        << pi->second << ", got " << mi->second 
                        << std::endl;
                }
            } else {
                bSuccess = FALSE;
                nMismatches++;
                std::cout << mi->first << ": got " << mi->second << ", but no corresponding key found" << std::endl;
            }
        }
        for(mi=pvInvocationCounts->begin(); mi!=pvInvocationCounts->end(); mi++) {
            map<std::string, UINT>::iterator pi = m_vDispatchMap.find(mi->first);
            if(pi==m_vDispatchMap.end()) {
                bSuccess = FALSE;
                nMismatches++;
                std::cout << mi->first << ": expected " << mi->second << ", but no corresponding key found" << std::endl;
            }
        }
        std::cout << "found " << nMismatches << " dispatch discrepancies" << std::endl;
        LeaveCriticalSection(&m_csDispatchMap);    
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/18/2013. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    DispatchCounter::DispatchCounter(
        Task * pTask
        )
    {
        m_pTask = pTask;
        m_nActualDispatches = 0;
        m_nExpectedDispatches = 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/18/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    DispatchCounter::~DispatchCounter(
        VOID
        )
    {
        m_pTask = NULL;
        m_nActualDispatches = 0;
        m_nExpectedDispatches = 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    DispatchCounter::RecordDispatch(
        VOID
        )
    {
        EnterCriticalSection(&m_csDispatchMap);
        UINT uiCount = 0;
        std::string str(m_pTask->GetTaskName());
        map<std::string, UINT>::iterator mi = m_vDispatchMap.find(str);
        if(mi != m_vDispatchMap.end()) 
            uiCount = mi->second;
        m_vDispatchMap[str] = ++uiCount;
        if(m_nExpectedDispatches!= 0) {
            if(++m_nActualDispatches > m_nExpectedDispatches) {
#if 0
                // this is our opportunity to set break-points, ignore certain tasks, etc.
                if(!strncmp(pTask->m_lpszTaskName, "imgInput", strlen("imgInput"))) {
                    // nevermind these...
                } else {
                    if(!strcmp(pTask->m_lpszTaskName, "LaplacianX_Level_4") || !strcmp(pTask->m_lpszTaskName, "LaplacianY_Level_4")) {
                        assert(pTask->m_nActualDispatches <= pTask->m_nExpectedDispatches+10);
                    } else {
                        assert(pTask->m_nActualDispatches <= pTask->m_nExpectedDispatches);
                    }
                }
#endif
            }
        }
        LeaveCriticalSection(&m_csDispatchMap);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an expected dispatch count. </summary>
    ///
    /// <remarks>   Crossbac, 2/28/2012. </remarks>
    ///
    /// <param name="nCount">   Number of. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    DispatchCounter::SetExpectedDispatchCount(
        UINT nCount
        )
    {
        m_nExpectedDispatches = nCount;
    }

};

