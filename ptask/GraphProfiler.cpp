///-------------------------------------------------------------------------------------------------
// file:	GraphProfiler.cpp
//
// summary:	Implements the graph profiler class
///-------------------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "graph.h"
#include "GraphProfiler.h"
#include "task.h"
#include <iostream>
#include <assert.h>
using namespace std;
using namespace PTask::Runtime;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    ///-------------------------------------------------------------------------------------------------

    GraphProfiler::GraphProfiler(
        __in Graph * pGraph
        )
    {
        m_pGraph = pGraph;
        Initialize();
    }

    GraphProfiler::~GraphProfiler(
        VOID
        )
    {
        Destroy();
        m_pGraph = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the ad hoc graph statistics. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    GraphProfiler::Initialize(
        VOID
        )
    {
        InitializeCriticalSection(&m_csGraphStats);
        m_uiMaxConcurrentInflightThreads = 0;
        m_uiMaxConcurrentInflightDispatches = 0;
        m_uiMaxTaskQueueOccupancy = 0;
        m_uiMinConcurrentInflightThreads = MAXDWORD;
        m_uiMinConcurrentInflightDispatches = MAXDWORD;
        m_uiMinTaskQueueOccupancy = MAXDWORD;
        m_uiAliveThreads = 0;
        m_uiAwakeThreads = 0;
        m_uiBlockedRunningThreads = 0;
        m_uiBlockedTaskAvailableThreads = 0;
        m_uiExitedThreads = 0;
        m_uiInflightThreads = 0;
        m_uiInflightDispatchAttempts = 0;
        m_uiInflightThreadUpdates = 0;
        m_uiInflightDispatchUpdates = 0;
        m_uiDispatchAttempts = 0;
        m_uiSuccessfulDispatchAttempts = 0;
        m_uiDequeueAttempts = 0;
        m_uiSuccessfulDequeueAttempts = 0;
        m_uiTaskQueueOccupancyAccumulator = 0;
        m_uiTaskQueueSamples = 0;
        m_uiConcurrentInflightThreadAccumulator = 0;
        m_uiConcurrentInflightDispatchAccumulator = 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Print graph statistics. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    GraphProfiler::Report(
        __inout std::ostream& ss
        )
    {
        EnterCriticalSection(&m_csGraphStats);
        double dAvgQueueOcc = m_uiTaskQueueSamples ? 
            ((double)m_uiTaskQueueOccupancyAccumulator/(double)m_uiTaskQueueSamples) : 0.0;
        double dAvgInflightDispatch = m_uiInflightDispatchUpdates ?
            ((double)m_uiConcurrentInflightDispatchAccumulator/(double)m_uiInflightDispatchUpdates) : 0.0;
        double dAvgInflightThreads = m_uiInflightThreadUpdates ? 
            ((double)m_uiConcurrentInflightThreadAccumulator/(double)m_uiInflightThreadUpdates) : 0.0;
        UINT uiMinTQOcc = m_uiMinTaskQueueOccupancy == MAXDWORD ? 0 : m_uiMinTaskQueueOccupancy;
        UINT uiMinConc = m_uiMinConcurrentInflightDispatches == MAXDWORD ? 0 : m_uiMinConcurrentInflightDispatches;
        UINT uiMinThreads = m_uiMinConcurrentInflightThreads == MAXDWORD ? 0 : m_uiMinConcurrentInflightThreads;

        ss   << "Graph " << m_pGraph->m_lpszGraphName << " stats:" << std::endl;
        ss   << "task queue occupancy: [" 
             << uiMinTQOcc << ".."  
             << m_uiMaxTaskQueueOccupancy << "] avg = " 
             << dAvgQueueOcc << " over "
             << m_uiTaskQueueSamples << " samples."
             << std::endl
             << m_uiSuccessfulDequeueAttempts << " of " << m_uiDequeueAttempts << " dispatch attempts succeeded."
             << std::endl;
        ss   << "dispatch: " 
             << m_uiSuccessfulDispatchAttempts << " of " 
             << m_uiDispatchAttempts << " dispatch attempts successful." << std::endl << "  dispatch concurrency: "
             << uiMinConc << ".." << m_uiMaxConcurrentInflightDispatches 
             << "] avg = " << dAvgInflightDispatch << " over " << m_uiInflightDispatchUpdates << " samples."
             << std::endl;
        ss   << "thread pool stats: ["
             << uiMinThreads << ".." << m_uiMaxConcurrentInflightThreads
             << "] avg = " << dAvgInflightThreads << " over " << m_uiInflightThreadUpdates << " samples."
             << std::endl;
        LeaveCriticalSection(&m_csGraphStats);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the ad hoc graph statistics. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    GraphProfiler::Destroy(
        VOID
        )
    {
        DeleteCriticalSection(&m_csGraphStats);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the task thread alive action. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GraphProfiler::OnTaskThreadAlive(
        VOID
        )
    {
        EnterCriticalSection(&m_csGraphStats);
        m_uiAliveThreads++;
        m_uiAwakeThreads++;
        LeaveCriticalSection(&m_csGraphStats);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the task thread exit action. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GraphProfiler::OnTaskThreadExit(
        VOID
        )
    {
        EnterCriticalSection(&m_csGraphStats);
        m_uiAliveThreads--;
        m_uiAwakeThreads--;
        assert(m_uiAliveThreads >= 0);
        assert(m_uiAwakeThreads >= 0);
        LeaveCriticalSection(&m_csGraphStats);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the task thread block running graph action. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GraphProfiler::OnTaskThreadBlockRunningGraph(
        VOID
        )
    {
        EnterCriticalSection(&m_csGraphStats);
        assert(m_uiAwakeThreads <= m_uiAliveThreads);
        assert(m_uiAwakeThreads > 0);
        assert(m_uiBlockedRunningThreads < m_uiAliveThreads);
        m_uiAwakeThreads--;
        m_uiBlockedRunningThreads++;
        LeaveCriticalSection(&m_csGraphStats);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the task thread wake running graph action. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GraphProfiler::OnTaskThreadWakeRunningGraph(
        VOID
        )
    {
        EnterCriticalSection(&m_csGraphStats);
        assert(m_uiBlockedRunningThreads > 0);
        m_uiAwakeThreads++;
        m_uiBlockedRunningThreads--;
        assert(m_uiAwakeThreads <= m_uiAliveThreads);
        LeaveCriticalSection(&m_csGraphStats);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the task thread block tasks available action. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GraphProfiler::OnTaskThreadBlockTasksAvailable(
        VOID
        )
    {
        EnterCriticalSection(&m_csGraphStats);
        assert(m_uiAwakeThreads <= m_uiAliveThreads);
        assert(m_uiAwakeThreads > 0);
        assert(m_uiBlockedTaskAvailableThreads < m_uiAliveThreads);
        m_uiAwakeThreads--;
        m_uiBlockedTaskAvailableThreads++;
        LeaveCriticalSection(&m_csGraphStats);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the task thread wake tasks available action. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GraphProfiler::OnTaskThreadWakeTasksAvailable(
        VOID
        )
    {
        EnterCriticalSection(&m_csGraphStats);
        assert(m_uiBlockedTaskAvailableThreads > 0);
        m_uiAwakeThreads++;
        m_uiBlockedTaskAvailableThreads--;
        assert(m_uiAwakeThreads <= m_uiAliveThreads);
        LeaveCriticalSection(&m_csGraphStats);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the task thread dequeue attempt action. The thread is about to 
    ///             attempt to dequeue a task. Update the inflight watermarks. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GraphProfiler::OnTaskThreadDequeueAttempt(
        VOID
        )
    {
        EnterCriticalSection(&m_csGraphStats);
        m_uiInflightThreads++;
        m_uiInflightThreadUpdates++;
        m_uiDequeueAttempts++;
        m_uiConcurrentInflightThreadAccumulator += m_uiInflightThreads;
        assert(m_uiInflightThreads <= m_uiAliveThreads);
        assert(m_uiInflightThreads <= m_uiAwakeThreads);
        EnterCriticalSection(&m_pGraph->m_csReadyQ);
        m_uiTaskQueueSamples++;
        m_uiTaskQueueOccupancyAccumulator += (UINT)m_pGraph->m_vReadyQ.size();
        m_uiMinTaskQueueOccupancy = min((UINT)m_pGraph->m_vReadyQ.size(), m_uiMinTaskQueueOccupancy);
        m_uiMaxTaskQueueOccupancy = max((UINT)m_pGraph->m_vReadyQ.size(), m_uiMaxTaskQueueOccupancy);
        LeaveCriticalSection(&m_pGraph->m_csReadyQ);
        m_uiMinConcurrentInflightThreads = min(m_uiMinConcurrentInflightThreads, m_uiInflightThreads);
        m_uiMaxConcurrentInflightThreads = max(m_uiMaxConcurrentInflightThreads, m_uiInflightThreads);
        LeaveCriticalSection(&m_csGraphStats);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the task thread dequeue complete action. If we didn't get 
    ///             a task, the thread moves to the non "inflight" state. Otherwise, update
    ///             the number of successful dispatch attempts. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    GraphProfiler::OnTaskThreadDequeueComplete(
        __in Task * pTask
        )
    {
        EnterCriticalSection(&m_csGraphStats);
        assert(m_uiInflightThreads > 0); // this one
        assert(m_uiInflightThreads <= m_uiAliveThreads);
        assert(m_uiInflightThreads <= m_uiAwakeThreads);
        assert(m_uiBlockedTaskAvailableThreads < m_uiAliveThreads);
        if(pTask != NULL) {
            m_uiSuccessfulDequeueAttempts++;
        } else {
            assert(m_uiInflightThreads > 0);
            m_uiInflightThreads--;
        }
        LeaveCriticalSection(&m_csGraphStats);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the task thread dispatch attempt action. increment the inflight
    ///             dispatch count, do basic bounds checking, and update high water marks on
    ///             concurrent inflight dispatches.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    GraphProfiler::OnTaskThreadDispatchAttempt(
        VOID
        )
    {
        EnterCriticalSection(&m_csGraphStats);
        m_uiDispatchAttempts++;
        m_uiInflightDispatchAttempts++;
        m_uiInflightDispatchUpdates++;
        m_uiConcurrentInflightDispatchAccumulator += m_uiInflightDispatchAttempts;
        assert(m_uiInflightThreads >= m_uiInflightDispatchAttempts);
        assert(m_uiInflightDispatchAttempts <= m_uiAliveThreads);
        assert(m_uiInflightDispatchAttempts <= m_uiAwakeThreads);
        m_uiMinConcurrentInflightDispatches = min(m_uiMinConcurrentInflightDispatches, m_uiInflightDispatchAttempts);
        m_uiMaxConcurrentInflightDispatches = max(m_uiMaxConcurrentInflightDispatches, m_uiInflightDispatchAttempts);
        LeaveCriticalSection(&m_csGraphStats);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the task thread dispatch complete action. If we succeeded,
    ///             increment the number of successful dispatches. Decrement the inflight
    ///             dispatch count, and the inflight count. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///
    /// <param name="bSuccess"> true if the operation was a success, false if it failed. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    GraphProfiler::OnTaskThreadDispatchComplete(
        BOOL bSuccess
        )
    {
        EnterCriticalSection(&m_csGraphStats);
        assert(m_uiInflightDispatchAttempts > 0);
        m_uiInflightDispatchAttempts--;
        m_uiSuccessfulDispatchAttempts += (bSuccess?1:0);
        assert(m_uiInflightThreads > 0);
        m_uiInflightThreads--;
        LeaveCriticalSection(&m_csGraphStats);
    }


};