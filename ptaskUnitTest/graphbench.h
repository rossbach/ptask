//--------------------------------------------------------------------------------------
// File: graphbench.h
// Benchmark PTask peformance via new graph-based API.
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_BENCH_H_
#define _GRAPH_BENCH_H_

#include "ptaskapi.h"
#include "confighelpers.h"
#include "shaderparms.h"

using namespace std;
using namespace PTask;

class BenchGraphCoordinator;

class BenchGraph {
public:

    // Different graph types that can be benchmarked.
    typedef enum GraphType_t {
        RECT,   // A rectangular graph.
                // Bredth b is the number of independent chains of tasks.
                // Each chain has depth d tasks.
        BINTREE // A binary tree, fat at the top and bredth 1 at the bottom.
                // Graph has depth d.
                // Bredth is different at each depth. See binTreeBredthAtDepth()
    } GraphType;

    // For BINTREE, each depth of the graph has bredth 2 ** (d - (c+1)),
    // where d = the (one-based) graph depth, and c is the (zero-based) current depth.
    int binTreeBredthAtDepth(int graphDepth, int currentDepth);
    int binTreeTotalNumTasks(int graphDepth);

    BenchGraph(
        int graphId,
        BenchGraphCoordinator* coordinator,
        GraphType type,
        char * szfile,
        char * szshader,
        int graphDepth,
        int graphBredth,
        int rows,
        int cols,
        int ptaskPriority,
        int threadPriority,
        bool emulateModular,
        bool steadyStateBenchmark,
        int verbosity);
    ~BenchGraph();

    Graph* buildGraph(
        GraphType type,
        char * szfile,
        char * szshader,
        int graphDepth,
        int graphBredth,
        int ptaskPriority,
        int threadPriority,
        bool emulateModular);

    void resetTimer();

    void initializeBenchmark(int iterations, bool instrument, bool singleThreaded, int schedulingMode);

    int getNumTasks();
    int getMaxBredth();
    double getStartTime();
    double getEndTime();
    double getTotalTime();
    int getTotalIterationsProduced();
    int getTotalIterationsConsumed();

    void producer(bool runAllIterations);
    void consumer(bool runAllIterations);
    void STProducerConsumer();

    int getMaxOutstandingInvocations();

    void terminateProducerAndConsumer() { m_terminated = TRUE; }

private:
    int                 m_graphId;
    BenchGraphCoordinator* m_coordinator;
    HANDLE              m_producerThread;
    HANDLE              m_consumerThread;
    bool                m_steadyStateBenchmark;
    bool                m_terminated;
    Graph*              m_graph;
    int                 m_numTasks;
    int                 m_depth;
    int                 m_maxBredth;

    DatablockTemplate*  m_dataTemplate;
    DatablockTemplate*  m_parmTemplate;
    int                 m_rows;
    int                 m_cols;
    UINT                m_stride;
    UINT                m_elements;

    CSimpleMatrix<ELEMTYPE>* m_inputMatrix;
    CSimpleMatrix<ELEMTYPE>* m_outputMatrix;
    MATADD_PARAMS       m_matrixParams;

    CHighResolutionTimer * m_timer;
    double              m_startTime;
    double              m_endTime;

    int                 m_timingStartIteration;
    int                 m_timingStopIteration;
    int                 m_currentProducerIteration;
    int                 m_currentConsumerIteration;
    int                 m_numInputDataChannels;
    int                 m_numInputParmChannels;
    int                 m_numOutputChannels;
    GraphInputChannel** m_inputDataChannels;
    GraphInputChannel** m_inputParmChannels;
    GraphOutputChannel** m_outputChannels;

    int                 m_verbosity;

    // Instrumentation is only used if m_instrument is TRUE, 
    // since it introduces locking around state shared between the producer and consumer.
    bool                m_instrument;
    int                 m_pushed;
    int                 m_pulled;
    CRITICAL_SECTION    m_csProducerConsumerState;
    // State below this is shared.
    int                 m_outstandingInvocations;
    int                 m_maxOutstandingInvocations;
};

class BenchGraphCoordinator {
public:
    BenchGraphCoordinator()
    {
        WCHAR lpwszProducersStartEventName[50] = L"ProducersStartEvent";
	    m_ProducersStartEvent = CreateEvent(NULL, TRUE, FALSE, lpwszProducersStartEventName);
        WCHAR lpwszConsumersFinishedEventName[50] = L"ConsumersFinishedEvent";
	    m_AllConsumersFinishedEvent = CreateEvent(NULL, TRUE, FALSE, lpwszConsumersFinishedEventName);
        InitializeCriticalSection(&m_lock);
    }

    ~BenchGraphCoordinator()
    {
        CloseHandle(m_ProducersStartEvent);
        CloseHandle(m_AllConsumersFinishedEvent);
    }

    void reset(int numGraphs)
    {
        ResetEvent(m_ProducersStartEvent);
        ResetEvent(m_AllConsumersFinishedEvent);
    	EnterCriticalSection(&m_lock);
        m_graphsRemaining = numGraphs;
	    LeaveCriticalSection(&m_lock);
    }

    void waitForProducersStartSignal()
    {
        DWORD dwWait = WaitForSingleObject(m_ProducersStartEvent, INFINITE);
    }

    void signalProducersToStart()
    {
        SetEvent(m_ProducersStartEvent);
    }

    void waitForAllConsumersToFinish()
    {
        DWORD dwWait = WaitForSingleObject(m_AllConsumersFinishedEvent, INFINITE);
    }

    void consumerFinished()
    {
    	EnterCriticalSection(&m_lock);
        if (--m_graphsRemaining == 0)
        {
        	SetEvent(m_AllConsumersFinishedEvent);
        }
	    LeaveCriticalSection(&m_lock);
    }

private:
    HANDLE      m_ProducersStartEvent;
    HANDLE      m_AllConsumersFinishedEvent;
    int         m_graphsRemaining;
    CRITICAL_SECTION m_lock;
};

int run_graph_bench_task(
    char * szfile,
    char * szshader,
    int depthMin,
    int depthStep,
    int depthMax,
    int bredthMin,
    int bredthMax,
    int bredthStep,
    int dimensionMin,
    int dimensionMax,
    int iterations,
    int nInstances,
    int* ptaskPriorities,
    int* threadPriorities,
    int schedulingMode,
    int verbosity,
    bool instrument,
    int gtype
    );
#endif
