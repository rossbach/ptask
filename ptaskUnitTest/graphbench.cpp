//--------------------------------------------------------------------------------------
// File: graphbench.cpp
//--------------------------------------------------------------------------------------
#include <stdio.h>
#include <crtdbg.h>
#include "accelerator.h"
#include "assert.h"
#include "sort.h"
#include <vector>
#include <algorithm>
#include "matrixtask.h"
#include "SimpleMatrix.h"
#include "matmul.h"
#include "graphmatmul.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "graphbench.h"
#include "hrperft.h"
#include "confighelpers.h"

//#define DEBUGLOG(x) printf(x)
//#define DEBUGLOG2(x,y) printf(x,y)
//#define DEBUGLOG3(x,y,z) printf(x,y,z)
#define DEBUGLOG(x)
#define DEBUGLOG2(x,y)
#define DEBUGLOG3(x,y,z)

#pragma warning(disable:4482)

// HACK: Yuck, a global. No time to refactor right now.
BenchGraphCoordinator* g_coordinator = NULL;

DWORD WINAPI 
    BenchGraphProducerThread(LPVOID p) 
{
    BenchGraph* graph = (BenchGraph*) p;
    graph->producer(TRUE);
    return 0;
}

DWORD WINAPI 
    BenchGraphConsumerThread(LPVOID p) 
{
    BenchGraph* graph = (BenchGraph*) p;
    graph->consumer(TRUE);
    return 0;
}

DWORD WINAPI 
    BenchGraphSTProducerConsumerThread(LPVOID p) 
{
    DEBUGLOG("\n** BenchGraphSTProducerConsumerThread starting\n");
    BenchGraph* graph = (BenchGraph*) p;
    graph->STProducerConsumer();
    DEBUGLOG("\n** BenchGraphSTProducerConsumerThread ending\n");
    return 0;
}

// For BINTREE, each depth of the graph has bredth 2 ** (d - (c+1)),
// where d = the (one-based) graph depth, and c is the (zero-based) current depth.
// d=1: c=0 -> 1
// d=2: c=0 -> 2, c=1 -> 1
// d=3: c=0 -> 4, c=1 -> 2, c=2 -> 1 etc.
int BenchGraph::binTreeBredthAtDepth(int graphDepth, int currentDepth)
{
    return (int)pow((double)2, (double)(graphDepth - (currentDepth + 1)));
}

int BenchGraph::binTreeTotalNumTasks(int graphDepth)
{
    int numTasks = 0;
    for (int c=0; c < graphDepth; c++)
    {
        numTasks += binTreeBredthAtDepth(graphDepth, c);
    }
    return numTasks;
}

BenchGraph::BenchGraph(
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
    int verbosity)
{
    assert(rows == cols);
    m_rows = rows;
    m_cols = cols;
    m_verbosity = verbosity;

    m_graphId = graphId;
    m_coordinator = coordinator;

    m_stride = sizeof(ELEMTYPE);
    m_elements = m_rows*m_cols;

    m_dataTemplate	= PTask::Runtime::GetDatablockTemplate("RxC_elem_matrix", m_stride, m_cols, m_rows, 1);
    m_parmTemplate	= PTask::Runtime::GetDatablockTemplate("mat_mul_parms", sizeof(MATADD_PARAMS), 1, 1, 1);

    m_graph = buildGraph(type, szfile, szshader, graphDepth, graphBredth, ptaskPriority, threadPriority, emulateModular);

    m_timer = new CHighResolutionTimer(gran_msec);

    InitializeCriticalSection(&m_csProducerConsumerState);
    m_producerThread = NULL;
    m_consumerThread = NULL;
    m_steadyStateBenchmark = steadyStateBenchmark;
}

BenchGraph::~BenchGraph()
{
    /*    delete [] taskInputPorts;
    delete [] taskOutputPorts;
    delete [] pAxBxCInputPorts;
    delete [] pAxBxCOutputPorts;
    */

	m_graph->Stop();
    m_graph->Teardown();

    delete m_inputMatrix;
    delete m_outputMatrix;

    if (m_producerThread != NULL) CloseHandle(m_producerThread);
    if (m_consumerThread != NULL) CloseHandle(m_consumerThread);
    delete m_timer;
    Graph::DestroyGraph(m_graph);
    DeleteCriticalSection(&m_csProducerConsumerState);
}

#pragma warning(disable:4996)
Graph* BenchGraph::buildGraph(
    GraphType type,
    char * szfile,
    char * szshader,
    int graphDepth,
    int graphBredth,
    int ptaskPriority,
    int threadPriority,
    bool emulateModular)
{
    m_depth = graphDepth;

    if (type == GraphType::BINTREE)
    {
        assert(graphBredth == -1); // Must pass in -1. Will be replaced with the max bredth (= the bredth at depth == 1)
    }

    CompiledKernel * kernel = PTask::Runtime::GetCompiledKernel(szfile, szshader, g_szCompilerOutputBuffer, COBBUFSIZE);
    CheckCompileSuccess(szfile, szshader, kernel);

    char graphName[50];
    sprintf(graphName, "graph_%d", m_graphId);
    Graph* graph = new Graph(graphName);

    // Debug mode causes the runtime to:
    // a) always materialize a host-domain view of data
    //    pushed into internal channels, even if the source and
    //    sink ports indicate GPU->GPU domain
    // b) always mark the block incoherent after materializing
    //    the view to force a CPU->GPU copy before dispatching
    //    the consumer, allowing any host-side modifications to
    //    to the block to be reflected in the inputs of the consumer.
    // These two features also conveniently give us a lower bound
    // on the PCI/memory traffic that would be required in a modular
    // implementation.
    CONFIGUREPTASKU(DebugMode, emulateModular);

    int inputPortsPerTask = 3;
    int outputPortsPerTask = 1;

    // Number of channels and tasks is different for each graph type.
    int expectedNumTasks = -1;
    if (type == GraphType::RECT)
    {
        // Number of tasks expected is graph depth times bredth.
        expectedNumTasks = graphDepth * graphBredth;

        // The graph bredth is the same at all depths.
        m_maxBredth = graphBredth;

        // All tasks have one external data input ... except those at depth zero, which have two.
        m_numInputDataChannels =  graphBredth * (graphDepth + 1);

        // All tasks have one external parm input.
        m_numInputParmChannels = expectedNumTasks;

        // Number of external outputs equals the bredth of the graph, as the last task in each chain
        // produces one output.
        m_numOutputChannels = graphBredth;
    }
    else if (type == GraphType::BINTREE)
    {
        // Number of tasks expected..
        expectedNumTasks = binTreeTotalNumTasks(graphDepth);

        // The graph has maximum bredth at depth zero.
        int bredthAtDeptZero = binTreeBredthAtDepth(graphDepth, 0);
        m_maxBredth = bredthAtDeptZero;

        // Only tasks at depth 0 have an external data input. Each of them has two.
        m_numInputDataChannels =  bredthAtDeptZero * 2;

        // All tasks have one external parm input.
        m_numInputParmChannels = expectedNumTasks;

        // Number of external outputs equals 1.
        m_numOutputChannels = 1;
    }
    else
    {
        assert(FALSE);
    }

    if (m_verbosity >= 2) // JC
    {
        printf("\nGraph for %s depth %d, (max) bredth %d. Task priorities=(%d, %d)\n", 
            type == GraphType::RECT ? "RECT" : "BINTREE",
            graphDepth, m_maxBredth, ptaskPriority, threadPriority);
    }

    // Increase as we add tasks to the graph (as a sanity check of expected total number of tasks).
    m_numTasks = 0;

    int nextInputDataChannel = 0;
    int nextInputParmChannel = 0;
    int nextInternalChannel = 0;
    int nextOutputChannel = 0;

    m_inputDataChannels = new GraphInputChannel*[m_numInputDataChannels];
    m_inputParmChannels = new GraphInputChannel*[m_numInputParmChannels];
    m_outputChannels = new GraphOutputChannel*[m_numOutputChannels];

    char name[100];
    UINT uiUidCounter = 0;
    Port** upstreamOutputPorts = new Port*[m_maxBredth];
    for (int d = 0; d < graphDepth; d++)
    {
        int bredthAtCurrentDepth = -1;
        // Bredth at given graph depth depends on graph type.
        if (type == GraphType::RECT)
        {
            // For RECT, the bredth is the same at all depths.
            bredthAtCurrentDepth = graphBredth;
        }
        else if (type == GraphType::BINTREE)
        {
            // For BINTREE, the bredth is a function of the total graph depth and
            // the current depth.
            bredthAtCurrentDepth = binTreeBredthAtDepth(graphDepth, d);
        }
        else
        {
            assert(FALSE);
        }

        for (int b = 0; b < bredthAtCurrentDepth; b++)
        {
            sprintf(name, "Task(%d,%d)", d, b);
            Port ** taskInputPorts = new Port*[inputPortsPerTask];
            Port ** taskOutputPorts = new Port*[outputPortsPerTask];
            taskInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, m_dataTemplate, uiUidCounter++);
            taskInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, m_dataTemplate, uiUidCounter++);
            taskInputPorts[2]		= PTask::Runtime::CreatePort(STICKY_PORT, m_parmTemplate, uiUidCounter++);
            taskOutputPorts[0]		= PTask::Runtime::CreatePort(OUTPUT_PORT, m_dataTemplate, uiUidCounter++);

            Task * task = graph->AddTask(kernel, 
                inputPortsPerTask,
                taskInputPorts,
                outputPortsPerTask,
                taskOutputPorts,
                name);

            task->SetComputeGeometry(m_rows, m_cols, 1);
            task->SetPriority(ptaskPriority, threadPriority);
            m_numTasks++;

            // Connect task's first data input (c.f. A or AxB in RECT AxBxC scenario).
            if (d == 0)
            {
                // For tasks at depth zero, this input comes from an external source,
                // for both graph type RECT and BINTREE.
                sprintf(name, "InputDataChannel(%d)", nextInputDataChannel);
                m_inputDataChannels[nextInputDataChannel++] =
                    graph->AddInputChannel(taskInputPorts[0], name);
                if (m_verbosity >= 2)
                {
                    printf("%s -> input %d of %s\n", 
                        name, 0, taskInputPorts[0]->GetTask()->GetTaskName());
                }
            }
            else
            {
                // For all subsequent tasks, it comes from an upstream task,
                // for both graph type RECT and BINTREE.
                sprintf(name, "InternalDataChannel(%d)", nextInternalChannel++);
                int upstreamTaskIndex = -1;
                // Which upstream task it comes from depends on the graph type.
                if (type == GraphType::RECT)
                {
                    upstreamTaskIndex = b;
                }
                else if (type == GraphType::BINTREE)
                {
                    upstreamTaskIndex = b * 2;
                }
                graph->AddInternalChannel(upstreamOutputPorts[upstreamTaskIndex], taskInputPorts[0], name);
                if (m_verbosity >= 2)
                {
                    printf("%s from output 0 of %s -> input 0 of %s\n", 
                        name, upstreamOutputPorts[upstreamTaskIndex]->GetTask()->GetTaskName(),
                        taskInputPorts[0]->GetTask()->GetTaskName());
                }
            }

            // Connect task's second data input (c.f. B or C in RECT AxBxC scenario).
            // For graph type RECT this is always from an external source.
            // For graph type BINTREE this is only from an external source at depth = 0.
            if (type == GraphType::RECT ||
               (type == GraphType::BINTREE && d == 0))
            {
                sprintf(name, "InputDataChannel(%d)", nextInputDataChannel);
                m_inputDataChannels[nextInputDataChannel++] =
                    graph->AddInputChannel(taskInputPorts[1], name);
                if (m_verbosity >= 2)
                {
                    printf("%s -> input %d of %s\n", 
                        name, 1, taskInputPorts[1]->GetTask()->GetTaskName());
                }
            }
            else if (type == GraphType::BINTREE && d > 0)
            {
                // For graph type BINTREE and depth > 0, it comes from an upstream task.
                sprintf(name, "InternalDataChannel(%d)", nextInternalChannel++);
                int upstreamTaskIndex = (b * 2) + 1;
                graph->AddInternalChannel(upstreamOutputPorts[upstreamTaskIndex], taskInputPorts[1], name);
                if (m_verbosity >= 2)
                {
                    printf("%s from output 0 of %s -> input 1 of %s\n", 
                        name, upstreamOutputPorts[upstreamTaskIndex]->GetTask()->GetTaskName(),
                        taskInputPorts[1]->GetTask()->GetTaskName());
                }
            }
            else
            {
                assert(FALSE);
            }

            // Connect task's parm input.
            // This is always from an external source.
            sprintf(name, "InputParmChannel(%d)", nextInputParmChannel);
            m_inputParmChannels[nextInputParmChannel++] =
                graph->AddInputChannel(taskInputPorts[2], name);
            if (m_verbosity >= 2)
            {
                printf("%s -> input %d of %s\n", 
                    name, 2, taskInputPorts[2]->GetTask()->GetTaskName());
            }

            // If the task is at the deepest depth in the graph, 
            // connect its output port to an output channel.        
            if (d == (graphDepth - 1)) // Last 'row' of graph - may also be the first.
            {
                sprintf(name, "OutputChannel(%d)", nextOutputChannel);
                m_outputChannels[nextOutputChannel++] =
                    graph->AddOutputChannel(taskOutputPorts[0], name);
                if (m_verbosity >= 2)
                {
                    printf("%s -> output %d of %s\n", 
                        name, 0, taskOutputPorts[0]->GetTask()->GetTaskName());
                }
            }
            // Otherwise save its output port, as a task at the next depth 
            // will want to connect an internal channel it.
            else
            {
                upstreamOutputPorts[b] = taskOutputPorts[0];
            }
        }
    }
    assert(nextInputDataChannel == m_numInputDataChannels);
    assert(nextInputParmChannel == m_numInputParmChannels);
    assert(nextOutputChannel == m_numOutputChannels);
    assert(expectedNumTasks == m_numTasks);

    if (m_verbosity >= 2)
    {
        printf("Num tasks in graph = %d\n", m_numTasks);
    }

    return graph;
    #pragma warning(default:4996)
}

void BenchGraph::producer(bool runAllIterations)
{
    // Wait for start signal and start timer.
    // Only do this the first time are called - in ST case, will be called multiple times.
    if (m_currentProducerIteration == 0)
    {
        DEBUGLOG2(">>> Producer %d waiting for start signal\n", m_graphId);
        m_coordinator->waitForProducersStartSignal();
        DEBUGLOG2(">>> Producer %d starting\n", m_graphId);
    }

    while(!m_terminated)
    {
        DEBUGLOG3("      >>> Producer %d pushing input for iteration %d\n", m_graphId, m_currentProducerIteration);

        if (m_currentProducerIteration == m_timingStartIteration)
        {
            m_startTime = m_timer->elapsed(false) / (double)1000; // Convert from msec to sec.
            DEBUGLOG2(">>> Producer %d starting timing\n", m_graphId);
        }

        // Only push parm datablocks for the first iteration, since their channels are sticky and
        // the parm values are the same for all iterations.
        if (m_currentProducerIteration == 0)
        {
            for (int j = 0; j < m_numInputParmChannels; j++)
            {
                Datablock * db = PTask::Runtime::AllocateDatablock(m_parmTemplate, &m_matrixParams, sizeof(m_matrixParams), m_inputParmChannels[j]);
                m_inputParmChannels[j]->Push(db);
            }
        }

        // Updating instrumentation in midst of this batch of pushes. Can't avoid possibilty of context switch.
        // Will possibly push this down into PTask at some point.
        if (m_instrument)
        {
            EnterCriticalSection(&m_csProducerConsumerState);
            m_pushed++;
            m_outstandingInvocations = m_pushed - m_pulled;
            if (m_outstandingInvocations > m_maxOutstandingInvocations)
            {
                m_maxOutstandingInvocations = m_outstandingInvocations;   
            }
            LeaveCriticalSection(&m_csProducerConsumerState);
        }

        for (int j = 0; j < m_numInputDataChannels; j++)
        {
            Datablock * db = PTask::Runtime::AllocateDatablock(m_dataTemplate, 
                                                               m_inputMatrix->cells(), 
                                                               m_inputMatrix->arraysize(), 
                                                               m_inputDataChannels[j]);
            m_inputDataChannels[j]->Push(db);
        }

        m_currentProducerIteration++;
        if (!runAllIterations)
        {
            break;
        }
    }
    DEBUGLOG2(">>> Producer %d finished\n", m_graphId);
}

void BenchGraph::consumer(bool runAllIterations)
{
    while (!m_terminated)
    {
        for (int j = 0; j < m_numOutputChannels; j++)
        {
            Datablock * pResultBlock = m_outputChannels[j]->Pull();
            assert(pResultBlock != NULL);
            ELEMTYPE * psrc = (ELEMTYPE*) pResultBlock->GetDataPointer(FALSE);
            ELEMTYPE * pdst = m_outputMatrix->cells();
            memcpy(pdst, psrc, m_elements*m_stride);
            pResultBlock->Release();
        }
        DEBUGLOG3("      <<< Consumer %d pulled output for iteration %d\n", m_graphId, m_currentConsumerIteration);

        if (m_instrument)
        {
            EnterCriticalSection(&m_csProducerConsumerState);
            m_pulled++;
            m_outstandingInvocations = m_pushed - m_pulled;
            LeaveCriticalSection(&m_csProducerConsumerState);
        }

        m_currentConsumerIteration++;
        if (m_currentConsumerIteration == m_timingStopIteration)
        {
            m_endTime = m_timer->elapsed(false) / (double)1000; // Convert from msec to sec.
            m_coordinator->consumerFinished();
            DEBUGLOG2("   <<< Consumer %d stopping timing\n", m_graphId);
            if (!m_steadyStateBenchmark)
            {
                m_terminated = TRUE;
                DEBUGLOG2("   <<< Consumer %d terminating\n", m_graphId);
            }
        }
/*
        if (m_currentConsumerIteration == m_timingStopIteration * 3)
        {
            DEBUGLOG2("   <<< Consumer %d GIVING UP\n", m_graphId);
            m_terminated = TRUE;
        }
*/
        if (!runAllIterations)
        {
            break;
        }
    }
    DEBUGLOG2("   <<< Consumer %d finished\n", m_graphId);
}

void BenchGraph::STProducerConsumer()
{
    while (!m_terminated)
    {
        // False runAllIterations flag tells each invocation of producer or consumer 
        // to process only one iteration.
        producer(FALSE);
        consumer(FALSE);
    }
}

void BenchGraph::resetTimer()
{
    m_timer->reset();
}

void BenchGraph::initializeBenchmark(int iterations, bool instrument, bool singleThreaded, int schedulingMode)
{
    if (m_steadyStateBenchmark)
    {
        // Ramp-up avoidance commented out for now. Revisit at some point...
        //        m_timingStartIteration = iterations;
        //        m_timingStopIteration = iterations * 2;
        m_timingStartIteration = 0;
        m_timingStopIteration = iterations;
    }
    else
    {
        m_timingStartIteration = 0;
        m_timingStopIteration = iterations;
    }

    m_currentProducerIteration = 0;
    m_currentConsumerIteration = 0;
    m_instrument = instrument;

    CONFIGUREPTASKU(SchedulingMode, singleThreaded?0:schedulingMode);
    m_graph->Run(singleThreaded);

    if (m_instrument)
    {
        EnterCriticalSection(&m_csProducerConsumerState);
        m_pushed = 0;
        m_pulled = 0;
        m_outstandingInvocations = 0;
        m_maxOutstandingInvocations = 0;
        LeaveCriticalSection(&m_csProducerConsumerState);
    }

    configure_raw_matrix(m_rows, m_cols, &m_inputMatrix);
    m_outputMatrix = new CSimpleMatrix<ELEMTYPE>(m_rows, m_cols);

    m_matrixParams.g_tex_cols = m_cols;
    m_matrixParams.g_tex_rows = m_rows;

    m_terminated = FALSE;
    // Launch producer and consumer (either on separate threads or combined on one thread).
    // Don't need to wait on them here, as they will signal completion via BenchGraphCoordinator.
    if (singleThreaded)
    {
        m_producerThread = CreateThread(NULL, 0, BenchGraphSTProducerConsumerThread, this, NULL, NULL);
        m_consumerThread = NULL;
    }
    else
    {
        m_producerThread = CreateThread(NULL, 0, BenchGraphProducerThread, this, NULL, NULL);
        m_consumerThread = CreateThread(NULL, 0, BenchGraphConsumerThread, this, NULL, NULL);
    }
}

int BenchGraph::getNumTasks()
{
    return m_numTasks;
}

int BenchGraph::getMaxBredth()
{
    return m_maxBredth;
}

double BenchGraph::getStartTime()
{
    return m_startTime;
}

double BenchGraph::getEndTime()
{
    return m_endTime;
}

double BenchGraph::getTotalTime()
{
    return m_endTime - m_startTime;
}

int BenchGraph::getTotalIterationsProduced()
{
    return m_currentProducerIteration;
}

int BenchGraph::getTotalIterationsConsumed()
{
    return m_currentConsumerIteration;
}

int BenchGraph::getMaxOutstandingInvocations()
{
    return m_maxOutstandingInvocations;
}

void run_one_configuration(
    BenchGraph::GraphType type, char * szfile, char * szshader,
    int graphDepth, int graphBredth, int rows, int cols,
    int iterations, int nInstances, int* ptaskPriorities, int* threadPriorities, int schedulingMode, bool steadyStateBenchmark,
    bool instrument, bool singleThreaded, bool emulateModular, int verbosity)
{
    g_coordinator->reset(nInstances);
    BenchGraph** graphs = new BenchGraph*[nInstances];
    for (int i=0; i<nInstances; i++)
    {
        graphs[i] = new BenchGraph(i, g_coordinator, type, szfile, szshader, graphDepth, graphBredth, rows, cols, 
            ptaskPriorities[i], threadPriorities[i], emulateModular, steadyStateBenchmark, verbosity);
        graphs[i]->initializeBenchmark(iterations, instrument, singleThreaded, schedulingMode);
    }
    for (int i=0; i<nInstances; i++)
    {
        graphs[i]->resetTimer();
    }
    g_coordinator->signalProducersToStart();
    g_coordinator->waitForAllConsumersToFinish();

    // If doing a steady state benchmark, each graph did not terminate its producer and consumer
    // when it reached the end of its timing run. So terminate them now.
    if (steadyStateBenchmark)
    {
        for (int i=0; i<nInstances; i++)
        {
            graphs[i]->terminateProducerAndConsumer();
        }
        Sleep(1000); // Give the threads a while to notice that they should terminate.
    }

    printf("%s, %s, %s, %s, %dx%d, %d, %d, %d",
        schedulingMode == 0 ? "COOP" : schedulingMode == 1 ? "PRIO" : schedulingMode == 2 ? "DATA" : "FIFO",
        type == BenchGraph::GraphType::RECT ? "RECT" : "BINTREE",
        singleThreaded ? "ST" : "MT",
        emulateModular ? "    MOD" : "NON-MOD",
        rows, cols, graphDepth, graphs[0]->getMaxBredth(), graphs[0]->getNumTasks());
 
    for (int i=0; i<nInstances; i++)
    {
        double totalTime = graphs[i]->getTotalTime();
        double iterationsPerSec = (double)iterations / totalTime;
        double tasksPerSec = iterationsPerSec * (double)(graphs[i]->getNumTasks());
        // printf(", %f, %f", iterationsPerSec, tasksPerSec);
        printf(", %.1f", tasksPerSec);
    }

/*   for (int i=0; i<nInstances; i++)
   {
        //printf(" [%.1f to %.1f = %.1f (%d/%d)] ",
        printf(" [%.1f to %.1f = %.1f] ",
            graphs[i]->getStartTime(), 
            graphs[i]->getEndTime(), 
            graphs[i]->getTotalTime() 
            //graphs[i]->getTotalIterationsProduced(), 
            //graphs[i]->getTotalIterationsConsumed()
            );
   }*/
    printf("\n");

    if (instrument)
    {
        printf("Max outstanding invocations = \n");
        for (int i=0; i<nInstances; i++)
        {
            printf("%d ", graphs[i]->getMaxOutstandingInvocations());
        }
        printf("\n");
    }

    for (int i=0; i<nInstances; i++)
    {
        delete graphs[i];
    }
    delete [] graphs;

}

void run_range_matrix_and_graph_sizes(
    BenchGraph::GraphType type,
    char* szfile, char* szshader,
    int dimensionMin, int dimensionMax,
    int depthMin, int depthMax, int depthStep,
    int bredthMin, int bredthMax, int bredthStep,
    int iterations, int nInstances, int* ptaskPriorities, int* threadPriorities, int schedulingMode, bool steadyStateBenchmark,
    bool instrument, bool singleThreaded, bool emulateModular, int verbosity
    )
{
    // For BINTREE, ignore bredth, since only one value makes sense, and it will be derived from the depth.
    if (type == BenchGraph::GraphType::BINTREE)
    {
        bredthMin = bredthMax = -1;
        bredthStep = 1;
    }

    for (int dimension = dimensionMin; dimension <= dimensionMax; dimension *= 2)
    {
        for (int depth = depthMin; depth <= depthMax; depth += depthStep)
        {
            for (int bredth = bredthMin; bredth <= bredthMax; bredth += bredthStep)
            {
                run_one_configuration(type, szfile, szshader, depth, bredth, 
                    dimension, dimension, iterations, nInstances, ptaskPriorities, threadPriorities, schedulingMode, steadyStateBenchmark,
                    instrument, singleThreaded, emulateModular, verbosity);

                // We want to go e.g. 1, 2, 4, 6 instead of 1, 3, 5, 7 ...
                if (bredth == 1 && (bredthStep > 1))
                {
                    bredth = 0;
                }
            }

            // We want to go e.g. 1, 2, 4, 6 instead of 1, 3, 5, 7 ...
            if (depth == 1 && (depthStep > 1))
            {
                depth = 0;
            }
        }
    }
}

extern double s_gWeight_CurrentWaitBump = 3.5;

int run_graph_bench_task(
    char * szfile,
    char * szshader,
    int depthMin,
    int depthMax,
    int depthStep,
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
    ) 
{
    bool steadyStateBenchmark = true;

    g_coordinator = new BenchGraphCoordinator();
    BenchGraph::GraphType type = (BenchGraph::GraphType) gtype;
    CONFIGUREPTASKU(UseOpenCL, FALSE);
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    printf("\nBenchmark %s\n", szfile);
    printf("depthMin=%d, depthMax=%d, depthStep=%d\n", depthMin, depthMax, depthStep);
    printf("bredthMin=%d, bredthMax=%d, bredthStep=%d\n", bredthMin, bredthMax, bredthStep);
    printf("dimensionMin=%d, dimensionMax=%d\n", dimensionMin, dimensionMax);
    printf("Iterations = %d\n", iterations);
    printf("Num graph instances = %d. ptaskPrio(threadPrio) = ", nInstances);
    for (int i=0; i<nInstances; i++)
    {
        printf("%d(%d), ", ptaskPriorities[i], threadPriorities[i]);
    }
    printf("\n\nSchedMode, Type, Threading, Modular, Dim, Depth, Bredth, NumTasks");
    for (int i=0; i<nInstances; i++)
    {
        //JC printf(", itersec%d, tasksec%d", i, i);
        printf(", tasksec%d", i);
    }
    printf("\n");

    // Sweep across all scheduling modes.
    if (schedulingMode == -1)
    {
        //for (double w=2.0; w+=0.5; w<10)
        {
            //s_gWeight_CurrentWaitBump = w;
            //printf("s_gWeight_CurrentWaitBump = %f:\n", s_gWeight_CurrentWaitBump);
        for (int m=0; m<=3; m++) // Adjust if add extra modes.
        {
            // Potentially run multiple iterations on each, to check stability.
            for (int i=0; i<3; i++)
            {
                // Multi-Threaded, Non-Modular
                run_range_matrix_and_graph_sizes(type, szfile, szshader, dimensionMin, dimensionMax, 
                    depthMin, depthMax, depthStep, bredthMin, bredthMax, bredthStep,
                    iterations, nInstances, ptaskPriorities, threadPriorities, m, steadyStateBenchmark, instrument, false, false, verbosity);
            }
            printf("\n");
        }
        }
    }
    else
    {
    // Single-Threaded, Modular
    run_range_matrix_and_graph_sizes(type, szfile, szshader, dimensionMin, dimensionMax, 
        depthMin, depthMax, depthStep, bredthMin, bredthMax, bredthStep,
        iterations, nInstances, ptaskPriorities, threadPriorities, schedulingMode, steadyStateBenchmark, instrument, true, true, verbosity);

    // Single-Threaded, Non-Modular
    run_range_matrix_and_graph_sizes(type, szfile, szshader, dimensionMin, dimensionMax, 
        depthMin, depthMax, depthStep, bredthMin, bredthMax, bredthStep,
        iterations, nInstances, ptaskPriorities, threadPriorities, schedulingMode, steadyStateBenchmark, instrument, true, false, verbosity);

    // Multi-Threaded, Modular
    run_range_matrix_and_graph_sizes(type, szfile, szshader, dimensionMin, dimensionMax, 
        depthMin, depthMax, depthStep, bredthMin, bredthMax, bredthStep,
        iterations, nInstances, ptaskPriorities, threadPriorities, schedulingMode, steadyStateBenchmark, instrument, false, true, verbosity);

    // Multi-Threaded, Non-Modular
    run_range_matrix_and_graph_sizes(type, szfile, szshader, dimensionMin, dimensionMax, 
        depthMin, depthMax, depthStep, bredthMin, bredthMax, bredthStep,
        iterations, nInstances, ptaskPriorities, threadPriorities, schedulingMode, steadyStateBenchmark, instrument, false, false, verbosity);
    }

    PTask::Runtime::Terminate();
    return 0;
}
