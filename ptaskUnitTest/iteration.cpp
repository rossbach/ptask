//--------------------------------------------------------------------------------------
// File: iteration.cpp
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#include <stdio.h>
#include <crtdbg.h>
#include "accelerator.h"
#include "assert.h"
#include "shaderparms.h"
#include "sort.h"
#include <vector>
#include <algorithm>
#include "matrixtask.h"
#include "SimpleMatrix.h"
#include "SimpleVector.h"
#include "matmul.h"
#include "iteration.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

using namespace std;
using namespace PTask;

extern float * random_vector(int n);
extern BOOL compare_vectors(float*pA, float*pB, int n);
extern float * vector_scale(float * pA, float scalar, int n);
extern float * vector_scale_repeatedly(float * pA, float scalar, int n, int nIterations);
extern void vector_scale_in_place(float * pA, float scalar, int n);

void 
perform_graph_diagnostics(
    Graph * pGraph,
    char * lpszDotFileName,
    BOOL bForceGraphDisplay
    ) 
{
    // Output the graph in graphviz 'dot' format.
    // (Use dot -Tpng GeneralIterationGraph.dot -o GeneralIterationGraph.png to render as PNG image.)
    pGraph->WriteDOTFile(lpszDotFileName);
    PTRESULT ptr = pGraph->CheckGraphSemantics();
    BOOL bMalformed = !PTSUCCESS(ptr);
    if(bMalformed || bForceGraphDisplay) {
        char szCommandLine[1024];
        if(bMalformed) {
            printf("malformed graph (code 0x%.8X! exiting...\n", ptr);
        }
        sprintf_s(szCommandLine, 1024, "dot -Tpng %s -o iteration.png", lpszDotFileName);
        system(szCommandLine);
        system("start iteration.png");
        if(bMalformed) { 
            exit(ptr);
        }
    }
}

void 
perform_graph_diagnostics(
    Graph * pGraph,
    char * lpszDotFileName
    )
{
    perform_graph_diagnostics(pGraph, lpszDotFileName, FALSE);
}

int run_simple_iteration_task(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    ) 
{
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    assert(rows == 1 || cols == 1);		// vector, not matrix!
    int n = rows * cols;

    float* vAVector = random_vector(n);
    float* vBVector = random_vector(n);

    VECADD_PARAMS params;
    float fScalar = 2.0f;
    params.N = n; 
    UINT uiParamsStride = sizeof(VECADD_PARAMS);
    UINT stride = sizeof(float);
    UINT nPulse = 1;

    float* vAReference = vector_scale_repeatedly(vAVector, fScalar, n, iterations);

    int nChannelCount = 0;

    CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
    pTimer->reset();

   
    Graph * pGraph = new Graph();
    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
    DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", stride, PTPARM_FLOAT);
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
    DatablockTemplate * pIterTemplate	= PTask::Runtime::GetDatablockTemplate("dbiter", sizeof(int), PTPARM_INT);
    CompiledKernel * pKernel		    = COMPILE_KERNEL(szfile, szshader);

    const UINT uiInputCount = 4;
    const UINT uiOutputCount = 1;
    
    Port ** pTopInputPorts = new Port*[uiInputCount];
    Port ** pTopOutputPorts = new Port*[uiOutputCount];

    UINT uiUidCounter		= 0;
    pTopInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_top", 0, 0);
    pTopInputPorts[1]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_top", 1);
    pTopInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_top", 2);
    pTopInputPorts[3]   = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "iter_meta", 3);
    pTopOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_top", 0);

    PTask::Runtime::SetPortMetaFunction(pTopInputPorts[3], MF_SIMPLE_ITERATOR);
    Task * pTopTask = PTask::Runtime::AddTask(pGraph,
                                              pKernel, 
                                              uiInputCount,
                                              pTopInputPorts,
                                              uiOutputCount,
                                              pTopOutputPorts,
                                              "top");


    assert(pTopTask);
    pTopTask->SetComputeGeometry(n, 1, 1);
    PTASKDIM3 threadBlockSize(256, 1, 1);
    PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
    pTopTask->SetBlockAndGridSize(gridSize, threadBlockSize);

    GraphInputChannel * pTopAInput				= pGraph->AddInputChannel(pTopInputPorts[0], "AInputChannel");
    GraphInputChannel * pTopAxBScaleInput		= pGraph->AddInputChannel(pTopInputPorts[1], "ScalarChannel");
    GraphInputChannel * pIterationCount         = pGraph->AddInputChannel(pTopInputPorts[3], "IterationChannel");
    GraphOutputChannel * pAOutputA			    = pGraph->AddOutputChannel(pTopOutputPorts[0], "outputChannel_left_A");

    PTask::Runtime::BindDerivedPort(pGraph, pTopInputPorts[0], pTopInputPorts[2]);

    perform_graph_diagnostics(pGraph, "SimpleIteration.dot");

    PTask::Runtime::RunGraph(pGraph);

    Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pTopAxBScaleInput);
    pTopAxBScaleInput->Push(pScPrm);

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pTopAInput);
    Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), pIterationCount);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();

    PTask::Runtime::Push(pTopAInput, pA);
    PTask::Runtime::Push(pIterationCount, pIters);
    pA->Release();
    pIters->Release();
    pScPrm->Release();

    Datablock * pLAout = PTask::Runtime::Pull(pAOutputA);

    float * psrc = NULL;
    int nFailures = 0;
    pLAout->Lock();
    psrc = (float*) pLAout->GetDataPointer(FALSE);
    // Sleep(100);
    if(!compare_vectors(vAReference, psrc, n))
        nFailures++;
    pLAout->Unlock();
    pLAout->Release();

    if(nFailures) {
        printf("failure.\n");
    } else  {
        printf( "%s succeeded\n", szshader );
    }

    pGraph->Stop();
    pGraph->Teardown();

    delete [] vAVector;
    delete [] vBVector;
    delete [] vAReference;
	Graph::DestroyGraph(pGraph);
    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;
    delete pTimer;

    PTask::Runtime::Terminate();

    return 0;
}

int run_general_iteration_task(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    ) 
{
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    assert(rows == 1 || cols == 1);		// vector, not matrix!
    int n = rows * cols;
    float* vAVector = random_vector(n);

    VECADD_PARAMS params;
    float fScalar = 2.0f;
    params.N = n; 
    UINT uiParamsStride = sizeof(VECADD_PARAMS);
    UINT stride = sizeof(float);
    UINT nPulse = 1;

    bool insertPreLoopTask = true;

    int totalNumScalings = iterations*2;
    if (insertPreLoopTask)
    {
        totalNumScalings++; // One extra scaling, performed by the pre-loop task.
    }
    float* vAReference = vector_scale_repeatedly(vAVector, fScalar, n, totalNumScalings);

    int nChannelCount = 0;

    CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
    pTimer->reset();

    Graph * pGraph = new Graph();

    UINT uiUidCounter		= 0;
    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
    DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", stride, PTPARM_FLOAT);
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
    DatablockTemplate * pIterTemplate	= PTask::Runtime::GetDatablockTemplate("dbiter", sizeof(int), PTPARM_INT);
    CompiledKernel * pKernel		    = COMPILE_KERNEL(szfile, szshader);

    Port ** pPreLoopInputPorts = NULL;
    Port ** pPreLoopOutputPorts = NULL;
    if (insertPreLoopTask)
    {
        const UINT uiInputCount = 3;
        const UINT uiOutputCount = 1;

        pPreLoopInputPorts = new Port*[uiInputCount];
        pPreLoopOutputPorts = new Port*[uiOutputCount];

        pPreLoopInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_preloop", 0, 0);
        pPreLoopInputPorts[1]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_preloop", 1);
        pPreLoopInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_preloop", 2);
        pPreLoopOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_preloop", 0);

        Task * pPreLoopTask = pGraph->AddTask(pKernel, 
            uiInputCount,
            pPreLoopInputPorts,
            uiOutputCount,
            pPreLoopOutputPorts,
            "preloop");

        assert(pPreLoopTask);
        pPreLoopTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pPreLoopTask->SetBlockAndGridSize(gridSize, threadBlockSize);
    }

    Port ** pTopInputPorts = NULL;
    Port ** pTopOutputPorts = NULL;
    {
        const UINT uiInputCount = 3;
        const UINT uiOutputCount = 1;

        pTopInputPorts = new Port*[uiInputCount];
        pTopOutputPorts = new Port*[uiOutputCount];

        pTopInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_top", 0, 0);
        pTopInputPorts[1]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_top", 1);
        pTopInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_top", 2);
        pTopOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_top", 0);


        Task * pTopTask = pGraph->AddTask(pKernel, 
            uiInputCount,
            pTopInputPorts,
            uiOutputCount,
            pTopOutputPorts,
            "top");

        assert(pTopTask);
        pTopTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pTopTask->SetBlockAndGridSize(gridSize, threadBlockSize);
    }

    Port ** pBottomInputPorts = NULL;
    Port ** pBottomOutputPorts = NULL;
    {
        const UINT uiInputCount = 4;
        const UINT uiOutputCount = 1;

        pBottomInputPorts = new Port*[uiInputCount];
        pBottomOutputPorts = new Port*[uiOutputCount];

        UINT uiUidCounter		= 0;
        pBottomInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_Bottom", 0, 0);
        pBottomInputPorts[1]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_Bottom", 1);
        pBottomInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_bottom", 2);
        pBottomInputPorts[3]    = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "Meta_bottom", 3);
        pBottomOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_Bottom", 0);

        pBottomInputPorts[3]->SetMetaFunction(MF_GENERAL_ITERATOR);

        Task * pBottomTask = pGraph->AddTask(pKernel, 
            uiInputCount,
            pBottomInputPorts,
            uiOutputCount,
            pBottomOutputPorts,
            "bottom");

        assert(pBottomTask);
        pBottomTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pBottomTask->SetBlockAndGridSize(gridSize, threadBlockSize);
    }

    // Set up graph input channels.
    GraphInputChannel * pAInputChannel = NULL;
    GraphInputChannel * pScalarInputChannelPreLoop = NULL;
    if (insertPreLoopTask)
    {
        pAInputChannel				= pGraph->AddInputChannel(pPreLoopInputPorts[0], "AInputChannel"); // Not a switch channel
        pScalarInputChannelPreLoop	= pGraph->AddInputChannel(pPreLoopInputPorts[1], "ScalarInputChannelPreLoop");

    } else {
        pAInputChannel				= pGraph->AddInputChannel(pTopInputPorts[0], "AInputChannel", TRUE); // A switch channel
    }
    GraphInputChannel * pScalarInputChannelTop		= pGraph->AddInputChannel(pTopInputPorts[1], "ScalarInputChannelTop");
    GraphInputChannel * pScalarInputChannelBottom   = pGraph->AddInputChannel(pBottomInputPorts[1], "ScalarInputChannelBottom");
    GraphInputChannel * pIterationCount             = pGraph->AddInputChannel(pBottomInputPorts[3], "IterationChannel");

    // Bind descriptor ports.
    // For all the tasks, port 0 (A) drives the value of port 2 (N).
    if (insertPreLoopTask)
    {
        pGraph->BindDescriptorPort(pPreLoopInputPorts[0], pPreLoopInputPorts[2]);
    }
    pGraph->BindDescriptorPort(pTopInputPorts[0], pTopInputPorts[2]);
    pGraph->BindDescriptorPort(pBottomInputPorts[0], pBottomInputPorts[2]);

    // Set up internal channels (including back channel for A from Bottom to Top task) and output channel.
    InternalChannel * pBackChannel                  = pGraph->AddInternalChannel(pBottomOutputPorts[0], pTopInputPorts[0], "back-channel");
    GraphOutputChannel * pAOutputChannel		    = pGraph->AddOutputChannel(pBottomOutputPorts[0], "outputChannel_left_A");
    if (insertPreLoopTask)
    {
        InternalChannel * pPreLoopToTop             = pGraph->AddInternalChannel(
            pPreLoopOutputPorts[0], pTopInputPorts[0], "A_PreLoopToTop", TRUE); // A switch channel
    }
    InternalChannel * pTopToBottom                  = pGraph->AddInternalChannel(pTopOutputPorts[0], pBottomInputPorts[0], "A_TopToBottom");

    // Configure looping.
    pGraph->BindControlPropagationPort(pBottomInputPorts[3], pBottomOutputPorts[0]);
    pBackChannel->SetPredicationType(CE_SRC, CGATEFN_CLOSE_ON_ENDITERATION);
    pAOutputChannel->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_ENDITERATION);

    perform_graph_diagnostics(pGraph, "GeneralIteration.dot");

    pGraph->Run();

    Datablock * pScalarParamPreLoop	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pScalarInputChannelPreLoop);
    Datablock * pScalarParamTop		 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pScalarInputChannelTop);
    Datablock * pScalarParamBottom	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pScalarInputChannelBottom);
    if (insertPreLoopTask)
    {
            pScalarInputChannelPreLoop->Push(pScalarParamPreLoop);
    }
    pScalarInputChannelTop->Push(pScalarParamTop);
    pScalarInputChannelBottom->Push(pScalarParamBottom);
    pScalarParamPreLoop->Release();
    pScalarParamTop->Release();
    pScalarParamBottom->Release();

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pAInputChannel);
    Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), pIterationCount);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();

    pAInputChannel->Push(pA);
    pIterationCount->Push(pIters);
    pA->Release();
    pIters->Release();

    Datablock * pLAout = pAOutputChannel->Pull();

    float * psrc = NULL;
    int nFailures = 0;
    pLAout->Lock();
    psrc = (float*) pLAout->GetDataPointer(FALSE);
    if(!compare_vectors(vAReference, psrc, n))
        nFailures++;
    pLAout->Unlock();  
    pLAout->Release();

    if(nFailures) {
        printf("failure.\n");
    } else  {
        printf( "%s succeeded\n", szshader );
    }

    pGraph->Stop();
    pGraph->Teardown();
	Graph::DestroyGraph(pGraph);

    if(pTopInputPorts) delete [] pTopInputPorts;
    if(pTopOutputPorts) delete [] pTopOutputPorts;
    if(pPreLoopInputPorts) delete [] pPreLoopInputPorts;
    if(pPreLoopOutputPorts) delete [] pPreLoopOutputPorts;
    if(pBottomInputPorts) delete [] pBottomInputPorts;
    if(pBottomOutputPorts) delete [] pBottomOutputPorts;
    if(vAVector) delete [] vAVector;
    if(vAReference) delete [] vAReference;
    if(pTimer) delete pTimer;

    PTask::Runtime::Terminate();

    return 0;
}

int run_general_iteration2_task(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    ) 
{
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    assert(rows == 1 || cols == 1);		// vector, not matrix!
    int n = rows * cols;
    float* vAVector = random_vector(n);

    VECADD_PARAMS params;
    float fScalar = 2.0f;
    params.N = n; 
    UINT uiParamsStride = sizeof(VECADD_PARAMS);
    UINT stride = sizeof(float);
    UINT nPulse = 1;

    int totalNumScalings = iterations*2;
    float* vAReference = vector_scale_repeatedly(vAVector, fScalar, n, totalNumScalings);

    int nChannelCount = 0;

    CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
    pTimer->reset();

    Graph * pGraph = new Graph();
    UINT uiUidCounter		= 0;
    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
    DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", stride, PTPARM_FLOAT);
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
    DatablockTemplate * pIterTemplate	= PTask::Runtime::GetDatablockTemplate("dbiter", sizeof(int), PTPARM_INT);
    CompiledKernel * pKernel		    = COMPILE_KERNEL(szfile, szshader);


    Port ** pTopInputPorts = NULL;
    Port ** pTopOutputPorts = NULL;
    {
        const UINT uiInputCount = 4;
        const UINT uiOutputCount = 2;

        pTopInputPorts = new Port*[uiInputCount];
        pTopOutputPorts = new Port*[uiOutputCount];

        pTopInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_top", 0, 0);
        pTopInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_top", 1, 1);
        pTopInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_top", 2);
        pTopInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_top", 3);
        pTopOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_top", 0);
        pTopOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B(out)_top", 1);

        Task * pTopTask = pGraph->AddTask(pKernel, 
            uiInputCount,
            pTopInputPorts,
            uiOutputCount,
            pTopOutputPorts,
            "top");

        assert(pTopTask);
        pTopTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pTopTask->SetBlockAndGridSize(gridSize, threadBlockSize);
    }

    Port ** pBottomInputPorts = NULL;
    Port ** pBottomOutputPorts = NULL;
    {
        const UINT uiInputCount = 5;
        const UINT uiOutputCount = 2;

        pBottomInputPorts = new Port*[uiInputCount];
        pBottomOutputPorts = new Port*[uiOutputCount];

        UINT uiUidCounter		= 0;
        pBottomInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_Bottom", 0, 0);
        pBottomInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_Bottom", 1, 1);
        pBottomInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_Bottom", 2);
        pBottomInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_bottom", 3);
        pBottomInputPorts[4]    = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "Meta_bottom", 4);
        pBottomOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_Bottom", 0);
        pBottomOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B(out)_Bottom", 1);

        pBottomInputPorts[4]->SetMetaFunction(MF_GENERAL_ITERATOR);

        Task * pBottomTask = pGraph->AddTask(pKernel, 
            uiInputCount,
            pBottomInputPorts,
            uiOutputCount,
            pBottomOutputPorts,
            "bottom");

        assert(pBottomTask);
        pBottomTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pBottomTask->SetBlockAndGridSize(gridSize, threadBlockSize);
    }

    // Set up graph input channels.
    GraphInputChannel * pAInputChannel				= pGraph->AddInputChannel(pTopInputPorts[0], "AInputChannel", TRUE); // A switch channel
    GraphInputChannel * pBInputChannel				= pGraph->AddInputChannel(pTopInputPorts[1], "BInputChannel", TRUE); // A switch channel
    GraphInputChannel * pScalarInputChannelTop		= pGraph->AddInputChannel(pTopInputPorts[2], "ScalarInputChannelTop");
    GraphInputChannel * pScalarInputChannelBottom   = pGraph->AddInputChannel(pBottomInputPorts[2], "ScalarInputChannelBottom");
    GraphInputChannel * pIterationCount             = pGraph->AddInputChannel(pBottomInputPorts[4], "IterationChannel");

    // Bind descriptor ports.
    // For all the tasks, port 0 (A) drives the value of port 3 (N).
    pGraph->BindDescriptorPort(pTopInputPorts[0], pTopInputPorts[3]);
    pGraph->BindDescriptorPort(pBottomInputPorts[0], pBottomInputPorts[3]);

    // Set up internal channels (including back channel for A from Bottom to Top task) and output channel.
    InternalChannel * pBackChannelA                 = pGraph->AddInternalChannel(pBottomOutputPorts[0], pTopInputPorts[0], "back-channelA");
    InternalChannel * pBackChannelB                 = pGraph->AddInternalChannel(pBottomOutputPorts[1], pTopInputPorts[1], "back-channelB");
    GraphOutputChannel * pOutputChannelA		    = pGraph->AddOutputChannel(pBottomOutputPorts[0], "outputChannel_left_A");
    GraphOutputChannel * pOutputChannelB		    = pGraph->AddOutputChannel(pBottomOutputPorts[1], "outputChannel_left_B");
    InternalChannel * pTopToBottomA                  = pGraph->AddInternalChannel(pTopOutputPorts[0], pBottomInputPorts[0], "A_TopToBottom");
    InternalChannel * pTopToBottomB                  = pGraph->AddInternalChannel(pTopOutputPorts[1], pBottomInputPorts[1], "B_TopToBottom");

    // Configure looping.
    // ----------------
    // XXXX: cjr: wrong port indeces! The META port is at index 4!
    // In the previous test cases, it was at index 3, but this test
    // adds an additional in/out parameter, forcing the metaport index up!
    // Unfortunatey, no control propagation route information is set up for 
    // the scalar "N" parameter (which is at index 3), so when the iteration ends, 
    // those blocks are marked with ENDITERATION, but that control information never
    // gets to the output ports. Hence, the hang. 
    // -------------------------------------------------------------------------------
    //pGraph->BindControlPropagationPort(pBottomInputPorts[3], pBottomOutputPorts[0]);
    //pGraph->BindControlPropagationPort(pBottomInputPorts[3], pBottomOutputPorts[1]);
    pGraph->BindControlPropagationPort(pBottomInputPorts[4], pBottomOutputPorts[0]);
    pGraph->BindControlPropagationPort(pBottomInputPorts[4], pBottomOutputPorts[1]);
    pBackChannelA->SetPredicationType(CE_SRC, CGATEFN_CLOSE_ON_ENDITERATION);
    pBackChannelB->SetPredicationType(CE_SRC, CGATEFN_CLOSE_ON_ENDITERATION);
    pOutputChannelA->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_ENDITERATION);
    pOutputChannelB->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_ENDITERATION);

    perform_graph_diagnostics(pGraph, "GeneralIteration2.dot");

    pGraph->Run();

    Datablock * pScalarParamTop		 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pScalarInputChannelTop);
    Datablock * pScalarParamBottom	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pScalarInputChannelBottom);
    pScalarInputChannelTop->Push(pScalarParamTop);
    pScalarInputChannelBottom->Push(pScalarParamBottom);
    pScalarParamTop->Release();
    pScalarParamBottom->Release();

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pAInputChannel);
    Datablock * pB		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pBInputChannel);
    Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), pIterationCount);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();

    pAInputChannel->Push(pA);
    pBInputChannel->Push(pB);
    pIterationCount->Push(pIters);
    pA->Release();
    pB->Release();
    pIters->Release();

    Datablock * pLAout = pOutputChannelA->Pull();
    Datablock * pLBout = pOutputChannelB->Pull();

    int nFailures = 0;
    float * psrcA = NULL;
    pLAout->Lock();
    psrcA = (float*) pLAout->GetDataPointer(FALSE);
    if(!compare_vectors(vAReference, psrcA, n))
        nFailures++;
    pLAout->Unlock();    
    float * psrcB = NULL;
    pLBout->Lock();
    psrcB = (float*) pLBout->GetDataPointer(FALSE);
    if(!compare_vectors(vAReference, psrcB, n))
        nFailures++;
    pLBout->Unlock();    

    if(nFailures) {
        printf("failure.\n");
    } else  {
        printf( "%s succeeded\n", szshader );
    }

    pGraph->Stop();
    pGraph->Teardown();

    delete [] vAVector;
	Graph::DestroyGraph(pGraph);
    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;

    delete pTimer;

    PTask::Runtime::Terminate();

    return 0;
}

int run_general_iteration2_with_preloop_task(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    ) 
{
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    assert(rows == 1 || cols == 1);		// vector, not matrix!
    int n = rows * cols;
    float* vAVector = random_vector(n);

    VECADD_PARAMS params;
    float fScalar = 2.0f;
    params.N = n; 
    UINT uiParamsStride = sizeof(VECADD_PARAMS);
    UINT stride = sizeof(float);
    UINT nPulse = 1;

    int totalNumScalings = iterations*2;
    totalNumScalings++; // One extra scaling, performed by the pre-loop task.
    float* vAReference = vector_scale_repeatedly(vAVector, fScalar, n, totalNumScalings);

    int nChannelCount = 0;

    CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
    pTimer->reset();

    Graph * pGraph = new Graph();
    UINT uiUidCounter		= 0;
    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
    DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", stride, PTPARM_FLOAT);
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
    DatablockTemplate * pIterTemplate	= PTask::Runtime::GetDatablockTemplate("dbiter", sizeof(int), PTPARM_INT);
    CompiledKernel * pKernel		    = COMPILE_KERNEL(szfile, szshader);

    Port ** pPreLoopInputPorts = NULL;
    Port ** pPreLoopOutputPorts = NULL;

    {
        const UINT uiInputCount = 4;
        const UINT uiOutputCount = 2;

        pPreLoopInputPorts = new Port*[uiInputCount];
        pPreLoopOutputPorts = new Port*[uiOutputCount];

        pPreLoopInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_preloop", 0, 0);
        pPreLoopInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_preloop", 1, 1);
        pPreLoopInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_preloop", 2);
        pPreLoopInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_preloop", 3);
        pPreLoopOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_preloop", 0);
        pPreLoopOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B(out)_preloop", 1);

        Task * pPreLoopTask = pGraph->AddTask(pKernel, 
            uiInputCount,
            pPreLoopInputPorts,
            uiOutputCount,
            pPreLoopOutputPorts,
            "preloop");

        assert(pPreLoopTask);
        pPreLoopTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pPreLoopTask->SetBlockAndGridSize(gridSize, threadBlockSize);
    }

    Port ** pTopInputPorts = NULL;
    Port ** pTopOutputPorts = NULL;
    {
        const UINT uiInputCount = 4;
        const UINT uiOutputCount = 2;

        pTopInputPorts = new Port*[uiInputCount];
        pTopOutputPorts = new Port*[uiOutputCount];

        pTopInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_top", 0, 0);
        pTopInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_top", 1, 1);
        pTopInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_top", 2);
        pTopInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_top", 3);
        pTopOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_top", 0);
        pTopOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B(out)_top", 1);

        Task * pTopTask = pGraph->AddTask(pKernel, 
            uiInputCount,
            pTopInputPorts,
            uiOutputCount,
            pTopOutputPorts,
            "top");

        assert(pTopTask);
        pTopTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pTopTask->SetBlockAndGridSize(gridSize, threadBlockSize);
    }

    Port ** pBottomInputPorts = NULL;
    Port ** pBottomOutputPorts = NULL;
    {
        const UINT uiInputCount = 5;
        const UINT uiOutputCount = 2;

        pBottomInputPorts = new Port*[uiInputCount];
        pBottomOutputPorts = new Port*[uiOutputCount];

        UINT uiUidCounter		= 0;
        pBottomInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_Bottom", 0, 0);
        pBottomInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_Bottom", 1, 1);
        pBottomInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_Bottom", 2);
        pBottomInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_bottom", 3);
        pBottomInputPorts[4]    = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "Meta_bottom", 4);
        pBottomOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_Bottom", 0);
        pBottomOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B(out)_Bottom", 1);

        pBottomInputPorts[4]->SetMetaFunction(MF_GENERAL_ITERATOR);

        Task * pBottomTask = pGraph->AddTask(pKernel, 
            uiInputCount,
            pBottomInputPorts,
            uiOutputCount,
            pBottomOutputPorts,
            "bottom");

        assert(pBottomTask);
        pBottomTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pBottomTask->SetBlockAndGridSize(gridSize, threadBlockSize);
    }

    // Set up graph input channels.
    GraphInputChannel * pAInputChannel = NULL;
    GraphInputChannel * pBInputChannel = NULL;
    GraphInputChannel * pScalarInputChannelPreLoop = NULL;
    pAInputChannel				= pGraph->AddInputChannel(pPreLoopInputPorts[0], "AInputChannel"); // Not a switch channel
    pBInputChannel				= pGraph->AddInputChannel(pPreLoopInputPorts[1], "BInputChannel"); // Not a switch channel
    pScalarInputChannelPreLoop	= pGraph->AddInputChannel(pPreLoopInputPorts[2], "ScalarInputChannelPreLoop");
    GraphInputChannel * pScalarInputChannelTop		= pGraph->AddInputChannel(pTopInputPorts[2], "ScalarInputChannelTop");
    GraphInputChannel * pScalarInputChannelBottom   = pGraph->AddInputChannel(pBottomInputPorts[2], "ScalarInputChannelBottom");
    GraphInputChannel * pIterationCount             = pGraph->AddInputChannel(pBottomInputPorts[4], "IterationChannel");

    // Bind descriptor ports.
    // For all the tasks, port 0 (A) drives the value of port 3 (N).
    pGraph->BindDescriptorPort(pPreLoopInputPorts[0], pPreLoopInputPorts[3]);
    pGraph->BindDescriptorPort(pTopInputPorts[0], pTopInputPorts[3]);
    pGraph->BindDescriptorPort(pBottomInputPorts[0], pBottomInputPorts[3]);

    // Set up internal channels (including back channel for A from Bottom to Top task) and output channel.
    InternalChannel * pBackChannelA                 = pGraph->AddInternalChannel(pBottomOutputPorts[0], pTopInputPorts[0], "back-channelA");
    InternalChannel * pBackChannelB                 = pGraph->AddInternalChannel(pBottomOutputPorts[1], pTopInputPorts[1], "back-channelB");
    GraphOutputChannel * pOutputChannelA		    = pGraph->AddOutputChannel(pBottomOutputPorts[0], "outputChannel_left_A");
    GraphOutputChannel * pOutputChannelB		    = pGraph->AddOutputChannel(pBottomOutputPorts[1], "outputChannel_left_B");
    InternalChannel * pPreLoopToTopA                = pGraph->AddInternalChannel(pPreLoopOutputPorts[0], 
                                                                                 pTopInputPorts[0], 
                                                                                 "A_PreLoopToTop", 
                                                                                 TRUE); // A switch channel
    InternalChannel * pPreLoopToTopB                = pGraph->AddInternalChannel(pPreLoopOutputPorts[1], 
                                                                                 pTopInputPorts[1], 
                                                                                 "B_PreLoopToTop", 
                                                                                 TRUE); // A switch channel
    InternalChannel * pTopToBottomA                  = pGraph->AddInternalChannel(pTopOutputPorts[0], pBottomInputPorts[0], "A_TopToBottom");
    InternalChannel * pTopToBottomB                  = pGraph->AddInternalChannel(pTopOutputPorts[1], pBottomInputPorts[1], "B_TopToBottom");

    // Configure looping.
    pGraph->BindControlPropagationPort(pBottomInputPorts[4], pBottomOutputPorts[0]);
    pGraph->BindControlPropagationPort(pBottomInputPorts[4], pBottomOutputPorts[1]);
    pBackChannelA->SetPredicationType(CE_SRC, CGATEFN_CLOSE_ON_ENDITERATION);
    pBackChannelB->SetPredicationType(CE_SRC, CGATEFN_CLOSE_ON_ENDITERATION);
    pOutputChannelA->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_ENDITERATION);
    pOutputChannelB->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_ENDITERATION);

    perform_graph_diagnostics(pGraph, "GeneralIteration2_withPreLoop.dot");

    pGraph->Run();

    Datablock * pScalarParamPreLoop	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pScalarInputChannelPreLoop);
    Datablock * pScalarParamTop		 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pScalarInputChannelTop);
    Datablock * pScalarParamBottom	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pScalarInputChannelBottom);

    pScalarInputChannelPreLoop->Push(pScalarParamPreLoop);
    pScalarInputChannelTop->Push(pScalarParamTop);
    pScalarInputChannelBottom->Push(pScalarParamBottom);
    pScalarParamPreLoop->Release();
    pScalarParamTop->Release();
    pScalarParamBottom->Release();

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pAInputChannel);
    Datablock * pB		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pBInputChannel);
    Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), pIterationCount);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();

    pAInputChannel->Push(pA);
    pBInputChannel->Push(pB);
    pIterationCount->Push(pIters);
    pA->Release();
    pB->Release();
    pIters->Release();

    Datablock * pLAout = pOutputChannelA->Pull();
    Datablock * pLBout = pOutputChannelB->Pull();

    int nFailures = 0;
    float * psrcA = NULL;
    pLAout->Lock();
    psrcA = (float*) pLAout->GetDataPointer(FALSE);
    if(!compare_vectors(vAReference, psrcA, n))
        nFailures++;
    pLAout->Unlock();    
    float * psrcB = NULL;
    pLBout->Lock();
    psrcB = (float*) pLBout->GetDataPointer(FALSE);
    if(!compare_vectors(vAReference, psrcB, n))
        nFailures++;
    pLBout->Unlock();    

    if(nFailures) {
        printf("failure.\n");
    } else  {
        printf( "%s succeeded\n", szshader );
    }

    pGraph->Stop();
    pGraph->Teardown();

    delete [] vAVector;
	Graph::DestroyGraph(pGraph);
    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;

    delete pTimer;

    PTask::Runtime::Terminate();

    return 0;
}

int run_general_iteration_task_single_node(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    ) 
{
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    assert(rows == 1 || cols == 1);		// vector, not matrix!
    int n = rows * cols;
    float* vAVector = random_vector(n);

    VECADD_PARAMS params;
    float fScalar = 2.0f;
    params.N = n; 
    UINT uiParamsStride = sizeof(VECADD_PARAMS);
    UINT stride = sizeof(float);
    UINT nPulse = 1;

    float* vAReference = vector_scale_repeatedly(vAVector, fScalar, n, iterations);

    int nChannelCount = 0;

    CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
    pTimer->reset();

    Graph * pGraph = new Graph();
    UINT uiUidCounter		= 0;

    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
    DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", stride, PTPARM_FLOAT);
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
    DatablockTemplate * pIterTemplate	= PTask::Runtime::GetDatablockTemplate("dbiter", sizeof(int), PTPARM_INT);
    CompiledKernel * pKernel		    = COMPILE_KERNEL(szfile, szshader);


    Port ** pTopInputPorts = NULL;
    Port ** pTopOutputPorts = NULL;
    {
        const UINT uiInputCount = 4;
        const UINT uiOutputCount = 1;

        pTopInputPorts = new Port*[uiInputCount];
        pTopOutputPorts = new Port*[uiOutputCount];

        pTopInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_top", 0, 0);
        pTopInputPorts[1]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_top", 1);
        pTopInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_top", 2);
        pTopInputPorts[3] = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "N_top", 3);
        pTopOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_top", 0);

        pTopInputPorts[3]->SetMetaFunction(MF_GENERAL_ITERATOR);

        Task * pTopTask = pGraph->AddTask(pKernel, 
            uiInputCount,
            pTopInputPorts,
            uiOutputCount,
            pTopOutputPorts,
            "top");

        assert(pTopTask);
        pTopTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pTopTask->SetBlockAndGridSize(gridSize, threadBlockSize);
    }

    Port ** pBottomInputPorts = NULL;
    Port ** pBottomOutputPorts = NULL;

    GraphInputChannel * pAInputChannel				= pGraph->AddInputChannel(pTopInputPorts[0], "AInputChannel", TRUE);
    GraphInputChannel * pScalarInputChannel1		= pGraph->AddInputChannel(pTopInputPorts[1], "ScalarInputChannel1");
    GraphInputChannel * pScalarInputChannel2 = NULL;

    GraphInputChannel * pIterationCount         = pGraph->AddInputChannel(pTopInputPorts[3], "IterationChannel");
    InternalChannel * pBackChannel = NULL;
    GraphOutputChannel * pAOutputChannel		    = NULL;

    pBackChannel = pGraph->AddInternalChannel(pTopOutputPorts[0], pTopInputPorts[0], "back-channel");
    pAOutputChannel = pGraph->AddOutputChannel(pTopOutputPorts[0], "outputChannel_left_A");

    pGraph->BindDescriptorPort(pTopInputPorts[0], pTopInputPorts[2]);
    pGraph->BindControlPropagationPort(pTopInputPorts[3], pTopOutputPorts[0]);
    pBackChannel->SetPredicationType(CE_SRC, CGATEFN_CLOSE_ON_ENDITERATION);
    pAOutputChannel->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_ENDITERATION);

    // Output the graph in graphviz 'dot' format.
    // (Use dot -Tpng GeneralIterationGraph.dot -o GeneralIterationGraph.png to render as PNG image.)
    pGraph->WriteDOTFile("GeneralIterationGraph.dot");

    pGraph->Run();

    Datablock * pScPrm1	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pScalarInputChannel1);
    pScalarInputChannel1->Push(pScPrm1);
    pScPrm1->Release();

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pAInputChannel);
    Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), pIterationCount);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();

    pAInputChannel->Push(pA);
    pIterationCount->Push(pIters);
    pA->Release();
    pIters->Release();

    Datablock * pLAout = pAOutputChannel->Pull();

    float * psrc = NULL;
    int nFailures = 0;
    pLAout->Lock();
    psrc = (float*) pLAout->GetDataPointer(FALSE);
    if(!compare_vectors(vAReference, psrc, n))
        nFailures++;
    pLAout->Unlock();    

    if(nFailures) {
        printf("failure.\n");
    } else  {
        printf( "%s succeeded\n", szshader );
    }

    pGraph->Stop();
    pGraph->Teardown();

    delete [] vAVector;
	Graph::DestroyGraph(pGraph);
    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;

    delete pTimer;

    PTask::Runtime::Terminate();

    return 0;
}



int run_scoped_iteration_task(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    ) 
{
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    assert(rows == 1 || cols == 1);		// vector, not matrix!
    int n = rows * cols;
    float* vAVector = random_vector(n);

    VECADD_PARAMS params;
    float fScalar = 2.0f;
    params.N = n; 
    UINT uiParamsStride = sizeof(VECADD_PARAMS);
    UINT stride = sizeof(float);
    UINT nPulse = 1;

    int totalNumScalings = iterations*2;
    float* vAReference = vector_scale_repeatedly(vAVector, fScalar, n, totalNumScalings);

    int nChannelCount = 0;

    CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
    pTimer->reset();

    Graph * pGraph = new Graph();
    UINT uiUidCounter		= 0;
    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
    DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", stride, PTPARM_FLOAT);
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
    DatablockTemplate * pIterTemplate	= PTask::Runtime::GetDatablockTemplate("dbiter", sizeof(int), PTPARM_INT);
    CompiledKernel * pKernel		    = COMPILE_KERNEL(szfile, szshader);


    Port ** pTopInputPorts = NULL;
    Port ** pTopOutputPorts = NULL;
    {
        const UINT uiInputCount = 4;
        const UINT uiOutputCount = 2;

        pTopInputPorts = new Port*[uiInputCount];
        pTopOutputPorts = new Port*[uiOutputCount];

        pTopInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_top", 0, 0);
        pTopInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_top", 1, 1);
        pTopInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_top", 2);
        pTopInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_top", 3);
        pTopOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_top", 0);
        pTopOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B(out)_top", 1);

        Task * pTopTask = pGraph->AddTask(pKernel, 
            uiInputCount,
            pTopInputPorts,
            uiOutputCount,
            pTopOutputPorts,
            "top");

        assert(pTopTask);
        pTopTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pTopTask->SetBlockAndGridSize(gridSize, threadBlockSize);
    }

    Port ** pBottomInputPorts = NULL;
    Port ** pBottomOutputPorts = NULL;
    {
        const UINT uiInputCount = 5;
        const UINT uiOutputCount = 2;

        pBottomInputPorts = new Port*[uiInputCount];
        pBottomOutputPorts = new Port*[uiOutputCount];

        UINT uiUidCounter		= 0;
        pBottomInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_Bottom", 0, 0);
        pBottomInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_Bottom", 1, 1);
        pBottomInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_Bottom", 2);
        pBottomInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_bottom", 3);
        pBottomInputPorts[4]    = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "Meta_bottom", 4);
        pBottomOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_Bottom", 0);
        pBottomOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B(out)_Bottom", 1);

        pBottomInputPorts[4]->SetMetaFunction(MF_GENERAL_ITERATOR);

        Task * pBottomTask = pGraph->AddTask(pKernel, 
            uiInputCount,
            pBottomInputPorts,
            uiOutputCount,
            pBottomOutputPorts,
            "bottom");

        assert(pBottomTask);
        pBottomTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pBottomTask->SetBlockAndGridSize(gridSize, threadBlockSize);
    }

    // Set up graph input channels.
    GraphInputChannel * pAInputChannel				= pGraph->AddInputChannel(pTopInputPorts[0], "AInputChannel", TRUE); // A switch channel
    GraphInputChannel * pBInputChannel				= pGraph->AddInputChannel(pTopInputPorts[1], "BInputChannel", FALSE); // not A switch channel (no back channel)
    GraphInputChannel * pScalarInputChannelTop		= pGraph->AddInputChannel(pTopInputPorts[2], "ScalarInputChannelTop");
    GraphInputChannel * pScalarInputChannelBottom   = pGraph->AddInputChannel(pBottomInputPorts[2], "ScalarInputChannelBottom");
    GraphInputChannel * pIterationCount             = pGraph->AddInputChannel(pBottomInputPorts[4], "IterationChannel");

    // Bind descriptor ports.
    // For all the tasks, port 0 (A) drives the value of port 3 (N).
    pGraph->BindDescriptorPort(pTopInputPorts[0], pTopInputPorts[3]);
    pGraph->BindDescriptorPort(pBottomInputPorts[0], pBottomInputPorts[3]);

    // Set up internal channels (including back channel for A from Bottom to Top task) and output channel,
    // don't set up a back channel for input B. Let the scoped iteration cause it to be replayed at each input.
    InternalChannel * pBackChannelA                 = pGraph->AddInternalChannel(pBottomOutputPorts[0], pTopInputPorts[0], "back-channelA");
    GraphOutputChannel * pOutputChannelA		    = pGraph->AddOutputChannel(pBottomOutputPorts[0], "outputChannel_left_A");
    GraphOutputChannel * pOutputChannelB		    = pGraph->AddOutputChannel(pBottomOutputPorts[1], "outputChannel_left_B");
    InternalChannel * pTopToBottomA                  = pGraph->AddInternalChannel(pTopOutputPorts[0], pBottomInputPorts[0], "A_TopToBottom");
    InternalChannel * pTopToBottomB                  = pGraph->AddInternalChannel(pTopOutputPorts[1], pBottomInputPorts[1], "B_TopToBottom");

    // Configure looping.
    pGraph->BindControlPropagationPort(pBottomInputPorts[4], pBottomOutputPorts[0]);
    pGraph->BindControlPropagationPort(pBottomInputPorts[4], pBottomOutputPorts[1]);
    pGraph->BindIterationScope(pBottomInputPorts[4], pTopInputPorts[1]);
    pBackChannelA->SetPredicationType(CE_SRC, CGATEFN_CLOSE_ON_ENDITERATION);
    pOutputChannelA->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_ENDITERATION);
    pOutputChannelB->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_ENDITERATION);

    perform_graph_diagnostics(pGraph, "ScopedIteration.dot");

    pGraph->Run();

    Datablock * pScalarParamTop		 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pScalarInputChannelTop);
    Datablock * pScalarParamBottom	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pScalarInputChannelBottom);
    pScalarInputChannelTop->Push(pScalarParamTop);
    pScalarInputChannelBottom->Push(pScalarParamBottom);
    pScalarParamTop->Release();
    pScalarParamBottom->Release();

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pAInputChannel);
    Datablock * pB		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pBInputChannel);
    Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), pIterationCount);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();

    pAInputChannel->Push(pA);
    pBInputChannel->Push(pB);
    pIterationCount->Push(pIters);
    pA->Release();
    pB->Release();
    pIters->Release();

    Datablock * pLAout = pOutputChannelA->Pull();
    Datablock * pLBout = pOutputChannelB->Pull();

    int nFailures = 0;
    float * psrcA = NULL;
    pLAout->Lock();
    psrcA = (float*) pLAout->GetDataPointer(FALSE);
    if(!compare_vectors(vAReference, psrcA, n))
        nFailures++;
    pLAout->Unlock();    
    float * psrcB = NULL;
    pLBout->Lock();
    psrcB = (float*) pLBout->GetDataPointer(FALSE);
    if(!compare_vectors(vAReference, psrcB, n))
        nFailures++;
    pLBout->Unlock();    

    if(nFailures) {
        printf("failure.\n");
    } else  {
        printf( "%s succeeded\n", szshader );
    }

    pGraph->Stop();
    pGraph->Teardown();

    delete [] vAVector;
	Graph::DestroyGraph(pGraph);
    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;

    delete pTimer;

    PTask::Runtime::Terminate();

    return 0;
}