///-------------------------------------------------------------------------------------------------
// file:	rttests.cpp
//
// summary:	Declares simple test cases for PTask like init/teardown
///-------------------------------------------------------------------------------------------------

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
#include "Scheduler.h"

using namespace std;
using namespace PTask;

extern float * random_vector(int n);
extern BOOL compare_vectors(float*pA, float*pB, int n);
extern float * vector_scale(float * pA, float scalar, int n);
extern float * vector_scale_repeatedly(float * pA, float scalar, int n, int nIterations);
extern void vector_scale_in_place(float * pA, float scalar, int n);
extern void perform_graph_diagnostics(Graph * pGraph,
                                      char * lpszDotFileName,
                                      BOOL bForceGraphDisplay
                                      );

int run_init_teardown_test(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    ) 
{
    PTask::Runtime::Initialize();
    PTask::Runtime::Terminate();
    return 0;
}

int run_init_teardown_test_with_objects_no_run(	
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
    
    VECADD_PARAMS params;
    float fScalar = 2.0f;
    params.N = n; 
    UINT uiParamsStride = sizeof(VECADD_PARAMS);
    UINT stride = sizeof(float);
    UINT nPulse = 1;

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
    pTopInputPorts[3] = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "iter_meta", 3);
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

    pGraph->Teardown();

	Graph::DestroyGraph(pGraph);
    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;

    PTask::Runtime::Terminate();

    return 0;
}


int run_init_teardown_test_with_run(	
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
    
    VECADD_PARAMS params;
    float fScalar = 2.0f;
    params.N = n; 
    UINT uiParamsStride = sizeof(VECADD_PARAMS);
    UINT stride = sizeof(float);
    UINT nPulse = 1;

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
    pTopInputPorts[3] = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "iter_meta", 3);
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

    PTask::Runtime::RunGraph(pGraph);

    pGraph->Stop();
    pGraph->Teardown();

	Graph::DestroyGraph(pGraph);
    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;

    PTask::Runtime::Terminate();

    return 0;
}


int run_init_teardown_test_with_single_push(	
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
    
    VECADD_PARAMS params;
    float fScalar = 2.0f;
    params.N = n; 
    UINT uiParamsStride = sizeof(VECADD_PARAMS);
    UINT stride = sizeof(float);
    UINT nPulse = 1;

    float* vAVector = random_vector(n);
    float* vBVector = random_vector(n);

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
    pTopInputPorts[3] = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "iter_meta", 3);
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

    PTask::Runtime::RunGraph(pGraph);

    Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pTopAxBScaleInput);
    pTopAxBScaleInput->Push(pScPrm);
    pScPrm->Release();

    pGraph->Stop();
    pGraph->Teardown();

	Graph::DestroyGraph(pGraph);

    PTask::Runtime::Terminate();

    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;
    delete [] vAVector;
    delete [] vBVector;

    return 0;
}

int run_init_teardown_test_with_multi_push(	
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
    
    VECADD_PARAMS params;
    float fScalar = 2.0f;
    params.N = n; 
    UINT uiParamsStride = sizeof(VECADD_PARAMS);
    UINT stride = sizeof(float);
    UINT nPulse = 1;

    float* vAVector = random_vector(n);
    float* vBVector = random_vector(n);

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
    pTopInputPorts[3] = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "iter_meta", 3);
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

    PTask::Runtime::RunGraph(pGraph);

    Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pTopAxBScaleInput);
    pTopAxBScaleInput->Push(pScPrm);
    pScPrm->Release();

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pTopAInput);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();

    PTask::Runtime::Push(pTopAInput, pA);
    pA->Release();

    pGraph->Stop();
    pGraph->Teardown();

	Graph::DestroyGraph(pGraph);
    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;
    delete [] vAVector; 
    delete [] vBVector;

    PTask::Runtime::Terminate();

    return 0;
}

int run_init_teardown_test_with_all_push_no_pull(	
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
    pTopInputPorts[3] = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "iter_meta", 3);
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

    PTask::Runtime::RunGraph(pGraph);

    Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pTopAxBScaleInput);
    pTopAxBScaleInput->Push(pScPrm);
    pScPrm->Release();

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pTopAInput);
    Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), pIterationCount);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();

    PTask::Runtime::Push(pTopAInput, pA);
    PTask::Runtime::Push(pIterationCount, pIters);
    pA->Release();
    pIters->Release();


    Sleep(500);

    pGraph->Stop();
    pGraph->Teardown();

	Graph::DestroyGraph(pGraph);
    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;
    delete [] vAVector;
    delete [] vBVector;

    PTask::Runtime::Terminate();

    return 0;
}

int run_init_teardown_test_pull_and_discard(	
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
    pTopInputPorts[3] = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "iter_meta", 3);
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

    PTask::Runtime::RunGraph(pGraph);

    Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pTopAxBScaleInput);
    pTopAxBScaleInput->Push(pScPrm);
    pScPrm->Release();

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pTopAInput);
    Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), pIterationCount);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();

    PTask::Runtime::Push(pTopAInput, pA);
    PTask::Runtime::Push(pIterationCount, pIters);
    pA->Release();
    pIters->Release();

    Datablock * pLAout = PTask::Runtime::Pull(pAOutputA);
    pLAout->Release();

    pGraph->Stop();
    pGraph->Teardown();

	Graph::DestroyGraph(pGraph);
    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;
    delete [] vAVector;
    delete [] vBVector;

    PTask::Runtime::Terminate();

    return 0;
}


int run_init_teardown_test_pull_and_open(	
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
    pTopInputPorts[3] = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "iter_meta", 3);
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

    PTask::Runtime::RunGraph(pGraph);

    Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pTopAxBScaleInput);
    pTopAxBScaleInput->Push(pScPrm);
    pScPrm->Release();

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pTopAInput);
    Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), pIterationCount);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();

    PTask::Runtime::Push(pTopAInput, pA);
    PTask::Runtime::Push(pIterationCount, pIters);
    pA->Release();
    pIters->Release();

    Datablock * pLAout = PTask::Runtime::Pull(pAOutputA);

    pLAout->Lock();
    float * psrc = (float*) pLAout->GetDataPointer(FALSE);
    pLAout->Unlock();    
    pLAout->Release();

    pGraph->Stop();
    pGraph->Teardown();

	Graph::DestroyGraph(pGraph);
    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;
    delete [] vAVector;
    delete [] vBVector;

    PTask::Runtime::Terminate();

    return 0;
}

int run_multi_init_test(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    )
{
    for(int i=0; i<iterations; i++) {
        PTask::Runtime::Initialize();
        PTask::Runtime::Terminate();
    }
    return 0;
}

int run_multi_init_teardown_test(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    )
{

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

    for(int i=0; i<iterations; i++) {
        PTask::Runtime::Initialize();
        CheckPlatformSupport(szfile, szshader);

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
        pTopInputPorts[3] = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "iter_meta", 3);
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

        PTask::Runtime::RunGraph(pGraph);

        Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pTopAxBScaleInput);
        pTopAxBScaleInput->Push(pScPrm);
        pScPrm->Release();

        Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pTopAInput);
        Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), pIterationCount);

        pA->Lock();
        pA->SetRecordCount(params.N);
        pA->Unlock();

        PTask::Runtime::Push(pTopAInput, pA);
        PTask::Runtime::Push(pIterationCount, pIters);
        pA->Release();
        pIters->Release();

        Datablock * pLAout = PTask::Runtime::Pull(pAOutputA);

        pLAout->Lock();
        float * psrc = (float*) pLAout->GetDataPointer(FALSE);
        pLAout->Unlock();    
        pLAout->Release();

        pGraph->Stop();
        pGraph->Teardown();
	    Graph::DestroyGraph(pGraph);
        delete [] pTopInputPorts;
        delete [] pTopOutputPorts;

        PTask::Runtime::Terminate();
    }

    delete [] vAVector;
    delete [] vBVector;
    return 0;
}

int run_multi_graph_test(	
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
    for(int i=0; i<iterations; i++) {

        CheckPlatformSupport(szfile, szshader);

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
        pTopInputPorts[3] = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "iter_meta", 3);
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

        PTask::Runtime::RunGraph(pGraph);

        Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pTopAxBScaleInput);
        pTopAxBScaleInput->Push(pScPrm);
        pScPrm->Release();

        Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pTopAInput);
        Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), pIterationCount);

        pA->Lock();
        pA->SetRecordCount(params.N);
        pA->Unlock();

        PTask::Runtime::Push(pTopAInput, pA);
        PTask::Runtime::Push(pIterationCount, pIters);
        pA->Release();
        pIters->Release();

        Datablock * pLAout = PTask::Runtime::Pull(pAOutputA);

        pLAout->Lock();
        float * psrc = (float*) pLAout->GetDataPointer(FALSE);
        pLAout->Unlock();    
        pLAout->Release();

        pGraph->Stop();
        pGraph->Teardown();
	    Graph::DestroyGraph(pGraph);
        delete [] pTopInputPorts;
        delete [] pTopOutputPorts;

    }

    delete [] vAVector;
    delete [] vBVector;
    PTask::Runtime::Terminate();
    return 0;
}

int run_concurrent_graph_test(	
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

    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
    DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", stride, PTPARM_FLOAT);
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
    DatablockTemplate * pIterTemplate	= PTask::Runtime::GetDatablockTemplate("dbiter", sizeof(int), PTPARM_INT);
    CompiledKernel * pKernel		    = COMPILE_KERNEL(szfile, szshader);
    Graph ** ppGraphs = new Graph*[iterations];
    Port *** ppTopInputPorts = new Port**[iterations];
    Port *** ppTopOutputPorts = new Port**[iterations];
    Channel ** ppTopAInput		  = new Channel*[iterations];
    Channel ** ppTopAxBScaleInput = new Channel*[iterations];
    Channel ** ppIterationCount   = new Channel*[iterations]; 
    Channel ** ppAOutputA		  = new Channel*[iterations];

    for(int i=0; i<iterations; i++) {

        ppGraphs[i] = new Graph();

        const UINT uiInputCount = 4;
        const UINT uiOutputCount = 1;
    
        ppTopInputPorts[i] = new Port*[uiInputCount];
        ppTopOutputPorts[i] = new Port*[uiOutputCount];

        UINT uiUidCounter		= 0;
        ppTopInputPorts[i][0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_top", 0, 0);
        ppTopInputPorts[i][1]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_top", 1);
        ppTopInputPorts[i][2]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_top", 2);
        ppTopInputPorts[i][3] = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "iter_meta", 3);
        ppTopOutputPorts[i][0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_top", 0);

        PTask::Runtime::SetPortMetaFunction(ppTopInputPorts[i][3], MF_SIMPLE_ITERATOR);
        Task * pTopTask = PTask::Runtime::AddTask(ppGraphs[i],
                                                  pKernel, 
                                                  uiInputCount,
                                                  ppTopInputPorts[i],
                                                  uiOutputCount,
                                                  ppTopOutputPorts[i],
                                                  "top");


        assert(pTopTask);
        pTopTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pTopTask->SetBlockAndGridSize(gridSize, threadBlockSize);

        ppTopAInput[i]	      = ppGraphs[i]->AddInputChannel(ppTopInputPorts[i][0], "AInputChannel");
        ppTopAxBScaleInput[i] = ppGraphs[i]->AddInputChannel(ppTopInputPorts[i][1], "ScalarChannel");
        ppIterationCount[i]   = ppGraphs[i]->AddInputChannel(ppTopInputPorts[i][3], "IterationChannel");
        ppAOutputA[i]	      = ppGraphs[i]->AddOutputChannel(ppTopOutputPorts[i][0], "outputChannel_left_A");
        PTask::Runtime::BindDerivedPort(ppGraphs[i], ppTopInputPorts[i][0], ppTopInputPorts[i][2]);

        PTask::Runtime::RunGraph(ppGraphs[i]);

    }

    // now we've got a bunch of graphs running. feed them and tear them down. 
    
    for(int i=0; i<iterations; i++) {

        Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), ppTopAxBScaleInput[i]);
        ppTopAxBScaleInput[i]->Push(pScPrm);
        pScPrm->Release();

        Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, ppTopAInput[i]);
        Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), ppIterationCount[i]);

        pA->Lock();
        pA->SetRecordCount(params.N);
        pA->Unlock();

        PTask::Runtime::Push(ppTopAInput[i], pA);
        PTask::Runtime::Push(ppIterationCount[i], pIters);
        pA->Release();
        pIters->Release();

        Datablock * pLAout = PTask::Runtime::Pull(ppAOutputA[i]);

        pLAout->Lock();
        float * psrc = (float*) pLAout->GetDataPointer(FALSE);
        pLAout->Unlock();    
        pLAout->Release();

        ppGraphs[i]->Stop();
        ppGraphs[i]->Teardown();
	    Graph::DestroyGraph(ppGraphs[i]);
        delete [] ppTopInputPorts[i];
        delete [] ppTopOutputPorts[i];

    }

    delete [] ppGraphs;
    delete [] ppTopInputPorts;
    delete [] ppTopOutputPorts;
    delete [] ppTopAInput;
    delete [] ppTopAxBScaleInput;
    delete [] ppIterationCount;
    delete [] ppAOutputA;
    delete [] vAVector;
    delete [] vBVector;
    PTask::Runtime::Terminate();
    return 0;
}


int run_ultra_concurrent_graph_test(	
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

    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
    DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", stride, PTPARM_FLOAT);
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
    DatablockTemplate * pIterTemplate	= PTask::Runtime::GetDatablockTemplate("dbiter", sizeof(int), PTPARM_INT);
    CompiledKernel * pKernel		    = COMPILE_KERNEL(szfile, szshader);
    Graph ** ppGraphs = new Graph*[iterations];
    Port *** ppTopInputPorts = new Port**[iterations];
    Port *** ppTopOutputPorts = new Port**[iterations];
    Channel ** ppTopAInput		  = new Channel*[iterations];
    Channel ** ppTopAxBScaleInput = new Channel*[iterations];
    Channel ** ppIterationCount   = new Channel*[iterations]; 
    Channel ** ppAOutputA		  = new Channel*[iterations];

    for(int i=0; i<iterations; i++) {

        ppGraphs[i] = new Graph();

        const UINT uiInputCount = 4;
        const UINT uiOutputCount = 1;
    
        ppTopInputPorts[i] = new Port*[uiInputCount];
        ppTopOutputPorts[i] = new Port*[uiOutputCount];

        UINT uiUidCounter		= 0;
        ppTopInputPorts[i][0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_top", 0, 0);
        ppTopInputPorts[i][1]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_top", 1);
        ppTopInputPorts[i][2]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_top", 2);
        ppTopInputPorts[i][3] = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "iter_meta", 3);
        ppTopOutputPorts[i][0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_top", 0);

        PTask::Runtime::SetPortMetaFunction(ppTopInputPorts[i][3], MF_SIMPLE_ITERATOR);
        Task * pTopTask = PTask::Runtime::AddTask(ppGraphs[i],
                                                  pKernel, 
                                                  uiInputCount,
                                                  ppTopInputPorts[i],
                                                  uiOutputCount,
                                                  ppTopOutputPorts[i],
                                                  "top");


        assert(pTopTask);
        pTopTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pTopTask->SetBlockAndGridSize(gridSize, threadBlockSize);

        ppTopAInput[i]	      = ppGraphs[i]->AddInputChannel(ppTopInputPorts[i][0], "AInputChannel");
        ppTopAxBScaleInput[i] = ppGraphs[i]->AddInputChannel(ppTopInputPorts[i][1], "ScalarChannel");
        ppIterationCount[i]   = ppGraphs[i]->AddInputChannel(ppTopInputPorts[i][3], "IterationChannel");
        ppAOutputA[i]	      = ppGraphs[i]->AddOutputChannel(ppTopOutputPorts[i][0], "outputChannel_left_A");
        PTask::Runtime::BindDerivedPort(ppGraphs[i], ppTopInputPorts[i][0], ppTopInputPorts[i][2]);

        PTask::Runtime::RunGraph(ppGraphs[i]);

    }

    // now we've got a bunch of graphs running. feed them and tear them down. 
    
    for(int i=0; i<iterations; i++) {

        Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), ppTopAxBScaleInput[i]);
        ppTopAxBScaleInput[i]->Push(pScPrm);
        pScPrm->Release();

        Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, ppTopAInput[i]);
        Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), ppIterationCount[i]);

        pA->Lock();
        pA->SetRecordCount(params.N);
        pA->Unlock();

        PTask::Runtime::Push(ppTopAInput[i], pA);
        PTask::Runtime::Push(ppIterationCount[i], pIters);
        pA->Release();
        pIters->Release();

    }
    Sleep(100);

    for(int i=iterations-1; i>0; i--) {

        Datablock * pLAout = PTask::Runtime::Pull(ppAOutputA[i]);

        pLAout->Lock();
        float * psrc = (float*) pLAout->GetDataPointer(FALSE);
        pLAout->Unlock();    
        pLAout->Release();
        Sleep(10);
    }

    for(int i=iterations-1; i>=0; i--) {

        ppGraphs[i]->Stop();
        ppGraphs[i]->Teardown();
	    Graph::DestroyGraph(ppGraphs[i]);
        delete [] ppTopInputPorts[i];
        delete [] ppTopOutputPorts[i];

    }

    delete [] ppGraphs;
    delete [] ppTopInputPorts;
    delete [] ppTopOutputPorts;
    delete [] ppTopAInput;
    delete [] ppTopAxBScaleInput;
    delete [] ppIterationCount;
    delete [] ppAOutputA;
    delete [] vAVector;
    delete [] vBVector;
    PTask::Runtime::Terminate();
    return 0;
}

int run_extreme_concurrent_graph_test(	
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

    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
    DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", stride, PTPARM_FLOAT);
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
    DatablockTemplate * pIterTemplate	= PTask::Runtime::GetDatablockTemplate("dbiter", sizeof(int), PTPARM_INT);
    CompiledKernel * pKernel		    = COMPILE_KERNEL(szfile, szshader);
    Graph ** ppGraphs = new Graph*[iterations];
    Port *** ppTopInputPorts = new Port**[iterations];
    Port *** ppTopOutputPorts = new Port**[iterations];
    Channel ** ppTopAInput		  = new Channel*[iterations];
    Channel ** ppTopAxBScaleInput = new Channel*[iterations];
    Channel ** ppIterationCount   = new Channel*[iterations]; 
    Channel ** ppAOutputA		  = new Channel*[iterations];

    for(int i=0; i<iterations; i++) {

        ppGraphs[i] = new Graph();

        const UINT uiInputCount = 4;
        const UINT uiOutputCount = 1;
    
        ppTopInputPorts[i] = new Port*[uiInputCount];
        ppTopOutputPorts[i] = new Port*[uiOutputCount];

        UINT uiUidCounter		= 0;
        ppTopInputPorts[i][0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_top", 0, 0);
        ppTopInputPorts[i][1]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_top", 1);
        ppTopInputPorts[i][2]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_top", 2);
        ppTopInputPorts[i][3] = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "iter_meta", 3);
        ppTopOutputPorts[i][0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_top", 0);

        PTask::Runtime::SetPortMetaFunction(ppTopInputPorts[i][3], MF_SIMPLE_ITERATOR);
        Task * pTopTask = PTask::Runtime::AddTask(ppGraphs[i],
                                                  pKernel, 
                                                  uiInputCount,
                                                  ppTopInputPorts[i],
                                                  uiOutputCount,
                                                  ppTopOutputPorts[i],
                                                  "top");


        assert(pTopTask);
        pTopTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pTopTask->SetBlockAndGridSize(gridSize, threadBlockSize);

        ppTopAInput[i]	      = ppGraphs[i]->AddInputChannel(ppTopInputPorts[i][0], "AInputChannel");
        ppTopAxBScaleInput[i] = ppGraphs[i]->AddInputChannel(ppTopInputPorts[i][1], "ScalarChannel");
        ppIterationCount[i]   = ppGraphs[i]->AddInputChannel(ppTopInputPorts[i][3], "IterationChannel");
        ppAOutputA[i]	      = ppGraphs[i]->AddOutputChannel(ppTopOutputPorts[i][0], "outputChannel_left_A");
        PTask::Runtime::BindDerivedPort(ppGraphs[i], ppTopInputPorts[i][0], ppTopInputPorts[i][2]);

        PTask::Runtime::RunGraph(ppGraphs[i]);

    }

    // now we've got a bunch of graphs running. feed them and tear them down. 
    
    for(int i=0; i<iterations; i++) {

        Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), ppTopAxBScaleInput[i]);
        ppTopAxBScaleInput[i]->Push(pScPrm);
        pScPrm->Release();

        Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, ppTopAInput[i]);
        Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), ppIterationCount[i]);

        pA->Lock();
        pA->SetRecordCount(params.N);
        pA->Unlock();

        PTask::Runtime::Push(ppTopAInput[i], pA);
        PTask::Runtime::Push(ppIterationCount[i], pIters);
        pA->Release();
        pIters->Release();

        Datablock * pLAout = PTask::Runtime::Pull(ppAOutputA[i]);

        pLAout->Lock();
        float * psrc = (float*) pLAout->GetDataPointer(FALSE);
        pLAout->Unlock();    
        pLAout->Release();

        ppGraphs[i]->Stop();
        ppGraphs[i]->Teardown();
	    Graph::DestroyGraph(ppGraphs[i]);
        delete [] ppTopInputPorts[i];
        delete [] ppTopOutputPorts[i];

    }

    delete [] ppGraphs;
    delete [] ppTopInputPorts;
    delete [] ppTopOutputPorts;
    delete [] ppTopAInput;
    delete [] ppTopAxBScaleInput;
    delete [] ppIterationCount;
    delete [] ppAOutputA;
    delete [] vAVector;
    delete [] vBVector;
    PTask::Runtime::Terminate();
    return 0;
}


int run_accelerator_disable_test(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    )
{
    int nErrors = 0;
    PTRESULT pt = PTASK_OK;
    PTask::Runtime::Initialize();
    ACCELERATOR_CLASS accClass = ptaskutils::SelectAcceleratorClass(szfile);
    
    UINT uiIndex = 0;
    ACCELERATOR_DESCRIPTOR * pDescriptor;
    std::vector<ACCELERATOR_DESCRIPTOR*> descriptors;
    std::vector<ACCELERATOR_DESCRIPTOR*>::iterator vi;
    while(PTASK_ERR_NOT_FOUND != PTask::Runtime::EnumerateAccelerators(accClass, uiIndex, &pDescriptor)) {
        descriptors.push_back(pDescriptor);
        uiIndex++;
    }
    
    for(vi=descriptors.begin(); vi!=descriptors.end(); vi++) {
        
        // for each available device, try disabling it, check that it is not in the available list,
        // re-enable it, and check that it is in the available device list. 
        ACCELERATOR_DESCRIPTOR * pAccDesc = *vi;
        Accelerator * pAccelerator = reinterpret_cast<Accelerator*>(pAccDesc->pAccelerator);
        pt = PTask::Runtime::DynamicDisableAccelerator(pAccDesc);
        if(!PTSUCCESS(pt)) {
            nErrors++;
            printf("PTask::Runtime::DisableAccelerator(pAccDesc) returned %d!\n", pt);
            continue;
        }
        
        set<Accelerator*> vaccs;
        Scheduler::EnumerateAvailableAccelerators(accClass, vaccs);
        if(vaccs.find(pAccelerator) != vaccs.end()) {
            nErrors++;
            std::cout 
                << "run_accelerator_disable_test FAILED! " 
                << pAccelerator
                << " found in available list after ostensible disable!"
                << std::endl;
        }
        vaccs.clear();

        pt = PTask::Runtime::DynamicEnableAccelerator(pAccDesc);
        if(!PTSUCCESS(pt)) {
            nErrors++;
            printf("PTask::Runtime::EnableAccelerator(pAccDesc) returned %d!\n", pt);
            continue;
        }
        Scheduler::EnumerateAvailableAccelerators(accClass, vaccs);
        if(vaccs.find(pAccelerator) == vaccs.end()) {
            nErrors++;
            std::cout 
                << "run_accelerator_enable_test FAILED! " 
                << pAccelerator
                << " not found in available list after ostensible enable!"
                << std::endl;
        }
        vaccs.clear();        

    }

    // now disable them all, see that there are none,
    // reenable them all and see that all the known accelerators
    // are present.
    for(vi=descriptors.begin(); vi!=descriptors.end(); vi++) {
        
        // for each available device, try disabling it, check that it is not in the available list,
        // re-enable it, and check that it is in the available device list. 
        ACCELERATOR_DESCRIPTOR * pAccDesc = *vi;
        Accelerator * pAccelerator = reinterpret_cast<Accelerator*>(pAccDesc->pAccelerator);
        pt = PTask::Runtime::DynamicDisableAccelerator(pAccDesc);
        if(!PTSUCCESS(pt)) {
            nErrors++;
            printf("PTask::Runtime::DisableAccelerator(pAccDesc) returned %d!\n", pt);
            continue;
        }
    }
        
    set<Accelerator*> vaccs;
    Scheduler::EnumerateAvailableAccelerators(accClass, vaccs);
    if(vaccs.size()>0) {
        nErrors++;
        std::cout 
            << "run_accelerator_disable_test FAILED! " 
            << vaccs.size()
            << " accelerators found in available list after all disabled!"
            << std::endl;
    }
    vaccs.clear();

    for(vi=descriptors.begin(); vi!=descriptors.end(); vi++) {

        ACCELERATOR_DESCRIPTOR * pAccDesc = *vi;
        Accelerator * pAccelerator = reinterpret_cast<Accelerator*>(pAccDesc->pAccelerator);
        pt = PTask::Runtime::DynamicEnableAccelerator(pAccDesc);
        if(!PTSUCCESS(pt)) {
            nErrors++;
            printf("PTask::Runtime::EnableAccelerator(pAccDesc) returned %d!\n", pt);
            continue;
        }
    }

    Scheduler::EnumerateAvailableAccelerators(accClass, vaccs);
    if(vaccs.size() != descriptors.size()) {
        nErrors++;
        std::cout 
            << "run_accelerator_enable_test FAILED! " 
            << vaccs.size()
            << " found after ostensible enable, wanted "
            << descriptors.size()
            << "!"
            << std::endl;
    }
    vaccs.clear();        

    if(nErrors == 0)
        printf("run_accelerator_disable_test succeeded!\n");

    for(vi=descriptors.begin(); vi!=descriptors.end(); vi++) {
        free(*vi);
    }

    PTask::Runtime::Terminate();        
    return nErrors;
}

int run_mismatched_template_detection(	
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
    int nChannelCount = 0;

    int totalNumScalings = iterations*2 + 1;
    float* vAReference = vector_scale_repeatedly(vAVector, fScalar, n, totalNumScalings);

    CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
    pTimer->reset();

    Graph * pGraph = new Graph();
    UINT uiUidCounter		= 0;
    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
    DatablockTemplate * pData2Template	= PTask::Runtime::GetDatablockTemplate("dbV2_double", stride*2, n+3, 1, 1);
    DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", stride, PTPARM_FLOAT);
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
    DatablockTemplate * pIterTemplate	= PTask::Runtime::GetDatablockTemplate("dbiter", sizeof(int), PTPARM_INT);
    CompiledKernel * pKernel		    = COMPILE_KERNEL(szfile, szshader);

    Port ** pPreLoopInputPorts = NULL;
    Port ** pPreLoopOutputPorts = NULL;
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
        pBottomInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pData2Template, uiUidCounter++, "A_Bottom", 0, 0);
        pBottomInputPorts[1]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_Bottom", 1);
        pBottomInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_bottom", 2);
        pBottomInputPorts[3]    = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "Meta_bottom", 3);
        pBottomOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pData2Template, uiUidCounter++, "A(out)_Bottom", 0);

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
    pAInputChannel				= pGraph->AddInputChannel(pPreLoopInputPorts[0], "AInputChannel"); // Not a switch channel
    pScalarInputChannelPreLoop	= pGraph->AddInputChannel(pPreLoopInputPorts[1], "ScalarInputChannelPreLoop");
    GraphInputChannel * pScalarInputChannelTop		= pGraph->AddInputChannel(pTopInputPorts[1], "ScalarInputChannelTop");
    GraphInputChannel * pScalarInputChannelBottom   = pGraph->AddInputChannel(pBottomInputPorts[1], "ScalarInputChannelBottom");
    GraphInputChannel * pIterationCount             = pGraph->AddInputChannel(pBottomInputPorts[3], "IterationChannel");

    // Bind descriptor ports.
    // For all the tasks, port 0 (A) drives the value of port 2 (N).
    pGraph->BindDescriptorPort(pPreLoopInputPorts[0], pPreLoopInputPorts[2]);
    pGraph->BindDescriptorPort(pTopInputPorts[0], pTopInputPorts[2]);
    pGraph->BindDescriptorPort(pBottomInputPorts[0], pBottomInputPorts[2]);

    // Set up internal channels (including back channel for A from Bottom to Top task) and output channel.
    InternalChannel * pBackChannel          = pGraph->AddInternalChannel(pBottomOutputPorts[0], pTopInputPorts[0], "back-channel");
    GraphOutputChannel * pAOutputChannel	= pGraph->AddOutputChannel(pBottomOutputPorts[0], "outputChannel_left_A");
    InternalChannel * pPreLoopToTop         = pGraph->AddInternalChannel(pPreLoopOutputPorts[0], pTopInputPorts[0], "A_PreLoopToTop", TRUE); // A switch channel
    InternalChannel * pTopToBottom          = pGraph->AddInternalChannel(pTopOutputPorts[0], pBottomInputPorts[0], "A_TopToBottom");

    // Configure looping.
    pGraph->BindControlPropagationPort(pBottomInputPorts[3], pBottomOutputPorts[0]);
    pBackChannel->SetPredicationType(CE_SRC, CGATEFN_CLOSE_ON_ENDITERATION);
    pAOutputChannel->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_ENDITERATION);

    // the graph checker should find that the graph is malformed
    // because pBottomInputPorts[0]'s template doesn't match the 
    // template at the source end of it's connecting channel.
    PTRESULT ptr = pGraph->CheckGraphSemantics();
    BOOL bMalformed = !PTSUCCESS(ptr);
    if(!bMalformed) {
        printf("failure.\n");
    } else  {
        printf( "%s succeeded\n", szshader );
    }

    pGraph->Teardown();

    delete [] vAVector;
	Graph::DestroyGraph(pGraph);
    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;
    delete [] vAReference;

    delete pTimer;

    PTask::Runtime::Terminate();

    return 0;
}
