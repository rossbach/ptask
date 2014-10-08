//--------------------------------------------------------------------------------------
// File: tee.cpp
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
#include "gatedports.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"

using namespace std;
using namespace PTask;

extern float * random_vector(int n);
extern BOOL compare_vectors(float*pA, float*pB, int n);
extern float * vector_scale(float * pA, float scalar, int n);

int run_graph_cuda_tee_task(	
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
	params.N = n;
    float fScalar = 2.0f;
	UINT stride = sizeof(float);
    UINT nPulse = 1;

	int nChannelCount = 0;

	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	Graph * pGraph = new Graph();
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
	DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", sizeof(float), PTPARM_FLOAT);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
	CompiledKernel * pMatmulKernel		= COMPILE_KERNEL(szfile, szshader);
    
	const UINT uiInputCount = 3;
	const UINT uiOutputCount = 1;
	
	Port ** pAInputPorts = new Port*[uiInputCount];
	Port ** pAOutputPorts = new Port*[uiOutputCount];
	Port ** pBInputPorts = new Port*[uiInputCount];
	Port ** pBOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAInputPorts[0]	 = PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_A", 0, 0);
	pAInputPorts[1]	 = PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "A_scalar", 1);
	pAInputPorts[2]	 = PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "A_N", 2);
	pAOutputPorts[0] = PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A_A(out)", 0);
	pBInputPorts[0]	 = PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_A", 0, 0);
	pBInputPorts[1]	 = PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "B_scalar", 1);
	pBInputPorts[2]	 = PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "B_N", 2);
	pBOutputPorts[0] = PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B_A(out)", 0);

	Task * pATask = pGraph->AddTask(pMatmulKernel, 
									uiInputCount,
									pAInputPorts,
									uiOutputCount,
									pAOutputPorts,
									"A_AxScalar");

	Task * pBTask = pGraph->AddTask(pMatmulKernel, 
									uiInputCount,
									pBInputPorts,
									uiOutputCount,
									pBOutputPorts,
									"B_AxScalar");


	assert(pATask);
	assert(pBTask);
	pATask->SetComputeGeometry(n, 1, 1);
	pBTask->SetComputeGeometry(n, 1, 1);
	PTASKDIM3 threadBlockSize(256, 1, 1);
	PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
	pATask->SetBlockAndGridSize(gridSize, threadBlockSize);
	pBTask->SetBlockAndGridSize(gridSize, threadBlockSize);

	GraphInputChannel * pA_AInput			= pGraph->AddInputChannel(pAInputPorts[0], "A_AInputChannel");
	GraphInputChannel * pA_AxBScaleInput	= pGraph->AddInputChannel(pAInputPorts[1], "A_ScalarChannel");
	GraphOutputChannel * pA_Output			= pGraph->AddOutputChannel(pAOutputPorts[0], "A_outputChannel");
    pGraph->BindDescriptorPort(pAInputPorts[0], pAInputPorts[2]);

	GraphInputChannel * pB_AxBScaleInput	= pGraph->AddInputChannel(pBInputPorts[1], "B_ScalarChannel");
	GraphOutputChannel * pB_Output			= pGraph->AddOutputChannel(pBOutputPorts[0], "B_outputChannel");
    pGraph->BindDescriptorPort(pBInputPorts[0], pBInputPorts[2]);

    // now add an internal channel from A_A output to 
    // to B_A input. This will creaet a situation where
    // there is an output channel to the user on the same
    // port where there is an internal channel to an in/out
    // port. In order to ensure the user gets a consistent view
    // of the A_A output, the runtime has to clone the input
    // to B_A. We will wait to pull from the A_A output until
    // after pulling from B_A output, so that if the A_A output
    // can be overwritten, we will be sure to detect it.
    pGraph->AddInternalChannel(pAOutputPorts[0], pBInputPorts[0], "A->B");

	pGraph->Run();

	Datablock * pA		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pA_AInput);
	Datablock * pScParm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pA_AxBScaleInput);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();
	pA_AxBScaleInput->Push(pScParm);
    pB_AxBScaleInput->Push(pScParm);
    pScParm->Release();
    pA_AInput->Push(pA);
    pA->Release();

    int nFailures = 0;
    Datablock * pB_out = pB_Output->Pull();
    Datablock * pA_out = pA_Output->Pull();
    pB_out->Lock();
	float * psrc = (float*) pB_out->GetDataPointer(TRUE);
	float* vBReference = vector_scale(vAVector, fScalar*fScalar, n);
	if(!compare_vectors(vBReference, psrc, n)) {
        nFailures++;
    } 
    pB_out->Unlock();
    pA_out->Lock();
	psrc = (float*) pA_out->GetDataPointer(TRUE);
	float* vAReference = vector_scale(vAVector, fScalar, n);
	if(!compare_vectors(vAReference, psrc, n)) {
        nFailures++;
    } 
    pA_out->Unlock();

    if(nFailures) {
        printf("tee/cloning failed!\n");
    } else {
        printf("%s succeeded\n", szshader);
    }

    delete [] vAReference;
    delete [] vBReference;
	pA_out->Release();
    pB_out->Release();
	pGraph->Stop();
	pGraph->Teardown();

	delete [] vAVector;
    delete [] vBVector;
	Graph::DestroyGraph(pGraph);
	delete [] pAInputPorts;
	delete [] pAOutputPorts;
	delete [] pBInputPorts;
	delete [] pBOutputPorts;

	delete pTimer;

	PTask::Runtime::Terminate();

	return 0;
}

