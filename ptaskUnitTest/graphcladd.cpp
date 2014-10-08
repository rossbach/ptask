//--------------------------------------------------------------------------------------
// File: graphcladd.cpp
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
#include "graphcladd.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

using namespace std;
using namespace PTask;

int run_graph_opencl_vecadd_task(	
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
    CSimpleVector<float>* vAVector = new CSimpleVector<float>(n, randfloat);
    CSimpleVector<float>* vBVector = new CSimpleVector<float>(n, randfloat);

	VECADD_PARAMS params;
	params.N = n;
	UINT stride = sizeof(float);

	int nChannelCount = 0;

	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	Graph * pGraph = new Graph();
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("matmulparms", sizeof(VECADD_PARAMS), PTPARM_INT);

    CompiledKernel * pMatmulKernel = PTask::Runtime::GetCompiledKernel(szfile, szshader, g_szCompilerOutputBuffer, COBBUFSIZE);
    CheckCompileSuccess(szfile, szshader, pMatmulKernel);

	const UINT uiInputCount = 3;
	const UINT uiOutputCount = 1;
	
	Port ** pAplusBInputPorts = new Port*[uiInputCount];
	Port ** pAplusBOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAplusBInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A", 0);
	pAplusBInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B", 1);
	pAplusBInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N", 3);
	pAplusBOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "C", 2);

	Task * pAplusBTask = pGraph->AddTask(pMatmulKernel, 
										  uiInputCount,
										  pAplusBInputPorts,
										  uiOutputCount,
										  pAplusBOutputPorts,
										  "AplusBTask");

	assert(pAplusBTask);
	pAplusBTask->SetComputeGeometry(n, 1, 1);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAplusBInputPorts[0], "AInputChannel");
	GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAplusBInputPorts[1], "BInputChannel");
	GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAplusBInputPorts[2], "NConstChannel");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAplusBOutputPorts[0], "outputChannel");

	pGraph->Run();

	Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector->cells(), vAVector->arraysize(), pAInput);
	Datablock * pB		= PTask::Runtime::AllocateDatablock(pDataTemplate, vBVector->cells(), vBVector->arraysize(), pBInput);
	Datablock * pABPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pAxBParmsInput);

	double dInitTime = pTimer->elapsed(false);
	pAInput->Push(pA);
	pBInput->Push(pB);
	pAxBParmsInput->Push(pABPrm);
    pA->Release();
    pB->Release();
    pABPrm->Release();
	double dCopyToDeviceEnd = pTimer->elapsed(false);
	double dCopyToDeviceTime = dCopyToDeviceEnd - dInitTime;
	Datablock * pResultBlock = pOutput->Pull();
	double dComputeEnd = pTimer->elapsed(false);
	double dComputeTime = dComputeEnd - dCopyToDeviceEnd;
    pResultBlock->Lock();
    CSimpleVector<float> * pResult = new CSimpleVector<float>(n, (float*) pResultBlock->GetDataPointer(FALSE));
    pResultBlock->Unlock();
	double dCopyType = pTimer->elapsed(false) - dComputeTime;

	printf( "\nVerifying against CPU result..." );
	int nErrorTolerance = 20;
	double dHostStart = pTimer->elapsed(false);
	CSimpleVector<float> * vReference = CSimpleVector<float>::vadd(vAVector, vBVector);
	double dHostEnd = pTimer->elapsed(false) - dHostStart;

	if(!vReference->equals(pResult, fdelta, fgt, FEPSILON)) {
		printf("failure\n");
        printf("ref:\n");
        vReference->print(fprintfn, 8);
        printf("ptask-result:\n");
        pResult->print(fprintfn, 8);
    } else {
        printf( "%s succeeded\n", szshader );
    }

	double dTeardownStart = pTimer->elapsed(false);
	pResultBlock->Release();
	pGraph->Stop();
	pGraph->Teardown();

	delete vAVector;
	delete vBVector;
	delete vReference;
    delete pResult;
    Graph::DestroyGraph(pGraph);
	delete [] pAplusBInputPorts;
	delete [] pAplusBOutputPorts;
	double dTeardownTime = pTimer->elapsed(false) - dTeardownStart;

	printf("InitTime:\t%f\n", dInitTime);
	printf("CopyToGPU:\t%f\n", dCopyToDeviceTime);
	printf("GPU Compute:\t%f\n", dComputeTime);
	printf("CPU Compute:\t%f\n", dHostEnd);
	printf("Teardown:\t%f\n", dTeardownTime);

	delete pTimer;

	PTask::Runtime::Terminate();

	return 0;
}

