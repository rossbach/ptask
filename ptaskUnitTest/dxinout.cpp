//--------------------------------------------------------------------------------------
// File: dxinout.cpp
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
#include "dxinout.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

using namespace std;
using namespace PTask;

extern float * random_vector(int n);
extern BOOL compare_vectors(float*pA, float*pB, int n);

typedef struct dxinoutparm_t {
	int N;
    int scalar;
} DXINOUT_PARAMS;

float *
vector_add_scalar(
	float * pA,
	float scalar,
    int n
	) 
{
	float * pC = new float[n];
	for(int i=0; i<n; i++)
		pC[i] = pA[i] + scalar;
	return pC;
}

int run_dxinout_task(	
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

	DXINOUT_PARAMS params;
	params.N = n;
    params.scalar = 10;
	UINT stride = sizeof(float);

	int nChannelCount = 0;

	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	Graph * pGraph = new Graph();
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(DXINOUT_PARAMS), 1, 1, 1);

    CompiledKernel * pKernel = PTask::Runtime::GetCompiledKernel(szfile, szshader, g_szCompilerOutputBuffer, COBBUFSIZE);
    CheckCompileSuccess(szfile, szshader, pKernel);

	const UINT uiInputCount = 2;
	const UINT uiOutputCount = 1;
	
	Port ** pAplusBInputPorts = new Port*[uiInputCount];
	Port ** pAplusBOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAplusBInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A", 0, 0);
	pAplusBInputPorts[1]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "parms", 1);
	pAplusBOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)", 0);

	Task * pAplusBTask = pGraph->AddTask(pKernel, 
										  uiInputCount,
										  pAplusBInputPorts,
										  uiOutputCount,
										  pAplusBOutputPorts,
										  "AplusScalar");

	assert(pAplusBTask);
	pAplusBTask->SetComputeGeometry(n, 1, 1);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAplusBInputPorts[0], "AInputChannel");
	GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAplusBInputPorts[1], "ParmChannel");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAplusBOutputPorts[0], "outputChannel");

	pGraph->Run();

	Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pAInput);
	Datablock * pABPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pAxBParmsInput);

	double dInitTime = pTimer->elapsed(false);
	pAInput->Push(pA);
	pAxBParmsInput->Push(pABPrm);
    pA->Release();
    pABPrm->Release();
	double dCopyToDeviceEnd = pTimer->elapsed(false);
	double dCopyToDeviceTime = dCopyToDeviceEnd - dInitTime;
	Datablock * pResultBlock = pOutput->Pull();
	double dComputeEnd = pTimer->elapsed(false);
	double dComputeTime = dComputeEnd - dCopyToDeviceEnd;
    pResultBlock->Lock();
	float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
    pResultBlock->Unlock();
	double dCopyType = pTimer->elapsed(false) - dComputeTime;

	printf( "\nVerifying against CPU result..." );
	int nErrorTolerance = 20;
	double dHostStart = pTimer->elapsed(false);
    float* vReference = vector_add_scalar(vAVector, (float) params.scalar, n);
	double dHostEnd = pTimer->elapsed(false) - dHostStart;

	if(!compare_vectors(vReference, psrc, n))
        printf("failure. ref[0] = %+f, out[0] = %+f\n", vReference[0], psrc[0]);
	else 
        printf( "%s succeeded\n", szshader );

	double dTeardownStart = pTimer->elapsed(false);
	pResultBlock->Release();
	pGraph->Stop();
	pGraph->Teardown();

	delete [] vAVector;
	delete [] vReference;
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

