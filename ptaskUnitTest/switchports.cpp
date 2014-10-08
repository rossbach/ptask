//--------------------------------------------------------------------------------------
// File: switchports.cpp
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#include <stdio.h>
#include <crtdbg.h>

#include "assert.h"
#include "shaderparms.h"
#include "sort.h"
#include <vector>
#include <algorithm>
#include "matrixtask.h"
#include "SimpleMatrix.h"
#include "SimpleVector.h"
#include "matmul.h"
#include "switchports.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"

using namespace std;
using namespace PTask;

extern float * random_vector(int n);
extern BOOL compare_vectors(float*pA, float*pB, int n);
extern float * vector_scale(float * pA, float scalar, int n);

int run_graph_cuda_switchport_task(	
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
	params.N = n;
    float fScalar = 2.0f;
	UINT stride = sizeof(float);
    UINT nPulse = 1;

	int nChannelCount = 0;

	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	Graph * pGraph = new Graph();
	DatablockTemplate * pPulseTemplate	= PTask::Runtime::GetDatablockTemplate("pulse", sizeof(int), 1, 1, 1);
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
	DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", sizeof(float), PTPARM_FLOAT);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
	CompiledKernel * pMatmulKernel		= COMPILE_KERNEL(szfile, szshader);

	const UINT uiInputCount = 4;
	const UINT uiOutputCount = 1;
	
	Port ** pAplusBInputPorts = new Port*[uiInputCount];
	Port ** pAplusBOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAplusBInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pPulseTemplate, uiUidCounter++, "pulse", 0);
	pAplusBInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A", 1, 0);
	pAplusBInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar", 2);
	pAplusBInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N", 3);
	pAplusBOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)", 0);

	Task * pAplusBTask = pGraph->AddTask(pMatmulKernel, 
										  uiInputCount,
										  pAplusBInputPorts,
										  uiOutputCount,
										  pAplusBOutputPorts,
										  "AxScalar");

	assert(pAplusBTask);
	pAplusBTask->SetComputeGeometry(n, 1, 1);
	PTASKDIM3 threadBlockSize(256, 1, 1);
	PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
	pAplusBTask->SetBlockAndGridSize(gridSize, threadBlockSize);

	GraphInputChannel * pPulseInput		    = pGraph->AddInputChannel(pAplusBInputPorts[0], "PulseInputChannel");
	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAplusBInputPorts[1], "AInputChannel", TRUE);
	GraphInputChannel * pAxBScaleInput		= pGraph->AddInputChannel(pAplusBInputPorts[2], "ScalarChannel");
	GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAplusBInputPorts[3], "NConstChannel");
    InternalChannel * pBackChannel          = pGraph->AddInternalChannel(pAplusBOutputPorts[0], pAplusBInputPorts[1], "BackEdge");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAplusBOutputPorts[0], "outputChannel");

	pGraph->Run();

    Datablock * pTrigger = PTask::Runtime::AllocateDatablock(pPulseTemplate, &nPulse, sizeof(nPulse), pPulseInput); 
	Datablock * pA		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pAInput);
	Datablock * pACopy	 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pAInput);
	Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pAxBScaleInput);
	Datablock * pABPrm	 = PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pAxBParmsInput);

	pAxBScaleInput->Push(pScPrm);
	pAxBParmsInput->Push(pABPrm);
    pScPrm->Release();
    pABPrm->Release();

    int nSuccessCount = 0;
    int nFailCount = 0;
    pAInput->Push(pA);
    float factor = 1.0f;
    for(int j=1; j<4; j++) {
        factor = factor * fScalar;
        pPulseInput->Push(pTrigger);
        Datablock * pResultBlock = pOutput->Pull();
        pResultBlock->Lock();
	    float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
        pResultBlock->Unlock();
	    int nErrorTolerance = 20;
	    float* vReference = vector_scale(vAVector, factor, n);
	    if(!compare_vectors(vReference, psrc, n)) {
            nFailCount++;
            printf("failure. ref[0] = %+f, out[0] = %+f\n", vReference[0], psrc[0]);
        } else  {
            nSuccessCount++; 
        }
        delete [] vReference;
	    pResultBlock->Release();
    }
    pA->Release();

    pAInput->Push(pACopy);
    pPulseInput->Push(pTrigger);
    pACopy->Release();
    pTrigger->Release();
    Datablock * pResultBlock = pOutput->Pull();
    pResultBlock->Lock();
	float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
    pResultBlock->Unlock();
	int nErrorTolerance = 20;
	float* vReference = vector_scale(vAVector, fScalar, n);
	if(!compare_vectors(vReference, psrc, n)) {
        nFailCount++;
        printf("failure. ref[0] = %+f, out[0] = %+f\n", vReference[0], psrc[0]);
    } else  {
        nSuccessCount++; 
    }
    delete [] vReference;
	pResultBlock->Release();

    if(nFailCount == 0) {
        printf( "%s succeeded\n", szshader );
    }
	pGraph->Stop();
	pGraph->Teardown();

	delete [] vAVector;
	Graph::DestroyGraph(pGraph);
	delete [] pAplusBInputPorts;
	delete [] pAplusBOutputPorts;

	delete pTimer;

	PTask::Runtime::Terminate();

	return 0;
}

