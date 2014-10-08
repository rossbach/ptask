//--------------------------------------------------------------------------------------
// File: graphsort.cpp
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#include <stdio.h>
#include <crtdbg.h>
#include "accelerator.h"
#include "assert.h"
#include "shaderparms.h"
#include "sort.h"
#include <vector>
#include <string>
#include <algorithm>
#include <assert.h>
#include "graphsort.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"
using namespace std;
using namespace PTask;

typedef struct invdesc_t {
	char * op;
	SORT_PARAMS parms;
//	UINT level;
//	UINT mask;
	UINT X;
	UINT Y;
	UINT Z;
	Port ** pInputPorts;
	Port ** pOutputPorts;
	Task * pTask;
	Channel * pInput;
	Channel * pOutput;
	GraphInputChannel * pParmsInput;
	Datablock * pParmBlock;
} SORTSTEPDESC;

static const unsigned int SORT_BLOCK_SIZE = 512;
static const unsigned int TRANSPOSE_SIZE = 16;

//SORTSTEPDESC g_sortsteps512x16[]  = { 
//	{ "sort",		{ 2, 2, 16, 512 },		16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 4, 4, 16, 512 },		16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 8, 8, 16, 512 },		16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 16, 16, 16, 512 },	16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 32, 32, 16, 512 },	16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 64, 64, 16, 512 },	16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 128, 128, 16, 512 },	16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 256, 256, 16, 512 },	16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 512, 512, 16, 512 },	16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "transpose",	{ 2, 2, 512, 16 },		32, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 2, 2, 512, 16 },		16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "transpose",	{ 512, 1024, 16, 512 },	1, 32, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 512, 1024, 16, 512 },	16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "transpose",	{ 4, 4, 512, 16 },		32, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 4, 4, 512, 16 },		16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "transpose",	{ 512, 2048, 16, 512 },	1, 32, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 512, 2048, 16, 512 },	16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "transpose",	{ 8, 8, 512, 16 },		32, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 8, 8, 512, 16 },		16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "transpose",	{ 512, 4096, 16, 512 },	1, 32, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 512, 4096, 16, 512 },	16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "transpose",	{ 16, 0, 512, 16 },		32, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 16, 0, 512, 16 },		16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "transpose",	{ 512, 8192, 16, 512 },	1, 32, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ "sort",		{ 512, 8192, 16, 512 },	16, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
//	{ NULL,			{ 0, 0, 0, 0 },			0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL }
//};

vector<SORTSTEPDESC*>*
summarize_graph_steps(
	UINT rows,
	UINT cols
	)
{
	SORTSTEPDESC * pDesc = NULL;
	vector<SORTSTEPDESC*> * pSteps = new vector<SORTSTEPDESC*>();
	for(UINT level=2 ; level<= SORT_BLOCK_SIZE; level *= 2) {
		pDesc = new SORTSTEPDESC();
		memset(pDesc, 0, sizeof(*pDesc));
		pDesc->parms.iLevel = level;
		pDesc->parms.iLevelMask = level;
		pDesc->parms.iWidth = rows;  // yes, this is intentional.
		pDesc->parms.iHeight = cols;
		pDesc->X = (rows * cols) / SORT_BLOCK_SIZE;
		pDesc->Y = 1;
		pDesc->Z = 1;
		pDesc->op = "sort";
		pSteps->push_back(pDesc);
	}

    for(UINT level=(SORT_BLOCK_SIZE*2); level<=(rows*cols); level *= 2) {

		pDesc = new SORTSTEPDESC();
		memset(pDesc, 0, sizeof(*pDesc));
		pDesc->parms.iLevel = (level / SORT_BLOCK_SIZE);
		pDesc->parms.iLevelMask = (level & ~(rows*cols)) / SORT_BLOCK_SIZE; 
		pDesc->parms.iWidth = cols;
		pDesc->parms.iHeight = rows;
		pDesc->X = cols / TRANSPOSE_SIZE;
		pDesc->Y = rows / TRANSPOSE_SIZE;
		pDesc->Z = 1;
		pDesc->op = "transpose";
		pSteps->push_back(pDesc);

		pDesc = new SORTSTEPDESC();
		memset(pDesc, 0, sizeof(*pDesc));
		pDesc->parms.iLevel = (level / SORT_BLOCK_SIZE);
		pDesc->parms.iLevelMask = (level & ~(rows*cols)) / SORT_BLOCK_SIZE; 
		pDesc->parms.iWidth = cols;
		pDesc->parms.iHeight = rows;
		pDesc->X = (rows * cols) / SORT_BLOCK_SIZE;
		pDesc->Y = 1;
		pDesc->Z = 1;
		pDesc->op = "sort";
		pSteps->push_back(pDesc);

		pDesc = new SORTSTEPDESC();
		memset(pDesc, 0, sizeof(*pDesc));
		pDesc->parms.iLevel = SORT_BLOCK_SIZE;
		pDesc->parms.iLevelMask = level;
		pDesc->parms.iWidth = rows;
		pDesc->parms.iHeight = cols;
		pDesc->X = rows / TRANSPOSE_SIZE;
		pDesc->Y = cols / TRANSPOSE_SIZE;
		pDesc->Z = 1;
		pDesc->op = "transpose";
		pSteps->push_back(pDesc);

		pDesc = new SORTSTEPDESC();
		memset(pDesc, 0, sizeof(*pDesc));
		pDesc->parms.iLevel = SORT_BLOCK_SIZE;
		pDesc->parms.iLevelMask = level;
		pDesc->parms.iWidth = rows;
		pDesc->parms.iHeight = cols;
		pDesc->X = (rows * cols) / SORT_BLOCK_SIZE;
		pDesc->Y = 1;
		pDesc->Z = 1;
		pDesc->op = "sort";
		pSteps->push_back(pDesc);
	}

#ifdef VERIFY_GRAPH_CONSTRUCTION
	int index = 0;
	vector<SORTSTEPDESC*>::iterator vi;
	for(vi=pSteps->begin(); vi!=pSteps->end(); vi++) {
		SORTSTEPDESC * pCand = *vi;
		SORTSTEPDESC * pRef = &g_sortsteps512x16[index++];
		if(!memcmp(pCand, pRef, sizeof(*pCand))) {
			assert(false);
		}
	}
#endif
	return pSteps;
}

void 
free_graph_steps(
	vector<SORTSTEPDESC*>* pSteps
	)
{
	vector<SORTSTEPDESC*>::iterator vi;
	for(vi=pSteps->begin(); vi!=pSteps->end(); vi++) {
		delete [] (*vi)->pInputPorts;
		delete [] (*vi)->pOutputPorts;
		delete *vi;
	}
}



#pragma warning(disable:4996)
DWORD WINAPI 
graph_sort_thread(
	LPVOID p
	) 
{
    PPSORTDESC desc = (PPSORTDESC) p;
    PTask::Runtime::Initialize();
    CheckPlatformSupport(desc->szfile, desc->szSortShader);

	int rows = desc->rows;
	int cols = desc->cols;
	vector<SORTSTEPDESC*>::iterator vi;
	vector<SORTSTEPDESC*>* pSortSteps = summarize_graph_steps(rows, cols);
	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	SORT_PARAMS params;
	params.iHeight = rows;
	params.iWidth = cols;
	params.iLevel = 2;
	params.iLevelMask = 512;
	UINT stride = sizeof(UINT);
	UINT elements = rows*cols;
	vector<UINT>* pData = randdata(rows*cols);
	vector<UINT>* pReference = new vector<UINT>(rows*cols);

	double dHostStart = pTimer->elapsed(false);
	pReference->assign(pData->begin(), pData->end());
	hostsort(pReference);
	double dHostEnd = pTimer->elapsed(true) - dHostStart;

	vector<UINT>* pResult = new vector<UINT>(rows*cols);

	int nChannelCount = 0;
	const UINT SORT_BLOCK_SIZE = rows;
	const UINT TRANSPOSE_SIZE = 16;

	Graph * pGraph = new Graph();
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbInput_uint", stride, rows, cols, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("sortparms", sizeof(SORT_PARAMS), 1, 1, 1);
	CompiledKernel * pSortKernel		= COMPILE_KERNEL(desc->szfile, desc->szSortShader);
	CompiledKernel * pTransposeKernel	= COMPILE_KERNEL(desc->szfile, desc->szTransposeShader);

	const UINT uiInputCount = 2;
	const UINT uiOutputCount = 1;
	UINT uiUidCounter = 0;
	int nTotalSteps = (int) pSortSteps->size();
	for(vi=pSortSteps->begin();
		vi!=pSortSteps->end(); vi++) {
		char szTaskName[256];
        char szPortName[256];
        int pN = 0;
		SORTSTEPDESC * step = *vi;
		CompiledKernel * pKernel = ((!strcmp(step->op, desc->szSortShader))?pSortKernel:pTransposeKernel);
		sprintf(szTaskName, "%s_step_%d", step->op, nTotalSteps);
		step->pInputPorts		= new Port*[uiInputCount];
		step->pOutputPorts		= new Port*[uiOutputCount];
        sprintf(szPortName, "%s_port_%d", step->op, pN++);
		step->pInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, szPortName);
        sprintf(szPortName, "%s_port_%d", step->op, pN++);
		step->pInputPorts[1]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, szPortName);
        sprintf(szPortName, "%s_port_%d", step->op, pN++);
		step->pOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, szPortName);
		step->pTask				= pGraph->AddTask(pKernel, 
													uiInputCount,
													step->pInputPorts,
													uiOutputCount,
													step->pOutputPorts,
													szTaskName);
		assert( (step->pInputPorts[0] != NULL) && (step->pInputPorts[1] != NULL) &&
				(step->pOutputPorts[0] != NULL) && (step->pTask != NULL));
		step->pTask->SetComputeGeometry(step->X, step->Y, step->Z);
	}

	int i;
	for(vi=pSortSteps->begin(), i=0;
		vi!=pSortSteps->end(); vi++, i++) {
		char szInputName[256];
		char szParmsName[256];
		char szOutputName[256];
		SORTSTEPDESC * step = *vi;
		sprintf(szInputName, "%s_InputChannel_%d", step->op, i);
		sprintf(szParmsName, "%s_ParmsChannel_%d", step->op, i);
		sprintf(szOutputName, "%s_OutputChannel_%d", step->op, i);
		step->pParmsInput =	pGraph->AddInputChannel(step->pInputPorts[1], szParmsName);
		step->pInput = ((i==0)?pGraph->AddInputChannel(step->pInputPorts[0], szInputName):(*(vi-1))->pOutput);
		if(i<nTotalSteps-1) {
			Port * pLocalOut = step->pOutputPorts[0];
			Port * pRemoteIn = (*(vi+1))->pInputPorts[0];
			step->pOutput = pGraph->AddInternalChannel(pLocalOut, pRemoteIn, szOutputName);
		} else {
			step->pOutput = pGraph->AddOutputChannel(step->pOutputPorts[0], szOutputName);
		}
		//printf("p_%d: (%d, %d, %d, %d, X=%d, Y=%d, Z=%d)\n",
		//	i,
		//	step->parms.iLevel,
		//	step->parms.iLevelMask,
		//	step->parms.iWidth,
		//	step->parms.iHeight,
		//	step->X, step->Y, step->Z);
		step->pParmBlock = PTask::Runtime::AllocateDatablock(pParmTemplate, &step->parms, sizeof(step->parms), step->pParmsInput);
	}	
    printf("graph.run...");

	pGraph->Run();

	for(vi=pSortSteps->begin();
		vi!=pSortSteps->end(); vi++) {
		SORTSTEPDESC * step = *vi;
		step->pParmsInput->Push(step->pParmBlock);
        step->pParmBlock->Release();
	}

	double dInitTime = pTimer->elapsed(false);
	GraphOutputChannel * pOutput = (GraphOutputChannel*) (*pSortSteps)[nTotalSteps-1]->pOutput;
	GraphInputChannel * pInput = (GraphInputChannel*) (*pSortSteps)[0]->pInput;
	Datablock * pDataBlock = PTask::Runtime::AllocateDatablock(pDataTemplate, &(*pData)[0], stride*elements, pInput);
	pInput->Push(pDataBlock);
    pDataBlock->Release();
	double dCopyToDeviceEnd = pTimer->elapsed(false);
	double dCopyToDeviceTime = dCopyToDeviceEnd - dInitTime;
	Datablock * pResultBlock = pOutput->Pull();
	double dComputeEnd = pTimer->elapsed(false);
	double dComputeTime = dComputeEnd - dCopyToDeviceEnd;

    pResultBlock->Lock();
	UINT * psrc = (UINT*) pResultBlock->GetDataPointer(FALSE);
	UINT * pdst = (UINT*) &(*pResult)[0];
	memcpy(pdst, psrc, elements*stride);
    pResultBlock->Unlock();

	// do we get the result we expect?
	i=0;
	printf( "Thread %d Verifying against CPU result (iteration %d)...", desc->threadid, i );
	if(!check_sort_result(pReference, pResult)) 
		printf("Thread %d failed (iteration %d)\n", desc->threadid, i);
	else 
		printf( "Thread %d: %s succeeded (iteration %d)\n", desc->threadid, desc->szSortShader, i );	

	double dTeardownStart = pTimer->elapsed(false);
	pResultBlock->Release();
	pGraph->Stop();
	pGraph->Teardown();

	delete pData;
	delete pResult;
	delete pReference;
	Graph::DestroyGraph(pGraph);
	double dTeardownTime = pTimer->elapsed(false) - dTeardownStart;
	free_graph_steps(pSortSteps);

	printf("InitTime:\t%f\n", dInitTime);
	printf("CopyToGPU:\t%f\n", dCopyToDeviceTime);
	printf("GPU Compute:\t%f\n", dComputeTime);
	printf("CPU Compute:\t%f\n", dHostEnd);
	printf("Teardown:\t%f\n", dTeardownTime);

	delete pTimer;

	PTask::Runtime::Terminate();

	return 0;
}

int run_graph_sort_task(	
	char * szfile,
	char * szshader,
	int rows,
	int cols,
	int siblings,
	int iterations
	) 
{
	HANDLE * vThreads = new HANDLE[siblings];
	PSORTDESC * descs = new PSORTDESC[siblings];
	for(int i=0; i<siblings; i++) {
		descs[i].rows = rows;
		descs[i].cols = cols;
		descs[i].threadid = i;
		descs[i].iterations = iterations;
		strcpy_s(descs[i].szfile, MAXPATH, szfile);
		strcpy_s(descs[i].szSortShader, MAXPATH, szshader);
		strcpy_s(descs[i].szTransposeShader, MAXPATH, "transpose");
		vThreads[i] = ::CreateThread(NULL, NULL, graph_sort_thread, &descs[i], NULL, 0);
	}
	
	WaitForMultipleObjects(siblings, vThreads, TRUE, INFINITE);
	for(int i=0; i<siblings; i++) 
		CloseHandle(vThreads[i]);
	delete [] vThreads;
	delete [] descs;
	return 0;
}
