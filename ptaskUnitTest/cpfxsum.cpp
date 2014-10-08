///-------------------------------------------------------------------------------------------------
// file:	cpfxsum.cpp
//
// summary:	cuda prefix sum
///-------------------------------------------------------------------------------------------------

#include <stdio.h>
#include <crtdbg.h>
#include "accelerator.h"
#include "assert.h"
#include "shaderparms.h"
#include "sort.h"
#include <vector>
#include <algorithm>
#include "cpfxsum.h"
#include <stdio.h>
#include "ptaskapi.h"
#include "confighelpers.h"
#include "platformcheck.h"
#include <math.h>

using namespace std;
using namespace PTask;

int g_cdata[] = {
    17,15,7,18,1,7,8,16,
    4,8,10,12,10,18,18,15,
    9,9,4,6,10,20,5,19,
    1,18,5,15,14,2,13,6,
    18,2,12,7,2,13,2,4,
    5,8,8,18,16,0,14,14,
    6,4,0,17,18,11,15,2,
    2,15,15,0,18,14,15,13,
    10,8,9,10,18,19,9,17,
    5,3,4,0,17,13,20,2,
    16,3,3,19,2,8,10,17,
    7,4,19,18,18,7,5,20,
    20,9,6,1,4,6,16,2,
    8,15,5,12,19,20,12,18,
    17,4,20,8,6,5,7,14,
    9,8,5,19,9,16,16,0,
    17,15,7,18,1,7,8,16,
    4,8,10,12,10,18,18,15,
    9,9,4,6,10,20,5,19,
    1,18,5,15,14,2,13,6,
    18,2,12,7,2,13,2,4,
    5,8,8,18,16,0,14,14,
    6,4,0,17,18,11,15,2,
    2,15,15,0,18,14,15,13,
    10,8,9,10,18,19,9,17,
    5,3,4,0,17,13,20,2,
    16,3,3,19,2,8,10,17,
    7,4,19,18,18,7,5,20,
    20,9,6,1,4,6,16,2,
    8,15,5,12,19,20,12,18,
    17,4,20,8,6,5,7,14,
    9,8,5,19,9,16,16,0,
    17,15,7,18,1,7,8,16,
    4,8,10,12,10,18,18,15,
    9,9,4,6,10,20,5,19,
    1,18,5,15,14,2,13,6,
    18,2,12,7,2,13,2,4,
    5,8,8,18,16,0,14,14,
    6,4,0,17,18,11,15,2,
    2,15,15,0,18,14,15,13,
    10,8,9,10,18,19,9,17,
    5,3,4,0,17,13,20,2,
    16,3,3,19,2,8,10,17,
    7,4,19,18,18,7,5,20,
    20,9,6,1,4,6,16,2,
    8,15,5,12,19,20,12,18,
    17,4,20,8,6,5,7,14,
    9,8,5,19,9,16,16,0,
    17,15,7,18,1,7,8,16,
    4,8,10,12,10,18,18,15,
    9,9,4,6,10,20,5,19,
    1,18,5,15,14,2,13,6,
    18,2,12,7,2,13,2,4,
    5,8,8,18,16,0,14,14,
    6,4,0,17,18,11,15,2,
    2,15,15,0,18,14,15,13,
    10,8,9,10,18,19,9,17,
    5,3,4,0,17,13,20,2,
    16,3,3,19,2,8,10,17,
    7,4,19,18,18,7,5,20,
    20,9,6,1,4,6,16,2,
    8,15,5,12,19,20,12,18,
    17,4,20,8,6,5,7,14,
    9,8,5,19,9,16,16,0,
    17,15,7,18,1,7,8,16,
    4,8,10,12,10,18,18,15,
    9,9,4,6,10,20,5,19,
    1,18,5,15,14,2,13,6,
    18,2,12,7,2,13,2,4,
    5,8,8,18,16,0,14,14,
    6,4,0,17,18,11,15,2,
    2,15,15,0,18,14,15,13,
    10,8,9,10,18,19,9,17,
    5,3,4,0,17,13,20,2,
    16,3,3,19,2,8,10,17,
    7,4,19,18,18,7,5,20,
    20,9,6,1,4,6,16,2,
    8,15,5,12,19,20,12,18,
    17,4,20,8,6,5,7,14,
    9,8,5,19,9,16,16,0,
    17,15,7,18,1,7,8,16,
    4,8,10,12,10,18,18,15,
    9,9,4,6,10,20,5,19,
    1,18,5,15,14,2,13,6,
    18,2,12,7,2,13,2,4,
    5,8,8,18,16,0,14,14,
    6,4,0,17,18,11,15,2,
    2,15,15,0,18,14,15,13,
    10,8,9,10,18,19,9,17,
    5,3,4,0,17,13,20,2,
    16,3,3,19,2,8,10,17,
    7,4,19,18,18,7,5,20,
    20,9,6,1,4,6,16,2,
    8,15,5,12,19,20,12,18,
    17,4,20,8,6,5,7,14,
    9,8,5,19,9,16,16,0,
    17,15,7,18,1,7,8,16,
    4,8,10,12,10,18,18,15,
    9,9,4,6,10,20,5,19,
    1,18,5,15,14,2,13,6,
    18,2,12,7,2,13,2,4,
    5,8,8,18,16,0,14,14,
    6,4,0,17,18,11,15,2,
    2,15,15,0,18,14,15,13,
    10,8,9,10,18,19,9,17,
    5,3,4,0,17,13,20,2,
    16,3,3,19,2,8,10,17,
    7,4,19,18,18,7,5,20,
    20,9,6,1,4,6,16,2,
    8,15,5,12,19,20,12,18,
    17,4,20,8,6,5,7,14,
    9,8,5,19,9,16,16,0,
    17,15,7,18,1,7,8,16,
    4,8,10,12,10,18,18,15,
    9,9,4,6,10,20,5,19,
    1,18,5,15,14,2,13,6,
    18,2,12,7,2,13,2,4,
    5,8,8,18,16,0,14,14,
    6,4,0,17,18,11,15,2,
    2,15,15,0,18,14,15,13,
    10,8,9,10,18,19,9,17,
    5,3,4,0,17,13,20,2,
    16,3,3,19,2,8,10,17,
    7,4,19,18,18,7,5,20,
    20,9,6,1,4,6,16,2,
    8,15,5,12,19,20,12,18,
    17,4,20,8,6,5,7,14,
    9,8,5,19,9,16,16,0,
};
int g_nCArraySize = 1024;

extern void dump_array(
    int N,
    int * parray
    );
extern void host_pfxsum(
    int N,
    int * pin,
    int * pout
    );
extern BOOL
arrays_equal(
    int N,
    int * pKV0,
    int * pKV1
    );


int run_cpfxsum_task(	
	char * szfile,
	char * szshader,
	int iterations
	) 
{
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    printf("ptask CUDA pfxsum!\n");
    printf("Input Array:\n");
	g_nCArraySize = 24;
    dump_array(g_nCArraySize, g_cdata);

	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	Graph * pGraph = new Graph();
    UINT uiMaxDataSize = (UINT) g_nCArraySize;
    DatablockTemplate * pTemplate	    = PTask::Runtime::GetDatablockTemplate("dbpfxdata", sizeof(int), uiMaxDataSize, 1, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("parms", sizeof(g_nCArraySize), PTPARM_INT);
	CompiledKernel * pKernel		    = COMPILE_KERNEL(szfile, szshader);
	
	int nThreadBlockSize = 256;
    int nChannelCount = 0;
	const UINT uiInputCount = 2;
	const UINT uiOutputCount = 1;
	
	Port ** pInputPorts = new Port*[uiInputCount];
	Port ** pOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter	= 0;
	pInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pTemplate, uiUidCounter++, "pin", 0);
	pInputPorts[1]		= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N", 2);
	pOutputPorts[0]	    = PTask::Runtime::CreatePort(OUTPUT_PORT, pTemplate, uiUidCounter++, "pout", 1);	

	Task * pTask = pGraph->AddTask(pKernel, 
								   uiInputCount,
								   pInputPorts,
								   uiOutputCount,
								   pOutputPorts,
								   "PfxsumTask");
	assert(pTask);
    PTASKDIM3 threadBlockSize(nThreadBlockSize, 1, 1);
    PTASKDIM3 gridSize(static_cast<int>(ceil(g_nCArraySize/(double)nThreadBlockSize)), 1, 1);
	gridSize.x = 16;
    pTask->SetBlockAndGridSize(gridSize, threadBlockSize);

	GraphInputChannel * pInput		= pGraph->AddInputChannel(pInputPorts[0], "DataInputChannel");
	GraphInputChannel * pParmsInput	= pGraph->AddInputChannel(pInputPorts[1], "ParmsInputChannel");
	GraphOutputChannel * pOutput	= pGraph->AddOutputChannel(pOutputPorts[0], "DataOutputChannel");

	pGraph->Run();

	double dInitTime;
	double dCopyToDeviceEnd; 
	double dCopyToDeviceTime;
	double dComputeEnd;
	double dComputeTime;
	double dCopyType;
	double dHostStart;
	double dHostEnd;
	double dTeardownStart;
	for(int i=0;i<iterations;i++) {

        BUFFERACCESSFLAGS rawFlags = (PT_ACCESS_HOST_WRITE | PT_ACCESS_ACCELERATOR_READ);
        Datablock * pDB	= PTask::Runtime::AllocateDatablock(pTemplate, g_cdata, sizeof(int)*g_nCArraySize, pInput, rawFlags);        
        Datablock * pPrmDB	= PTask::Runtime::AllocateDatablock(pParmTemplate, &g_nCArraySize, sizeof(g_nCArraySize), pParmsInput);

		dInitTime = pTimer->elapsed(false);
        pInput->Push(pDB);
        pParmsInput->Push(pPrmDB);
        pDB->Release();
        pPrmDB->Release();
		dCopyToDeviceEnd = pTimer->elapsed(false);
		dCopyToDeviceTime = dCopyToDeviceEnd - dInitTime;
		Datablock * poDB = pOutput->Pull();
		dComputeEnd = pTimer->elapsed(false);
		dComputeTime = dComputeEnd - dCopyToDeviceEnd;
        poDB->Lock();
        int* paccResult = (int*) poDB->GetDataPointer(FALSE);
        poDB->Unlock();
		dCopyType = pTimer->elapsed(false) - dComputeTime;

		printf( "Verifying against CPU result..." );
        int * psum = (int*) malloc(g_nCArraySize*sizeof(int));
		dHostStart = pTimer->elapsed(false);
        host_pfxsum(g_nCArraySize, g_cdata, psum);
        dHostEnd = pTimer->elapsed(false) - dHostStart;

        if(!arrays_equal(g_nCArraySize, paccResult, psum)) {
			printf("failure: erroneous output\n");
        } else {
			printf( "%s succeeded\n", szshader );
        }

        printf("\n\nHOST PFXSUM:\n");
        dump_array(g_nCArraySize, psum);
        printf("\n\nCUDA PFXSUM:\n");
        dump_array(g_nCArraySize, paccResult);

		dTeardownStart = pTimer->elapsed(false);
		poDB->Release();
        free(psum);
	}

	pGraph->Stop();
	pGraph->Teardown();

	Graph::DestroyGraph(pGraph);
	delete [] pInputPorts;
	delete [] pOutputPorts;
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
