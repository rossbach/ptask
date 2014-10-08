//--------------------------------------------------------------------------------------
// File: pfxsum.cpp
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
#include "pfxsum.h"
#include <stdio.h>
#include "ptaskapi.h"
#include "confighelpers.h"
#include "platformcheck.h"

using namespace std;
using namespace PTask;

int g_data[] = {
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
int g_nArraySize = 128;

void 
dump_array(
    int N,
    int * parray
    )
{
    for(int i=0; i<N; i++) {
        printf("%d, ", parray[i]);
        if(i%16 == 15) printf("\n");
    }
    printf("\n");
}

void
host_pfxsum(
    int N,
    int * pin,
    int * pout
    )
{
    assert(N > 1);
    assert(pin);
    assert(pout);
    pout[0] = 0; // pin[0];
    for(int i=1; i<N; i++) {
        pout[i] = pout[i-1] + pin[i-1];
    }
}

BOOL
arrays_equal(
    int N,
    int * pKV0,
    int * pKV1
    )
{
    if(!pKV0 || !pKV1) return FALSE;
    for(int i=0; i<N; i++)
        if(pKV0[i] != pKV1[i]) return FALSE;
    return TRUE;
}

int run_pfxsum_task(	
	char * szfile,
	char * szshader,
	int iterations
	) 
{
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    printf("ptask pfxsum!\n");
    printf("Input Array:\n");
    dump_array(g_nArraySize, g_data);
    PFXSUM_PARAMS params;
    params.N = g_nArraySize;

	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	Graph * pGraph = new Graph();
    UINT uiMaxDataSize = (UINT) g_nArraySize;
    DatablockTemplate * pTemplate	    = PTask::Runtime::GetDatablockTemplate("dbpfxdata", sizeof(int), uiMaxDataSize, 1, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("parms", sizeof(PFXSUM_PARAMS), 1, 1, 1);
	CompiledKernel * pKernel		    = COMPILE_KERNEL(szfile, szshader);

    int nChannelCount = 0;
	const UINT uiInputCount = 2;
	const UINT uiOutputCount = 1;
	
	Port ** pInputPorts = new Port*[uiInputCount];
	Port ** pOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter	= 0;
	pInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pTemplate, uiUidCounter++, "p1");
	pInputPorts[1]		= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "p2");
	pOutputPorts[0]	    = PTask::Runtime::CreatePort(OUTPUT_PORT, pTemplate, uiUidCounter++, "p3");	

	Task * pTask = pGraph->AddTask(pKernel, 
									uiInputCount,
									pInputPorts,
									uiOutputCount,
									pOutputPorts,
									"PfxsumTask");
	assert(pTask);
	pTask->SetComputeGeometry(1, 1, 1);
        // uiMaxDataSize/2, 1, 1);

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
        Datablock * pDB	= PTask::Runtime::AllocateDatablock(pTemplate, g_data, sizeof(g_data), pInput, rawFlags);        
        Datablock * pPrmDB	= PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pParmsInput);

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
        int * psum = (int*) malloc(g_nArraySize*sizeof(int));
		dHostStart = pTimer->elapsed(false);
        host_pfxsum(g_nArraySize, g_data, psum);
        dHostEnd = pTimer->elapsed(false) - dHostStart;

        if(!arrays_equal(g_nArraySize, paccResult, psum)) {
			printf("failure: erroneous output\n");
        } else {
			printf( "%s succeeded\n", szshader );
        }

        printf("\n\nHOST PFXSUM:\n");
        dump_array(g_nArraySize, psum);
        printf("\n\nGPU PFXSUM:\n");
        dump_array(g_nArraySize, paccResult);

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
