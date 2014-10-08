//--------------------------------------------------------------------------------------
// File: graphmatmul.cpp
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
#include "matmul.h"
#include "graphhostmm.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

using namespace std;
using namespace PTask;


int run_graph_host_matmul_task_hard(	
	char * szfile,
	char * szshader,
	int rows,
	int cols,
	int siblings,
	int iterations
	) 
{
    CONFIGUREPTASKU(UseDirectX, TRUE);
    CONFIGUREPTASKU(UseOpenCL, FALSE);
    CONFIGUREPTASKU(UseCUDA, TRUE);
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

#ifdef DANDELION_DEBUG
    rows = 3;
    cols = 3;
#endif

	assert(rows == cols);
	CSimpleMatrix<ELEMTYPE>* vAMatrix;
	CSimpleMatrix<ELEMTYPE>* vBMatrix;
	CSimpleMatrix<ELEMTYPE>* vCMatrix;
	CSimpleMatrix<ELEMTYPE>* vDMatrix = new CSimpleMatrix<ELEMTYPE>(rows, cols);
#ifdef DANDELION_DEBUG
	configure_raw_matrix(rows, cols, &vAMatrix, 1);
	configure_raw_matrix(rows, cols, &vBMatrix, 2);
	configure_raw_matrix(rows, cols, &vCMatrix, 3);
#else
	configure_raw_matrix(rows, cols, &vAMatrix);
	configure_raw_matrix(rows, cols, &vBMatrix);
	configure_raw_matrix(rows, cols, &vCMatrix);
#endif

    MATADD_PARAMS params;
	params.g_tex_cols = cols;
	params.g_tex_rows = rows;
	UINT stride = sizeof(ELEMTYPE);
	UINT elements = rows*cols;

	int nChannelCount = 0;

	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	Graph * pGraph = new Graph();
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbRxC_uint", stride, cols, rows, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("matmulparms", sizeof(MATADD_PARAMS), 1, 1, 1);
	CompiledKernel * pMatmulKernel		= COMPILE_KERNEL(szfile, szshader);

	const UINT uiInputCount = 3;
	const UINT uiOutputCount = 1;
	
	Port ** pAxBInputPorts = new Port*[uiInputCount];
	Port ** pAxBOutputPorts = new Port*[uiOutputCount];
	Port ** pAxBxCInputPorts = new Port*[uiInputCount];
	Port ** pAxBxCOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAxBInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A", 0);
	pAxBInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B", 1);
	pAxBInputPorts[2]		= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "parms", 3);
	pAxBOutputPorts[0]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "AxBo", 2);
	pAxBxCInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "AxBi", 0);
	pAxBxCInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "C", 1);
	pAxBxCInputPorts[2]		= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "parms", 3);
	pAxBxCOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "AxBxC", 2);	

	Task * pAxBTask = pGraph->AddTask(pMatmulKernel, 
									  uiInputCount,
									  pAxBInputPorts,
									  uiOutputCount,
									  pAxBOutputPorts,
									  "AxBTask");
	Task * pABCTask = pGraph->AddTask(pMatmulKernel, 
									  uiInputCount,
									  pAxBxCInputPorts,
									  uiOutputCount,
									  pAxBxCOutputPorts,
									  "ABxCTask");

	assert(pAxBTask && pABCTask);
	pAxBTask->SetComputeGeometry(rows, cols, 1);
	pABCTask->SetComputeGeometry(rows, cols, 1);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAxBInputPorts[0], "AInputChannel");
	GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAxBInputPorts[1], "BInputChannel");
	GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAxBInputPorts[2], "ABConstChannel");
	GraphInputChannel * pCInput				= pGraph->AddInputChannel(pAxBxCInputPorts[1], "CInputChannel");
	GraphInputChannel * pABCParmsInput		= pGraph->AddInputChannel(pAxBxCInputPorts[2], "ABCConstChannel");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAxBxCOutputPorts[0], "outputChannel");
	pGraph->AddInternalChannel(pAxBOutputPorts[0], pAxBxCInputPorts[0], "internalChannel");

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

		Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAMatrix->cells(), vAMatrix->arraysize(), pAInput);
		Datablock * pB		= PTask::Runtime::AllocateDatablock(pDataTemplate, vBMatrix->cells(), vBMatrix->arraysize(), pBInput);
		Datablock * pC		= PTask::Runtime::AllocateDatablock(pDataTemplate, vCMatrix->cells(), vCMatrix->arraysize(), pCInput);
		Datablock * pABPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pAxBParmsInput);
		Datablock * pABCPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pABCParmsInput);

		dInitTime = pTimer->elapsed(false);
		pAInput->Push(pA);
		pBInput->Push(pB);
		pCInput->Push(pC);
		pAxBParmsInput->Push(pABPrm);
		pABCParmsInput->Push(pABCPrm);
        pA->Release();
        pB->Release();
        pC->Release();
        pABPrm->Release();
        pABCPrm->Release();
		dCopyToDeviceEnd = pTimer->elapsed(false);
		dCopyToDeviceTime = dCopyToDeviceEnd - dInitTime;
		Datablock * pResultBlock = pOutput->Pull();
		dComputeEnd = pTimer->elapsed(false);
		dComputeTime = dComputeEnd - dCopyToDeviceEnd;

        pResultBlock->Lock();
		ELEMTYPE * psrc = (ELEMTYPE*) pResultBlock->GetDataPointer(FALSE);
		ELEMTYPE * pdst = vDMatrix->cells();
		memcpy(pdst, psrc, elements*stride);
        pResultBlock->Unlock();
		dCopyType = pTimer->elapsed(false) - dComputeTime;

		printf( "Verifying against CPU result..." );
		int nErrorTolerance = 20;
		dHostStart = pTimer->elapsed(false);
		CSimpleMatrix<ELEMTYPE>* pAxB = matmul(vAMatrix, vBMatrix); // on CPU
		CSimpleMatrix<ELEMTYPE>* pAxBxC = matmul(pAxB, vCMatrix);
		dHostEnd = pTimer->elapsed(false) - dHostStart;

		if(!check_matrix_result(vDMatrix, pAxBxC, &nErrorTolerance)) {
			printf("failure: (%d of %d) erroneous cells\n", nErrorTolerance, rows*cols);
            print_matrix("D: ", vDMatrix);
            print_matrix("AxBxC: ", pAxBxC);
            print_matrix("AxB: ", pAxB);
        } else {
			printf( "%s succeeded\n", szshader );
        }

		dTeardownStart = pTimer->elapsed(false);
		pResultBlock->Release();
		delete pAxB;
		delete pAxBxC;

	}

	pGraph->Stop();
	pGraph->Teardown();

	delete vAMatrix;
	delete vBMatrix;
	delete vCMatrix;
	delete vDMatrix;
	Graph::DestroyGraph(pGraph);
	delete [] pAxBInputPorts;
	delete [] pAxBOutputPorts;
	delete [] pAxBxCInputPorts;
	delete [] pAxBxCOutputPorts;
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

int run_graph_host_matmul_task(	
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

    assert(rows == cols);
	CSimpleMatrix<ELEMTYPE>* vAMatrix;
	CSimpleMatrix<ELEMTYPE>* vBMatrix;
	CSimpleMatrix<ELEMTYPE>* vDMatrix = new CSimpleMatrix<ELEMTYPE>(rows, cols);
	configure_raw_matrix(rows, cols, &vAMatrix);
	configure_raw_matrix(rows, cols, &vBMatrix);

	MATADD_PARAMS params;
	params.g_tex_cols = cols;
	params.g_tex_rows = rows;
	UINT stride = sizeof(ELEMTYPE);
	UINT elements = rows*cols;

	int nChannelCount = 0;

    Graph * pGraph = new Graph();
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbRxC_uint", stride, cols, rows, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("matmulparms", sizeof(MATADD_PARAMS), 1, 1, 1);
	CompiledKernel * pMatmulKernel		= COMPILE_KERNEL(szfile, szshader);

	const UINT uiInputCount = 3;
	const UINT uiOutputCount = 1;
	
	Port ** pAxBInputPorts = new Port*[uiInputCount];
	Port ** pAxBOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAxBInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A", 0);
	pAxBInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B", 1);
	pAxBInputPorts[2]		= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "parms", 3);
	pAxBOutputPorts[0]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "C", 2);

	Task * pAxBTask = pGraph->AddTask(pMatmulKernel, 
									  uiInputCount,
									  pAxBInputPorts,
									  uiOutputCount,
									  pAxBOutputPorts,
									  "AxBTask");

	assert(pAxBTask);
	// in this case, the iterations are handled 
    // inside the host code. Perhaps we should actually
    // pass the requested geometry to the host code?
    // pAxBTask->SetComputeGeometry(rows, cols, 1);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAxBInputPorts[0], "AInputChannel");
	GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAxBInputPorts[1], "BInputChannel");
	GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAxBInputPorts[2], "ABConstChannel");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAxBOutputPorts[0], "outputChannel");

	pGraph->Run();

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAMatrix->cells(), vAMatrix->arraysize(), pAInput);
	Datablock * pB		= PTask::Runtime::AllocateDatablock(pDataTemplate, vBMatrix->cells(), vBMatrix->arraysize(), pBInput);
	Datablock * pABPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pAxBParmsInput);

	pAInput->Push(pA);
	pBInput->Push(pB);
	pAxBParmsInput->Push(pABPrm);
	Datablock * pResultBlock = pOutput->Pull();

    pResultBlock->Lock();
	ELEMTYPE * psrc = (ELEMTYPE*) pResultBlock->GetDataPointer(FALSE);
	ELEMTYPE * pdst = vDMatrix->cells();
	memcpy(pdst, psrc, elements*stride);
    pResultBlock->Unlock();

	printf( "Verifying against CPU result..." );
	CSimpleMatrix<ELEMTYPE>* pAxB = matmul(vAMatrix, vBMatrix); // on CPU
	if(!check_matrix_result(vDMatrix, pAxB)) {
		printf("failure\n");
        print_matrix("D:", vDMatrix, 16);
        print_matrix("ref:", pAxB, 16);
    } else {
        printf( "%s succeeded\n", szshader );
    }

	pResultBlock->Release();
	pGraph->Stop();
	pGraph->Teardown();

	delete vAMatrix;
	delete vBMatrix;
	delete vDMatrix;
	delete pAxB;
	Graph::DestroyGraph(pGraph);
	delete [] pAxBInputPorts; 
	delete [] pAxBOutputPorts;

	PTask::Runtime::Terminate();

	return 0;
}

void __stdcall
hostfunc_htmatmul(
    UINT nArguments,
    void **ppArguments
    )
{
    assert(nArguments == 4);
    assert(ppArguments != NULL);
    MATADD_PARAMS * pParams = (MATADD_PARAMS*) ppArguments[3];
    int nStorageBytes = pParams->g_tex_rows * pParams->g_tex_cols * sizeof(ELEMTYPE);
    CSimpleMatrix<ELEMTYPE> * pA = new CSimpleMatrix<ELEMTYPE>(pParams->g_tex_rows, pParams->g_tex_cols);
    CSimpleMatrix<ELEMTYPE> * pB = new CSimpleMatrix<ELEMTYPE>(pParams->g_tex_rows, pParams->g_tex_cols);
    ELEMTYPE * A = (ELEMTYPE*) ppArguments[0];
    ELEMTYPE * B = (ELEMTYPE*) ppArguments[1];
    ELEMTYPE * C = (ELEMTYPE*) ppArguments[2];
    memcpy(pA->cells(), A, nStorageBytes);    
    memcpy(pB->cells(), B, nStorageBytes);
    CSimpleMatrix<ELEMTYPE> * pC = matmul(pA, pB);
    memcpy(C, pC->cells(), nStorageBytes);
    delete pA;
    delete pB;
    delete pC;
}

void __stdcall application_context_callback(
    APPLICATIONCONTEXTCALLBACKPOINT eCallbackPoint,
    const Datablock * pDatablock,
    void ** ppApplicationContext
    )
{
    static int count = 0;
    printf("ApplicationContext callback called: %s, %p, %p\n",
        eCallbackPoint == 0 ? "CREATE" : 
        eCallbackPoint == 1 ? "CLONE" : 
        eCallbackPoint == 2 ? "DESTROY" : "UNKNOWN", 
        pDatablock, *ppApplicationContext);

    if (NULL == *ppApplicationContext)
    {
        *ppApplicationContext = (void*)(++count);
        printf("** Setting NULL application context to %p ** \n",
            *ppApplicationContext);
    }
}

int run_graph_hostfunc_matmul_task(	
	int rows,
	int cols,
	int siblings,
	int iterations
	) 
{
    PTask::Runtime::Initialize();
    // CheckPlatformSupport(szfile, szshader);

    assert(rows == cols);
	CSimpleMatrix<ELEMTYPE>* vAMatrix;
	CSimpleMatrix<ELEMTYPE>* vBMatrix;
	CSimpleMatrix<ELEMTYPE>* vDMatrix = new CSimpleMatrix<ELEMTYPE>(rows, cols);
	configure_raw_matrix(rows, cols, &vAMatrix);
	configure_raw_matrix(rows, cols, &vBMatrix);

	MATADD_PARAMS params;
	params.g_tex_cols = cols;
	params.g_tex_rows = rows;
	UINT stride = sizeof(ELEMTYPE);
	UINT elements = rows*cols;

	int nChannelCount = 0;

    Graph * pGraph = new Graph();
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbRxC_uint", stride, cols, rows, 1);
    pDataTemplate->SetApplicationContextCallback(application_context_callback);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("matmulparms", sizeof(MATADD_PARAMS), 1, 1, 1);
	//CompiledKernel * pMatmulKernel		= COMPILE_KERNEL(szfile, szshader);
    CompiledKernel * pMatmulKernel      = PTask::Runtime::GetHostFunctionCompiledKernel("MatMulHostFunction", (FARPROC)hostfunc_htmatmul);
         
	const UINT uiInputCount = 3;
	const UINT uiOutputCount = 1;
	
	Port ** pAxBInputPorts = new Port*[uiInputCount];
	Port ** pAxBOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAxBInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A", 0);
	pAxBInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B", 1);
	pAxBInputPorts[2]		= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "parms", 3);
	pAxBOutputPorts[0]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "C", 2);

	Task * pAxBTask = pGraph->AddTask(pMatmulKernel, 
									  uiInputCount,
									  pAxBInputPorts,
									  uiOutputCount,
									  pAxBOutputPorts,
									  "AxBTask");

	assert(pAxBTask);
	// in this case, the iterations are handled 
    // inside the host code. Perhaps we should actually
    // pass the requested geometry to the host code?
    // pAxBTask->SetComputeGeometry(rows, cols, 1);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAxBInputPorts[0], "AInputChannel");
	GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAxBInputPorts[1], "BInputChannel");
	GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAxBInputPorts[2], "ABConstChannel");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAxBOutputPorts[0], "outputChannel");

	pGraph->Run();

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAMatrix->cells(), vAMatrix->arraysize(), pAInput);
	Datablock * pB		= PTask::Runtime::AllocateDatablock(pDataTemplate, vBMatrix->cells(), vBMatrix->arraysize(), pBInput);
	Datablock * pABPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pAxBParmsInput);

	pAInput->Push(pA);
	pBInput->Push(pB);
	pAxBParmsInput->Push(pABPrm);
	Datablock * pResultBlock = pOutput->Pull();

    pResultBlock->Lock();
	ELEMTYPE * psrc = (ELEMTYPE*) pResultBlock->GetDataPointer(FALSE);
	ELEMTYPE * pdst = vDMatrix->cells();
	memcpy(pdst, psrc, elements*stride);
    pResultBlock->Unlock();

	printf( "Verifying against CPU result..." );
	CSimpleMatrix<ELEMTYPE>* pAxB = matmul(vAMatrix, vBMatrix); // on CPU
	if(!check_matrix_result(vDMatrix, pAxB)) {
		printf("failure\n");
        print_matrix("D:", vDMatrix, 16);
        print_matrix("ref:", pAxB, 16);
    } else {
        // printf( "%s succeeded\n", szshader );
        printf( "run_graph_hostfunc_matmul_task succeeded\n");
    }

	pResultBlock->Release();
	pGraph->Stop();
	pGraph->Teardown();

	delete vAMatrix;
	delete vBMatrix;
	delete vDMatrix;
	delete pAxB;
	Graph::DestroyGraph(pGraph);
	delete [] pAxBInputPorts; 
	delete [] pAxBOutputPorts;

	PTask::Runtime::Terminate();

	return 0;
}
