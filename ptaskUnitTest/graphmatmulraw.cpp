//--------------------------------------------------------------------------------------
// File: graphmatmulraw.cpp
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
#include "graphmatmulraw.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

using namespace std;
using namespace PTask;


int run_graph_matmulraw_task(	
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
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbRxC_uint", stride*cols*rows, 1, 1, 1, false, true);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("matmulparms", sizeof(MATADD_PARAMS), 1, 1, 1, false, true);
	CompiledKernel * pMatmulKernel		= COMPILE_KERNEL(szfile, szshader);

	const UINT uiInputCount = 4;
	const UINT uiOutputCount = 2;
	
	Port ** pAxBInputPorts = new Port*[uiInputCount];
	Port ** pAxBOutputPorts = new Port*[uiOutputCount];
	Port ** pAxBxCInputPorts = new Port*[uiInputCount];
	Port ** pAxBxCOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAxBInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "p1");
	pAxBInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pParmTemplate, uiUidCounter++, "p2");
	pAxBInputPorts[2]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "p3");
	pAxBInputPorts[3]		= PTask::Runtime::CreatePort(INPUT_PORT, pParmTemplate, uiUidCounter++, "p4");
	pAxBOutputPorts[0]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "p5");;
	pAxBOutputPorts[1]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pParmTemplate, uiUidCounter++, "p6");;
	pAxBxCInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "p7");
	pAxBxCInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pParmTemplate, uiUidCounter++, "p8");
	pAxBxCInputPorts[2]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "p9");
	pAxBxCInputPorts[3]		= PTask::Runtime::CreatePort(INPUT_PORT, pParmTemplate, uiUidCounter++, "p10");
	pAxBxCOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "p11");;	
	pAxBxCOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pParmTemplate, uiUidCounter++, "p12");;	

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
	GraphInputChannel * pAInput_md			= pGraph->AddInputChannel(pAxBInputPorts[1], "AMetadata");
	GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAxBInputPorts[2], "BInputChannel");
	GraphInputChannel * pBInput_md			= pGraph->AddInputChannel(pAxBInputPorts[3], "BMetadata");
	GraphInputChannel * pCInput				= pGraph->AddInputChannel(pAxBxCInputPorts[2], "CInputChannel");
	GraphInputChannel * pCInput_md		    = pGraph->AddInputChannel(pAxBxCInputPorts[3], "CMetadata");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAxBxCOutputPorts[0], "outputChannel");
	GraphOutputChannel * pOutput_md			= pGraph->AddOutputChannel(pAxBxCOutputPorts[1], "outputChannel_md");
	
    pGraph->AddInternalChannel(pAxBOutputPorts[0], pAxBxCInputPorts[0], "internalChannel");
    pGraph->AddInternalChannel(pAxBOutputPorts[1], pAxBxCInputPorts[1], "internalMD");

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

        BUFFERACCESSFLAGS rawFlags = (PT_ACCESS_HOST_WRITE | PT_ACCESS_ACCELERATOR_READ | PT_ACCESS_BYTE_ADDRESSABLE);
        UINT uiDataSize = vAMatrix->rows() * vAMatrix->cols() * sizeof(ELEMTYPE);
        UINT uiMetadataSize = 2 * sizeof(int);

        Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAMatrix->cells(), vAMatrix->arraysize(), pAInput, rawFlags, uiDataSize);
        Datablock * pAmd    = PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pAInput_md, rawFlags, uiMetadataSize);
        Datablock * pB		= PTask::Runtime::AllocateDatablock(pDataTemplate, vBMatrix->cells(), vBMatrix->arraysize(), pBInput, rawFlags, uiDataSize);
        Datablock * pBmd    = PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pBInput_md, rawFlags, uiMetadataSize);
		Datablock * pC		= PTask::Runtime::AllocateDatablock(pDataTemplate, vCMatrix->cells(), vCMatrix->arraysize(), pCInput, rawFlags, uiDataSize);
        Datablock * pCmd    = PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pCInput_md, rawFlags, uiMetadataSize);

		dInitTime = pTimer->elapsed(false);
		pAInput->Push(pA);
        pAInput_md->Push(pAmd);
		pBInput->Push(pB);
        pBInput_md->Push(pBmd);
		pCInput->Push(pC);
        pCInput_md->Push(pCmd);
        pA->Release();
        pAmd->Release();
        pB->Release();
        pBmd->Release();
        pC->Release();
        pCmd->Release();

		dCopyToDeviceEnd = pTimer->elapsed(false);
		dCopyToDeviceTime = dCopyToDeviceEnd - dInitTime;
		Datablock * pResultBlock = pOutput->Pull();
        Datablock * pMDResult = pOutput_md->Pull();
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
            print_matrix("D:", vDMatrix);
            print_matrix("ref:", pAxB);
        } else {
			printf( "%s succeeded\n", szshader );
        }

		dTeardownStart = pTimer->elapsed(false);
		pResultBlock->Release();
        pMDResult->Release();
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

int run_graph_matmulraw_task_easy(	
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
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbRxC_uint", stride * cols * rows, 1, 1, 1, true);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("matmulparms", sizeof(MATADD_PARAMS), 1, 1, 1, true);
	CompiledKernel * pMatmulKernel		= COMPILE_KERNEL(szfile, szshader);

	const UINT uiInputCount = 4;
	const UINT uiOutputCount = 2;
	
	Port ** pAxBInputPorts = new Port*[uiInputCount];
	Port ** pAxBOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAxBInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++);
	pAxBInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pParmTemplate, uiUidCounter++);
	pAxBInputPorts[2]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++);
	pAxBInputPorts[3]		= PTask::Runtime::CreatePort(INPUT_PORT, pParmTemplate, uiUidCounter++);
	pAxBOutputPorts[0]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++);
	pAxBOutputPorts[1]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pParmTemplate, uiUidCounter++);

	Task * pAxBTask = pGraph->AddTask(pMatmulKernel, 
									  uiInputCount,
									  pAxBInputPorts,
									  uiOutputCount,
									  pAxBOutputPorts,
									  "AxBTask");

	assert(pAxBTask);
	pAxBTask->SetComputeGeometry(rows, cols, 1);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAxBInputPorts[0], "AInputChannel");
	GraphInputChannel * pAInput_md			= pGraph->AddInputChannel(pAxBInputPorts[1], "AInputChannel_md");
	GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAxBInputPorts[2], "BInputChannel");
	GraphInputChannel * pBInput_md			= pGraph->AddInputChannel(pAxBInputPorts[3], "BInputChannel_md");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAxBOutputPorts[0], "outputChannel");
	GraphOutputChannel * pOutput_md			= pGraph->AddOutputChannel(pAxBOutputPorts[1], "outputChannel_md");

	pGraph->Run();

    BUFFERACCESSFLAGS rawFlags = (PT_ACCESS_HOST_WRITE | PT_ACCESS_ACCELERATOR_READ | PT_ACCESS_BYTE_ADDRESSABLE);
    UINT uiDataSize = vAMatrix->rows() * vAMatrix->cols() * sizeof(ELEMTYPE);
    UINT uiMetadataSize = 2 * sizeof(int);

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAMatrix->cells(), vAMatrix->arraysize(), pAInput, rawFlags, uiDataSize);
    Datablock * pAmd    = PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pAInput_md, rawFlags, uiMetadataSize);
    Datablock * pB		= PTask::Runtime::AllocateDatablock(pDataTemplate, vBMatrix->cells(), vBMatrix->arraysize(), pBInput, rawFlags, uiDataSize);
    Datablock * pBmd    = PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pBInput_md, rawFlags, uiMetadataSize);

	pAInput->Push(pA);
    pAInput_md->Push(pAmd);
	pBInput->Push(pB);
    pBInput_md->Push(pBmd);
	Datablock * pResultBlock = pOutput->Pull();
    Datablock * pResultMD = pOutput_md->Pull();

	ELEMTYPE * psrc = (ELEMTYPE*) pResultBlock->GetDataPointer(FALSE);
	ELEMTYPE * pdst = vDMatrix->cells();
	memcpy(pdst, psrc, elements*stride);

	printf( "Verifying against CPU result..." );
	CSimpleMatrix<ELEMTYPE>* pAxB = matmul(vAMatrix, vBMatrix); // on CPU
	if(!check_matrix_result(vDMatrix, pAxB)) {
		printf("failure\n");
        print_matrix("D:", vDMatrix);
        print_matrix("ref:", pAxB);
    } else {
        printf( "%s succeeded\n", szshader );
    }

	pResultBlock->Release();
    pResultMD->Release();
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
