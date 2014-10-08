//--------------------------------------------------------------------------------------
// File: graphmdmatmul.cpp
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
#include "graphmdmatmul.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

using namespace std;
using namespace PTask;

typedef struct MATRIX_METADATA_t
{
    int offset;
    int N;
} MATRIX_METADATA;

void 
create_matrix_stream(
    int object_rows,
    int object_cols,
    int N,
    vector<CSimpleMatrix<ELEMTYPE>* >& vAMatrices,
    vector<MATRIX_METADATA>& vAMetadata,
    vector<CSimpleMatrix<ELEMTYPE>* >& vBMatrices,
    vector<MATRIX_METADATA>& vBMetadata,
    vector<CSimpleMatrix<ELEMTYPE>* >& vCMatrices,
    vector<MATRIX_METADATA>& vCMetadata,
    vector<CSimpleMatrix<ELEMTYPE>* >& vReferenceMatrices,
    vector<MATRIX_METADATA>& vRefMetadata,
    void ** ppADatablock,
    void ** ppBDatablock,
    void ** ppMetadataBlock
    )
{
    int nMetadataSize = object_rows * object_cols * sizeof(MATRIX_METADATA);
    int nDataSize = object_rows * object_cols * N * N * sizeof(ELEMTYPE);
    ELEMTYPE * pAData = (ELEMTYPE*) new BYTE[nDataSize];
    ELEMTYPE * pBData = (ELEMTYPE*) new BYTE[nDataSize];
    MATRIX_METADATA * pMetadata = (MATRIX_METADATA*) new BYTE[nMetadataSize];
    int index = 0;
    for(int r=0; r<object_rows; r++) {
        for(int c=0; c<object_cols; c++) {            
            // configure metadata
            int curidx = r * object_cols + c;
            MATRIX_METADATA amd, bmd, refmd;
            amd.N = bmd.N = refmd.N = N;
            amd.offset = bmd.offset = refmd.offset = (N*N*curidx);
            vAMetadata[index] = amd;
            vBMetadata[index] = bmd;
            vCMetadata[index] = refmd; // we'll overwrite this on the pull
            vRefMetadata[index] = refmd;
            // configure data, and reference;
            CSimpleMatrix<ELEMTYPE>* vAMatrix;
	        CSimpleMatrix<ELEMTYPE>* vBMatrix;
	        CSimpleMatrix<ELEMTYPE>* vRefMatrix;
	        configure_raw_matrix(N, N, &vAMatrix);
	        configure_raw_matrix(N, N, &vBMatrix);
            vRefMatrix = matmul(vAMatrix, vBMatrix);
            vAMatrices[index] = vAMatrix;
            vBMatrices[index] = vBMatrix;
            vCMatrices[index] = new CSimpleMatrix<ELEMTYPE>(N, N); // allocate storage for result
            vReferenceMatrices[index] = vRefMatrix;
            memcpy(&pMetadata[index], &refmd, sizeof(MATRIX_METADATA));
            memcpy(&pAData[amd.offset], vAMatrix->cells(), N*N*sizeof(ELEMTYPE));
            memcpy(&pBData[amd.offset], vBMatrix->cells(), N*N*sizeof(ELEMTYPE));
            index++;
        }
    }
    *ppADatablock = pAData;
    *ppBDatablock = pBData;
    *ppMetadataBlock = pMetadata;
}

void 
free_matrix_stream(
    vector<CSimpleMatrix<ELEMTYPE>* >& vAMatrices,
    vector<CSimpleMatrix<ELEMTYPE>* >& vBMatrices,
    vector<CSimpleMatrix<ELEMTYPE>* >& vReferenceMatrices,
    void * pAData,
    void * pBData,
    void * pMetadata
    )
{
    vector<CSimpleMatrix<ELEMTYPE>*>::iterator vi;
    for(vi=vAMatrices.begin(); vi!=vAMatrices.end(); vi++) delete (*vi);
    for(vi=vBMatrices.begin(); vi!=vBMatrices.end(); vi++) delete (*vi);
    for(vi=vReferenceMatrices.begin(); vi!=vReferenceMatrices.end(); vi++) delete (*vi);
    vAMatrices.clear();
    vBMatrices.clear();
    vReferenceMatrices.clear();
    delete [] pAData;
    delete [] pBData;
    delete [] pMetadata;
}

void 
verify_result(
    void * pData,
    void * pMetadata,
    void * pRefData,
    void * pRefMetadata
    ) 
{
}
    

int 
run_graph_md_matmul_task(	
	char * szfile,
	char * szshader,
	int object_rows,
	int object_cols,
	int N, 
	int siblings,
	int iterations
	) 
{
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);
    
    void * pAData;
    void * pBData;
    void * pMetadata;
    vector<MATRIX_METADATA> vAMetadata, vBMetadata, vCMetadata, vRefMetadata;
	vector<CSimpleMatrix<ELEMTYPE>*> vAMatrices;
	vector<CSimpleMatrix<ELEMTYPE>*> vBMatrices;
	vector<CSimpleMatrix<ELEMTYPE>*> vCMatrices;
	vector<CSimpleMatrix<ELEMTYPE>*> vReferenceMatrices;
    create_matrix_stream(object_rows, object_cols, N, 
                        vAMatrices,
                        vAMetadata,
                        vBMatrices,
                        vBMetadata,
                        vCMatrices,
                        vCMetadata,
                        vReferenceMatrices,
                        vRefMetadata,
                        &pAData,
                        &pBData,
                        &pMetadata
                        );



    MATADD_PARAMS params;
	params.g_tex_cols = object_cols;
	params.g_tex_rows = object_rows;
	UINT stride = sizeof(ELEMTYPE)*N*N;
	UINT elements = object_rows*object_cols;

	int nChannelCount = 0;

	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	Graph * pGraph = new Graph();
	DatablockTemplate * pDataTemplate	    = PTask::Runtime::GetDatablockTemplate("dbRxC_NxNmat", stride, object_cols, object_rows, 1);
    DatablockTemplate * pMetadataTemplate   = PTask::Runtime::GetDatablockTemplate("dbRxC_uint2", sizeof(MATRIX_METADATA), object_cols, object_rows, 1);
	DatablockTemplate * pParmTemplate	    = PTask::Runtime::GetDatablockTemplate("matmulparms", sizeof(MATADD_PARAMS), 1, 1, 1);
	CompiledKernel * pMatmulKernel		    = COMPILE_KERNEL(szfile, szshader);

	const UINT uiInputCount = 5;
	const UINT uiOutputCount = 2;
	
	Port ** pAxBInputPorts = new Port*[uiInputCount];
	Port ** pAxBOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAxBInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++);
	pAxBInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pMetadataTemplate, uiUidCounter++);
    pAxBInputPorts[2]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++);
	pAxBInputPorts[3]		= PTask::Runtime::CreatePort(INPUT_PORT, pMetadataTemplate, uiUidCounter++);
	pAxBInputPorts[4]		= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++);
	pAxBOutputPorts[0]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++);
	pAxBOutputPorts[1]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pMetadataTemplate, uiUidCounter++);

	Task * pAxBTask = pGraph->AddTask(pMatmulKernel, 
									  uiInputCount,
									  pAxBInputPorts,
									  uiOutputCount,
									  pAxBOutputPorts,
									  "AxBTask");

	assert(pAxBTask);
	pAxBTask->SetComputeGeometry(object_rows, object_cols, 1);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAxBInputPorts[0], "AInputChannel");
	GraphInputChannel * pAMDInput			= pGraph->AddInputChannel(pAxBInputPorts[1], "AMetadataChannel");
	GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAxBInputPorts[2], "BInputChannel");
	GraphInputChannel * pBMDInput			= pGraph->AddInputChannel(pAxBInputPorts[3], "BMetadataChannel");
	GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAxBInputPorts[4], "ABConstChannel");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAxBOutputPorts[0], "outputChannel");
	GraphOutputChannel * pOutputMD			= pGraph->AddOutputChannel(pAxBOutputPorts[0], "outputMetadataChannel");

	pGraph->Run();

	double dInitTime;
	double dCopyToDeviceEnd; 
	double dCopyToDeviceTime;
	double dComputeEnd;
	double dComputeTime;
	double dTeardownStart;
	for(int i=0;i<iterations;i++) {

		Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, pAData, stride*elements, pAInput);
        Datablock * pAMeta  = PTask::Runtime::AllocateDatablock(pMetadataTemplate, pMetadata, sizeof(MATRIX_METADATA)*elements, pAMDInput);
		Datablock * pB		= PTask::Runtime::AllocateDatablock(pDataTemplate, pBData, stride*elements, pBInput);
        Datablock * pBMeta  = PTask::Runtime::AllocateDatablock(pMetadataTemplate, pMetadata, sizeof(MATRIX_METADATA)*elements, pBMDInput);
		Datablock * pABPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pAxBParmsInput);

		dInitTime = pTimer->elapsed(false);
		pAInput->Push(pA);
        pAMDInput->Push(pAMeta);
		pBInput->Push(pB);
        pBMDInput->Push(pBMeta);
		pAxBParmsInput->Push(pABPrm);
		dCopyToDeviceEnd = pTimer->elapsed(false);
		dCopyToDeviceTime = dCopyToDeviceEnd - dInitTime;
		Datablock * pResultBlock = pOutput->Pull();
        Datablock * pResultMeta = pOutputMD->Pull();
		dComputeEnd = pTimer->elapsed(false);
		dComputeTime = dComputeEnd - dCopyToDeviceEnd;

        // XXX: TODO:
        verify_result(pResultBlock, pResultMeta, NULL, NULL);

		dTeardownStart = pTimer->elapsed(false);
		pResultBlock->Release();
        pResultMeta->Release();
	}

	pGraph->Stop();
	pGraph->Teardown();

	free_matrix_stream(vAMatrices,
                        vBMatrices,
                        vReferenceMatrices,
                        pAData,
                        pBData,
                        pMetadata
                        );

	Graph::DestroyGraph(pGraph);
	delete [] pAxBInputPorts;
	delete [] pAxBOutputPorts;
	double dTeardownTime = pTimer->elapsed(false) - dTeardownStart;

	printf("InitTime:\t%f\n", dInitTime);
	printf("CopyToGPU:\t%f\n", dCopyToDeviceTime);
	printf("GPU Compute:\t%f\n", dComputeTime);
	printf("CPU Compute:\tunknown(verification unimplemented)\n");
	printf("Teardown:\t%f\n", dTeardownTime);

	delete pTimer;

	PTask::Runtime::Terminate();

	return 0;
}

