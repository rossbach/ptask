///-------------------------------------------------------------------------------------------------
// file:	descportsout.cpp
//
// summary:	test kernal for output descriptor ports test case:
//          The test does a normal vector scale, but the output data
//          block should also have 'N' in the metadata channel and
//          the entire contents of the pMetaData array in the template channel.
///-------------------------------------------------------------------------------------------------

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
#include "deferredports.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

using namespace std;
using namespace PTask;

extern float * random_vector(int n);
extern BOOL compare_vectors(float*pA, float*pB, int n);
extern float * vector_scale(float * pA, float scalar, int n);

int run_graph_cuda_descportsout_task(	
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
    const UINT nMetaChannelStride = sizeof(int);
    const UINT nMetaChannelElems = 1;
    const UINT nTmplChannelStride = sizeof(float);
    const UINT nTmplChannelElems = 100;
    const UINT cbTmplChannelData = nTmplChannelElems * nTmplChannelStride;
    float * pTmplChannelData = new float[nTmplChannelElems];
    for(int i=0; i<nTmplChannelElems; i++) pTmplChannelData[i] = (float) i;

	int nChannelCount = 0;

	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();
	Graph * pGraph = new Graph();
    CompiledKernel * pKernel = Runtime::GetCompiledKernel(szfile, szshader, g_szCompilerOutputBuffer, COBBUFSIZE);
    CheckCompileSuccess(szfile, szshader, pKernel);


    /*
        // set up templates for the following signature:
        scale(
            0. __in  float* A,             // matrix
            1. __in  float scalar,         // scalar for A * scalar
            2. __in  int N,                // size of A, pOut
            3. __out float * pOut,         // output result
            4. __out int * pMDOut,         // meta data channel out (should have N)
            5. __in  float * pTmplDataIn,  // input data destined for output template channel
            6. __in  int nTmpl,            // size of pTmplDataIn
            7. __out float * pTmplDataOut  // output buffer for tmpl channel
            )

        On output, we want not just the scaled vector, but
        the metadata channel of the block should have the value of N,
        and the channel for pMetaOut should contain all of pMetaData's contents.
    */
	DatablockTemplate * pDataTemplate	= Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
	DatablockTemplate * pScaleTemplate	= Runtime::GetDatablockTemplate("vscale", sizeof(float), PTPARM_FLOAT);
	DatablockTemplate * pParmTemplate	= Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
	DatablockTemplate * pMDTemplate	    = Runtime::GetDatablockTemplate("dbV1_uint1", nMetaChannelStride, nMetaChannelElems, 1, 1);
	DatablockTemplate * pMDSizeTemplate	= Runtime::GetDatablockTemplate("mdsize", sizeof(UINT), PTPARM_INT);
	DatablockTemplate * pTCTemplate	    = Runtime::GetDatablockTemplate("dbV1_float10", nTmplChannelStride, nTmplChannelElems, 1, 1);
	DatablockTemplate * pTCSizeTemplate	= Runtime::GetDatablockTemplate("tcsize", sizeof(UINT), PTPARM_INT);

	const UINT uiInputCount = 8;    // 5 kernel params + 3 meta port allocator for pMetaOut
	const UINT uiOutputCount = 3;    //	
	Port ** pInputPorts = new Port*[uiInputCount];
	Port ** pOutputPorts = new Port*[uiOutputCount];
	UINT uiUidCounter		= 0;
	pInputPorts[0]	= Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A", 0);
	pInputPorts[1]	= Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar", 1);
	pInputPorts[2]	= Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N", 2);
	pInputPorts[3]	= Runtime::CreatePort(INPUT_PORT, pTCTemplate, uiUidCounter++, "pTmplDataIn", 5);
	pInputPorts[4]	= Runtime::CreatePort(STICKY_PORT, pTCSizeTemplate, uiUidCounter++, "nTmpl", 6);
	pInputPorts[5]	= Runtime::CreatePort(META_PORT, pParmTemplate, uiUidCounter++, "pOut-meta", 0);
	pInputPorts[6]	= Runtime::CreatePort(META_PORT, pMDSizeTemplate, uiUidCounter++, "pMDOut-meta", 1);
	pInputPorts[7]	= Runtime::CreatePort(META_PORT, pTCSizeTemplate, uiUidCounter++, "pTmplDataOut-meta", 2);
	pOutputPorts[0]	= Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "pOut", 3);
	pOutputPorts[1]	= Runtime::CreatePort(OUTPUT_PORT, pMDTemplate, uiUidCounter++, "pMDOut", 4);
	pOutputPorts[2]	= Runtime::CreatePort(OUTPUT_PORT, pTCTemplate, uiUidCounter++, "pTmplDataOut", 7);

	Task * pTask = pGraph->AddTask(pKernel, 
								   uiInputCount,
								   pInputPorts,
								   uiOutputCount,
								   pOutputPorts,
								   "AxScalar");

	assert(pTask);
	pTask->SetComputeGeometry(n, 1, 1);
	PTASKDIM3 threadBlockSize(256, 1, 1);
	PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
	pTask->SetBlockAndGridSize(gridSize, threadBlockSize);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pInputPorts[0], "AInputChannel");
	GraphInputChannel * pAxBScaleInput		= pGraph->AddInputChannel(pInputPorts[1], "ScalarChannel");
	GraphInputChannel * pTmplDataInput		= pGraph->AddInputChannel(pInputPorts[3], "TmplChannelIn");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pOutputPorts[0], "outputChannel");

    // set up meta channels;
    // actually--not needed: use descriptor ports!
    // GraphInputChannel * pMainMetaInput	= pGraph->AddInputChannel(pInputPorts[5], "MainMetaChannel");
    // GraphInputChannel * pMDMetaInput		= pGraph->AddInputChannel(pInputPorts[6], "MDMetaChannel");
    // GraphInputChannel * pTmplMetaInput	= pGraph->AddInputChannel(pInputPorts[7], "TmplMetaChannel");
    
    pGraph->BindDescriptorPort(pInputPorts[0], pInputPorts[2]);   // N derived from A input
    pGraph->BindDescriptorPort(pInputPorts[3], pInputPorts[4]);   // nTmpl derived from pTmplDataIn input
    pGraph->BindDescriptorPort(pInputPorts[0], pInputPorts[5]);   // out size allocator derived from in
    pGraph->BindDescriptorPort(pInputPorts[1], pInputPorts[6]);   // size alloc for MD out from scale (just taking advantage of same size--these are not really coupled data)
    pGraph->BindDescriptorPort(pInputPorts[3], pInputPorts[7]);   // nTmpl out size derived from pTmplDataIn input

    pGraph->BindDescriptorPort(pOutputPorts[0], pOutputPorts[1], DF_METADATA_SOURCE);  // get meta data channel from port 1
    pGraph->BindDescriptorPort(pOutputPorts[0], pOutputPorts[2], DF_TEMPLATE_SOURCE);  // get template data channel from port 2

	pGraph->Run();

	Datablock * pA		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pAInput);
	Datablock * pScParm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pAxBScaleInput);
	Datablock * pTmplIn  = PTask::Runtime::AllocateDatablock(pTCTemplate, pTmplChannelData, cbTmplChannelData, pTmplDataInput);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();
	pAxBScaleInput->Push(pScParm);
    pScParm->Release();

    pAInput->Push(pA);
    pA->Release();
    pTmplDataInput->Push(pTmplIn);
    pTmplIn->Release();

    Datablock * pResultBlock = pOutput->Pull();
    pResultBlock->Lock();
	float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
    int * pMD = (int*) pResultBlock->GetMetadataPointer(FALSE);
    float * pTC = (float*) pResultBlock->GetTemplatePointer(FALSE);
	int nErrorTolerance = 20;
	float* vReference = vector_scale(vAVector, fScalar, n);

    BOOL bSuccess = TRUE;
	if(!compare_vectors(vReference, psrc, n)) {
        bSuccess = FALSE;
        printf("failure in data channel. ref[0] = %+f, out[0] = %+f\n", vReference[0], psrc[0]);
    } 
    if(*pMD != n) {
        bSuccess = FALSE;
        printf("failure in metadata channel. expected %d, got %d\n", n, *pMD);
    }
    for(int i=0; i<nTmplChannelElems; i++) {
        if(pTC[i] != pTmplChannelData[i]) {
            bSuccess = FALSE;
            printf("failure in template channel[%d]. expected %+f, got %+f\n", i, pTmplChannelData[i], pTC[i]);
            break;
        }
    }

    if(bSuccess) {
        printf( "%s succeeded\n", szshader );
    }
    delete [] vReference;
    pResultBlock->Unlock();
	pResultBlock->Release();
	pGraph->Stop();
	pGraph->Teardown();

	delete [] vAVector;
    Graph::DestroyGraph(pGraph);
	delete [] pInputPorts;
	delete [] pOutputPorts;
    delete [] pTmplChannelData;

	delete pTimer;

	PTask::Runtime::Terminate();

	return 0;
}

