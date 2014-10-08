///-------------------------------------------------------------------------------------------------
// file:	permanentblocks.cpp
//
// summary:	Implements the permanentblocks class
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
#include "permanentblocks.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

using namespace std;
using namespace PTask;

extern float * random_vector(int n);
extern BOOL compare_vectors(float*pA, float*pB, int n);
extern float * vector_scale(float * pA, float scalar, int n);

int run_graph_cuda_permanentblocks_task(	
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

    CompiledKernel * pKernel = PTask::Runtime::GetCompiledKernel(szfile, szshader, g_szCompilerOutputBuffer, COBBUFSIZE);
    CheckCompileSuccess(szfile, szshader, pKernel);

    assert(rows == 1 || cols == 1);		// vector, not matrix!
    int n = rows * cols;
    float* vAVector = random_vector(n);
    float* vBVector = random_vector(n);

    VECADD_PARAMS params;
    params.N = n;
    float fScalar = 2.0f;
    UINT stride = sizeof(float);
    UINT nPulse = 1;

    int nChannelCount = 0;

    CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
    pTimer->reset();

    Graph * pGraph = new Graph();
    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
    DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", sizeof(float), PTPARM_FLOAT);
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);

    const UINT uiInputCount = 4;
    const UINT uiOutputCount = 1;
    
    Port ** pAplusBInputPorts = new Port*[uiInputCount];
    Port ** pAplusBOutputPorts = new Port*[uiOutputCount];

    UINT uiUidCounter		= 0;
    pAplusBInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A", 0, 0);
    pAplusBInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B", 1);
    pAplusBInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar", 2);
    pAplusBInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N", 3);
    pAplusBOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)", 0);

    Task * pAplusBTask = pGraph->AddTask(pKernel, 
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

    GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAplusBInputPorts[0], "AInputChannel");
    GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAplusBInputPorts[1], "BInputChannel");
    GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAplusBOutputPorts[0], "outputChannel");
    pGraph->BindDescriptorPort(pAplusBInputPorts[0], pAplusBInputPorts[3]);
    pGraph->BindControlPort(pAplusBInputPorts[1], pAplusBOutputPorts[0]);

    Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), NULL);
    pAplusBInputPorts[2]->SetPermanentBlock(pScPrm);
    pScPrm->Release();

    pGraph->Run();

    Datablock * pA		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, n*stride, pAInput);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();

    for(int i=0; i<3; i++) {
        pAInput->Push(pA);
        Datablock * pB = PTask::Runtime::AllocateDatablock(pDataTemplate, vBVector, n*stride, pBInput);
        pBInput->Push(pB);
        pB->Release();
    }
    Datablock * pB = PTask::Runtime::AllocateDatablock(pDataTemplate, vBVector, n*stride, pBInput);
    pB->Lock();
    pB->SetControlSignal(DBCTLC_EOF);
    pB->Unlock();
    pAInput->Push(pA);
    pBInput->Push(pB);
    pB->Release();
    Datablock * pResultBlock = pOutput->Pull();
    pResultBlock->Lock();
    float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
    pResultBlock->Unlock();
    int nErrorTolerance = 20;
    float* vReference = vector_scale(vAVector, fScalar*fScalar*fScalar*fScalar, n);
    if(!compare_vectors(vReference, psrc, n)) {
        printf("failure. ref[0] = %+f, out[0] = %+f\n", vReference[0], psrc[0]);
    } else  {
        printf( "%s succeeded\n", szshader );
    }
    delete [] vReference;
    pA->Release();
    pResultBlock->Release();
    pGraph->Stop();
    pGraph->Teardown();

    delete [] vAVector;
    delete [] vBVector;
	Graph::DestroyGraph(pGraph);
    delete [] pAplusBInputPorts;
    delete [] pAplusBOutputPorts;

    delete pTimer;

    PTask::Runtime::Terminate();

    return 0;
}

