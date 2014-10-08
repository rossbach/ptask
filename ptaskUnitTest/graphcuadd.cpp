//--------------------------------------------------------------------------------------
// File: graphcuadd.cpp
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
#include <string>
#include "matrixtask.h"
#include "SimpleMatrix.h"
#include "SimpleVector.h"
#include "matmul.h"
#include "graphcuadd.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

extern int  g_serializationMode;

using namespace std;
using namespace PTask;

float * 
random_vector(
    int n
    )
{
    float * pVector = new float[n];
    for(int i=0; i<n; i++) 
        pVector[i] = rand()/(float)RAND_MAX;
    return pVector;
}

float *
vector_add(
    float * pA,
    float * pB,
    int n
    ) 
{
    float * pC = new float[n];
    for(int i=0; i<n; i++)
        pC[i] = pA[i]+pB[i];
    return pC;
}

BOOL
compare_vectors(
    float * pA,
    float * pB,
    int n
    )
{
    for(int i=0; i<n; i++) {
        if(fabs(pA[i]-pB[i]) > 1e-7f) {
            printf("diff at index %d: A=%f B=%f\n", i, pA[i], pB[i]);
            return FALSE;
        }
    }
    return TRUE;
}

Graph * initialize_graphcuadd_graph(char * szfile, char * szshader, UINT stride, int n, const char * graphFileName)
{

    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    Graph * pGraph = new Graph();
    if (2 == g_serializationMode)
    {
        pGraph->Deserialize(graphFileName);
        return pGraph;
    }    

    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("matmulparms", sizeof(VECADD_PARAMS), PTPARM_INT);

    CompiledKernel * pKernel = PTask::Runtime::GetCompiledKernel(szfile, szshader, g_szCompilerOutputBuffer, COBBUFSIZE);
    CheckCompileSuccess(szfile, szshader, pKernel);

    const UINT uiInputCount = 3;
    const UINT uiOutputCount = 1;
    Port ** pAplusBInputPorts = new Port*[uiInputCount];
    Port ** pAplusBOutputPorts = new Port*[uiOutputCount];

    UINT uiUidCounter		= 0;
    pAplusBInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A", 0);
    pAplusBInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B", 1);
    pAplusBInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N", 3);
    pAplusBOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "C", 2);

    Task * pAplusBTask = pGraph->AddTask(pKernel, 
                                          uiInputCount,
                                          pAplusBInputPorts,
                                          uiOutputCount,
                                          pAplusBOutputPorts,
                                          "AplusBTask");

    assert(pAplusBTask);
    pAplusBTask->SetComputeGeometry(n, 1, 1);
    PTASKDIM3 threadBlockSize(256, 1, 1);
    PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
    pAplusBTask->SetBlockAndGridSize(gridSize, threadBlockSize);

    GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAplusBInputPorts[0], "AInputChannel");
    GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAplusBInputPorts[1], "BInputChannel");
    GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAplusBInputPorts[2], "NConstChannel");
    GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAplusBOutputPorts[0], "outputChannel");

    delete [] pAplusBInputPorts;
    delete [] pAplusBOutputPorts;
    return pGraph;
}

int run_graph_cuda_vecadd_task(	
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
    float* vBVector = random_vector(n);

    UINT stride = sizeof(float);
    VECADD_PARAMS params;
    params.N = n;

    int nChannelCount = 0;

    CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
    pTimer->reset();

    // Create, serialize or deserialize PTask Graph, depending on value of g_serializationMode.
    const char * graphFileName = "graphcuadd.xml";
    Graph * pGraph = initialize_graphcuadd_graph(szfile, szshader, stride, n, graphFileName);
    if (g_serializationMode == 1)
    {
        pGraph->Serialize(graphFileName);
        printf( "%s succeeded\n", szshader );
        return 0;
    }

    pGraph->Run();

    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float");
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("matmulparms");

    GraphInputChannel * pAInput				= (GraphInputChannel*)pGraph->GetChannel("AInputChannel");
    GraphInputChannel * pBInput				= (GraphInputChannel*)pGraph->GetChannel("BInputChannel");
    GraphInputChannel * pAxBParmsInput		= (GraphInputChannel*)pGraph->GetChannel("NConstChannel");
    GraphOutputChannel * pOutput			= (GraphOutputChannel*)pGraph->GetChannel("outputChannel");

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, n*stride, pAInput);
    Datablock * pB		= PTask::Runtime::AllocateDatablock(pDataTemplate, vBVector, n*stride, pBInput);
    Datablock * pABPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pAxBParmsInput);

    double dInitTime = pTimer->elapsed(false);
    pAInput->Push(pA);
    pBInput->Push(pB);
    pAxBParmsInput->Push(pABPrm);
    pA->Release();
    pB->Release();
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
    float* vReference = vector_add(vAVector, vBVector, n);
    double dHostEnd = pTimer->elapsed(false) - dHostStart;

    if(!compare_vectors(vReference, psrc, n))
        printf("failure\n");
    else 
        printf( "%s succeeded\n", szshader );

    double dTeardownStart = pTimer->elapsed(false);
    pResultBlock->Release();
    pGraph->Stop();
    pGraph->Teardown();

    delete [] vAVector;
    delete [] vBVector;
    delete [] vReference;
    Graph::DestroyGraph(pGraph);
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

