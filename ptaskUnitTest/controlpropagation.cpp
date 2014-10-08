//--------------------------------------------------------------------------------------
// File: controlpropagation.cpp
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
#include "SimpleVector.h"
#include "matmul.h"
#include "controlpropagation.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

extern int  g_serializationMode;

using namespace std;
using namespace PTask;

extern float * random_vector(int n);
extern BOOL compare_vectors(float*pA, float*pB, int n);
extern float * vector_scale(float * pA, float scalar, int n);

int run_graph_cuda_controlpropagation_task_simple(	
	char * szfile,
	char * szshader,
	int rows,
	int cols,
	int siblings,
	int iterations
	) 
{
    if (0 != g_serializationMode)
    {
        printf("run_graph_cuda_controlpropagation_task_simple not referenced in main.cpp\n");
        printf("so has not been upgraded to work with graph serialization. Use run_graph_cuda_controlpropagation_task\n");
        printf("or port this method to use the same pattern.\n");
        printf("failure.");
        return 1;
    }

    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

	assert(rows == 1 || cols == 1);		// vector, not matrix!
	int n = rows * cols;
	float* vAVector = random_vector(n);
    float* vBVector = random_vector(n);

	VECADD_PARAMS params;
	params.N = n;
    float fScalar = 2.0f;
	UINT stride = sizeof(float);
    UINT uiVectorBytes = stride * n;
    UINT nPulse = 1;

	int nChannelCount = 0;

	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	Graph * pGraph = new Graph();
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
	DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", sizeof(float), PTPARM_FLOAT);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);

    CompiledKernel * pKernel = PTask::Runtime::GetCompiledKernel(szfile, szshader, g_szCompilerOutputBuffer, COBBUFSIZE);
    CheckCompileSuccess(szfile, szshader, pKernel);

	const UINT uiInputCount = 4;
	const UINT uiOutputCount = 2;
	
	Port ** pAInputPorts = new Port*[uiInputCount];
	Port ** pAOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_top", 0, 0);
	pAInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_top", 1, 1);
	pAInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_top", 2);
	pAInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_top", 3);
	pAOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_top", 0);
	pAOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B(out)_top", 1);

	Task * pTopTask = pGraph->AddTask(pKernel, 
									uiInputCount,
									pAInputPorts,
									uiOutputCount,
									pAOutputPorts,
									"top");

	assert(pTopTask);
	pTopTask->SetComputeGeometry(n, 1, 1);
	PTASKDIM3 threadBlockSize(256, 1, 1);
	PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
	pTopTask->SetBlockAndGridSize(gridSize, threadBlockSize);


	GraphInputChannel * pTopAInput				= pGraph->AddInputChannel(pAInputPorts[0], "AInputChannel");
	GraphInputChannel * pTopBInput				= pGraph->AddInputChannel(pAInputPorts[1], "BInputChannel");
	GraphInputChannel * pTopAxBScaleInput		= pGraph->AddInputChannel(pAInputPorts[2], "ScalarChannel");
    pGraph->BindDescriptorPort(pAInputPorts[0], pAInputPorts[3]);
    pGraph->BindControlPropagationPort(pAInputPorts[0], pAOutputPorts[0]);
    pGraph->BindControlPropagationPort(pAInputPorts[0], pAOutputPorts[1]);

    GraphOutputChannel * pAOutputA			= pGraph->AddOutputChannel(pAOutputPorts[0], "outputChannel_left_A");
    GraphOutputChannel * pAOutputB			= pGraph->AddOutputChannel(pAOutputPorts[1], "outputChannel_left_B");

    PTask::Runtime::CheckGraphSemantics(pGraph, TRUE, TRUE);

	pGraph->Run();

    Datablock * pScParm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pTopAxBScaleInput);
    pTopAxBScaleInput->Push(pScParm);
    pScParm->Release();

    int nFailures = 0;
    int nIter = 1;
    for(int i=0; i<nIter; i++) {
	    Datablock * pA		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pTopAInput);
	    Datablock * pB		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pTopBInput);

        pA->Lock();
        pA->SetRecordCount(params.N);
        if(i == nIter - 1) 
            pA->SetControlSignal(DBCTLC_EOF);
        pA->Unlock();


        pTopAInput->Push(pA);
        pTopBInput->Push(pB);
        pA->Release();
        pB->Release();

        Datablock * pLAout = pAOutputA->Pull();
        Datablock * pLBout = pAOutputB->Pull();
        pLAout->Lock();
        pLBout->Lock();
        if(!pLAout->HasAnyControlSignal()) {
            printf("Failed--did not get expected EOF on left B output!");
            nFailures++;
        }
        if(!pLBout->HasAnyControlSignal()) {
            printf("Failed--did not get expected EOF on right B output!");
            nFailures++;
        }
        pLAout->Unlock();
        pLBout->Unlock();
        pLAout->Release();
        pLBout->Release();
    }

    if(nFailures) {
        printf("failure.");
    } else  {
        printf( "%s succeeded", szshader );
    }

    pGraph->Stop();
	pGraph->Teardown();

	delete [] vAVector;
    Graph::DestroyGraph(pGraph);
	delete [] pAInputPorts;
	delete [] pAOutputPorts;

	delete pTimer;

	PTask::Runtime::Terminate();

	return 0;
}

Graph * initialize_controlpropagation_graph(char * szfile, char * szshader, UINT stride, int n, const char * graphFileName)
{
    Graph * pGraph = new Graph();
    if (2 == g_serializationMode)
    {
        pGraph->Deserialize(graphFileName);
        return pGraph;
    }    

	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
	DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", sizeof(float), PTPARM_FLOAT);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);

    CompiledKernel * pKernel = PTask::Runtime::GetCompiledKernel(szfile, szshader, g_szCompilerOutputBuffer, COBBUFSIZE);
    CheckCompileSuccess(szfile, szshader, pKernel);

	const UINT uiInputCount = 4;
	const UINT uiOutputCount = 2;
	
	Port ** pAInputPorts = new Port*[uiInputCount];
	Port ** pAOutputPorts = new Port*[uiOutputCount];
	Port ** pLeftInputPorts = new Port*[uiInputCount];
	Port ** pLeftOutputPorts = new Port*[uiOutputCount];
	Port ** pRightInputPorts = new Port*[uiInputCount];
	Port ** pRightOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_top", 0, 0);
	pAInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_top", 1, 1);
	pAInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_top", 2);
	pAInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_top", 3);
	pAOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_top", 0);
	pAOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B(out)_top", 1);

	pLeftInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_left", 0, 0);
	pLeftInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_left", 1, 1);
	pLeftInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_left", 2);
	pLeftInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_left", 3);
	pLeftOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_left", 0);
	pLeftOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B(out)_left", 1);

	pRightInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_left", 0, 0);
	pRightInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_left", 1, 1);
	pRightInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_left", 2);
	pRightInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_left", 3);
	pRightOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_left", 0);
	pRightOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B(out)_left", 1);

	Task * pTopTask = pGraph->AddTask(pKernel, 
									uiInputCount,
									pAInputPorts,
									uiOutputCount,
									pAOutputPorts,
									"top");

	Task * pLeftTask = pGraph->AddTask(pKernel, 
									uiInputCount,
									pLeftInputPorts,
									uiOutputCount,
									pLeftOutputPorts,
									"left");

	Task * pRightTask = pGraph->AddTask(pKernel, 
									uiInputCount,
									pRightInputPorts,
									uiOutputCount,
									pRightOutputPorts,
									"right");

	assert(pTopTask);
	assert(pLeftTask);
	assert(pRightTask);
	pTopTask->SetComputeGeometry(n, 1, 1);
	pLeftTask->SetComputeGeometry(n, 1, 1);
	pRightTask->SetComputeGeometry(n, 1, 1);
	PTASKDIM3 threadBlockSize(256, 1, 1);
	PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
	pTopTask->SetBlockAndGridSize(gridSize, threadBlockSize);
	pLeftTask->SetBlockAndGridSize(gridSize, threadBlockSize);
	pRightTask->SetBlockAndGridSize(gridSize, threadBlockSize);

	GraphInputChannel * pTopAInput				= pGraph->AddInputChannel(pAInputPorts[0], "AInputChannel");
	GraphInputChannel * pTopBInput				= pGraph->AddInputChannel(pAInputPorts[1], "BInputChannel");
	GraphInputChannel * pTopAxBScaleInput		= pGraph->AddInputChannel(pAInputPorts[2], "ScalarChannel");
    pGraph->BindDescriptorPort(pAInputPorts[0], pAInputPorts[3]);

	GraphInputChannel * pLeftBInput				= pGraph->AddInputChannel(pLeftInputPorts[1], "BInputChannel_left");
	GraphInputChannel * pLeftAxBScaleInput		= pGraph->AddInputChannel(pLeftInputPorts[2], "ScalarChannel_left");
    pGraph->BindDescriptorPort(pLeftInputPorts[0], pLeftInputPorts[3]);

	GraphInputChannel * pRightAInput			= pGraph->AddInputChannel(pRightInputPorts[0], "AInputChannel_right");
	GraphInputChannel * pRightAxBScaleInput		= pGraph->AddInputChannel(pRightInputPorts[2], "ScalarChannel_right");
    pGraph->BindDescriptorPort(pRightInputPorts[0], pRightInputPorts[3]);   

    GraphOutputChannel * pLeftOutputA			= pGraph->AddOutputChannel(pLeftOutputPorts[0], "outputChannel_left_A");
    GraphOutputChannel * pLeftOutputB			= pGraph->AddOutputChannel(pLeftOutputPorts[1], "outputChannel_left_B");
    GraphOutputChannel * pRightOutputA			= pGraph->AddOutputChannel(pRightOutputPorts[0], "outputChannel_Right_A");
    GraphOutputChannel * pRightOutputB			= pGraph->AddOutputChannel(pRightOutputPorts[1], "outputChannel_Right_B");

    pGraph->AddInternalChannel(pAOutputPorts[0], pLeftInputPorts[0]);
    pGraph->AddInternalChannel(pAOutputPorts[1], pRightInputPorts[1]);

    pGraph->BindControlPropagationPort(pAInputPorts[0], pAOutputPorts[0]);
    pGraph->BindControlPropagationPort(pAInputPorts[0], pAOutputPorts[1]);
    pGraph->BindControlPropagationPort(pLeftInputPorts[0], pLeftOutputPorts[1]);
    pGraph->BindControlPropagationPort(pRightInputPorts[1], pRightOutputPorts[1]);
    pGraph->BindControlPort(pRightInputPorts[1], pRightOutputPorts[0], TRUE);
    pGraph->BindControlPort(pRightInputPorts[1], pRightOutputPorts[1], FALSE);

	delete [] pAInputPorts;
	delete [] pAOutputPorts;
	delete [] pLeftInputPorts;
	delete [] pLeftOutputPorts;
	delete [] pRightInputPorts;
	delete [] pRightOutputPorts;

    return pGraph;
}

int run_graph_cuda_controlpropagation_task(	
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

	VECADD_PARAMS params;
	params.N = n;
    float fScalar = 2.0f;
	UINT stride = sizeof(float);
    UINT nPulse = 1;

	int nChannelCount = 0;

	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

    // Create, serialize or deserialize PTask Graph, depending on value of g_serializationMode.
    const char * graphFileName = "controlpropagation.xml";
    Graph * pGraph = initialize_controlpropagation_graph(szfile, szshader, stride, n, graphFileName);
    pGraph->CheckGraphSemantics(TRUE, TRUE);
    if (g_serializationMode == 1)
    {
        pGraph->Serialize(graphFileName);
        printf( "%s succeeded\n", szshader );
        return 0;
    }

    pGraph->Run();

	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float");
	DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale");
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims");

	GraphInputChannel * pTopAxBScaleInput		= (GraphInputChannel*)pGraph->GetChannel("ScalarChannel");
	GraphInputChannel * pLeftAxBScaleInput		= (GraphInputChannel*)pGraph->GetChannel("ScalarChannel_left");
	GraphInputChannel * pRightAxBScaleInput		= (GraphInputChannel*)pGraph->GetChannel("ScalarChannel_right");

    Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pTopAxBScaleInput);
    pTopAxBScaleInput->Push(pScPrm);
    pLeftAxBScaleInput->Push(pScPrm);
    pRightAxBScaleInput->Push(pScPrm);
    pScPrm->Release();

	GraphInputChannel * pTopAInput		= (GraphInputChannel*)pGraph->GetChannel("AInputChannel");
	GraphInputChannel * pTopBInput		= (GraphInputChannel*)pGraph->GetChannel("BInputChannel");
	GraphInputChannel * pLeftBInput		= (GraphInputChannel*)pGraph->GetChannel("BInputChannel_left");
	GraphInputChannel * pRightAInput	= (GraphInputChannel*)pGraph->GetChannel("AInputChannel_right");
	GraphOutputChannel * pLeftOutputA	= (GraphOutputChannel*)pGraph->GetChannel("outputChannel_left_A");
	GraphOutputChannel * pLeftOutputB	= (GraphOutputChannel*)pGraph->GetChannel("outputChannel_left_B");
	GraphOutputChannel * pRightOutputA	= (GraphOutputChannel*)pGraph->GetChannel("outputChannel_Right_A");
	GraphOutputChannel * pRightOutputB	= (GraphOutputChannel*)pGraph->GetChannel("outputChannel_Right_B");

    int nFailures = 0;
    int nIter = 4;
    for(int i=0; i<nIter; i++) {
	    Datablock * pA		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pTopAInput);
	    Datablock * pB		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pTopBInput);
	    Datablock * pLB		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pLeftBInput);
	    Datablock * pRA		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pRightAInput);

        pA->Lock();
        pA->SetRecordCount(params.N);
        if(i == nIter - 1) 
            pA->SetControlSignal(DBCTLC_EOF);
        pA->Unlock();

        pRA->Lock();
        pRA->SetRecordCount(params.N);
        pRA->Unlock();

        pTopAInput->Push(pA);
        pTopBInput->Push(pB);
        pRightAInput->Push(pRA);
        pLeftBInput->Push(pLB);
        pA->Release();
        pB->Release();
        pRA->Release();
        pLB->Release();

        Datablock * pLAout = pLeftOutputA->Pull();
        Datablock * pLBout = pLeftOutputB->Pull();
        if(i == nIter - 1) {
            Datablock * pRAout = pRightOutputA->Peek();
            Datablock * pRBout = pRightOutputB->Pull();
            if(pRAout != NULL) {
                printf("Failed--got non-null from gated right output A!\n");
                nFailures++;
            }
            pLBout->Lock();
            if(!pLBout->HasAnyControlSignal()) {
                printf("Failed--did not get expected EOF on left B output!\n");
                nFailures++;
            }
            pLBout->Unlock();
            pRBout->Lock();
            if(!pRBout->HasAnyControlSignal()) {
                printf("Failed--did not get expected EOF on right B output!\n");
                nFailures++;
            }
            pRBout->Unlock();
            pRBout->Release();
        } else {
            Datablock * pRAout = pRightOutputA->Pull();
            Datablock * pRBout = pRightOutputB->Peek();
            if(pRBout != NULL) {
                printf("Failed--got non-null from gated right output B!\n");
                nFailures++;
            }
            pRAout->Release();
        }
        pLAout->Release();
        pLBout->Release();
    }

    if(nFailures) {
        printf("failure.");
    } else  {
        printf( "%s succeeded", szshader );
    }

    pGraph->Stop();
	pGraph->Teardown();

	delete [] vAVector;
    delete [] vBVector;
    Graph::DestroyGraph(pGraph);
	delete pTimer;

	PTask::Runtime::Terminate();

	return 0;
}
