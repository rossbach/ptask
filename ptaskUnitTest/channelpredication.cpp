//--------------------------------------------------------------------------------------
// File: channelpredication.cpp
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
#include "channelpredication.h"
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

Graph * initialize_channelpredication_graph(char * szfile, char * szshader, UINT stride, int n, const char * graphFileName)
{
    Graph * pGraph = new Graph();
    if (2 == g_serializationMode)
    {
        pGraph->Deserialize(graphFileName);
        return pGraph;
    }    

	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
	DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", stride, PTPARM_FLOAT);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
	CompiledKernel * pKernel		    = PTask::Runtime::GetCompiledKernel(szfile, szshader, g_szCompilerOutputBuffer, COBBUFSIZE);

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
    pAOutputA->SetPredicationType(CE_SRC, CGATEFN_CLOSE_ON_EOF);
    pAOutputB->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_EOF);

	delete [] pAInputPorts;
	delete [] pAOutputPorts;

    return pGraph;
}

int run_channelpredication_task(	
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
    float fScalar = 2.0f;
    params.N = n; 
    UINT uiParamsStride = sizeof(VECADD_PARAMS);
	UINT stride = sizeof(float);
    UINT uiVectorBytes = stride * n;
    UINT nPulse = 1;

	int nChannelCount = 0;

	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

    // Create, serialize or deserialize PTask Graph, depending on value of g_serializationMode.
    const char * graphFileName = "channelpredication.xml";
    Graph * pGraph = initialize_channelpredication_graph(szfile, szshader, stride, n, graphFileName);

    pGraph->CheckGraphSemantics(TRUE, TRUE);
    if (g_serializationMode == 1 && false)
    {
        pGraph->Serialize(graphFileName);
        printf( "%s succeeded\n", szshader );
        return 0;
    }

    pGraph->Run();

	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float");
	DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale");
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims");

	GraphInputChannel * pTopAxBScaleInput = (GraphInputChannel*)pGraph->GetChannel("ScalarChannel");
	GraphInputChannel * pTopAInput		  = (GraphInputChannel*)pGraph->GetChannel("AInputChannel");
	GraphInputChannel * pTopBInput		  = (GraphInputChannel*)pGraph->GetChannel("BInputChannel");

    Datablock * pScPrm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pTopAxBScaleInput);
    pTopAxBScaleInput->Push(pScPrm);
    pScPrm->Release();

	GraphOutputChannel * pAOutputA	= (GraphOutputChannel*)pGraph->GetChannel("outputChannel_left_A");
	GraphOutputChannel * pAOutputB	= (GraphOutputChannel*)pGraph->GetChannel("outputChannel_left_B");

    int nFailures = 0;
    int nIter = 1;
    DWORD dwTimeout = 1000;
    for(int i=0; i<nIter; i++) {

        BOOL bLastTime       = (i == nIter - 1);
        Datablock * pA		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, uiVectorBytes, pTopAInput);
	    Datablock * pB		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vBVector, uiVectorBytes, pTopBInput);

        pA->Lock();
        pA->SetRecordCount(params.N);
        if(bLastTime)
            pA->SetControlSignal(DBCTLC_EOF);
        pA->Unlock();

        pTopAInput->Push(pA);
        pTopBInput->Push(pB);
        pA->Release();
        pB->Release();

        Datablock * pLAout = pAOutputA->Pull(dwTimeout);
        Datablock * pLBout = pAOutputB->Pull(dwTimeout);
        if(bLastTime) {
            if(pLAout != NULL) {
                nFailures++;
                printf("got non-null block from A output on final iteration\n");
                pLAout->Release();
            }
            if(pLBout == NULL) {
                nFailures++;
                printf("got null block from B output on final iteration\n");
            } else {
                pLBout->Lock();
                if(!pLBout->IsEOF()) {
                    printf("Failed--did not get expected EOF on B output!");
                    nFailures++;
                }
                pLBout->Unlock();
                pLBout->Release();
            }
        } else {
            if(pLBout != NULL) {
                nFailures++;
                printf("got non-null block from B output on iteration %d\n", i);
                pLBout->Release();
            }
            if(pLAout == NULL) {
                nFailures++;
                printf("got null block from A output on iteration %d\n", i);
            } else {
                pLAout->Lock();
                CONTROLSIGNAL luiCode = pLAout->GetControlSignals();
                if(luiCode != DBCTLC_NONE) {
                    printf("Failed--get unexpected control value on A output:%d\n", luiCode);
                    nFailures++;
                }
                pLAout->Unlock();
                pLAout->Release();
            }
        }
    }

    if(nFailures) {
        printf("failure.\n");
    } else  {
        printf( "%s succeeded\n", szshader );
    }
    pGraph->Stop();
	pGraph->Teardown();

    Graph::DestroyGraph(pGraph);
	delete [] vAVector;
	delete [] vBVector;
	delete pTimer;

	PTask::Runtime::Terminate();

	return 0;
}
