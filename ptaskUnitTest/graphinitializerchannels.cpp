//--------------------------------------------------------------------------------------
// File: iteration.cpp
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
#include "graphinitializerchannels.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

using namespace std;
using namespace PTask;

extern float * random_vector(int n);
extern float * uniform_vector(float val, int n);
extern BOOL compare_vectors(float*pA, float*pB, int n);
extern float * vector_scale(float * pA, float scalar, int n);
extern float * vector_scale_repeatedly(float * pA, float scalar, int n, int nIterations);
extern float * vector_scale_and_add_repeatedly(float * pA, float * pB, float scalar, int n, int nIterations);
extern void vector_scale_in_place(float * pA, float scalar, int n);
extern float * vector_add_scalar(float * pA, float scalar, int n);

extern void perform_graph_diagnostics(Graph * pGraph, char * lpszDotFileName);

int run_initializer_channel_task(	
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
    float* vBVector = uniform_vector(0.0f, n);

    VECADD_PARAMS params;
    float fScalar = 2.0f;
    params.N = n; 
    UINT uiParamsStride = sizeof(VECADD_PARAMS);
    UINT stride = sizeof(float);
    UINT nPulse = 1;

    int totalNumScalings = iterations*2;
    float* vAReference = vector_scale_and_add_repeatedly(vAVector, vBVector, fScalar, n, totalNumScalings);

    int nChannelCount = 0;

    CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
    pTimer->reset();

    Graph * pGraph = new Graph();
    UINT uiUidCounter		= 0;
    DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
    DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", stride, PTPARM_FLOAT);
    DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
    DatablockTemplate * pIterTemplate	= PTask::Runtime::GetDatablockTemplate("dbiter", sizeof(int), PTPARM_INT);
    CompiledKernel * pKernel		    = COMPILE_KERNEL(szfile, szshader);

    float * pInitialValue = uniform_vector(0.0, n);
    pDataTemplate->SetInitialValue(pInitialValue, 4 * n, n);

    Port ** pTopInputPorts = NULL;
    Port ** pTopOutputPorts = NULL;
    {
        const UINT uiInputCount = 4;
        const UINT uiOutputCount = 2;

        pTopInputPorts = new Port*[uiInputCount];
        pTopOutputPorts = new Port*[uiOutputCount];

        pTopInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_top", 0, 0);
        pTopInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_top", 1, 1);
        pTopInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_top", 2);
        pTopInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_top", 3);
        pTopOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_top", 0);
        pTopOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B(out)_top", 1);

        Task * pTopTask = pGraph->AddTask(pKernel, 
            uiInputCount,
            pTopInputPorts,
            uiOutputCount,
            pTopOutputPorts,
            "top");

        assert(pTopTask);
        pTopTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pTopTask->SetBlockAndGridSize(gridSize, threadBlockSize);
    }

    Port ** pBottomInputPorts = NULL;
    Port ** pBottomOutputPorts = NULL;
    {
        const UINT uiInputCount = 5;
        const UINT uiOutputCount = 2;

        pBottomInputPorts = new Port*[uiInputCount];
        pBottomOutputPorts = new Port*[uiOutputCount];

        UINT uiUidCounter		= 0;
        pBottomInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_Bottom", 0, 0);
        pBottomInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B_Bottom", 1, 1);
        pBottomInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_Bottom", 2);
        pBottomInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_bottom", 3);
        pBottomInputPorts[4]    = PTask::Runtime::CreatePort(META_PORT, pIterTemplate, uiUidCounter++, "Meta_bottom", 4);
        pBottomOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_Bottom", 0);
        pBottomOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "B(out)_Bottom", 1);

        pBottomInputPorts[4]->SetMetaFunction(MF_GENERAL_ITERATOR);

        Task * pBottomTask = pGraph->AddTask(pKernel, 
            uiInputCount,
            pBottomInputPorts,
            uiOutputCount,
            pBottomOutputPorts,
            "bottom");

        assert(pBottomTask);
        pBottomTask->SetComputeGeometry(n, 1, 1);
        PTASKDIM3 threadBlockSize(256, 1, 1);
        PTASKDIM3 gridSize(static_cast<int>(ceil(n/256.0)), 1, 1);
        pBottomTask->SetBlockAndGridSize(gridSize, threadBlockSize);
    }

    // Set up graph input channels.
    GraphInputChannel * pAInputChannel				= pGraph->AddInputChannel(pTopInputPorts[0], "AInputChannel", FALSE); // A switch channel
    InitializerChannel * pBInitChannel				= pGraph->AddInitializerChannel(pTopInputPorts[1], "BInitChannel", FALSE); // A switch channel
    GraphInputChannel * pScalarInputChannelTop		= pGraph->AddInputChannel(pTopInputPorts[2], "ScalarInputChannelTop");
    GraphInputChannel * pScalarInputChannelBottom   = pGraph->AddInputChannel(pBottomInputPorts[2], "ScalarInputChannelBottom");
    GraphInputChannel * pIterationCount             = pGraph->AddInputChannel(pBottomInputPorts[4], "IterationChannel");

    // Bind descriptor ports.
    // For all the tasks, port 0 (A) drives the value of port 3 (N).
    pGraph->BindDescriptorPort(pTopInputPorts[0], pTopInputPorts[3]);
    pGraph->BindDescriptorPort(pBottomInputPorts[0], pBottomInputPorts[3]);

    // Set up internal channels (including back channel for A from Bottom to Top task) and output channel.
    InternalChannel * pBackChannelA                 = pGraph->AddInternalChannel(pBottomOutputPorts[0], pTopInputPorts[0], "back-channelA", TRUE);
    InternalChannel * pBackChannelB                 = pGraph->AddInternalChannel(pBottomOutputPorts[1], pTopInputPorts[1], "back-channelB", TRUE);
    GraphOutputChannel * pOutputChannelA		    = pGraph->AddOutputChannel(pBottomOutputPorts[0], "outputChannel_left_A");
    GraphOutputChannel * pOutputChannelB		    = pGraph->AddOutputChannel(pBottomOutputPorts[1], "outputChannel_left_B");
    InternalChannel * pTopToBottomA                  = pGraph->AddInternalChannel(pTopOutputPorts[0], pBottomInputPorts[0], "A_TopToBottom");
    InternalChannel * pTopToBottomB                  = pGraph->AddInternalChannel(pTopOutputPorts[1], pBottomInputPorts[1], "B_TopToBottom");

    // Configure looping.
    // ----------------
    // XXXX: cjr: wrong port indeces! The META port is at index 4!
    // In the previous test cases, it was at index 3, but this test
    // adds an additional in/out parameter, forcing the metaport index up!
    // Unfortunatey, no control propagation route information is set up for 
    // the scalar "N" parameter (which is at index 3), so when the iteration ends, 
    // those blocks are marked with ENDITERATION, but that control information never
    // gets to the output ports. Hence, the hang. 
    // -------------------------------------------------------------------------------
    // pGraph->BindControlPropagationPort(pBottomInputPorts[3], pBottomOutputPorts[0]);
    // pGraph->BindControlPropagationPort(pBottomInputPorts[3], pBottomOutputPorts[1]);
    pGraph->BindControlPropagationPort(pBottomInputPorts[4], pBottomOutputPorts[0]);
    pGraph->BindControlPropagationPort(pBottomInputPorts[4], pBottomOutputPorts[1]);
    pGraph->BindControlPropagationChannel(pBottomInputPorts[4], pBInitChannel);
    pGraph->BindControlPropagationChannel(pBottomInputPorts[4], pAInputChannel);
    pAInputChannel->SetPropagatedControlSignal(DBCTLC_ENDITERATION);
    pAInputChannel->SetPredicationType(CE_DST, CGATEFN_OPEN_ON_ENDITERATION);
    pBInitChannel->SetPropagatedControlSignal(DBCTLC_ENDITERATION);
    pBInitChannel->SetPredicationType(CE_DST, CGATEFN_OPEN_ON_ENDITERATION);
    pBackChannelA->SetPredicationType(CE_SRC, CGATEFN_CLOSE_ON_ENDITERATION);
    pBackChannelB->SetPredicationType(CE_SRC, CGATEFN_CLOSE_ON_ENDITERATION);
    pOutputChannelA->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_ENDITERATION);
    pOutputChannelB->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_ENDITERATION);

    perform_graph_diagnostics(pGraph, "InitializerChannels.dot");

    pGraph->Run();

    Datablock * pScalarParamTop		 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pScalarInputChannelTop);
    Datablock * pScalarParamBottom	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pScalarInputChannelBottom);

    Datablock * pA		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pAInputChannel);
    Datablock * pIters  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), pIterationCount);
    Datablock * pA2		= PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, stride*n, pAInputChannel);
    Datablock * pIters2  = PTask::Runtime::AllocateDatablock(pIterTemplate, &iterations, sizeof(iterations), pIterationCount);

    pA->Lock();
    pA->SetRecordCount(params.N);
    pA->Unlock();

    pAInputChannel->Push(pA);
    pIterationCount->Push(pIters);
    pA->Release();
    pIters->Release();

    pScalarInputChannelTop->Push(pScalarParamTop);
    pScalarInputChannelBottom->Push(pScalarParamBottom);
    pScalarParamTop->Release();
    pScalarParamBottom->Release();

    pAInputChannel->Push(pA2);
    pIterationCount->Push(pIters2);
    pA2->Release();
    pIters2->Release();

    Datablock * pLAout = pOutputChannelA->Pull();
    Datablock * pLBout = pOutputChannelB->Pull();
    Datablock * pLAout2 = pOutputChannelA->Pull();
    Datablock * pLBout2 = pOutputChannelB->Pull();

    int nFailures = 0;
    float * psrcA = NULL;
    pLAout->Lock();
    psrcA = (float*) pLAout->GetDataPointer(FALSE);
    printf("comparing A...\n");
    if(!compare_vectors(vAReference, psrcA, n))
        nFailures++;
    pLAout->Unlock();    
    float * psrcB = NULL;
    pLBout->Lock();
    psrcB = (float*) pLBout->GetDataPointer(FALSE);
    printf("comparing B...\n");
    if(!compare_vectors(vBVector, psrcB, n))
        nFailures++;
    pLBout->Unlock();    

    if(nFailures) {
        printf("failure.\n");
    } else  {
        printf( "%s succeeded\n", szshader );
    }

    pLAout->Release();
    pLBout->Release();
    pLAout2->Release();
    pLBout2->Release();
    delete [] vAVector;
    delete [] vBVector;
    delete [] vAReference;
    delete [] pInitialValue;



    pGraph->Stop();
    pGraph->Teardown();

	Graph::DestroyGraph(pGraph);
    delete [] pTopInputPorts;
    delete [] pTopOutputPorts;
    delete [] pBottomInputPorts;
    delete [] pBottomOutputPorts;
    delete pTimer;

    PTask::Runtime::Terminate();

    return 0;
}


int run_initializer_channel_task_bof(	
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

	Graph * pGraph = new Graph();
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float", stride, n, 1, 1);
	DatablockTemplate * pBDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbV1_float_b", stride, n, 1, 1);
	DatablockTemplate * pScaleTemplate	= PTask::Runtime::GetDatablockTemplate("vscale", stride, PTPARM_FLOAT);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("vecdims", sizeof(VECADD_PARAMS), PTPARM_INT);
	CompiledKernel * pKernel		    = PTask::Runtime::GetCompiledKernel(szfile, szshader, g_szCompilerOutputBuffer, COBBUFSIZE);
    pBDataTemplate->SetInitialValue(vBVector, n*sizeof(float), n);

    CheckCompileSuccess(szfile, szshader, pKernel);

	const UINT uiInputCount = 4;
	const UINT uiOutputCount = 2;
	
	Port ** pAInputPorts = new Port*[uiInputCount];
	Port ** pAOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAInputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A_top", 0, 0);
	pAInputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, pBDataTemplate, uiUidCounter++, "B_top", 1, 1);
	pAInputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, pScaleTemplate, uiUidCounter++, "scalar_top", 2);
	pAInputPorts[3]	= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "N_top", 3);
	pAOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "A(out)_top", 0);
	pAOutputPorts[1]	= PTask::Runtime::CreatePort(OUTPUT_PORT, pBDataTemplate, uiUidCounter++, "B(out)_top", 1);

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
	InitializerChannel * pTopBInit              = pGraph->AddInitializerChannel(pAInputPorts[1], "BInitChannel", FALSE);
	GraphInputChannel * pTopAxBScaleInput		= pGraph->AddInputChannel(pAInputPorts[2], "ScalarChannel");
    pGraph->BindDescriptorPort(pAInputPorts[0], pAInputPorts[3]);
    pGraph->BindControlPropagationPort(pAInputPorts[0], pAOutputPorts[0]);
    pGraph->BindControlPropagationChannel(pAInputPorts[0], pTopBInit);
    GraphOutputChannel * pAOutputA			    = pGraph->AddOutputChannel(pAOutputPorts[0], "outputChannel_left_A");
    InternalChannel * pBackChannelB			    = pGraph->AddInternalChannel(pAOutputPorts[1], pAInputPorts[1], "back_channel_B", TRUE);
    pAOutputA->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_EOF);
    pTopBInit->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_BOF);

    pGraph->Run();

    Datablock * pScParm	 = PTask::Runtime::AllocateDatablock(pScaleTemplate, &fScalar, sizeof(fScalar), pTopAxBScaleInput);
    pTopAxBScaleInput->Push(pScParm);
    pScParm->Release();

    int nFailures = 0;
    int nIter = iterations;
    DWORD dwTimeout = 1000;
    for(int i=0; i<nIter; i++) {

        BOOL bLastTime       = (i == nIter - 1);
        Datablock * pA		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, uiVectorBytes, pTopAInput);
	    // Datablock * pB		 = PTask::Runtime::AllocateDatablock(pDataTemplate, vAVector, uiVectorBytes, pTopBInput);

        pA->Lock();
        pA->SetRecordCount(params.N);
        if(i==0)
            pA->SetControlSignal(DBCTLC_BOF);
        if(bLastTime)
            pA->SetControlSignal(DBCTLC_EOF);
        pA->Unlock();


        pTopAInput->Push(pA);
        pA->Release();
    }

    Datablock * pLAout = pAOutputA->Pull();
    if(pLAout == NULL) {
        nFailures++;
        printf("got null block from A output\n");
    } else {
        pLAout->Lock();
        CONTROLSIGNAL luiCode = pLAout->GetControlSignals();
        if(!TESTSIGNAL(luiCode, DBCTLC_EOF)) {
            printf("Failed--get unexpected control value on A output:%d\n", luiCode);
            nFailures++;
        }
        float * pdst = (float*)malloc(n*sizeof(float));
        float * psrc = (float*)pLAout->GetDataPointer(FALSE);
        memcpy(pdst, psrc, n*sizeof(float));
        pLAout->Unlock();
        pLAout->Release();

        // output should be vAVector * scale + [vAVector + nIter*scale];
        float * pAScaled = vector_scale(vAVector, fScalar, n);
        float * pBSum = vector_add_scalar(vBVector, fScalar*nIter, n);

        for(int i=0; i<n; i++) {
            float fRef = pAScaled[i]+pBSum[i];
            float fGPU = pdst[i];
            if(fabs(fRef-fGPU) > 0.0001) {
                nFailures++;
                printf("failed at idx %d: ref[%d]=%.4f, gpu[%d]=%0.4f\n",
                       i, i, fRef, i, fGPU);
            }
        }

        free(pdst);
        free(pAScaled);
        free(pBSum);
    }

    if(nFailures) {
        printf("failure.\n");
    } else  {
        printf( "%s succeeded\n", szshader );
    }

    pGraph->Stop();
	pGraph->Teardown();

	delete [] vAVector;
    delete [] vBVector;
	Graph::DestroyGraph(pGraph);
	delete [] pAInputPorts;
	delete [] pAOutputPorts;

	delete pTimer;

	PTask::Runtime::Terminate();

	return 0;
}
