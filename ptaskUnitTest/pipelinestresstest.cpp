///-------------------------------------------------------------------------------------------------
// file:	pipelinestresstest.cpp
//
// summary:	Implements the tests that stress the ability of PTask to 
//          fill the graphics driver pipeline. Generally speaking, 
//          any failure to do so is a result of avoidable synchrony,
//          so these tests are designed to tease out places where we 
//          are unintentionally syncing the device/driver. 
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
#include "matmul.h"
#include "pipelinestresstest.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"
#include "Scheduler.h"

using namespace std;
using namespace PTask;

///-------------------------------------------------------------------------------------------------
/// <summary>   Create a new random matrix. </summary>
///
/// <remarks>   Crossbac, 3/11/2014. </remarks>
///
/// <param name="rows">         The rows. </param>
/// <param name="cols">         The cols. </param>
/// <param name="fMinValue">    (Optional) the minimum value. </param>
/// <param name="fMaxValue">    (Optional) the maximum value. </param>
/// <param name="nRandSeed">    (Optional) the random seed. </param>
///
/// <returns>   null if it fails, else a CSimpleMatrix&lt;float&gt;*. </returns>
///-------------------------------------------------------------------------------------------------

CSimpleMatrix<float>* 
RandomMatrix(
	UINT rows,
	UINT cols,
    unsigned int nRandSeed = 0,
    float fMinValue = FLT_MIN,
    float fMaxValue = FLT_MAX
	) 
{
    if(nRandSeed != 0) 
        srand(nRandSeed);
	CSimpleMatrix<float>* pA = new CSimpleMatrix<float>(rows, cols);

	for(UINT r=0; r<rows; r++) {
		for(UINT c=0; c<cols; c++) {
            int nRandValue = rand();            
            float fValue = (float) nRandValue;
            if(fMinValue != FLT_MIN && fMaxValue != FLT_MAX) {
                bool bNegative = rand() % 2 == 0;
                float fRandMax = (float) RAND_MAX;
                fValue = bNegative ? ((fValue/fRandMax)*fMinValue) : ((fValue/fRandMax)*fMaxValue);
                assert(fValue <= fMaxValue && fValue >= fMinValue);
            }
			pA->setv(r, c, fValue);
		}
    }
	return pA;
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Calculates the same result as the HLSL. </summary>
///
/// <remarks>   Crossbac, 3/11/2014. </remarks>
///
/// <param name="pA">       [in,out] If non-null, the p a. </param>
/// <param name="pB">       [in,out] If non-null, the p b. </param>
/// <param name="pParms">   [in,out] If non-null, options for controlling the operation. </param>
///
/// <returns>   null if it fails, else the calculated result. </returns>
///-------------------------------------------------------------------------------------------------

CSimpleMatrix<float>* 
ComputeResultCPU(
    CSimpleMatrix<float>* pA, 
    CSimpleMatrix<float>* pB,
    PSTRESSPARMS * pParms
    )
{
    CSimpleMatrix<float>* pC = new CSimpleMatrix<float>(pParms->g_tex_rows, pParms->g_tex_cols);
    for(int i=0; i<pParms->g_tex_rows; i++) {
        for(int j=0; j<pParms->g_tex_cols; j++) {
            int yidx = i;
            int xidx = j;
	        float t = 0;
            int nCells = 0;
            for(int di=-pParms->g_tex_halfwin; di<pParms->g_tex_halfwin; di++) {
                for(int dj=-pParms->g_tex_halfwin; dj<pParms->g_tex_halfwin; dj++) {		
                    if(yidx+di < 0 || yidx+di >= pParms->g_tex_rows) continue;
                    if(xidx+dj < 0 || xidx+dj >= pParms->g_tex_cols) continue;
                    // int idx = ((yidx+di) * pParms->g_tex_cols) + (xidx+dj);
                    float fA = pA->v(yidx+di, xidx+dj);
                    float fB = pB->v(yidx+di, xidx+dj);
                    float fProd = fA*fB;
                    float fTan = tanf(fProd);
                    float fSin = sinf(fA);
                    float fCos = cosf(fB);
                    float fInc = fTan/(fSin*fCos);
		            t+=fInc;
                    nCells++;
                }
	        }
            float v = (nCells>0?t/nCells:0.0f);
            pC->setv(yidx, xidx, v);
        }
    }
    return pC;
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Calculates the same result as the HLSL. </summary>
///
/// <remarks>   Crossbac, 3/11/2014. </remarks>
///
/// <param name="pA">       [in,out] If non-null, the p a. </param>
/// <param name="pB">       [in,out] If non-null, the p b. </param>
/// <param name="pC">       [in,out] If non-null, the p c. </param>
/// <param name="pParms">   [in,out] If non-null, options for controlling the operation. </param>
///
/// <returns>   null if it fails, else the calculated result. </returns>
///-------------------------------------------------------------------------------------------------

CSimpleMatrix<float>* 
ComputeMultiTaskResultCPU(
    CSimpleMatrix<float>* pA, 
    CSimpleMatrix<float>* pB,
    CSimpleMatrix<float>* pC,
    PSTRESSPARMS * pParms
    )
{
    CSimpleMatrix<float>* pAB = ComputeResultCPU(pA, pB, pParms);
    CSimpleMatrix<float>* pABC = ComputeResultCPU(pC, pAB, pParms);
    delete pAB;
    return pABC;
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Executes the pipestress simple operation. </summary>
///
/// <remarks>   Crossbac, 3/11/2014. </remarks>
///
/// <param name="szfile">       [in,out] If non-null, the szfile. </param>
/// <param name="szshader">     [in,out] If non-null, the szshader. </param>
/// <param name="rows">         The rows. </param>
/// <param name="cols">         The cols. </param>
/// <param name="siblings">     The siblings. </param>
/// <param name="iterations">   The iterations. </param>
/// <param name="bVerify">      (Optional) the verify. </param>
/// <param name="bCopyback">    (Optional) the copyback. </param>
///
/// <returns>   An int. </returns>
///-------------------------------------------------------------------------------------------------

int run_pipestress_simple(	
	char * szfile,
	char * szshader,
	int rows,
	int cols,
	int siblings,
	int iterations,
    bool bVerify,
    bool bCopyback
	) 
{
    bCopyback |= bVerify;
    CONFIGUREPTASKU(UseDirectX, TRUE);
    CONFIGUREPTASKU(UseOpenCL, FALSE);
    CONFIGUREPTASKU(UseCUDA, TRUE);
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    int seed = 2;
    int tgx = 2;
    int tgy = 2;
    int tgz = 1;

	CSimpleMatrix<float>* vAMatrix = RandomMatrix(rows, cols, 2, -1.0f, 1.0f);
	CSimpleMatrix<float>* vBMatrix = RandomMatrix(rows, cols, 0, -1.0f, 1.0f);
	CSimpleMatrix<float>* vCMatrix = new CSimpleMatrix<float>(rows, cols);

    PSTRESSPARMS params;
	params.g_tex_cols = cols;
	params.g_tex_rows = rows;
    params.g_tex_halfwin = min(10, rows-2);
	UINT stride = sizeof(float);
	UINT elements = rows*cols;
	int nChannelCount = 0;
	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	Graph * pGraph = new Graph();

	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbRxC_float", stride, cols, rows, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("pstressparms", sizeof(PSTRESSPARMS), 1, 1, 1);
	CompiledKernel * pKernel		    = CompileWithGeometry(szfile, szshader, tgx, tgy, tgz);

	const UINT uiInputCount = 3;
	const UINT uiOutputCount = 1;
	
	Port ** pInputPorts = new Port*[uiInputCount];
	Port ** pOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pInputPorts[0]		= Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "p1");
	pInputPorts[1]		= Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "p2");
	pInputPorts[2]		= Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "p3");
	pOutputPorts[0]		= Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "p4");


	Task * pTask = pGraph->AddTask(pKernel, 
								   uiInputCount,
								   pInputPorts,
								   uiOutputCount,
								   pOutputPorts,
								   "HighLatencyTask");

	pTask->SetComputeGeometry(rows/tgx, cols/tgy, 1);

	GraphInputChannel * pAInput		= pGraph->AddInputChannel(pInputPorts[0], "AInputChannel");
	GraphInputChannel * pBInput		= pGraph->AddInputChannel(pInputPorts[1], "BInputChannel");
	GraphInputChannel * pParmsInput	= pGraph->AddInputChannel(pInputPorts[2], "ABConstChannel");
	GraphOutputChannel * pOutput	= pGraph->AddOutputChannel(pOutputPorts[0], "outputChannel");

    Datablock * pA		= Runtime::AllocateDatablock(pDataTemplate, vAMatrix->cells(), vAMatrix->arraysize(), pAInput);
	Datablock * pB		= Runtime::AllocateDatablock(pDataTemplate, vBMatrix->cells(), vBMatrix->arraysize(), pBInput);
	Datablock * pABPrm	= Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pParmsInput);

    pGraph->Run();

	double dInitTime         = pTimer->elapsed(false);;
	double dCopyToDeviceTime = 0.0;
	double dComputeTime      = 0.0;
	double dCopyType         = 0.0;
	double dHostStart        = 0.0;
	double dHostEnd          = 0.0;
	double dTeardownStart    = 0.0;
    int nFailedIterations    = 0;

	for(int i=0;i<iterations;i++) {

        double dIterStart = pTimer->elapsed(false);
		pAInput->Push(pA);              // don't release until all iterations complete
		pBInput->Push(pB);              // don't release until all iterations complete
		pParmsInput->Push(pABPrm);      // don't release until all iterations complete

		double dCopyToDeviceEnd = pTimer->elapsed(false);
		dCopyToDeviceTime += dCopyToDeviceEnd - dIterStart;
		Datablock * pResultBlock = pOutput->Pull();
		double dComputeEnd = pTimer->elapsed(false);
        double dIterCompute = dComputeEnd - dCopyToDeviceEnd;
		dComputeTime += dIterCompute;

        if(bVerify || bCopyback || i == iterations-1) {

            pResultBlock->Lock();
		    float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
		    float * pdst = vCMatrix->cells();
		    memcpy(pdst, psrc, elements*stride);
            pResultBlock->Unlock();
		    dCopyType += pTimer->elapsed(false) - dComputeEnd;

            if(bVerify) {

		        printf( "Verifying against CPU result..." );
		        int nErrorTolerance = max((int)(((double)(rows*cols))*0.01), 1);
		        dHostStart = pTimer->elapsed(false);
		        CSimpleMatrix<float>* pCPU = ComputeResultCPU(vAMatrix, vBMatrix, &params); // on CPU
		        dHostEnd = pTimer->elapsed(false) - dHostStart;

		        if(!vCMatrix->compare(pCPU, 0.01f, &nErrorTolerance, (rows<=16)&(cols<=16))) {
			        printf("failure: (%d of %d) erroneous cells\n", nErrorTolerance, rows*cols);
                    nFailedIterations++;
                } 
                delete pCPU;
            }
        }

		pResultBlock->Release();
    }

    if(nFailedIterations == 0) {
        printf( "%s succeeded\n", szshader );
    }

	dTeardownStart = pTimer->elapsed(false);

    pA->Release();
    pB->Release();
    pABPrm->Release();

	pGraph->Stop();
	pGraph->Teardown();

	delete vAMatrix;
	delete vBMatrix;
	delete vCMatrix;
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

///-------------------------------------------------------------------------------------------------
/// <summary>   Executes the pipestress general operation. </summary>
///
/// <remarks>   Crossbac, 3/11/2014. </remarks>
///
/// <param name="szfile">       [in,out] If non-null, the szfile. </param>
/// <param name="szshader">     [in,out] If non-null, the szshader. </param>
/// <param name="rows">         The rows. </param>
/// <param name="cols">         The cols. </param>
/// <param name="siblings">     The siblings. </param>
/// <param name="iterations">   The iterations. </param>
/// <param name="bVerify">      true to verify. </param>
/// <param name="bCopyback">    true to copyback. </param>
///
/// <returns>   An int. </returns>
///-------------------------------------------------------------------------------------------------

int run_pipestress_general(	
	char * szfile,
	char * szshader,
	int rows,
	int cols,
	int siblings,
	int iterations,
    bool bVerify,
    bool bCopyback
	) 
{
    PSTRESSPARMS params;
	params.g_tex_cols = cols;
	params.g_tex_rows = rows;
    params.g_tex_halfwin = min(10, rows-2);
	UINT stride = sizeof(float);
	UINT elements = rows*cols;

    bCopyback |= bVerify;
    CONFIGUREPTASKU(UseDirectX, TRUE);
    CONFIGUREPTASKU(UseOpenCL, FALSE);
    CONFIGUREPTASKU(UseCUDA, TRUE);

	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbRxC_float", stride, cols, rows, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("pstressparms", sizeof(PSTRESSPARMS), 1, 1, 1);

    PTask::Runtime::RequireBlockPool(pParmTemplate, 2);
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    UINT nAccID = 0;
    Accelerator * pABAccelerator = NULL;
    Accelerator * pABCAccelerator = NULL;
    std::set<Accelerator*> vAffinitizedAccelerators;
    if(!PTask::Scheduler::EnumerateEnabledAccelerators(ACCELERATOR_CLASS_DIRECT_X, 
                                                       vAffinitizedAccelerators)) {
        assert(FALSE);
        PTask::Runtime::Terminate();
        return -1;
    }
                                                            
    if(vAffinitizedAccelerators.size() < 2 || !PTask::Runtime::MultiGPUEnvironment()) {
#if 1
        pABAccelerator = *(vAffinitizedAccelerators.begin());
        pABCAccelerator = *(vAffinitizedAccelerators.begin());
#else
        printf("The pipelinestresstestmulti task applies only to multi-GPU environments.\n"
               "This is not such an environment (have you limited GPU count with PTask::Runtime::SetMaximumConcurrency?\n"
               "pipelinestresstestmulti succeeded by default\n");
        PTask::Runtime::Terminate();
        return 0;
#endif
    } else {
        std::set<Accelerator*>::iterator si = vAffinitizedAccelerators.begin();
        pABAccelerator = *si++;
        pABCAccelerator = *si++;
    }

    int seed = 2;
    int tgx = 2;
    int tgy = 2;
    int tgz = 1;

	CSimpleMatrix<float>* vAMatrix = RandomMatrix(rows, cols, 2, -1.0f, 1.0f);
	CSimpleMatrix<float>* vBMatrix = RandomMatrix(rows, cols, 0, -1.0f, 1.0f);
	CSimpleMatrix<float>* vCMatrix = RandomMatrix(rows, cols, 0, -1.0f, 1.0f);
	CSimpleMatrix<float>* vDMatrix = new CSimpleMatrix<float>(rows, cols);

	int nChannelCount = 0;
	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	Graph * pGraph = new Graph();

	CompiledKernel * pKernel		    = CompileWithGeometry(szfile, szshader, tgx, tgy, tgz);

	const UINT uiInputCount = 3;
	const UINT uiOutputCount = 1;
	
	Port ** pABInputPorts = new Port*[uiInputCount];
	Port ** pABOutputPorts = new Port*[uiOutputCount];
	Port ** pABCInputPorts = new Port*[uiInputCount];
	Port ** pABCOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pABInputPorts[0]		= Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++,  "ab_p1");
	pABInputPorts[1]		= Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++,  "ab_p2");
	pABInputPorts[2]		= Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "ab_p3");
	pABOutputPorts[0]		= Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "ab_p4");
	pABCInputPorts[0]		= Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++,  "abc_p1");
	pABCInputPorts[1]		= Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++,  "abc_p2");
	pABCInputPorts[2]		= Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "abc_p3");
	pABCOutputPorts[0]		= Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "abc_p4");

	pABInputPorts[0] ->SetUpstreamChannelPool(1, FALSE, 0);
	pABInputPorts[1] ->SetUpstreamChannelPool(1, FALSE, 0);
	pABCInputPorts[0]->SetUpstreamChannelPool(1, FALSE, 0);

	Task * pABTask = pGraph->AddTask(pKernel, 
								     uiInputCount,
								     pABInputPorts,
								     uiOutputCount,
								     pABOutputPorts,
								     "AB");
	Task * pABCTask = pGraph->AddTask(pKernel, 
								      uiInputCount,
								      pABCInputPorts,
								      uiOutputCount,
								      pABCOutputPorts,
								      "ABC");


	pABTask->SetComputeGeometry(rows/tgx, cols/tgy, 1);
	pABCTask->SetComputeGeometry(rows/tgx, cols/tgy, 1);
    pABTask->SetAffinity(pABAccelerator, AFFINITYTYPE_MANDATORY);
    pABCTask->SetAffinity(pABCAccelerator, AFFINITYTYPE_MANDATORY);

	GraphInputChannel * pAInput		    = pGraph->AddInputChannel(pABInputPorts[0], "AInputChannel");
	GraphInputChannel * pBInput		    = pGraph->AddInputChannel(pABInputPorts[1], "BInputChannel");
	GraphInputChannel * pABParmsInput	= pGraph->AddInputChannel(pABInputPorts[2], "ABConstChannel");
	GraphInputChannel * pCInput		    = pGraph->AddInputChannel(pABCInputPorts[0], "CInputChannel");
    InternalChannel *   pABCInternal    = pGraph->AddInternalChannel(pABOutputPorts[0], pABCInputPorts[1], "ABout_ABCin");
	GraphInputChannel * pABCParmsInput	= pGraph->AddInputChannel(pABCInputPorts[2], "ABCConstChannel");
	GraphOutputChannel * pOutput	    = pGraph->AddOutputChannel(pABCOutputPorts[0], "outputChannel");

	pAInput->SetCapacity(1);
	pBInput->SetCapacity(1);
	pCInput->SetCapacity(1);

    pGraph->Run();

    Datablock * pA		= Runtime::AllocateDatablockAsync(pDataTemplate, vAMatrix->cells(), vAMatrix->arraysize(), pAInput, PT_ACCESS_DEFAULT, DBCTLC_NONE, TRUE);
	Datablock * pB		= Runtime::AllocateDatablockAsync(pDataTemplate, vBMatrix->cells(), vBMatrix->arraysize(), pBInput, PT_ACCESS_DEFAULT, DBCTLC_NONE, TRUE);
	Datablock * pC		= Runtime::AllocateDatablockAsync(pDataTemplate, vCMatrix->cells(), vCMatrix->arraysize(), pCInput, PT_ACCESS_DEFAULT, DBCTLC_NONE, TRUE);
	Datablock * pABPrm	= Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pABParmsInput);  
	Datablock * pABCPrm	= Runtime::AllocateDatablock(pParmTemplate, &params, sizeof(params), pABCParmsInput);  

	double dInitTime         = pTimer->elapsed(false);
	double dCopyToDeviceTime = 0.0;
	double dComputeTime      = 0.0;
	double dCopyType         = 0.0;
	double dHostStart        = 0.0;
	double dHostEnd          = 0.0;
	double dTeardownStart    = 0.0;
    int nFailedIterations    = 0;

	for(int i=0;i<iterations;i++) {

        double dIterStart = pTimer->elapsed(false);
		pAInput->Push(pA);              // don't release until all iterations complete
		pBInput->Push(pB);              // don't release until all iterations complete
        pCInput->Push(pC);              // see above...
		pABParmsInput->Push(pABPrm);    // don't release until all iterations complete
		pABCParmsInput->Push(pABCPrm);  // likewise...

		double dCopyToDeviceEnd = pTimer->elapsed(false);
		dCopyToDeviceTime += dCopyToDeviceEnd - dIterStart;
		Datablock * pResultBlock = pOutput->Pull();
		double dComputeEnd = pTimer->elapsed(false);
        double dIterCompute = dComputeEnd - dCopyToDeviceEnd;
		dComputeTime += dIterCompute;

        if(bVerify || bCopyback || i == iterations-1) {

            // pResultBlock->Lock();
            Accelerator * pHostAcc = MemorySpace::GetAcceleratorFromMemorySpaceId(HOST_MEMORY_SPACE_ID);
            Accelerator * pAcc = pResultBlock->LockForViewSync(TRUE);
		    float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
		    float * pdst = vDMatrix->cells();
		    memcpy(pdst, psrc, elements*stride);
            pResultBlock->UnlockForViewSync(TRUE, pAcc);
		    dCopyType += pTimer->elapsed(false) - dComputeEnd;

            if(bVerify) {

		        printf( "Verifying against CPU result..." );
		        int nErrorTolerance = max((int)(((double)(rows*cols))*0.01), 1);
		        dHostStart = pTimer->elapsed(false);
		        CSimpleMatrix<float>* pCPU = ComputeMultiTaskResultCPU(vAMatrix, vBMatrix, vCMatrix, &params); // on CPU
		        dHostEnd = pTimer->elapsed(false) - dHostStart;

		        if(!vDMatrix->compare(pCPU, 0.01f, &nErrorTolerance, rows <= 16 || cols <= 16)) {
			        printf("failure: (%d of %d) erroneous cells\n", nErrorTolerance, rows*cols);
                    nFailedIterations++;
                } 
                delete pCPU;
            }
        }

		pResultBlock->Release();
    }

    if(nFailedIterations == 0) {
        printf( "%s succeeded\n", szshader );
    }

	dTeardownStart = pTimer->elapsed(false);

    pA->Release();
    pB->Release();
    pC->Release();
    pABPrm->Release();
    pABCPrm->Release();

	pGraph->Stop();
	pGraph->Teardown();

	delete vAMatrix;
	delete vBMatrix;
	delete vCMatrix;
    delete vDMatrix;
	Graph::DestroyGraph(pGraph);
    delete [] pABInputPorts;
	delete [] pABOutputPorts;
    delete [] pABCInputPorts;
	delete [] pABCOutputPorts;
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
