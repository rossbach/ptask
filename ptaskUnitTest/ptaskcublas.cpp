///-------------------------------------------------------------------------------------------------
// file:	ptaskcublas.cpp
//
// summary:	Implements the ptaskcublas class
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
#include "ptaskcublas.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

#include <cublas_v2.h>
#include "hosttask.h"

using namespace std;
using namespace PTask;

void
reference_sgemm(
    int n, 
    float alpha, 
    const float *A, 
    const float *B,
    float beta, 
    float *C
    )
{
    int i;
    int j;
    int k;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            float prod = 0;

            for (k = 0; k < n; ++k)
            {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

void
reference_sgemm(
    int m,          // A->m x k
    int n,          // B->k x n
    int K,          // C->m x n
    float alpha, 
    const float *A, 
    const float *B, 
    float beta, 
    float *C
    )
{
    int i;
    int j;
    int k;
    for (i = 0; i < m; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            float prod = 0;

            for (k = 0; k < K; ++k)
            {
                prod += A[k * m + i] * B[j * K + k];
            }

            C[j * m + i] = alpha * prod + beta * C[j * m + i];
        }
    }
}

void
reference_sgemm_atransposed(
    int m,          // A->m x k
    int n,          // B->k x n
    int K,          // C->m x n
    float alpha, 
    const float *A, 
    const float *B, 
    float beta, 
    float *C
    )
{
    int i;
    int j;
    int k;
    for (i = 0; i < m; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            float prod = 0;

            for (k = 0; k < K; ++k)
            {
                prod += A[i*K+k] * B[j * K + k];
            }

            C[j * m + i] = alpha * prod + beta * C[j * m + i];
        }
    }
}

void
reference_sgemm_atransposed(
    int m,          // A->m x k
    int n,          // B->k x n
    int K,          // C->m x n
    float alpha, 
    const float *A, 
    const float *B, 
    float beta, 
    float *C,
    int colpitch
    )
{
    int i;
    int j;
    int k;
    assert(colpitch >= K);
    for (i = 0; i < m; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            float prod = 0;

            for (k = 0; k < K; ++k)
            {
                prod += A[i * colpitch + k] * B[j * colpitch + k];
            }

            C[j * colpitch + i] = alpha * prod + beta * C[j * colpitch + i];
        }
    }
}



void print_mat_colmajor(
    float * p,
    int n,
    int m,
    int colpitch
    ) {
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            printf("%.2f, ", p[colpitch*j+i]);
        }
        printf("\n");
    }

}

void print_mat_square(
    float * p,
    int n
    ) {
    print_mat_colmajor(p, n, n, n);
}

typedef struct _sgemm_parms_t {
    int ax;
    int ay;
    int bx;
    int by;
    int cx;
    int cy;
} SGEMMPARMS;

typedef struct _p_sgemm_parms_t {
    int ax;
    int ay;
    int acolpitch;
    int bx;
    int by;
    int bcolpitch;
    int cx;
    int cy;
    int ccolpitch;
} SGEMMPARMSPITCH;

int run_cublas_task_pitched(	
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

    int colpitch = rows & 0x3 ? ((rows&~0x3)+4) : rows;

    float * pA = (float*) calloc(1, cols*colpitch*sizeof(float));  // A[c,r]
    float * pB = (float*) calloc(1, cols*colpitch*sizeof(float));  // B[r,c]
    float * pC = (float*) calloc(1, cols*cols*sizeof(float));  // C[c,c]
    for(int i = 0; i < cols; i++)
    {
        for(int j = 0; j < cols; j++) {
            int idx = j*colpitch+i;
            if(i<rows && j<cols) {
                pA[idx] = rand() / (float)RAND_MAX;
                pB[idx] = rand() / (float)RAND_MAX;
            }
            pC[idx] = 0.0f; 
        }
    }

    SGEMMPARMSPITCH sizeparams; 
    sizeparams.ax = rows;
    sizeparams.ay = cols;
    sizeparams.acolpitch = colpitch;
    sizeparams.bx = rows;
    sizeparams.by = cols;
    sizeparams.bcolpitch = colpitch;
    sizeparams.cx = cols;
    sizeparams.cy = cols;
    sizeparams.ccolpitch = colpitch;

	UINT stride = sizeof(float);
	int nChannelCount = 0;

    Graph * pGraph = new Graph();
	DatablockTemplate * pADataTemplate	= PTask::Runtime::GetDatablockTemplate("dbCxR_float", stride, colpitch, rows, 1);
	DatablockTemplate * pBDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbRxC_float", stride, rows, colpitch, 1);
	DatablockTemplate * pCDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbCxC_float", stride, cols, colpitch, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("sgemmparms", sizeof(SGEMMPARMSPITCH), 1, 1, 1);
    float * pInitialValue = (float*)malloc(rows*colpitch*sizeof(float));
    float * pInitialValueC = (float*)malloc(cols*colpitch*sizeof(float));
    memset(pInitialValue, 0, sizeof(float)*rows*colpitch);
    memset(pInitialValueC, 0, sizeof(float)*cols*colpitch);
    pADataTemplate->SetInitialValue(pInitialValue, stride * rows*colpitch, rows*colpitch);
    pBDataTemplate->SetInitialValue(pInitialValue, stride * rows*colpitch, rows*colpitch);
    pCDataTemplate->SetInitialValue(pInitialValueC, stride * cols*colpitch, cols*colpitch);

	CompiledKernel * pMatmulKernel		= COMPILE_KERNEL(szfile, szshader);

	const UINT uiInputCount = 4;
	const UINT uiOutputCount = 1;
	
	Port ** pAxBInputPorts = new Port*[uiInputCount];
	Port ** pAxBOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAxBInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pADataTemplate, uiUidCounter++, "A", 0);
	pAxBInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pBDataTemplate, uiUidCounter++, "B", 1);
	pAxBInputPorts[2]		= PTask::Runtime::CreatePort(INPUT_PORT, pCDataTemplate, uiUidCounter++, "C(in)", 2, 0);
	pAxBInputPorts[3]		= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "parms", 3);
	pAxBOutputPorts[0]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pCDataTemplate, uiUidCounter++, "C(out)", 2);

    for (int i=0; i < uiInputCount-1;i++)
        pAxBInputPorts[i]->BindDependentAccelerator(ACCELERATOR_CLASS_CUDA, 0);
    pAxBOutputPorts[0]->BindDependentAccelerator(ACCELERATOR_CLASS_CUDA, 0);

	Task * pAxBTask = pGraph->AddTask(pMatmulKernel, 
									  uiInputCount,
									  pAxBInputPorts,
									  uiOutputCount,
									  pAxBOutputPorts,
									  "AxBTask");

	assert(pAxBTask);

    pAxBTask->BindDependentAcceleratorClass(ACCELERATOR_CLASS_CUDA, 1, TRUE);

	// in this case, the iterations are handled 
    // inside the host code. Perhaps we should actually
    // pass the requested geometry to the host code?
    // pAxBTask->SetComputeGeometry(rows, cols, 1);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAxBInputPorts[0], "AInputChannel");
	GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAxBInputPorts[1], "BInputChannel");
	GraphInputChannel * pCInput				= pGraph->AddInputChannel(pAxBInputPorts[2], "CInputChannel");
	GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAxBInputPorts[3], "ABConstChannel");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAxBOutputPorts[0], "outputChannel");

	pGraph->Run();

    Datablock * pblkA   = PTask::Runtime::AllocateDatablock(pADataTemplate, pA, rows*colpitch*sizeof(float), pAInput);
	Datablock * pblkB	= PTask::Runtime::AllocateDatablock(pBDataTemplate, pB, rows*colpitch*sizeof(float), pBInput);
	Datablock * pblkC	= PTask::Runtime::AllocateDatablock(pCDataTemplate, pC, cols*colpitch*sizeof(float), pCInput);
	Datablock * pABPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &sizeparams, sizeof(sizeparams), pAxBParmsInput);

	pAInput->Push(pblkA);
	pBInput->Push(pblkB);
    pCInput->Push(pblkC);
	pAxBParmsInput->Push(pABPrm);
	Datablock * pResultBlock = pOutput->Pull();

    pResultBlock->Lock();
	float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
	float * pdst = (float*) malloc(cols*colpitch*sizeof(float));
	memcpy(pdst, psrc, cols*colpitch*stride);
    pResultBlock->Unlock();

	printf( "Verifying against CPU result..." );
    reference_sgemm_atransposed(cols, cols, rows, 1.0f, pA, pB, 0.0f, pC, colpitch);
    bool bSuccess = true;
    for(int i=0; i<cols; i++) {
        for(int j=0; j<cols; j++) {
            int idx = j*colpitch+i;
	        if(abs(pdst[idx] - pC[idx]) > 0.0001f) {
                if(rows < 16) {
		            printf("failure at index %d,%d\n", i, j);
                    printf("reference:\n");
                    print_mat_colmajor(pC, cols, cols, colpitch);
                    printf("gpu:\n");
                    print_mat_colmajor(pdst, cols, cols, colpitch);
                    bSuccess = false;
                    break;
                } else {
		            printf("failure at index %d: %.2f != %.2f\n", i, pdst[i], pC[i]);
                    bSuccess = false;
                }
            }
        }
        if(!bSuccess && rows < 16) 
            break;
    }
    if(bSuccess) {
        printf( "%s succeeded\n", szshader );
    }

	pResultBlock->Release();
	pGraph->Stop();
	pGraph->Teardown();

    free(pA);
    free(pB);
    free(pC);
    free(pdst);    
    free(pInitialValue);
    free(pInitialValueC);
	Graph::DestroyGraph(pGraph);
	delete [] pAxBInputPorts; 
	delete [] pAxBOutputPorts;

	PTask::Runtime::Terminate();

	return 0;
}

int run_cublas_task(	
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

    float * pA = (float*) malloc(rows*cols*sizeof(float));  // A[c,r]
    float * pB = (float*) malloc(rows*cols*sizeof(float));  // B[r,c]
    float * pC = (float*) malloc(cols*cols*sizeof(float));  // C[c,c]
    for(int i = 0; i < cols*cols; i++)
    {
        if(i<rows*cols) pA[i] = rand() / (float)RAND_MAX;
        if(i<rows*cols) pB[i] = rand() / (float)RAND_MAX;
        pC[i] = 0.0f; 
    }

    SGEMMPARMS sizeparams; 
    sizeparams.ax = rows;
    sizeparams.ay = cols;
    sizeparams.bx = rows;
    sizeparams.by = cols;
    sizeparams.cx = cols;
    sizeparams.cy = cols;

	UINT stride = sizeof(float);
	int nChannelCount = 0;

    Graph * pGraph = new Graph();
	DatablockTemplate * pADataTemplate	= PTask::Runtime::GetDatablockTemplate("dbCxR_float", stride, cols, rows, 1);
	DatablockTemplate * pBDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbRxC_float", stride, rows, cols, 1);
	DatablockTemplate * pCDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbCxC_float", stride, cols, cols, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("sgemmparms", sizeof(SGEMMPARMS), 1, 1, 1);
    float * pInitialValue = (float*)malloc(rows*cols*sizeof(float));
    float * pInitialValueC = (float*)malloc(cols*cols*sizeof(float));
    memset(pInitialValue, 0, sizeof(float)*rows*cols);
    memset(pInitialValueC, 0, sizeof(float)*cols*cols);
    pADataTemplate->SetInitialValue(pInitialValue, stride * rows*cols, rows*cols);
    pBDataTemplate->SetInitialValue(pInitialValue, stride * rows*cols, rows*cols);
    pCDataTemplate->SetInitialValue(pInitialValueC, stride * cols*cols, cols*cols);

	CompiledKernel * pMatmulKernel		= COMPILE_KERNEL(szfile, szshader);

	const UINT uiInputCount = 4;
	const UINT uiOutputCount = 1;
	
	Port ** pAxBInputPorts = new Port*[uiInputCount];
	Port ** pAxBOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAxBInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pADataTemplate, uiUidCounter++, "A", 0);
	pAxBInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pBDataTemplate, uiUidCounter++, "B", 1);
	pAxBInputPorts[2]		= PTask::Runtime::CreatePort(INPUT_PORT, pCDataTemplate, uiUidCounter++, "C(in)", 2, 0);
	pAxBInputPorts[3]		= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "parms", 3);
	pAxBOutputPorts[0]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pCDataTemplate, uiUidCounter++, "C(out)", 2);

    for (int i=0; i < uiInputCount-1;i++)
        pAxBInputPorts[i]->BindDependentAccelerator(ACCELERATOR_CLASS_CUDA, 0);
    pAxBOutputPorts[0]->BindDependentAccelerator(ACCELERATOR_CLASS_CUDA, 0);

	Task * pAxBTask = pGraph->AddTask(pMatmulKernel, 
									  uiInputCount,
									  pAxBInputPorts,
									  uiOutputCount,
									  pAxBOutputPorts,
									  "AxBTask");

	assert(pAxBTask);

    pAxBTask->BindDependentAcceleratorClass(ACCELERATOR_CLASS_CUDA, 1, TRUE);

	// in this case, the iterations are handled 
    // inside the host code. Perhaps we should actually
    // pass the requested geometry to the host code?
    // pAxBTask->SetComputeGeometry(rows, cols, 1);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAxBInputPorts[0], "AInputChannel");
	GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAxBInputPorts[1], "BInputChannel");
	GraphInputChannel * pCInput				= pGraph->AddInputChannel(pAxBInputPorts[2], "CInputChannel");
	GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAxBInputPorts[3], "ABConstChannel");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAxBOutputPorts[0], "outputChannel");

	pGraph->Run();

    Datablock * pblkA   = PTask::Runtime::AllocateDatablock(pADataTemplate, pA, rows*cols*sizeof(float), pAInput);
	Datablock * pblkB	= PTask::Runtime::AllocateDatablock(pBDataTemplate, pB, rows*cols*sizeof(float), pBInput);
	Datablock * pblkC	= PTask::Runtime::AllocateDatablock(pCDataTemplate, pC, cols*cols*sizeof(float), pCInput);
	Datablock * pABPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &sizeparams, sizeof(sizeparams), pAxBParmsInput);

	pAInput->Push(pblkA);
	pBInput->Push(pblkB);
    pCInput->Push(pblkC);
	pAxBParmsInput->Push(pABPrm);
	Datablock * pResultBlock = pOutput->Pull();

    pResultBlock->Lock();
	float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
	float * pdst = (float*) malloc(cols*cols*sizeof(float));
	memcpy(pdst, psrc, cols*cols*stride);
    pResultBlock->Unlock();

	printf( "Verifying against CPU result..." );
    reference_sgemm_atransposed(cols, cols, rows, 1.0f, pA, pB, 0.0f, pC);
    bool bSuccess = true;
    for(int i=0; i<cols*cols; i++) {
	    if(abs(pdst[i] - pC[i]) > 0.0001f) {
            if(rows < 16) {
		        printf("failure at index %d\n", i);
                printf("reference:\n");
                print_mat_colmajor(pC, cols, cols, cols);
                printf("gpu:\n");
                print_mat_colmajor(pdst, cols, cols, cols);
                bSuccess = false;
                break;
            } else {
		        printf("failure at index %d: %.2f != %.2f\n", i, pdst[i], pC[i]);
                bSuccess = false;
            }
        }
    }
    if(bSuccess) {
        printf( "%s succeeded\n", szshader );
    }

	pResultBlock->Release();
	pGraph->Stop();
	pGraph->Teardown();

    free(pA);
    free(pB);
    free(pC);
    free(pdst);    
    free(pInitialValue);
    free(pInitialValueC);
	Graph::DestroyGraph(pGraph);
	delete [] pAxBInputPorts; 
	delete [] pAxBOutputPorts;

	PTask::Runtime::Terminate();

	return 0;
}

/* Host implementation of a CUBLAS sgemm call 
   supports non-square matrices.
   corresponds to the 
    run_hostfunc_cublas_matmul_task()
   test in PTaskUnitTest (defined below).
*/
void
hostfunc_SGemmTrA(LPDEPENDENTCONTEXT depContext)
{
    assert(depContext->nArguments == 4);
    assert(depContext->ppArguments != NULL);

    assert(depContext->ppDatablocks[0]->GetApplicationContext() == (void*)1);
    assert(depContext->ppDatablocks[1]->GetApplicationContext() == (void*)2);
    assert(depContext->ppDatablocks[2]->GetApplicationContext() == (void*)3);
    assert(depContext->ppDatablocks[3]->GetApplicationContext() == (void*)4);
                

    if(depContext->pbIsDependentBinding[0]) {
		cublasHandle_t handle;
		cublasCreate(&handle);
        cudaStream_t stream = (cudaStream_t) depContext->pStreams[0];
		cublasSetStream(handle, stream);
        // in this case the depContext->ppArguments[*]  are device pointers
        SGEMMPARMS * pParms = ((SGEMMPARMS*) depContext->ppArguments[3]);
        float * A = (float *)depContext->ppArguments[0];
        float * B = (float *)depContext->ppArguments[1];
        float * C = (float *)depContext->ppArguments[2];
        int lda = pParms->ax;
        int ldb = pParms->bx;
        int ldc = pParms->cx;
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm (handle, CUBLAS_OP_T, CUBLAS_OP_N, pParms->ay, pParms->by, pParms->bx, &alpha, A, lda, B, ldb, &beta, C, ldc);
		cublasDestroy(handle);
    } else {
        // in this case the depContext->ppArguments[*]  are host pointers                 
        // call host version if it exists
        assert(false);  // no native version
    }
}

int run_hostfunc_cublas_matmul_task(	
	int rows,
	int cols,
	int siblings,
	int iterations
	) 
{
    PTask::Runtime::Initialize();
    // CheckPlatformSupport(szfile, szshader);

    float * pA = (float*) malloc(rows*cols*sizeof(float));  // A[c,r]
    float * pB = (float*) malloc(rows*cols*sizeof(float));  // B[r,c]
    float * pC = (float*) malloc(cols*cols*sizeof(float));  // C[c,c]
    for(int i = 0; i < cols*cols; i++)
    {
        if(i<rows*cols) pA[i] = rand() / (float)RAND_MAX;
        if(i<rows*cols) pB[i] = rand() / (float)RAND_MAX;
        pC[i] = 0.0f; 
    }

    SGEMMPARMS sizeparams; 
    sizeparams.ax = rows;
    sizeparams.ay = cols;
    sizeparams.bx = rows;
    sizeparams.by = cols;
    sizeparams.cx = cols;
    sizeparams.cy = cols;

	UINT stride = sizeof(float);
	int nChannelCount = 0;

    Graph * pGraph = new Graph();
	DatablockTemplate * pADataTemplate	= PTask::Runtime::GetDatablockTemplate("dbCxR_float", stride, cols, rows, 1);
	DatablockTemplate * pBDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbRxC_float", stride, rows, cols, 1);
	DatablockTemplate * pCDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbCxC_float", stride, cols, cols, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("sgemmparms", sizeof(SGEMMPARMS), 1, 1, 1);
    float * pInitialValue = (float*)malloc(rows*cols*sizeof(float));
    float * pInitialValueC = (float*)malloc(cols*cols*sizeof(float));
    memset(pInitialValue, 0, sizeof(float)*rows*cols);
    memset(pInitialValueC, 0, sizeof(float)*cols*cols);
    pADataTemplate->SetInitialValue(pInitialValue, stride * rows*cols, rows*cols);
    pBDataTemplate->SetInitialValue(pInitialValue, stride * rows*cols, rows*cols);
    pCDataTemplate->SetInitialValue(pInitialValueC, stride * cols*cols, cols*cols);

	// CompiledKernel * pMatmulKernel		= COMPILE_KERNEL(szfile, szshader);
    CompiledKernel * pMatmulKernel		= PTask::Runtime::GetHostFunctionCompiledKernel("CublasMatMulHostFunction", (FARPROC)hostfunc_SGemmTrA);

	const UINT uiInputCount = 4;
	const UINT uiOutputCount = 1;
	
	Port ** pAxBInputPorts = new Port*[uiInputCount];
	Port ** pAxBOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAxBInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pADataTemplate, uiUidCounter++, "A", 0);
	pAxBInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pBDataTemplate, uiUidCounter++, "B", 1);
	pAxBInputPorts[2]		= PTask::Runtime::CreatePort(INPUT_PORT, pCDataTemplate, uiUidCounter++, "C(in)", 2, 0);
	pAxBInputPorts[3]		= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "parms", 3);
	pAxBOutputPorts[0]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pCDataTemplate, uiUidCounter++, "C(out)", 2);

    for (int i=0; i < uiInputCount-1;i++)
        pAxBInputPorts[i]->BindDependentAccelerator(ACCELERATOR_CLASS_CUDA, 0);
    pAxBOutputPorts[0]->BindDependentAccelerator(ACCELERATOR_CLASS_CUDA, 0);

	Task * pAxBTask = pGraph->AddTask(pMatmulKernel, 
									  uiInputCount,
									  pAxBInputPorts,
									  uiOutputCount,
									  pAxBOutputPorts,
									  "AxBTask");

	assert(pAxBTask);

    pAxBTask->BindDependentAcceleratorClass(ACCELERATOR_CLASS_CUDA, 1, TRUE);

	// in this case, the iterations are handled 
    // inside the host code. Perhaps we should actually
    // pass the requested geometry to the host code?
    // pAxBTask->SetComputeGeometry(rows, cols, 1);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAxBInputPorts[0], "AInputChannel");
	GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAxBInputPorts[1], "BInputChannel");
	GraphInputChannel * pCInput				= pGraph->AddInputChannel(pAxBInputPorts[2], "CInputChannel");
	GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAxBInputPorts[3], "ABConstChannel");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAxBOutputPorts[0], "outputChannel");

	pGraph->Run();

    Datablock * pblkA   = PTask::Runtime::AllocateDatablock(pADataTemplate, pA, rows*cols*sizeof(float), pAInput);
	Datablock * pblkB	= PTask::Runtime::AllocateDatablock(pBDataTemplate, pB, rows*cols*sizeof(float), pBInput);
	Datablock * pblkC	= PTask::Runtime::AllocateDatablock(pCDataTemplate, pC, cols*cols*sizeof(float), pCInput);
	Datablock * pABPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &sizeparams, sizeof(sizeparams), pAxBParmsInput);

    pblkA->SetApplicationContext((void*)1); // Input A
    pblkB->SetApplicationContext((void*)2); // Input B
    pblkC->SetApplicationContext((void*)3); // In/Out C
    pABPrm->SetApplicationContext((void*)4); // Const params

	pAInput->Push(pblkA);
	pBInput->Push(pblkB);
    pCInput->Push(pblkC);
	pAxBParmsInput->Push(pABPrm);
	Datablock * pResultBlock = pOutput->Pull();

    pResultBlock->Lock();
	float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
	float * pdst = (float*) malloc(cols*cols*sizeof(float));
	memcpy(pdst, psrc, cols*cols*stride);
    // The output block should be the same one as for Input C.
    assert(pResultBlock->GetApplicationContext() == (void*)3);
    pResultBlock->Unlock();

	printf( "Verifying against CPU result..." );
    reference_sgemm_atransposed(cols, cols, rows, 1.0f, pA, pB, 0.0f, pC);
    bool bSuccess = true;
    for(int i=0; i<cols*cols; i++) {
	    if(abs(pdst[i] - pC[i]) > 0.0001f) {
            if(rows < 16) {
		        printf("failure at index %d\n", i);
                printf("reference:\n");
                print_mat_colmajor(pC, cols, cols, cols);
                printf("gpu:\n");
                print_mat_colmajor(pdst, cols, cols, cols);
                bSuccess = false;
                break;
            } else {
		        printf("failure at index %d: %.2f != %.2f\n", i, pdst[i], pC[i]);
                bSuccess = false;
            }
        }
    }
    if(bSuccess) {
        // printf( "%s succeeded\n", szshader );
        printf( "run_hostfunc_cublas_matmul_task succeeded\n");
    }

	pResultBlock->Release();
	pGraph->Stop();
	pGraph->Teardown();

    free(pA);
    free(pB);
    free(pC);
    free(pdst);    
    free(pInitialValue);
    free(pInitialValueC);
	Graph::DestroyGraph(pGraph);
	delete [] pAxBInputPorts; 
	delete [] pAxBOutputPorts;

	PTask::Runtime::Terminate();

	return 0;
}

int run_cublas_task_nonsq(	
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

    float * pA = (float*) malloc(rows*cols*sizeof(float));  // A[c,r]
    float * pB = (float*) malloc(rows*cols*sizeof(float));  // B[r,c]
    float * pC = (float*) malloc(cols*cols*sizeof(float));  // C[c,c]
    for(int i = 0; i < cols*cols; i++)
    {
        if(i<rows*cols) pA[i] = rand() / (float)RAND_MAX;
        if(i<rows*cols) pB[i] = rand() / (float)RAND_MAX;
        pC[i] = 0.0f; 
    }

    SGEMMPARMS sizeparams; 
    sizeparams.ax = cols;
    sizeparams.ay = rows;
    sizeparams.bx = rows;
    sizeparams.by = cols;
    sizeparams.cx = cols;
    sizeparams.cy = cols;

	UINT stride = sizeof(float);
	int nChannelCount = 0;

    Graph * pGraph = new Graph();
	DatablockTemplate * pADataTemplate	= PTask::Runtime::GetDatablockTemplate("dbCxR_float", stride, cols, rows, 1);
	DatablockTemplate * pBDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbRxC_float", stride, rows, cols, 1);
	DatablockTemplate * pCDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbCxC_float", stride, cols, cols, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("sgemmparms", sizeof(SGEMMPARMS), 1, 1, 1);
    float * pInitialValue = (float*)malloc(rows*cols*sizeof(float));
    float * pInitialValueC = (float*)malloc(cols*cols*sizeof(float));
    memset(pInitialValue, 0, sizeof(float)*rows*cols);
    memset(pInitialValueC, 0, sizeof(float)*cols*cols);
    pADataTemplate->SetInitialValue(pInitialValue, stride * rows*cols, rows*cols);
    pBDataTemplate->SetInitialValue(pInitialValue, stride * rows*cols, rows*cols);
    pCDataTemplate->SetInitialValue(pInitialValueC, stride * cols*cols, cols*cols);

	CompiledKernel * pMatmulKernel		= COMPILE_KERNEL(szfile, szshader);

	const UINT uiInputCount = 4;
	const UINT uiOutputCount = 1;
	
	Port ** pAxBInputPorts = new Port*[uiInputCount];
	Port ** pAxBOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAxBInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pADataTemplate, uiUidCounter++, "A", 0);
	pAxBInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pBDataTemplate, uiUidCounter++, "B", 1);
	pAxBInputPorts[2]		= PTask::Runtime::CreatePort(INPUT_PORT, pCDataTemplate, uiUidCounter++, "C(in)", 2, 0);
	pAxBInputPorts[3]		= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "parms", 3);
	pAxBOutputPorts[0]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pCDataTemplate, uiUidCounter++, "C(out)", 2);

    for (int i=0; i < uiInputCount-1;i++)
        pAxBInputPorts[i]->BindDependentAccelerator(ACCELERATOR_CLASS_CUDA, 0);
    pAxBOutputPorts[0]->BindDependentAccelerator(ACCELERATOR_CLASS_CUDA, 0);

	Task * pAxBTask = pGraph->AddTask(pMatmulKernel, 
									  uiInputCount,
									  pAxBInputPorts,
									  uiOutputCount,
									  pAxBOutputPorts,
									  "AxBTask");

	assert(pAxBTask);

    pAxBTask->BindDependentAcceleratorClass(ACCELERATOR_CLASS_CUDA, 1, TRUE);

	// in this case, the iterations are handled 
    // inside the host code. Perhaps we should actually
    // pass the requested geometry to the host code?
    // pAxBTask->SetComputeGeometry(rows, cols, 1);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAxBInputPorts[0], "AInputChannel");
	GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAxBInputPorts[1], "BInputChannel");
	GraphInputChannel * pCInput				= pGraph->AddInputChannel(pAxBInputPorts[2], "CInputChannel");
	GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAxBInputPorts[3], "ABConstChannel");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAxBOutputPorts[0], "outputChannel");

	pGraph->Run();

    Datablock * pblkA   = PTask::Runtime::AllocateDatablock(pADataTemplate, pA, rows*cols*sizeof(float), pAInput);
	Datablock * pblkB	= PTask::Runtime::AllocateDatablock(pBDataTemplate, pB, rows*cols*sizeof(float), pBInput);
	Datablock * pblkC	= PTask::Runtime::AllocateDatablock(pCDataTemplate, pC, cols*cols*sizeof(float), pCInput);
	Datablock * pABPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &sizeparams, sizeof(sizeparams), pAxBParmsInput);

	pAInput->Push(pblkA); pblkA->Release();
	pBInput->Push(pblkB); pblkB->Release();
    pCInput->Push(pblkC); pblkC->Release();
	pAxBParmsInput->Push(pABPrm);
	Datablock * pResultBlock = pOutput->Pull();

    pResultBlock->Lock();
	float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
	float * pdst = (float*) malloc(cols*cols*sizeof(float));
	memcpy(pdst, psrc, cols*cols*stride);
    pResultBlock->Unlock();

	printf( "Verifying against CPU result..." );
    reference_sgemm(cols, cols, rows, 1.0f, pA, pB, 0.0f, pC);
    bool bSuccess = true;
    for(int i=0; i<cols*cols; i++) {
	    if(abs(pdst[i] - pC[i]) > 0.0001f) {
            if(rows < 16) {
		        printf("failure at index %d\n", i);
                printf("reference:\n");
                print_mat_colmajor(pC, cols, cols, cols);
                printf("gpu:\n");
                print_mat_colmajor(pdst, cols, cols, cols);
                bSuccess = false;
                break;
            } else {
		        printf("failure at index %d: %.2f != %.2f\n", i, pdst[i], pC[i]);
                bSuccess = false;
            }
        }
    }
    if(bSuccess) {
        printf( "%s succeeded\n", szshader );
    }

	pResultBlock->Release();
	pGraph->Stop();
	pGraph->Teardown();

    free(pA);
    free(pB);
    free(pC);
    free(pdst);    
    free(pInitialValue);
    free(pInitialValueC);
	Graph::DestroyGraph(pGraph);
	delete [] pAxBInputPorts; 
	delete [] pAxBOutputPorts;

	PTask::Runtime::Terminate();

	return 0;
}


int run_cublas_task_square(	
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
    float * pA = (float*) malloc(rows*cols*sizeof(float));
    float * pB = (float*) malloc(rows*cols*sizeof(float));
    float * pC = (float*) malloc(rows*cols*sizeof(float));
    for(int i = 0; i < rows*cols; i++)
    {
        pA[i] = rand() / (float)RAND_MAX;
        pB[i] = rand() / (float)RAND_MAX;
        pC[i] = 0.0f; // rand() / (float)RAND_MAX;
    }

    int sizeparams = rows;
	UINT stride = sizeof(float);
	UINT elements = rows*cols;
	int nChannelCount = 0;

    Graph * pGraph = new Graph();
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbRxC_float", stride, cols, rows, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("sgemmparms", sizeof(int), 1, 1, 1);
    float * pInitialValue = (float*)malloc(rows*cols*sizeof(float));
    memset(pInitialValue, 0, sizeof(float)*rows*cols);
    pDataTemplate->SetInitialValue(pInitialValue, stride * rows*cols, rows*cols);

	CompiledKernel * pMatmulKernel		= COMPILE_KERNEL(szfile, szshader);

	const UINT uiInputCount = 4;
	const UINT uiOutputCount = 1;
	
	Port ** pAxBInputPorts = new Port*[uiInputCount];
	Port ** pAxBOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pAxBInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "A", 0);
	pAxBInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "B", 1);
	pAxBInputPorts[2]		= PTask::Runtime::CreatePort(INPUT_PORT, pDataTemplate, uiUidCounter++, "C(in)", 2, 0);
	pAxBInputPorts[3]		= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "parms", 3);
	pAxBOutputPorts[0]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pDataTemplate, uiUidCounter++, "C(out)", 2);

    for (int i=0; i < uiInputCount-1;i++)
        pAxBInputPorts[i]->BindDependentAccelerator(ACCELERATOR_CLASS_CUDA, 0);
    pAxBOutputPorts[0]->BindDependentAccelerator(ACCELERATOR_CLASS_CUDA, 0);

	Task * pAxBTask = pGraph->AddTask(pMatmulKernel, 
									  uiInputCount,
									  pAxBInputPorts,
									  uiOutputCount,
									  pAxBOutputPorts,
									  "AxBTask");

	assert(pAxBTask);

    pAxBTask->BindDependentAcceleratorClass(ACCELERATOR_CLASS_CUDA, 1, TRUE);

	// in this case, the iterations are handled 
    // inside the host code. Perhaps we should actually
    // pass the requested geometry to the host code?
    // pAxBTask->SetComputeGeometry(rows, cols, 1);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAxBInputPorts[0], "AInputChannel");
	GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAxBInputPorts[1], "BInputChannel");
	GraphInputChannel * pCInput				= pGraph->AddInputChannel(pAxBInputPorts[2], "CInputChannel");
	GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAxBInputPorts[3], "ABConstChannel");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAxBOutputPorts[0], "outputChannel");

	pGraph->Run();

    Datablock * pblkA   = PTask::Runtime::AllocateDatablock(pDataTemplate, pA, rows*cols*sizeof(float), pAInput);
	Datablock * pblkB	= PTask::Runtime::AllocateDatablock(pDataTemplate, pB, rows*cols*sizeof(float), pBInput);
	Datablock * pblkC	= PTask::Runtime::AllocateDatablock(pDataTemplate, pC, rows*cols*sizeof(float), pCInput);
	Datablock * pABPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &sizeparams, sizeof(sizeparams), pAxBParmsInput);

	pAInput->Push(pblkA);
	pBInput->Push(pblkB);
    pCInput->Push(pblkC);
	pAxBParmsInput->Push(pABPrm);
	Datablock * pResultBlock = pOutput->Pull();

    pResultBlock->Lock();
	float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
	float * pdst = (float*) malloc(rows*cols*sizeof(float));
	memcpy(pdst, psrc, elements*stride);
    pResultBlock->Unlock();

	printf( "Verifying against CPU result..." );
    reference_sgemm(rows, 1.0f, pA, pB, 0.0f, pC);
    bool bSuccess = true;
    for(int i=0; i<rows*cols; i++) {
	    if(abs(pdst[i] - pC[i]) > 0.0001f) {
            if(rows < 16) {
		        printf("failure at index %d\n", i);
                printf("reference:\n");
                print_mat_square(pC, rows);
                printf("gpu:\n");
                print_mat_square(pdst, rows);
                bSuccess = false;
                break;
            } else {
		        printf("failure at index %d: %.2f != %.2f\n", i, pdst[i], pC[i]);
                bSuccess = false;
            }
        }
    }
    if(bSuccess) {
        printf( "%s succeeded\n", szshader );
    }

	pResultBlock->Release();
	pGraph->Stop();
	pGraph->Teardown();

    free(pA);
    free(pB);
    free(pC);
    free(pdst);    
	Graph::DestroyGraph(pGraph);
	delete [] pAxBInputPorts; 
	delete [] pAxBOutputPorts;

	PTask::Runtime::Terminate();

	return 0;
}

int run_cublas_task_no_inout(	
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
    float * pA = (float*) malloc(rows*cols*sizeof(float));
    float * pB = (float*) malloc(rows*cols*sizeof(float));
    float * pC = (float*) malloc(rows*cols*sizeof(float));
    for(int i = 0; i < rows*cols; i++)
    {
        pA[i] = rand() / (float)RAND_MAX;
        pB[i] = rand() / (float)RAND_MAX;
        pC[i] = 0.0f; // rand() / (float)RAND_MAX;
    }

    int sizeparams = rows;
	UINT stride = sizeof(float);
	UINT elements = rows*cols;
	int nChannelCount = 0;

    Graph * pGraph = new Graph();
	DatablockTemplate * pDataTemplate	= PTask::Runtime::GetDatablockTemplate("dbRxC_float", stride, cols, rows, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("sgemmparms", sizeof(int), 1, 1, 1);
    float * pInitialValue = (float*)malloc(rows*cols*sizeof(float));
    memset(pInitialValue, 0, sizeof(float)*rows*cols);
    pDataTemplate->SetInitialValue(pInitialValue, stride * rows*cols, rows*cols);

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

    for (int i=0; i < uiInputCount-1;i++)
        pAxBInputPorts[i]->BindDependentAccelerator(ACCELERATOR_CLASS_CUDA, 0);
    pAxBOutputPorts[0]->BindDependentAccelerator(ACCELERATOR_CLASS_CUDA, 0);

	Task * pAxBTask = pGraph->AddTask(pMatmulKernel, 
									  uiInputCount,
									  pAxBInputPorts,
									  uiOutputCount,
									  pAxBOutputPorts,
									  "AxBTask");

	assert(pAxBTask);

    pAxBTask->BindDependentAcceleratorClass(ACCELERATOR_CLASS_CUDA, 1, TRUE);

	// in this case, the iterations are handled 
    // inside the host code. Perhaps we should actually
    // pass the requested geometry to the host code?
    // pAxBTask->SetComputeGeometry(rows, cols, 1);

	GraphInputChannel * pAInput				= pGraph->AddInputChannel(pAxBInputPorts[0], "AInputChannel");
	GraphInputChannel * pBInput				= pGraph->AddInputChannel(pAxBInputPorts[1], "BInputChannel");
	GraphInputChannel * pAxBParmsInput		= pGraph->AddInputChannel(pAxBInputPorts[2], "ABConstChannel");
	GraphOutputChannel * pOutput			= pGraph->AddOutputChannel(pAxBOutputPorts[0], "outputChannel");

	pGraph->Run();

    Datablock * pblkA   = PTask::Runtime::AllocateDatablock(pDataTemplate, pA, rows*cols*sizeof(float), pAInput);
	Datablock * pblkB	= PTask::Runtime::AllocateDatablock(pDataTemplate, pB, rows*cols*sizeof(float), pBInput);
	Datablock * pABPrm	= PTask::Runtime::AllocateDatablock(pParmTemplate, &sizeparams, sizeof(sizeparams), pAxBParmsInput);

	pAInput->Push(pblkA); pblkA->Release();
	pBInput->Push(pblkB); pblkB->Release();
	pAxBParmsInput->Push(pABPrm); pABPrm->Release();
	Datablock * pResultBlock = pOutput->Pull();

    pResultBlock->Lock();
	float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
	float * pdst = (float*) malloc(rows*cols*sizeof(float));
	memcpy(pdst, psrc, elements*stride);
    pResultBlock->Unlock();

	printf( "Verifying against CPU result..." );
    reference_sgemm(rows, 1.0f, pA, pB, 0.0f, pC);
    bool bSuccess = true;
    for(int i=0; i<rows*cols; i++) {
	    if(abs(pdst[i] - pC[i]) > 0.00001f) {
		    printf("failure at index %d\n", i);
            printf("reference:\n");
            print_mat_square(pC, rows);
            printf("gpu:\n");
            print_mat_square(pdst, rows);
            bSuccess = false;
            break;
        }
    }
    if(bSuccess) {
        printf( "%s succeeded\n", szshader );
    }

	pResultBlock->Release();
	pGraph->Stop();
	pGraph->Teardown();

    free(pA);
    free(pB);
    free(pC);
    free(pdst);    
	Graph::DestroyGraph(pGraph);
	delete [] pAxBInputPorts; 
	delete [] pAxBOutputPorts;

	PTask::Runtime::Terminate();

	return 0;
}
