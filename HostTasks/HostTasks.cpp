// HostTasks.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "HostTasks.h"
#include <assert.h>
#include "shaderparms.h"
#include "elemtype.h"
#include "SimpleMatrix.h"
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "hosttask.h"


// export variable example, in case we need them.
HOSTTASKS_API int nHostTasks=0;

static CSimpleMatrix<ELEMTYPE>*
matmul(
	CSimpleMatrix<ELEMTYPE>* pA,
	CSimpleMatrix<ELEMTYPE>* pB
	)
{
	int rows = pA->rows();
	int cols = pA->cols();
	CSimpleMatrix<ELEMTYPE>* pC = new CSimpleMatrix<ELEMTYPE>(rows, cols);

	for(int r=0; r<rows; r++) {
		for(int c=0; c<cols; c++) {
			int itot = 0;
			float ftot = 0;
			for(int k=0; k<rows; k++) {
				ELEMTYPE aelem = pA->v(r, k);
				ELEMTYPE belem = pB->v(k, c);
				itot += aelem.i * belem.i;
				ftot += aelem.f * belem.f;
			}
			ELEMTYPE celem = { itot, ftot };
			pC->setv(r, c, celem);
		}
	}
	return pC;
}

// A is assumed to be initialized by an
// initializer port to be uniformly 0. 
// the length of the output is determined
// by a metaport.
// Given an input matrix A of length N'>N
// the ptask runtime code for this will
// allocate an output of size N. Output
// should be uniform scalar of size N
static void
vectoraddN(
    float * A,
    float scalar,
    int N,
    float * B
    )
{
    for(int i=0; i<N; i++) {
        B[i] = A[i] + scalar;
    }
}

extern "C" {
HOSTTASKS_API void __stdcall
htmatmul(
    UINT nArguments,
    void **ppArguments
    )
{
    assert(nArguments == 4);
    assert(ppArguments != NULL);
    MATADD_PARAMS * pParams = (MATADD_PARAMS*) ppArguments[3];
    int nStorageBytes = pParams->g_tex_rows * pParams->g_tex_cols * sizeof(ELEMTYPE);
    CSimpleMatrix<ELEMTYPE> * pA = new CSimpleMatrix<ELEMTYPE>(pParams->g_tex_rows, pParams->g_tex_cols);
    CSimpleMatrix<ELEMTYPE> * pB = new CSimpleMatrix<ELEMTYPE>(pParams->g_tex_rows, pParams->g_tex_cols);
    ELEMTYPE * A = (ELEMTYPE*) ppArguments[0];
    ELEMTYPE * B = (ELEMTYPE*) ppArguments[1];
    ELEMTYPE * C = (ELEMTYPE*) ppArguments[2];
    memcpy(pA->cells(), A, nStorageBytes);    
    memcpy(pB->cells(), B, nStorageBytes);
    CSimpleMatrix<ELEMTYPE> * pC = matmul(pA, pB);
    memcpy(C, pC->cells(), nStorageBytes);
    delete pA;
    delete pB;
    delete pC;
}
}

extern "C" HOSTTASKS_API void __stdcall
htvadd(
    UINT nArguments,
    void **ppArguments
    )
{
    assert(nArguments == 4);
    assert(ppArguments != NULL);
    float * pA = (float*) ppArguments[0];
    float scalar = *((float*)(&ppArguments[1]));
    int N = ((int)ppArguments[2]);
    float * pB = (float*) ppArguments[3];
    vectoraddN(pA, scalar, N, pB);
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Host task to coalesce groups into a datablock
///             with data channel containing: <key-id><grp-item_0,...grp-item_(grp-cnt_0-1)>...
///             and  meta channel containing: <grp-cnt-0><grp-cnt-1>...
///             </summary>
///
/// <remarks>   Crossbac, 2/12/2013. </remarks>
///
/// <typeparam name="typename K">   Type of the typename k. </typeparam>
/// <typeparam name="typename T">   Type of the typename t. </typeparam>
/// <param name="pPrimaryInput">    [in,out] If non-null, the primary input. </param>
/// <param name="nInputElements">   The input elements. </param>
/// <param name="pGroupIDs">        [in,out] If non-null, the group i ds. </param>
/// <param name="nNumGroups">       Number of groups. </param>
/// <param name="pnPerGroupCounts"> [in,out] If non-null, the pn per group counts. </param>
/// <param name="pPrimaryOutput">   [in,out] If non-null, the primary output. </param>
/// <param name="pMetaOutput">      [in,out] If non-null, the meta output. </param>
///-------------------------------------------------------------------------------------------------

template<
    typename K, 
    typename T>
void 
CoalesceGroups(
    T * pPrimaryInput,
    int nInputElements,
    K * pGroupIDs,
    int nNumGroups,
    int * pnPerGroupCounts,
    void * pPrimaryOutput,
    int * pMetaOutput
    )
{
    int i, n;
    void * pP = pPrimaryOutput;
    int * pPGCnt = pMetaOutput;
    for(i=0, n=0; i<nNumGroups && n<nInputElements; i++) {
        int nGroupSize = pnPerGroupCounts[i];
        K * pOutKey = (K*) pP;
        T * pOutGroup = (T*) (pOutKey+1);
        K * pInKey = (K*) &pGroupIDs[n];
        T * pInGroup = (T*) &pPrimaryInput[n];
        *pPGCnt++ = pnPerGroupCounts[i];
        *pOutKey = *pInKey;
        for(int j=0; j<nGroupSize; j++) {
            *pOutGroup++ = *pInGroup++;
            n++;
        }
    }
}

extern "C" HOSTTASKS_API void __stdcall
coalesce_int_float(
    UINT nArguments,
    void **ppArguments
    )
{
    ///             void 
    ///             <m_strKernelName>(
    ///                 __in    T *        pPrimaryInput
    ///                 __in    int        nPrimaryInputElements,
    ///                 __in    int *      pGroupIDs,
    ///                 __in    int        nNumGroups,
    ///                 __in    int *      pnPerGroupCounts,
    ///                 __out   (int,T[])  pPrimaryOutput
    ///                 __out   int *      pnMetaOutput
    ///                 );

    assert(nArguments == 7);
    assert(ppArguments != NULL);
    float * pPrimaryInput = (float*) ppArguments[0];
    int nInputElements = ((int)ppArguments[1]);
    int * pGroupIDs = (int*) ppArguments[2];
    int nNumGroups = ((int)ppArguments[3]);
    int * pnPerGroupCounts = (int*) ppArguments[4];
    void * pPrimaryOutput = (void*) ppArguments[5];
    int * pMetaOutput = (int*) ppArguments[6];

    CoalesceGroups<int, float>(pPrimaryInput, 
                               nInputElements,
                               pGroupIDs,
                               nNumGroups,
                               pnPerGroupCounts,
                               pPrimaryOutput,
                               pMetaOutput);
}


/* Host implementation of a simple version of sgemm */
extern "C" HOSTTASKS_API void __stdcall
simple_sgemm(
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

/* Host implementation of a CUBLAS sgemm call
   supports square matrices only.
   corresponds to the 
    run_cublas_task_no_inout()
    run_cublas_task_square()
   tests in PTaskUnitTest
*/
extern "C" HOSTTASKS_API void __stdcall
SGemmSq(LPDEPENDENTCONTEXT depContext)
{
    assert(depContext->nArguments == 4);
    assert(depContext->ppArguments != NULL);
                
    if(depContext->pbIsDependentBinding[0]) {
		cublasHandle_t handle;
		cublasCreate(&handle);
		// cublasSetStream(handle, GetCurrentStream()) || cublasfailure("cublasSetStream") ;
	    // This function seems limited to 2048 components, but inconsistent. Another meh! moment!
		//CUdevice dev = (CUdevice)depContext->pDependentDevices[0];
		//cudaSetDevice((int)dev);

        // in this case the depContext->ppArguments[*]  are device pointers
        int n = *((int*) depContext->ppArguments[3]);
        float * A = (float *)depContext->ppArguments[0];
        float * B = (float *)depContext->ppArguments[1];
        float * C = (float *)depContext->ppArguments[2];
        int lda = n;
        int ldb = n;
        int ldc = n;
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
		cublasDestroy(handle);
    } else {
        // in this case the depContext->ppArguments[*]  are host pointers                 
        // call host version if it exists
        assert(false);  // no native version
    }
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

/* Host implementation of a CUBLAS sgemm call 
   supports non-square matrices.
   corresponds to the 
    run_cublas_task()
   test in PTaskUnitTest
*/
extern "C" HOSTTASKS_API void __stdcall
SGemm(LPDEPENDENTCONTEXT depContext)
{
    assert(depContext->nArguments == 4);
    assert(depContext->ppArguments != NULL);
                
    if(depContext->pbIsDependentBinding[0]) {
		cublasHandle_t handle;
		cublasCreate(&handle);
        cudaStream_t stream = (cudaStream_t) depContext->pStreams[0];
		cublasSetStream(handle, stream);
		// cublasSetStream(handle, GetCurrentStream()) || cublasfailure("cublasSetStream") ;
	    // This function seems limited to 2048 components, but inconsistent. Another meh! moment!
		//CUdevice dev = (CUdevice)depContext->pDependentDevices[0];
		//cudaSetDevice((int)dev);

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
        cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, pParms->ax, pParms->by, pParms->bx, &alpha, A, lda, B, ldb, &beta, C, ldc);
		cublasDestroy(handle);
    } else {
        // in this case the depContext->ppArguments[*]  are host pointers                 
        // call host version if it exists
        assert(false);  // no native version
    }
}

/* Host implementation of a CUBLAS sgemm call 
   supports non-square matrices.
   corresponds to the 
    run_cublas_task()
   test in PTaskUnitTest
*/
extern "C" HOSTTASKS_API void __stdcall
SGemmTrA(LPDEPENDENTCONTEXT depContext)
{
    assert(depContext->nArguments == 4);
    assert(depContext->ppArguments != NULL);
                
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

/* Host implementation of a CUBLAS sgemm call 
   supports non-square matrices.
   corresponds to the 
    run_cublas_task()
   test in PTaskUnitTest
*/
extern "C" HOSTTASKS_API void __stdcall
SGemmTrAPitch(LPDEPENDENTCONTEXT depContext)
{
    assert(depContext->nArguments == 4);
    assert(depContext->ppArguments != NULL);
                
    if(depContext->pbIsDependentBinding[0]) {
		cublasHandle_t handle;
		cublasCreate(&handle);
        cudaStream_t stream = (cudaStream_t) depContext->pStreams[0];
		cublasSetStream(handle, stream);
	    // This function seems limited to 2048 components, but inconsistent. Another meh! moment!
		//CUdevice dev = (CUdevice)depContext->pDependentDevices[0];
		//cudaSetDevice((int)dev);

        // in this case the depContext->ppArguments[*]  are device pointers
        SGEMMPARMSPITCH * pParms = ((SGEMMPARMSPITCH*) depContext->ppArguments[3]);
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

// This is the constructor of a class that has been exported.
// see HostTasks.h for the class definition
CHostTasks::CHostTasks()
{
	return;
}
