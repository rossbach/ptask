//--------------------------------------------------------------------------------------
// File: matrixtask.cpp
// 
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#include <stdio.h>
#include <crtdbg.h>
#include <Windows.h>
#include "assert.h"
#include "SimpleMatrix.h"
#include "matrixtask.h"

void print_matrix(
	char * label,
	CSimpleMatrix<ELEMTYPE>* v,
    int nMaxElems
	)
{
	printf("%s:\n", label);
	int rows = v->rows();
	int cols = v->cols();
    int nPrintedElems = 0;
	for(int r=0; r<rows; r++) {
		for(int c=0; c<cols; c++) {
            ELEMTYPE e = v->v(r, c);
			// printf("%.2f,", e.f);
			printf("%d,", e.i);
            ++nPrintedElems;
            if(nMaxElems && nPrintedElems >= nMaxElems) {
                printf("\n");
                return;
            }
		}
		printf("\n");
	}
}

extern BOOL g_verbose;

bool 
check_matrix_result(
	CSimpleMatrix<ELEMTYPE>* vCMatrix,
	CSimpleMatrix<ELEMTYPE>* pReferenceResult,
	int * pErrorTolerance
	) 
{
	if(g_verbose) {
		print_matrix("result", vCMatrix);
		print_matrix("reference", pReferenceResult);
	}
	int rows = vCMatrix->rows();
	int cols = vCMatrix->cols();
	int errorCount = 0;
	for(int r=0; r<rows; r++) {
	    for (int c=0; c<cols; c++) {
			ELEMTYPE rm, cm;
			rm = pReferenceResult->v(r, c);
			cm = vCMatrix->v(r, c);
            // note that we require both int and float results
            // to come back wrong--the floating point implementation
            // on the GPU hardware may be imprecise, and because the 
            // numbers we are dealing with are very large, it is difficult
            // to specify a single threshold. What we *should* do is tolerate
            // a fixed error delta in the mantissa alone, but for now, trust
            // the integer-result if the float doesn't match
			if ( (cm.i != rm.i) && (cm.f != rm.f) ) 
				errorCount++;
		}
	}
    if(pErrorTolerance && *pErrorTolerance) {
		if(errorCount < *pErrorTolerance) {
			*pErrorTolerance = errorCount;
			return true;
		}
		*pErrorTolerance = errorCount;
		return false;
	}
    return errorCount == 0;
}

