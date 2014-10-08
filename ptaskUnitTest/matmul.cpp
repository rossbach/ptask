//--------------------------------------------------------------------------------------
// File: matmul.cpp
// Array A * Array B
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#include <stdio.h>
#include <crtdbg.h>
#include "accelerator.h"
#include "assert.h"
#include "matmul.h"
#include "shaderparms.h"
#include "SimpleMatrix.h"
#include "matrixtask.h"

void 
configure_raw_matrix(
	UINT rows,
	UINT cols,
	CSimpleMatrix<ELEMTYPE>** A,
    int v
	) 
{
	CSimpleMatrix<ELEMTYPE>* pA = new CSimpleMatrix<ELEMTYPE>(rows, cols);
	for(UINT r=0; r<rows; r++) {
		for(UINT c=0; c<cols; c++) {
			ELEMTYPE e;
            if (v == -1)
            {
			    e.i = r+c;
			    e.f = (float)(r+c);
            }
            else
            {
                e.i = v;
                e.f = (float) v;
            }
			pA->setv(r, c, e);
		}
    }
	*A = pA;
}

void 
configure_raw_matrices(
	UINT rows,
	UINT cols,
	CSimpleMatrix<ELEMTYPE>** A,
	CSimpleMatrix<ELEMTYPE>** B
	) 
{
	configure_raw_matrix(rows, cols, A);
	configure_raw_matrix(rows, cols, B);
}

CSimpleMatrix<ELEMTYPE>*
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
