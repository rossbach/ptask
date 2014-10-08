//--------------------------------------------------------------------------------------
// File: matmul.h
// functions for dense matrix multiply
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _MATMUL_H_
#define _MATMUL_H_
#include "SimpleMatrix.h"
struct ELEMTYPE;
void 
configure_raw_matrix(
	unsigned int rows,
	unsigned int cols,
	CSimpleMatrix<ELEMTYPE>** A,
    int v = -1
	);
void 
configure_raw_matrices(
	unsigned int rows,
	unsigned int cols,
	CSimpleMatrix<ELEMTYPE>** A,
	CSimpleMatrix<ELEMTYPE>** B
	);
CSimpleMatrix<ELEMTYPE>*
matmul(
	CSimpleMatrix<ELEMTYPE>* pA,
	CSimpleMatrix<ELEMTYPE>* pB
	);
int run_matrix_mul_task(char * szfile, char * szshader, int rows, int cols);
#endif