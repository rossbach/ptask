//--------------------------------------------------------------------------------------
// File: matrixtask.h
// 
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _MATRIX_TASK_H_
#define _MATRIX_TASK_H_
#include "elemtype.h"
#include "SimpleMatrix.h"
int run_matrix_task(
	CSimpleMatrix<ELEMTYPE>* vAMatrix,
	CSimpleMatrix<ELEMTYPE>* vBMatrix,
	CSimpleMatrix<ELEMTYPE>** ppResult,
	char * szfile,
	char * szshader,
	bool bRawBuffers
	);
bool check_matrix_result(
	CSimpleMatrix<ELEMTYPE>* vCMatrix,
	CSimpleMatrix<ELEMTYPE>* pReferenceResult,
	int * pErrorTolerance = 0
	);
void print_matrix(
    char * label,
	CSimpleMatrix<ELEMTYPE>* v,
    int nMaxElems=0
	);

#endif