///-------------------------------------------------------------------------------------------------
// file:	ptaskcublas.h
//
// summary:	Declares the ptaskcublas class
///-------------------------------------------------------------------------------------------------

#ifndef _PTASK_CUBLAS_H_
#define _PTASK_CUBLAS_H_
int run_cublas_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
int run_cublas_task_nonsq(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
int run_cublas_task_square(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
int run_cublas_task_no_inout(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
int run_hostfunc_cublas_matmul_task(	
	int rows,
	int cols,
	int siblings,
	int iterations
	);
#endif