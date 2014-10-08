//--------------------------------------------------------------------------------------
// File: tee.h
// test gated ports with different initial states
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _TEE_H_
#define _TEE_H_
int run_graph_cuda_tee_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif