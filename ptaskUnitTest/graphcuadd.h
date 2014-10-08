//--------------------------------------------------------------------------------------
// File: graphcuadd.h
// test cuda vector addition *using task graph*
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_CUDA_VEC_ADD_H_
#define _GRAPH_CUDA_VEC_ADD_H_
int run_graph_cuda_vecadd_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif