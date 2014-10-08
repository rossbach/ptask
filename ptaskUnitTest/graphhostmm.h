//--------------------------------------------------------------------------------------
// File: graphhostmm.h
// test chained matrix multiplies *using task graph*
// tasks execute host-implementations of matmul
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_HOST_MATMUL_H_
#define _GRAPH_HOST_MATMUL_H_
int run_graph_host_matmul_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
int run_graph_hostfunc_matmul_task(
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
int run_graph_host_matmul_task_easy(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif