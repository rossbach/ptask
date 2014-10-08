//--------------------------------------------------------------------------------------
// File: graphmatmul.h
// test chained matrix multiplies *using task graph*
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_MATMUL_H_
#define _GRAPH_MATMUL_H_
int run_graph_matmul_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
int run_graph_matmul_task_easy(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif