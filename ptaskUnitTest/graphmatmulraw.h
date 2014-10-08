//--------------------------------------------------------------------------------------
// File: graphmatmulraw.h
// test chained matrix multiplies *using task graph*
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_MATMUL_RAW_H_
#define _GRAPH_MATMUL_RAW_H_
int run_graph_matmulraw_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
int run_graph_matmulraw_task_easy(	
	char * szfile,
	char * szshader,
	int rows,
	int cols,
	int siblings,
	int iterations
	);
#endif