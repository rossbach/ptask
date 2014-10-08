//--------------------------------------------------------------------------------------
// File: gatedports.h
// test gated ports
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_GATEDPORTS_H_
#define _GRAPH_GATEDPORTS_H_
int run_graph_cuda_gatedport_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif