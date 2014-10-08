//--------------------------------------------------------------------------------------
// File: controlpropagation.h
// test propagation of control signals on datablocks
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_CONTROLPROPAGATION_H_
#define _GRAPH_CONTROLPROPAGATION_H_
int run_graph_cuda_controlpropagation_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif