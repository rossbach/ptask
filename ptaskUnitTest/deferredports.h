//--------------------------------------------------------------------------------------
// File: deferredports.h
// test deferred ports
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_DEFERREDPORTS_H_
#define _GRAPH_DEFERREDPORTS_H_
int run_graph_cuda_deferredport_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif