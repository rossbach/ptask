//--------------------------------------------------------------------------------------
// File: graphhostmetaports.h
// test meta port use
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_HOSTMETAPORTS_H_
#define _GRAPH_HOSTMETAPORTS_H_
int run_graph_cuda_hostmetaports_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif