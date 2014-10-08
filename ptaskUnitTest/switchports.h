//--------------------------------------------------------------------------------------
// File: switchports.h
// test switching ports
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_SWITCHPORTS_H_
#define _GRAPH_SWITCHPORTS_H_
int run_graph_cuda_switchport_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif