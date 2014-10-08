//--------------------------------------------------------------------------------------
// File: graphsort.h
// test using bitonic sort *and task graph*
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_SORT_H_
#define _GRAPH_SORT_H_
int run_graph_sort_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif