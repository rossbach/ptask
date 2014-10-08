//--------------------------------------------------------------------------------------
// File: graphmdmatmul.h
// test chained matrix multiplies *using multiple matrices per data block and metadata*
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_MD_MATMUL_H_
#define _GRAPH_MD_MATMUL_H_
int run_graph_md_matmul_task(
	char * szfile, 
	char * szshader, 
    int object_rows,
    int object_cols,
	int N, 
	int siblings,
	int iterations
	);
#endif