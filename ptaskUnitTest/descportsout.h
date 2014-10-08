///-------------------------------------------------------------------------------------------------
// file:	descportsout.h
//
// summary:	Declares the descportsout class
///-------------------------------------------------------------------------------------------------

#ifndef _GRAPH_DESCPORTSOUT_H_
#define _GRAPH_DESCPORTSOUT_H_
int run_graph_cuda_descportsout_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif