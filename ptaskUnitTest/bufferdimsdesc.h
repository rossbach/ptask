///-------------------------------------------------------------------------------------------------
// file:	bufferdimsdesc.h
//
// summary:	Declares the bufferdimsdesc class
///-------------------------------------------------------------------------------------------------

#ifndef _BUFFER_DIMS_DESC_H_
#define _BUFFER_DIMS_DESC_H_
int run_graph_cuda_bufferdimsdesc_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif