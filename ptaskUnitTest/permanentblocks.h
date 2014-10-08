///-------------------------------------------------------------------------------------------------
// file:	permanentblocks.h
//
// summary:	Declares the permanentblocks class
///-------------------------------------------------------------------------------------------------

#ifndef _PERMANENT_BLOCKS_H_
#define _PERMANENT_BLOCKS_H_
int run_graph_cuda_permanentblocks_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif