//--------------------------------------------------------------------------------------
// File: cpfxsum.h
// ptask graph for cuda prefix sum
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _PTASK_CPFXSUM_H_
#define _PTASK_CPFXSUM_H_
int run_cpfxsum_task(
	char * szfile, 
	char * szshader,
	int iterations
	);
#endif