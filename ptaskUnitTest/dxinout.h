//--------------------------------------------------------------------------------------
// File: dxinout.h
// test init port usage
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _DXINOUT_H_
#define _DXINOUT_H_
int run_dxinout_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);
#endif