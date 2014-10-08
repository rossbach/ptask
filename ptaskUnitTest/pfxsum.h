//--------------------------------------------------------------------------------------
// File: select.h
// ptask graph for select
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _PTASK_PFXSUM_H_
#define _PTASK_PFXSUM_H_
int run_pfxsum_task(
	char * szfile, 
	char * szshader,
	int iterations
	);
typedef struct _pfxsum_parms_t {
    int N;
} PFXSUM_PARAMS;
#endif