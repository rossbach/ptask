//--------------------------------------------------------------------------------------
// File: select.h
// ptask graph for select
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _PTASK_SELECT_H_
#define _PTASK_SELECT_H_
int run_select_task(
	char * szfile, 
	char * szshader, 
	char * szdata,
	int iterations
	);

struct ENTRY {
    int key_offset;
    int key_length;
    int val_offset;
    int val_length;
};

#endif