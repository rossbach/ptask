///-------------------------------------------------------------------------------------------------
// file:	pipelinestresstest.h
//
// summary:	Implements the tests that stress the ability of PTask to 
//          fill the graphics driver pipeline. Generally speaking, 
//          any failure to do so is a result of avoidable synchrony,
//          so these tests are designed to tease out places where we 
//          are unintentionally syncing the device/driver. 
///-------------------------------------------------------------------------------------------------

#ifndef __PIPELINESTRESSTEST_H__
#define __PIPELINESTRESSTEST_H__

int run_pipestress_simple(	
	char * szfile,
	char * szshader,
	int rows,
	int cols,
	int siblings,
	int iterations,
    bool bVerify=false,
    bool bCopyback=false
	);

int run_pipestress_general(	
	char * szfile,
	char * szshader,
	int rows,
	int cols,
	int siblings,
	int iterations,
    bool bVerify=false,
    bool bCopyback=false
	);

#include <PshPack1.h>
typedef struct _pstress_params_t {
	int g_tex_cols;
	int g_tex_rows;
	int g_tex_halfwin;
	int g_pad1;
} PSTRESSPARMS;
#include <PopPack.h>

#endif