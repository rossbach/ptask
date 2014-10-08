//--------------------------------------------------------------------------------------
// File: shaderparms.h
//
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _SHADER_PARMS_H_
#define _SHADER_PARMS_H_
#include <PshPack1.h>
typedef struct _matadd_params_t {
	unsigned int g_tex_cols;
	unsigned int g_tex_rows;
	unsigned int g_pad0;
	unsigned int g_pad1;
} MATADD_PARAMS;
#include <PopPack.h>
#endif