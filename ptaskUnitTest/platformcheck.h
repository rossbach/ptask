///-------------------------------------------------------------------------------------------------
// file:	platformcheck.h
//
// summary:	Declares the platformcheck class
///-------------------------------------------------------------------------------------------------

#ifndef __PLATFORM_CHECK_H__
#define __PLATFORM_CHECK_H__

#include "PTaskRuntime.h"

static const int COBBUFSIZE = 4906;
extern char g_szCompilerOutputBuffer[];
int CheckPlatformSupport(char * szFile, char * szShader);
int CheckCompileSuccess(char * szfile, char* szshader, void * pKernel);

inline PTask::CompiledKernel * 
COMPILE_KERNEL(
    char * szfile,
    char * szshader
    )
{
    PTask::CompiledKernel * pKernel = 
        PTask::Runtime::GetCompiledKernel(szfile, 
                                          szshader, 
                                          g_szCompilerOutputBuffer, 
                                          COBBUFSIZE);

    if(CheckCompileSuccess(szfile, szshader, pKernel))
        return pKernel;
    return NULL;
}

inline PTask::CompiledKernel * 
CompileWithGeometry(
    char * szfile,
    char * szshader,
    int tgx,
    int tgy,
    int tgz
    )
{
    PTask::CompiledKernel * pKernel = 
        PTask::Runtime::GetCompiledKernel(szfile, 
                                          szshader, 
                                          g_szCompilerOutputBuffer, 
                                          COBBUFSIZE,
                                          tgx,
                                          tgy,
                                          tgz);

    if(CheckCompileSuccess(szfile, szshader, pKernel))
        return pKernel;
    return NULL;
}

#endif