//--------------------------------------------------------------------------------------
// File: graphinitializerchannels.h
// test channel initializer primitives
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_INITIALIZER_CHANNELS_H_
#define _GRAPH_INITIALIZER_CHANNELS_H_
int run_initializer_channel_task(
    char * szfile, 
    char * szshader, 
    int rows, 
    int cols,
    int siblings,
    int iterations
    );
int run_initializer_channel_task_bof(
    char * szfile, 
    char * szshader, 
    int rows, 
    int cols,
    int siblings,
    int iterations
    );
#endif