//--------------------------------------------------------------------------------------
// File: channelpredication.h
// test channel predication
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_CHANNELPREDICATION_H_
#define _GRAPH_CHANNELPREDICATION_H_
int run_channelpredication_task(
    char * szfile, 
    char * szshader, 
    int rows, 
    int cols,
    int siblings,
    int iterations
    );
#endif