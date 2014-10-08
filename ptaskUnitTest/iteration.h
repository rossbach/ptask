//--------------------------------------------------------------------------------------
// File: iteration.h
// test iteration primitives
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_ITERATION_H_
#define _GRAPH_ITERATION_H_
int run_simple_iteration_task(
    char * szfile, 
    char * szshader, 
    int rows, 
    int cols,
    int siblings,
    int iterations
    );
int run_general_iteration_task(
    char * szfile, 
    char * szshader, 
    int rows, 
    int cols,
    int siblings,
    int iterations
    );
int run_general_iteration2_task(
    char * szfile, 
    char * szshader, 
    int rows, 
    int cols,
    int siblings,
    int iterations
    );
int run_general_iteration2_with_preloop_task(
    char * szfile, 
    char * szshader, 
    int rows, 
    int cols,
    int siblings,
    int iterations
    );
int run_scoped_iteration_task(
    char * szfile, 
    char * szshader, 
    int rows, 
    int cols,
    int siblings,
    int iterations
    );
#endif