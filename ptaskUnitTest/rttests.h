///-------------------------------------------------------------------------------------------------
// file:	rttests.h
//
// summary:	Declares simple test cases for PTask like init/teardown
///-------------------------------------------------------------------------------------------------

#ifndef __RTTESTS_H__
#define __RTTESTS_H__

int run_init_teardown_test(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_init_teardown_test_with_objects_no_run(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_init_teardown_test_with_run(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_init_teardown_test_with_single_push(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_init_teardown_test_with_multi_push(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_init_teardown_test_with_all_push_no_pull(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_init_teardown_test_pull_and_discard(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_init_teardown_test_pull_and_open(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_multi_init_test(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_multi_init_teardown_test(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_multi_graph_test(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_concurrent_graph_test(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_ultra_concurrent_graph_test(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_extreme_concurrent_graph_test(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_accelerator_disable_test(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

int run_mismatched_template_detection(	
    char * szfile,
    char * szshader,
    int rows,
    int cols,
    int siblings,
    int iterations
    );

#endif
