//--------------------------------------------------------------------------------------
// File: sort.h
// test using bitonic sort
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _PTASK_SORT_H_
#define _PTASK_SORT_H_
#include <vector>

static const int MAXPATH = 4096;
typedef struct _sortcb_t {
    unsigned int iLevel;
    unsigned int iLevelMask;
    unsigned int iWidth;
    unsigned int iHeight;
} SORT_PARAMS;

int run_sort_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int siblings,
	int iterations
	);

unsigned int orand(
	void
	);

void
hostsort(
	std::vector<unsigned int>* data
	);

std::vector<unsigned int> *
randdata( 
	unsigned int nElems 
	);
 
bool
check_sort_result(
	std::vector<unsigned int> * pData,
	std::vector<unsigned int> * pCandidate
	);

typedef struct ptaskdesc_t { 
	int threadid;
	int iterations;
	int rows;
	int cols;
	char szfile[MAXPATH];
	char szSortShader[MAXPATH];
	char szTransposeShader[MAXPATH];
} PSORTDESC, *PPSORTDESC;

void SetSortParams( 
	unsigned int iLevel, 
	unsigned int iLevelMask, 
	unsigned int iWidth, 
	unsigned int iHeight,
	SORT_PARAMS * pParams
	);
#endif