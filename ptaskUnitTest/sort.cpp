//--------------------------------------------------------------------------------------
// File: sort.cpp
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#include <stdio.h>
#include <crtdbg.h>
#include "accelerator.h"
#include "assert.h"
#include "shaderparms.h"
#include "sort.h"
#include <vector>
#include <algorithm>
using namespace std;

UINT orand() {
    return (rand() * rand() + rand());
}

vector<UINT> *
randdata( 
	UINT nElems 
	) 
{
    srand( GetTickCount() );
	vector<UINT>* res = new vector<UINT>(nElems);
    std::generate( res->begin(), res->end(), orand );
	return res;
}

void
hostsort(
	vector<UINT>* data
	)
{
    std::sort(data->begin(), data->end());
}

void SetSortParams( 
	UINT iLevel, 
	UINT iLevelMask, 
	UINT iWidth, 
	UINT iHeight,
	SORT_PARAMS * pParams
	)
{
	pParams->iLevel = iLevel;
	pParams->iLevelMask = iLevelMask;
	pParams->iWidth = iWidth;
	pParams->iHeight = iHeight;
}

bool
check_sort_result(
	vector<UINT> * pData,
	vector<UINT> * pCandidate
	)
{
	if(pData->size() != pCandidate->size())
		return false;
	int index = 0;
	vector<UINT>::iterator i, j;
	for(i=pData->begin(),j=pCandidate->begin();
		i!=pData->end() && j!=pCandidate->end(); i++, j++) {
		if(*i != *j)
			return false;
		index++;
	}
	return true;
}


