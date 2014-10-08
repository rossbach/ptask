//--------------------------------------------------------------------------------------
// File: simplevector.h
//
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _SIMPLE_VECTOR_H_
#define _SIMPLE_VECTOR_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct vectaddparm_t {
	int N;
} VECADD_PARAMS;

static float randfloat() { return rand()/(float)RAND_MAX; }
static float fdelta(float a, float b) { return fabs(a - b); }
static int fgt(float a, float b) { return (a > b); }
static void fprintfn(float a) { printf("%.2f", a); }
static int randint() { return rand(); }
static int idelta(int a, int b) { return abs(a - b); }
static int igt(int a, int b) { return (a > b); }
static void iprintfn(int a) { printf("%d", a); }
static const float FEPSILON = 1e-7f;

template <class T>
class CSimpleVector
{
public:
	CSimpleVector(int n) {
		_n = n;
		_arraysize = _n*sizeof(T);
		_cells = (T*) calloc(1, _arraysize);
	}
	CSimpleVector(int n, T* pdata) {
		_n = n;
		_arraysize = _n*sizeof(T);
		_cells = (T*) calloc(1, _arraysize);
        memcpy(_cells, pdata, n*sizeof(T));
	}
	CSimpleVector(int n, T (*pfn)()) {
		_n = n;
		_arraysize = _n*sizeof(T);
		_cells = (T*) calloc(1, _arraysize);
    	for(int i=0; i<n; i++)  {
		    _cells[i] = (*pfn)();
        }
	}
	~CSimpleVector(void) {
		delete [] _cells;
	}
    int arraysize() { return _arraysize; }
	T* cells() { return _cells; }
	T& v(int r) {
		return _cells[r];
	}
	void setv(int r, T &v) {
		_cells[r] = v;
	}
	int N() { return _n; }
    static CSimpleVector* vadd(
        CSimpleVector* pA,
        CSimpleVector* pB
        )
    {
        CSimpleVector* pC = new CSimpleVector(pA->N());
        int n = pA->N();
	    for(int i=0; i<n; i++)
		    pC->_cells[i] = pA->_cells[i]+pB->_cells[i];
        return pC;
    }
    int equals(CSimpleVector * v, T (*pdelta)(T, T), int (*pgt)(T, T), T epsilon) {
        for(int i=0; i<_n; i++) {
            T delta = (*pdelta)(_cells[i], v->_cells[i]);
            if((*pgt)(delta, epsilon))
		        return 0;
        }
        return 1;
    }
    void print(void (*printfn)(T), int nMaxElems = 0, int colwidth = 8) {
        for(int i=0; i<_n; i++) {
            if(i >= nMaxElems) break;
            if(i % colwidth == (colwidth - 1))
                printf("\n");
            T val = _cells[i];
            (*printfn)(val);
            printf(", ");
        }
        printf("\n");
    }
protected:
	int _n;
    int _arraysize;
	T * _cells;
};
#endif