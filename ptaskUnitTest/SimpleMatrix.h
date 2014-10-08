//--------------------------------------------------------------------------------------
// File: simplematrix.h
//
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _SIMPLE_MATRIX_H_
#define _SIMPLE_MATRIX_H_
#include <cstdio>

template <class T>
class CSimpleMatrix
{
public:

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 3/11/2014. </remarks>
    ///
    /// <param name="rows"> The rows. </param>
    /// <param name="cols"> The cols. </param>
    ///-------------------------------------------------------------------------------------------------

	CSimpleMatrix(
        int rows, 
        int cols
        ) 
    {
		_r = rows;
		_c = cols;
		_arraysize = rows*cols*sizeof(T);
		int hdrsize = rows*sizeof(T*);
		int nsize = _arraysize + hdrsize;
		_storage = (BYTE*) calloc(1, nsize);
		_rows = (T**)_storage;
		_cells = (T*)&_storage[hdrsize];
		for(int r=0;r<rows;r++) {
			_rows[r] = &_cells[r*cols];
		}
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 3/11/2014. </remarks>
    ///-------------------------------------------------------------------------------------------------

	~CSimpleMatrix(
        void
        ) 
    {
		delete [] _storage;
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Compares two matrices. </summary>
    ///
    /// <remarks>   Crossbac, 3/11/2014. </remarks>
    ///
    /// <param name="pB">               [in,out] If non-null, c simple matrix&lt; t&gt;* to be
    ///                                 compared. </param>
    /// <param name="epsilon">          T to be compared. </param>
    /// <param name="pErrorTolerance">  [in,out] (Optional) If non-null, (Optional) int * to be
    ///                                 compared. </param>
    /// <param name="bVerbose">         (Optional) bool to be compared. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool 
    compare(
        CSimpleMatrix<T>* pB, 
        T epsilon,
        int * pErrorTolerance=NULL,
        bool bVerbose=false
        ) 
    { 
        return compare(this, pB, epsilon, pErrorTolerance, bVerbose); 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Compares objects. </summary>
    ///
    /// <remarks>   Crossbac, 3/11/2014. </remarks>
    ///
    /// <param name="pA">               [in,out] If non-null, c simple matrix&lt; t&gt;* to be
    ///                                 compared. </param>
    /// <param name="pB">               [in,out] If non-null, c simple matrix&lt; t&gt;* to be
    ///                                 compared. </param>
    /// <param name="epsilon">          T to be compared. </param>
    /// <param name="pErrorTolerance">  [in,out] (Optional) If non-null, (Optional) int * to be
    ///                                 compared. </param>
    /// <param name="bVerbose">         (Optional) bool to be compared. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    static bool 
    compare(
        CSimpleMatrix<T>* pA, 
        CSimpleMatrix<T>* pB, 
        T epsilon,
        int * pErrorTolerance=NULL,
        bool bVerbose=false
        ) 
    {
	    if(bVerbose) {
            pA->print("result");
            pB->print("reference");
	    }

	    int rows = pA->rows();
	    int cols = pA->cols();
        if(rows != pB->rows() || cols != pB->cols()) {
            if(bVerbose) 
                std::cout << "different matrix sizes!" 
                          << std::endl;
            return false;
        }

	    int errorCount = 0;
	    for(int r=0; r<rows; r++) {
	        for (int c=0; c<cols; c++) {
			    T rm, cm;
			    rm = pB->v(r, c);
			    cm = pA->v(r, c);
                if(fabs(rm-cm) > epsilon) 
				    errorCount++;
            }
	    }

        if(pErrorTolerance && *pErrorTolerance) {
		    if(errorCount < *pErrorTolerance) {
			    *pErrorTolerance = errorCount;
			    return true;
		    }
		    *pErrorTolerance = errorCount;
		    return false;
	    }
        return errorCount == 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Prints this matrix. </summary>
    ///
    /// <remarks>   Crossbac, 3/11/2014. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    print(
	    char * szLabel=0,
        int nMaxElems=0
	    )
    {
        if(szLabel) std::cout << szLabel << std::endl;
        int nPrintedElems = 0;
	    for(int r=0; r<_r; r++) {
		    for(int c=0; c<_c; c++) {
                T e = v(r, c);
                std::cout << e << ", ";
                ++nPrintedElems;
                if(nMaxElems && nPrintedElems >= nMaxElems) {
                    std::cout << std::endl;
                    return;
                }
		    }
            std::cout << std::endl;
        }
    }

    // simple getters and setters
	T* cells() { return _cells; }
    T& v(int r, int c) { return _rows[r][c]; }
    void setv(int r, int c, T &v) { _rows[r][c] = v; }
	int rows() { return _r; }
	int cols() { return _c; }
    int arraysize() { return _arraysize; }

protected:
	int _r;
	int _c;
	T ** _rows;
	T * _cells;
	unsigned char * _storage;
    int _arraysize;
};
#endif