#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <memory.h>

#include "hrperft.h"

//#define JC_DEBUG

double calculate_rate(const char * msg, int iterations, double duration);

// Use own matrix implementation rather than CSimpleMatrix<double> as need column-major form
// to be compatible with BLAS.

double* init_matrix(int M, int N, double V);
double* init_matrix_random(int M, int N, double max);
double* init_matrix_zero(int M, int N);
double* init_matrix_value(int M, int N, double value);
void row_normalize(double* X, int M, int N);
void output_matrix(FILE* outputFile, char* msg, double* X, int M, int N);
void copy_matrix(double* X, double* Y, int M, int N);
void compare_matrices(double* X, char* nameX, double* Y, char* nameY, int M, int N, bool verbose = false);

double pca_cpu(double* X, unsigned int M, unsigned int N, unsigned int K, 
	double* T, double* P, double* R, int iterations, int innerIterations, bool preNorm, bool shortNorm, FILE* debugFile);

// Use own matrix implementation rather than CSimpleMatrix<double> as need column-major form
// to be compatible with BLAS.

// matrix indexing convention:
//   Column-major storage (like Fortran/BLAS). ** 0-based dimensions (unlike Fortran/BLAS) **.
// M rows, N columns. LD = leading dimension. (LD = M unless operating on a sub-matrix.)
#define COLUMN_MAJOR_INDEX(m, n, ld) (((n) * (ld) + (m)))
