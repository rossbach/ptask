#include "pca_cpu.h"

double calculate_rate(const char * msg, int iterations, double duration)
{
    double rate = (double)(iterations)/duration*1000.0; // iteration per second
    return rate;
}
double* init_matrix_zero(int M, int N)
{
    double* X = (double*)malloc(M*N * sizeof(X[0]));
    if(X == 0)
    {
        fprintf (stderr, "! host memory allocation error: X\n");
        return NULL;
    }
    memset(X, 0, M*N*sizeof(double));
    return X;
}

double* init_matrix_value(int M, int N, double value)
{
    double* X = (double*)malloc(M*N * sizeof(X[0]));
    if(X == 0)
    {
        fprintf (stderr, "! host memory allocation error: X\n");
        return NULL;
    }
    int m;
    for(m = 0; m < M; m++)
    {
        int n;
        for(n = 0; n < N; n++)
        {
            X[COLUMN_MAJOR_INDEX(m, n, M)] = value;
        }
    }
    return X;
}

double* init_matrix_random(int M, int N, double max)
{
    double* X = (double*)malloc(M*N * sizeof(X[0]));
    if(X == 0)
    {
        fprintf (stderr, "! host memory allocation error: X\n");
        return NULL;
    }
    int m;
    for(m = 0; m < M; m++)
    {
        int n;
        for(n = 0; n < N; n++)
        {
            X[COLUMN_MAJOR_INDEX(m, n, M)] = rand() / (double)RAND_MAX * max;
        }
    }
    return X;
}

void row_normalize(double* X, int M, int N)
{
    // Set vector U to X[col 0]
    double* U = new double[M];
    for (int m=0; m<M; m++)
    {
        U[m] = X[COLUMN_MAJOR_INDEX(m, 0, M)];
    }
    // Add columns 1 to N-1 to U
    // (iterate on column first since column major format has each column sequential in memory)
    for (int n=1; n<N; n++)
    {
        for (int m=0; m<M; m++)
        {
            U[m] += X[COLUMN_MAJOR_INDEX(m, n, M)];
        }
    }
    // U -> U/N
    for (int m=0; m<M; m++)
    {
        U[m] /= N;
    }
    // Subtract U from all columns of X.
    // (iterate on column first since column major format has each column sequential in memory)
    for (int n=0; n<N; n++)
    {
        for (int m=0; m<M; m++)
        {
            X[COLUMN_MAJOR_INDEX(m, n, M)] -= U[m];
        }
    }
    delete U;
}

void output_matrix(FILE* outputFile, char* msg, double* X, int M, int N)
{
    fprintf(outputFile, "%s (%dx%d):\n", msg, M, N);
    int m;
    for(m = 0; m < M; m++)
    {
        int n;
        for(n = 0; n < N; n++)
        {
            fprintf(outputFile, "%6.2f ", X[COLUMN_MAJOR_INDEX(m, n, M)]);
        }
        fprintf(outputFile, "\n");
    }
}

void copy_matrix(double* X, double* Y, int M, int N)
{
    memcpy(Y, X, M*N * sizeof(X[0]));
}

void compare_matrices(double* X, char* nameX, double* Y, char* nameY, int M, int N, bool verbose)
{
    // max error
    // All outputs show as different if 1.0e-6 is used.
    //	double er = 1.0e-6;
    double er = 1.0e-5;
    double maxDiff = 0.0;

    int m;
    for(m = 0; m < M; m++)
    {
        int n;
        for(n = 0; n < N; n++)
        {
            double diff = fabs(X[COLUMN_MAJOR_INDEX(m, n, M)] - Y[COLUMN_MAJOR_INDEX(m, n, M)]);
            if (diff > maxDiff)
                maxDiff = diff;
            if (diff > er)
            {
                printf("XXX Matrices %s and %s differ at row %d, col %d (X=%f, Y=%f) diff=%f XXX\n", 
                    nameX, nameY, m, n, X[COLUMN_MAJOR_INDEX(m, n, M)], Y[COLUMN_MAJOR_INDEX(m, n, M)], diff);
                //output_matrix(nameX, X, M, N);
                //output_matrix(nameY, Y, M, N);
                //exit(1);
                return;
            };
        }
    }
    if (verbose)
        printf("*** Matrices %s and %s match (max diff = %f, threshold = %f) ***\n", nameX, nameY, maxDiff, er);
}

// Naive, single-threaded implementations of BLAS routines used for PCA.
void cpublasDcopy(int n, const double *x, int incx, double *y, int incy)
{
    for (int col=0; col<n; col++)
    {
        y[col*incy] = x[col*incx];
    }
}

void cpublasDaxpy(int n, double alpha, const double *x, int incx, double *y, int incy)
{
    for (int col=0; col<n; col++)
    {
        y[col*incy] += alpha * x[col*incx];
    }
}

void cpublasDgemv(bool transpose, int m, int n, double alpha,
    const double *A, int lda, const double *x,
    int incx, double beta, double *y, int incy)
{
    // Num rows depends on whether are transposing or not.
    int numRows = (transpose == true) ? n : m;

    for (int row=0; row<numRows; row++)
    {
        // Compute one row of: y = alpha * op(A) * x + beta * y
        // beta * y
        double result = beta * y[row * incy];
        // alpha * op(A) * x
        if (!transpose)
        {
            double A_x = A[row] * x[0]; // Init to value from A[row,0]*x[0]
            for (int i=1; i<n; i++)
            {
                A_x += A[row + (i * lda)] * x[i * incx]; // Add value from A[row, i]*x[i]
            }
            result += (alpha * A_x);
        } else {		
            // To work with A', just flip row and column from non-transpose case.
            double A_x = A[row * lda] * x[0]; // Init to value from A[0, row]*x[0]
            //if (debug) printf("Row %d: Initial value A[0, %d]*x[0] = %f * %f = %f\n", row, row, A[row * lda], x[0], A_x);
            for (int i=1; i<m; i++) // Column 1 to m rather than 1 to n, since transposed.
            {	
                double val = A[i + (row * lda)] * x[i * incx]; // Add value from A[i, row]*x[i] 
                A_x += val;
                //if (debug) printf("Row %d: Add value A[%d, %d]*x[%d] = %f * %f = %f\n", row, i, row, i, A[i + (row * lda)], x[i * incx], val);
            }
            result += (alpha * A_x);
        }

        y[row * incy] = result;
    }
}

void cpublasDscal(int n, double alpha, double *x, int incx)
{
    for (int col=0; col<n; col++)
    {
        x[col*incx] *= alpha;
    }
}

void cpublasDger(int m, int n, double alpha, const double *x,
    int incx, const double *y, int incy, double *A, int lda)
{
/*
            printf("\n\n>>> CPU Dger: alpha = %f\n", alpha);
            printf("Matrix before:\n");
            for(int r=0; r<m; r++)
            {
                for(int c=0; c<n; c++)
                {
                    printf("%6.2f\t", A[(c * lda) + r]);
                }
                printf("\n");
            }
*/
    for (int row=0; row<m; row++)
    {
        for (int col=0; col<n; col++)
        {
            // A = alpha * x * y' + A
            //
            // x * y' is the vector outer product of x and y.
            // We are only interested in the first m elements of x and 
            // the first n elements of y, resulting in an m by n matrix, which 
            // can then be added to A.

            // Since both the outer vector product and matrix addition involve no
            // cross-over between elements of the m by n matrix, can do the whole
            // operation in parallel.

            A[(col * lda) + row] += alpha * x[row * incx] * y[col * incy];
        }
    }
/*
            printf("Matrix after:\n");
            for(int r=0; r<m; r++)
            {
                for(int c=0; c<n; c++)
                {
                    printf("%6.2f\t", A[(c * lda) + r]);
                }
                printf("\n");
            }
*/
}

double cpublasDnrm2(int n, const double *x, int incx)
{
    double sum = 0.0;
    for(int i=0; i<n; i++)
    {
        double val = x[i * incx];
        sum += (val * val);
        //printf("val %f, sum %f\n", val, sum);
    }
    return sqrt(sum);
}

double pca_cpu(double* X, unsigned int M, unsigned int N, unsigned int K, 
    double* T_out, double* P_out, double* R_out, int iterations, int innerIterations, bool preNorm, bool shortNorm, FILE* debugFile)
{
    // PCA model: X = TP’ + R
    // input: X, MxN matrix (data)
    // input: M = number of rows in X
    // input: N = number of columns in X
    // input: K = number of components (K<=N)
    // output: T, MxK scores matrix
    // output: P, NxK loads matrix
    // output: R, MxN residual matrix

    // max error
    double er = 1.0e-7;
    unsigned int n, j, k;

    CHighResolutionTimer * timer = new CHighResolutionTimer(gran_msec);
    timer->reset();
    double startTime = timer->elapsed(false);

    double * T = NULL;
    double * P = NULL;
    double * R = NULL;

    for (int iter=0; iter<iterations; iter++)
    {
        // Initialize T and P each time. 
        // *** JCJC Need to do this? ***
        if (T) free(T);
        if (P) free(P);
        if (R) free(R);
        T = init_matrix_zero(M, K);
        P = init_matrix_zero(N, K);
        R = init_matrix_zero(M, N);

        // Initialize R from X.
        // (X is not written to below, so we are non-destructive of the only input).
        copy_matrix(X, R, M, N);

        // allocate memory for eigenvalues.
        double *L;
        L = (double*)calloc(K, sizeof(L[0]));;
        if(L == 0)
        {
            fprintf (stderr, "! host memory allocation error: T\n");
            return EXIT_FAILURE;
        }

        // mean center the data
        // allocate memory for per-row sums. (Used again later).
        double *U = (double*)calloc(M, sizeof(U[0]));
        if (!preNorm)
        {
            if(U == NULL)
            {
                fprintf (stderr, "! memory allocation error (U)\n");
                return EXIT_FAILURE;
            }
            cpublasDcopy(M, &R[0], 1, U, 1);

            for(n=1; n<N; n++)
            {
                cpublasDaxpy (M, 1.0, &R[n*M], 1, U, 1);
            }
            for(n=0; n<N; n++)
            {
                cpublasDaxpy (M, -1.0/N, U, 1, &R[n*M], 1);
            }
        }

        // GS-PCA
        //	printf("Performing GS-PCA for %d components\n", K);
        unsigned int J = innerIterations;
        // double a;
        for(k=0; k<K; k++)
        {
            cpublasDcopy (M, &R[k*M], 1, &T[k*M], 1);
            // a = 0.0;
            for(j=0; j<J; j++)
            {
                // blasDgemv (bool transpose, int m, int n, double alpha,
                // const double *A, int lda, const double *x,
                // int incx, double beta, double *y, int incy)
                //
                // y = alpha * op(A) * x + beta * y
                // d_P[col k] = d_R' * d_T[col k]
                cpublasDgemv (true, M, N, 1.0, R, M, &T[k*M], 1, 0.0, &P[k*N], 1);

                if(k>0)
                {
                    // blasDgemv (bool transpose, int m, int n, double alpha,
                    // const double *A, int lda, const double *x,
                    // int incx, double beta, double *y, int incy)
                    //
                    // y = alpha * op(A) * x + beta * y
                    // d_U = d_P[col 1-k]' * d_P[col k]
                    cpublasDgemv (true, N, k, 1.0, P, N, &P[k*N], 1, 0.0, U, 1);

                    // blasDgemv (bool transpose, int m, int n, double alpha,
                    // const double *A, int lda, const double *x,
                    // int incx, double beta, double *y, int incy)
                    //
                    // y = alpha * op(A) * x + beta * y
                    // d_P[colk] = -1.0 * d_P[col 1-k] * d_U + d_P[col k]
                    cpublasDgemv (false, N, k, -1.0, P, N, U, 1, 1.0, &P[k*N], 1);
                }

                double dnrm2 = cpublasDnrm2(N, &P[k*N], 1);

                cpublasDscal (N, 1.0/dnrm2, &P[k*N], 1);
                cpublasDgemv (false, M, N, 1.0, R, M, &P[k*N], 1, 0.0, &T[k*M], 1);
/*
                if(k>0)
                {
                    cpublasDgemv (true, M, k, 1.0, T, M, &T[k*M], 1, 0.0, U, 1);
                    cpublasDgemv (false, M, k, -1.0, T, M, U, 1, 1.0, &T[k*M], 1);
                }
*/
                L[k] = cpublasDnrm2(M, &T[k*M], 1);
                cpublasDscal(M, 1.0/L[k], &T[k*M], 1);
                /* Not testing convergence at the moment
                if(fabs(a - L[k]) < er*L[k]) 
                {
                    // printf("  Done in %d iterations\n", j+1);
                    break;
                }
                a = L[k]; */
            }
            // A = alpha * x * y' + A
            //   R = -L[k] * T[col k] * P[col k]' + R
/*			output_matrix(stdout, "\nDGER_CPU: T before", T, M, K);
            output_matrix(stdout, "DGER_CPU: P before", P, N, K);
            output_matrix(stdout, "DGER_CPU: R before", R, M, N);
            output_matrix(stdout, "DGER_CPU: L before", L, K, 1); */
            cpublasDger (M, N, - L[k], &T[k*M], 1, &P[k*N], 1, R, M);
        }
#if 1
        for(k=0; k<K; k++)
        {
            cpublasDscal(M, L[k], &T[k*M], 1);
        }
#endif
        free(L);
        free(U);

    }

    // Copy back the results of the final iteration, for comparison with other methods.
    memcpy(T_out, T, M*K*sizeof(double));
    memcpy(P_out, P, N*K*sizeof(double));
    memcpy(R_out, R, M*N*sizeof(double));

    double endTime = timer->elapsed(false);
    double duration = endTime - startTime;
    double rate = calculate_rate("CPU", iterations, duration);

    return rate;
}
