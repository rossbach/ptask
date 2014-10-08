//--------------------------------------------------------------------------------------
// File: graphcupca.cpp
//--------------------------------------------------------------------------------------
#include <stdio.h>
#include <crtdbg.h>
#include "accelerator.h"
#include "assert.h"
#include "shaderparms.h"
#include "sort.h"
#include <vector>
#include <algorithm>
#include "matrixtask.h"
#include "SimpleMatrix.h"
#include "matmul.h"
#include "graphcupca.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "pca_cpu.h"
#include "confighelpers.h"

#pragma warning(disable:4996)

using namespace std;
using namespace PTask;

int THREADBLOCKSIZE_X = 16;
int THREADBLOCKSIZE_Y = 16;

int MAX_PARAMS_INPUTS = 4096;

extern BOOL g_bSingleThreaded;
extern BOOL g_bRunCPUReference;

double pca_ptask(char * szfile, double* X, unsigned int M, unsigned int N, unsigned int K, 
    double* T, double* P, double* R, int iterations, int innerIterations, bool pipelineInputs, bool preNorm, bool shortNorm, FILE* debugFile)
{
    PCAGraph * graph = new PCAGraph();
    graph->BuildGraph(szfile, M, N, K, innerIterations, preNorm, shortNorm);
    double rate = graph->RunTest(X, M, N, K, T, P, R, iterations, pipelineInputs, preNorm);
    return rate;
}

int run_graph_cuda_pca_task(
    char * szfile, char * szshader, int rows, int cols,
    int iterations, int innerIterations, int components, BOOL pipelineInputs) 
{
    CONFIGUREPTASKU(UseOpenCL, FALSE);
    CONFIGUREPTASKU(UseDirectX, FALSE);
    CONFIGUREPTASKU(ICBlockPoolSize, 2);
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    bool preNorm = (strstr(szshader, "preNorm") != NULL) ? true : false;
    bool shortNorm = (strstr(szshader, "short") != NULL) ? true : false;

    bool debug = false;
    FILE* cpuLogFile = NULL;
    FILE* ptaskLogFile = NULL;
    if (debug)
    {
        cpuLogFile = fopen("log_cpu.txt", "w");
        if (cpuLogFile == NULL)
        {
            printf("\n**** Error opening cpu debug log file.\n");
            return EXIT_FAILURE;
        }
        ptaskLogFile = fopen("log_ptask.txt", "w");
        if (ptaskLogFile == NULL)
        {
            printf("\n**** Error opening ptask debug log file.\n");
            return EXIT_FAILURE;
        }
    }

    // M, N: Rows and Columns in input matrix.
    int M = rows;
    int N = cols;

    srand(2011);

    // X: Input matrix. 
    double* X = init_matrix_random(M, N, 1.0);

    // In 'preNorm' mode, normalize the input just once, so don't have to do in PCA implementation.
    if (preNorm)
    {
        row_normalize(X, M, N);
    }

    // K: Num princpal components to compute.
    int K = components;

    // T, P, R: Output matrices
    //   (One set per calculatin method)
    // T = MxK Scores matrix
    // P = NxK Loads matrix (aka the principal components)
    // R = MxN Residual matrix
    double* T_ptask = init_matrix_zero(M, K);
    double* P_ptask = init_matrix_zero(N, K);
    double* R_ptask = init_matrix_zero(M, N);

    double* T_cpu = init_matrix_zero(M, K);
    double* P_cpu = init_matrix_zero(N, K);
    double* R_cpu = init_matrix_zero(M, N);

    if(g_bRunCPUReference) {
        printf("%-4dx%-4d\t",
            M, N);
    } else {
        printf("%dx%d\t",
            M, N);
    }

    // Calculate using PTask-based CUDA implementation.
    double rate_ptask = pca_ptask(
        szfile, X, M, N, K, T_ptask, P_ptask, R_ptask, 
        iterations, innerIterations, (pipelineInputs != 0), preNorm, shortNorm, ptaskLogFile);
    //	double rate_ptask = 1.0;

    // Calculate reference result using CPU implementation.
    double rate_cpu;
    if(g_bRunCPUReference) {
        rate_cpu = pca_cpu(X, M, N, K, T_cpu, P_cpu, R_cpu, iterations, innerIterations, preNorm, shortNorm, cpuLogFile);
    }

#if 0
    output_matrix(stdout, "\nX", X, M, N);
    output_matrix(stdout, "T_ptask", T_ptask, M, K);
    output_matrix(stdout, "T_cpu", T_cpu, M, K);
    output_matrix(stdout, "P_ptask", P_ptask, N, K);
    output_matrix(stdout, "P_cpu", R_cpu, N, K);
    output_matrix(stdout, "R_ptask", R_ptask, M, N);
    output_matrix(stdout, "R_cpu", R_cpu, M, N);
#endif

    if(g_bRunCPUReference) {
        printf("%.6f\t%.6f\t  PTask is %.1fx slower than CPU.\n",
            rate_ptask, rate_cpu, 
            rate_cpu/rate_ptask);
        compare_matrices(T_ptask, "T_ptask", T_cpu, "T_cpu", M, K);
        compare_matrices(P_ptask, "P_ptask", P_cpu, "P_cpu", N, K);
        compare_matrices(R_ptask, "R_ptask", R_cpu, "R_cpu", M, N);
    } else {
        printf("%.6f\n",rate_ptask);
    }
    
/*
    output_matrix(stdout, "X", X, M, N);
    output_matrix(stdout, "T_ptask", T_ptask, M, K);
    output_matrix(stdout, "T_cpu", T_cpu, M, K);
    output_matrix(stdout, "P_ptask", P_ptask, N, K);
    output_matrix(stdout, "P_cpu", P_cpu, N, K);
    output_matrix(stdout, "R_ptask", R_ptask, M, N);
    output_matrix(stdout, "R_cpu", R_cpu, M, N);
*/

    PTask::Runtime::Terminate();
    return 0;
}

#define PARAM_ALIGNMENT 1

typedef struct __declspec(align(PARAM_ALIGNMENT)) DcopyParams_t {
    int N;          // num elements in input vector
    int OFFSETX;    // offset into X to start at
    int OFFSETY;    // offset into Y to start at
    int INCX;       // storage space between elements of X
    int INCY;       // storage space between elements of Y
} DCOPY_PARAMS;

typedef struct __declspec(align(PARAM_ALIGNMENT)) DaxpyParams_t {
    int N;			// num elements in input vector
    double ALPHA;	// scalar multiplier
    int OFFSETX;	// offset into X to start at
    int OFFSETY;	// offset into Y to start at
    int INCX;		// storage space between elements of X
    int INCY;		// storage space between elements of Y
} DAXPY_PARAMS;

typedef struct __declspec(align(PARAM_ALIGNMENT)) DgemvParams_t {
    bool	TRANSPOSE;	// transpose A or not
    int		M;			// num rows in input matrix A/elements in input vectors X and Y
    int		N;			// num columns in input matrix
    double	ALPHA;		// scalar multiplier
    double	BETA;		// scalar multiplier
    int		LDA;		// leading dimension of array used to store matrix A
    int		OFFSETX;	// offset into X to start at
    int		OFFSETY;	// offset into Y to start at
    int		INCX;		// storage space between elements of X
    int		INCY;		// storage space between elements of Y
} DGEMV_PARAMS;

typedef struct __declspec(align(PARAM_ALIGNMENT)) Dnrm2Params_t {
    int		N;			// num elements in input vector X
    int		OFFSETX;	// offset into X to start at
    int		OFFSETY;	// offset into Y to start at
    int		INCX;		// storage space between elements of X
} DNRM2_PARAMS;

typedef struct __declspec(align(PARAM_ALIGNMENT)) DscalParams_t {
    int		N;					// num elements in input vector
    double	ALPHA_CONST;		// scalar multiplier
    int		INCX;				// storage space between elements of X
    int		OFFSETX;			// offset into X to start at
    int		OFFSETALPHA_VAR;	// offset into ALPHA_VAR to read value from
    bool	INVERT_ALPHA_VAR;	// whether to use alpha_var or 1 / alpha_var
} DSCAL_PARAMS;

typedef struct __declspec(align(PARAM_ALIGNMENT)) DgerParams_t {
//JC hard coding ALPHA_CONST to -1 for now, to work around struct alignment issue. (Only value used at present).
//JC	double	ALPHA_CONST;	// scalar multiplier
    int		OFFSETY;		// offset into Y to start at
    int		INCX;			// storage space between elements of X
    int		INCY;			// storage space between elements of Y
    int		LDA;			// leading dimension of array used to store matrix A
    int		M;				// num rows in input matrix A/elements in input vectors X and Y
    int		N;				// num columns in input matrix
    int		OFFSETALPHA_VAR;// offset into ALPHA_VAR to read value from
    int		OFFSETX;		// offset into X to start at
} DGER_PARAMS;

PCAGraph::PCAGraph()
{
    m_ptaskGraph = NULL;

    m_TInputChannel = NULL;
    m_PInputChannel = NULL;
    m_RInputChannel = NULL;
    m_UInputChannel = NULL;
    m_LInputChannel = NULL;

    m_TOutputChannel = NULL;
    m_POutputChannel = NULL;
    m_ROutputChannel = NULL;
    m_UOutputChannel = NULL;

    m_numParamsChannels = 0;
    m_ParamsInputChannels = new PTask::GraphInputChannel*[MAX_PARAMS_INPUTS];
    m_ParamsDatablocks = new Datablock*[MAX_PARAMS_INPUTS];
}

PCAGraph::~PCAGraph()
{
    Graph::DestroyGraph(m_ptaskGraph);
}

bool PCAGraph::BuildGraph(char * szfile, int M, int N, int K, int innerIterations, bool preNorm, bool shortNorm)
{
    // PCA model: X = TP’ + R
    // input: X, MxN matrix (data)
    // input: M = number of rows in X
    // input: N = number of columns in X
    // input: K = number of components (K<=N)
    // output: T, MxK scores matrix
    // output: P, NxK loads matrix
    // output: R, MxN residual matrix

    //DebugBreak();

#pragma region templates, kernels etc
    PTASKDIM3 threadBlockSize(THREADBLOCKSIZE_X, THREADBLOCKSIZE_Y, 1);
#pragma warning(disable:4244)
    PTASKDIM3 gridSize(ceil((float)M/(float)THREADBLOCKSIZE_X), ceil((float)N/(float)THREADBLOCKSIZE_Y), 1);

    //
    // Datablock templates
    //
    // Output matrices: T, P and R.
    m_TTemplate = PTask::Runtime::GetDatablockTemplate("T_double", sizeof(double), M, K, 1);
    m_PTemplate = PTask::Runtime::GetDatablockTemplate("P_double", sizeof(double), N, K, 1);
    m_RTemplate = PTask::Runtime::GetDatablockTemplate("R_double", sizeof(double), M, N, 1);

    // No datablock template for input, X. 
    // X is used to set the initial value for R, which is then used as an input to the PTask graph.

    // Datablock template for U : M-element vector used to hold intermediate values.
    m_UTemplate = PTask::Runtime::GetDatablockTemplate("U_double", sizeof(double), M, 1, 1);

    // Datablock template for L : K-element vector used to hold the eigenvalue of each component.
    m_LTemplate = PTask::Runtime::GetDatablockTemplate("L_double", sizeof(double), K, 1, 1);

    // Datablock template for params struct for each kernel type.
    m_DcopyParamsTemplate	= PTask::Runtime::GetDatablockTemplate("DcopyParams", sizeof(DCOPY_PARAMS), PTPARM_BYVALSTRUCT);
    m_DaxpyParamsTemplate	= PTask::Runtime::GetDatablockTemplate("DaxpyParams", sizeof(DAXPY_PARAMS), PTPARM_BYVALSTRUCT);
    m_DgemvParamsTemplate	= PTask::Runtime::GetDatablockTemplate("DgemvParams", sizeof(DGEMV_PARAMS), PTPARM_BYVALSTRUCT);
    m_Dnrm2ParamsTemplate	= PTask::Runtime::GetDatablockTemplate("Dnrm2Params", sizeof(DNRM2_PARAMS), PTPARM_BYVALSTRUCT);
    m_DscalParamsTemplate	= PTask::Runtime::GetDatablockTemplate("DscalParams", sizeof(DSCAL_PARAMS), PTPARM_BYVALSTRUCT);
    m_DgerParamsTemplate	= PTask::Runtime::GetDatablockTemplate("DgerParams",  sizeof(DGER_PARAMS),  PTPARM_BYVALSTRUCT);

    //
    // Compiled kernels
    //

    CompiledKernel * ptaskDcopyKernel = COMPILE_KERNEL(szfile, "ptaskDcopy");
    CompiledKernel * ptaskDaxpyKernel = COMPILE_KERNEL(szfile, "ptaskDaxpy");
    CompiledKernel * ptaskDgemvKernel = COMPILE_KERNEL(szfile, "ptaskDgemv");
    CompiledKernel * ptaskDnrm2Kernel = COMPILE_KERNEL(szfile, "ptaskDnrm2");
    CompiledKernel * ptaskDscalKernel = COMPILE_KERNEL(szfile, "ptaskDscal");
    CompiledKernel * ptaskDgerKernel  = COMPILE_KERNEL(szfile, "ptaskDger");
#pragma endregion

    //
    // Build graph
    //
    m_ptaskGraph = new Graph();
    UINT portUID = 0;
    char taskName[100];
    char channelName[100];
    Port * previousToutput = NULL;
    Port * previousPoutput = NULL;
    Port * previousRoutput = NULL;
    Port * previousLoutput = NULL;
    Port * previousUoutput = NULL;

    //DebugBreak();

#pragma region normalize
    if (!preNorm)
    {
        // Mean-center the initial values of R
        //
#pragma region Dcopy_1
        {
            // Copy R[column 0] to U:
            //   blasDcopy(M, &R[0], 1, U, 1)
            const UINT inputPortCount = 3;
            const UINT outputPortCount = 2;
            Port ** inputPorts = new Port*[inputPortCount];
            Port ** outputPorts = new Port*[outputPortCount];

            inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_RTemplate, portUID++, "R_in", 0, 0);
            inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_UTemplate, portUID++, "U_in", 1, 1);
            inputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, m_DcopyParamsTemplate, portUID++, "PARAMS", 2);
            outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_RTemplate, portUID++, "R_out", 0);
            outputPorts[1]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_UTemplate, portUID++, "U_out", 1);

            sprintf(taskName, "Dcopy_1");
            Task * ptask = m_ptaskGraph->AddTask(ptaskDcopyKernel, 
                inputPortCount,
                inputPorts,
                outputPortCount,
                outputPorts,
                taskName);
            assert(ptask);

            ptask->SetComputeGeometry(M, N, 1);
            ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

            // Create and store graph input channel and datablock for params.
            GraphInputChannel * paramsChannel =
                m_ptaskGraph->AddInputChannel(inputPorts[2], "paramsConstChannel");
            DCOPY_PARAMS params;
            params.N = M;
            params.OFFSETX = 0;
            params.OFFSETY = 0;
            params.INCX = 1;
            params.INCY = 1;
            Datablock * paramsDatablock =
                PTask::Runtime::AllocateDatablock(m_DcopyParamsTemplate, &params, sizeof(params), paramsChannel);
            AddParamsChannel(paramsChannel, paramsDatablock);

            // Create graph input channels for R and U.
            m_RInputChannel = m_ptaskGraph->AddInputChannel(inputPorts[0], "RInputChannel");
            m_UInputChannel = m_ptaskGraph->AddInputChannel(inputPorts[1], "UInputChannel");

            previousRoutput = outputPorts[0];
            previousUoutput = outputPorts[1];
        }
#pragma endregion
#pragma region Daxpy_1
        {
            // For n=1 to N-1, add R[column n] to U:
            //   blasDaxpy (M, 1.0, &R[n*M], 1, U, 1)
            int maxColumnToAdd = N-1;
            const UINT inputPortCount = 3;
            const UINT outputPortCount = 2;
            Port ** inputPorts = new Port*[inputPortCount];
            Port ** outputPorts = new Port*[outputPortCount];
            for(int n=1; n<=maxColumnToAdd; n++)
            {
                //TODO: Check is OK to just overwrite the ports in the port arrays each iteration.
                inputPorts[0]  = PTask::Runtime::CreatePort(INPUT_PORT, m_RTemplate, portUID++, "R_in", 0, 0);
                inputPorts[1]  = PTask::Runtime::CreatePort(INPUT_PORT, m_UTemplate, portUID++, "U_in", 1, 1);
                inputPorts[2]  = PTask::Runtime::CreatePort(STICKY_PORT, m_DaxpyParamsTemplate, portUID++, "PARAMS", 2);
                outputPorts[0] = PTask::Runtime::CreatePort(OUTPUT_PORT, m_RTemplate, portUID++, "R_out", 0);
                outputPorts[1] = PTask::Runtime::CreatePort(OUTPUT_PORT, m_UTemplate, portUID++, "U_out", 1);

                sprintf(taskName, "Daxpy_1_column_%d", n);
                Task * ptask = m_ptaskGraph->AddTask(ptaskDaxpyKernel, 
                    inputPortCount,
                    inputPorts,
                    outputPortCount,
                    outputPorts,
                    taskName);
                assert(ptask);

                ptask->SetComputeGeometry(M, N, 1);
                ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

                // Create and store graph input channel and datablock for params.
                GraphInputChannel * paramsChannel =
                    m_ptaskGraph->AddInputChannel(inputPorts[2], "paramsConstChannel");
                DAXPY_PARAMS params;
                params.N = M;
                params.ALPHA = 1.0;
                params.OFFSETX = n*M; // only param that varies with n
                params.OFFSETY = 0;
                params.INCX = 1;
                params.INCY = 1;
                Datablock * paramsDatablock =
                    PTask::Runtime::AllocateDatablock(m_DaxpyParamsTemplate, &params, sizeof(params), paramsChannel);
                AddParamsChannel(paramsChannel, paramsDatablock);

                // Create internal channels connecting upstream R and U outputs to current task's R and U inputs.
                sprintf(channelName, "InternalChannel_R(%s to %s)",
                    previousRoutput->GetTask()->GetTaskName(),
                    inputPorts[0]->GetTask()->GetTaskName());
                m_ptaskGraph->AddInternalChannel(previousRoutput, inputPorts[0], channelName);

                sprintf(channelName, "InternalChannel_U(%s to %s)",
                    previousUoutput->GetTask()->GetTaskName(),
                    inputPorts[1]->GetTask()->GetTaskName());
                m_ptaskGraph->AddInternalChannel(previousUoutput, inputPorts[1], channelName);

                // Save current output ports, to connect to downstream consumer.
                previousRoutput = outputPorts[0];
                previousUoutput = outputPorts[1];
            }
        }
#pragma endregion
#pragma region Daxpy_2
        {
            // For n=0 to N-1, subtract U/N from R[column n]:
            //   cpublasDaxpy (M, -1.0/N, U, 1, &R[n*M], 1)
            int maxColumnToAdd = N-1;
            const UINT inputPortCount = 3;
            const UINT outputPortCount = 2;
            Port ** inputPorts = new Port*[inputPortCount];
            Port ** outputPorts = new Port*[outputPortCount];

            for(int n=0; n<=maxColumnToAdd; n++)
            {
                //TODO: Check is OK to just overwrite the ports in the port arrays each iteration.
                inputPorts[0]  = PTask::Runtime::CreatePort(INPUT_PORT, m_UTemplate, portUID++, "U_in", 0, 0);
                inputPorts[1]  = PTask::Runtime::CreatePort(INPUT_PORT, m_RTemplate, portUID++, "R_in", 1, 1);
                inputPorts[2]  = PTask::Runtime::CreatePort(STICKY_PORT, m_DaxpyParamsTemplate, portUID++, "PARAMS", 2);
                outputPorts[0] = PTask::Runtime::CreatePort(OUTPUT_PORT, m_UTemplate, portUID++, "U_out", 0);
                outputPorts[1] = PTask::Runtime::CreatePort(OUTPUT_PORT, m_RTemplate, portUID++, "R_out", 1);

                sprintf(taskName, "Daxpy_2_column_%d", n);
                Task * ptask = m_ptaskGraph->AddTask(ptaskDaxpyKernel,
                    inputPortCount,
                    inputPorts,
                    outputPortCount,
                    outputPorts,
                    taskName);
                assert(ptask);

                ptask->SetComputeGeometry(M, N, 1);
                ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

                // Create internal channels connecting upstream R and U outputs to current task's R and U inputs.
                sprintf(channelName, "InternalChannel_U(%s to %s)",
                    previousUoutput->GetTask()->GetTaskName(),
                    inputPorts[0]->GetTask()->GetTaskName());
                m_ptaskGraph->AddInternalChannel(previousUoutput, inputPorts[0], channelName);

                sprintf(channelName, "InternalChannel_R(%s to %s)",
                    previousRoutput->GetTask()->GetTaskName(),
                    inputPorts[1]->GetTask()->GetTaskName());
                m_ptaskGraph->AddInternalChannel(previousRoutput, inputPorts[1], channelName);


                // Create and store graph input channel and datablock for params.
                GraphInputChannel * paramsChannel =
                    m_ptaskGraph->AddInputChannel(inputPorts[2], "paramsConstChannel");
                DAXPY_PARAMS params;
                params.N = M;
                params.ALPHA = -1.0/N;
                params.OFFSETX = 0;
                params.OFFSETY = n*M; // only param that varies with n
                params.INCX = 1;
                params.INCY = 1;
                Datablock * paramsDatablock =
                    PTask::Runtime::AllocateDatablock(m_DaxpyParamsTemplate, &params, sizeof(params), paramsChannel);
                AddParamsChannel(paramsChannel, paramsDatablock);

                // Save current output ports, to connect to downstream consumer.
                previousUoutput = outputPorts[0];
                previousRoutput = outputPorts[1];
            }
        }
#pragma endregion
    }
#pragma endregion

    // Unroll loop for each component
    for(int k=0; k<K; k++)
    {
#pragma region Dcopy_2
        {
            // Copy R[col k] to T[col k]:
            //   blasDcopy(M, &R[k*M], 1, &T[k*M], 1);
            const UINT inputPortCount = 3;
            const UINT outputPortCount = 2;
            Port ** inputPorts = new Port*[inputPortCount];
            Port ** outputPorts = new Port*[outputPortCount];

            inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_RTemplate, portUID++, "R_in", 0, 0);
            inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_TTemplate, portUID++, "T_in", 1, 1);
            inputPorts[2]   = PTask::Runtime::CreatePort(STICKY_PORT, m_DcopyParamsTemplate, portUID++, "PARAMS", 2);
            outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_RTemplate, portUID++, "R_out", 0);
            outputPorts[1]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_TTemplate, portUID++, "T_out", 1);

            sprintf(taskName, "Dcopy_2_component_%d", k);
            Task * ptask = m_ptaskGraph->AddTask(ptaskDcopyKernel, 
                inputPortCount,
                inputPorts,
                outputPortCount,
                outputPorts,
                taskName);
            assert(ptask);

            ptask->SetComputeGeometry(M, N, 1);
            ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

            // Create and store graph input channel and datablock for params.
            GraphInputChannel * paramsChannel =
                m_ptaskGraph->AddInputChannel(inputPorts[2], "paramsConstChannel");
            DCOPY_PARAMS params;
            params.N = M;
            params.OFFSETX = k*M; // varies with k
            params.OFFSETY = k*M; // varies with k
            params.INCX = 1;
            params.INCY = 1;
            Datablock * paramsDatablock =
                PTask::Runtime::AllocateDatablock(m_DcopyParamsTemplate, &params, sizeof(params), paramsChannel);
            AddParamsChannel(paramsChannel, paramsDatablock);

            // Connect R input.
            if (preNorm && (0 == k))
            {
                // If preNorm was done, and k==0, this PTask is the first consumer of R, so create graph input channel.
                assert(m_RInputChannel == NULL);
                assert(previousRoutput == NULL);
                m_RInputChannel = m_ptaskGraph->AddInputChannel(inputPorts[0], "RInputChannel");
            } else {
                // Otherwise, create internal channel to connect it to the previous producer of R.
                assert(m_RInputChannel != NULL);
                assert(previousRoutput != NULL);
                sprintf(channelName, "InternalChannel_R(%s to %s)",
                    previousRoutput->GetTask()->GetTaskName(),
                    inputPorts[0]->GetTask()->GetTaskName());
                m_ptaskGraph->AddInternalChannel(previousRoutput, inputPorts[0], channelName);
            }

            // Connect T input.
            if (0 == k)
            {
                // If k==0, this PTask is the first consumer of T, so create graph input channel.
                assert(m_TInputChannel == NULL);
                assert(previousToutput == NULL);
                m_TInputChannel = m_ptaskGraph->AddInputChannel(inputPorts[1], "TInputChannel");
            } else {
                // Otherwise, create internal channel to connect it to the previous producer of T.
                assert(m_TInputChannel != NULL);
                assert(previousToutput != NULL);
                sprintf(channelName, "InternalChannel_R(%s to %s)",
                    previousToutput->GetTask()->GetTaskName(),
                    inputPorts[1]->GetTask()->GetTaskName());
                m_ptaskGraph->AddInternalChannel(previousToutput, inputPorts[1], channelName);
            }

            previousRoutput = outputPorts[0];
            previousToutput = outputPorts[1];
        }
#pragma endregion
        // Unroll loop for each inner iteration

        // Not testing for convergence at the moment.
        // a = 0.0;
        for(int j=0; j<innerIterations; j++)
        {
#pragma region Dgemv_1
            {
                // blasDgemv (bool transpose, int m, int n, double alpha,
                // const double *A, int lda, const double *x,
                // int incx, double beta, double *y, int incy)
                //
                // y = alpha * op(A) * x + beta * y
                // d_P[col k] = d_R' * d_T[col k]
                //   blasDgemv (true, M, N, 1.0, R, M, &T[k*M], 1, 0.0, &P[k*N], 1);
                const UINT inputPortCount = 4;
                const UINT outputPortCount = 1;
                Port ** inputPorts = new Port*[inputPortCount];
                Port ** outputPorts = new Port*[outputPortCount];

                inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_RTemplate, portUID++, "R_in", 0);
                inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_TTemplate, portUID++, "T_in", 1);
                inputPorts[2]	= PTask::Runtime::CreatePort(INPUT_PORT, m_PTemplate, portUID++, "P_in", 2, 0);
                inputPorts[3]   = PTask::Runtime::CreatePort(STICKY_PORT, m_DgemvParamsTemplate, portUID++, "PARAMS", 3);
                outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_PTemplate, portUID++, "P_out", 0);

                sprintf(taskName, "Dgemv_1_comp_%d_iter_%d", k, j);
                Task * ptask = m_ptaskGraph->AddTask(ptaskDgemvKernel, 
                    inputPortCount,
                    inputPorts,
                    outputPortCount,
                    outputPorts,
                    taskName);
                assert(ptask);

                ptask->SetComputeGeometry(M, N, 1);
                ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

                // Create and store graph input channel and datablock for params.
                GraphInputChannel * paramsChannel =
                    m_ptaskGraph->AddInputChannel(inputPorts[3], "paramsConstChannel");
                DGEMV_PARAMS params;
                params.TRANSPOSE = true;
                params.M = M;
                params.N = N;
                params.ALPHA = 1.0;
                params.BETA = 0.0;
                params.LDA = M;
                params.OFFSETX = k*M;
                params.OFFSETY = k*N;
                params.INCX = 1;
                params.INCY = 1;
                Datablock * paramsDatablock =
                    PTask::Runtime::AllocateDatablock(m_DgemvParamsTemplate, &params, sizeof(params), paramsChannel);
                AddParamsChannel(paramsChannel, paramsDatablock);

                // Connect R and T inputs.
                assert(m_RInputChannel != NULL);
                assert(previousRoutput != NULL);
                sprintf(channelName, "InternalChannel_R(%s to %s)",
                    previousRoutput->GetTask()->GetTaskName(),
                    inputPorts[0]->GetTask()->GetTaskName());
                m_ptaskGraph->AddInternalChannel(previousRoutput, inputPorts[0], channelName);

                assert(m_TInputChannel != NULL);
                assert(previousToutput != NULL);
                sprintf(channelName, "InternalChannel_T(%s to %s)",
                    previousToutput->GetTask()->GetTaskName(),
                    inputPorts[1]->GetTask()->GetTaskName());
                m_ptaskGraph->AddInternalChannel(previousToutput, inputPorts[1], channelName);

                // Connect P input.
                if ((0 == k) && (0 == j))
                {
                    // If k==0 and j==0, this PTask is the first consumer of P, so create graph input channel.
                    assert(m_PInputChannel == NULL);
                    assert(previousPoutput == NULL);
                    m_PInputChannel = m_ptaskGraph->AddInputChannel(inputPorts[2], "PInputChannel");
                } else {
                    // Otherwise, create internal channel to connect it to the previous producer of T.
                    assert(m_PInputChannel != NULL);
                    assert(previousPoutput != NULL);
                    sprintf(channelName, "InternalChannel_P(%s to %s)",
                        previousPoutput->GetTask()->GetTaskName(),
                        inputPorts[2]->GetTask()->GetTaskName());
                    m_ptaskGraph->AddInternalChannel(previousPoutput, inputPorts[2], channelName);
                }

                previousPoutput = outputPorts[0];
            }
#pragma endregion

            if(k>0)
            {
#pragma region Dgemv_2
                {
                    // blasDgemv (bool transpose, int m, int n, double alpha,
                    // const double *A, int lda, const double *x,
                    // int incx, double beta, double *y, int incy)
                    //
                    // y = alpha * op(A) * x + beta * y
                    // d_U = d_P[col 1-k]' * d_P[col k]
                    //   cpublasDgemv (true, N, k, 1.0, P, N, &P[k*N], 1, 0.0, U, 1);
                    const UINT inputPortCount = 4;
                    const UINT outputPortCount = 1;
                    Port ** inputPorts = new Port*[inputPortCount];
                    Port ** outputPorts = new Port*[outputPortCount];

                    inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_PTemplate, portUID++, "P_in", 0);
                    inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_PTemplate, portUID++, "P_in_again", 1);
                    inputPorts[2]	= PTask::Runtime::CreatePort(INPUT_PORT, m_UTemplate, portUID++, "U_in", 2, 0);
                    inputPorts[3]   = PTask::Runtime::CreatePort(STICKY_PORT, m_DgemvParamsTemplate, portUID++, "PARAMS", 3);
                    outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_UTemplate, portUID++, "U_out", 0);

                    sprintf(taskName, "Dgemv_2_comp_%d_iter_%d", k, j);
                    Task * ptask = m_ptaskGraph->AddTask(ptaskDgemvKernel, 
                        inputPortCount,
                        inputPorts,
                        outputPortCount,
                        outputPorts,
                        taskName);
                    assert(ptask);

                    ptask->SetComputeGeometry(M, N, 1);
                    ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

                    // Create and store graph input channel and datablock for params.
                    GraphInputChannel * paramsChannel =
                        m_ptaskGraph->AddInputChannel(inputPorts[3], "paramsConstChannel");
                    DGEMV_PARAMS params;
                    params.TRANSPOSE = true;
                    params.M = N;
                    params.N = k;
                    params.ALPHA = 1.0;
                    params.BETA = 0.0;
                    params.LDA = N;
                    params.OFFSETX = k*N;
                    params.OFFSETY = 0;
                    params.INCX = 1;
                    params.INCY = 1;
                    Datablock * paramsDatablock =
                        PTask::Runtime::AllocateDatablock(m_DgemvParamsTemplate, &params, sizeof(params), paramsChannel);
                    AddParamsChannel(paramsChannel, paramsDatablock);

                    // Connect P input (twice).
                    assert(m_PInputChannel != NULL);
                    assert(previousPoutput != NULL);
                    sprintf(channelName, "InternalChannel_P(%s to %s)",
                        previousPoutput->GetTask()->GetTaskName(),
                        inputPorts[0]->GetTask()->GetTaskName());
                    m_ptaskGraph->AddInternalChannel(previousPoutput, inputPorts[0], channelName);

                    assert(m_PInputChannel != NULL);
                    assert(previousPoutput != NULL);
                    sprintf(channelName, "InternalChannel_P(%s to %s)",
                        previousPoutput->GetTask()->GetTaskName(),
                        inputPorts[1]->GetTask()->GetTaskName());
                    m_ptaskGraph->AddInternalChannel(previousPoutput, inputPorts[1], channelName);

                    // Connect U input.
                    if (preNorm && (1 == k) && (0 == j))
                    {
                        // If preNorm was done, k==1 and j==0, this PTask is the first consumer of U, so create graph input channel.
                        // (k==1, rather than k==0 as this ptask is skipped for k==0)
                        assert(m_UInputChannel == NULL);
                        assert(previousUoutput == NULL);
                        m_UInputChannel = m_ptaskGraph->AddInputChannel(inputPorts[2], "UInputChannel");
                    } else {
                        // Otherwise, create internal channel to connect it to the previous producer of U.
                        assert(m_UInputChannel != NULL);
                        assert(previousUoutput != NULL);
                        sprintf(channelName, "InternalChannel_U(%s to %s)",
                            previousUoutput->GetTask()->GetTaskName(),
                            inputPorts[2]->GetTask()->GetTaskName());
                        m_ptaskGraph->AddInternalChannel(previousUoutput, inputPorts[2], channelName);
                    }

                    previousUoutput = outputPorts[0];
                }
#pragma endregion
#pragma region Dgemv_3
                {
                    // blasDgemv (bool transpose, int m, int n, double alpha,
                    // const double *A, int lda, const double *x,
                    // int incx, double beta, double *y, int incy)
                    //
                    // y = alpha * op(A) * x + beta * y
                    // d_P[col k] = -1.0 * d_P[col 1-k] * d_U + d_P[col k]
                    //   cpublasDgemv (false, N, k, -1.0, P, N, U, 1, 1.0, &P[k*N], 1);
                    const UINT inputPortCount = 4;
                    const UINT outputPortCount = 1;
                    Port ** inputPorts = new Port*[inputPortCount];
                    Port ** outputPorts = new Port*[outputPortCount];

                    inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_PTemplate, portUID++, "P_in", 0);
                    inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_UTemplate, portUID++, "U_in", 1);
                    inputPorts[2]	= PTask::Runtime::CreatePort(INPUT_PORT, m_PTemplate, portUID++, "P_in_again", 2, 0);
                    inputPorts[3]   = PTask::Runtime::CreatePort(STICKY_PORT, m_DgemvParamsTemplate, portUID++, "PARAMS", 3);
                    outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_PTemplate, portUID++, "P_out", 0);

                    sprintf(taskName, "Dgemv_3_comp_%d_iter_%d", k, j);
                    Task * ptask = m_ptaskGraph->AddTask(ptaskDgemvKernel, 
                        inputPortCount,
                        inputPorts,
                        outputPortCount,
                        outputPorts,
                        taskName);
                    assert(ptask);

                    ptask->SetComputeGeometry(M, N, 1);
                    ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

                    // Create and store graph input channel and datablock for params.
                    GraphInputChannel * paramsChannel =
                        m_ptaskGraph->AddInputChannel(inputPorts[3], "paramsConstChannel");
                    DGEMV_PARAMS params;
                    params.TRANSPOSE = false;
                    params.M = N;
                    params.N = k;
                    params.ALPHA = -1.0;
                    params.BETA = 1.0;
                    params.LDA = N;
                    params.OFFSETX = 0;
                    params.OFFSETY = k*N;
                    params.INCX = 1;
                    params.INCY = 1;
                    Datablock * paramsDatablock =
                        PTask::Runtime::AllocateDatablock(m_DgemvParamsTemplate, &params, sizeof(params), paramsChannel);
                    AddParamsChannel(paramsChannel, paramsDatablock);

                    // Connect P input (twice).
                    assert(m_PInputChannel != NULL);
                    assert(previousPoutput != NULL);
                    sprintf(channelName, "InternalChannel_P(%s to %s)",
                        previousPoutput->GetTask()->GetTaskName(),
                        inputPorts[0]->GetTask()->GetTaskName());
                    m_ptaskGraph->AddInternalChannel(previousPoutput, inputPorts[0], channelName);

                    assert(m_PInputChannel != NULL);
                    assert(previousPoutput != NULL);
                    sprintf(channelName, "InternalChannel_P(%s to %s)",
                        previousPoutput->GetTask()->GetTaskName(),
                        inputPorts[2]->GetTask()->GetTaskName());
                    m_ptaskGraph->AddInternalChannel(previousPoutput, inputPorts[2], channelName);

                    // Connect U input.
                    assert(m_UInputChannel != NULL);
                    assert(previousUoutput != NULL);
                    sprintf(channelName, "InternalChannel_U(%s to %s)",
                        previousUoutput->GetTask()->GetTaskName(),
                        inputPorts[1]->GetTask()->GetTaskName());
                    m_ptaskGraph->AddInternalChannel(previousUoutput, inputPorts[1], channelName);

                    previousPoutput = outputPorts[0];
                }
#pragma endregion
            } // if(k>0)
#ifndef JC_DEBUG
//JC matches if comment out to here
#pragma region Dnrm2_1
        {
            // Store norm2(P[col k]) into L[k] even though it is not the eigenvalue
            // ... L[k] will get overwritten below with the eigenvalue. Just squatting the 
            // space before that to save creating another temp buffer.
            //   blasDnrm2(N, &P[k*N], 1, L[k])
            const UINT inputPortCount = 3;
            const UINT outputPortCount = 1;
            Port ** inputPorts = new Port*[inputPortCount];
            Port ** outputPorts = new Port*[outputPortCount];

            inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_PTemplate, portUID++, "P_in", 0);
            inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_LTemplate, portUID++, "L_in", 1, 0);
            inputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, m_Dnrm2ParamsTemplate, portUID++, "PARAMS", 2);
            outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_LTemplate, portUID++, "L_out", 0);

            sprintf(taskName, "Dnrm2_1");
            Task * ptask = m_ptaskGraph->AddTask(ptaskDnrm2Kernel, 
                inputPortCount,
                inputPorts,
                outputPortCount,
                outputPorts,
                taskName);
            assert(ptask);

            ptask->SetComputeGeometry(M, N, 1);
            ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

            // Create and store graph input channel and datablock for params.
            GraphInputChannel * paramsChannel =
                m_ptaskGraph->AddInputChannel(inputPorts[2], "paramsConstChannel");
            DNRM2_PARAMS params;
            params.N = N;
            params.OFFSETX = k*N;
            params.OFFSETY = k;
            params.INCX = 1;
            Datablock * paramsDatablock =
                PTask::Runtime::AllocateDatablock(m_DcopyParamsTemplate, &params, sizeof(params), paramsChannel);
            AddParamsChannel(paramsChannel, paramsDatablock);

            // Connect P input.
            assert(m_PInputChannel != NULL);
            assert(previousPoutput != NULL);
            sprintf(channelName, "InternalChannel_P(%s to %s)",
                previousPoutput->GetTask()->GetTaskName(),
                inputPorts[0]->GetTask()->GetTaskName());
            m_ptaskGraph->AddInternalChannel(previousPoutput, inputPorts[0], channelName);

            // Connect L input.
            if ((0 == k) && (0 == j))
            {
                // If k==0 and j==0, this PTask is the first consumer of L, so create graph input channel.
                assert(m_LInputChannel == NULL);
                assert(previousLoutput == NULL);
                m_LInputChannel = m_ptaskGraph->AddInputChannel(inputPorts[1], "LInputChannel");
            } else {
                // Otherwise, create internal channel to connect it to the previous producer of L.
                assert(m_LInputChannel != NULL);
                assert(previousLoutput != NULL);
                sprintf(channelName, "InternalChannel_P(%s to %s)",
                    previousLoutput->GetTask()->GetTaskName(),
                    inputPorts[1]->GetTask()->GetTaskName());
                m_ptaskGraph->AddInternalChannel(previousLoutput, inputPorts[1], channelName);
            }

            previousLoutput = outputPorts[0];
        }
#pragma endregion
//JC#ifndef JC_DEBUG
//JC mismatch if comment out to here

#pragma region Dscal_1
        {
            // cpublasDscal(N, 1.0/L[k], &P[k*N], 1);
            const UINT inputPortCount = 3;
            const UINT outputPortCount = 1;
            Port ** inputPorts = new Port*[inputPortCount];
            Port ** outputPorts = new Port*[outputPortCount];

            inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_LTemplate, portUID++, "L_in", 0);
            inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_PTemplate, portUID++, "P_in", 1, 0);
            inputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, m_DscalParamsTemplate, portUID++, "PARAMS", 2);
            outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_PTemplate, portUID++, "P_out", 0);

            sprintf(taskName, "Dscal_1");
            Task * ptask = m_ptaskGraph->AddTask(ptaskDscalKernel, 
                inputPortCount,
                inputPorts,
                outputPortCount,
                outputPorts,
                taskName);
            assert(ptask);

            ptask->SetComputeGeometry(M, N, 1);
            ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

            // Create and store graph input channel and datablock for params.
            GraphInputChannel * paramsChannel =
                m_ptaskGraph->AddInputChannel(inputPorts[2], "paramsConstChannel");
            DSCAL_PARAMS params;
            params.N = N;
            params.ALPHA_CONST = 1.0;
            params.INCX = 1;
            params.OFFSETX = k*N;
            params.OFFSETALPHA_VAR = k;
            params.INVERT_ALPHA_VAR = true;
            Datablock * paramsDatablock =
                PTask::Runtime::AllocateDatablock(m_DscalParamsTemplate, &params, sizeof(params), paramsChannel);
            AddParamsChannel(paramsChannel, paramsDatablock);

            // Connect L input.
            assert(m_LInputChannel != NULL);
            assert(previousLoutput != NULL);
            sprintf(channelName, "InternalChannel_L(%s to %s)",
                previousLoutput->GetTask()->GetTaskName(),
                inputPorts[0]->GetTask()->GetTaskName());
            m_ptaskGraph->AddInternalChannel(previousLoutput, inputPorts[0], channelName);

            // Connect P input.
            assert(m_PInputChannel != NULL);
            assert(previousPoutput != NULL);
            sprintf(channelName, "InternalChannel_P(%s to %s)",
                previousPoutput->GetTask()->GetTaskName(),
                inputPorts[1]->GetTask()->GetTaskName());
            m_ptaskGraph->AddInternalChannel(previousPoutput, inputPorts[1], channelName);

            previousPoutput = outputPorts[0];
        }
#pragma endregion

//JC#ifndef JC_DEBUG
//JC mismatch if comment out to here

#pragma region Dgemv_4
                {
                    // blasDgemv (bool transpose, int m, int n, double alpha,
                    // const double *A, int lda, const double *x,
                    // int incx, double beta, double *y, int incy)
                    //
                    // y = alpha * op(A) * x + beta * y
                    // T[col k] = R * P[col k]
                    //   blasDgemv (false, M, N, 1.0, R, M, &P[k*N], 1, 0.0, &T[k*M], 1);
                    const UINT inputPortCount = 4;
                    const UINT outputPortCount = 1;
                    Port ** inputPorts = new Port*[inputPortCount];
                    Port ** outputPorts = new Port*[outputPortCount];

                    inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_RTemplate, portUID++, "R_in", 0);
                    inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_PTemplate, portUID++, "P_in", 1);
                    inputPorts[2]	= PTask::Runtime::CreatePort(INPUT_PORT, m_TTemplate, portUID++, "T_in", 2, 0);
                    inputPorts[3]   = PTask::Runtime::CreatePort(STICKY_PORT, m_DgemvParamsTemplate, portUID++, "PARAMS", 3);
                    outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_TTemplate, portUID++, "T_out", 0);

                    sprintf(taskName, "Dgemv4_comp_%d_iter_%d", k, j);
                    Task * ptask = m_ptaskGraph->AddTask(ptaskDgemvKernel, 
                        inputPortCount,
                        inputPorts,
                        outputPortCount,
                        outputPorts,
                        taskName);
                    assert(ptask);

                    ptask->SetComputeGeometry(M, N, 1);
                    ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

                    // Create and store graph input channel and datablock for params.
                    GraphInputChannel * paramsChannel =
                        m_ptaskGraph->AddInputChannel(inputPorts[3], "paramsConstChannel");
                    DGEMV_PARAMS params;
                    params.TRANSPOSE = false;
                    params.M = N;
                    params.N = N;
                    params.ALPHA = 1.0;
                    params.BETA = 0.0;
                    params.LDA = M;
                    params.OFFSETX = k*N;
                    params.OFFSETY = k*M;
                    params.INCX = 1;
                    params.INCY = 1;
                    Datablock * paramsDatablock =
                        PTask::Runtime::AllocateDatablock(m_DgemvParamsTemplate, &params, sizeof(params), paramsChannel);
                    AddParamsChannel(paramsChannel, paramsDatablock);

                    // Connect R, P and T inputs.
                    assert(m_RInputChannel != NULL);
                    assert(previousRoutput != NULL);
                    sprintf(channelName, "InternalChannel_R(%s to %s)",
                        previousRoutput->GetTask()->GetTaskName(),
                        inputPorts[0]->GetTask()->GetTaskName());
                    m_ptaskGraph->AddInternalChannel(previousRoutput, inputPorts[0], channelName);

                    assert(m_PInputChannel != NULL);
                    assert(previousPoutput != NULL);
                    sprintf(channelName, "InternalChannel_P(%s to %s)",
                        previousPoutput->GetTask()->GetTaskName(),
                        inputPorts[1]->GetTask()->GetTaskName());
                    m_ptaskGraph->AddInternalChannel(previousPoutput, inputPorts[1], channelName);

                    assert(m_TInputChannel != NULL);
                    assert(previousToutput != NULL);
                    sprintf(channelName, "InternalChannel_T(%s to %s)",
                        previousToutput->GetTask()->GetTaskName(),
                        inputPorts[2]->GetTask()->GetTaskName());
                    m_ptaskGraph->AddInternalChannel(previousToutput, inputPorts[2], channelName);

                    previousToutput = outputPorts[0];
                }
#pragma endregion
#if 0
                if(k>0)
                {					
#pragma region Dgemv_5
                    {
                        // blasDgemv (bool transpose, int m, int n, double alpha,
                        // const double *A, int lda, const double *x,
                        // int incx, double beta, double *y, int incy)
                        //
                        // y = alpha * op(A) * x + beta * y
                        // U = T[col 1 to k]' * T[col k]
                        //   blasDgemv (true, M, k, 1.0, T, M, &T[k*M], 1, 0.0, U, 1)
                        const UINT inputPortCount = 4;
                        const UINT outputPortCount = 1;
                        Port ** inputPorts = new Port*[inputPortCount];
                        Port ** outputPorts = new Port*[outputPortCount];

                        inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_TTemplate, portUID++, "T_in", 0);
                        inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_TTemplate, portUID++, "T_in_again", 1);
                        inputPorts[2]	= PTask::Runtime::CreatePort(INPUT_PORT, m_UTemplate, portUID++, "U_in", 2, 0);
                        inputPorts[3]   = PTask::Runtime::CreatePort(STICKY_PORT, m_DgemvParamsTemplate, portUID++, "PARAMS", 3);
                        outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_UTemplate, portUID++, "U_out", 0);

                        sprintf(taskName, "Dgemv_5_comp_%d_iter_%d", k, j);
                        Task * ptask = m_ptaskGraph->AddTask(ptaskDgemvKernel, 
                            inputPortCount,
                            inputPorts,
                            outputPortCount,
                            outputPorts,
                            taskName);
                        assert(ptask);

                        ptask->SetComputeGeometry(M, N, 1);
                        ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

                        // Create and store graph input channel and datablock for params.
                        GraphInputChannel * paramsChannel =
                            m_ptaskGraph->AddInputChannel(inputPorts[3], "paramsConstChannel");
                        DGEMV_PARAMS params;
                        params.TRANSPOSE = true;
                        params.M = M;
                        params.N = k;
                        params.ALPHA = 1.0;
                        params.BETA = 0.0;
                        params.LDA = M;
                        params.OFFSETX = k*M;
                        params.OFFSETY = 0;
                        params.INCX = 1;
                        params.INCY = 1;
                        Datablock * paramsDatablock =
                            PTask::Runtime::AllocateDatablock(m_DgemvParamsTemplate, &params, sizeof(params), paramsChannel);
                        AddParamsChannel(paramsChannel, paramsDatablock);

                        // Connect T input (twice).
                        assert(m_TInputChannel != NULL);
                        assert(previousToutput != NULL);
                        sprintf(channelName, "InternalChannel_T(%s to %s)",
                            previousToutput->GetTask()->GetTaskName(),
                            inputPorts[0]->GetTask()->GetTaskName());
                        m_ptaskGraph->AddInternalChannel(previousToutput, inputPorts[0], channelName);

                        assert(m_TInputChannel != NULL);
                        assert(previousToutput != NULL);
                        sprintf(channelName, "InternalChannel_T(%s to %s)",
                            previousToutput->GetTask()->GetTaskName(),
                            inputPorts[1]->GetTask()->GetTaskName());
                        m_ptaskGraph->AddInternalChannel(previousToutput, inputPorts[1], channelName);

                        // Connect U input.
                        assert(m_UInputChannel != NULL);
                        assert(previousUoutput != NULL);
                        sprintf(channelName, "InternalChannel_U(%s to %s)",
                            previousUoutput->GetTask()->GetTaskName(),
                            inputPorts[2]->GetTask()->GetTaskName());
                        m_ptaskGraph->AddInternalChannel(previousUoutput, inputPorts[2], channelName);

                        previousUoutput = outputPorts[0];
                    }
#pragma endregion
#pragma region Dgemv_6
                {
                    // blasDgemv (bool transpose, int m, int n, double alpha,
                    // const double *A, int lda, const double *x,
                    // int incx, double beta, double *y, int incy)
                    //
                    // y = alpha * op(A) * x + beta * y
                    // T[col k] = -1.0 * T[col 1 to k] * U + T[col k]
                    //   blasDgemv (false, M, k, -1.0, T, M, U, 1, 1.0, &T[k*M], 1)
                    const UINT inputPortCount = 4;
                    const UINT outputPortCount = 1;
                    Port ** inputPorts = new Port*[inputPortCount];
                    Port ** outputPorts = new Port*[outputPortCount];

                    inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_TTemplate, portUID++, "T_in", 0);
                    inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_UTemplate, portUID++, "U_in", 1);
                    inputPorts[2]	= PTask::Runtime::CreatePort(INPUT_PORT, m_TTemplate, portUID++, "T_in_again", 2, 0);
                    inputPorts[3]   = PTask::Runtime::CreatePort(STICKY_PORT, m_DgemvParamsTemplate, portUID++, "PARAMS", 3);
                    outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_TTemplate, portUID++, "T_out", 0);

                    sprintf(taskName, "Dgemv_6_comp_%d_iter_%d", k, j);
                    Task * ptask = m_ptaskGraph->AddTask(ptaskDgemvKernel, 
                        inputPortCount,
                        inputPorts,
                        outputPortCount,
                        outputPorts,
                        taskName);
                    assert(ptask);

                    ptask->SetComputeGeometry(M, N, 1);
                    ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

                    // Create and store graph input channel and datablock for params.
                    GraphInputChannel * paramsChannel =
                        m_ptaskGraph->AddInputChannel(inputPorts[3], "paramsConstChannel");
                    DGEMV_PARAMS params;
                    params.TRANSPOSE = false;
                    params.M = M;
                    params.N = k;
                    params.ALPHA = -1.0;
                    params.BETA = 1.0;
                    params.LDA = M;
                    params.OFFSETX = 0;
                    params.OFFSETY = k*M;
                    params.INCX = 1;
                    params.INCY = 1;
                    Datablock * paramsDatablock =
                        PTask::Runtime::AllocateDatablock(m_DgemvParamsTemplate, &params, sizeof(params), paramsChannel);
                    AddParamsChannel(paramsChannel, paramsDatablock);

                    // Connect T input (twice).
                    assert(m_TInputChannel != NULL);
                    assert(previousToutput != NULL);
                    sprintf(channelName, "InternalChannel_T(%s to %s)",
                        previousToutput->GetTask()->GetTaskName(),
                        inputPorts[0]->GetTask()->GetTaskName());
                    m_ptaskGraph->AddInternalChannel(previousToutput, inputPorts[0], channelName);

                    assert(m_TInputChannel != NULL);
                    assert(previousToutput != NULL);
                    sprintf(channelName, "InternalChannel_T(%s to %s)",
                        previousToutput->GetTask()->GetTaskName(),
                        inputPorts[2]->GetTask()->GetTaskName());
                    m_ptaskGraph->AddInternalChannel(previousToutput, inputPorts[2], channelName);

                    // Connect U input.
                    assert(m_UInputChannel != NULL);
                    assert(previousUoutput != NULL);
                    sprintf(channelName, "InternalChannel_U(%s to %s)",
                        previousUoutput->GetTask()->GetTaskName(),
                        inputPorts[1]->GetTask()->GetTaskName());
                    m_ptaskGraph->AddInternalChannel(previousUoutput, inputPorts[1], channelName);

                    previousToutput = outputPorts[0];
                }
#pragma endregion
                } // if(k>0)
#endif
#pragma region Dnrm2_2
        {
            // Store eigenvalue of score for component k into L[k]
            // (overwriting temp value that was squatting there earlier in the inner (j) iteration)
            //   L[k] = cpublasDnrm2(M, &T[k*M], 1);
            const UINT inputPortCount = 3;
            const UINT outputPortCount = 1;
            Port ** inputPorts = new Port*[inputPortCount];
            Port ** outputPorts = new Port*[outputPortCount];

            inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_TTemplate, portUID++, "T_in", 0);
            inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_LTemplate, portUID++, "L_in", 1, 0);
            inputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, m_Dnrm2ParamsTemplate, portUID++, "PARAMS", 2);
            outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_LTemplate, portUID++, "L_out", 0);

            sprintf(taskName, "Dnrm2_2");
            Task * ptask = m_ptaskGraph->AddTask(ptaskDnrm2Kernel, 
                inputPortCount,
                inputPorts,
                outputPortCount,
                outputPorts,
                taskName);
            assert(ptask);

            ptask->SetComputeGeometry(M, N, 1);
            ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

            // Create and store graph input channel and datablock for params.
            GraphInputChannel * paramsChannel =
                m_ptaskGraph->AddInputChannel(inputPorts[2], "paramsConstChannel");
            DNRM2_PARAMS params;
            params.N = M;
            params.OFFSETX = k*M;
            params.OFFSETY = k;
            params.INCX = 1;
            Datablock * paramsDatablock =
                PTask::Runtime::AllocateDatablock(m_DcopyParamsTemplate, &params, sizeof(params), paramsChannel);
            AddParamsChannel(paramsChannel, paramsDatablock);

            // Connect T input.
            assert(m_TInputChannel != NULL);
            assert(previousToutput != NULL);
            sprintf(channelName, "InternalChannel_T(%s to %s)",
                previousToutput->GetTask()->GetTaskName(),
                inputPorts[0]->GetTask()->GetTaskName());
            m_ptaskGraph->AddInternalChannel(previousToutput, inputPorts[0], channelName);

            // Connect L input.
            assert(m_LInputChannel != NULL);
            assert(previousLoutput != NULL);
            sprintf(channelName, "InternalChannel_L(%s to %s)",
                previousLoutput->GetTask()->GetTaskName(),
                inputPorts[1]->GetTask()->GetTaskName());
            m_ptaskGraph->AddInternalChannel(previousLoutput, inputPorts[1], channelName);

            previousLoutput = outputPorts[0];
        }
#pragma endregion
#pragma region Dscal_2
        {
            // blasDscal(M, 1.0/L[k], &T[k*M], 1);
            const UINT inputPortCount = 3;
            const UINT outputPortCount = 1;
            Port ** inputPorts = new Port*[inputPortCount];
            Port ** outputPorts = new Port*[outputPortCount];

            inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_LTemplate, portUID++, "L_in", 0);
            inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_TTemplate, portUID++, "T_in", 1, 0);
            inputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, m_DscalParamsTemplate, portUID++, "PARAMS", 2);
            outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_TTemplate, portUID++, "T_out", 0);

            sprintf(taskName, "Dscal_2");
            Task * ptask = m_ptaskGraph->AddTask(ptaskDscalKernel, 
                inputPortCount,
                inputPorts,
                outputPortCount,
                outputPorts,
                taskName);
            assert(ptask);

            ptask->SetComputeGeometry(M, N, 1);
            ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

            // Create and store graph input channel and datablock for params.
            GraphInputChannel * paramsChannel =
                m_ptaskGraph->AddInputChannel(inputPorts[2], "paramsConstChannel");
            DSCAL_PARAMS params;
            params.N = M;
            params.ALPHA_CONST = 1.0;
            params.INCX = 1;
            params.OFFSETX = k*M;
            params.OFFSETALPHA_VAR = k;
            params.INVERT_ALPHA_VAR = true;
            Datablock * paramsDatablock =
                PTask::Runtime::AllocateDatablock(m_DscalParamsTemplate, &params, sizeof(params), paramsChannel);
            AddParamsChannel(paramsChannel, paramsDatablock);

            // Connect L input.
            assert(m_LInputChannel != NULL);
            assert(previousLoutput != NULL);
            sprintf(channelName, "InternalChannel_L(%s to %s)",
                previousLoutput->GetTask()->GetTaskName(),
                inputPorts[0]->GetTask()->GetTaskName());
            m_ptaskGraph->AddInternalChannel(previousLoutput, inputPorts[0], channelName);

            // Connect T input.
            assert(m_TInputChannel != NULL);
            assert(previousToutput != NULL);
            sprintf(channelName, "InternalChannel_T(%s to %s)",
                previousToutput->GetTask()->GetTaskName(),
                inputPorts[1]->GetTask()->GetTaskName());
            m_ptaskGraph->AddInternalChannel(previousToutput, inputPorts[1], channelName);

            previousToutput = outputPorts[0];
        }
#pragma endregion

                /* Not testing convergence at the moment
                if(fabs(a - L[k]) < er*L[k]) 
                {
                    // printf("  Done in %d iterations\n", j+1);
                    break;
                }
                a = L[k]; */
#endif // ifndef JC_DEBUG
        } // for(j)


#if 1
#pragma region Dger_1
        {
            // A = alpha * x * y' + A
            //   blasDger(M, N, - L[k], &T[k*M], 1, &P[k*N], 1, R, M);
            const UINT inputPortCount = 5;
#ifndef JC_DEBUG
            const UINT outputPortCount = 1;
#else
            const UINT outputPortCount = 2;
#endif
            Port ** inputPorts = new Port*[inputPortCount];
            Port ** outputPorts = new Port*[outputPortCount];

            inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_LTemplate, portUID++, "L_in", 0);
            inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_TTemplate, portUID++, "T_in", 1);
#ifndef JC_DEBUG
            inputPorts[2]	= PTask::Runtime::CreatePort(INPUT_PORT, m_PTemplate, portUID++, "P_in", 2);
#else
            inputPorts[2]	= PTask::Runtime::CreatePort(INPUT_PORT, m_PTemplate, portUID++, "P_in", 2, 1);
#endif
            inputPorts[3]	= PTask::Runtime::CreatePort(INPUT_PORT, m_RTemplate, portUID++, "R_in", 3, 0);
            inputPorts[4]	= PTask::Runtime::CreatePort(STICKY_PORT, m_DscalParamsTemplate, portUID++, "PARAMS", 4);
            outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_RTemplate, portUID++, "R_out", 0);
#ifdef JC_DEBUG
            outputPorts[1]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_PTemplate, portUID++, "P_out", 1);
#endif
            sprintf(taskName, "Dger_1");
            Task * ptask = m_ptaskGraph->AddTask(ptaskDgerKernel, 
                inputPortCount,
                inputPorts,
                outputPortCount,
                outputPorts,
                taskName);
            assert(ptask);

            ptask->SetComputeGeometry(M, N, 1);
            ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

            // Create and store graph input channel and datablock for params.
            GraphInputChannel * paramsChannel =
                m_ptaskGraph->AddInputChannel(inputPorts[4], "paramsConstChannel");
            DGER_PARAMS params;
            params.M = M;
            params.N = N;
            //params.ALPHA_CONST = -1.0;   Hardcoding for now to work around struct alignment issue.
            params.OFFSETALPHA_VAR = k;
            params.INCX = 1;
            params.INCY = 1;
            params.OFFSETX = k*M;
            params.OFFSETY = k*N;
            params.LDA = M;

            assert(M != 0);
/*            printf("*** DGER_PARAMS on host: M=%d, N=%d, ALPHA_CONST=%f, ALPHA_OFFSET=%d, INCX=%d, INCY=%d, OFFSETX=%d, OFFSETY=%d, OFFSETALPHA_VAR=%d, LDA=%d. SizeOf=%d\n", 
                params.M, params.N, params.ALPHA_CONST, params.OFFSETALPHA_VAR, params.INCX, params.INCY, params.OFFSETX, params.OFFSETY, params.OFFSETALPHA_VAR, 
                params.LDA, sizeof(DGER_PARAMS));
*/
            Datablock * paramsDatablock =
                PTask::Runtime::AllocateDatablock(m_DgerParamsTemplate, &params, sizeof(params), paramsChannel);
            AddParamsChannel(paramsChannel, paramsDatablock);

            // Connect L input.
//            assert(m_LInputChannel != NULL);
//            assert(previousLoutput != NULL);

            if (m_LInputChannel == NULL)
            {
                m_LInputChannel = m_ptaskGraph->AddInputChannel(inputPorts[0], "LInputChannel");
            } 
            else
            {
            sprintf(channelName, "InternalChannel_L(%s to %s)",
                previousLoutput->GetTask()->GetTaskName(),
                inputPorts[0]->GetTask()->GetTaskName());
            m_ptaskGraph->AddInternalChannel(previousLoutput, inputPorts[0], channelName);
            }

            // Connect T input.
//            assert(m_TInputChannel != NULL);
//            assert(previousToutput != NULL);

            if (m_TInputChannel == NULL)
            {
                m_TInputChannel = m_ptaskGraph->AddInputChannel(inputPorts[1], "TInputChannel");
            } 
            else
            {
            sprintf(channelName, "InternalChannel_T(%s to %s)",
                previousToutput->GetTask()->GetTaskName(),
                inputPorts[1]->GetTask()->GetTaskName());
            m_ptaskGraph->AddInternalChannel(previousToutput, inputPorts[1], channelName);
            }
            // Connect P input.
//            assert(m_PInputChannel != NULL);
//            assert(previousPoutput != NULL);

            if (m_PInputChannel == NULL)
            {
                m_PInputChannel = m_ptaskGraph->AddInputChannel(inputPorts[2], "PInputChannel");
            } 
            else
            {
            sprintf(channelName, "InternalChannel_P(%s to %s)",
                previousPoutput->GetTask()->GetTaskName(),
                inputPorts[2]->GetTask()->GetTaskName());
            m_ptaskGraph->AddInternalChannel(previousPoutput, inputPorts[2], channelName);
            }
            // Connect R input.
//            assert(m_RInputChannel != NULL);
//            assert(previousRoutput != NULL);

            if (m_RInputChannel == NULL)
            {
                m_RInputChannel = m_ptaskGraph->AddInputChannel(inputPorts[3], "RInputChannel");
            } 
            else
            {
            sprintf(channelName, "InternalChannel_R(%s to %s)",
                previousRoutput->GetTask()->GetTaskName(),
                inputPorts[3]->GetTask()->GetTaskName());
            m_ptaskGraph->AddInternalChannel(previousRoutput, inputPorts[3], channelName);
            }
            previousRoutput = outputPorts[0];
#ifdef JC_DEBUG
            previousPoutput = outputPorts[1];
#endif
        }
#pragma endregion
#endif
    } // for(k)
#if 1
    for(int k=0; k<K; k++)
    {
#pragma region Dscal_3
        {
            // blasDscal(M, L[k], &T[k*M], 1);
            const UINT inputPortCount = 3;
            const UINT outputPortCount = 1;
            Port ** inputPorts = new Port*[inputPortCount];
            Port ** outputPorts = new Port*[outputPortCount];

            inputPorts[0]	= PTask::Runtime::CreatePort(INPUT_PORT, m_LTemplate, portUID++, "L_in", 0);
            inputPorts[1]	= PTask::Runtime::CreatePort(INPUT_PORT, m_TTemplate, portUID++, "T_in", 1, 0);
            inputPorts[2]	= PTask::Runtime::CreatePort(STICKY_PORT, m_DscalParamsTemplate, portUID++, "PARAMS", 2);
            outputPorts[0]  = PTask::Runtime::CreatePort(OUTPUT_PORT, m_TTemplate, portUID++, "T_out", 0);

            sprintf(taskName, "Dscal_3");
            Task * ptask = m_ptaskGraph->AddTask(ptaskDscalKernel, 
                inputPortCount,
                inputPorts,
                outputPortCount,
                outputPorts,
                taskName);
            assert(ptask);

            ptask->SetComputeGeometry(M, N, 1);
            ptask->SetBlockAndGridSize(gridSize, threadBlockSize);

            // Create and store graph input channel and datablock for params.
            GraphInputChannel * paramsChannel =
                m_ptaskGraph->AddInputChannel(inputPorts[2], "paramsConstChannel");
            DSCAL_PARAMS params;
            params.N = M;
            params.ALPHA_CONST = 1.0;
            params.INCX = 1;
            params.OFFSETX = k*M;
            params.OFFSETALPHA_VAR = k;
            params.INVERT_ALPHA_VAR = false;
            Datablock * paramsDatablock =
                PTask::Runtime::AllocateDatablock(m_DscalParamsTemplate, &params, sizeof(params), paramsChannel);
            AddParamsChannel(paramsChannel, paramsDatablock);

            // Connect L input.
            assert(m_LInputChannel != NULL);
            assert(previousLoutput != NULL);
            sprintf(channelName, "InternalChannel_L(%s to %s)",
                previousLoutput->GetTask()->GetTaskName(),
                inputPorts[0]->GetTask()->GetTaskName());
            m_ptaskGraph->AddInternalChannel(previousLoutput, inputPorts[0], channelName);

            // Connect T input.
            assert(m_TInputChannel != NULL);
            assert(previousToutput != NULL);
            sprintf(channelName, "InternalChannel_T(%s to %s)",
                previousToutput->GetTask()->GetTaskName(),
                inputPorts[1]->GetTask()->GetTaskName());
            m_ptaskGraph->AddInternalChannel(previousToutput, inputPorts[1], channelName);

            previousToutput = outputPorts[0];
        }
#pragma endregion
    }
#endif

    // Create graph output channels for the outputs, connected to the outputs of the last ptask to emit them.
    m_TOutputChannel = m_ptaskGraph->AddOutputChannel(previousToutput, "TOutputChannel");
    m_POutputChannel = m_ptaskGraph->AddOutputChannel(previousPoutput, "POutputChannel");
    m_ROutputChannel = m_ptaskGraph->AddOutputChannel(previousRoutput, "ROutputChannel");
    //m_UOutputChannel = m_ptaskGraph->AddOutputChannel(previousUoutput, "UOutputChannel");

    m_ptaskGraph->WriteDOTFile("graphPCA.dot", true);
    // if(g_bSingleThreaded) printf("single-thread\n");
    m_ptaskGraph->Run(g_bSingleThreaded);
    // m_ptaskGraph->Run(TRUE);

    return true;
}

void PCAGraph::AddParamsChannel(PTask::GraphInputChannel* channel, PTask::Datablock* datablock)
{
    for(int i=0; i<m_numParamsChannels; i++) {
        if(m_ParamsInputChannels[i] == channel)
            assert(false);
    }
    m_ParamsInputChannels[m_numParamsChannels] = channel;
    m_ParamsDatablocks[m_numParamsChannels] = datablock;
    m_numParamsChannels++;
    assert(m_numParamsChannels < MAX_PARAMS_INPUTS);
}

double PCAGraph::RunTest(
    double* X, unsigned int M, unsigned int N, unsigned int K, 
    double* T, double* P, double* R, int iterations, bool pipelineInputs, bool preNorm)
{

    //
    // Push sticky inputs (that only need to submit once per run, not once per iteration).
    //
    for(int i=0; i<m_numParamsChannels; i++)
    {
        m_ParamsInputChannels[i]->Push(m_ParamsDatablocks[i]);
    }

    //
    // Iterate pushing inputs and pulling outputs.
    //

    CHighResolutionTimer * timer = new CHighResolutionTimer(gran_msec);
    timer->reset();
    double startTime = timer->elapsed(false);

    for (int iter=0; iter<iterations; iter++)
    {
        //
        // Input data blocks. Create each time, for fair comparison with CPU.
        // (Creating once and re-using also seemed to result in values changing between iterations. 
        // Not surprising, since InOut data-blocks are modified.)
        //

        // Initial values of T and P : Set to values passed in (which will be zeros).
        Datablock * pTInput = PTask::Runtime::AllocateDatablock(m_TTemplate, T, m_TTemplate->GetDatablockByteCount(), m_TInputChannel);
        Datablock * pPInput = PTask::Runtime::AllocateDatablock(m_PTemplate, P, m_PTemplate->GetDatablockByteCount(), m_PInputChannel);

        // Initial value of R : Set to value of X.
        Datablock * pRInput = PTask::Runtime::AllocateDatablock(m_RTemplate, X, m_RTemplate->GetDatablockByteCount(), m_RInputChannel);

        // Initial value of U : All 0.0 values.
        double* initialU = (double*)calloc(M, sizeof(double));
        assert(initialU != NULL);
        for (unsigned int i=0; i<M; i++) initialU[i] = 0.0;
        Datablock * pUInput = PTask::Runtime::AllocateDatablock(m_UTemplate, initialU, m_UTemplate->GetDatablockByteCount(), m_UInputChannel);

        // Initial value of L : All 0.0 values.
        double* initialL = (double*)calloc(K, sizeof(double));
        assert(initialL != NULL);
        for (unsigned int i=0; i<K; i++) initialU[i] = 0.0;
        Datablock * pLInput = PTask::Runtime::AllocateDatablock(m_LTemplate, initialL, m_LTemplate->GetDatablockByteCount(), m_LInputChannel);

        m_TInputChannel->Push(pTInput); pTInput->Release();
        m_PInputChannel->Push(pPInput); pPInput->Release();
        m_RInputChannel->Push(pRInput); pRInput->Release();
        if (m_LInputChannel) { m_LInputChannel->Push(pLInput); pLInput->Release(); }
        if (m_UInputChannel) { m_UInputChannel->Push(pUInput); pUInput->Release(); }

        if (!pipelineInputs)
        {
            //if (iter == 0) printf("not pipelining inputs\n");
            this->PullOutput(M, N, K, (iter == (iterations - 1)), T, P, R);
        }			
    }

    if (pipelineInputs)
    {
        for (int iter=0; iter<iterations; iter++)
        {
            //if (iter == 0) printf("pipelining inputs\n");
            this->PullOutput(M, N, K, (iter == (iterations - 1)), T, P, R);
        }
    }

    double endTime = timer->elapsed(false);
    double duration = endTime - startTime;
    double rate = calculate_rate("PTask", iterations, duration);

    m_ptaskGraph->Stop();
    m_ptaskGraph->Teardown();

    //JC	delete [] inputPorts;
    //JC	delete [] outputPorts;

    // JC Call more dtors, inc timer.

    return rate;
}

void PCAGraph::PullOutput(unsigned int M, unsigned int N, unsigned int K, bool lastIteration, double* T, double* P, double* R)
{
        Datablock * pTOutputDatablock = m_TOutputChannel->Pull();
        Datablock * pPOutputDatablock = m_POutputChannel->Pull();
        Datablock * pROutputDatablock = m_ROutputChannel->Pull();
        //Datablock * pUOutputDatablock = m_UOutputChannel->Pull();

        pTOutputDatablock->Lock();
        pPOutputDatablock->Lock();
        pROutputDatablock->Lock();
        //pUOutputDatablock->Lock();

        BOOL bWriteAccess = FALSE;
        double * TOutputData = (double*) pTOutputDatablock->GetDataPointer(bWriteAccess);
        double * POutputData = (double*) pPOutputDatablock->GetDataPointer(bWriteAccess);
        double * ROutputData = (double*) pROutputDatablock->GetDataPointer(bWriteAccess);
        //double * UOutputData = (double*) pUOutputDatablock->GetDataPointer(bWriteAccess);

        //output_matrix(stdout, "R_ptask", ROutputData, M, N);

        // If this is the last iteration, copy output back to provided matrix, for comparison with results of other methods.
        if (lastIteration)
        {
            memcpy(T, TOutputData, M * K * sizeof(double));
            memcpy(P, POutputData, N * K * sizeof(double));
            memcpy(R, ROutputData, M * N * sizeof(double));
        }
        pTOutputDatablock->Unlock();
        pPOutputDatablock->Unlock();
        pROutputDatablock->Unlock();
        //pUOutputDatablock->Unlock();

        pTOutputDatablock->Release();
        pPOutputDatablock->Release();
        pROutputDatablock->Release();
        //pUOutputDatablock->Release();
}
