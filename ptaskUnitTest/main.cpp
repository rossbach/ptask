//--------------------------------------------------------------------------------------
// File: main.cpp
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif
#include <stdio.h>
#include <assert.h>
#include "matmul.h"
#include "clvectoradd.h"
#include "sort.h"
#include "graphsort.h"
#include "graphmatmul.h"
#include "graphhostmm.h"
#include "ptaskcublas.h"
#include "graphmdmatmul.h"
#include "graphmatmulraw.h"
#include "graphemgmm.h"
#include "graphcuadd.h"
#include "graphcladd.h"
#include "graphcupca.h"
#include "graphbench.h"
#include "select.h"
#include "pfxsum.h"
#include "cpfxsum.h"
#include "graphfdtd.h"
#include "graphinout.h"
#include "dxinout.h"
#include "switchports.h"
#include "deferredports.h"
#include "descportsout.h"
#include "gatedports.h"
#include "graphinitports.h"
#include "graphmetaports.h"
#include "graphhostmetaports.h"
#include "graphgroupby.h"
#include "tee.h"
#include "controlpropagation.h"
#include "channelpredication.h"
#include "iteration.h"
#include "graphinitializerchannels.h"
#include "permanentblocks.h"
#include "bufferdimsdesc.h"
#include "primitive_types.h"
#include "rttests.h"
#include "platformcheck.h"
#include "Tracer.h"
#include "confighelpers.h"
#include "pipelinestresstest.h"

static const int COOPERATIVE = 0;
static const int PRIORITY = 1;
static const int PATHBUFSIZE = 4096;
static char g_sztask[PATHBUFSIZE];
static char g_szfile[PATHBUFSIZE];
static char g_szshader[PATHBUFSIZE];
static char g_szxformdata[PATHBUFSIZE];
static char g_szxformout[PATHBUFSIZE];
static char g_szselectdata[PATHBUFSIZE];
char g_szCompilerOutputBuffer[COBBUFSIZE];
BOOL g_verbose = FALSE;
BOOL g_fileset = FALSE;
BOOL g_shaderset = FALSE;
BOOL g_taskset = FALSE;
BOOL g_useOpenCL = FALSE;
BOOL g_bUseGraph = FALSE;
BOOL g_bSingleThreaded = FALSE;
BOOL g_bExtremeTrace = FALSE;
BOOL g_bForceSynchronous = FALSE;
BOOL g_bSetICMaterializationPolicy = FALSE;
BOOL g_bSetOCMaterializationPolicy = FALSE;
BOOL g_bThreadPoolSignalPerThread = TRUE;
BOOL g_bSetThreadPoolSignalPolicy = FALSE;
BOOL g_bExitOnFailure = TRUE;
BOOL g_bSetAllowAppThreadPrimaryCtxt = FALSE;
BOOL g_bAllowAppThreadPrimaryCtxt = TRUE;
PTask::Runtime::VIEWMATERIALIZATIONPOLICY g_eICPolicy;
PTask::Runtime::VIEWMATERIALIZATIONPOLICY g_eOCPolicy;
int g_cols = 320;
int g_rows = 200;
int g_nInstances = 1;
int g_nIterations = 1;
BOOL g_bPipelineInputs = FALSE;
int g_nInnerIterations = 10;
int g_nComponents = 3;
UINT g_nElements = 100;
int g_depthMin=1;       // -z
int g_depthMax=5;       // -Z
int g_depthStep=1;      // -x
int g_bredthMin=1;      // -y
int g_bredthMax=5;      // -Y
int g_bredthStep=1;     // -X
int g_dimensionMin=32;  // -w
int g_dimensionMax=256; // -W
int g_iterations=100;   // -i
int g_gtype=0;          // -T
bool g_instrument=false;// -I
int* g_ptaskPriorities=NULL;         // -p  Default = { 5 }
int* g_threadPriorities=NULL;        // -P  Default = { -1 }
int g_nSchedulingMode = PTask::Runtime::SCHEDMODE_DATADRIVEN; // -m
int g_serializationMode = 0;        // -S : See usage for details
int g_nMaximumConcurrency = 0;       // -C, 0 means no limit
BOOL g_threadPrioritiesSet = FALSE;  //  thread priorities for graphbench
BOOL g_bSetChannelCapacity = FALSE;
int g_nChannelCapacity = 0;
BOOL g_bRunCPUReference=TRUE;
int COMPILER_BUF_SIZE = 1024;
char g_szCompilerOutput[1024];
BOOL g_bSetBlockPoolSize = FALSE;
int g_nBlockPoolSize = 0;
BOOL g_bLogDispatches = FALSE;
BOOL g_bDispatchTracingEnabled = FALSE;
BOOL g_bSetThreadPoolPolicy = FALSE; // true to set the thread pool policy in ptask
THREADPOOLPOLICY g_eThreadPoolPolicy = TPP_AUTOMATIC; // default policy
BOOL g_bSetThreadPoolSize = FALSE; // true if options set the pool size
UINT g_uiThreadPoolSize = 0; // means runtime chooses!
UINT g_uiThreadPoolRunQThreshold = 0; // graphs bigger than 32 
BOOL g_bSetTPRQThreshold = FALSE; // true if options set the thread pool threshold (g_uiThreadPoolRunQThreshold)
BOOL g_bTaskProfileMode = FALSE;
BOOL g_bSetSchedulerThreads = FALSE;
UINT g_uiSchedulerThreads = 1;
bool g_bVerify = false;
bool g_bCopyback = false;

void 
usage(
    char ** argv
    )
{
    printf("\nUsage for %s:\n", argv[0]);
    printf("%s [options]\n\n", argv[0]);
    printf("     -t (task)   : task to run [matadd|mataddraw|matmul|multi|scheduler]\n");
    printf("                     multi option runs parallel instances \n");
    printf("                     of the given shader, operation\n");
    printf("     -f (file)   : file (containing shader programs, current list below:) \n");
    printf("                   matrixadd.hlsl \n");
    printf("                   matrixaddraw.hlsl \n");
    printf("                   matrixmul.hlsl \n");
    printf("                   streamadd.cl \n");
    printf("     -d (dir)    : directory containing input files for xform task\n");
    printf("     -o (file)   : output file (bmp) for xform task\n");
    printf("     -s (shader) : shader to invoke\n");
    printf("     -r (rows)   : row size for matrix operations\n");
    printf("     -c (cols)   : col size for matrix operations\n");
    printf("     -n (num)    : number of parallel instances to run\n");
    printf("     -i (iter)   : number of iterations for each parallel instance to run\n");
    printf("     -L (flag)   : Choose whether to pipeline all inputs (1) or submit them sequentially, after the previous output is read (1)\n");
    printf("     -j (iter)   : number of inner iterations to run within each iteration (for tasks that have such inner iteration)\n");
    printf("     -K (comp)   : number of PCA components to produce\n");
    printf("     -v          : verbose output\n");
    printf("     -O          : use OpenCL accelerators as well as DirectX\n");
    printf("     -G          : use graph+tasks instead of ptasks...(temporary debug feature)\n");
    printf("     -z (dmin)   : depth min for graph bench (default == 1)\n");
    printf("     -Z (dmax)   : depth max for graph bench (default == 5)\n");
    printf("     -x (dstep)  : depth step for graph bench (default == 1)\n");
    printf("     -y (bmin)   : breadth min for graph bench (default == 1)\n");
    printf("     -Y (bmax)   : breadth max for graph bench (default == 5)\n");
    printf("     -X (bstep)  : breadth step for graph bench (default == 1)\n");
    printf("     -w (dimmin) : dimension min for graph bench (default == 32, must be a power of 2)\n");
    printf("     -W (dimmax) : dimension max for graph bench (default == 256, must be a power of 2)\n");
    printf("     -T (type)   : graph type for graph bench (0==RECT, 1=BINTREE)\n");
    printf("     -I          : instrument (profile) graph bench\n");
    printf("     -p (prios)  : PTask priorities of instances of graphs to run in graph bench (default = 5). Comma separated list, no spaces.\n");
    printf("     -P (prios)  : thread priorities of instances of graphs to run in graph bench (default = -1 = not set). Comma separated list, no spaces.\n");
    printf("                   Setting -p and/or -P implicitly sets -n to the number of values specified.\n");
    printf("                   -p has to be set if -P is, but not v.v. If both are set, number of prioroties must match.\n");
    printf("     -m (mode)   : set scheduling mode (0==COOPERATIVE, 1==PRIORITY, 2==DATADRIVEN, 3==FIFO ... -1=Sweep all modes, 5 runs of each)\n");
    printf("     -S (mode)   : Graph serialization mode\n");
    printf("                   0 (default) : Don't serialize graph. Run the test case using graph constructed programmatically.\n");
    printf("                   1           : Construct and serialize graph to XML file. Exit without running test case against the graph.\n");
    printf("                   2           : Deserialize graph from XML file and run test case against it instead of constructing graph programatically.\n");
    printf("     -C (conc)   : set maximum concurrency in GPUs. 0->no limit, 1->use only 1 GPU (even if there are more), etc.\n");
    printf("     -J          : single-threaded runtime\n");
    printf("     -a (cap)    : default channel capacity\n");
    printf("     -R          : do not run CPU reference workload (PCA only)\n");
    printf("     -U (size)   : set the default output port block pool size\n");
    printf("     -l          : log dispatches to the console\n");
    printf("     -A          : enable API tracing (requires that PTask is built with EXTREME_TRACE)\n");
    printf("     -F          : enable \"force synchronous mode\" in PTask, which syncs device contexts after sensitive ops\n");
    printf("     -B          : enable dispatch tracing (ETW)\n");
    printf("     -Q policy   : sets thread pool policy: 0->automatic, 1->explicit, 2->force task-per-thread\n");
    printf("     -k sigpol   : sets the signalling policy for worker threads: 1->per-thread signalling/random select, 0->all threads use the same event\n");
    printf("     -q size     : sets the thread pool size (requires explicit policy--see\"-Q\")\n");
    printf("     -u thresh   : sets the threshold (graph size) at which ptask will change from 1:N threads:tasks to M:N\n");
    printf("     -H threads  : sets the number of scheduler threads\n");
    printf("     -h          : enable task profile mode\n");
    printf("     -E          : copy back data on each iteration (depends on test scenario support)\n");
    printf("     -V          : verify on each iteration (depends on test scenario support)\n");
    printf("     -g policy   : (policy->\"DEMAND\"|\"EAGER\"): sets the default view materialization policy for channels other than output channels\n");
    printf("     -b policy   : (policy->\"DEMAND\"|\"EAGER\"): sets the default view materialization policy for *output* channels\n");
    printf("     -M [0|1]    : 1=>application threads have default device context, 0=>no default context on app threads\n");

    // available!
    //    printf("     -E          : emulate modular\n");

    printf("\n\nCommon command lines:\n\n");
    printf(
        "sort:\t\t"
        "-G -t sort -f sort.hlsl -s sort -r 512 -c 512 -n 1 -i 1\n"
        "select:\t\t"
        "-t select -f select.hlsl -D selectdata.txt -s main \n"
        "pfxsum:\t\t"
        "-t pfxsum -f pfxsum.hlsl -s main \n"
        "graphbench:\t"
        "-t graphbench -f matrixmul.hlsl -s op -z 1 -Z 6 -x 2 -y 1 -Y 6 -X 2 -w 32 -W 256 -i 50 -T 0\n"
        "matmul:\t\t"
        "-G -t matmul -f matrixmul.hlsl -s op -r 64 -c 64 -n 1 -i 1\n"
        "using byte-addressable buffers:\n"
        "matmulraw:\t"
        "-G -t matmulraw -f matrixmul_4ch.hlsl -s op -r 16 -c 16 -n 1 -i 1\n"
        "using CUDA:\n"
        "vector add:\t"
        "-G -t cuvecadd -f vectorAddC2.ptx -s VecAdd -r 50000 -c 1 -n 1 -i 1\n"
        );
}

#pragma warning(disable:4996)
int
set_priorities(int*& outputArray, char* inputString)
{
    int numPriorities = 0;
    char* inputCopy = new char[strlen(inputString)+1];
    char* tok;

    // Free any existing output array.
    if(outputArray != NULL) 
        delete [] outputArray;

    // Parse the string once to determine the number of tokens.
    strcpy(inputCopy, inputString);
    tok = strtok(inputCopy, ",");
    while (tok != NULL)
    {
        numPriorities++;
        tok = strtok(NULL, ",");
    }

    // Parse the string again, converting the tokens to signed integers.
    // Store them in a new array.
    outputArray = new int[numPriorities];
    strcpy(inputCopy, inputString);
    int currentIndex = 0;
    tok = strtok(inputCopy, ",");
    while (tok != NULL)
    {
        int val = atoi(tok);
        outputArray[currentIndex++] = val;
        tok = strtok(NULL, ",");
    }

    delete [] inputCopy;
    return numPriorities;
#pragma warning(default:4996)
}

class GetOpt
{
    int m_iarg; // argument number we are on
    std::string m_options;
    int m_argc;
    char **m_argv;
public:
    GetOpt(int argc, char *argv[], std::string options) :
        m_argc(argc), m_argv(argv), m_iarg(1), m_options(options)
    {}

    // get the next option and option argument
    // optarg - argument for this option
    // returns: character that is the argument
    char GetNext(char*& optarg)
    {
        // start out assuming no parameter
        optarg = nullptr;
        // check to see if we are past the last argument
        if (m_iarg >= m_argc) 
            return 0;
        std::string arg = m_argv[m_iarg++];
        // need an argument identifier, accept '/' or '-'
        if (arg[0] != '-' && arg[0] != '/')
            return '?';

        int c = arg[1];

        // see if it's a known option
        auto pos = m_options.find_first_of(c);

        // option not found in our list, return '?'
        if (pos == std::string::npos)
            return '?';

        // see if this option takes a parameter
        if (m_options.size() > pos+1 && m_options[pos+1] == ':')
        {
            // check to see if we are past the last argument
            if (m_iarg >= m_argc) 
                return 0;
            // it takes a parameter, so return the next argument as a parameter
            optarg = m_argv[m_iarg++];
        }
        return c;
    }
};

BOOL 
get_options(
    int argc, 
    char * argv[]
    )  
{
    if(argc < 2) {
        return FALSE;
    }
    char c = 0;
    int dimtmp = 0;
    int bitcount = 0;
    int numPriorities;
    int nPolicy;
    GetOpt getopt(argc, argv, "a:Ab:Bc:C:d:D:e:Ef:Fg:GhH:i:Ij:Jk:K:lL:m:M:n:o:p:P:Q:q:r:Rs:S:t:T:u:U:vw:W:x:X:y:Y:z:Z:V");
    char* optarg;
    while(c=getopt.GetNext(optarg)) {
        switch(c) {	
        case 'M':
            if(optarg == NULL) {
                printf("\n-b requires an integer-valued application thread default context parameter (0,1)\n");
                return FALSE;
            }
            g_bSetAllowAppThreadPrimaryCtxt = TRUE;
            g_bAllowAppThreadPrimaryCtxt = atoi(optarg) != 0;
            break;
        case 'k':
            if(optarg == NULL) {
                printf("\n-k requires an integer-valued thread-pool signal policy\n");
                return FALSE;
            }
            g_bSetThreadPoolSignalPolicy = TRUE;
            g_bThreadPoolSignalPerThread = atoi(optarg) != 0;
            break;
        case 'g':
            if(optarg == NULL) {
                printf("\n-g requires a policy argument (\"DEMAND\" or \"EAGER\")\n");
                return FALSE;
            }
            g_bSetICMaterializationPolicy = TRUE;
            if(!strcmp(optarg, "DEMAND") || !strcmp(optarg, "\"DEMAND\"")) {
                g_eICPolicy = VIEWMATERIALIZATIONPOLICY_ON_DEMAND;
            } else if(!strcmp(optarg, "EAGER") || !strcmp(optarg, "\"EAGER\"")) {
                g_eICPolicy = VIEWMATERIALIZATIONPOLICY_EAGER;
            } else {
                printf("unknown view materialization policy: %s\n", optarg);
                return FALSE;
            }
            break;
        case 'b':
            if(optarg == NULL) {
                printf("\n-b requires a policy argument (\"DEMAND\" or \"EAGER\")\n");
                return FALSE;
            }
            g_bSetOCMaterializationPolicy = TRUE;
            if(!strcmp(optarg, "DEMAND") || !strcmp(optarg, "\"DEMAND\"")) {
                g_eOCPolicy = VIEWMATERIALIZATIONPOLICY_ON_DEMAND;
            } else if(!strcmp(optarg, "EAGER") || !strcmp(optarg, "\"EAGER\"")) {
                g_eOCPolicy = VIEWMATERIALIZATIONPOLICY_EAGER;
            } else {
                printf("unknown view materialization policy: %s\n", optarg);
                return FALSE;
            }
            break;
        case 'h':
            g_bTaskProfileMode = TRUE;
            break;
        case 'E':
            g_bCopyback = true;
            break;
        case 'V':
            g_bVerify = true;
            break;
        case 'H':
            if(optarg == NULL) {
                printf("\n-H requires an integer-valued scheduler thread count\n");
                return FALSE;
            }
            g_bSetSchedulerThreads = TRUE;
            g_uiSchedulerThreads = atoi(optarg);
            break;
        case 'Q':
            if(optarg == NULL) {
                printf("\n-Q requires an integer-valued policy argument (0->automatic, 1->explicit)\n");
                return FALSE;
            }
            g_bSetThreadPoolPolicy = TRUE;
            nPolicy = atoi(optarg);
            switch(nPolicy) {
            case 0: g_eThreadPoolPolicy = TPP_AUTOMATIC;     break;
            case 1: g_eThreadPoolPolicy = TPP_EXPLICIT;      break;
            case 2: g_eThreadPoolPolicy = TPP_THREADPERTASK; break;
            default:
                printf("unknown thread pool policy: %d\n", nPolicy);
                return FALSE;
            }
            break;
        case 'q':
            if(optarg == NULL) {
                printf("\n-q requires an integer-valued thread pool size\n");
                return FALSE;
            }
            g_bSetThreadPoolSize = TRUE;
            g_uiThreadPoolSize = atoi(optarg);
            break;
        case 'u':
            if(optarg == NULL) {
                printf("\n-q requires an integer-valued graph size threshold\n");
                return FALSE;
            }
            ::g_bSetTPRQThreshold = TRUE;
            g_uiThreadPoolRunQThreshold = atoi(optarg);
            break;
        case 'F':
            g_bForceSynchronous = TRUE;
            break;
        case 'A':
            g_bExtremeTrace = TRUE;
            break;
        case 'l':
            g_bLogDispatches = TRUE;
            break;
        case 'B':
            g_bDispatchTracingEnabled = TRUE;
            break;
        case 'I':
            g_instrument = true;
            break;
        case 'J':
            g_bSingleThreaded = TRUE;
            break;
        case 'a':
            if(optarg == NULL) {
                printf("\n-a requires an integer-valued channel capacity (0->no limit!\n\n"); 
                return FALSE;
            }
            g_bSetChannelCapacity = TRUE;
            g_nChannelCapacity = atoi(optarg);
            break;		
        case 'C':
            if(optarg == NULL) {
                printf("\n-C requires an integer-valued max GPU count (0->no limit!\n\n"); 
                return FALSE;
            }
            g_nMaximumConcurrency = atoi(optarg);
            break;			
        case 'm':
            if(optarg == NULL) {
                printf("\n-m requires an integer-valued scheduling mode parameter (0->COOP, 1->PRIO)!\n\n"); 
                return FALSE;
            }
            g_nSchedulingMode = atoi(optarg);
            break;			
        case 'S':
            if(optarg == NULL) {
                printf("\n-S requires an integer-valued serialization mode flag. (See usage for values)!\n\n"); 
                return FALSE;
            }
            g_serializationMode = atoi(optarg);
            break;			
        case 'z':
            if(optarg == NULL) {
                printf("\n-z requires an integer-valued depth min parameter!\n\n"); 
                return FALSE;
            }
            g_depthMin = atoi(optarg);
            break;			
        case 'Z':
            if(optarg == NULL) {
                printf("\n-Z requires an integer-valued depth max parameter!\n\n"); 
                return FALSE;
            }
            g_depthMax = atoi(optarg);
            break;			
        case 'y':
            if(optarg == NULL) {
                printf("\n-y requires an integer-valued breadth min parameter!\n\n"); 
                return FALSE;
            }
            g_bredthMin = atoi(optarg);
            break;			
        case 'Y':
            if(optarg == NULL) {
                printf("\n-Y requires an integer-valued breadth max parameter!\n\n"); 
                return FALSE;
            }
            g_bredthMax = atoi(optarg);
            break;			
        case 'x':
            if(optarg == NULL) {
                printf("\n-x requires an integer-valued depth step parameter!\n\n"); 
                return FALSE;
            }
            g_depthStep = atoi(optarg);
            break;			
        case 'X':
            if(optarg == NULL) {
                printf("\n-X requires an integer-valued breadth step parameter!\n\n"); 
                return FALSE;
            }
            g_bredthStep = atoi(optarg);
            break;			
        case 'w':
            if(optarg == NULL) {
                printf("\n-w requires an integer-valued min grid dimension parameter (must be power of 2)!\n\n"); 
                return FALSE;
            }
            g_dimensionMin = atoi(optarg);
            dimtmp = g_dimensionMin;
            bitcount = 0;
            while(dimtmp) {
                if(dimtmp & 0x1)
                    bitcount++;
                dimtmp >>= 1;
            }
            if(bitcount != 1) {
                printf("dimension parameters must be a power of 2. Min dimension defaulting to 32\n");
                g_dimensionMin = 32;
            }
            break;			
        case 'W':
            if(optarg == NULL) {
                printf("\n-W requires an integer-valued max grid dimension parameter (must be power of 2)!\n\n"); 
                return FALSE;
            }
            g_dimensionMax = atoi(optarg);
            dimtmp = g_dimensionMax;
            bitcount = 0;
            while(dimtmp) {
                if(dimtmp & 0x1)
                    bitcount++;
                dimtmp >>= 1;
            }
            if(bitcount != 1) {
                printf("dimension parameters must be a power of 2. Max dimension defaulting to 256\n");
                g_dimensionMax = 256;
            }
            break;			
        case 'T':
            if(optarg == NULL) {
                printf("\n-T requires an integer-valued graph type parameter (0==RECT, 1=BINTREE)!\n\n"); 
                return FALSE;
            }
            g_gtype = atoi(optarg);
            if(g_gtype != 0 && g_gtype != 1) {
                printf("\n%d is not a valid graph type parameter (0==RECT, 1=BINTREE)!\n\n", g_gtype); 
                return FALSE;
            }
            break;			
        case 'G':
            g_bUseGraph = TRUE;
            break;
        case 'o':
            if(optarg == NULL) {
                printf("\n-o requires a path to write a bmp file!\n\n"); 
                return FALSE;
            }			
            strcpy_s(g_szxformout, PATHBUFSIZE, optarg);
            break;
        case 'd':
            if(optarg == NULL) {
                printf("\n-d requires a directory!\n\n"); 
                return FALSE;
            }			
            strcpy_s(g_szxformdata, PATHBUFSIZE, optarg);
            break;
        case 'e':
            if(optarg == NULL) {
                printf("\n-e requires an integer-valued array length (for streamadd)!\n\n"); 
                return FALSE;
            }
            g_nElements = (UINT) atoi(optarg);
            break;			
        case 'i':
            if(optarg == NULL) {
                printf("\n-i requires an integer-valued iteration count parameter!\n\n"); 
                return FALSE;
            }
            g_nIterations = atoi(optarg);
            break;			
        case 'L':
            if(optarg == NULL) {
                printf("\n-S requires an integer-valued pipeline input flag (0->NO (default), 1->YES)!\n\n"); 
                return FALSE;
            }
            g_bPipelineInputs = atoi(optarg) != 0;
            break;			
        case 'j':
            if(optarg == NULL) {
                printf("\n-j requires an integer-valued inner iteration count parameter!\n\n"); 
                return FALSE;
            }
            g_nInnerIterations = atoi(optarg);
            break;			
        case 'K':
            if(optarg == NULL) {
                printf("\n-K requires an integer-valued number of PCA components to discover!\n\n"); 
                return FALSE;
            }
            g_nComponents = atoi(optarg);
            break;
        case 'n':
            if(optarg == NULL) {
                printf("\n-n requires an integer-valued instance count parameter!\n\n"); 
                return FALSE;
            }
            g_nInstances = atoi(optarg);
            break;			
        case 'p':
            if(optarg == NULL) {
                printf("\n-p requires a comma separated list of ptask priorities for graph instances!\n\n"); 
                return FALSE;
            }
            g_nInstances = set_priorities(g_ptaskPriorities, optarg);
            break;			
        case 'P':
            if(optarg == NULL) {
                printf("\n-P requires a comma separated list of thread priorities for graph instances!\n\n"); 
                return FALSE;
            }
            numPriorities = set_priorities(g_threadPriorities, optarg);
            g_threadPrioritiesSet = TRUE;
            if (numPriorities != g_nInstances)
            {
                printf("\n-P must specify the same number of priorities as -p!\n\n"); 
                return FALSE;
            }
            break;			
        case 'r':
            if(optarg == NULL) {
                printf("\n-r requires an integer-valued row parameter!\n\n"); 
                return FALSE;
            }
            g_rows = atoi(optarg);
            break;			
        case 'c':
            if(optarg == NULL) {
                printf("\n-c requires an integer-valued column parameter!\n\n"); 
                return FALSE;
            }
            g_cols = atoi(optarg);
            break;			
        case 't':
            if(optarg == NULL) {
                printf("\n-t requires a task!\n\n"); 
                return FALSE;
            }			
            strcpy_s(g_sztask, PATHBUFSIZE, optarg);
            g_taskset = TRUE;
            break;
        case 'U':
            if(optarg == NULL) {
                printf("\n-U requires an integer-valued block pool size!\n\n"); 
                return FALSE;
            }
            g_nBlockPoolSize = atoi(optarg);
            g_bSetBlockPoolSize = TRUE;
            break;			
        case 'f':
            if(optarg == NULL) {
                printf("\n-s requires a path to an HLSL File!\n\n"); 
                return FALSE;
            }			
            strcpy_s(g_szfile, PATHBUFSIZE, optarg);
            g_fileset = TRUE;
            break;
        case 'D':
            if(optarg == NULL) {
                printf("\n-D requires a path to a K-V pair File!\n\n"); 
                return FALSE;
            }			
            strcpy_s(g_szselectdata, PATHBUFSIZE, optarg);
            break;
        case 's':
            if(optarg == NULL) {
                printf("\n-s requires a shader function!\n\n"); 
                return FALSE;
            }			
            strcpy_s(g_szshader, PATHBUFSIZE, optarg);
            g_shaderset = TRUE;
            break;
        case 'v': 			
            g_verbose = TRUE;
            break;
        case 'O': 			
            g_useOpenCL = TRUE;
            break;
        case 'R':
            g_bRunCPUReference = FALSE;
            break;
        default:			printf("Unknown command line switch: %c", c);			
            return FALSE;
        }
    }
    return TRUE;
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Check compile success. </summary>
///
/// <remarks>   Crossbac, 7/12/2012. </remarks>
///
/// <param name="lpvKernel">    [in,out] If non-null, the lpv kernel. </param>
///
/// <returns>   true if it succeeds, false if it fails. </returns>
///-------------------------------------------------------------------------------------------------

BOOL
CheckCompileSuccess(
    char * szfile,
    char * szshader,
    void * lpvKernel
    )
{
    if(lpvKernel == NULL) {
        printf("failed to compile %s\\%s\n", szfile, szshader);
        printf("compiler output:\n\n%s\n\n", g_szCompilerOutputBuffer);
        exit(2);
        return FALSE;
    }
    return TRUE;
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Check platform support. </summary>
///
/// <remarks>   Crossbac, 7/12/2012. </remarks>
///
/// <param name="szShaderFile"> [in,out] If non-null, the shader file. </param>
///
/// <returns>   true if it succeeds, false if it fails. </returns>
///-------------------------------------------------------------------------------------------------

BOOL
CheckPlatformSupport(
    char * szShaderFile,
    char * szShaderOp
    )
{
    BOOL bInitialized = PTask::Runtime::IsInitialized();
    if(!bInitialized) {
        printf("Error: checking platform support before runtime initialization!\n");
        exit(1);
    }
    BOOL bPlatformSupport = PTask::Runtime::CanExecuteKernel(szShaderFile);
    if(bPlatformSupport) {
        printf("PTask support for %s OK\n", g_szshader);
        return TRUE;
    } else {
        printf("\n"
               "\tPTask found no accelerators that can execute %s\n"
               "\tCheck that you have called Runtime::SetUseXXX(TRUE) if you believe the system has the right hardware.\n"
               "\tIf you are using RDP, recall that you will be unable to use non-TCC driver CUDA systems.\n"
               "\tIf you are using Tesla cards, recall that they are not video cards and cannot be found by DirectX.\n"
               "\tIf you need to run DirectX workloads and cannot find a card, try calling SetUseReferenceDrivers.\n"
               , g_szshader);
        if(bInitialized) {
            PTask::Runtime::Terminate();
        }
        exit(0);
    }
    return FALSE;
}

//--------------------------------------------------------------------------------------
// Entry point to the program
//--------------------------------------------------------------------------------------
int __cdecl main(int argc, char ** argv)
{
#if defined(DEBUG) || defined(_DEBUG)
    // Enable run-time memory check for debug builds.
    // also, note handy tool (commented out) for breaking on a particular
    // allocation number once you've isolated a leaked alloc
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
    //int MYALLOCNUMBER=1392;//whatever
    //_CrtSetBreakAlloc(MYALLOCNUMBER);
#endif    

    // Set default values for options that cannot be set by static initializers.
    // Note these are used only by the graphbench benchmark.
    
    set_priorities(g_ptaskPriorities, "5");
    set_priorities(g_threadPriorities, "-1");
    
    if(!get_options(argc, argv)) {
        usage(argv);
        return -1;
    }

    if(!g_taskset) {
        if(!g_fileset) {
            printf("please specify an HLSL file!\n");
            return -2;
        }
        if(!g_shaderset) {
            printf("please specify a shader function within %s!\n", g_szfile);
            return -3;
        }
    }
    if (g_verbose)
    {
        for(int m=0; m<argc; m++) {
            printf("%s ", argv[m]);
        }
        printf("\n");
    }

    // -------------------------------------------------
    // If -p was specified, but -P was not, make 
    // length of thread priority array 
    // match that of PTask priority array.
    // -------------------------------------------------
    if (!g_threadPrioritiesSet && g_nInstances > 1)
    {
        delete [] g_threadPriorities;
        g_threadPriorities = new int[g_nInstances];
        for (int i=0; i<g_nInstances; i++)
        {
            g_threadPriorities[i] = -1;
        }
    }

    // cjr: hack: 1/22/13: disabling OpenCL support temporarily
    // PTask::Runtime::SetUseOpenCL(FALSE);

    std::cout << std::endl << std::endl;
    CONFIGUREPTASKI(Verbose, TRUE, g_verbose);
    CONFIGUREPTASK(MaximumConcurrency, true, g_nMaximumConcurrency);
    CONFIGUREPTASK(ThreadPoolSignalPerThread, g_bSetThreadPoolSignalPolicy, g_bThreadPoolSignalPerThread);
    CONFIGUREPTASKE(DefaultViewMaterializationPolicy, g_bSetICMaterializationPolicy, g_eICPolicy, vptostr);
    CONFIGUREPTASKE(DefaultOutputViewMaterializationPolicy, g_bSetOCMaterializationPolicy, g_eOCPolicy, vptostr);
    CONFIGUREPTASK(DefaultChannelCapacity, g_bSetChannelCapacity, g_nChannelCapacity);
    CONFIGUREPTASK(ICBlockPoolSize, g_bSetBlockPoolSize, g_nBlockPoolSize);
    CONFIGUREPTASK(DispatchLoggingEnabled, g_bLogDispatches, g_bLogDispatches);
    CONFIGUREPTASK(DispatchTracingEnabled, g_bDispatchTracingEnabled, g_bDispatchTracingEnabled);
    CONFIGUREPTASK(ForceSynchronous, g_bForceSynchronous, g_bForceSynchronous);
    CONFIGUREPTASK(ExtremeTraceMode, g_bExtremeTrace, g_bExtremeTrace);
    CONFIGUREPTASKE(TaskThreadPoolPolicy, g_bSetThreadPoolPolicy, g_eThreadPoolPolicy, tpptostr);
    CONFIGUREPTASK(TaskThreadPoolSize, g_bSetThreadPoolSize, g_uiThreadPoolSize);
    CONFIGUREPTASK(SchedulerThreadPerTaskThreshold, g_bSetTPRQThreshold, g_uiThreadPoolRunQThreshold);
    CONFIGUREPTASK(SchedulerThreadCount, g_bSetSchedulerThreads, g_uiSchedulerThreads);
    CONFIGUREPTASK(UseGraphMonitorWatchdog, TRUE, FALSE);
    CONFIGUREPTASK(TrackDeviceMemory, TRUE, FALSE);
    CONFIGUREPTASK(TaskProfileMode, g_bTaskProfileMode, g_bTaskProfileMode);
    CONFIGUREPTASK(ExitOnRuntimeFailure, g_bExitOnFailure, g_bExitOnFailure);    
    CONFIGUREPTASK(ApplicationThreadsManagePrimaryContext, g_bSetAllowAppThreadPrimaryCtxt, g_bAllowAppThreadPrimaryCtxt);       

    // -----------------------------------------------------------
    // NB: we currently aren't supporting runs that don't use a graph. 
    // eventually, we will want to be able to manipulate ptasks
    // directly, but none of the current unit tests work this way.
    // -----------------------------------------------------------
    int res = 0;
    if(!strcmp("sort", g_sztask)) {
        // -G -t sort -f c:\SVC\Dandelion\accelerators\PTaskUnitTest\sort.hlsl -s sort -r 512 -c 512 -n 1 -i 1
        res = run_graph_sort_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("graphemgmm", g_sztask)) {
        // command line: Jon?
        res = run_graph_emgmm_task(g_szfile, g_szshader);
    } else if(!strcmp("select", g_sztask)) {
        // -t select -f "c:\SVC\Dandelion\accelerators\PTaskUnitTest\select.hlsl" -D "selectdata.txt" -s main 
        res = run_select_task(g_szfile, g_szshader, g_szselectdata, g_nIterations);
    } else if(!strcmp("pfxsum", g_sztask)) {
        // -t pfxsum -f "c:\SVC\Dandelion\accelerators\PTaskUnitTest\pfxsum.hlsl" -s main 
        res = run_pfxsum_task(g_szfile, g_szshader, g_nIterations);
    } else if(!strcmp("cpfxsum", g_sztask)) {
        // -t cpfxsum -f "c:\SVC\Dandelion\accelerators\PTaskUnitTest\pfxsum.ptx" -s pfxsum 
        res = run_cpfxsum_task(g_szfile, g_szshader, g_nIterations);
    } else if(!strcmp("graphbench", g_sztask)) {
        // -t graphbench -f "..\..\..\PTaskUnitTest\matrixmul.hlsl" -s op -z 1 -Z 6 -x 2 -y 1 -Y 6 -X 2 -w 32 -W 256 -i 50 -T 0
        res = run_graph_bench_task(
                    g_szfile,
                    g_szshader,
                    g_depthMin,
                    g_depthMax,
                    g_depthStep,
                    g_bredthMin,
                    g_bredthMax,
                    g_bredthStep,
                    g_dimensionMin,
                    g_dimensionMax,
                    g_nIterations,
                    g_nInstances,
                    g_ptaskPriorities,
                    g_threadPriorities,
                    g_nSchedulingMode,
                    g_verbose,
                    g_instrument,
                    g_gtype);
    } else if(!strcmp("matmul", g_sztask)) {
        // -G -t matmul -f c:\SVC\Dandelion\accelerators\PTaskUnitTest\matrixmul.hlsl -s op -r 64 -c 64 -n 1 -i 1
        res = run_graph_matmul_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("pipelinestresstest", g_sztask)) {
        // -G -t pipelinestresstest -f c:\SVC\Dandelion\accelerators\PTaskUnitTest\pipelinestresstest.hlsl -s op -r 1024 -c 1024 -n 1 -i 1
        res = run_pipestress_simple(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations, g_bVerify, g_bCopyback);
    } else if(!strcmp("pipelinestresstestmulti", g_sztask)) {
        // -G -t pipelinestresstestmulti -f c:\SVC\Dandelion\accelerators\PTaskUnitTest\pipelinestresstest.hlsl -s op -r 1024 -c 1024 -n 1 -i 1 *-C 0*
        res = run_pipestress_general(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations, g_bVerify, g_bCopyback);
    } else if(!strcmp("hostmatmul", g_sztask)) {
        // -G -t hostmatmul -f d:\SVC\Dandelion\accelerators\HostTasks\x64\Debug\HostTasks.dll -s htmatmul -r 64 -c 64 -n 1 -i 1
        res = run_graph_host_matmul_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("ptaskcublas", g_sztask)) {
        // -G -t ptaskcublas -f d:\SVC\Dandelion\accelerators\HostTasks\x64\Debug\HostTasks.dll -s SGemmTrA -r 32 -c 64 -n 1 -i 1
        res = run_cublas_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("hostfuncmatmul", g_sztask)) {
        // -G -t hostfuncmatmul -r 64 -c 64 -n 1 -i 1
        res = run_graph_hostfunc_matmul_task(g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("hostfunccublasmatmul", g_sztask)) {
        // -G -t hostfuncptaskcublas -s SGemmTrA -r 32 -c 64 -n 1 -i 1
        res = run_hostfunc_cublas_matmul_task(g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("ptaskcublasnonsq", g_sztask)) {
        // -G -t ptaskcublasnonsq -f d:\SVC\Dandelion\accelerators\HostTasks\x64\Debug\HostTasks.dll -s SGemm -r 32 -c 64 -n 1 -i 1
        res = run_cublas_task_nonsq(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);    
    } else if(!strcmp("ptaskcublassq", g_sztask)) {
        // -G -t ptaskcublassq -f d:\SVC\Dandelion\accelerators\HostTasks\x64\Debug\HostTasks.dll -s SGemmSq -r 64 -c 64 -n 1 -i 1
        res = run_cublas_task_nonsq(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);    
    } else if(!strcmp("ptaskcublasnoinout", g_sztask)) {
        // -G -t ptaskcublasnoinout -f d:\SVC\Dandelion\accelerators\HostTasks\x64\Debug\HostTasks.dll -s SGemmSq -r 64 -c 64 -n 1 -i 1
        res = run_cublas_task_no_inout(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);    
    } else if(!strcmp("matmulraw", g_sztask)) {
        // -G -t matmulraw -f c:\SVC\Dandelion\accelerators\PTaskUnitTest\matrixmul_4ch.hlsl -s op -r 16 -c 16 -n 1 -i 1
        res = run_graph_matmulraw_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("cuvecadd", g_sztask)) {
        // -G -t cuvecadd -f vectorAddC2.ptx -s VecAdd -r 50000 -c 1 -n 1 -i 1
        res = run_graph_cuda_vecadd_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("inout", g_sztask)) {
        // -G -t inout -f inout.ptx -s scale -r 50000 -c 1 -n 1 -i 1
        res = run_inout_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("dxinout", g_sztask)) {
        // -G -t dxinout -f inout.hlsl -s op -r 50000 -c 1 -n 1 -i 1
        res = run_dxinout_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("switchports", g_sztask)) {
        // -G -t switchports -f switchports.ptx -s scale -r 50000 -c 1 -n 1 -i 1
        res = run_graph_cuda_switchport_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("deferredports", g_sztask)) {
        // -G -t deferredports -f deferredports.ptx -s scale -r 50000 -c 1 -n 1 -i 1
        res = run_graph_cuda_deferredport_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("descportsout", g_sztask)) {
        // -G -t descportsout -f descportsout.ptx -s scale -r 50000 -c 1 -n 1 -i 1
        res = run_graph_cuda_descportsout_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("gatedports", g_sztask)) {
        // -G -t gatedports -f gatedports.ptx -s scale -r 50000 -c 1 -n 1 -i 1
        res = run_graph_cuda_gatedport_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("bufferdimsdesc", g_sztask)) {
        // -G -t bufferdimsdesc -f bufferdimsdesc.ptx -s scale -r 100 -c 100 -n 1 -i 1
        res = run_graph_cuda_bufferdimsdesc_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("permanentblocks", g_sztask)) {
        // -G -t permanentblocks -f permanentblocks.ptx -s scale -r 50000 -c 1 -n 1 -i 1
        res = run_graph_cuda_permanentblocks_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("controlpropagation", g_sztask)) {
        // -G -t controlpropagation -f controlpropagation.ptx -s scale -r 50000 -c 1 -n 1 -i 1
        res = run_graph_cuda_controlpropagation_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("channelpredication", g_sztask)) {
        // -G -t channelpredication -f channelpredication.ptx -s scale -r 50000 -c 1 -n 1 -i 1
        // -G -t channelpredication -f channelpredication.hlsl -s scale -r 50000 -c 1 -n 1 -i 1
        res = run_channelpredication_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("simpleiteration", g_sztask)) {
        // -G -t simpleiteration -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t simpleiteration -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t simpleiteration -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_simple_iteration_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("generaliteration", g_sztask)) {
        // -G -t generaliteration -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t generaliteration -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t generaliteration -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_general_iteration_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("generaliteration2", g_sztask)) {
        // -G -t generaliteration -f iteration2.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t generaliteration -f iteration2.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t generaliteration -f iteration2.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_general_iteration2_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("scopediteration", g_sztask)) {
        // -G -t scopediteration -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t scopediteration -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t scopediteration -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_scoped_iteration_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("initializerchannels", g_sztask)) {
        // -G -t initializerchannels -f initializerchannels.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_initializer_channel_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("initializerchannelsbof", g_sztask)) {
        // -G -t initializerchannelsbof -f initializerchannels.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_initializer_channel_task_bof(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("generaliteration2_preloop", g_sztask)) {
        res = run_general_iteration2_with_preloop_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("tee", g_sztask)) {
        // -G -t tee -f tee.ptx -s scale -r 50000 -c 1 -n 1 -i 1
        res = run_graph_cuda_tee_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);    
    } else if(!strcmp("initports", g_sztask)) {
        // -G -t initports -f initports.ptx -s scale -r 50000 -c 1 -n 1 -i 1
        res = run_initports_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("metaports", g_sztask)) {
        // -G -t metaports -f metaports.ptx -s op -r 50000 -c 1 -n 1 -i 1
        res = run_metaports_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("hostmetaports", g_sztask)) {
        // -G -t hostmetaports -f c:\SVC\Dandelion\accelerators\HostTasks\x64\Debug\HostTasks.dll -s htvadd -r 64 -c 1 -n 1 -i 1
        // -C 0 -m 2 -G -t hostmetaports -f c:\SVC\Dandelion\bin\x64\Debug\HostTasks.dll -s htvadd -r 20000 -c 1 -n 1 -i 1
        res = run_graph_cuda_hostmetaports_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("clvecadd", g_sztask)) {
        // -G -t clvecadd -f vectoradd.cl -s vadd -r 50000 -c 1 -n 1 -i 1
        res = run_graph_opencl_vecadd_task(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("cupca", g_sztask)) {
        // -G -t cupca -f C:\SVC\Dandelion\accelerators\PTaskUnitTest\pca.ptx -s preNorm -r   128 -c 128 -n 1 -i 10 -j 10 -K 3 -L 1
        res = run_graph_cuda_pca_task(g_szfile, g_szshader, g_rows, g_cols, g_nIterations, g_nInnerIterations, g_nComponents, g_bPipelineInputs);
    } else if(!strcmp("hlslfdtd", g_sztask)) {
        // -G -t hlslfdtd -d c:\SVC\Dandelion\accelerators\PTaskUnitTest\fdtdKernels -r 8 -c 8 -z 4 -n 1 -i 1
        res = run_graph_fdtd_task(g_szxformdata, g_szshader, g_rows, g_cols, g_depthMin, g_nInstances, g_nIterations, HLSL);
    } else if(!strcmp("cufdtd", g_sztask)) {
        // -G -t cufdtd -d c:\SVC\Dandelion\accelerators\PTaskUnitTest\fdtdKernels -r 8 -c 8 -z 4 -n 1 -i 1
        res = run_graph_fdtd_task(g_szxformdata, g_szshader, g_rows, g_cols, g_depthMin, g_nInstances, g_nIterations, CUDA);
    } else if(!strcmp("cupfxsum", g_sztask)) {
        //Inclusive prefix sum
        // -G -t cupfxsum -d c:\SVC\Dandelion\accelerators\PTaskUnitTest\cupfxsumkernel.ptx -r 1024 -n 5
    } else if(!strcmp("cugroupby", g_sztask)) {
        // -r : length of input array
        // -c : number of keys
        // -i : number of groupbys to run
        // -G -t cugroupby -d c:\SVC\Dandelion\accelerators\PTaskUnitTest\groupbyKernels -r 1024 -c 32 -i 1
        res = run_graph_groupby_task(g_szxformdata, g_szshader, g_rows, g_cols, g_nIterations);
    } else if(!strcmp("rttest1", g_sztask)) {
        // -G -t rttest1 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest1 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest1 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_init_teardown_test(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest2", g_sztask)) {
        // -G -t rttest2 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest2 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest2 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_init_teardown_test_with_objects_no_run(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest3", g_sztask)) {
        // -G -t rttest3 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest3 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest3 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_init_teardown_test_with_run(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest4", g_sztask)) {
        // -G -t rttest4 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest4 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest4 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_init_teardown_test_with_single_push(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest5", g_sztask)) {
        // -G -t rttest5 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest5 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest5 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_init_teardown_test_with_multi_push(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest6", g_sztask)) {
        // -G -t rttest6 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest6 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest6 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_init_teardown_test_with_all_push_no_pull(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest7", g_sztask)) {
        // -G -t rttest7 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest7 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest7 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_init_teardown_test_pull_and_discard(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest8", g_sztask)) {
        // -G -t rttest8 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest8 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest8 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_init_teardown_test_pull_and_open(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest9", g_sztask)) {
        // -G -t rttest9 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest9 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest9 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_multi_init_teardown_test(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest10", g_sztask)) {
        // -G -t rttest10 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest10 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest10 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_multi_graph_test(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest11", g_sztask)) {
        //-G -t rttest11 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        //-G -t rttest11 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        //-G -t rttest11 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        //res = run_accelerator_disable_test(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest12", g_sztask)) {
        // -G -t rttest12 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest12 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest12 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_multi_init_test(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest13", g_sztask)) {
        // -G -t rttest13-f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest13 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest13 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_concurrent_graph_test(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest14", g_sztask)) {
        // -G -t rttest14 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest14 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest14 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_ultra_concurrent_graph_test(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("rttest15", g_sztask)) {
        // -G -t rttest15 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest15 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t rttest15 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_extreme_concurrent_graph_test(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else if(!strcmp("graphdiagnostics1", g_sztask)) {
        // -G -t graphdiagnostics1 -f iteration.ptx -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t graphdiagnostics1 -f iteration.hlsl -s op -r 50000 -c 1 -n 1 -i 10
        // -G -t graphdiagnostics1 -f iteration.cl -s op -r 50000 -c 1 -n 1 -i 10
        res = run_mismatched_template_detection(g_szfile, g_szshader, g_rows, g_cols, g_nInstances, g_nIterations);
    } else {
        res = -1;
        printf("unknown workload requested: %s\n", g_sztask);
    }

    if(g_ptaskPriorities)
        delete [] g_ptaskPriorities;
    if(g_threadPriorities)
        delete [] g_threadPriorities;

    if (FALSE)
    {
        printf("\nPress <Enter> to Quit....\n");                  
        getchar();                                                           
    }

    if(!IsDebuggerPresent()) {
        _CrtSetReportMode( _CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG );
        _CrtSetReportFile( _CRT_WARN, _CRTDBG_FILE_STDERR );
    }
    printf("exiting...\n");
    return res;
}

