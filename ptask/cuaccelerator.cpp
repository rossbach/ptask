//--------------------------------------------------------------------------------------
// File: CUAccelerator.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#ifdef CUDA_SUPPORT 

#define CU_USE_MEMHOST_ALLOC
#define USE_PAGELOCKED_MEMORY

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "CUAccelerator.h"
#include "PCUBuffer.h"
#include "CUTask.h"
#include "MemorySpace.h"
#include "CUAsyncContext.h"
#include "datablock.h"
#include "extremetrace.h"
#include <cublas_v2.h>  // cjr--added for cudamatrixop::l_vhandles
#include <assert.h>
#include <algorithm>
#include <set>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "nvtxmacros.h"
using namespace std;

#define PTASSERT(x) assert(x)

#define ACQUIRE_CTXNL(acc)                            \
        BOOL bFCC = !acc->IsDeviceContextCurrent();   \
        if(bFCC) acc->MakeDeviceContextCurrent();           
#define RELEASE_CTXNL(acc)                            \
        if(bFCC) acc->ReleaseCurrentDeviceContext();  \

#define ACQUIRE_CTX(acc)                              \
        acc->Lock();                                  \
        ACQUIRE_CTXNL(acc);                           
#define RELEASE_CTX(acc)                              \
        RELEASE_CTXNL(acc);                           \
        acc->Unlock();

#define ONFAILURE(apicall, code) {                                      \
    assert(FALSE);                                                      \
    PTask::Runtime::HandleError("%s::%s--%s failed res=%d (line:%d)\n", \
                                __FILE__,                               \
                                __FUNCTION__,                           \
                                #apicall,                               \
                                code,                                   \
                                __LINE__); }

#ifdef DEBUG
#define CHECK_CONTEXT_INVARIANTS() CheckContextInvariants()
#else
#define CHECK_CONTEXT_INVARIANTS()
#endif


namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Convert streaming multiprocessor count to core count
    /// 			based on version. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="major">    The major version. </param>
    /// <param name="minor">    The minor version. </param>
    ///
    /// <returns>   number of physical cores </returns>
    ///-------------------------------------------------------------------------------------------------

    inline int ConvertSMVer2Cores(int major, int minor)
    {
        // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
        typedef struct {
            int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
            int Cores;
        } sSMtoCores;

        sSMtoCores nGpuArchCoresPerSM[] = 
        {
			{ 0x10,  8 }, // Tesla Generation  (SM 1.0) G80 class
			{ 0x11,  8 }, // Tesla Generation  (SM 1.1) G8x class
			{ 0x12,  8 }, // Tesla Generation  (SM 1.2) G9x class
			{ 0x13,  8 }, // Tesla Generation  (SM 1.3) GT200 class
			{ 0x20, 32 }, // Fermi Generation  (SM 2.0) GF100 class
			{ 0x21, 48 }, // Fermi Generation  (SM 2.1) GF10x class
			{ 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
			{ 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
			{   -1, -1 }
        };

        int index = 0;
        while (nGpuArchCoresPerSM[index].SM != -1) {
            if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
                return nGpuArchCoresPerSM[index].Cores;
            }
            index++;
        }
        printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
        return -1;
    }

    /// <summary> true to if CUDA initializer has been called. </summary>
    BOOL CUAccelerator::s_bCUDAInitialized = FALSE;

    /// <summary>   context of primary device. This is essentially the "primary" CUDA context, but
    ///             might not actually be the primary if user code is also managing device contexts
    ///             or using cuda runtime API calls.
    ///             </summary>
    CUcontext        CUAccelerator::s_pRootContext = NULL;

    /// <summary> true if the root context is valid </summary>
    BOOL             CUAccelerator::s_bRootContextValid = FALSE;

    /// <summary>   device id for the root context. </summary>
    CUdevice         CUAccelerator::s_nRootContext = MAXINT;
    CUcontext       CUAccelerator::s_vKnownPTaskContexts[MAXCTXTS];
    UINT            CUAccelerator::s_nKnownPTaskContexts = 0;
    CUcontext       CUAccelerator::s_vKnownUserContexts[MAXCTXTS];
    UINT            CUAccelerator::s_nKnownUserContexts = 0;

    /// <summary>   Thread-local storage for caching device contexts,
    ///             enabling some heuristics to avoid unnecessary and occasionally
    ///             expensive calls to cuCtx[Push|Pop]Current. </summary>
    __declspec(thread) CUAccelerator*  CUAccelerator::s_pDefaultDeviceCtxt = NULL;
    __declspec(thread) CUAccelerator*  CUAccelerator::s_pCurrentDeviceCtxt = NULL;
    __declspec(thread) int             CUAccelerator::s_vContextDepthMap[MAXCTXTS];
    __declspec(thread) CUAccelerator **CUAccelerator::s_pContextChangeMap[MAXCTXTS];
    __declspec(thread) CUAccelerator * CUAccelerator::s_vContextChangeMap[MAXCTXTS*MAXCTXDEPTH];    
    __declspec(thread) BOOL            CUAccelerator::s_bContextTLSInit = FALSE;
    __declspec(thread) PTTHREADROLE    CUAccelerator::s_eThreadRole = PTTR_UNKNOWN;
    __declspec(thread) CUcontext       CUAccelerator::s_pUserStackTop = NULL;
    __declspec(thread) BOOL            CUAccelerator::s_bThreadPoolThread = FALSE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'ctx' is a known context. (helper) </summary>
    ///
    /// <remarks>   crossbac, 6/18/2014. </remarks>
    ///
    /// <param name="ctx">          The context. </param>
    /// <param name="pContexts">    [in,out] If non-null, the contexts. </param>
    /// <param name="uiCtxCount">   Number of contexts. </param>
    ///
    /// <returns>   true if known context, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CUAccelerator::IsKnownContext(
        __in CUcontext ctx, 
        __in CUcontext * pContexts, 
        __in UINT uiCtxCount
        ) {
        if(ctx == NULL) return FALSE;
        for(UINT ui=0; ui<uiCtxCount; ui++) 
            if(pContexts[ui] == ctx)
                return TRUE;
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a known context. (helper) </summary>
    ///
    /// <remarks>   crossbac, 6/18/2014. </remarks>
    ///
    /// <param name="ctx">          The context. </param>
    /// <param name="pContexts">    [in,out] If non-null, the contexts. </param>
    /// <param name="puiCtxCount">  [in,out] If non-null, number of pui contexts. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    CUAccelerator::AddKnownContext(
        __in CUcontext ctx, 
        __in CUcontext * pContexts, 
        __inout UINT * puiCtxCount
        )
    {
        if(ctx == NULL) return FALSE;
        UINT uiIndex = ::InterlockedIncrement(puiCtxCount) - 1;
        if(*puiCtxCount >= MAXCTXTS) {
            if(PTask::Runtime::HandleError("%s::%s(%d): PANIC! tracking too many %s contexts(%d)!\n",
                                           __FILE__,
                                           __FUNCTION__,
                                           __LINE__,
                                           (pContexts == s_vKnownPTaskContexts ? "PTask" : "User"),
                                           *puiCtxCount)) return FALSE;
        }
        pContexts[uiIndex] = ctx;
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check context provenance. </summary>
    ///
    /// <remarks>   crossbac, 6/18/2014. </remarks>
    ///
    /// <param name="ctx">  The context. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CUAccelerator::CheckContextProvenance(
        CUcontext ctx
        )
    {
        if(ctx == NULL) return TRUE;
        if(IsKnownPTaskContext(ctx)) return TRUE;
        if(IsKnownUserContext(ctx)) return TRUE;
        if(s_eThreadRole == PTTR_GRAPHRUNNER || 
           s_eThreadRole == PTTR_SCHEDULER ||
           s_eThreadRole == PTTR_GC) {            
            // we've never seen this context before. 
            // if the context is being used on a graph runner or a scheduler 
            // thread, we've got a problem. Complain. 
            assert(!(s_eThreadRole == PTTR_GRAPHRUNNER || s_eThreadRole == PTTR_GC));
            PTask::Runtime::HandleError("%s::%s(%d): PANIC! unknown current context on ptask private thread!\n",
                                        __FILE__,
                                        __FUNCTION__,
                                        __LINE__);
            return FALSE;
        }

        // we've never seen this context before, but the
        // thread is a known application thread or a thread
        // we've not seen (e.g. gc finalizer) making it an app thread
        // as far as we're concerned. Assume conservatively that
        // this is a user context, add it (if it still fits) and
        // then declare it safe. 
        return AddKnownUserContext(ctx);
    }

    BOOL CUAccelerator::IsKnownContext(CUcontext ctx) { return IsKnownUserContext(ctx) || IsKnownPTaskContext(ctx); }
    BOOL CUAccelerator::IsUserContext(CUcontext ctx) { return (IsKnownUserContext(ctx) || !IsKnownPTaskContext(ctx));}
    BOOL CUAccelerator::IsPTaskContext(CUcontext ctx) { return IsKnownPTaskContext(ctx); }
    BOOL CUAccelerator::IsKnownPTaskContext(CUcontext ctx) { return IsKnownContext(ctx, s_vKnownPTaskContexts, s_nKnownPTaskContexts); }
    BOOL CUAccelerator::IsKnownUserContext(CUcontext ctx) { return IsKnownContext(ctx, s_vKnownUserContexts, s_nKnownUserContexts); }
    BOOL CUAccelerator::AddKnownPTaskContext(CUcontext ctx) { return AddKnownContext(ctx, s_vKnownPTaskContexts, &s_nKnownPTaskContexts); }
    BOOL CUAccelerator::AddKnownUserContext(CUcontext ctx) { return AddKnownContext(ctx, s_vKnownUserContexts, &s_nKnownUserContexts); }
    

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the cuda initialized flag. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="b">    true to b. </param>
    ///-------------------------------------------------------------------------------------------------

    void                    
    CUAccelerator::SetCUDAInitialized(
        BOOL b
        )
    {
        //if(b) assert(!m_bCUDAInitialized); // don't do this twice!
       s_bCUDAInitialized = b; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pAttribs"> [in,out] If non-null, the attributes. </param>
    ///-------------------------------------------------------------------------------------------------

    CUAccelerator::CUAccelerator(
        CUDA_DEVICE_ATTRIBUTES * pAttribs
        ) 
    {
        assert(pAttribs != NULL);
        m_pDeviceAttributes = &m_attrs;
        memcpy(m_pDeviceAttributes, pAttribs, sizeof(CUDA_DEVICE_ATTRIBUTES));
        m_bInitialized = FALSE;
        m_pContext = NULL;
        m_pDevice = NULL;
        m_class = ACCELERATOR_CLASS_CUDA;
        size_t nLength = strlen(pAttribs->deviceName)+10;
        m_lpszDeviceName = (char*) malloc(nLength);
        sprintf_s(m_lpszDeviceName, nLength, "%s:%d", pAttribs->deviceName, (int) pAttribs->dev);
        m_uiMemorySpaceId = MemorySpace::AssignUniqueMemorySpaceIdentifier();
        std::string strDeviceName(m_lpszDeviceName);
        m_pMemorySpace = new MemorySpace(strDeviceName, m_uiMemorySpaceId);
        m_pMemorySpace->UpdateSpaceSizeBytes(m_pDeviceAttributes->totalGlobalMem);
        MemorySpace::RegisterMemorySpace(m_pMemorySpace, this);
        MemorySpace * pHostSpace = MemorySpace::GetMemorySpaceFromId(HOST_MEMORY_SPACE_ID);
        pHostSpace->AddDeferredAllocationEntry(this);
        m_nDeviceId = (int) pAttribs->dev;
        m_bApplicationPrimaryContext = FALSE;
        Open(m_nDeviceId);
        CreateAsyncContext(NULL, ASYNCCTXT_DEFAULT);
        CreateAsyncContext(NULL, ASYNCCTXT_XFERDTOD);
        CreateAsyncContext(NULL, ASYNCCTXT_XFERDTOH);
        CreateAsyncContext(NULL, ASYNCCTXT_XFERHTOD);
        assert(m_bInitialized);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    CUAccelerator::~CUAccelerator() {
        ReleaseAsyncContexts();
        Lock();
        cuCtxDetach(m_pContext); trace("cuCtxDetach");
        m_bInitialized = FALSE;
        Unlock();
        if(PTask::Runtime::GetTrackDeviceMemory()) {
            PTask::Runtime::MandatoryInform("accelerator(%s)::dtor: memory statistics:\n", m_lpszDeviceName);
            m_pMemorySpace->Report(std::cout);
        }
        assert(m_lpszDeviceName != NULL);
        free(m_lpszDeviceName);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enumerate all CUDA-capable accelerators in the system. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="devices">  [out] the devices. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CUAccelerator::EnumerateAccelerators(
        std::vector<Accelerator*> &devices
        )
    {
        if(!PTask::Runtime::GetUseCUDA())
            return;

        CUdevice dev;
        int deviceCount = 0;
        char msg[USER_OUTPUT_BUF_SIZE];

        PTask::Runtime::Inform("CUDA devices:");
        CUresult err = cuInit(0); trace("cuInit");
        if(err != CUDA_SUCCESS) {
            PTask::Runtime::Warning("There is no device supporting CUDA");
            return;
        }
        CUAccelerator::SetCUDAInitialized(err == CUDA_SUCCESS);
        cuDeviceGetCount(&deviceCount); trace("cuDeviceGetCount");
        if (deviceCount == 0) {
            PTask::Runtime::Warning("There is no device supporting CUDA");
            return;
        }

        int realDevCount = 0;
        set<CUdevice> vCUDADevices;
        vector<Accelerator*> candidateAccelerators;
        for (dev = 0; dev < deviceCount; ++dev) {
            CUDA_DEVICE_ATTRIBUTES attrs;
            memset(&attrs, 0, sizeof(attrs));
            cuDeviceComputeCapability(&attrs.major, &attrs.minor, dev); trace("cuDeviceComputeCapability");
            if (attrs.major < 2) continue;
            realDevCount++;
            int nRuntimeVersion = attrs.major * 1000 + attrs.minor*100;
            if (dev == 0) {
                if (PTask::Runtime::IsVerbose()) {
                    // This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
                    if (attrs.major == 9999 && attrs.minor == 9999)
                        PTask::Runtime::Inform("There is no device supporting CUDA.");
                    else if (deviceCount == 1)
                        PTask::Runtime::Inform("There is 1 device supporting CUDA");
                    else {
                        PTask::Runtime::Inform("There are %d devices supporting CUDA\n", deviceCount);
                    }
                }
            }
            attrs.dev = dev;
            cuDeviceGetName(attrs.deviceName, 256, dev); trace("cuDeviceGetName");
            if (PTask::Runtime::IsVerbose()) { 
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "\nDevice %d: \"%s\"\n", dev, attrs.deviceName);
                PTask::Runtime::Inform(msg);
            }
            vCUDADevices.insert(dev);

            cuDriverGetVersion(&attrs.driverVersion); trace("cuDriverGetVersion");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "CUDA Driver Version:                           %d.%d\n", attrs.driverVersion/1000, attrs.driverVersion%100);
                PTask::Runtime::Inform(msg);
            }

            cuDeviceTotalMem(&attrs.totalGlobalMem, dev); trace("cuDeviceTotalMem");
            if (PTask::Runtime::IsVerbose()) { 
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Total amount of global memory:                 %llu bytes\n", (unsigned long long)attrs.totalGlobalMem);
                PTask::Runtime::Inform(msg);
            }

            cuDeviceGetAttribute( &attrs.multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(
                    msg, USER_OUTPUT_BUF_SIZE, "  Multiprocessors x Cores/MP = Cores:            %d (MP) x %d (Cores/MP) = %d (Cores)\n", 
                    attrs.multiProcessorCount, 
                    ConvertSMVer2Cores(attrs.major, attrs.minor), 
                ConvertSMVer2Cores(attrs.major, attrs.minor) * attrs.multiProcessorCount);
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.totalConstantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Total amount of constant memory:               %u bytes\n", attrs.totalConstantMemory);
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.sharedMemPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Total amount of shared memory per block:       %u bytes\n", attrs.sharedMemPerBlock);
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Total number of registers available per block: %d\n", attrs.regsPerBlock);
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Warp size:                                     %d\n",	attrs.warpSize);
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Maximum number of threads per block:           %d\n",	attrs.maxThreadsPerBlock);
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.maxBlockDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev ); trace("cuDeviceGetAttribute");
            cuDeviceGetAttribute( &attrs.maxBlockDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev ); trace("cuDeviceGetAttribute");
            cuDeviceGetAttribute( &attrs.maxBlockDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(
                    msg, USER_OUTPUT_BUF_SIZE, 
                    "  Maximum sizes of each dimension of a block:    %d x %d x %d\n", 
                    attrs.maxBlockDim[0], 
                    attrs.maxBlockDim[1], 
                    attrs.maxBlockDim[2]);
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.maxGridDim[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev ); trace("cuDeviceGetAttribute");
            cuDeviceGetAttribute( &attrs.maxGridDim[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev ); trace("cuDeviceGetAttribute");
            cuDeviceGetAttribute( &attrs.maxGridDim[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(
                    msg, USER_OUTPUT_BUF_SIZE, 
                    "  Maximum sizes of each dimension of a grid:     %d x %d x %d\n", 
                    attrs.maxGridDim[0], 
                    attrs.maxGridDim[1], 
                    attrs.maxGridDim[2]);
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.memPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Maximum memory pitch:                          %u bytes\n", attrs.memPitch);
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.textureAlign, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Texture alignment:                             %u bytes\n", attrs.textureAlign);
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Clock rate:                                    %.2f GHz\n", attrs.clockRate * 1e-6f);
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.gpuOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Concurrent copy and execution:                 %s\n",attrs.gpuOverlap ? "Yes" : "No");
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.kernelExecTimeoutEnabled, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Run time limit on kernels:                     %s\n", attrs.kernelExecTimeoutEnabled ? "Yes" : "No");
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Integrated:                                    %s\n", attrs.integrated ? "Yes" : "No");
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Support host page-locked memory mapping:       %s\n", attrs.canMapHostMemory ? "Yes" : "No");
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.concurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Concurrent kernel execution:                   %s\n", attrs.concurrentKernels ? "Yes" : "No");
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.eccEnabled,  CU_DEVICE_ATTRIBUTE_ECC_ENABLED, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Device has ECC support enabled:                %s\n", attrs.eccEnabled ? "Yes" : "No");
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.tccDriver ,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Device is using TCC driver mode:               %s\n", attrs.tccDriver ? "Yes" : "No");
                PTask::Runtime::Inform(msg);
            }
            cuDeviceGetAttribute( &attrs.unifiedAddressing ,  CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev ); trace("cuDeviceGetAttribute");
            if (PTask::Runtime::IsVerbose()) {
                sprintf_s(msg, USER_OUTPUT_BUF_SIZE, "  Device supports unified addressing:            %s\n", attrs.unifiedAddressing ? "Yes" : "No");
                PTask::Runtime::Inform(msg);
            }
            Accelerator * pAccelerator = new CUAccelerator(&attrs);
            pAccelerator->SetPlatformSpecificRuntimeVersion(nRuntimeVersion);
            pAccelerator->SetCoreClockRate(attrs.clockRate);
            pAccelerator->SetGlobalMemorySize((UINT)(attrs.totalGlobalMem/1000000)); // in MB!
            pAccelerator->SetCoreCount(ConvertSMVer2Cores(attrs.major, attrs.minor) * attrs.multiProcessorCount);
            pAccelerator->SetSupportsConcurrentKernels(attrs.concurrentKernels);
            pAccelerator->SetPlatformIndex(dev);
            candidateAccelerators.push_back(pAccelerator);
        }

        std::vector<Accelerator*>::iterator vi;
        for(vi=candidateAccelerators.begin(); vi!=candidateAccelerators.end(); vi++) {
            devices.push_back(*vi);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check context invariants. </summary>
    ///
    /// <remarks>   Crossbac, 3/20/2014. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    CUAccelerator::CheckContextInvariants(
        VOID
        )
    {
        // the general idea is that ptask wants to leave contexts current on the threads where they
        // will most likely be used, to avoid the overhead of cuCtxPush/Pop calls, which actually can
        // add up, experientially speaking. To do this, we use some TLS data structures to keep track
        // of the default context for each thread, and only make context state changes when the default
        // context isn't the one we happen to need. This function checks that our expectation about
        // context state meet a set of invariants before a sequence of device-related commands occurs--
        // in a world where PTask manages all contexts, this means one where either no context is
        // current because we haven't acquired the device yet, or where there is a current context, but
        // that's ok because it's the default context and the stack depth for that context is 0.
        // 
        // There are some complicating considerations for checking these invariants: 
        // 1. We need to mirror the stack-based context management in the actual CUDA
        //    API since we may need to switch contexts an arbitrary number of times
        //    in service of a single dispatch (e.g. to deal with gpu-gpu xfer and
        //    so on).
        // 2. If user code creates it's own contexts, either intentionally, or indirectly through
        //    calls to the CUDA runtime API (which will create a primary context and manage it
        //    transparently to the user), then user code can leave these contexts current, and we
        //    have no way to track them. We need to handle these cases differently. 
         
#ifdef DEBUG
        CUcontext ctxt;                                                              
        CUresult eResult = cuCtxGetCurrent(&ctxt);                                   
        assert(eResult == CUDA_SUCCESS);            // call should always succeed
        BOOL bNullContextOnEntry = ctxt == NULL;
        if(!bNullContextOnEntry && !CheckContextProvenance(ctxt)) return;
        BOOL bPTaskCtxtOnEntry = !bNullContextOnEntry && IsPTaskContext(ctxt);
        BOOL bUserCtxtOnEntry = !bNullContextOnEntry && IsUserContext(ctxt); 
        BOOL bUserOnlyCtxtOnEntry = bUserCtxtOnEntry && !bPTaskCtxtOnEntry;
        BOOL bSharedContextOnEntry = bPTaskCtxtOnEntry && bUserCtxtOnEntry;
        assert(!bSharedContextOnEntry || m_bApplicationPrimaryContext); 

        if(s_eThreadRole == PTTR_UNKNOWN) {

            // this can be a thread we've never seen before. 
            // it's particularly difficult to track application
            // threads in managed land, where thread pool threads
            // can wind up making calls that require a cuda context.
            
            assert(s_pDefaultDeviceCtxt == NULL);               // never set a default device if we don't know the thread role
            assert(s_pCurrentDeviceCtxt == NULL);               // never stash a device context. 
            assert(bNullContextOnEntry || !bPTaskCtxtOnEntry);  // ideally there was no context on entry.
                                                                // however, we will tolerate a user or shared context,
                                                                // since we don't really have a choice. 

        } else {
            
            // a registered application thread will only have a default context if 
            // PTask::Runtime::GetApplicationThreadsManagePrimaryContext() is FALSE.
            // When this setting is configured, cuCtxGetCurrent should *always* give us
            // a non-null context. If the context is not the same as the default, 
            // it had better be a user context...
            //
            // conversely if PTask::Runtime::GetApplicationThreadsManagePrimaryContext()
            // is TRUE, then we can have a null context on entry if there is no user
            // context, and there is no current device context already set up by 
            // a call to MakeDeviceContextCurrent(). 

            BOOL bAppThreadsHaveDefault = !PTask::Runtime::GetApplicationThreadsManagePrimaryContext();
            BOOL bShouldHaveDefault = ((s_eThreadRole == PTTR_GRAPHRUNNER) ||
                                       (s_eThreadRole == PTTR_GC) ||
                                       ((s_eThreadRole == PTTR_APPLICATION) && 
                                        (bAppThreadsHaveDefault)));

            assert(!bShouldHaveDefault || s_pDefaultDeviceCtxt != NULL);
            assert(s_eThreadRole == PTTR_APPLICATION || !bUserOnlyCtxtOnEntry);
            assert(ctxt != NULL || !bShouldHaveDefault);

            if(s_pDefaultDeviceCtxt != NULL) {

                assert(ctxt != NULL);                                                       // default device always current
                assert(ctxt == s_pCurrentDeviceCtxt->m_pContext || bUserCtxtOnEntry);       // current device always current, unless we enter with a user ctxt.
                assert(s_vContextDepthMap[s_pDefaultDeviceCtxt->m_nPlatformIndex] > 0);     // default device always at bottom of stack

                if(bUserOnlyCtxtOnEntry) {

                    // the context had better not be the current dev.
                    // more over, since there is a USER context installed we have
                    // to be either at the entry to the first MakeDeviceContextCurrent
                    // call or at the end of the last ReleaseDeviceContext call.
                    assert(ctxt != s_pCurrentDeviceCtxt->m_pContext);
                    assert(s_pCurrentDeviceCtxt != NULL);
                    assert(s_pCurrentDeviceCtxt == s_pDefaultDeviceCtxt);
                    assert(s_vContextDepthMap[s_pDefaultDeviceCtxt->m_nPlatformIndex] == 1);

                }  else {

                    // this is a context we know about and created, so
                    // we can be anywhere in a nested series of PTask api
                    // calls that acquire/release the device. If the context
                    // depth for the this device is 0, then it's context is not
                    // current. Consequently, it had better not be the default 
                    
                    if(s_vContextDepthMap[m_nPlatformIndex] == 0) {

                        // 0 context depth. This had better not be
                        // the current device, and had better not be the
                        // default device, which always has > 0 depth.
            
                        assert(ctxt != m_pContext);
                        assert(this != s_pDefaultDeviceCtxt);
                        assert(this != s_pCurrentDeviceCtxt);

                    } else {

                        // > 0 context depth. If this is not the current device,
                        // it better be the default and the depth == 1. If it is
                        // the current device, then the depth must be > 0.
            
                        if(this != s_pCurrentDeviceCtxt) {

                            // not the current, so must be default to have depth > 0
                            assert(this == s_pDefaultDeviceCtxt || s_pCurrentDeviceCtxt == s_pDefaultDeviceCtxt);
                            assert(s_vContextDepthMap[s_pCurrentDeviceCtxt->m_nPlatformIndex] > 0);

                        } else {

                            // this is the current device. if it is not the default, 
                            assert(ctxt == s_pCurrentDeviceCtxt->m_pContext);
                        }
                    }   
                }

            } else {

                // there is no default device context for this thread. 
                // it had better be an app, scheduler, or unknown thread. 
                
                if(bUserCtxtOnEntry) {

                    assert(s_eThreadRole == PTTR_APPLICATION || bSharedContextOnEntry);
                    assert(s_pDefaultDeviceCtxt == NULL || bSharedContextOnEntry); 
                    if(bPTaskCtxtOnEntry) {

                        assert(m_bApplicationPrimaryContext);
                        assert(s_pCurrentDeviceCtxt == NULL || s_pCurrentDeviceCtxt->m_bApplicationPrimaryContext);
                        BOOL bNoCurrentDevice = (s_pCurrentDeviceCtxt == NULL && s_vContextDepthMap[m_nPlatformIndex] == 0);
                        BOOL bThisDeviceCurrent = (s_pCurrentDeviceCtxt == this && s_vContextDepthMap[m_nPlatformIndex] > 0);
                        BOOL bOtherDeviceCurrent = (s_pCurrentDeviceCtxt != NULL && s_pCurrentDeviceCtxt != this && 
                                                    s_vContextDepthMap[s_pCurrentDeviceCtxt->m_nPlatformIndex] > 0);

                        // this is both a user and ptask context. Meaning it is ok for CUDA to think it is 
                        // current on entry if this device is not current. That can be the case if *we* think
                        // this is actually the current device, or if we think there is no current device. 
                        // Or if some other user context is actually current. 
                        assert(bNoCurrentDevice || bThisDeviceCurrent || bOtherDeviceCurrent);

                    } else {

                        // definitely an unknown context. 
                        assert(!m_bApplicationPrimaryContext);
                        assert(s_pCurrentDeviceCtxt == NULL);
                        assert(s_vContextDepthMap[m_nPlatformIndex] == 0);
                    }

                } else {

                    if(s_vContextDepthMap[m_nPlatformIndex] == 0) {

                        // 0 context depth. This had better not be
                        // the current device, and had better not be the
                        // default device, which always has > 0 depth.
            
                        assert(ctxt != m_pContext);
                        assert(this != s_pCurrentDeviceCtxt);

                    } else {

                        if(this != s_pCurrentDeviceCtxt) {
                            // > 0 context depth. If this is not the current device,
                            // it better be the default and the depth == 1. If it is
                            // the current device, then the depth must be > 0.
                            assert(s_vContextDepthMap[s_pCurrentDeviceCtxt->m_nPlatformIndex] > 0);
                        } else {
                            // this is the current device. 
                            assert(ctxt == s_pCurrentDeviceCtxt->m_pContext);
                        }
                    }   
                }
            }
        }
#endif
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Opens a CUAccelerator </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> none </param>
    ///
    /// <returns>   HRESULT--use SUCCEEDED() and FAILED() macros to check </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT
    CUAccelerator::Open(
        VOID
        )
    {
        // just use the default device
        CUdevice devID = 0; 
        return Open(devID);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Opens a CUAccelerator for a given device ID </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="devID">    Identifier for the device. </param>
    ///
    /// <returns>   HRESULT--use SUCCEEDED() and FAILED() macros to check. </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    CUAccelerator::Open(
        CUdevice devID
        )
    {
        CUresult error;
        assert(s_bCUDAInitialized);  // Scheduler.cpp should do this.

        int major, minor;
        char deviceName[100];
        trace("cuDeviceComputeCapability");
        if(CUDA_SUCCESS != (error = cuDeviceComputeCapability(&major, &minor, devID))) {
            assert(false);
            sprintf_s(
                m_lpszUserMessages, 
                USER_OUTPUT_BUF_SIZE, 
                "cannot open device %d: cuDeviceComputeCapability failed",
                devID);
            PTask::Runtime::ErrorMessage(m_lpszUserMessages);
            return E_FAIL;
        }

        // cuDeviceGetAttribute
        // CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
        trace("cuDeviceGetAttribute");
        int nAttribVal = 0;
        if(CUDA_SUCCESS != (error = cuDeviceGetAttribute(&nAttribVal, CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, devID))) {
            assert(false);
            sprintf_s(
                m_lpszUserMessages, 
                USER_OUTPUT_BUF_SIZE, 
                "cannot open device %d: cuDeviceComputeCapability failed",
                devID);
            PTask::Runtime::ErrorMessage(m_lpszUserMessages);
            return E_FAIL;
        }        

        trace("cuDeviceGetName");
        if(CUDA_SUCCESS != (error = cuDeviceGetName(deviceName, 256, devID))) {
            assert(false);
            sprintf_s(
                m_lpszUserMessages, 
                USER_OUTPUT_BUF_SIZE, 
                "cannot open device %d: cuDeviceGetName failed",
                devID);
            PTask::Runtime::ErrorMessage(m_lpszUserMessages);
            return E_FAIL;
        }
        if (PTask::Runtime::IsVerbose()) {
            sprintf_s(
                m_lpszUserMessages, 
                USER_OUTPUT_BUF_SIZE, 
                "> Using Device %d: \"%s\" with Compute capability %d.%d\n", 
                devID, deviceName, major, minor);
            PTask::Runtime::Inform(m_lpszUserMessages);
        }

        // pick up device with zero ordinal (default, or devID)
        trace("cuDeviceGet");
        if(CUDA_SUCCESS != (error = cuDeviceGet(&m_pDevice, devID))) {
            assert(false);
            sprintf_s(
                m_lpszUserMessages, 
                USER_OUTPUT_BUF_SIZE, 
                "cannot open device %d: cuDeviceGet failed",
                devID);
            PTask::Runtime::ErrorMessage(m_lpszUserMessages);
            return E_FAIL;
        }

        CUcontext pPreviousContext = NULL;
        BOOL bContextKnownPrimary = FALSE;
        BOOL bAnyContextAlreadyCurrent = FALSE;
        BOOL bThisContextAlreadyCurrent = FALSE;

        if(PTask::Runtime::GetApplicationThreadsManagePrimaryContext()) {

            // if the application manages the primary device contexts 
            // (or are implicitly in control of them because they use the CUDA
            // runtime API), then we don't necessarily want to create additional
            // contexts. We want to find the primary ones that are already
            // created. The best way to do that seems to be to use the CUDA 
            // runtime API. Note that if the contexts haven't already been created
            // by the application when we get here, these calls should result
            // in creation of contexts that will be found when the app gets around 
            // to it. 
            
            CUresult ePreGetCurrent = cuCtxGetCurrent(&pPreviousContext);
            if(CUDA_SUCCESS != ePreGetCurrent) {
                if(PTask::Runtime::HandleError("%s::%s(%d) %s failed with %d\n",
                                               __FILE__,
                                               __FUNCTION__,
                                               __LINE__,
                                               "cuCtxGetCurrent",
                                               ePreGetCurrent)) return E_FAIL;
            }
            bAnyContextAlreadyCurrent = pPreviousContext != NULL;

            cudaError_t eSetDevice = cudaSetDevice(m_pDevice);
            if(cudaSuccess != eSetDevice) {
                const char* errmsg = cudaGetErrorString(eSetDevice);
                if(PTask::Runtime::HandleError("%s::%s(%d) %s failed with %s\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                __LINE__,
                                                "cudaSetDevice",
                                                errmsg)) return E_FAIL;
            }

            // now just get the primary context by querying it...
            CUresult eGetContext = cuCtxGetCurrent(&m_pContext);
            if(CUDA_SUCCESS != eGetContext) {
                if(PTask::Runtime::HandleError("%s::%s(%d) %s failed with %d\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                __LINE__,
                                                "cuCtxGetCurrent",
                                                eGetContext)) return E_FAIL;            
            }
            bThisContextAlreadyCurrent = m_pContext == pPreviousContext;
            // there may be a context currently active, but it may not have been initialized yet
			// so call a runtime API to ensure it is active
            cudaFree(0);

            bContextKnownPrimary = TRUE;
            m_bApplicationPrimaryContext = TRUE;
            AddKnownUserContext(m_pContext);

        } else {

            // the application has no idea about cuda devices (or doesn't
            // claim to anyway. We can safely create contexts through the 
            // driver API without worrying about conflicting contexts later.

            trace("cuCtxCreate");
            unsigned int uiFlags = CU_CTX_MAP_HOST | CU_CTX_SCHED_SPIN;
            if(CUDA_SUCCESS != (error = cuCtxCreate(&m_pContext, uiFlags, m_pDevice))) {
                assert(false);
                sprintf_s(
                    m_lpszUserMessages, 
                    USER_OUTPUT_BUF_SIZE, 
                    "cannot open device %d: cuDeviceGet failed",
                    devID);
                PTask::Runtime::ErrorMessage(m_lpszUserMessages);
                return E_FAIL;
            }
        }

        AddKnownPTaskContext(m_pContext);

        // track the "root" context. The root should be
        // the lowest numbered device that we actually open.
        if(!s_bRootContextValid || s_nRootContext > devID) {
            s_pRootContext = m_pContext;
            s_nRootContext = devID;
            s_bRootContextValid = TRUE; 
        }

        size_t szMaxPending = 0;
        CUresult ePendingRes = cuCtxGetLimit(&szMaxPending, CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT);
        if(ePendingRes == CUDA_SUCCESS) {
            PTask::Runtime::Inform("CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT =%d!\n", szMaxPending); 
        } else if(ePendingRes == CUDA_ERROR_UNSUPPORTED_LIMIT) {
            PTask::Runtime::Inform("XXX: CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT unsupported!\n"); 
        } else {
            assert(FALSE);
            PTask::Runtime::HandleError("FAILURE!\n");
        }
        // size_t szMaxSyncDepth = 0;
        // CUresult ePrioRes = CUDA_SUCCESS; 
        // cuCtxGetStreamPriorityRange(&m_nMinStreamPriority, &m_nMaxStreamPriority);

        //set the heap size for device-side mallocs
        if(PTask::Runtime::g_bUserDefinedCUDAHeapSize) {
            char szMessage[1024];
            size_t prevSize = 0, readBackSize;
            size_t newSize = static_cast<size_t>(PTask::Runtime::GetCUDAHeapSize());
            CUresult getRes = cuCtxGetLimit(&prevSize, CU_LIMIT_MALLOC_HEAP_SIZE);
            sprintf_s(szMessage, 1024, "PTask: old cuda heap size: %lld\n", prevSize);
            PTask::Runtime::Inform(szMessage);
            CUresult setRes = cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, newSize);
            if(setRes != CUDA_SUCCESS) {
                assert(setRes == CUDA_SUCCESS);
                PTask::Runtime::Warning("Attempt to set CUDA heap size failed!");
            }
            getRes = cuCtxGetLimit(&readBackSize, CU_LIMIT_MALLOC_HEAP_SIZE);
            sprintf_s(szMessage, 1024, "PTask: new cuda heap size: %lld\n", readBackSize);
            PTask::Runtime::Inform(szMessage);
        }

        // first mem-alloc in CUDA incurs overhead of
        // run-time configuration. So let's take the
        // hit here, instead of when we are actually
        // doing some real computation. 
        CUdeviceptr incurAllocOverhead = NULL;
        void * incurAllocOverheadPinned = NULL;
        CUresult aRes = cuMemAlloc(&incurAllocOverhead, 4096);
        CUresult apRes = cuMemAllocHost(&incurAllocOverheadPinned, 4096);
        assert(aRes == CUDA_SUCCESS);
        assert(apRes == CUDA_SUCCESS);
        if(aRes == CUDA_SUCCESS) cuMemFree(incurAllocOverhead);
        if(apRes == CUDA_SUCCESS) cuMemFreeHost(incurAllocOverheadPinned);    

        if(PTask::Runtime::g_bInitCublas) {
#ifdef LINKCUBLAS
            // cjr: 5/1/2013
            // initializing and destroying a cublas context
            // per-device at init time gets a pathologically 
            // low-latency operation off the critical path at runtime.
            // (cublasCreate calls cuFree for some reason, at an astonishing performance
            // cost). This is a bit of a hack, but it works, and after having spent
            // nearly a week trying to fix this in a principled way, I'm happy to
            // introduce a temporary work-around
            cublasHandle_t handle;
            cublasCreate(&handle);
            cublasDestroy(handle);
#else
            PTask::Runtime::MandatoryInform("PTask::Runtime::SetInitCublas() used in a build that does not link cublas!\n");
#endif
        }
        

        if(m_bApplicationPrimaryContext) {

            // we're about to detach a context from this thread without knowing whether it was previously
            // current because user code called PTask::Initialize with a device already current. 
            // Emit a warning. This really should be handled by save/restore context at the beginning/end
            // of the scheduler's initialization process. For now, complain to keep it on our radar. 
             
            PTask::Runtime::MandatoryInform("%s::%s(%d) warning, detaching primary context from application thread...\n"
                                             "    context may have been current on entry and app may need to restore it\n",
                                             __FILE__,
                                             __FUNCTION__,
                                             __LINE__);
        }

        CUcontext pLastCtxt = NULL;
        CUresult eCtxRes = cuCtxPopCurrent(&pLastCtxt);
        if(eCtxRes != CUDA_SUCCESS) {
            assert(eCtxRes == CUDA_SUCCESS);
            PTask::Runtime::HandleError("%s::%s::%s returned %d!\n",
                                        __FILE__,
                                        __FUNCTION__,
                                        "cuCtxPopCurrent",
                                        eCtxRes);
        }
        assert(pLastCtxt == m_pContext);

        m_bInitialized = TRUE;
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if this accelerator has support for unified addressing. Unified
    ///             addressing means there is no distinction between device and host pointers (for
    ///             page-locked memory). This is important because the datablock abstraction
    ///             maintains a buffer per logical memory space, and if two memory spaces are
    ///             logically the same (unified), but only for pointers to page-locked memory, a
    ///             number of special cases arise for allocation, freeing, ownership, etc. Sadly,
    ///             this complexity is required in the common case, because asynchronous transfers
    ///             only work in CUDA when the host pointers are page-locked. We need to be able to
    ///             tell when a page-locked buffer in the host-memory space is different from a
    ///             device pointer in a CUAccelerator memory space.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    /// 
    /// <returns>   true if the device supports unified addressing. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    CUAccelerator::SupportsUnifiedAddressing(
        VOID
        )
    {
        assert(m_bInitialized);
        return m_attrs.unifiedAddressing;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the device. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the device. </returns>
    ///-------------------------------------------------------------------------------------------------

    void*			
    CUAccelerator::GetDevice() { 
        return (void*) m_pDevice;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the context. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the context. </returns>
    ///-------------------------------------------------------------------------------------------------

    void*	
    CUAccelerator::GetContext() {
        return m_pContext;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates an asynchronous context for the task. Create the cuda stream for this
    ///             ptask.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 12/20/2011.
    ///             
    ///             This method is required of all subclasses, and abstracts the work associated with
    ///             managing whatever framework-level asynchrony abstractions are supported by the
    ///             backend target. For example, CUDA supports the "stream", while DirectX supports
    ///             an ID3D11ImmediateContext, OpenCL has command queues, and so on.
    ///             </remarks>
    ///
    /// <param name="pTask">                [in] non-null, the CUDA-capable acclerator to which the
    ///                                     stream is bound. </param>
    /// <param name="eAsyncContextType">    Type of the asynchronous context. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    AsyncContext * 
    CUAccelerator::PlatformSpecificCreateAsyncContext(
        __in Task * pTask,
        __in ASYNCCONTEXTTYPE eAsyncContextType
        )
    {
        return new CUAsyncContext(this, pTask, eAsyncContextType);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Cache a compiled shader. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="szFile">   [in,out] If non-null, the file. </param>
    /// <param name="szFunc">   [in,out] If non-null, the func. </param>
    /// <param name="p">        The p. </param>
    /// <param name="m">        The m. </param>
    ///-------------------------------------------------------------------------------------------------

    void					
    CUAccelerator::CachePutShader(
        char * szFile, 
        char * szFunc, 
        CUfunction p,
        CUmodule m
        )
    {
        Lock();
        UNREFERENCED_PARAMETER(p);
        UNREFERENCED_PARAMETER(m);
        UNREFERENCED_PARAMETER(szFile);
        UNREFERENCED_PARAMETER(szFunc);
        // m_pCache->CachePut(szFile, szFunc, p);
        // p->AddRef();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check the cache for the given compiled shader </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="szFile">   [in,out] If non-null, the file. </param>
    /// <param name="szFunc">   [in,out] If non-null, the func. </param>
    /// <param name="p">        [in,out] The p. </param>
    /// <param name="m">        [in,out] The m. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    CUAccelerator::CacheGetShader(
        char * szFile, 
        char * szFunc,
        CUfunction &p,
        CUmodule &m
        )
    {
        Lock();
        UNREFERENCED_PARAMETER(szFile);
        UNREFERENCED_PARAMETER(szFunc);
        // result = m_pCache->CacheGet(szFile, szFunc);
        p = NULL;
        m = NULL;
        Unlock();
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a new platform specific buffer. This routine is called by CreateBuffer to
    ///             get a new instance of whatever buffer type corresponds to the platform
    ///             implementing this interface. For example, DXAccelerator will return a new
    ///             PDXBuffer object, where PDXBuffer is a subclass of PBuffer. The Accelerator super-
    ///             class can then perform the rest of the work required to initialize the PBuffer.
    ///             
    ///             We only create PBuffers to provide 'physical' views of the 'logical' buffer
    ///             abstraction provided by the Datablock. Datablocks can have up to three different
    ///             channels (data, metadata, template), so consequently, each of which must be
    ///             backed by its own PBuffer. A PBuffer should not have to know what channel it is
    ///             backing, but we include that information in it's creation to simplify the
    ///             materialization of views between different subclasses of PBuffer.
    ///             
    ///             The "proxy allocator" is present as parameter to handle two corner cases:
    ///             
    ///             1. Allocation of host-side buffers by the host-specific subclass of PBuffer
    ///                (PHBuffer)--for example, we prefer to use a CUDA accelerator object to
    ///                allocate host memory when a block will be touched by a CUDA-based PTask,
    ///                because we can use the faster async APIs with memory we allocate using CUDA
    ///                host allocation APIs. This requires that the HostAccelerator defer the host-
    ///                side memory allocation to the CUDA accelerator.
    ///             
    ///             2. Communication between runtimes that provide some interop support (e.g. CUDA
    ///                and DirectX can actually share texture objects, meaning there is no need to
    ///                actually allocate a new buffer to back a CUDA view that already has a DirectX
    ///                view, but the two accelerators must cooperate to assemble a PBuffer that
    ///                shares the underlying shared object.
    ///             
    ///             Case 1 is implemented, while case 2 is largely unimplemented. If no proxy
    ///             accelerator is provided, allocation will proceed using the accelerator object
    ///             whose member function is being called to allocate the PBuffer.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pLogicalParent">           [in] If non-null, the datablock that is the logical
    ///                                         buffer using this 'physical' buffer to back a particular
    ///                                         channel on this accelerator. </param>
    /// <param name="nDatblockChannelIndex">    Zero-based index of the channel being backed. Must be:
    ///                                         * DBDATA_IDX = 0, OR
    ///                                         * DBMETADATA_IDX = 1, OR
    ///                                         * DBTEMPLATE_IDX = 2. </param>
    /// <param name="uiBufferAccessFlags">      Access flags determining what views to create. </param>
    /// <param name="pProxyAllocator">          [in,out] If non-null, the proxy allocator. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    PBuffer*	
    CUAccelerator::NewPlatformSpecificBuffer(
        Datablock * pLogicalParent, 
        UINT nDatblockChannelIndex, 
        BUFFERACCESSFLAGS uiBufferAccessFlags, 
        Accelerator * pProxyAllocator
        )
    {
        return new PCUBuffer(pLogicalParent, 
                             uiBufferAccessFlags, 
                             nDatblockChannelIndex, 
                             this, 
                             pProxyAllocator);
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Compiles. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pszSrcFile">               [in,out] If non-null, source file. </param>
    /// <param name="pFunctionName">            [in,out] If non-null, name of the function. </param>
    /// <param name="ppPlatformSpecificBinary"> [in,out] If non-null, the platform specific binary. </param>
    /// <param name="ppPlatformSpecificModule"> [in,out] If non-null, the platform specific module. </param>
    /// <param name="lpszCompilerOutput">       [in,out] If non-null, the compiler output. </param>
    /// <param name="uiCompilerOutput">         The compiler output. </param>
    /// <param name="tgx">                      The tgx. </param>
    /// <param name="tgy">                      The tgy. </param>
    /// <param name="tgz">                      The tgz. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL  
    CUAccelerator::Compile( 
        char* pszSrcFile, 
        char* pFunctionName,
        void ** ppPlatformSpecificBinary,
        void ** ppPlatformSpecificModule,
        char * lpszCompilerOutput,
        int uiCompilerOutput,
        int tgx, 
        int tgy, 
        int tgz
        )
    {
    #pragma warning(disable:4996)
        FILE *fp = fopen(pszSrcFile, "rb");
        if (NULL == fp) {
            sprintf_s(
                m_lpszUserMessages, 
                USER_OUTPUT_BUF_SIZE, 
                "Could not find file %s\n", 
                pszSrcFile);
            PTask::Runtime::ErrorMessage(m_lpszUserMessages);
            assert(false);
            return NULL;
        }
        fseek(fp, 0, SEEK_END);

        int file_size = ftell(fp);
        char *buf = new char[file_size+1];
        fseek(fp, 0, SEEK_SET);
        fread(buf, sizeof(char), file_size, fp);
        fclose(fp);
        buf[file_size] = '\0';

		BOOL bResult = Compile(buf, 
                        file_size, 
                        pFunctionName, 
                        ppPlatformSpecificBinary, 
                        ppPlatformSpecificModule, 
                        lpszCompilerOutput,
                        uiCompilerOutput,
                        tgx, 
                        tgy, 
                        tgz);

		

        delete[] buf;

        if(bResult) {
            CachePutShader((char*)pszSrcFile, 
                           (char*)pFunctionName, 
                           (CUfunction)*ppPlatformSpecificBinary, 
                           (CUmodule)*ppPlatformSpecificModule);
            return TRUE; 
        }

        return FALSE;
    #pragma warning(default:4996)
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Compiles accelerator source code to create a PTask binary. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011.
    ///             
    ///             The function accepts a string of source code and an operation in that source to
    ///             build a binary for.  This is a convenience for source code that may not be stored
    ///             in files (e.g. dynamically generated code). On success the function will create
    ///             platform- specific binary and module objects that can be later used by the
    ///             runtime to invoke the shader code. The caller can provide a buffer for compiler
    ///             output, which if present, the runtime will fill *iff* the compilation fails.
    ///             
    ///             NB: Thread group dimensions are optional parameters here. This is because some
    ///             runtimes require them statically, and some do not. DirectX requires thread-group
    ///             sizes to be specified statically to enable compiler optimizations that cannot be
    ///             used otherwise. CUDA and OpenCL allow runtime specification of these parameters.
    ///             </remarks>
    ///
    /// <param name="lpszShaderCode">           [in] actual source. cannot be null. </param>
    /// <param name="uiShaderCodeSize">         Size of the shader code. </param>
    /// <param name="lpszOperation">            [in] Function name in source file. cannot be null. </param>
    /// <param name="ppPlatformSpecificBinary"> [out] On success, a platform specific binary. </param>
    /// <param name="ppPlatformSpecificModule"> [out] On success, a platform specific module handle. </param>
    /// <param name="lpszCompilerOutput">       (optional) [in,out] On failure, the compiler output. </param>
    /// <param name="uiCompilerOutput">         (optional) [in] length of buffer supplied for
    ///                                         compiler output. </param>
    /// <param name="nThreadGroupSizeX">        (optional) thread group X dimensions. (see remarks) </param>
    /// <param name="nThreadGroupSizeY">        (optional) thread group Y dimensions. (see remarks) </param>
    /// <param name="nThreadGroupSizeZ">        (optional) thread group Z dimensions. (see remarks) </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    CUAccelerator::Compile(
        __in char *  lpszShaderCode, 
        __in UINT    uiShaderCodeSize,
        __in char *  lpszOperation, 
        __in void ** ppPlatformSpecificBinary,
        __in void ** ppPlatformSpecificModule,
        __in char *  lpszCompilerOutput,
        __in int     uiCompilerOutput,
        __in int     nThreadGroupSizeX, 
        __in int     nThreadGroupSizeY, 
        __in int     nThreadGroupSizeZ 
        ) {
    #pragma warning(disable:4996)
        UNREFERENCED_PARAMETER(nThreadGroupSizeX);
        UNREFERENCED_PARAMETER(nThreadGroupSizeY);
        UNREFERENCED_PARAMETER(nThreadGroupSizeZ);
        UNREFERENCED_PARAMETER(uiShaderCodeSize);

        *ppPlatformSpecificBinary = NULL;
        *ppPlatformSpecificModule = NULL;
        CUmodule pModule = NULL;
        CUresult error = CUDA_SUCCESS;
        CUfunction pShader = NULL;

        // Create module from binary file (PTX only)
        const unsigned int jitNumOptions = 3;
        CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
        void **jitOptVals = new void*[jitNumOptions];

        // set up size of compilation log buffer
        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        int jitLogBufferSize = 1024;
        jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

        // set up pointer to the compilation log buffer
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;

        //// set up pointer to set the Maximum # of registers for a particular kernel
        //jitOptions[2] = CU_JIT_MAX_REGISTERS;
        //int jitRegCount = 32;
        //jitOptVals[2] = (void *)(size_t)jitRegCount;
        
        jitOptions[2] = CU_JIT_TARGET_FROM_CUCONTEXT;
        jitOptVals[2] = NULL;

        Lock();
        MakeDeviceContextCurrent();

        // CUresult quack = cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 100);
        // assert(quack == CUDA_SUCCESS);

        trace("cuModuleLoadDataEx");
        DWORD dwStartJIT = GetTickCount();
        if(CUDA_SUCCESS != (error = cuModuleLoadDataEx(&pModule, 
                                                       lpszShaderCode, 
                                                       jitNumOptions, 
                                                       jitOptions, 
                                                       (void **)jitOptVals))) {
            char * p = jitLogBuffer;
            std::stringstream ss;
            ss << "> PTX JIT log:\n" << p;
            while(*p != '\0' && p != NULL) {
                ss << ">\t" << p << std::endl;
                p = strchr(p, '\0');
            }
            const char * szOutput = ss.str().c_str();
            if(lpszCompilerOutput != NULL) {
                memset(lpszCompilerOutput, 0, uiCompilerOutput);
                UINT nOutputLength = (UINT) ss.str().length();
                UINT nCopyLength = ((UINT) uiCompilerOutput > nOutputLength) ? nOutputLength : uiCompilerOutput - 1;
                memcpy(lpszCompilerOutput, szOutput, nCopyLength);
            }
            PTask::Runtime::ErrorMessage(szOutput);
            delete [] jitOptions;
            delete [] jitOptVals;
            delete [] jitLogBuffer;
            assert(false);
            return NULL;
        } 
    
        if(PTask::Runtime::IsVerbose()) {
            // dump compiler output, but only if it's interesting!
            if(strlen(jitLogBuffer)) {
                sprintf_s(m_lpszUserMessages, USER_OUTPUT_BUF_SIZE, "> PTX JIT log:\n%s\n", jitLogBuffer);
                PTask::Runtime::Inform(m_lpszUserMessages);
            }
        }

        delete [] jitOptions;
        delete [] jitOptVals;
        delete [] jitLogBuffer;

        trace("cuModuleGetFunction");
        MARKRANGEENTER(L"cuModuleGetFunction");
        if(CUDA_SUCCESS != (error = cuModuleGetFunction(&pShader, pModule, lpszOperation))) {
            sprintf_s(m_lpszUserMessages, "Could not find function %s in module loaded.", lpszOperation);
            PTask::Runtime::Inform(m_lpszUserMessages);
            assert(false);
            return NULL;
        }
        MARKRANGEEXIT();
        DWORD dwEndJIT = GetTickCount();
        PTask::Runtime::Inform("JIT compilation: %d ms\n", dwEndJIT-dwStartJIT);
            
        ReleaseCurrentDeviceContext();
        Unlock();

        *ppPlatformSpecificBinary = (void*) pShader;
        *ppPlatformSpecificModule = (void*) pModule;
        return TRUE; 
    #pragma warning(default:4996)
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Makes a context current. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    CUAccelerator::MakeDeviceContextCurrent(
        VOID
        ) 
    {
        CheckContextTLSInitialized();
        CHECK_CONTEXT_INVARIANTS();

        // What is required to make this accelerator's device context current is dependent on a number
        // of factors, most of which can be inferred from the role the calling thread plays. This could
        // be a very simple operation since the cuCtxPush/Pop APIs enable a reentrant idiom that just
        // always pushes here, but our experience is that cuCtx* calls are more expensive than one
        // might think. So generally speaking, we make a concerted effort to avoid making these CUDA
        // api calls if it's possible, assigning each thread a default context that is always kept
        // current and can be used directly if it happens to be the right one. This strategy works
        // great when PTask is the only component creating device contexts, but when user code also
        // creates them (e.g. implicitly, by making cuda* RT API calls), we may be forced to make them
        // and we may be forced to manage and restore state we did not create.
        // 
        // We try to make the logic easier by keeping track of which threads are private to PTask (and
        // therefore should have no opportunity to encounter a device context created by user code). 
        // PTask threads can always safely use the default context if it is this one. If the thread
        // is either a known user (application) thread or a thread we've not encountered before (.NET
        // thread pool, GC finalizer, etc.) we need to be sure not to save/restore user context. 
        //
        // Start by trying to determine if we need to make an API call to check the current context.
        // This is required only when the thread is not ptask-private, and we are not already current
        // with this device context. 
        
        BOOL bUserThread = s_eThreadRole == PTTR_APPLICATION || s_eThreadRole == PTTR_UNKNOWN;
        BOOL bCurrentDeviceExists = s_pCurrentDeviceCtxt != NULL;
        BOOL bCurrentDeviceIsThis = s_pCurrentDeviceCtxt == this;
        UINT uiCurrentDevicePlatformIdx = bCurrentDeviceExists ? s_pCurrentDeviceCtxt->GetPlatformIndex() : 0xFFFFFFFF;
        UINT uiContextDepth = bCurrentDeviceExists ? s_vContextDepthMap[uiCurrentDevicePlatformIdx] : 0;
        BOOL bCurrentDevicePushed = (uiContextDepth > 1 && bCurrentDeviceIsThis) || (uiContextDepth > 0 && !bCurrentDeviceIsThis);
        BOOL bMustCheckUserContext = bUserThread &&                 // app thread or unknown worker thread
                                     s_pUserStackTop == NULL &&     // we haven't already stashed a user context
                                     !bCurrentDevicePushed;         // if the current device is already current because 
                                                                    // we explicitly *made* it current, meaning this is
                                                                    // a nested call. 

        BOOL bUserContextOnEntry = FALSE;
        if(bMustCheckUserContext) {

            // check for a user context on entry. 
            // if we find one, save it off before making
            // any change to the context state.
            CUcontext ctxt = NULL;
            CUresult eCtxGetResult = cuCtxGetCurrent(&ctxt);
            if(eCtxGetResult != CUDA_SUCCESS) {
                ONFAILURE(cuCtxGetCurrent, eCtxGetResult);
            }

            if(ctxt != NULL) {

                // if we encounter a current context on this
                // thread and it's not a ptask context, we
                // must track the user's context so we can restore it.
                
                BOOL bFamiliarContext = CheckContextProvenance(ctxt);
                BOOL bPTaskContext = IsPTaskContext(ctxt);
                BOOL bUserContext = IsUserContext(ctxt);
                assert(bFamiliarContext);
                if(bUserContext) {
                    assert(!bPTaskContext || m_bApplicationPrimaryContext);
                    assert(s_pUserStackTop == NULL);
                    assert(s_pUserStackTop != ctxt);
                    s_pUserStackTop = ctxt;
                    bUserContextOnEntry = TRUE;
                }                
            }
        }

        CUresult res = CUDA_SUCCESS;
        assert(!bUserContextOnEntry || s_pCurrentDeviceCtxt == s_pDefaultDeviceCtxt || s_pCurrentDeviceCtxt == NULL);
        if(s_pCurrentDeviceCtxt != this || bUserContextOnEntry) {
            
            // we are changing device context... if this is a ptask thread, the depth for this device on
            // this thread should be 0. If we can successfully push the new context, increment the depth,
            // set the thread-local acc ptr. if it's a user thread, we need to check whether there is
            // already a device context current and save it if there is. 
            
            assert(s_pDefaultDeviceCtxt != NULL || (s_eThreadRole == PTTR_UNKNOWN) || (s_eThreadRole == PTTR_APPLICATION));

            CUresult res = cuCtxPushCurrent(m_pContext);
            trace3("cuCtxPushCurrent(%8X:d%d)\n", m_pContext, s_vContextDepthMap[m_nPlatformIndex]);
            if(CUDA_SUCCESS == res) {
                int nNewDepth = ++s_vContextDepthMap[m_nPlatformIndex];
                CUAccelerator * pPushMapEntry = bUserContextOnEntry ? NULL : s_pCurrentDeviceCtxt;
                s_pContextChangeMap[m_nPlatformIndex][nNewDepth] = pPushMapEntry;
                s_pCurrentDeviceCtxt = this;
            } else {
                PTASSERT(res == CUDA_SUCCESS);
                ONFAILURE(cuCtxPushCurrent, res);
            } 
                
        } else {

            // this is already the current device context for this thread. 
            PTASSERT(s_eThreadRole != PTTR_UNKNOWN);
            PTASSERT(s_vContextDepthMap[m_nPlatformIndex] > 0);
            int nNewDepth = ++s_vContextDepthMap[m_nPlatformIndex];
            s_pContextChangeMap[m_nPlatformIndex][nNewDepth] = NULL;

        }
        CHECK_CONTEXT_INVARIANTS();
        return res == CUDA_SUCCESS;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the current context described by.  </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    CUAccelerator::ReleaseCurrentDeviceContext(
        VOID
        )
    {
        CHECK_CONTEXT_INVARIANTS();

        assert(s_vContextDepthMap[m_nPlatformIndex] > 0);
        assert(s_pCurrentDeviceCtxt == this);

        if(s_vContextDepthMap[m_nPlatformIndex] <= 0) {
            if(PTask::Runtime::HandleError("%s::%s: mismanaged device context depth for acc=%s on thread=%d!\n",
                                           __FILE__,
                                           __FUNCTION__,
                                           m_lpszDeviceName,
                                           ::GetCurrentThreadId()))
                                           return;

        } 


        if(s_vContextDepthMap[m_nPlatformIndex] == 1 && 
            s_pDefaultDeviceCtxt == s_pCurrentDeviceCtxt) {

            // this is the default device context for this thread.
            // don't actually do anything...leave the device
            // context current, but complain. this case really shouldn't happen
            // because it means there is a mismatched acquire/release pair. 
            assert(!(s_vContextDepthMap[m_nPlatformIndex] == 1 &&  s_pDefaultDeviceCtxt == s_pCurrentDeviceCtxt));
            assert(s_pUserStackTop == NULL);

        } else {

            assert(s_pCurrentDeviceCtxt != NULL);

            int nOldDepth = s_vContextDepthMap[m_nPlatformIndex]--;
            CUAccelerator * pNextCurrent = s_pContextChangeMap[m_nPlatformIndex][nOldDepth];
            s_pContextChangeMap[m_nPlatformIndex][nOldDepth] = NULL;
            BOOL bStackBottom = s_pDefaultDeviceCtxt == NULL ? nOldDepth == 1 : nOldDepth == 2;
            BOOL bRestoreUserContext = pNextCurrent == NULL && bStackBottom && s_pUserStackTop != NULL;
            BOOL bRestoreEmptyContext = bStackBottom && this == s_pCurrentDeviceCtxt && s_pDefaultDeviceCtxt == NULL;
            BOOL bCtxCallRequired = (pNextCurrent != NULL || bRestoreUserContext || bRestoreEmptyContext);

            if(bCtxCallRequired) {

                // there was actually a cuCtx* call associated with this
                // context change. We need to do corresponding pop call
                // to make sure we get back to the correct previous state                    

                CUresult popres = CUDA_SUCCESS;
                CUresult getres = CUDA_SUCCESS;
                CUcontext poppedContext = NULL;
                CUcontext newContext = NULL;
                popres = cuCtxPopCurrent(&poppedContext);
                getres = cuCtxGetCurrent(&newContext);
                trace3("cuCtxPopCurrent(%8X:d%d)\n", m_pContext, s_vContextDepthMap[m_nPlatformIndex]);
                if(popres != CUDA_SUCCESS) 
                    ONFAILURE(cuCtxPopCurrent, popres);
                if(getres != CUDA_SUCCESS)
                    ONFAILURE(cuCtxGetCurrent, getres);

                // should be back to previous context
                assert(poppedContext != NULL);
                assert(pNextCurrent == NULL || pNextCurrent->m_pContext == newContext);
                assert(s_pCurrentDeviceCtxt->m_pContext == poppedContext);
                assert(newContext == s_pUserStackTop || !bRestoreUserContext);
                assert(newContext == NULL || !(bRestoreEmptyContext && !bRestoreUserContext));

                if(bRestoreUserContext || bRestoreEmptyContext) {

                    assert(s_pUserStackTop != NULL || (bRestoreEmptyContext && !bRestoreUserContext));
                    assert(s_pDefaultDeviceCtxt == NULL);
                    assert(!Accelerator::ShouldHaveDefaultContext(s_eThreadRole));
                    s_pCurrentDeviceCtxt = NULL;
                    s_pUserStackTop = NULL;

                } else {
                    s_pCurrentDeviceCtxt = pNextCurrent;
                    assert(s_pCurrentDeviceCtxt != NULL);
                    assert(!bStackBottom);
                }
                    
            } else {

                // This release had no cuCtx* calls associated with it.
                // We have decremented the context depth and must take action only if we
                // have hit the logical "stack bottom" on a thread that does not maintain
                // a default context. Make sure that this is actually the current device 
                // if we have not reached the logical "stack bottom" (which should be handled
                // above...)
                
                    
                assert(s_vContextDepthMap[m_nPlatformIndex] != 0);
                assert(s_pCurrentDeviceCtxt == this);

            }
        }
        CHECK_CONTEXT_INVARIANTS();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Query if the given accelerator has a memory space accessible to this one through APIs
    ///     (rather than by copying through host memory).
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="p">    [in,out] The p. </param>
    ///
    /// <returns>   true if accessible memory space, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CUAccelerator::HasAccessibleMemorySpace(
        Accelerator*p
        )
    {
        if(p->GetClass() == ACCELERATOR_CLASS_CUDA)
            return TRUE;
        // XXX: TODO: 
        // there is probably a way to do this between
        // CUDA and DX accelerators, since CUDA supports
        // some DirectCompute interoperation.
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if this accelerator has some support for device side memcpy.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CUAccelerator::SupportsDeviceMemcpy(
        VOID
        )
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if this accelerator has some support for device to device transfer
    ///             with the given accelerator. This allows us to skip a trip through host memory in
    ///             many cases. The strategy here is to quickly eliminate cases that cannot be
    ///             supported or are not implemented. If we find a case where we have implemented
    ///             support for the combination of platforms, check for cached result. If we can't
    ///             find it, figure out through CUDA APIs whether P2P support is available, cache the
    ///             result, and return it.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   true if P2P support is available, false otherwise. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CUAccelerator::SupportsDeviceToDeviceTransfer(
        __in Accelerator * pAccelerator
        )
    {
        // in many cases we can tell just by looking at the accelerator class combination whether there
        // specialized P2P transfer operations are even an option. In general, except where there is
        // interop support from the platforms, different accelerator classes are assumed to be
        // incapable of P2P communication. So check the class combination first. 
        // 
        assert(pAccelerator != NULL);
        assert(pAccelerator != this);
        switch(pAccelerator->GetClass()) {           

        case ACCELERATOR_CLASS_OPEN_CL: 
        case ACCELERATOR_CLASS_REFERENCE:
        case ACCELERATOR_CLASS_HOST: 
        case ACCELERATOR_CLASS_UNKNOWN: 

            // peer-to-peer transfer from CUDA to these runtimes is unimplemented (AFAIK). There may be
            // some interop support for OpenCL, but for now, if it can be done, we don't expose it. 
            return FALSE;

        case ACCELERATOR_CLASS_DIRECT_X: 

            // CUDA supports DX interop, so presumably this is actually possible. However, we don't support
            // it yet. Return false accordingly, but complain about the missed opportunity so that if a use-
            // case arises where this is important, the programmer knows it's possible to do the right
            // thing. 
            PTask::Runtime::Warning("Request for CU->DX P2P transfer. Possible, but unimplemented!");
            return FALSE; 

        case ACCELERATOR_CLASS_CUDA: 

            // CUDA-CUDA peer transfer dependes on device caps.
            // logic to determine connectivity below.
            break;
        }

        CUAccelerator * pA = this;
        CUAccelerator * pB = (CUAccelerator*) pAccelerator;
        assert(pA->LockIsHeld());
        assert(pB->LockIsHeld());
        assert(pA != NULL);
        assert(pB != NULL);

        if(pA->m_vP2PAccessible.find(pB) != pA->m_vP2PAccessible.end()) {
            // If the two accelerators are on their respective known 
            // accessible lists, we are done. We've already checked, and 
            // there is P2P transfer support for this combination.
            assert(pB->m_vP2PAccessible.find(pA) != pB->m_vP2PAccessible.end());
            return TRUE;
        }

        if(pA->m_vP2PInaccessible.find(pB) != pA->m_vP2PInaccessible.end()) {
            // If the two accelerators are on their respective known 
            // inaccessible lists, we are done. We've already checked, and 
            // there is NO P2P transfer support for this combination.
            assert(pB->m_vP2PInaccessible.find(pA) != pB->m_vP2PInaccessible.end());
            return FALSE;
        }

        CUdevice aDev = (CUdevice) pA->GetDevice();
        CUdevice bDev = (CUdevice) pB->GetDevice();
        BOOL bBCanAccessA = FALSE;
        BOOL bACanAccessB = FALSE;
        BOOL bBAccessAEnabled = FALSE;
        BOOL bAAccessBEnabled = FALSE;

        pB->MakeDeviceContextCurrent();
        assert(!pA->IsDeviceContextCurrent());
        CUresult canAccessResA = cuDeviceCanAccessPeer(&bBCanAccessA, bDev, aDev);
        trace5("cuDeviceCanAccessPeer(%16llX, %16llX)->%d, res=%d\n", bDev, aDev, bBCanAccessA, canAccessResA);
        CUresult enableAccessResA = cuCtxEnablePeerAccess((CUcontext) pA->GetContext(), 0);
        trace3("cuCtxEnablePeerAccess(%16llX, 0), res=%d\n", aDev, enableAccessResA);
        bBAccessAEnabled = enableAccessResA == CUDA_SUCCESS;
        pB->ReleaseCurrentDeviceContext();

        pA->MakeDeviceContextCurrent();
        assert(!pB->IsDeviceContextCurrent());
        CUresult canAccessResB = cuDeviceCanAccessPeer(&bACanAccessB, aDev, bDev);
        trace5("cuDeviceCanAccessPeer(%16llX, %16llX)->%d, res=%d\n", aDev, bDev, bACanAccessB, canAccessResB);
        CUresult enableAccessResB = cuCtxEnablePeerAccess((CUcontext) pB->GetContext(), 0);
        trace3("cuCtxEnablePeerAccess(%16llX, 0), res=%d\n", bDev, enableAccessResB);
        bAAccessBEnabled = enableAccessResB == CUDA_SUCCESS;
        pA->ReleaseCurrentDeviceContext();

        BOOL bP2PSupport = canAccessResA == CUDA_SUCCESS && 
                           canAccessResB == CUDA_SUCCESS && 
                           bBCanAccessA && 
                           bACanAccessB;

        if(bP2PSupport) {
            pA->m_vP2PAccessible.insert(pB); 
            pB->m_vP2PAccessible.insert(pA);
            pA->m_vP2PEnabled.insert(pB);
            pB->m_vP2PEnabled.insert(pA);
        } else {
            pA->m_vP2PInaccessible.insert(pB);
            pB->m_vP2PInaccessible.insert(pA);
        }

        return bP2PSupport;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Device to device transfer. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pDstBuffer">       [in,out] If non-null, the accelerator. </param>
    /// <param name="pSrcBuffer">       [in,out] If non-null, buffer for source data. </param>
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    CUAccelerator::DeviceToDeviceTransfer(
        __inout PBuffer *       pDstBuffer,
        __in    PBuffer *       pSrcBuffer,
        __in    AsyncContext *  pAsyncContext
        ) 
    {
        PCUBuffer * pCUBuffer = dynamic_cast<PCUBuffer*>(pSrcBuffer);
        return pCUBuffer->DeviceToDeviceTransfer(pDstBuffer, pAsyncContext);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if this device supports pinned host memory. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CUAccelerator::SupportsPinnedHostMemory(
        VOID
        )
    {
        return this->m_attrs.canMapHostMemory;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate memory on the host. Some runtimes (esp. earlier versions of CUDA)
    ///             require that CUDA APIs be used to allocate host-side buffers, or support
    ///             specialized host allocators that can help improve DMA performance.
    ///             AllocatePagelockedHostMemory wraps these APIs for accelerators that have runtime support
    ///             for this, and uses normal system services (VirtualAlloc on Windows, malloc
    ///             elsewhere) to satisfy requests.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <param name="uiBytes">              Number of bytes to allocate. </param>
    /// <param name="pbResultPageLocked">   [in,out] If non-null, the result of whether the allocated
    ///                                     memory is page-locked is provided here. </param>
    ///
    /// <returns>   byte pointer on success, null on failure. </returns>
    ///-------------------------------------------------------------------------------------------------

    void *      
    CUAccelerator::AllocatePagelockedHostMemory(
        UINT uiBytes, 
        BOOL * pbResultPageLocked
        )
    {
        void * pResult = NULL;
        ACQUIRE_CTX(this);
        if(PTask::Runtime::GetPageLockingEnabled() && 
           ShouldAttemptPageLockedAllocation(uiBytes)) {
            unsigned int flags = 
                CU_MEMHOSTALLOC_PORTABLE | 
                CU_MEMHOSTALLOC_DEVICEMAP;
            CUresult res = cuMemHostAlloc(&pResult, uiBytes, flags);
            trace4("cuMemHostAlloc(%d)->%16llX, res=%d\n", uiBytes, pResult, res);
            if(res != CUDA_SUCCESS) {
                assert(res == CUDA_SUCCESS);
                PTask::Runtime::HandleError("%s: memallochost for %d bytes failed!\n", __FUNCTION__, uiBytes);
                *pbResultPageLocked = FALSE;
                pResult = NULL;
            } else {
                assert(pbResultPageLocked != NULL);
                *pbResultPageLocked = TRUE;
                RecordAllocation(pResult, TRUE, uiBytes);
            }
        } 

        if(pResult == NULL) {
            pResult = MemorySpace::AllocateMemoryExtent(HOST_MEMORY_SPACE_ID, uiBytes, 0);
            *pbResultPageLocked = FALSE;
        }

        RELEASE_CTX(this);
        return pResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Free host memory. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <param name="pBuffer">      If non-null, the buffer. </param>
    /// <param name="bPageLocked">  true if the memory was allocated in the page-locked area. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    CUAccelerator::FreeHostMemory(
        void * pBuffer,
        BOOL bPageLocked
        )
    {
        ACQUIRE_CTX(this);
        if(bPageLocked) {
            assert(PTask::Runtime::GetPageLockingEnabled());
            CUresult res = cuMemFreeHost(pBuffer);
            trace2("cuMemFreeHost()->res=%d\n", res);
            if(res != CUDA_SUCCESS) {
                PTASSERT(res == CUDA_SUCCESS);
                PTask::Runtime::HandleError("%s: cuMemFreeHost failed\n", __FUNCTION__);
            } else {
                RecordDeallocation(pBuffer);
            }
        } else {
            MemorySpace::DeallocateMemoryExtent(HOST_MEMORY_SPACE_ID, pBuffer);
        }
        RELEASE_CTX(this);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synchronizes the context. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="pContext"> [in,out] If non-null, the context. </param>
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL         
    CUAccelerator::Synchronize(
        Task * pTask
        )
    {
        BOOL bSuccess = TRUE;
        Lock();
        if(pTask) {
            std::map<Task*, AsyncContext*>::iterator mi;
            mi=m_vTaskAsyncContextMap.find(pTask);
            assert(mi!=m_vTaskAsyncContextMap.end());
            if(mi!=m_vTaskAsyncContextMap.end()) {
                AsyncContext * pAsyncContext = mi->second;
                pAsyncContext->SynchronizeContext();
            }
        } else { 
            ACQUIRE_CTXNL(this);
            trace("cuCtxSynchronize");
            CUresult err = cuCtxSynchronize();
            if(err != CUDA_SUCCESS) {
                assert(err == CUDA_SUCCESS);
                Runtime::HandleError("%s:%s (task=%s) failed with err=%d\n",
                                     __FILE__,
                                     __FUNCTION__,
                                     pTask->GetTaskName(),
                                     err);
            }
            RELEASE_CTXNL(this);
            std::map<Task*, AsyncContext*>::iterator mi;
            std::map<ASYNCCONTEXTTYPE, AsyncContext*>::iterator ti;
            for(mi=m_vTaskAsyncContextMap.begin(); mi!=m_vTaskAsyncContextMap.end(); mi++) {
                AsyncContext * pAsyncContext = mi->second;
                pAsyncContext->NotifyDeviceSynchronized();
            }
            for(ti=m_vDeviceAsyncContexts.begin(); ti!=m_vDeviceAsyncContexts.end(); ti++) {
                AsyncContext * pAsyncContext = ti->second;
                pAsyncContext->NotifyDeviceSynchronized();
            }
        }
        Unlock();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a context current. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CUAccelerator::IsDeviceContextCurrent(
        VOID
        ) 
    { 
        CheckContextTLSInitialized();
        CHECK_CONTEXT_INVARIANTS();

        // the device is current if s_pCurrentDeviceCtxt is this accelerator,
        // *unless* application threads can have default contexts and there
        // are CUDA contexts created in user code that we need to save/restore. 
        // This should only happen if there is a default device and it's at 
        // the bottom of the context stack

        BOOL bAccObjectMatch = s_pCurrentDeviceCtxt == this;
        if(!bAccObjectMatch)
            return bAccObjectMatch;     // this is always decisive. 
       
        BOOL bCurrentIsDefault = bAccObjectMatch && 
                                 s_pCurrentDeviceCtxt == s_pDefaultDeviceCtxt &&
                                 s_vContextDepthMap[s_pCurrentDeviceCtxt->m_nPlatformIndex] == 1;
        BOOL bConservativeCheckReq = bCurrentIsDefault && 
                                    s_eThreadRole == PTTR_APPLICATION &&                                    
                                    !PTask::Runtime::GetApplicationThreadsManagePrimaryContext();

        if(!bConservativeCheckReq) 
            return bAccObjectMatch;

        CUcontext ctxt = NULL;
        CUresult eCtxGetRes = cuCtxGetCurrent(&ctxt);
        if(eCtxGetRes != CUDA_SUCCESS) {
            ONFAILURE(cuCtxGetCurrent, eCtxGetRes);
        }
        return ctxt == m_pContext;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Supports function arguments. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CUAccelerator::SupportsFunctionArguments(
        VOID
        ) 
    { 
        return TRUE; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Supports byval arguments. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CUAccelerator::SupportsByvalArguments(
        VOID
        ) 
    { 
        return TRUE; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a device identifier. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The device identifier. </returns>
    ///-------------------------------------------------------------------------------------------------

    int             
    CUAccelerator::GetDeviceId(
        VOID
        ) 
    { 
        return m_nDeviceId; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the cuda runtime has been initialized. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   true if cuda initialized, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL             
    CUAccelerator::IsCUDAInitialized(
        VOID
        ) 
    { 
        return s_bCUDAInitialized; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the device attributes. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the device attributes. </returns>
    ///-------------------------------------------------------------------------------------------------

    CUDA_DEVICE_ATTRIBUTES* 
    CUAccelerator::GetDeviceAttributes(
        VOID
        ) 
    { 
        return m_pDeviceAttributes; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Determine if we should attempt page locked allocation. </summary>
    ///
    /// <remarks>   Crossbac, 9/24/2012. </remarks>
    ///
    /// <param name="uiAllocBytes"> The allocate in bytes. </param>
    ///
    /// <returns>   true if we should page-lock the requested buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CUAccelerator::ShouldAttemptPageLockedAllocation(
        UINT uiAllocBytes
        )
    {
        UNREFERENCED_PARAMETER(uiAllocBytes);
        if(!PTask::Runtime::GetPageLockingEnabled())
            return FALSE;
#if 0
        std::stringstream strMessage;
        strMessage 
            << "recommending page lock of " 
            << uiAllocBytes
            << ", currently "
            << m_uiPageLockedBytesAllocated
            << " page-locked bytes are allocated by "
            << m_uiPageLockedBuffersAllocated
            << " buffers"
            << std::endl;
        PTask::Runtime::Inform(strMessage.str());
#endif
        return TRUE;
        //if(m_uiPageLockedBytesAllocated >= PAGE_LOCKED_BYTES_MAX)
        //    return FALSE;
        //if(m_uiPageLockedBuffersAllocated >= PAGE_LOCKED_BUFFERS_MAX)
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if this accelerator encapsulates a backend framework that provides
    ///             explicit APIs for managing outstanding (Asynchronous) operations. When this is
    ///             the case, the corresponding AsyncContext subclass can manage outstanding
    ///             dependences explicitly to increase concurrency and avoid syncing with the device.
    ///             When it is *not* the case, we must synchronize when we data to and from this
    ///             accelerator context and contexts that *do* support an explicit async API. For
    ///             example, CUDA supports the stream and event API to explicitly manage dependences
    ///             and we use this feature heavily to allow task dispatch to get far ahead of device-
    ///             side dispatch. However when data moves between CUAccelerators and other
    ///             accelerator classes, we must use synchronous operations or provide a way to wait
    ///             for outstanding dependences from those contexts to resolve. This method is used
    ///             to tell us whether we can create an outstanding dependence after making calls
    ///             that queue work, or whether we need to synchronize.
    ///             
    ///             This override returns TRUE since this is the CUDA encapsulation class.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    CUAccelerator::SupportsExplicitAsyncOperations(
        VOID
        )
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Determines if we can requires thread local context initialization. </summary>
    ///
    /// <remarks>   Crossbac, 3/20/2014. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CUAccelerator::UsesTLSContextManagement(
        VOID
        )
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check context TLS initialized. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2014. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    CUAccelerator::CheckContextTLSInitialized(
        VOID
        )
    {
        if(!s_bContextTLSInit) {
            assert(s_eThreadRole == PTTR_UNKNOWN);
            InitializeTLSContext(PTTR_UNKNOWN, FALSE, FALSE);
        }
    }

    void 
    CUAccelerator::InitializeTLSContext(
        __in PTTHREADROLE eRole,
        __in BOOL bMakeDefault,
        __in BOOL bPooledThread
        ) 
    {
        // initializes the following TLS data structures:
        // 
        //__declspec(thread) static CUAccelerator *  s_pDefaultDeviceCtxt;
        //__declspec(thread) static CUAccelerator *  s_pCurrentDeviceCtxt;
        //__declspec(thread) static int              s_vContextDepthMap[MAXCTXTS];
        //__declspec(thread) static CUAccelerator ** s_pContextChangeMap[MAXCTXTS];
        //__declspec(thread) static CUAccelerator *  s_vContextChangeMap[MAXCTXTS*MAXCTXDEPTH];
        //__declspec(thread) static CUcontext        s_vUserStackTop;
        //__declspec(thread) static CUcontext        s_vPTaskContexts[MAXCTXTS];
        //__declspec(thread) static CUcontext        s_vUserContexts[MAXCTXTS];
        //__declspec(thread) static UINT             s_nPTaskContexts;
        //__declspec(thread) static UINT             s_nUserContexts;
        //__declspec(thread) static BOOL             s_bContextTLSInit;
        //__declspec(thread) static PTTHREADROLE     s_eThreadRole;

        if(s_bContextTLSInit) {

            // there is one valid case where the TLS area may already be
            // initialized and it's ok to do nothing: if there are multiple
            // graphs sharing a thread pool, calls to this method that 
            // are triggered by graph::run can make multiple calls per thread. 
            if(!(eRole == PTTR_GRAPHRUNNER && bPooledThread)) {
                assert(eRole == PTTR_GRAPHRUNNER && bPooledThread);
                if(PTask::Runtime::HandleError("%s::%s(%d): illegal combination of thread type and thread pool membership\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                __LINE__)) {

                    // recoverable...de-init and allow re-init below
                    s_bContextTLSInit = FALSE;
                    s_pDefaultDeviceCtxt = NULL;
                    s_pCurrentDeviceCtxt = NULL;
                }
            }  
            assert(eRole == s_eThreadRole);
            assert(s_bThreadPoolThread == bPooledThread);
        } 

        if(!s_bContextTLSInit) {

            assert(s_pDefaultDeviceCtxt == NULL);
            assert(s_pCurrentDeviceCtxt == NULL);

            memset(s_vContextDepthMap , 0, MAXCTXTS*sizeof(int));
            memset(s_pContextChangeMap, 0, MAXCTXTS*sizeof(CUAccelerator*));
            memset(s_vContextChangeMap, 0, MAXCTXTS*MAXCTXDEPTH*sizeof(CUAccelerator*));

            for(int i=0; i<MAXCTXTS; i++) 
                s_pContextChangeMap[i] = &s_vContextChangeMap[i*MAXCTXDEPTH];

            s_pDefaultDeviceCtxt = NULL;
            s_pCurrentDeviceCtxt = NULL;
            s_pUserStackTop = NULL;
            s_eThreadRole = eRole;
            s_bThreadPoolThread = bPooledThread;
            s_vContextDepthMap[m_nPlatformIndex] = 0;
            s_pContextChangeMap[m_nPlatformIndex][0] = NULL;
            s_pContextChangeMap[m_nPlatformIndex][1] = NULL;


            if(bMakeDefault && ShouldHaveDefaultContext(s_eThreadRole)) {

                // set the cuda context from the given accelerator to be the 
                // default for this thread (meaning  API call for that device 
                // usually require no context management). First check to see if
                // there is already a context current on this thread (set up by
                // user code). If there is, fail and require the user to change
                // PTask settings...
                    
                CUcontext ctxt;
                CUresult eGetCtxRes = cuCtxGetCurrent(&ctxt);
                if(eGetCtxRes != CUDA_SUCCESS) {
                    ONFAILURE(cuCtxGetCurrent, eGetCtxRes);
                }

                if(ctxt != NULL) {

                    // user context. don't  set a default                
                    assert(s_eThreadRole == PTTR_APPLICATION);
                    assert(!IsKnownPTaskContext(ctxt));
                    if(!IsKnownUserContext(ctxt))
                        AddKnownUserContext(ctxt);

                    if(PTask::Runtime::HandleError("%s::%s(%d): detected user device context at TLS init for %s...\n"
                                                   " This application should use PTask::Runtime::SetApplicationThreadsManagePrimaryContext(TRUE)\n"
                                                   " exiting...\n",
                                                   __FILE__,
                                                   __FUNCTION__,
                                                   __LINE__,
                                                   m_lpszDeviceName))
                                                   return;                   
                } 

                CUresult eResult = cuCtxSetCurrent(m_pContext);
                if(eResult != CUDA_SUCCESS) {
                    ONFAILURE(cuCtxSetCurrent, eResult);
                }

                s_pDefaultDeviceCtxt = this;
                s_pCurrentDeviceCtxt = this;
                s_vContextDepthMap[m_nPlatformIndex] = 1;
                s_pContextChangeMap[m_nPlatformIndex][0] = NULL;
                s_pContextChangeMap[m_nPlatformIndex][1] = NULL;
            }

            s_vTLSDeinitializers[s_nTLSDeinitializers++] = DeinitializeTLSContextManagement;
            s_bContextTLSInit = TRUE;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deinitializes any thread local context data structures associated with the
    ///             accelerator. Most back-ends do nothing. Subclasses that actually do some TLS-
    ///             based context management (CUDA)
    ///             override.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/20/2014. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    CUAccelerator::DeinitializeTLSContextManagement(
        VOID
        )
    {
        if(s_bContextTLSInit) {

            if(s_pCurrentDeviceCtxt != NULL) {

                // there is a device whose context is current still. make sure it's current only because it's
                // the default. if there *is* no default, or we are not at the bottom stack depth, complain--
                // this means that there is still a sequence of calls on that device context underway. 
                
                int nContextDepth = s_vContextDepthMap[s_pCurrentDeviceCtxt->m_nPlatformIndex];
                int nStackBottom = s_pDefaultDeviceCtxt == s_pCurrentDeviceCtxt ? 1 : 0;
                if(nContextDepth != nStackBottom) {
                    if(PTask::Runtime::HandleError("%s::%s(%d): device context tls teardown while context for %s in use!\n",
                                                   __FILE__,
                                                   __FUNCTION__,
                                                   __LINE__,
                                                   s_pCurrentDeviceCtxt->m_lpszDeviceName)) 
                                                   return;
                }
            }

            if(s_pDefaultDeviceCtxt != NULL) {

                // unbind the context. for safety, check that the
                // context we are about to unbind is in fact the 
                // one we set up when we bound the default device.

                CUcontext ctxt = NULL;
                assert(s_pDefaultDeviceCtxt == s_pCurrentDeviceCtxt);
                CUresult cuGetRes = cuCtxGetCurrent(&ctxt);
                if(cuGetRes != CUDA_SUCCESS) {
                    ONFAILURE(cuCtxGetCurrent, cuGetRes);
                }
                if(ctxt != NULL) {
                    assert(s_pCurrentDeviceCtxt != NULL);
                    assert(ctxt == s_pDefaultDeviceCtxt->m_pContext);
                    assert(ctxt == s_pCurrentDeviceCtxt->m_pContext);
                    CUresult cuSetRes = cuCtxSetCurrent(NULL);
                    if(cuSetRes != CUDA_SUCCESS) {
                        ONFAILURE(cuCtxSetCurrent, cuSetRes);
                    }
                }
                s_pDefaultDeviceCtxt = NULL;
                s_pCurrentDeviceCtxt = NULL;
            }

            assert(s_pDefaultDeviceCtxt == NULL);
            assert(s_pCurrentDeviceCtxt == NULL);
            s_pDefaultDeviceCtxt = NULL;
            s_pCurrentDeviceCtxt = NULL;
            s_bContextTLSInit = FALSE;
            s_eThreadRole = PTTR_UNKNOWN;
            memset(s_vContextDepthMap , 0, MAXCTXTS*sizeof(int));
            memset(s_pContextChangeMap, 0, MAXCTXTS*sizeof(CUAccelerator*));
            memset(s_vContextChangeMap, 0, MAXCTXTS*MAXCTXDEPTH*sizeof(CUAccelerator*));

            for(int i=0; i<MAXCTXTS; i++) 
                s_pContextChangeMap[i] = &s_vContextChangeMap[i*MAXCTXDEPTH];

            s_pUserStackTop = NULL;
            s_bThreadPoolThread = FALSE;
            s_bContextTLSInit = FALSE;
        }
    }



};

#endif // CUDA_SUPPORT
