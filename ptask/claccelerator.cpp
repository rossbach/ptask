//--------------------------------------------------------------------------------------
// File: CLAccelerator.cpp
// Accelerator built on OpenCL interface
// maintaner: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

#ifdef OPENCL_SUPPORT
#include "primitive_types.h"
#include <assert.h>
#include "oclhdr.h"
#include "pclbuffer.h"
#include "claccelerator.h"
#include "datablocktemplate.h"
#include "PTaskRuntime.h"
#include "MemorySpace.h"
#include "CLAsyncContext.h"

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="lpszName"> [in,out] If non-null, the name. </param>
    ///-------------------------------------------------------------------------------------------------

    CLAccelerator::CLAccelerator(
        char * lpszName
        ) 
    {
        m_bInitialized = FALSE;
        m_pContext = NULL;
        m_pDevice = NULL;	
        m_cpPlatform = NULL;
        m_cqCommandQueue = NULL;
        m_class = ACCELERATOR_CLASS_OPEN_CL;
        if(!lpszName) 
            lpszName = "anonymous-opencl-dev";
        size_t nLength = strlen(lpszName)+10;
        m_lpszDeviceName = (char*) malloc(nLength);
        sprintf_s(m_lpszDeviceName, nLength, "%s:%d", lpszName, 0);
        std::string strName(m_lpszDeviceName);
        m_uiMemorySpaceId = MemorySpace::AssignUniqueMemorySpaceIdentifier();
        MemorySpace * pMemorySpace = new MemorySpace(strName, m_uiMemorySpaceId);
        MemorySpace::RegisterMemorySpace(pMemorySpace, this);
        Open(NULL);
        CreateAsyncContext(NULL, ASYNCCTXT_DEFAULT);
        assert(m_bInitialized);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="id">       The identifier. </param>
    /// <param name="lpszName"> [in,out] If non-null, the name. </param>
    ///-------------------------------------------------------------------------------------------------

    CLAccelerator::CLAccelerator(
        cl_device_id id,
        char * lpszName
        )
    {
        m_bInitialized = FALSE;
        m_class = ACCELERATOR_CLASS_OPEN_CL;
        m_pContext = NULL;
        m_pDevice = NULL;	
        m_cpPlatform = NULL;
        m_cqCommandQueue = NULL;
        if(!lpszName) 
            lpszName = "anonymous-opencl-dev";
        size_t nLength = strlen(lpszName)+100;
        m_lpszDeviceName = (char*) malloc(nLength);
        sprintf_s(m_lpszDeviceName, nLength, "%s:%d", lpszName, (int) id);
        std::string strName(m_lpszDeviceName);
        m_uiMemorySpaceId = MemorySpace::AssignUniqueMemorySpaceIdentifier();
        MemorySpace * pMemorySpace = new MemorySpace(strName, m_uiMemorySpaceId);
        MemorySpace::RegisterMemorySpace(pMemorySpace, this);
        Open(id);
        CreateAsyncContext(NULL, ASYNCCTXT_DEFAULT);
        assert(m_bInitialized);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    CLAccelerator::~CLAccelerator() {
        if(m_cqCommandQueue)clReleaseCommandQueue(m_cqCommandQueue);
        if(m_pContext)clReleaseContext(m_pContext);
        assert(m_lpszDeviceName != NULL);
        free(m_lpszDeviceName);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enumerate platforms. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="platforms">    [in,out] The platforms. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CLAccelerator::EnumeratePlatforms(
        std::vector<cl_platform_id> &platforms
        ) 
    {
        #pragma warning (disable:4701)
        cl_uint num_platforms; 
        cl_platform_id* clPlatformIDs;
        cl_int ciErrNum;
        cl_platform_id clSelectedPlatformID = 0;

        // Get OpenCL platform count
        ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
        if (ciErrNum != CL_SUCCESS)
            return;
        if(num_platforms == 0)
            return;

        // if there's a platform or more, make space for ID's
        if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL) {
            PTask::Runtime::HandleError("%s: OOM", __FUNCTION__);
        }

        // get platform info for each platform and trap the NVIDIA platform if found
        ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
#if 0
        char chBuffer[1024];
        for(cl_uint i = 0; i < num_platforms; ++i) {
            ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
            if(ciErrNum == CL_SUCCESS) {
                if(strstr(chBuffer, "NVIDIA") != NULL) {
                    clSelectedPlatformID = clPlatformIDs[i];
                    break;
                }
            }
        }
#else
        // CJR: 2-18-13: struggling with CUDA 5.0 distribution:
        // for some reason, when we make the call below, we get an exception because the
        // entry point cannot be found:
        // -----------------------------
        //   ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
        // -----------------------------
        // It's quite perplexing because it is quite plainly there: 
        // 
        //    C:\Windows\System32>dumpbin nvopencl.dll /EXPORTS
        //    Microsoft (R) COFF/PE Dumper Version 10.00.30319.01
        //    Copyright (C) Microsoft Corporation.  All rights reserved.
        //
        //
        //    Dump of file nvopencl.dll
        //
        //    File Type: DLL
        //
        //      Section contains the following exports for nvopencl.dll
        //
        //        00000000 characteristics
        //        506B30F6 time date stamp Tue Oct 02 11:22:46 2012
        //            0.00 version
        //               1 ordinal base
        //               3 number of functions
        //               3 number of names
        //
        //        ordinal hint RVA      name
        //
        //              1    0 000867A0 clGetExportTable
        //              2    1 000A0F80 clGetExtensionFunctionAddress
        //              3    2 000941F0 clGetPlatformInfo        
        //              
        // I'm hoping a subsequent release of CUDA will fix this. For now, 
        // since the    clGetPlatformIDs call does succeed, we just skip the
        // part where we make sure it's NVIDIA's implementation (unnecessary anyway)
        // and bark loudly about the fact that we are doing this...
        
        PTask::Runtime::Warning("WARNING: (2-18-13) Skipping search for NVIDIA OpenCL platform, defaulting to first platform, due to clGetPlatformInfo instability in CUDA 5.0\n");
        clSelectedPlatformID = clPlatformIDs[0];

#endif

        // default to zeroeth platform if NVIDIA not found
        if(clSelectedPlatformID == NULL) {
            PTask::Runtime::Warning("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!");
            clSelectedPlatformID = clPlatformIDs[0];
        }

        free(clPlatformIDs);
        platforms.push_back(clSelectedPlatformID);
        #pragma warning (default:4701)
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enumerate accelerators. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="vAccelerators">    [in,out] [in,out] If non-null, the accelerators. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    CLAccelerator::EnumerateAccelerators(
        std::vector<Accelerator*> &vAccelerators
        ) 
    {
#ifdef OPENCL_SUPPORT
        if(!PTask::Runtime::GetUseOpenCL())
            return;
        
        // Get OpenCL platform name and version
        std::string sProfileString;
        std::vector<cl_platform_id> platforms;
        std::vector<cl_device_id> devices;
        std::vector<Accelerator*> candidates;
        EnumeratePlatforms(platforms);
        cl_platform_id clSelectedPlatformID = NULL; 
        if(platforms.size() == 0) {
            PTask::Runtime::Inform("No OpenCL platforms found!");
            return;
        }
        if(platforms.size() > 1) {
            PTask::Runtime::Warning("Multiple OpenCL platforms found! Using first available!");
        }
        clSelectedPlatformID = platforms[0];
#if 0
        char cBuffer[1024];
        cl_int ciErrNum = clGetPlatformInfo (clSelectedPlatformID, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
        if (ciErrNum == CL_SUCCESS) {
            if(PTask::Runtime::IsVerbose()) {
                std::string szInfo(" CL_PLATFORM_NAME: \t");
                szInfo += cBuffer;
                PTask::Runtime::Inform(szInfo);
            }
            sProfileString += cBuffer;
        } else {
            if (PTask::Runtime::IsVerbose()) {
                std::stringstream szWarn;
                szWarn << " Error " << ciErrNum << " in clGetPlatformInfo Call !!!";
                PTask::Runtime::Warning(szWarn.str());
            }
        }
        sProfileString += ", Platform Version = ";
        sProfileString += ", NumDevs = ";
#else
        PTask::Runtime::Warning("WARNING: (2-18-13) Skipping clGetPlatformInfo (instability in CUDA 5.0)\n");
#endif

        // Get and log OpenCL device info 
        cl_uint ciDeviceCount;
        cl_int ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &ciDeviceCount);

        // check for 0 devices found or errors... 
        if (ciDeviceCount == 0) {
            if (PTask::Runtime::IsVerbose()) {
                std::stringstream szWarn;
                szWarn << " No devices found supporting OpenCL (return code " << ciErrNum << ")!!!";
                PTask::Runtime::Warning(szWarn.str()); 
            }
            sProfileString += "0";
        } else if (ciErrNum != CL_SUCCESS) {
            if (PTask::Runtime::IsVerbose()) {
                std::stringstream szWarn;
                szWarn << " Error " << ciErrNum << " in clGetDeviceIDs Call !!!";
                PTask::Runtime::Warning(szWarn.str());
            }
        } else {
            // Get and log the OpenCL device ID's
            if (PTask::Runtime::IsVerbose()) {
                std::stringstream strInfo;
                strInfo << ciDeviceCount << " devices found supporting OpenCL.";
                PTask::Runtime::Inform(strInfo.str());
            }
            char cTemp[2];
            sprintf_s(cTemp, 2*sizeof(char), "%u", ciDeviceCount);
            sProfileString += cTemp;
            cl_device_id devices[20]; // too many!
            ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_ALL, ciDeviceCount, devices, &ciDeviceCount);
            if (ciErrNum != CL_SUCCESS) {
                if (PTask::Runtime::IsVerbose()) {
                    std::stringstream szWarn;
                    szWarn << " Error " << ciErrNum << " in clGetDeviceIDs Call !!!";
                    PTask::Runtime::Warning(szWarn.str());
                }
            }      
            if(ciDeviceCount == 0) 
                PTask::Runtime::Warning("WARNING: No OpenCL devices opened!");
            for(UINT i=0; i<ciDeviceCount; i++) {
                Accelerator * pAccelerator = new CLAccelerator(devices[i], NULL);
                pAccelerator->SetPlatformIndex(i);
                candidates.push_back(pAccelerator);
            }
            std::vector<Accelerator*>::iterator vi;
            for(vi=candidates.begin(); vi!=candidates.end(); vi++) {
                vAccelerators.push_back(*vi);
            }
        }
#endif
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the open. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
        CLAccelerator::Open()
    {
        return Open(NULL);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Opens. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="id">   The identifier. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    CLAccelerator::Open(
        cl_device_id id
        )
    {    
        cl_int ciErr1;
        if(id == NULL) {
            // find default device
            if(CL_SUCCESS != (ciErr1 = clGetPlatformIDs(1, &m_cpPlatform, NULL))) 
                return E_FAIL;
            if(CL_SUCCESS != (ciErr1 = clGetDeviceIDs(m_cpPlatform, CL_DEVICE_TYPE_GPU, 1, &m_pDevice, NULL)))
                return E_FAIL;
        } else {
            m_pDevice = id;
        }
        if(NULL == (m_pContext = clCreateContext(0, 1, &m_pDevice, NULL, NULL, &ciErr1)))
            return E_FAIL;
        if(CL_SUCCESS != ciErr1)
            return E_FAIL;
        if(NULL == (m_cqCommandQueue = clCreateCommandQueue(m_pContext, m_pDevice, 0, &ciErr1)))
            return E_FAIL;
        if(CL_SUCCESS != ciErr1)
            return E_FAIL;
        m_bInitialized = (m_pDevice != NULL && m_pContext != NULL && m_cqCommandQueue != NULL);
        return S_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a device. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the device. </returns>
    ///-------------------------------------------------------------------------------------------------

    void*			
    CLAccelerator::GetDevice(
        VOID
        ) 
    { 
        return (void*) m_pDevice;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the context. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the context. </returns>
    ///-------------------------------------------------------------------------------------------------

    void*
    CLAccelerator::GetContext() {
        return (void*) m_pContext;
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
    CLAccelerator::PlatformSpecificCreateAsyncContext(
        Task * pTask,
        ASYNCCONTEXTTYPE eAsyncContextType
        )
    {
        return new CLAsyncContext(this, pTask, eAsyncContextType);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the queue. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   The queue. </returns>
    ///-------------------------------------------------------------------------------------------------

    cl_command_queue
    CLAccelerator::GetQueue() {
        return m_cqCommandQueue;
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
    CLAccelerator::NewPlatformSpecificBuffer(
        Datablock * pLogicalParent, 
        UINT nDatblockChannelIndex, 
        BUFFERACCESSFLAGS uiBufferAccessFlags, 
        Accelerator * pProxyAllocator
        )
    {
        return new PCLBuffer(pLogicalParent, 
                             uiBufferAccessFlags, 
                             nDatblockChannelIndex, 
                             this, 
                             pProxyAllocator);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synchronizes the context. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pContext"> [in,out] If non-null, the context. </param>
    /// <param name="pTask">    [in,out] (optional)  If non-null, the task. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL         
    CLAccelerator::Synchronize(
        Task * pTask
        )
    {
        UNREFERENCED_PARAMETER(pTask);
        BOOL bSuccess = TRUE;
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'p' has accessible memory space. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="p">    [in,out] If non-null, the p. </param>
    ///
    /// <returns>   true if accessible memory space, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CLAccelerator::HasAccessibleMemorySpace(
        Accelerator*p
        )
    {
        // XXX: TODO: 
        // there is probably a way to do this for CL devices!
        UNREFERENCED_PARAMETER(p);
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Supports pinned host memory. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CLAccelerator::SupportsPinnedHostMemory(
        VOID
        )
    {
        return FALSE;
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
    CLAccelerator::AllocatePagelockedHostMemory(
        UINT uiBytes, 
        BOOL * pbResultPageLocked
        )
    {
        UNREFERENCED_PARAMETER(uiBytes);
        *pbResultPageLocked = FALSE;
        assert(FALSE);
        PTask::Runtime::HandleError("%s: CLAccelerator has no specialized host allocator!\n", __FUNCTION__);
        return NULL;
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
    CLAccelerator::FreeHostMemory(
        void * pBuffer,
        BOOL bPageLocked
        )
    {
        UNREFERENCED_PARAMETER(pBuffer);
        UNREFERENCED_PARAMETER(bPageLocked);
        PTask::Runtime::HandleError("%s: CLAccelerator has no specialized host allocator!\n", __FUNCTION__);
        assert(false);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Compiles accelerator source code to create a PTask binary. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. 
    /// 			</remarks>
    ///
    /// <param name="lpszFileName">             [in] filename+path of source. cannot be null.</param>
    /// <param name="lpszOperation">            [in] Function name in source file. cannot be null.</param>
    /// <param name="ppPlatformSpecificBinary"> [out] On success, a platform specific binary. </param>
    /// <param name="ppPlatformSpecificModule"> [out] On success, a platform specific module handle. </param>
    /// <param name="lpszCompilerOutput">       (optional) [in,out] On failure, the compiler output. </param>
    /// <param name="uiCompilerOutput">         (optional) [in] length of buffer supplied for 
    /// 										compiler output. </param>
    /// <param name="tgx">                      (optional) thread group X dimensions. (see remarks)</param>
    /// <param name="tgy">                      (optional) thread group Y dimensions. (see remarks)</param>
    /// <param name="tgz">                      (optional) thread group Z dimensions. (see remarks)</param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CLAccelerator::Compile( 
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
        static char szError[512];
        #pragma warning(disable:4996)
        UNREFERENCED_PARAMETER(tgx);
        UNREFERENCED_PARAMETER(tgy);
        UNREFERENCED_PARAMETER(tgz);

        *ppPlatformSpecificBinary = NULL;
        *ppPlatformSpecificModule = NULL;

        FILE *fp = fopen(pszSrcFile, "rb");
        if(fp == NULL) { 
            // Note: this is not necessarily fatal. Let the user handle it.
            sprintf_s(szError, 512, "cannot find shader source code %s", pszSrcFile);
            if(PTask::Runtime::IsVerbose()) {
                PTask::Runtime::Warning(szError);
            }
            return FALSE;
        }
        fseek(fp, 0, SEEK_END);
        size_t file_size = ftell(fp);
        char *buf = new char[file_size+1];
        fseek(fp, 0, SEEK_SET);
        fread(buf, sizeof(char), file_size, fp);
        fclose(fp);
        buf[file_size] = '\0';

        BOOL bResult = Compile(buf, (UINT)file_size, 
                               pFunctionName,
                               ppPlatformSpecificBinary, 
                               ppPlatformSpecificModule,
                               lpszCompilerOutput,
                               uiCompilerOutput,
                               tgx, tgy, tgz);

        delete [] buf;
        return bResult; 
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
    CLAccelerator::Compile(
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

        static char szError[512];
        UNREFERENCED_PARAMETER(nThreadGroupSizeX);
        UNREFERENCED_PARAMETER(nThreadGroupSizeY);
        UNREFERENCED_PARAMETER(nThreadGroupSizeZ);

        Lock();
        cl_int ciErr1;
        cl_program cpProgram = clCreateProgramWithSource(m_pContext, 1, 
                                                         (const char **)&lpszShaderCode, 
                                                         (const size_t*)&uiShaderCodeSize, 
                                                         &ciErr1);
        if (ciErr1 != CL_SUCCESS) {
            // not necessarily fatal, but user must handle
            Unlock();
            sprintf_s(m_lpszUserMessages, 
                      USER_OUTPUT_BUF_SIZE, 
                      "compile failed for %s\nLine %u!!!", 
                      __LINE__);
            if(PTask::Runtime::IsVerbose()) {
                PTask::Runtime::Warning(szError);
            }
            return FALSE;
        }

        ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
        if (ciErr1 != CL_SUCCESS) {
            Unlock();
            // not necessarily fatal, but user must handle
            sprintf_s(m_lpszUserMessages, 
                      USER_OUTPUT_BUF_SIZE, 
                      "Error in clBuildProgram, Line %u in file %s !!!", 
                      __LINE__, 
                      __FILE__);
            if(lpszCompilerOutput) {
                UINT nMessageLen = (UINT) strlen(m_lpszUserMessages);
                UINT nCopySize = ((UINT) uiCompilerOutput > nMessageLen) ? nMessageLen : uiCompilerOutput - 1;
                memset(lpszCompilerOutput, 0, uiCompilerOutput);
                memcpy(lpszCompilerOutput, m_lpszUserMessages, nCopySize);
            }
            if(PTask::Runtime::IsVerbose()) {
                PTask::Runtime::Warning(szError);
            }
            return FALSE;
        }

        cl_kernel ckKernel = clCreateKernel(cpProgram, lpszOperation, &ciErr1);
        Unlock();
        if (ciErr1 != CL_SUCCESS) {
            sprintf_s(m_lpszUserMessages, 
                      USER_OUTPUT_BUF_SIZE, 
                      "Error in clCreateKernel, Line %u in file %s !!!", 
                      __LINE__, 
                      __FILE__);
            if(PTask::Runtime::IsVerbose()) {
                PTask::Runtime::Warning(szError);
            }
            return FALSE;
        }

        // TODO: implement cache!
        // CachePutShader((char*)pszSrcFile, (char*)pFunctionName, ckKernel, cpProgram);
        *ppPlatformSpecificBinary = (void*) ckKernel;
        *ppPlatformSpecificModule = (void*) cpProgram;
        return TRUE; 
        #pragma warning(default:4996)    
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the context current. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CLAccelerator::IsDeviceContextCurrent(
        VOID
        ) 
    { 
        return TRUE; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Makes the context current. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CLAccelerator::MakeDeviceContextCurrent(
        VOID
        ) 
    { 
        return TRUE; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the current context. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void            
    CLAccelerator::ReleaseCurrentDeviceContext(
        VOID
        ) 
    { 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the supports function arguments. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CLAccelerator::SupportsFunctionArguments(
        VOID
        ) 
    { 
        return TRUE; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if this accelerator has some support for device to device transfer
    /// 			with the given accelerator. This allows us to skip a trip through host memory
    /// 			in many cases.
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CLAccelerator::SupportsDeviceToDeviceTransfer(
        __in Accelerator * pAccelerator
        )
    {
        assert(pAccelerator != NULL);
        assert(pAccelerator != this);
        switch(pAccelerator->GetClass()) {
        case ACCELERATOR_CLASS_DIRECT_X: return FALSE; 
        case ACCELERATOR_CLASS_OPEN_CL: return FALSE;
        case ACCELERATOR_CLASS_CUDA: return FALSE;
        case ACCELERATOR_CLASS_REFERENCE: return FALSE;
        case ACCELERATOR_CLASS_HOST: return FALSE;
        case ACCELERATOR_CLASS_UNKNOWN: return FALSE;
        }
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
    CLAccelerator::SupportsDeviceMemcpy(
        VOID
        )
    {
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the supports byval arguments. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    CLAccelerator::SupportsByvalArguments(
        VOID
        ) 
    { 
        return TRUE; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the platform identifier. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   The platform identifier. </returns>
    ///-------------------------------------------------------------------------------------------------

    cl_platform_id  
    CLAccelerator::GetPlatformId(
        VOID
        ) 
    { 
        return m_cpPlatform; 
    }

};
#endif