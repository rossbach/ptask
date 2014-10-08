//--------------------------------------------------------------------------------------
// File: HostAccelerator.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "ptdxhdr.h"
#include <assert.h>
#include "HostAccelerator.h"
#include "PHBuffer.h"
#include "HostTask.h"
#include "PTaskRuntime.h"
#include "MemorySpace.h"
#include "HostAsyncContext.h"
using namespace std;

#ifdef EXTREME_HOST_TRACE
#define MSGSIZE 256
#define htrace(x) {\
    char szMsg[MSGSIZE];\
    sprintf_s(szMsg, MSGSIZE, "%s\n", x);\
    printf("T[%4X]: %s", ::GetCurrentThreadId(), szMsg); }
#define htrace2(x, y) {\
    char szMsg[MSGSIZE];\
    sprintf_s(szMsg, MSGSIZE, x, y);\
    printf("T[%4X]: %s", ::GetCurrentThreadId(), szMsg); }
#define htrace3(x, y, z) {\
    char szMsg[MSGSIZE];\
    sprintf_s(szMsg, MSGSIZE, x, y, z);\
    printf("T[%4X]: %s", ::GetCurrentThreadId(), szMsg); }
#define htrace4(x, y, z, w){\
    char szMsg[MSGSIZE];\
    sprintf_s(szMsg, MSGSIZE, x, y, z, w);\
    printf("T[%4X]: %s", ::GetCurrentThreadId(), szMsg); }
#define htrace5(x, y, z, w, u) {\
    char szMsg[MSGSIZE];\
    sprintf_s(szMsg, MSGSIZE, x, y, z, w, u);\
    printf("T[%4X]: %s", ::GetCurrentThreadId(), szMsg); }
#else
#define htrace(x)
#define htrace2(x, y) 
#define htrace3(x, y, z) 
#define htrace4(x, y, z, w) 
#define htrace5(x, y, z, w, u) 
#endif

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="cpuid">    The cpuid. </param>
    /// <param name="lpszName"> [in,out] If non-null, the name. </param>
    ///-------------------------------------------------------------------------------------------------

    HostAccelerator::HostAccelerator(
        int cpuid,
        char * lpszName
        ) 
    {
        m_bInitialized = FALSE;
        m_class = ACCELERATOR_CLASS_HOST;
        if(!lpszName) 
            lpszName = "host-accelerator";
        size_t nLength = strlen(lpszName)+10;
        m_lpszDeviceName = (char*) malloc(nLength);
        sprintf_s(m_lpszDeviceName, nLength, "%s:%d", lpszName, (int) cpuid);
        m_nDeviceId = (int) cpuid;
        m_bInitialized = TRUE;
        m_nCoreCount = 1;
        m_uiMemorySpaceId = HOST_MEMORY_SPACE_ID;
        MemorySpace::RegisterMemorySpaceId(m_uiMemorySpaceId, this);
        CreateAsyncContext(NULL, ASYNCCTXT_DEFAULT);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    HostAccelerator::~HostAccelerator() {
        assert(!LockIsHeld() && "destroying HostAccelerator object while locks held in other threads!");
        Lock();
        m_bInitialized = FALSE;
        Unlock();
        assert(m_lpszDeviceName != NULL);
        free(m_lpszDeviceName);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enumerate accelerators. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="vAccelerators">    [in,out] [in,out] If non-null, the accelerators. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    HostAccelerator::EnumerateAccelerators(
        std::vector<Accelerator*> &vAccelerators
        ) 
    {
        if(!PTask::Runtime::GetUseHost()) 
            return;
        SYSTEM_INFO info;
        ::GetSystemInfo(&info);
        DWORD dwCPUs = info.dwNumberOfProcessors;
        DWORD dwArtMax = (DWORD) PTask::Runtime::GetMaximumHostConcurrency();
        DWORD dwUpperBound = dwArtMax > 0 ? min(dwCPUs, dwArtMax) : dwCPUs;
        for(UINT i=0; i<dwUpperBound; i++) {
            HostAccelerator * pHAccelerator = new HostAccelerator(0, "hostacc");
            vAccelerators.push_back(pHAccelerator);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the device. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the device. </returns>
    ///-------------------------------------------------------------------------------------------------

    void*			
    HostAccelerator::GetDevice() { 
        return (void*) NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the context. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the context. </returns>
    ///-------------------------------------------------------------------------------------------------

    void*	
    HostAccelerator::GetContext() {
        return NULL;
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
    HostAccelerator::PlatformSpecificCreateAsyncContext(
        __in Task * pTask,
        __in ASYNCCONTEXTTYPE eAsyncContextType
        )
    {
        return new HostAsyncContext(this, pTask, eAsyncContextType);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Cache put shader. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="szFile">   [in,out] If non-null, the file. </param>
    /// <param name="szFunc">   [in,out] If non-null, the func. </param>
    /// <param name="p">        The p. </param>
    /// <param name="m">        The m. </param>
    ///-------------------------------------------------------------------------------------------------

    void					
    HostAccelerator::CachePutShader(
        char * szFile, 
        char * szFunc, 
        FARPROC p,
        HMODULE m
        )
    {
        Lock();
        std::string strKey(szFile);
        strKey += "::";
        strKey += szFunc;
        assert(m_pCodeCache.find(strKey) == m_pCodeCache.end());
        assert(m_pModuleCache.find(strKey) == m_pModuleCache.end());
        m_pCodeCache[strKey] = p;
        m_pModuleCache[strKey] = m;
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Cache get shader. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="szFile">   [in,out] If non-null, the file. </param>
    /// <param name="szFunc">   [in,out] If non-null, the func. </param>
    /// <param name="p">        [in,out] If non-null, the p. </param>
    /// <param name="m">        [in,out] The m. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    HostAccelerator::CacheGetShader(
        char * szFile, 
        char * szFunc,
        FARPROC &p,
        HMODULE &m
        )
    {
        Lock();
        p = NULL;
        m = NULL;
        std::string strKey(szFile);
        strKey += "::";
        strKey += szFunc;
        map<std::string, FARPROC>::iterator fi = m_pCodeCache.find(strKey); 
        if(fi != m_pCodeCache.end())
            p = fi->second;
        map<string, HMODULE>::iterator mi = m_pModuleCache.find(strKey);
        if(mi != m_pModuleCache.end())
            m = mi->second;
        Unlock();
        return ((p != NULL) && (m != NULL));
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
    HostAccelerator::NewPlatformSpecificBuffer(
        Datablock * pLogicalParent, 
        UINT nDatblockChannelIndex, 
        BUFFERACCESSFLAGS uiBufferAccessFlags, 
        Accelerator * pProxyAllocator
        )
    {
        return new PHBuffer(pLogicalParent, 
                            uiBufferAccessFlags, 
                            nDatblockChannelIndex, 
                            this, 
                            pProxyAllocator);
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Compiles. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pszSrcFile">               [in,out] If non-null, source file. </param>
    /// <param name="pFunctionName">            [in,out] If non-null, name of the function. </param>
    /// <param name="ppPlatformSpecificBinary"> [out] On success, a platform specific binary. </param>
    /// <param name="ppPlatformSpecificModule"> [out] On success, a platform specific module handle. </param>
    /// <param name="lpszCompilerOutput">       [in,out] (optional)  On failure, the compiler output. </param>
    /// <param name="uiCompilerOutput">         (optional) [in] length of buffer supplied for
    ///                                         compiler output. </param>
    /// <param name="tgx">                      (optional) thread group X dimensions. (see remarks) </param>
    /// <param name="tgy">                      (optional) thread group Y dimensions. (see remarks) </param>
    /// <param name="tgz">                      (optional) thread group Z dimensions. (see remarks) </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL  
    HostAccelerator::Compile( 
        char* pszSrcFile,       /* <name>.dll */
        char* pFunctionName,    /* entry point */
        void ** ppPlatformSpecificBinary,
        void ** ppPlatformSpecificModule,
        char * lpszCompilerOutput,
        int uiCompilerOutput,
        int tgx, 
        int tgy, 
        int tgz
        )
    {
        UNREFERENCED_PARAMETER(tgx);
        UNREFERENCED_PARAMETER(tgy);
        UNREFERENCED_PARAMETER(tgz);

        FARPROC lpfn = NULL;
        HMODULE hModule = NULL;
        *ppPlatformSpecificBinary = NULL;
        *ppPlatformSpecificModule = NULL;

        if(CacheGetShader((char*)pszSrcFile, (char*)pFunctionName, lpfn, hModule)) {
            *ppPlatformSpecificBinary = (void*) lpfn;
            *ppPlatformSpecificModule = (void*) hModule;
            return TRUE; 
        }

        if(NULL == (hModule = LoadLibraryExA(pszSrcFile, NULL, NULL))) {
            static const char * szLLError = "LoadLibraryExA failure: cannot load dll in HostAccelerator::Compile\n";
            PTask::Runtime::Warning(szLLError);
            if(lpszCompilerOutput) {
                int nMsgSize = (int) strlen(szLLError);
                int nCopyBytes = (uiCompilerOutput > nMsgSize) ? nMsgSize : uiCompilerOutput - 1;
                memset(lpszCompilerOutput, 0, uiCompilerOutput);
                memcpy(lpszCompilerOutput, szLLError, nCopyBytes);
            }
            return FALSE;
        }

        if(NULL == (lpfn = GetProcAddress(hModule, pFunctionName))) {
            static const char * szGPAError = "GetProcAddress failed in HostAccelerator::Compile\n";
            PTask::Runtime::Warning(szGPAError);
            if(lpszCompilerOutput) {
                int nMsgSize = (int) strlen(szGPAError);
                int nCopyBytes = (uiCompilerOutput > nMsgSize) ? nMsgSize : uiCompilerOutput - 1;
                memset(lpszCompilerOutput, 0, uiCompilerOutput);
                memcpy(lpszCompilerOutput, szGPAError, nCopyBytes);
            }
            return FALSE;
        }

        CachePutShader((char*)pszSrcFile, (char*)pFunctionName, lpfn, hModule);
        *ppPlatformSpecificBinary = (void*) lpfn;
        *ppPlatformSpecificModule = (void*) hModule;
        return TRUE; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Compiles accelerator source code to create a PTask binary. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011.
    ///             
    ///             The function accepts a string of source code and an operation in that source to
    ///             build a binary for.  
    ///             
    ///             Currently, this is not implemented for host tasks because this involves
    ///             setting up infrastructure to choose a compiler and target a DLL, etc. 
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
    HostAccelerator::Compile(
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
        UNREFERENCED_PARAMETER(lpszShaderCode);
        UNREFERENCED_PARAMETER(uiShaderCodeSize);
        UNREFERENCED_PARAMETER(lpszOperation);
        UNREFERENCED_PARAMETER(ppPlatformSpecificBinary);
        UNREFERENCED_PARAMETER(ppPlatformSpecificModule);
        UNREFERENCED_PARAMETER(lpszCompilerOutput);
        UNREFERENCED_PARAMETER(uiCompilerOutput);
        UNREFERENCED_PARAMETER(nThreadGroupSizeX); 
        UNREFERENCED_PARAMETER(nThreadGroupSizeY);
        UNREFERENCED_PARAMETER(nThreadGroupSizeZ);
        PTask::Runtime::Warning("HostAccelerator compile from source unimplemented!");
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Makes a context current. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    HostAccelerator::MakeDeviceContextCurrent(
        VOID
        ) 
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the current context described by.  </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    HostAccelerator::ReleaseCurrentDeviceContext(
        VOID
        )
    {
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
    HostAccelerator::HasAccessibleMemorySpace(
        Accelerator*p
        )
    {
        // return true if the accelerator's memory
        // space is accessible from this one's. 
        // Host accelerators are assumed to have
        // coherent access to shared memory. TODO:
        // arch-specific implementations. 
        if(p->GetClass() == ACCELERATOR_CLASS_HOST)
            return TRUE;
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
    HostAccelerator::SupportsPinnedHostMemory(
        VOID
        )
    {
        return TRUE;
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
    /// <param name="bRequestPageLocked">   true to request page-locked memory, 
    /// 									false to request pageable memory. </param>
    /// <param name="pbResultPageLocked">   [in,out] If non-null, the result of whether the
    /// 									allocated memory is page-locked is provided here. </param>
    ///
    /// <returns>   byte pointer on success, null on failure. </returns>
    ///-------------------------------------------------------------------------------------------------

    void *      
    HostAccelerator::AllocatePagelockedHostMemory(
        UINT uiBytes, 
        BOOL * pbResultPageLocked
        )
    {
        *pbResultPageLocked = FALSE;
        return AllocateMemoryExtent((ULONG) uiBytes, 0);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Free host memory. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pBuffer">  [in,out] If non-null, the buffer. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    HostAccelerator::FreeHostMemory(
        void * pBuffer,
        BOOL bPageLocked
        ) 
    {
        UNREFERENCED_PARAMETER(bPageLocked);
        DeallocateMemoryExtent(pBuffer);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate host memory. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="uiBytes">  The bytes. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    void *          
    HostAccelerator::AllocateMemoryExtent(
        ULONG ulBytes,
        ULONG ulFlags
        )
    {
        ulFlags;
        #define MEMORY_ALIGNMENT  4096
        void * pBuffer = VirtualAlloc( NULL, (ulBytes + MEMORY_ALIGNMENT), MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE );    
        assert(pBuffer != NULL);
    #ifdef ALLOC_MAPPED_PINNED
        BOOL bLocked = VirtualLock(pBuffer, (ulBytes+MEMORY_ALIGNMENT));
        assert(bLocked);
    #endif
        return pBuffer;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Free host memory. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pBuffer">  [in,out] If non-null, the buffer. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    HostAccelerator::DeallocateMemoryExtent(
        void * pBuffer
        ) 
    {
    #ifdef ALLOC_MAPPED_PINNED
        assert(false && "HostAccelerator::FreeHostMemory requires buffer size to call VirtualUnlock!");
        VirtualUnlock(pBuffer, 0);
    #endif
        BOOL bFreeRes = VirtualFree(pBuffer, 0, MEM_RELEASE);
        if(!bFreeRes) {
            assert(bFreeRes);
            PTask::Runtime::HandleError("%s: VirtualFree failed (res=%d)\n",
                                        __FUNCTION__,
                                        bFreeRes);
        }
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
    HostAccelerator::SupportsDeviceMemcpy(
        VOID
        )
    {
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synchronizes the context. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pTask">    [in,out] (optional)  If non-null, the task. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL         
    HostAccelerator::Synchronize(
        Task * pTask
        )
    {
        UNREFERENCED_PARAMETER(pTask);
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Opens. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT		    
    HostAccelerator::Open(
        VOID
        ) 
    { 
        return S_OK; 
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
    HostAccelerator::IsDeviceContextCurrent(
        VOID
        ) 
    { 
        return TRUE; 
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
    HostAccelerator::SupportsFunctionArguments(
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
    HostAccelerator::SupportsDeviceToDeviceTransfer(
        __in Accelerator * pAccelerator
        )
    {
        UNREFERENCED_PARAMETER(pAccelerator);
        return FALSE;
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
    HostAccelerator::SupportsByvalArguments(
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
    HostAccelerator::GetDeviceId(   
        VOID
        ) 
    {   
        return m_nDeviceId; 
    }


};

