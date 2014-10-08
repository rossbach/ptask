//--------------------------------------------------------------------------------------
// File: DXAccelerator.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "primitive_types.h"
#include <stdlib.h>
#include "ptdxhdr.h"
#include <assert.h>
#include <algorithm>
#include "dxaccelerator.h"
#include "symbiostypes.h"
#include "PDXBuffer.h"
#include "Scheduler.h"
#include "PTaskRuntime.h"
#include "MemorySpace.h"
#include "DXAsyncContext.h"
using namespace std;

namespace PTask {

    /// <summary>   true to enable, false to disable code paths that
    ///             directly leverage direct x asyncrony. </summary>
    BOOL            DXAccelerator::s_bEnableDirectXAsyncrony = FALSE;

    /// <summary>   true to enable, false to disable code paths that
    ///             try to use resource sharing support in DX11. 
    ///             Turns out that the DX support for this is so limited
    ///             that it is effectively useles...only 2D non-mipmapped
    ///             surfaces can be shared through the APIs supported, and those
    ///             cannot be bound as SRVs and UAVs to compute shader. 
    ///             So this is disabled by default.
    ///             </summary>
    BOOL            DXAccelerator::s_bEnableDirectXP2PAPIs = FALSE;

    typedef HRESULT 
        (WINAPI * LPD3D11CREATEDEVICE)( 
            IDXGIAdapter*, 
            D3D_DRIVER_TYPE, 
            HMODULE, 
            UINT32, 
            CONST D3D_FEATURE_LEVEL*, 
            UINT, 
            UINT32, 
            ID3D11Device**, 
            D3D_FEATURE_LEVEL*, 
            ID3D11DeviceContext** );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Default constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    DXAccelerator::DXAccelerator() {
        char lpszAnonymousName[256];
        m_class = ACCELERATOR_CLASS_DIRECT_X;
        m_bInitialized = FALSE;
        m_pContext = NULL;
        m_pDevice = NULL;	
        m_pCache = new DXCodeCache();
        m_lpszDeviceName = NULL;
        m_pAdapter = NULL;
        m_nCoreCount = 0;
        m_nRuntimeVersion = 0;
        m_nMemorySize = 0;
        m_nClockRate = 0;
        m_bSupportsConcurrentKernels = FALSE;
        m_uiMemorySpaceId = MemorySpace::AssignUniqueMemorySpaceIdentifier();
        sprintf_s(lpszAnonymousName, "DirectXAcc_%d", m_uiMemorySpaceId);
        std::string strDXName(lpszAnonymousName);
        MemorySpace * pMemorySpace = new MemorySpace(strDXName, m_uiMemorySpaceId);
        MemorySpace::RegisterMemorySpace(pMemorySpace, this);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    DXAccelerator::~DXAccelerator() {
        PTSRELEASE( m_pContext );
        PTSRELEASE( m_pDevice );
        delete m_pCache;
        if(m_lpszDeviceName) 
            free(m_lpszDeviceName);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Creates a device. This is equivalent to D3D11CreateDevice, except it dynamically loads
    ///     d3d11.dll, this gives us a graceful way to message the user on systems with no d3d11
    ///     installed.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pAdapter">             [in,out] If non-null, the adapter. </param>
    /// <param name="DriverType">           Type of the driver. </param>
    /// <param name="Software">             external software rasterizer (always NULL!). </param>
    /// <param name="Flags">                creation flags to pass to DX runtime. </param>
    /// <param name="pFeatureLevels">       Acceptable DX feature levels list. </param>
    /// <param name="FeatureLevels">        Number of entries in feature levels list. </param>
    /// <param name="SDKVersion">           The sdk version. </param>
    /// <param name="ppDevice">             [out] If non-null, the device. </param>
    /// <param name="pFeatureLevel">        [out] If non-null, the feature level of the device. </param>
    /// <param name="ppImmediateContext">   [out] If non-null, context for the device. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT WINAPI 
    DXAccelerator::CreateDevice( 
        IDXGIAdapter* pAdapter,
        D3D_DRIVER_TYPE DriverType,
        HMODULE Software,
        UINT32 Flags,
        CONST D3D_FEATURE_LEVEL* pFeatureLevels,
        UINT FeatureLevels,
        UINT32 SDKVersion,
        ID3D11Device** ppDevice,
        D3D_FEATURE_LEVEL* pFeatureLevel,
        ID3D11DeviceContext** ppImmediateContext 
        )
    {
        static LPD3D11CREATEDEVICE  s_DynamicD3D11CreateDevice = NULL;
        if ( s_DynamicD3D11CreateDevice == NULL ) {            
            HMODULE hModD3D11 = LoadLibrary( L"d3d11.dll" );
            if ( hModD3D11 == NULL ) {
                // Ensure this "D3D11 absent" message is shown only once. As sometimes, the app would like to try
                // to create device multiple times
                static bool bMessageAlreadyShwon = false;
                if ( !bMessageAlreadyShwon ) {
                    OSVERSIONINFOEX osv;
                    memset( &osv, 0, sizeof(osv) );
                    osv.dwOSVersionInfoSize = sizeof(osv);
                    GetVersionEx( (LPOSVERSIONINFO)&osv );
                    if ( ( osv.dwMajorVersion > 6 )
                        || ( osv.dwMajorVersion == 6 && osv.dwMinorVersion >= 1 ) 
                        || ( osv.dwMajorVersion == 6 && osv.dwMinorVersion == 0 && osv.dwBuildNumber > 6002 ) ) {
                        PTask::Runtime::Warning("Direct3D 11 components were not found.");
                        // This should not happen, but is here for completeness as the system could be
                        // corrupted or some future OS version could pull D3D11.DLL for some reason
                    }
                    else if ( osv.dwMajorVersion == 6 && osv.dwMinorVersion == 0 && osv.dwBuildNumber == 6002 ) {                        
                        char * sz = "Direct3D 11 components were not found, but are available for"\
                                    " this version of Windows.\n"\
                                    "For details see Microsoft Knowledge Base Article #971644\n"\
                                    "http://support.microsoft.com/default.aspx/kb/971644/";
                        PTask::Runtime::Warning(sz);
                    }  else if ( osv.dwMajorVersion == 6 && osv.dwMinorVersion == 0 ) {
                        char * sz = "Direct3D 11 components were not found. Please install the latest Service Pack.\n"\
                                    "For details see Microsoft Knowledge Base Article #935791\n"\
                                    " http://support.microsoft.com/default.aspx/kb/935791";
                        PTask::Runtime::Warning(sz);
                    } else {
                        PTask::Runtime::Warning("Direct3D 11 is not supported on this OS.");
                    }
                    bMessageAlreadyShwon = true;
                }            
                return E_FAIL;
            }
            s_DynamicD3D11CreateDevice = ( LPD3D11CREATEDEVICE )GetProcAddress( hModD3D11, "D3D11CreateDevice" );           
        }

        return s_DynamicD3D11CreateDevice( pAdapter, DriverType, Software, Flags, pFeatureLevels, FeatureLevels,
            SDKVersion, ppDevice, pFeatureLevel, ppImmediateContext );
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enumerate accelerators. 
    //      Create Accelerator objects for all D3D devices matching the runtime settings:
    //      * max concurrency:   now handled in the scheduler--return a comprehensive list of candidates!
    //      * min feature level: ignore adapters that do not support DX<feature level> or greater
    //      * use ref driver:    create reference devices or not? 
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="devices">  [in,out] non-null, the devices. </param>
    ///-------------------------------------------------------------------------------------------------

    void             
    DXAccelerator::EnumerateAccelerators(
        std::vector<Accelerator*> &devices
        )
    {
        #pragma warning(disable:4996)
        vector<ADAPTERRECORD*> vAdapters;
        if(!PTask::Runtime::GetUseDirectX()) {
            PTask::Runtime::Warning("WARNING: DXAccelerator::EnumerateDevices() called without DirectX supported enabled!");
            return;
        }

        // first try to find out what adapters are available.
        // if we can find adapters it means we can use HLSL + DirectX PTasks.
        IDXGIFactory1 * pFactory;
        HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)(&pFactory) );
        if(FAILED(hr)) {
            PTask::Runtime::ErrorMessage("No DirectX capable devices found!");
            return;
        }

        UINT index = 0; 
        IDXGIAdapter1 * pAdapter; 
        while(pFactory->EnumAdapters1(index, &pAdapter) != DXGI_ERROR_NOT_FOUND) 
        { 
            DXGI_ADAPTER_DESC desc;
            pAdapter->GetDesc(&desc);
            if(!wcscmp(desc.Description, L"Microsoft Basic Render Driver") &&
                !PTask::Runtime::GetUseReferenceDrivers()) {
                PTask::Runtime::Inform("AVOIDING Microsoft Basic Render Driver");
                PTSRELEASE(pAdapter);
            } else {
                ADAPTERRECORD * rec = new ADAPTERRECORD();
                rec->pAdapter = pAdapter;
                pAdapter->GetDesc(&rec->desc);
                vAdapters.push_back(rec); 
            }
            ++index; 
        } 
        PTSRELEASE(pFactory);

        index=0;
        vector<ADAPTERRECORD*>::iterator i;   
        vector<Accelerator*> candidateAccelerators;
        for(i=vAdapters.begin(); i!=vAdapters.end(); i++) {	
            size_t nMemSize = (*i)->desc.DedicatedSystemMemory;
            if (PTask::Runtime::IsVerbose()) {
                char szDescription[USER_OUTPUT_BUF_SIZE];
                std::stringstream ss;
                ss << "Adapter " << index << std::endl;
                wcstombs(szDescription, (*i)->desc.Description, USER_OUTPUT_BUF_SIZE);
                ss << szDescription << std::endl;
                ss << "Dedicated sysmem=" << (*i)->desc.DedicatedSystemMemory/1024 << "K" << std::endl;
                ss << "Shared sysmem=" << (*i)->desc.SharedSystemMemory/1024 << "K" << std::endl;
                ss << "Dedicated vidmem=" << (*i)->desc.DedicatedVideoMemory/1024 << "K" << std::endl;
                PTask::Runtime::Inform(ss.str());
            }
            // as if things were not sufficiently fun already, DX libs have an idiosyncratic
            // approach to the default adapter--when creating a device context using the default
            // adapter, you are supposed to pass NULL as the adapter pointer, otherwise you 
            // pass the actual adapter interface. From:
            // http://msdn.microsoft.com/en-us/library/ff476082(VS.85).aspx
            // A pointer to the video adapter to use when creating a device. 
            // Pass NULL to use the default adapter, which is the first adapter that 
            // is enumerated by IDXGIFactory1::EnumAdapters. 
            IDXGIAdapter1 * pAdapter = (*i)->pAdapter;
            DXAccelerator * pAccelerator = new DXAccelerator();
            if(FAILED(pAccelerator->Open(pAdapter, index))) {
                delete pAccelerator;
                PTSRELEASE(pAdapter);
            } else {
                pAccelerator->SetGlobalMemorySize((UINT)(nMemSize/1000000)); // in MB
                candidateAccelerators.push_back(pAccelerator);
            }
            delete (*i);
            index++;
        }

        // if we've got nothing, create a DirectX software fallback
        // if the user has configured the run-time to do so.
        if(candidateAccelerators.size() == 0 && PTask::Runtime::GetUseReferenceDrivers()) {
            DXAccelerator * pAccelerator = new DXAccelerator();
            if(SUCCEEDED(pAccelerator->OpenReferenceDevice())) {
                candidateAccelerators.push_back(pAccelerator);
            } else {
                delete pAccelerator;
            }
        }

        std::vector<Accelerator*>::iterator vi;
        for(vi=candidateAccelerators.begin(); vi!=candidateAccelerators.end(); vi++) {
            devices.push_back(*vi);
        }
        #pragma warning(default:4996)
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Obsolete version of Open. We need to have an adapter pointer in the general case. If
    ///     someone calls this, find them an adapter and use it. But complain vociferously--the
    ///     scheduler should hide all of this and this version of the Open method should not be
    ///     called from within the PTask Runtime. If we pass NULL and a 0 enumeration index, the
    ///     DirectX APIs will just find us the default video card.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    DXAccelerator::Open( 
        VOID
        )
    {
        PTask::Runtime::ErrorMessage("Obsolete version of DXAccelerator::Open called!");
        return Open(NULL, 0);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Create the D3D device and device context suitable for running Compute Shaders(CS)
    ///     fail if the given device does not meet requirements for the current runtime settings.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pAdapter">             [in] If non-null, the adapter. </param>
    /// <param name="uiEnumerationIndex">   Zero-based index of the adapter when the OS enumerates
    ///                                     it. This is necessary because the D3D11 APIs for creating
    ///                                     a device are idiosyncratic in the presence of multiple
    ///                                     adapters. </param>
    ///
    /// <returns>
    ///     S_OK on success, E_FAIL otherwise. Use windows SUCCEEDED() and FAILED() macros.
    /// </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    DXAccelerator::Open( 
        IDXGIAdapter * pAdapter,
        UINT uiEnumerationIndex
        )
    {    
        HRESULT hr = E_FAIL;
#ifndef DIRECTXCOMPILESUPPORT
        UNREFERENCED_PARAMETER(pAdapter);
        UNREFERENCED_PARAMETER(uiEnumerationIndex);
#else // DIRECTXCOMPILESUPPORT
        Lock();
        ID3D11Device** ppDeviceOut = &this->m_pDevice; 
        ID3D11DeviceContext** ppContextOut = &this->m_pContext;
        *ppDeviceOut = NULL;
        *ppContextOut = NULL;
        m_nPlatformIndex = uiEnumerationIndex;
        m_pAdapter = pAdapter;
        pAdapter->GetDesc(&m_desc);
        if(!wcscmp(m_desc.Description, L"Microsoft Basic Render Driver") &&
            !PTask::Runtime::GetUseReferenceDrivers()) {
            PTask::Runtime::Inform("AVOIDING Microsoft Basic Render Driver");
            Unlock();
            return E_FAIL;
        }


        UINT uCreationFlags = 0;
        uCreationFlags |= D3D11_CREATE_DEVICE_DISABLE_GPU_TIMEOUT;
        uCreationFlags |= D3D11_CREATE_DEVICE_SINGLETHREADED;
#ifdef DEBUG
        uCreationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif


        // by default we prefer to use only DX11 capable
        // devices, but provide a global control to let
        // the programmer allow DX10 devices if they so chose.
        D3D_FEATURE_LEVEL flOut;
        static const D3D_FEATURE_LEVEL flvl_11[] = { D3D_FEATURE_LEVEL_11_0 };
        static const D3D_FEATURE_LEVEL flvl_10[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
        const D3D_FEATURE_LEVEL * flvl = PTask::Runtime::GetMinimumDirectXFeatureLevel() > 10 ? flvl_11 : flvl_10;
        const UINT nFeatureLevels = PTask::Runtime::GetMinimumDirectXFeatureLevel() > 10 ? 
            (sizeof(flvl_11) / sizeof(D3D_FEATURE_LEVEL)) :
            (sizeof(flvl_10) / sizeof(D3D_FEATURE_LEVEL));

        // among the more **ASININE** "features" of DX 11, the following, taken from:
        // http://msdn.microsoft.com/en-us/library/ff476082(VS.85).aspx
        //	If you set the pAdapter parameter to a non-NULL value, you must also set the DriverType 
        // parameter to the D3D_DRIVER_TYPE_UNKNOWN value. If you set the pAdapter parameter to a non-NULL 
        // value and the DriverType parameter to the D3D_DRIVER_TYPE_HARDWARE value, 
        // D3D11CreateDevice returns an HRESULT of E_INVALIDARG.
        // ------------------------------------------------------
        // Hence, we check to see if the adapter pointer is null--if it is, we are creating
        // an accelerator on the default device, and it's find to use D3D_..._TYPE_HARDWARE.
        // If it's non-null, we're not looking for the default device and must specify unknown.
        IDXGIAdapter * pAdapterParm = (uiEnumerationIndex == 0) ? NULL : pAdapter;
        D3D_DRIVER_TYPE driverType = (uiEnumerationIndex == 0) ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_UNKNOWN;
        hr = CreateDevice( 
                pAdapterParm,                // NULL means use default graphics card
                driverType,					 // Try to create a hardware accelerated device
                NULL,                        // Do not use external software rasterizer module
                uCreationFlags,              // Device creation flags
                flvl,
                nFeatureLevels,
                D3D11_SDK_VERSION,           // SDK version
                ppDeviceOut,                 // Device out
                &flOut,                      // Actual feature level created
                ppContextOut                 // Context out
                );

        if ( SUCCEEDED( hr ) )
        {
            m_class = ACCELERATOR_CLASS_DIRECT_X;
            m_d3dFeatureLevel = flOut;
            m_nRuntimeVersion = m_d3dFeatureLevel == D3D_FEATURE_LEVEL_11_0 ? 11 : 10;
            // A hardware accelerated device has been created, so check for Compute Shader support
            // If we have a device >= D3D_FEATURE_LEVEL_11_0 created, full CS5.0 
            // support is guaranteed, no need for further checks
            if ( flOut < D3D_FEATURE_LEVEL_11_0 && PTask::Runtime::GetMinimumDirectXFeatureLevel() < 11) {
                 // Otherwise, we need further check whether this device support CS4.x (Compute on 10)
                 D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS hwopts;
                 (*ppDeviceOut)->CheckFeatureSupport( D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, &hwopts, sizeof(hwopts) );
                 if ( !hwopts.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x ) {
                     sprintf_s(m_lpszUserMessages, 
                               USER_OUTPUT_BUF_SIZE, 
                               "No hardware Compute Shader capability for device at adapter slot #%d.", 
                               uiEnumerationIndex );
                     PTask::Runtime::Warning(m_lpszUserMessages);
                     hr = E_FAIL;
                 } 
            } 
            if(SUCCEEDED(hr) && pAdapter) {
                m_pAdapter = pAdapter;
                hr = pAdapter->GetDesc(&m_desc);
                if( SUCCEEDED( hr ) ) {
                    size_t desclen = wcslen(m_desc.Description);
                    size_t converted = 0;
                    m_lpszDeviceName = (char*) malloc(desclen + 10);
                    memset(m_lpszDeviceName, 0, desclen+10);
                    wcstombs_s(&converted, m_lpszDeviceName, desclen+10, m_desc.Description, desclen);
                    strcat_s(m_lpszDeviceName, desclen + 10, ":");
                    _ltoa_s(m_desc.DeviceId, &m_lpszDeviceName[desclen+1], 9, 10);
                } 
            } 
        }

        if(SUCCEEDED(hr)) {
            m_bInitialized = (m_pDevice != NULL && m_pContext != NULL);
            CreateAsyncContext(NULL, ASYNCCTXT_DEFAULT);
            CreateAsyncContext(NULL, ASYNCCTXT_XFERDTOD);
            CreateAsyncContext(NULL, ASYNCCTXT_XFERDTOH);
            CreateAsyncContext(NULL, ASYNCCTXT_XFERHTOD);
        } else {
            PTSRELEASE(*ppDeviceOut);
            PTSRELEASE(*ppContextOut);        
            m_bInitialized = FALSE;
        }

        WarmupPipeline();
        Unlock();
#endif // DIRECTXCOMPILESUPPORT

		return hr;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Opens a reference device. Create a software-emulated D3D device and device context
    ///     typically used for debugging on machines without the proper hardware.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    DXAccelerator::OpenReferenceDevice( 
        VOID
        )
    {    
        Lock();
        ID3D11Device** ppDeviceOut = &this->m_pDevice; 
        ID3D11DeviceContext** ppContextOut = &this->m_pContext;
        *ppDeviceOut = NULL;
        *ppContextOut = NULL;

        HRESULT hr = S_OK;

        // Re: creation flags: 
        // The D3D11_CREATE_DEVICE_SINGLETHREADED flag can enable the DX runtime
        // to elide some synchronization (which should entail some performance benefit)
        // but the app must then either provide its own synchronization on DX calls
        // or use only a single thread. We rely on thread-safety for concurrent calls
        // to ID3D11Device::CreateBuffer, so this flag is not an option for us.
        UINT uCreationFlags = D3D11_CREATE_DEVICE_SINGLETHREADED;
#if defined(DEBUG) || defined(_DEBUG)
        uCreationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        D3D_FEATURE_LEVEL flOut;
        static const D3D_FEATURE_LEVEL flvl_11[] = { D3D_FEATURE_LEVEL_11_0 };
        static const D3D_FEATURE_LEVEL flvl_10[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
        const D3D_FEATURE_LEVEL * flvl = PTask::Runtime::GetMinimumDirectXFeatureLevel() > 10 ? flvl_11 : flvl_10;
        const UINT nFeatureLevels = PTask::Runtime::GetMinimumDirectXFeatureLevel() > 10 ? 
            (sizeof(flvl_11) / sizeof(D3D_FEATURE_LEVEL)) :
            (sizeof(flvl_10) / sizeof(D3D_FEATURE_LEVEL));
        
        if(PTask::Runtime::IsVerbose()) {
            PTask::Runtime::Warning("Warning: Using DirectX Reference Driver for DirectX PTasks.");
        }

        hr = CreateDevice( NULL,                        
                D3D_DRIVER_TYPE_REFERENCE,   // Try to create a software-emulator
                NULL,                        // Do not use external software rasterizer module
                uCreationFlags,              // Device creation flags
                flvl,                        // what feature levels are we willing to accept?
                nFeatureLevels,              // number of feature levels in the flvl list
                D3D11_SDK_VERSION,           // SDK version
                ppDeviceOut,                 // Device out
                &flOut,                      // Actual feature level created
                ppContextOut );              // Context out
        m_d3dFeatureLevel = flOut;
        m_nRuntimeVersion = (UINT) m_d3dFeatureLevel;
        if ( FAILED(hr) )
        {
            PTask::Runtime::HandleError("%s: Reference rasterizer device create failure\n", __FUNCTION__);
            return hr;
        }
        m_class = ACCELERATOR_CLASS_REFERENCE;
        if(SUCCEEDED(hr)) {
            m_bInitialized = (m_pDevice != NULL && m_pContext != NULL);
        } else {
            PTSRELEASE(*ppDeviceOut);
            PTSRELEASE(*ppContextOut);        
            m_bInitialized = FALSE;
        }
        Unlock();
        return hr;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the device. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the device. </returns>
    ///-------------------------------------------------------------------------------------------------

    void*			
    DXAccelerator::GetDevice() { 
        return m_pDevice;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the context. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the context. </returns>
    ///-------------------------------------------------------------------------------------------------

    void*	
    DXAccelerator::GetContext() {
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
    DXAccelerator::PlatformSpecificCreateAsyncContext(
        __in Task * pTask,
        __in ASYNCCONTEXTTYPE eAsyncContextType
        )
    {
        return new DXAsyncContext(this, pTask, eAsyncContextType);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Cache put shader. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="szFile">   [in] non-null, the file name. </param>
    /// <param name="szFunc">   [in] non-null, the function name. </param>
    /// <param name="p">        [in] non-null, a pointer to a ID3D11ComputeShader. </param>
    ///-------------------------------------------------------------------------------------------------

    void					
    DXAccelerator::CachePutShader(
        char * szFile, 
        char * szFunc, 
        ID3D11ComputeShader*p
        )
    {
        Lock();
        m_pCache->CachePut(szFile, szFunc, p);
        p->AddRef();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Cache get shader. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="szFile">   [in] If non-null, the file. </param>
    /// <param name="szFunc">   [in] If non-null, the func. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    ID3D11ComputeShader*	
    DXAccelerator::CacheGetShader(
        char * szFile, 
        char * szFunc
        )
    {
        ID3D11ComputeShader* result = NULL;
        Lock();
        result = m_pCache->CacheGet(szFile, szFunc);
        Unlock();
        return result;
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
    /// <param name="nThreadGroupXSize">        Size of the thread group x coordinate. </param>
    /// <param name="nThreadGroupYSize">        Size of the thread group y coordinate. </param>
    /// <param name="nThreadGroupZSize">        Size of the thread group z coordinate. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    DXAccelerator::Compile( 
        char* pszSrcFile, 
        char* pFunctionName,
        void ** ppPlatformSpecificBinary,
        void ** ppPlatformSpecificModule,
        char * lpszCompilerOutput,
        int uiCompilerOutput,
        int nThreadGroupXSize, 
        int nThreadGroupYSize, 
        int nThreadGroupZSize
        )
    {
#ifdef DIRECTXCOMPILESUPPORT
        *ppPlatformSpecificBinary = NULL;
        *ppPlatformSpecificModule = NULL;
        ID3D11ComputeShader* pShader = CacheGetShader((char*)pszSrcFile, (char*)pFunctionName);
        if(pShader != NULL) {
            *ppPlatformSpecificBinary = pShader;
            return TRUE;
        }

        char szThreadGroupX[10];
        char szThreadGroupY[10];
        char szThreadGroupZ[10];
        sprintf_s(szThreadGroupX, "%d", nThreadGroupXSize);
        sprintf_s(szThreadGroupY, "%d", nThreadGroupYSize);
        sprintf_s(szThreadGroupZ, "%d", nThreadGroupZSize);
        const D3D_SHADER_MACRO defines[] = 
        {
            "thread_group_size_x", szThreadGroupX,
            "thread_group_size_y", szThreadGroupY,
            "thread_group_size_z", szThreadGroupZ,
            NULL, NULL
        };

        const UINT PATHBUFSIZE = 4096;
        WCHAR pSrcFile[PATHBUFSIZE];
        size_t converted = 0;
        mbstowcs_s(&converted, pSrcFile, pszSrcFile, PATHBUFSIZE);

        HRESULT hr = S_OK;
        WCHAR str[MAX_PATH];
        BOOL bCompilerOutput = PTask::Runtime::IsVerbose() || lpszCompilerOutput;
        hr = FindDXSDKShaderFileCch( str, MAX_PATH, pSrcFile );
        if(FAILED(hr)) {
            if(bCompilerOutput) {
                std::stringstream ss;
                ss << "could not find HLSL file " << pSrcFile << std::endl;
                const char * szError = ss.str().c_str();
                UINT nErrorLength = (UINT) ss.str().length();
                PTask::Runtime::Warning(szError);
                if(lpszCompilerOutput) {
                    UINT nCopyLength = ((UINT)uiCompilerOutput > nErrorLength) ? nErrorLength : uiCompilerOutput;
                    memset(lpszCompilerOutput, 0, uiCompilerOutput);
                    memcpy(lpszCompilerOutput, szError, nCopyLength);
                }
            }
            return FALSE;
        }

        void * pSourceCode = NULL;
        UINT uiSourceCodeLength = 0;
        if(!ptaskutils::LoadFileIntoMemory(str, &pSourceCode, &uiSourceCodeLength))
            return FALSE;

        BOOL bResult = CompileWithMacros((char*)pSourceCode, 
                                         uiSourceCodeLength,
                                         pFunctionName, 
                                         ppPlatformSpecificBinary, 
                                         ppPlatformSpecificModule, 
                                         lpszCompilerOutput,
                                         uiCompilerOutput,
                                         defines
                                         );

        if(bResult && pSourceCode) {
            pShader = (ID3D11ComputeShader*)*ppPlatformSpecificBinary;
            CachePutShader((char*)pszSrcFile, (char*)pFunctionName, pShader);
            free(pSourceCode);
        }
        return bResult;
#else
        printf("DirectX compiler support disabled in PTask build!\n");
        UNREFERENCED_PARAMETER(pszSrcFile);
        UNREFERENCED_PARAMETER(pFunctionName);
        UNREFERENCED_PARAMETER(ppPlatformSpecificBinary);
        UNREFERENCED_PARAMETER(ppPlatformSpecificModule);
        UNREFERENCED_PARAMETER(lpszCompilerOutput);
        UNREFERENCED_PARAMETER(uiCompilerOutput);
        UNREFERENCED_PARAMETER(nThreadGroupXSize); 
        UNREFERENCED_PARAMETER(nThreadGroupYSize);
        UNREFERENCED_PARAMETER(nThreadGroupZSize);
		return FALSE;
#endif
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Compile with macros. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="lpszShaderCode">           [in] filename+path of source. cannot be null. </param>
    /// <param name="uiShaderCodeSize">         Size of the shader code. </param>
    /// <param name="lpszOperation">            [in] Function name in source file. cannot be null. </param>
    /// <param name="ppPlatformSpecificBinary"> [out] On success, a platform specific binary. </param>
    /// <param name="ppPlatformSpecificModule"> [out] On success, a platform specific module handle. </param>
    /// <param name="lpszCompilerOutput">       (optional) [in,out] On failure, the compiler output. </param>
    /// <param name="uiCompilerOutput">         (optional) [in] length of buffer supplied for
    ///                                         compiler output. </param>
    /// <param name="pMacroDefs">               (optional) the macro defs. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    DXAccelerator::CompileWithMacros(
        __in char *                  lpszShaderCode, 
        __in UINT                    uiShaderCodeSize,                                    
        __in char *                  pFunctionName, 
        __out void **                ppPlatformSpecificBinary,
        __out void **                ppPlatformSpecificModule,
        __inout char *               lpszCompilerOutput,
        __in int                     uiCompilerOutput,
        __in const void*             lpvMacroDefinitions
        ) 
    {
#ifdef DIRECTXCOMPILESUPPORT
        const D3D_SHADER_MACRO* pMacroDefinitions = (const D3D_SHADER_MACRO*) lpvMacroDefinitions;
        *ppPlatformSpecificBinary = NULL;
        *ppPlatformSpecificModule = NULL;
        BOOL bCompilerOutput = PTask::Runtime::IsVerbose() || lpszCompilerOutput;
        DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
        dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

        // We generally prefer to use the higher CS shader profile 
        // when possible as CS 5.0 has better performance on 11-class hardware
        // if the user has set the runtime to allow DX10 hardare, we 
        // must use cs_4_0 for any hardware that has a lower feature level.
        LPCSTR pProfile = m_d3dFeatureLevel < D3D_FEATURE_LEVEL_11_0 ? "cs_4_0" : "cs_5_0";

        HRESULT hr = S_OK;
        ID3DBlob* pErrorBlob = NULL;
        ID3DBlob* pBlob = NULL;
        hr = D3DCompile(lpszShaderCode, 
                        uiShaderCodeSize,
                        NULL,
                        pMacroDefinitions, 
                        NULL, 
                        pFunctionName, 
                        pProfile, 
                        dwShaderFlags, 
                        NULL, 
                        &pBlob, 
                        &pErrorBlob);

        if(FAILED(hr)) {
            if(pErrorBlob && bCompilerOutput) {
                char * szError = (char*)pErrorBlob->GetBufferPointer();
                UINT nErrorLength = (UINT) strlen(szError);
                PTask::Runtime::Warning(szError);
                if(lpszCompilerOutput) {
                    UINT nCopyLength = ((UINT)uiCompilerOutput > nErrorLength) ? nErrorLength : uiCompilerOutput;
                    memset(lpszCompilerOutput, 0, uiCompilerOutput);
                    memcpy(lpszCompilerOutput, szError, nCopyLength);
                }
            }
            PTSRELEASE( pErrorBlob );
            PTSRELEASE( pBlob );    
            return FALSE;
        }    

        ID3D11ComputeShader * pShader = NULL;
        hr = m_pDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &pShader );

#if defined(DEBUG) || defined(PROFILE)
        if ( pShader )
            (pShader)->SetPrivateData( WKPDID_D3DDebugObjectName, lstrlenA(pFunctionName), pFunctionName );
#endif

        PTSRELEASE( pErrorBlob );
        PTSRELEASE( pBlob );

        if(FAILED(hr)) {
            PTask::Runtime::Warning("CreateComputeShader failed!");
            return FALSE;
        }
        *ppPlatformSpecificBinary = pShader;
        return TRUE;
#else
        UNREFERENCED_PARAMETER(lpszShaderCode); 
        UNREFERENCED_PARAMETER(uiShaderCodeSize);                                    
        UNREFERENCED_PARAMETER(pFunctionName); 
        UNREFERENCED_PARAMETER(ppPlatformSpecificBinary);
        UNREFERENCED_PARAMETER(ppPlatformSpecificModule);
        UNREFERENCED_PARAMETER(lpszCompilerOutput);
        UNREFERENCED_PARAMETER(uiCompilerOutput);
        UNREFERENCED_PARAMETER(lpvMacroDefinitions);
        return FALSE;
#endif
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
    DXAccelerator::Compile(
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
#ifdef DIRECTXCOMPILESUPPORT
        char szThreadGroupX[10];
        char szThreadGroupY[10];
        char szThreadGroupZ[10];
        sprintf_s(szThreadGroupX, "%d", nThreadGroupSizeX);
        sprintf_s(szThreadGroupY, "%d", nThreadGroupSizeY);
        sprintf_s(szThreadGroupZ, "%d", nThreadGroupSizeZ);
        const D3D_SHADER_MACRO defines[] = 
        {
            "thread_group_size_x", szThreadGroupX,
            "thread_group_size_y", szThreadGroupY,
            "thread_group_size_z", szThreadGroupZ,
            NULL, NULL
        };
        BOOL bResult = CompileWithMacros((char*)lpszShaderCode, 
                                         uiShaderCodeSize,
                                         lpszOperation, 
                                         ppPlatformSpecificBinary, 
                                         ppPlatformSpecificModule, 
                                         lpszCompilerOutput,
                                         uiCompilerOutput,
                                         defines
                                         );
        return bResult;
#else
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
		return FALSE;
#endif 
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
    DXAccelerator::NewPlatformSpecificBuffer(
        Datablock * pLogicalParent, 
        UINT nDatblockChannelIndex, 
        BUFFERACCESSFLAGS uiBufferAccessFlags, 
        Accelerator * pProxyAllocator
        )
    {
        return new PDXBuffer(pLogicalParent, 
                             uiBufferAccessFlags, 
                             nDatblockChannelIndex, 
                             this, 
                             pProxyAllocator);
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'p' has accessible memory space. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="p">    [in] non-null, a second accelerator. </param>
    ///
    /// <returns>   true if accessible memory space, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    DXAccelerator::HasAccessibleMemorySpace(
        Accelerator*p
        )
    {
        // XXX: TODO: 
        // there is a way to do this between
        // CUDA and DX accelerators, since CUDA supports
        // some DirectCompute interoperation.
        UNREFERENCED_PARAMETER(p);
        return FALSE;
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
    DXAccelerator::SupportsDeviceToDeviceTransfer(
        __in Accelerator * pAccelerator
        )
    {
        if(!s_bEnableDirectXP2PAPIs) 
            return FALSE;
        if(m_d3dFeatureLevel < D3D_FEATURE_LEVEL_11_0)
            return FALSE;
        if(pAccelerator == NULL) 
            return TRUE; 
        if(pAccelerator->GetClass() != ACCELERATOR_CLASS_DIRECT_X) 
            return FALSE;
        DXAccelerator * pDXAcc = reinterpret_cast<DXAccelerator*>(pAccelerator);
        return (pDXAcc->m_d3dFeatureLevel >= D3D_FEATURE_LEVEL_11_0);
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
    DXAccelerator::DeviceToDeviceTransfer(
        __inout PBuffer *       pDstBuffer,
        __in    PBuffer *       pSrcBuffer,
        __in    AsyncContext *  pAsyncContext
        ) 
    {
        if(!SupportsDeviceToDeviceTransfer(pDstBuffer->GetAccelerator()))
            return FALSE;
        PDXBuffer * pDXBuffer = dynamic_cast<PDXBuffer*>(pSrcBuffer);
        return pDXBuffer->DeviceToDeviceTransfer(pDstBuffer, pAsyncContext);
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
    DXAccelerator::SupportsPinnedHostMemory(
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
    /// <param name="pbResultPageLocked">   [in,out] If non-null, the result of whether the
    /// 									allocated memory is page-locked is provided here. </param>
    ///
    /// <returns>   byte pointer on success, null on failure. </returns>
    ///-------------------------------------------------------------------------------------------------

    void *      
    DXAccelerator::AllocatePagelockedHostMemory(
        UINT uiBytes, 
        BOOL * pbResultPageLocked
        )
    {
        UNREFERENCED_PARAMETER(uiBytes);
        *pbResultPageLocked = FALSE;
        assert(false && "DXAccelerator has no specialized host allocator!");
        PTask::Runtime::HandleError("%s: DXAccelerator has no specialized host allocator!", __FUNCTION__);
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
    DXAccelerator::FreeHostMemory(
        void * pBuffer,
        BOOL bPageLocked
        )
    {
        UNREFERENCED_PARAMETER(pBuffer);
        UNREFERENCED_PARAMETER(bPageLocked);
        assert(false && "DXAccelerator has no specialized host allocator!");
        PTask::Runtime::HandleError("%s: DXAccelerator has no specialized host allocator!", __FUNCTION__);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synchronizes the context. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pTask">    (optional) [in] If non-null, the task. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL         
    DXAccelerator::Synchronize(
        Task * pTask
        )
    {
        UNREFERENCED_PARAMETER(pTask);
        BOOL bSuccess = TRUE;
        return bSuccess;
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
    DXAccelerator::SupportsDeviceMemcpy(
        VOID
        )
    {
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Searches for the first dxsdk shader file cch. Tries to find the location of the shader
    ///     file This is a trimmed down version of DXUTFindDXSDKMediaFileCch. It only addresses the
    ///     following issue to allow the sample correctly run from within Sample Browser directly.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> shader file path. </param>
    ///
    /// <returns>   The found dxsdk shader file cch. </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    DXAccelerator::FindDXSDKShaderFileCch( 
        __in_ecount(cchDest) WCHAR* strDestPath,
        int cchDest, 
        __in LPCWSTR strFilename 
        )
    {
        if( NULL == strFilename || strFilename[0] == 0 || NULL == strDestPath || cchDest < 10 )
            return E_INVALIDARG;

        // Get the exe name, and exe path
        WCHAR strExePath[MAX_PATH] = { 0 };
        WCHAR strExeName[MAX_PATH] = { 0 };
        WCHAR* strLastSlash = NULL;
        GetModuleFileName( NULL, strExePath, MAX_PATH );
        strExePath[MAX_PATH - 1] = 0;
        strLastSlash = wcsrchr( strExePath, TEXT( '\\' ) );
        if( strLastSlash )
        {
            wcscpy_s( strExeName, MAX_PATH, &strLastSlash[1] );

            // Chop the exe name from the exe path
            *strLastSlash = 0;

            // Chop the .exe from the exe name
            strLastSlash = wcsrchr( strExeName, TEXT( '.' ) );
            if( strLastSlash )
                *strLastSlash = 0;
        }

        // Search in directories:
        //      .\
        //      %EXE_DIR%\..\..\%EXE_NAME%

        wcscpy_s( strDestPath, cchDest, strFilename );
        if( GetFileAttributes( strDestPath ) != 0xFFFFFFFF )
            return true;

        swprintf_s( strDestPath, cchDest, L"%s\\..\\..\\%s\\%s", strExePath, strExeName, strFilename );
        if( GetFileAttributes( strDestPath ) != 0xFFFFFFFF )
            return true;    

        // On failure, return the file as the path but also return an error code
        wcscpy_s( strDestPath, cchDest, strFilename );

        return E_FAIL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if this accelerator's device context is current. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <returns>   true if the context is current. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    DXAccelerator::IsDeviceContextCurrent(
        VOID
        ) 
    { 
        // this concept is not present in the 
        // DirectX 11 APIs we use to support DXAccelerator
        return TRUE; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Makes the context current. Return true always because DX11 contexts are always
    ///             current by construction.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    DXAccelerator::MakeDeviceContextCurrent(
        VOID
        ) 
    { 
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the current context. No-op for DX11. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void            
    DXAccelerator::ReleaseCurrentDeviceContext(
        VOID
        ) 
    { 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the D3D feature level for the hardware behind this accelerator object. There
    ///             is no requirement to hold the accelerator lock because the property is
    ///             initialized by the scheduler and read-only thereafter.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The feature level. </returns>
    ///-------------------------------------------------------------------------------------------------

    D3D_FEATURE_LEVEL 
    DXAccelerator::GetFeatureLevel(
        VOID
        ) 
    { 
        return m_d3dFeatureLevel; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if the front-end programming model supports function arguments for
    ///             top-level kernel invocations. DirectX requires top-level invocations to find
    ///             their inputs at global scope in constant buffers and
    ///             StructuredBuffers/RWStructuredBuffers, etc. so this function always returns false
    ///             for this class. There is no requirement to hold the accelerator lock because the
    ///             property is initialized by the scheduler and read-only thereafter.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   FALSE. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    DXAccelerator::SupportsFunctionArguments(
        VOID
        ) 
    { 
        return FALSE; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if the underlying platform supports byval arguments for kernel
    ///             invocations. If the platform does support this, PTask can elide explicit creation
    ///             and population of buffers to back these arguments, which is a performance win
    ///             when it is actually supported. DirectX does not support this sort of thing so we
    ///             always return false. There is no requirement to hold the accelerator lock because
    ///             the property is initialized by the scheduler and read-only thereafter.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   FALSE. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    DXAccelerator::SupportsByvalArguments(
        VOID
        ) 
    { 
        return FALSE; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the adapter for this accelerator. There is no requirement to hold the
    ///             accelerator lock because the property is initialized by the scheduler and read-
    ///             only thereafter.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the adapter. </returns>
    ///-------------------------------------------------------------------------------------------------

    IDXGIAdapter*   
    DXAccelerator::GetAdapter(
        VOID
        ) 
    { 
        return m_pAdapter; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the adapter description. There is no requirement to hold the accelerator
    ///             lock because the property is initialized by the scheduler and read-only
    ///             thereafter.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the adapter description. </returns>
    ///-------------------------------------------------------------------------------------------------

    DXGI_ADAPTER_DESC* 
    DXAccelerator::GetAdapterDesc(
        VOID
        ) 
    { 
        return &m_desc; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Warmup the DirectX pipeline--the first dispatch is always bloody slow.
    ///             Perform one when bringing the accelerator up to see if we can 
    ///             get some overheads out of the critical path.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/28/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    DXAccelerator::WarmupPipeline(
        VOID
        ) 
    {
#ifdef DIRECTXCOMPILESUPPORT
        const UINT NUM_COLS = 4;
        const UINT NUM_ROWS = 4;
        struct ELEMTYPE {    int i;    float f; };
        struct MATRIXPARAMS { 	
            UINT g_tex_cols;
            UINT g_tex_rows;
            UINT g_pad0;
            UINT g_pad1;
        } parms = { NUM_ROWS, NUM_COLS, 0, 0 };

        char * pszCode = 
        "struct BufType {    int i;    float f; };\n"
        "StructuredBuffer<BufType> Buffer0 : register(t0);\n"
        "StructuredBuffer<BufType> Buffer1 : register(t1);\n"
        "RWStructuredBuffer<BufType> BufferOut : register(u0);\n"
        "[numthreads(1, 1, 1)]\n"
        "void CSMain( uint3 DTid : SV_DispatchThreadID )\n"
        "{\n"
        "    BufferOut[DTid.x].i = Buffer0[DTid.x].i + Buffer1[DTid.x].i;\n"
        "    BufferOut[DTid.x].f = Buffer0[DTid.x].f + Buffer1[DTid.x].f;\n"
        "}\n";

	    const D3D_SHADER_MACRO defines[] = 
        {
		    "thread_group_size_x", "1",
		    "thread_group_size_y", "1",
		    "thread_group_size_z", "1",
            NULL, NULL
        };

        PTask::Runtime::Inform("Warming up direct X pipeline...");

        HRESULT hr = S_OK;
        ID3DBlob* pErrorBlob = NULL;
        ID3DBlob* pBlob = NULL;
        LPCSTR pProfile = ( m_pDevice->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0 ) ? "cs_5_0" : "cs_4_0";
        hr = D3DCompile(pszCode, strlen(pszCode), NULL, defines, NULL, "CSMain", pProfile, NULL, NULL, &pBlob, &pErrorBlob);
        if ( FAILED(hr) ) {
            if ( pErrorBlob )
                PTask::Runtime::ErrorMessage( (char*)pErrorBlob->GetBufferPointer() );
            PTSRELEASE( pErrorBlob );
            PTSRELEASE( pBlob );    
            return;
        }    

        ID3D11ComputeShader* pShaderOut = NULL;
        hr = m_pDevice->CreateComputeShader(pBlob->GetBufferPointer(), 
                                            pBlob->GetBufferSize(), 
                                            NULL, 
                                            &pShaderOut);
        PTSRELEASE( pErrorBlob );
        PTSRELEASE( pBlob );

        PTask::Runtime::Inform( "DX Warmup: Creating buffers and filling them with initial data..." );
	    ELEMTYPE * pA = new ELEMTYPE[NUM_ROWS*NUM_COLS];
	    ELEMTYPE * pB = new ELEMTYPE[NUM_ROWS*NUM_COLS];
        ELEMTYPE * pC = new ELEMTYPE[NUM_ROWS*NUM_COLS];

        ID3D11Buffer * pBufA = NULL;
        ID3D11Buffer * pBufB = NULL;
        ID3D11Buffer * pBufC = NULL;
        ID3D11Buffer * pConst = NULL;
        ID3D11Buffer* pResult = NULL;
        ID3D11ShaderResourceView * pBufASRV = NULL;
        ID3D11ShaderResourceView * pBufBSRV = NULL;
        ID3D11ShaderResourceView * pBufCSRV = NULL;
        ID3D11UnorderedAccessView * pBufCUAV = NULL;

        D3D11_BUFFER_DESC desc;
        D3D11_BUFFER_DESC resdesc;
        D3D11_SUBRESOURCE_DATA InitData;
        ZeroMemory( &desc, sizeof(desc) );
        desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        desc.ByteWidth = NUM_ROWS*NUM_COLS*sizeof(ELEMTYPE);
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(ELEMTYPE);
        InitData.pSysMem = pA;
        hr = m_pDevice->CreateBuffer( &desc, &InitData, &pBufA );
        InitData.pSysMem = pB;
        hr = m_pDevice->CreateBuffer( &desc, &InitData, &pBufB );
        InitData.pSysMem = pC;
        hr = m_pDevice->CreateBuffer( &desc, &InitData, &pBufC );

        PTask::Runtime::Inform( "DX Warmup: Creating buffer views..." );
        PTask::Runtime::Inform( "DX Warmup: Creating constant buffers..." );
        UINT byteWidth = sizeof(MATRIXPARAMS);
	    UINT pad = (byteWidth % 16)?(16-(byteWidth%16)):0;
	    BOOL bPadConstBuf = pad > 0;
	    VOID * pPaddedBuffer = NULL;
	    VOID * pInitPtr = &parms;
	    if(bPadConstBuf) {
		    pPaddedBuffer = malloc(byteWidth+pad);
		    memset(pPaddedBuffer, 0, byteWidth+pad);
		    memcpy(pPaddedBuffer, &parms, byteWidth);
		    pInitPtr = pPaddedBuffer;
	    }
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        desc.MiscFlags = 0;    
	    desc.ByteWidth = bPadConstBuf?(byteWidth+pad):byteWidth;
        InitData.pSysMem = pInitPtr;
        if(FAILED(hr = m_pDevice->CreateBuffer( &desc, &InitData, &pConst ))) {
			if(pPaddedBuffer) free(pPaddedBuffer);
			return;
		}
	    if(pPaddedBuffer) free(pPaddedBuffer);

        PTask::Runtime::Inform( "DX Warmup: Creating SRVs..." );
        D3D11_BUFFER_DESC descBuf;
        D3D11_SHADER_RESOURCE_VIEW_DESC rdesc;
        ZeroMemory( &descBuf, sizeof(descBuf) );
        pBufA->GetDesc( &descBuf );
        ZeroMemory( &rdesc, sizeof(rdesc) );
        rdesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
        rdesc.BufferEx.FirstElement = 0;
        rdesc.BufferEx.NumElements = descBuf.ByteWidth / descBuf.StructureByteStride;
        m_pDevice->CreateShaderResourceView( pBufA, &rdesc, &pBufASRV );
        m_pDevice->CreateShaderResourceView( pBufB, &rdesc, &pBufBSRV );
        m_pDevice->CreateShaderResourceView( pBufB, &rdesc, &pBufCSRV );

        PTask::Runtime::Inform( "DX Warmup: Creating UAVs..." );
        pBufC->GetDesc( &descBuf );        
        D3D11_UNORDERED_ACCESS_VIEW_DESC udesc;
        ZeroMemory( &udesc, sizeof(udesc) );
        udesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        udesc.Buffer.FirstElement = 0;
        udesc.Format = DXGI_FORMAT_UNKNOWN;      // Format must be must be DXGI_FORMAT_UNKNOWN, when creating a View of a Structured Buffer
        udesc.Buffer.NumElements = descBuf.ByteWidth / descBuf.StructureByteStride;     
        m_pDevice->CreateUnorderedAccessView( pBufC, &udesc, &pBufCUAV );
        PTask::Runtime::Inform( "DX Warmup: done setting up\n" );
        
        ZeroMemory( &resdesc, sizeof(resdesc) );
        pBufC->GetDesc( &resdesc );
        resdesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        resdesc.Usage = D3D11_USAGE_STAGING;
        resdesc.BindFlags = 0;
        resdesc.MiscFlags = 0;
        m_pDevice->CreateBuffer(&resdesc, NULL, &pResult);

        PTask::Runtime::Inform( "DX Warmup: Running Compute Shader(s)..." );
        for(int i=0;i<100;i++) {
            ID3D11ShaderResourceView* aRViews[2] = { pBufASRV, pBufBSRV };
            m_pContext->CSSetShader( pShaderOut, NULL, 0 );
            m_pContext->CSSetShaderResources( 0, 2, aRViews );
            m_pContext->CSSetUnorderedAccessViews( 0, 1, &pBufCUAV, NULL );
            m_pContext->CSSetConstantBuffers( 0, 1, &pConst );
            m_pContext->Dispatch( NUM_ROWS, NUM_COLS, 1);
            m_pContext->CSSetShader( NULL, NULL, 0 );
            ID3D11UnorderedAccessView* ppUAViewNULL[1] = { NULL };
            m_pContext->CSSetUnorderedAccessViews( 0, 1, ppUAViewNULL, NULL );
            ID3D11ShaderResourceView* ppSRVNULL[2] = { NULL, NULL };
            m_pContext->CSSetShaderResources( 0, 2, ppSRVNULL );
            ID3D11Buffer* ppCBNULL[1] = { NULL };
            m_pContext->CSSetConstantBuffers( 0, 1, ppCBNULL );
            m_pContext->CopyResource( pResult, pBufC );
        }
        PTask::Runtime::Inform( "DX Warmup: complete\n" );

        PTSRELEASE( pResult );
        PTSRELEASE( pBufASRV );
        PTSRELEASE( pBufBSRV );
        PTSRELEASE( pBufCSRV );
        PTSRELEASE( pBufCUAV );
        PTSRELEASE( pBufA );
        PTSRELEASE( pBufB );
        PTSRELEASE( pBufC );
        PTSRELEASE( pShaderOut );
	    PTSRELEASE( pConst );
	    delete pA;
	    delete pB;
	    delete pC;
#endif
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
    ///             The function is not abstract because most accelerator classes don't support async
    ///             operations yet. In DirectX it is unnecessary because the DX runtime manages these
    ///             dependences under the covers, and in OpenCL the API is present, but we do not
    ///             yet take advantage of it.  So it's simpler to override a default implementation
    ///             that returns FALSE.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    DXAccelerator::SupportsExplicitAsyncOperations(
        VOID
        )
    {
        return s_bEnableDirectXAsyncrony;
    }

};
