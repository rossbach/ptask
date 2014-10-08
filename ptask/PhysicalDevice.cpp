///-------------------------------------------------------------------------------------------------
// file:	PhysicalDevice.cpp
//
// summary:	Implements the physical device class
///-------------------------------------------------------------------------------------------------

#include "PhysicalDevice.h"
#include <vector>
#include <assert.h>
#include <algorithm>
#include "dxaccelerator.h"
#include "claccelerator.h"
#include "cuaccelerator.h"
#include "hostaccelerator.h"
#include "PTaskRuntime.h"
#include "hrperft.h"
#include "oclhdr.h"
#ifdef CUDA_SUPPORT
#include "cudaD3D11.h"
#endif
#ifdef OPENCL_SUPPORT
#include "CL\cl_d3d11_ext.h"
#endif
#include "extremetrace.h"
using namespace std;




namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    PhysicalDevice::PhysicalDevice(
        VOID
        ) : Lockable(NULL)

    {
#ifdef CUDA_SUPPORT
        m_pCUDADevice = NULL;
        m_pCUAccelerator = NULL;
#endif
#ifdef OPENCL_SUPPORT
        m_pOpenCLDevice = NULL;
        m_pCLAccelerator = NULL;
#endif
        m_pDirectXDevice = NULL;
        m_pDXAccelerator = NULL;
        m_bInFlight = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    PhysicalDevice::~PhysicalDevice(
        void
        )
    {
        assert(!LockIsHeld());
        Lock();
#ifdef CUDA_SUPPORT
        if(m_pCUDADevice) {
            delete m_pCUDADevice;
        }
#endif
#ifdef OPENCL_SUPPORT
        if(m_pOpenCLDevice) {
            delete m_pOpenCLDevice;
        }
#endif
        if(m_pDirectXDevice) {
            PTSRELEASE(m_pDirectXDevice->pAdapter);
            delete m_pDirectXDevice;
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pDevice' is same device. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pDevice">          [in,out] If non-null, the device. </param>
    /// <param name="pDesc">            [in,out] If non-null, the description. </param>
    /// <param name="nPlatformIndex">   Zero-based index of the n platform. </param>
    ///
    /// <returns>   true if same device, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    PhysicalDevice::IsSameDevice(
        IDXGIAdapter * pDevice, 
        DXGI_ADAPTER_DESC * pDesc,
        UINT nPlatformIndex
        )
    {
        assert(LockIsHeld());

        if(m_pDirectXDevice != NULL) {

            // could be the same adapter record
            // object or IDXGIAdapter pointer
            if(pDevice == m_pDirectXDevice->pAdapter) {
                assert(m_pDXAccelerator->GetPlatformIndex() == nPlatformIndex);
                return TRUE;
            }
            // could still be different interfaces opened
            // on the same device. In this case the
            // descriptor will tell us. 
            if(m_pDirectXDevice->desc.AdapterLuid.HighPart == pDesc->AdapterLuid.HighPart &&
                m_pDirectXDevice->desc.AdapterLuid.LowPart == pDesc->AdapterLuid.LowPart) {
                assert(m_pDXAccelerator->GetPlatformIndex() == nPlatformIndex);
                return TRUE;
            }
            return FALSE;
        }

#ifdef CUDA_SUPPORT
        if(m_pCUDADevice != NULL) {
            // the cuD3D11GetDevice API appears to 
            // be unreliable. Rely on the platform index
            // before trusting the result of that function call.
            CUdevice dev;
            trace("cuD3D11GetDevice\n");
            CUresult res = cuD3D11GetDevice(&dev, pDevice);
            if(res != CUDA_SUCCESS) {
                assert(false);
                printf("cuD3D11GetDevice failed with err=%d!\n", res);
                return FALSE;
            }
            if(dev == m_pCUDADevice->device) {
                assert(m_pCUAccelerator->GetPlatformIndex() == nPlatformIndex);
                return TRUE;
            }
            return m_pCUAccelerator->GetPlatformIndex() == nPlatformIndex;
        }
#endif

#ifdef OPENCL_SUPPORT
        if(m_pOpenCLDevice != NULL) {
            // API doesn't work!
            if(m_pCUAccelerator != NULL) {
                return m_pCLAccelerator->GetPlatformIndex() == nPlatformIndex;
            }
            return FALSE;
            //cl_uint nFoundDevices;
            //cl_device_id vDevices[10];
            //cl_uint nMaxDevices = sizeof(vDevices)/sizeof(cl_device_id);
            //cl_int err = clGetDeviceIDsFromD3D11KHR(m_pOpenCLDevice->platform,  
            //                                        CL_D3D11_DXGI_ADAPTER_KHR,
            //                                        (void*) pDevice,
            //                                        CL_PREFERRED_DEVICES_FOR_D3D11_KHR,
            //                                        nMaxDevices,
            //                                        vDevices,
            //                                        &nFoundDevices);
            //switch(err) {
            //case CL_SUCCESS:
            //    assert(nFoundDevices == 1);
            //    return vDevices[0] == m_pOpenCLDevice->device;
            //case CL_INVALID_PLATFORM:
            //case CL_INVALID_VALUE:
            //    assert(false);
            //    return FALSE;
            //case CL_DEVICE_NOT_FOUND:
            //    return FALSE;
            // }
        }
#endif
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'platform' is same device. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="platform">         The platform. </param>
    /// <param name="device">           The device. </param>
    /// <param name="nPlatformIndex">   Zero-based index of the n platform. </param>
    ///
    /// <returns>   true if same device, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------
#ifdef OPENCL_SUPPORT
    BOOL 
    PhysicalDevice::IsSameDevice(
        cl_platform_id platform,
        cl_device_id device,
        UINT nPlatformIndex
        )
    {
        assert(LockIsHeld());
        if(m_pDirectXDevice != NULL) {
            if(m_pDXAccelerator->GetPlatformIndex() == nPlatformIndex) {
                return TRUE;
            }
        }
        if(m_pCUDADevice != NULL) {            
            return m_pCUAccelerator->GetPlatformIndex() == nPlatformIndex;
        }
        if(m_pOpenCLDevice != NULL) {
            if(m_pOpenCLDevice->platform == platform && m_pOpenCLDevice->device == device)
                return TRUE;
            return (m_pCUAccelerator != NULL && m_pCLAccelerator->GetPlatformIndex() == nPlatformIndex);
        }
        return FALSE;
    }
#endif

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'device' is same device. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="device">           The device. </param>
    /// <param name="nPlatformIndex">   Zero-based index of the n platform. </param>
    ///
    /// <returns>   true if same device, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

#ifdef CUDA_SUPPORT
    BOOL 
    PhysicalDevice::IsSameDevice(
        CUdevice device,
        UINT nPlatformIndex
        )
    {
        assert(LockIsHeld());
        if(m_pDirectXDevice != NULL) {
            CUdevice dev;
            trace("cuD3D11GetDevice\n");
            CUresult res = cuD3D11GetDevice(&dev, m_pDirectXDevice->pAdapter);
			if(res == CUDA_ERROR_NOT_FOUND) {
                PTask::Runtime::Warning("cuD3D11GetDevice could not find a given DX adapter...XXXX!\n");
                return FALSE;
			} else if(res != CUDA_SUCCESS) {
                assert(false);
                printf("cuD3D11GetDevice failed with err=%d!\n", res);
                return FALSE;
            }
            if(m_pDXAccelerator->GetPlatformIndex() == nPlatformIndex)
                return dev == device;
            return FALSE;
        }
        if(m_pCUDADevice != NULL) {
            return device == m_pCUDADevice->device;
        }
#ifdef OPENCL_SUPPORT
        if(m_pOpenCLDevice != NULL) {
            return m_pCUAccelerator->GetPlatformIndex() == nPlatformIndex;
        }
#endif
        return FALSE;    
    }
#endif

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pAccelerator' is same device. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
    ///
    /// <returns>   true if same device, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    PhysicalDevice::IsSameDevice(
        Accelerator * pAccelerator
        )
    {
        assert(LockIsHeld());
        if(pAccelerator == NULL) {
            assert(FALSE);
            return FALSE;
        }
        DXAccelerator * pDX = NULL;
        ACCELERATOR_CLASS aclass = pAccelerator->GetClass();
        switch(aclass) {

        case ACCELERATOR_CLASS_DIRECT_X: {
            pDX = (DXAccelerator*) pAccelerator;
            IDXGIAdapter * pAdapter = pDX->GetAdapter();
            DXGI_ADAPTER_DESC * pDesc = pDX->GetAdapterDesc();
            return IsSameDevice(pAdapter, pDesc, pDX->GetPlatformIndex());
        }

        case ACCELERATOR_CLASS_CUDA: {
#ifdef CUDA_SUPPORT
            return IsSameDevice((CUdevice)pAccelerator->GetDevice(), pAccelerator->GetPlatformIndex());
#else
            assert(FALSE);
            return FALSE;
#endif
        }

        case ACCELERATOR_CLASS_OPEN_CL: {
#ifdef OPENCL_SUPPORT
            CLAccelerator * pCL = (CLAccelerator*) pAccelerator;
            return IsSameDevice(pCL->GetPlatformId(), ((cl_device_id) pCL->GetDevice()), pCL->GetPlatformIndex());
#else
            assert(FALSE);
            return FALSE;
#endif
        }

        case ACCELERATOR_CLASS_HOST:
        default:
            return FALSE;
        }
    } 

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an interface. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    PhysicalDevice::AddInterface(
        Accelerator * pAccelerator
        )
    {
        BOOL bSuccess = FALSE;
        if(pAccelerator == NULL) {
            assert(FALSE);
            return bSuccess;
        } 
        DXAccelerator * pDX = NULL;
        ACCELERATOR_CLASS aclass = pAccelerator->GetClass();

        Lock();
        switch(aclass) {

        case ACCELERATOR_CLASS_DIRECT_X: {
            assert(m_pDirectXDevice == NULL);
            pDX = (DXAccelerator*) pAccelerator;
            IDXGIAdapter * pAdapter = pDX->GetAdapter();
            DXGI_ADAPTER_DESC * pDesc = pDX->GetAdapterDesc();
            m_pDirectXDevice = new DIRECTX_DEVICERECORD();
            memcpy(&m_pDirectXDevice->desc, pDesc, sizeof(DXGI_ADAPTER_DESC));
            m_pDirectXDevice->pAdapter = pAdapter;
            m_pDXAccelerator = pDX;
            bSuccess = TRUE;
            break;
        }

        case ACCELERATOR_CLASS_CUDA:  {
#ifdef CUDA_SUPPORT
            assert(m_pCUDADevice == NULL);
            m_pCUDADevice = new CUDA_DEVICERECORD();
            m_pCUDADevice->device = (CUdevice)pAccelerator->GetDevice();
            m_pCUAccelerator = (CUAccelerator*) pAccelerator;
            bSuccess = TRUE;
#else
            assert(FALSE);
#endif
            break;
        }

        case ACCELERATOR_CLASS_OPEN_CL: {
#ifdef OPENCL_SUPPORT
            assert(m_pOpenCLDevice == NULL);
            CLAccelerator* pCL = (CLAccelerator*) pAccelerator;
            m_pOpenCLDevice = new OPENCL_DEVICERECORD();
            m_pOpenCLDevice->device = ((cl_device_id) pCL->GetDevice());
            m_pOpenCLDevice->platform = pCL->GetPlatformId();
            m_pCLAccelerator = pCL;
            bSuccess = TRUE;
#else
            assert(FALSE);
#endif
            break;        
        }

        case ACCELERATOR_CLASS_HOST:
        default:
            break;
        }
        Unlock();
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return TRUE if this device supports the given accelerator class. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="cls">  The accelerator class. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    PhysicalDevice::Supports(
        ACCELERATOR_CLASS cls
        )
    {
        assert(LockIsHeld());
        switch(cls) {
        case ACCELERATOR_CLASS_DIRECT_X: 
            return m_pDirectXDevice != NULL;
        case ACCELERATOR_CLASS_CUDA: 
#ifdef CUDA_SUPPORT
            return m_pCUDADevice != NULL;
#else
            return FALSE;
#endif
        case ACCELERATOR_CLASS_OPEN_CL:
#ifdef OPENCL_SUPPORT
            return m_pOpenCLDevice != NULL;
#else
            return FALSE;
#endif
        case ACCELERATOR_CLASS_HOST:
        default:
            return FALSE;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets an accelerator interface on this physical device for the given class. Return
    ///             NULL if no such interface is present on this device.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="cls">  The accelerator class. </param>
    ///
    /// <returns>   null if it fails, else the accelerator interface. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    PhysicalDevice::GetAcceleratorInterface(
        ACCELERATOR_CLASS cls
        )
    {
        assert(LockIsHeld());
        switch(cls) {
        case ACCELERATOR_CLASS_DIRECT_X: 
            return m_pDXAccelerator;
        case ACCELERATOR_CLASS_CUDA: 
#ifdef CUDA_SUPPORT
            return m_pCUAccelerator;
#else   
            assert(FALSE);
            return NULL;
#endif
        case ACCELERATOR_CLASS_OPEN_CL:
#ifdef OPENCL_SUPPORT      
            return m_pCLAccelerator;
#else
            assert(FALSE);
            return NULL;
#endif
        case ACCELERATOR_CLASS_HOST:
        default:
            return NULL;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Mark this device busy. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="bBusy">    true to busy. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    PhysicalDevice::SetBusy(
        BOOL bBusy
        )
    {
        assert(LockIsHeld());
        m_bInFlight = bBusy;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the physical device is busy. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if busy, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    PhysicalDevice::IsBusy(
        VOID
        )
    {
        assert(LockIsHeld());
        return m_bInFlight;
    }
};
