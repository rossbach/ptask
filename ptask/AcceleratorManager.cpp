///-------------------------------------------------------------------------------------------------
// file:	AcceleratorManager.cpp
//
// summary:	Implements the accelerator manager class
///-------------------------------------------------------------------------------------------------

#include "AcceleratorManager.h"
#include <vector>
#include <assert.h>
#include <algorithm>
#include "accelerator.h"
using namespace std;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    AcceleratorManager::AcceleratorManager(
        VOID
        ) : Lockable("AcceleratorManager")
    {        
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    AcceleratorManager::~AcceleratorManager(
        void
        )
    {
        Lock();
        vector<PhysicalDevice*>::iterator vi;
        for(vi=m_devices.begin(); vi!=m_devices.end(); vi++) {
            PhysicalDevice * pD = (*vi);
            delete pD;
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a device. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    AcceleratorManager::AddDevice(
        Accelerator * pAccelerator
        )
    {
        Lock();
        vector<PhysicalDevice*>::iterator vi;
        PhysicalDevice * pDevice = Find(pAccelerator);
        if(pDevice != NULL) {
            pDevice->AddInterface(pAccelerator);
            pAccelerator->SetPhysicalDevice(pDevice);
        } else {
            pDevice = new PhysicalDevice();
            pDevice->AddInterface(pAccelerator);
            pAccelerator->SetPhysicalDevice(pDevice);
            m_devices.push_back(pDevice);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Searches for the first match for the given accelerator*. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    PhysicalDevice * 
    AcceleratorManager::Find(
        Accelerator* pAccelerator
        )
    {
        assert(LockIsHeld());
        vector<PhysicalDevice*>::iterator vi;
        PhysicalDevice * pDevice = NULL;
        BOOL bFound = FALSE;
        for(vi=m_devices.begin(); vi!=m_devices.end() && !bFound; vi++) {
            PhysicalDevice * pD = (*vi);
            pD->Lock();
            if(pD->IsSameDevice(pAccelerator)) {
                pDevice = pD;
                bFound = TRUE;
            }
            pD->Unlock();
        }
        return pDevice;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'pAccelerator' is available. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   true if available, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    AcceleratorManager::IsAvailable(
        Accelerator * pAccelerator
        )
    {
        assert(LockIsHeld());
        PhysicalDevice * pDevice = Find(pAccelerator);
        if(pDevice == NULL) {
            assert(FALSE);
            return FALSE;
        }
        return !pDevice->IsBusy();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Searches for the first available. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="cls">  The cls. </param>
    /// <param name="v">    [in,out] [in,out] If non-null, the v. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    AcceleratorManager::FindAvailable(
        ACCELERATOR_CLASS cls, 
        std::vector<Accelerator*> &v
        )
    {
        assert(LockIsHeld());
        vector<PhysicalDevice*>::iterator vi;
        for(vi=m_devices.begin(); vi!=m_devices.end(); vi++) {
            PhysicalDevice * pD = (*vi);
            pD->Lock();
            if(!pD->IsBusy()) {
                Accelerator * pAccelerator = pD->GetAcceleratorInterface(cls);
                if(pAccelerator != NULL) {
                    v.push_back(pAccelerator);
                }
            }
            pD->Unlock();
        }
        return v.size() > 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a physical accelerator count. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The physical accelerator count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    AcceleratorManager::GetPhysicalAcceleratorCount(
        VOID
        )
    {
        UINT nAccelerators = 0;
        Lock();
        vector<PhysicalDevice*>::iterator vi;
        for(vi=m_devices.begin(); vi!=m_devices.end(); vi++) {
            PhysicalDevice * pD = (*vi);
            pD->Lock();
            // we don't want to count "host" accelerators,
            // since they are not really accelerators. What 
            // this function really returns is the number of
            // accelerators with private memory spaces.
            if(pD->Supports(ACCELERATOR_CLASS_DIRECT_X) ||
                pD->Supports(ACCELERATOR_CLASS_CUDA) ||
                pD->Supports(ACCELERATOR_CLASS_OPEN_CL)) {
                nAccelerators++;
            }
            pD->Unlock();
        }
        Unlock();
        return nAccelerators;
    }

};
