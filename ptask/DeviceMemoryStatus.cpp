///-------------------------------------------------------------------------------------------------
// file:	DeviceMemoryStatus.cpp
//
// summary:	Implements the device memory status class
///-------------------------------------------------------------------------------------------------

#include "DeviceMemoryStatus.h"
#include "ptaskutils.h"
#include "assert.h"
#include "accelerator.h"
#include "HostAccelerator.h"

namespace PTask {

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        DeviceMemoryStatus_t::DeviceMemoryStatus_t(
            std::string &szName,
            char * lpszUniquifier
            )
        {
            m_name = szName;
            if(lpszUniquifier) 
               m_name += lpszUniquifier;
            m_uiMemorySpaceSize = 0;
            Reset();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        DeviceMemoryStatus_t::~DeviceMemoryStatus_t() {
            m_vAllocations.clear();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets this object. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        DeviceMemoryStatus_t::Reset(
            VOID
            ) 
        {
            m_vAllocations.clear();
            m_uiMinAllocExtentSize = MAXDWORD64;
            m_uiMaxAllocExtentSize = 0;
            m_uiLowWaterMarkBytes = MAXDWORD64;
            m_uiHighWaterMarkBytes = 0;
            m_uiCurrentlyAllocatedBytes = 0;
            m_uiCurrentlyAllocatedBuffers = 0;
            m_uiAllocationRequests = 0;
            m_uiDeallocationRequests = 0;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record a memory allocation. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pMemoryExtent">    [in,out] If non-null, extent of the memory. </param>
        /// <param name="uiBytes">          The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    
        DeviceMemoryStatus_t::RecordAllocation(
            __in void * pMemoryExtent,
            __in size_t uiBytes
            )
        {
            m_vAllocations[pMemoryExtent] = uiBytes;
            m_uiAllocationRequests++;
            m_uiCurrentlyAllocatedBuffers++;
            m_uiCurrentlyAllocatedBytes += uiBytes;
            m_uiMinAllocExtentSize = min(m_uiMinAllocExtentSize, uiBytes);
            m_uiMaxAllocExtentSize = max(m_uiMaxAllocExtentSize, uiBytes);
            m_uiLowWaterMarkBytes = min(m_uiLowWaterMarkBytes, m_uiCurrentlyAllocatedBytes);
            m_uiHighWaterMarkBytes = max(m_uiHighWaterMarkBytes, m_uiCurrentlyAllocatedBytes);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record a memory deallocation. We provide "require entry" flag to
        ///             simplify tracking of page-locked allocations which are a strict subset
        ///             of all allocations. If we are removing an entry from the global tracking,
        ///             we require that an entry for it be found, otherwise we complain. If
        ///             we are removing entries from the page-locked tracking, it is not an  
        ///             error if there is no entry present.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pMemoryExtent">    [in,out] If non-null, extent of the memory. </param>
        /// <param name="bRequireEntry">    true to pinned allocation. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    
        DeviceMemoryStatus_t::RecordDeallocation(
            __in void * pMemoryExtent,
            __in BOOL bRequireEntry
            )
        {
            std::map<void*, unsigned __int64>::iterator mi;
            mi = m_vAllocations.find(pMemoryExtent); 
            if(bRequireEntry) {
                assert(mi != m_vAllocations.end());
            }
            if(mi != m_vAllocations.end()) {
                unsigned __int64 uiBytes = mi->second;
                m_vAllocations.erase(mi);
                m_uiDeallocationRequests++;
                m_uiCurrentlyAllocatedBuffers--;
                m_uiCurrentlyAllocatedBytes -= uiBytes;
                m_uiLowWaterMarkBytes = min(m_uiLowWaterMarkBytes, m_uiCurrentlyAllocatedBytes);
                m_uiHighWaterMarkBytes = max(m_uiHighWaterMarkBytes, m_uiCurrentlyAllocatedBytes);
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the allocation statistics. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        DeviceMemoryStatus_t::Report(
            std::ostream &ios
            ) 
        {
            if((m_vAllocations.size() == 0) && 
               (m_uiMinAllocExtentSize == MAXDWORD64) &&
               (m_uiMaxAllocExtentSize == 0) &&
               (m_uiLowWaterMarkBytes == MAXDWORD64) &&
               (m_uiHighWaterMarkBytes == 0) && 
               (m_uiCurrentlyAllocatedBytes == 0) && 
               (m_uiCurrentlyAllocatedBuffers == 0) &&
               (m_uiAllocationRequests == 0) &&
               (m_uiDeallocationRequests == 0)) {

                // this memory space is in it default initial state. 
                // we can therefore safely conclude it entirely unused
                // and safe a lot of space in the output simply by stating
                // this fact.
                
                ios << "memory state for " << m_name << ": UNUSED" << std::endl;

            } else {

                // this memory space was actually used...report on it.

                std::stringstream ssMinExtent;
                std::stringstream ssLowWaterMark;

                if(m_uiMinAllocExtentSize == MAXDWORD64) ssMinExtent << "---";
                else ssMinExtent << m_uiMaxAllocExtentSize;

                if(m_uiLowWaterMarkBytes == MAXDWORD64) ssLowWaterMark << "---";
                else ssLowWaterMark << m_uiLowWaterMarkBytes;

                ios << "memory state for " << m_name << ":" << std::endl;
                ios << "\tm_uiMemorySpaceSize       :" << m_uiMemorySpaceSize            << std::endl;
                ios << "\tMinAllocExtentSize        :" << ssMinExtent.str()              << std::endl;
                ios << "\tMaxAllocExtentSize        :" << m_uiMaxAllocExtentSize         << std::endl;
                ios << "\tLowWaterMarkBytes         :" << ssLowWaterMark.str()           << std::endl;
                ios << "\tHighWaterMarkBytes        :" << m_uiHighWaterMarkBytes         << std::endl;
                ios << "\tCurrentlyAllocatedBytes   :" << m_uiCurrentlyAllocatedBytes    << std::endl;
                ios << "\tCurrentlyAllocatedBuffers :" << m_uiCurrentlyAllocatedBuffers  << std::endl;
                ios << "\tAllocationRequests        :" << m_uiAllocationRequests         << std::endl;
                ios << "\tDeallocationRequests      :" << m_uiDeallocationRequests       << std::endl;

            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the memory space size described by uiBytes. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="uiBytes">  The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        DeviceMemoryStatus_t::UpdateMemorySpaceSize(
            unsigned __int64 uiBytes
            ) 
        {
            m_uiMemorySpaceSize = uiBytes;
        }
        
        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        GlobalDeviceMemoryState_t::GlobalDeviceMemoryState_t(
            std::string& szDeviceName
            ) : m_global(szDeviceName, NULL),
                m_pagelocked(szDeviceName, "_pagelocked") 
                
        {
             InitializeCriticalSection(&m_lock);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        GlobalDeviceMemoryState_t::~GlobalDeviceMemoryState_t(
            VOID
            )
        {
            Lock();
            m_global.Reset();
            m_pagelocked.Reset();
            Unlock();
            DeleteCriticalSection(&m_lock);
        }

        /// <summary>   synchronization. </summary>
        void GlobalDeviceMemoryState_t::Lock() { EnterCriticalSection(&m_lock); }
        void GlobalDeviceMemoryState_t::Unlock() { LeaveCriticalSection(&m_lock); }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Resets the global memory state stats object. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        GlobalDeviceMemoryState_t::Reset(
            VOID
            ) 
        {
            Lock();
            m_global.Reset();
            m_pagelocked.Reset();
            Unlock();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record a memory allocation. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pMemoryExtent">    [in,out] If non-null, extent of the memory. </param>
        /// <param name="uiBytes">          The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    
        GlobalDeviceMemoryState_t::RecordAllocation(
            __in void * pMemoryExtent,
            __in size_t uiBytes,
            __in BOOL bPinned
            )
        {
            Lock();
            m_global.RecordAllocation(pMemoryExtent, uiBytes);
            if(bPinned) {
                m_pagelocked.RecordAllocation(pMemoryExtent, uiBytes);
            }
            Unlock();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Record a memory deallocation. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="pMemoryExtent">        [in,out] If non-null, extent of the memory. </param>
        /// <param name="bPinnedAllocation">    true to pinned allocation. </param>
        /// <param name="uiBytes">              The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void                    
        GlobalDeviceMemoryState_t::RecordDeallocation(
            __in void * pMemoryExtent
            )
        {
            Lock();
            m_global.RecordDeallocation(pMemoryExtent, TRUE);
            m_pagelocked.RecordDeallocation(pMemoryExtent, FALSE);
            Unlock();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Dumps the allocation statistics. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        void 
        GlobalDeviceMemoryState_t::Report(
            std::ostream &ios
            ) 
        {
            Lock();
            m_global.Report(ios);
            m_pagelocked.Report(ios);
            Unlock();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets global memory state. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the global memory state. </returns>
        ///-------------------------------------------------------------------------------------------------

        MEMSTATEDESC * 
        GlobalDeviceMemoryState_t::GetGlobalMemoryState(
            VOID
            )
        {
            return &m_global;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets global memory state. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <returns>   null if it fails, else the global memory state. </returns>
        ///-------------------------------------------------------------------------------------------------

        MEMSTATEDESC * 
        GlobalDeviceMemoryState_t::GetPageLockedMemoryState(
            VOID
            )
        {
            return &m_pagelocked;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the memory space size described by uiBytes. </summary>
        ///
        /// <remarks>   Crossbac, 3/15/2013. </remarks>
        ///
        /// <param name="uiBytes">  The bytes. </param>
        ///-------------------------------------------------------------------------------------------------

        void
        GlobalDeviceMemoryState_t::UpdateMemorySpaceSize(
            unsigned __int64 uiBytes
            ) 
        {
            Lock();
            m_global.UpdateMemorySpaceSize(uiBytes);
            m_pagelocked.UpdateMemorySpaceSize(uiBytes);
            Unlock();            
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return the percentage of this memory space that is allocated. </summary>
        ///
        /// <remarks>   crossbac, 9/10/2013. </remarks>
        ///
        /// <returns>   The allocated percent. </returns>
        ///-------------------------------------------------------------------------------------------------

        UINT 
        GlobalDeviceMemoryState_t::GetAllocatedPercent(
            void
            )
        {
            Lock();
            double dSize = (double)m_global.m_uiMemorySpaceSize;
            double dAllocated = (double)m_global.m_uiCurrentlyAllocatedBytes;
            double dAllocFraction = dSize == 0.0?0.0:dAllocated/dSize;
            UINT uiPercent = (UINT) (dAllocFraction*100.0);
            Unlock();            
            return uiPercent;
        }

};


