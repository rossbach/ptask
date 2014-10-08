///-------------------------------------------------------------------------------------------------
// file:	MemorySpace.cpp
//
// summary:	Implements the memory space class
///-------------------------------------------------------------------------------------------------

#include "MemorySpace.h"
#include "ptaskutils.h"
#include "assert.h"
#include "accelerator.h"
#include "HostAccelerator.h"
#include "DeviceMemoryStatus.h"

namespace PTask {

    /// <summary> The memory space identifier counter. We start at 0 because
    /// 		  HostAccelerator will claim memory space id 0 and does not use
    /// 		  this variable to assign an id. Therefore, any accelerator
    /// 		  object claiming a memory space id using this variable must
    /// 		  start at id 1, which will be the first value returned by
    /// 		  InterlockedIncrement. </summary>
    unsigned int		MemorySpace::m_uiMemorySpaceIdCounter = 0;   

    /// <summary> Static map from ids to memory space objects. </summary>
    MemorySpace*        MemorySpace::m_vMemorySpaceMap[MAX_MEMORY_SPACES];

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the accelerator from memory space identifier. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="id">   The identifier. </param>
    ///
    /// <returns>   null if it fails, else the accelerator from memory space identifier. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    MemorySpace::GetAcceleratorFromMemorySpaceId(
        UINT id
        )
    {
        if(id >= GetNumberOfMemorySpaces())
            return NULL;
        MemorySpace * pSpace = m_vMemorySpaceMap[id];
        if(pSpace == NULL) 
            return NULL;
        return pSpace->GetAnyAccelerator();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Registers the memory space identifier. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="nMemorySpaceId">   Identifier for the memory space. </param>
    /// <param name="pAccelerator">     [in,out] If non-null, the accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    MemorySpace::RegisterMemorySpace(
        MemorySpace * pSpace, 
        Accelerator * pAccelerator
        )
    {
        UINT id = pSpace->m_nMemorySpaceId;
        if(id >= PTask::MAX_MEMORY_SPACES)
            return;
        assert(!m_vMemorySpaceMap[id]);
        if(pAccelerator != NULL) 
            pSpace->AddAccelerator(pAccelerator);
        m_vMemorySpaceMap[id] = pSpace;
    }    

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Registers the memory space identifier. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="nMemorySpaceId">   Identifier for the memory space. </param>
    /// <param name="pAccelerator">     [in,out] If non-null, the accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    MemorySpace::RegisterMemorySpaceId(
        UINT id, 
        Accelerator * pAccelerator
        )
    {
        if(id >= PTask::MAX_MEMORY_SPACES)
            return;
        MemorySpace * pSpace = m_vMemorySpaceMap[id];
        if(pSpace == NULL) {
            assert(false);
            pSpace->AddAccelerator(pAccelerator);
        } else {
            if(pSpace != NULL)
                pSpace->AddAccelerator(pAccelerator);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the memory spaces. </summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    MemorySpace::InitializeMemorySpaces(
        VOID
        )
    {
        memset(MemorySpace::m_vMemorySpaceMap, 0, MAX_MEMORY_SPACES*sizeof(MemorySpace*));
        std::string strHostMemspace("HostMem");
        MemorySpace * pHostMemorySpace = new MemorySpace(strHostMemspace, HOST_MEMORY_SPACE_ID);
        pHostMemorySpace->SetStaticAllocator(HostAccelerator::AllocateMemoryExtent,
                                             HostAccelerator::DeallocateMemoryExtent);
        RegisterMemorySpace(pHostMemorySpace, NULL);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unregisters the memory spaces at tear-down time. </summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    MemorySpace::UnregisterMemorySpaces(
        VOID
        )
    {
        for(UINT i=0; i<GetNumberOfMemorySpaces(); i++) {
            if(MemorySpace::m_vMemorySpaceMap[i] != NULL) {
                delete MemorySpace::m_vMemorySpaceMap[i];
                MemorySpace::m_vMemorySpaceMap[i] = NULL;
            }
        }
        m_uiMemorySpaceIdCounter = 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the accelerator from memory space identifier. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="id">   The identifier. </param>
    ///
    /// <returns>   null if it fails, else the accelerator from memory space identifier. </returns>
    ///-------------------------------------------------------------------------------------------------

    MemorySpace * 
    MemorySpace::GetMemorySpaceFromId(
        UINT id
        )
    {
        if(id >= MAX_MEMORY_SPACES)
            return NULL;
        return m_vMemorySpaceMap[id];
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the accelerator from memory space identifier. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="id">   The identifier. </param>
    ///
    /// <returns>   null if it fails, else the accelerator from memory space identifier. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    MemorySpace::HasStaticAllocator(
        UINT id
        )
    {
        MemorySpace * pSpace = GetMemorySpaceFromId(id);
        if(pSpace != NULL) {
            return pSpace->HasStaticAllocator();
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate an extent in this memory space. Fails if 
    /// 			no static allocator is present. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="uiMemorySpace">        The identifier. </param>
    /// <param name="ulBytesToAllocate">    The ul bytes to allocate. </param>
    /// <param name="ulFlags">              The ul flags. </param>
    ///
    /// <returns>   null if it fails, else the accelerator from memory space identifier. </returns>
    ///-------------------------------------------------------------------------------------------------

    void * 
    MemorySpace::AllocateMemoryExtent(
        UINT uiMemorySpace, 
        ULONG ulBytesToAllocate, 
        ULONG ulFlags
        )
    {
        MemorySpace * pSpace = GetMemorySpaceFromId(uiMemorySpace);
        if(pSpace == NULL || !pSpace->HasStaticAllocator()) 
            return NULL;
        return pSpace->AllocateMemoryExtent(ulBytesToAllocate, ulFlags);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   deallocate an extent in this memory space. Fails if 
    /// 			no static allocator is present. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="uiMemorySpace">        The identifier. </param>
    /// <param name="ulBytesToAllocate">    The ul bytes to allocate. </param>
    /// <param name="ulFlags">              The ul flags. </param>
    ///
    /// <returns>   null if it fails, else the accelerator from memory space identifier. </returns>
    ///-------------------------------------------------------------------------------------------------

    void 
    MemorySpace::DeallocateMemoryExtent(
        UINT id,
        void * pMemoryExtent
        )
    {
        MemorySpace * pSpace = GetMemorySpaceFromId(id);
        if(pSpace == NULL || !pSpace->HasStaticAllocator()) 
            return;
        pSpace->DeallocateMemoryExtent(pMemoryExtent);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Assign a unique memory space identifier. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    MemorySpace::AssignUniqueMemorySpaceIdentifier(
        VOID
        )
    {
        return ::InterlockedIncrement(&m_uiMemorySpaceIdCounter);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the number of memory spaces active in the system. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <returns>   The number of memory spaces. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    MemorySpace::GetNumberOfMemorySpaces(
        VOID
        )
    {
        return m_uiMemorySpaceIdCounter + 1;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Default constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="lpszProtectedObjectName">  [in] If non-null, name of the protected object. </param>
    ///-------------------------------------------------------------------------------------------------

    MemorySpace::MemorySpace(
        std::string& szDeviceName,
        UINT nMemorySpaceId
        ) : Lockable(NULL)
    {
        m_nMemorySpaceId = nMemorySpaceId;
        m_lpfnStaticAllocator = NULL;
        m_lpfnStaticDeallocator = NULL;
        m_strDeviceName = szDeviceName;
        m_pMemoryState = new DEVICEMEMORYSTATE(m_strDeviceName);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    MemorySpace::~MemorySpace(
        VOID
        )
    {
        delete m_pMemoryState;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this memory space has a static buffer allocator function. </summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///
    /// <returns>   true if static allocator, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    MemorySpace::HasStaticAllocator(
        VOID
        )
    {
        return (m_lpfnStaticAllocator != NULL);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate a memory extent in this memory space of the given
    /// 			size. If this memory space does not have a static allocator,
    /// 			return NULL. </summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///
    /// <param name="ulNumberOfBytes">  The ul number of in bytes. </param>
    /// <param name="ulFlags">          The ul flags. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    void *          
    MemorySpace::AllocateMemoryExtent(
        ULONG ulNumberOfBytes, 
        ULONG ulFlags
        )
    {
        assert(ulNumberOfBytes != 0);
        assert(m_lpfnStaticAllocator != NULL);
        if(m_lpfnStaticAllocator == NULL) 
            return NULL;
        return (*m_lpfnStaticAllocator)(ulNumberOfBytes, ulFlags);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deallocate a memory extent in this memory space of the given
    /// 			size. If this memory space does not have a static allocator,
    /// 			return NULL. </summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///
    /// <param name="ulNumberOfBytes">  The ul number of in bytes. </param>
    /// <param name="ulFlags">          The ul flags. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    void       
    MemorySpace::DeallocateMemoryExtent(
        void * pMemory
        )
    {
        assert(pMemory != 0);
        assert(m_lpfnStaticDeallocator != NULL);
        if(m_lpfnStaticDeallocator == NULL) 
            return;
        (*m_lpfnStaticDeallocator)(pMemory);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a pointer to any accelerator mapped to this space. Most spaces
    /// 			have just one, so this simplifies the process of getting an object
    /// 			that can provide allocation services if no static allocator is present. 
    /// 		    </summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///
    /// <returns>   null if it fails, else any accelerator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator *   
    MemorySpace::GetAnyAccelerator(
        VOID
        )
    {
        if(m_pAccelerators.size()) 
            return *(m_pAccelerators.begin());
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the number of accelerators mapped to this space. </summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///
    /// <returns>   The number of accelerators. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT            
    MemorySpace::GetNumberOfAccelerators(
        VOID
        )
    {
        return (UINT) m_pAccelerators.size();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets all accelerators in this space, by putting them in the user-provided buffer.
    ///             At most nMaxAccelerators will be provided.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///
    /// <param name="ppAccelerators">   [in,out] If non-null, the accelerators. </param>
    /// <param name="nMaxAccelerators"> The maximum accelerators. </param>
    ///
    /// <returns>   The number of accelerators in the result buffer, which may be different from
    ///             nMaxAccelerators!
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT            
    MemorySpace::GetAccelerators(
        Accelerator ** ppAccelerators, 
        UINT nMaxAccelerators
        )
    {
        if(ppAccelerators == NULL || nMaxAccelerators == 0)
            return 0;

        UINT nReturned = 0;
        std::set<Accelerator*>::iterator ai;
        for(ai=m_pAccelerators.begin(); 
            ai!=m_pAccelerators.end() && nReturned < nMaxAccelerators; 
            ai++) {
            ppAccelerators[nReturned++] = *ai;
        }
        return nReturned;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a static allocator function for this memory space. </summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///
    /// <param name="lpfnStaticAllocatorFunction">  The lpfn static allocator function. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    MemorySpace::SetStaticAllocator(
        LPFNSTATICALLOCATOR lpfnStaticAllocatorFunction,
        LPFNSTATICDEALLOCATOR lpfnStaticDeallocatorFunction
        )
    {
        assert(lpfnStaticAllocatorFunction != NULL);
        assert(lpfnStaticDeallocatorFunction != NULL);
        m_lpfnStaticAllocator = lpfnStaticAllocatorFunction;
        m_lpfnStaticDeallocator = lpfnStaticDeallocatorFunction;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a deferred allocation entry for the proxy accelerator, indicating that
    ///             allocations for this space should be deferred to accelerators for that space,
    ///             when the resulting buffers will be used to commnunicate between those spaces.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///
    /// <param name="pProxyAllocatorAccelerator">   [in,out] If non-null, the proxy allocator
    ///                                             accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    MemorySpace::AddDeferredAllocationEntry(
        Accelerator* pProxyAllocatorAccelerator
        )
    {
        assert(pProxyAllocatorAccelerator != NULL);
        assert(pProxyAllocatorAccelerator->GetMemorySpaceId() != m_nMemorySpaceId);
        m_pDeferredAllocatorSpaces.insert(pProxyAllocatorAccelerator->GetMemorySpaceId());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an accelerator to this memory space. </summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    MemorySpace::AddAccelerator(
        Accelerator * pAccelerator
        )
    {
        assert(pAccelerator != NULL);
        if(pAccelerator != NULL) 
            m_pAccelerators.insert(pAccelerator);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resets the memory space. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    MemorySpace::Reset(
        VOID
        ) 
    {
        Lock();
        m_pMemoryState->Reset();
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
    MemorySpace::RecordAllocation(
        __in void * pMemoryExtent,
        __in size_t uiBytes,
        __in BOOL bPinned
        )
    {
        Lock();
        m_pMemoryState->RecordAllocation(pMemoryExtent, uiBytes, bPinned);
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
    MemorySpace::RecordDeallocation(
        __in void * pMemoryExtent
        )
    {
        Lock();
        m_pMemoryState->RecordDeallocation(pMemoryExtent);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps the allocation statistics. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    MemorySpace::Report(
        std::ostream &ios
        ) 
    {
        Lock();
        m_pMemoryState->Report(ios);
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets memory state. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the memory state. </returns>
    ///-------------------------------------------------------------------------------------------------

    DEVICEMEMORYSTATE* 
    MemorySpace::GetMemoryState(
        VOID
        )
    {
        assert(LockIsHeld());
        return m_pMemoryState;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Updates the space size bytes described by uiBytes. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="uiBytes">  The bytes. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    MemorySpace::UpdateSpaceSizeBytes(
        unsigned __int64 uiBytes
        )
    {
        m_pMemoryState->UpdateMemorySpaceSize(uiBytes);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the percentage of this space already allocated. </summary>
    ///
    /// <remarks>   crossbac, 9/10/2013. </remarks>
    ///
    /// <returns>   The allocated percent. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT            
    MemorySpace::__GetAllocatedPercent(
        void
        )
    {
        return m_pMemoryState->GetAllocatedPercent();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the percentage of this space already allocated. </summary>
    ///
    /// <remarks>   crossbac, 9/10/2013. </remarks>
    ///
    /// <returns>   The allocated percent. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT            
    MemorySpace::GetAllocatedPercent(
        UINT uiMemorySpaceId
        )
    {
        MemorySpace * pSpace = GetMemorySpaceFromId(uiMemorySpaceId);
        if(pSpace != NULL) {
            return pSpace->__GetAllocatedPercent();
        }
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets allocation percentages. </summary>
    ///
    /// <remarks>   crossbac, 9/10/2013. </remarks>
    ///
    /// <param name="vDeviceMemories">  [in,out] The device memories. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    MemorySpace::GetAllocationPercentages(
        __inout std::map<UINT, UINT>& vDeviceMemories
        )
    {
        for(UINT ui=HOST_MEMORY_SPACE_ID+1; ui<GetNumberOfMemorySpaces(); ui++) {
            vDeviceMemories[ui] = GetAllocatedPercent(ui);
        }
    }
};


