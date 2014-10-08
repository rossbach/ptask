///-------------------------------------------------------------------------------------------------
// file:	RefCountProfiler.cpp
//
// summary:	Implements the reference count profiler class
///-------------------------------------------------------------------------------------------------

#include "PTaskRuntime.h"
#include "RefCountProfiler.h"
#include "ReferenceCounted.h"
#include <assert.h>

namespace PTask {
    
    LONG ReferenceCountedProfiler::m_nRCAllocations = 0;
    LONG ReferenceCountedProfiler::m_nRCDeletions = 0;
    LONG ReferenceCountedProfiler::m_nRCProfilerInit = 0;
    LONG ReferenceCountedProfiler::m_nRCProfilerEnable = 0;
    LONG ReferenceCountedProfiler::m_nRCProfilerIDCount = 0;
    std::set<PTask::ReferenceCounted*> ReferenceCountedProfiler::m_vAllAllocations;
    CRITICAL_SECTION ReferenceCountedProfiler::m_csRCProfiler;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initialize the Refcount object profiler. </summary>
    ///
    /// <remarks>   crossbac, 7/9/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    ReferenceCountedProfiler::Initialize(
        BOOL bEnable
        )
    {
        BOOL bSuccess = FALSE;        
#ifdef PROFILE_REFCOUNT_OBJECTS
        if(PTask::Runtime::GetRCProfilingEnabled()) {
            if(!m_nRCProfilerInit) {
                InitializeCriticalSection(&m_csRCProfiler);
                m_nRCProfilerEnable = bEnable;
                m_nRCProfilerInit = 1;
            } else {
                EnterCriticalSection(&m_csRCProfiler);
                m_nRCProfilerEnable = bEnable;
                LeaveCriticalSection(&m_csRCProfiler);
            }
            bSuccess = m_nRCProfilerInit;
        }
#else
        UNREFERENCED_PARAMETER(bEnable);
#endif
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deinitialize the Refcount object profiler. </summary>
    ///
    /// <remarks>   crossbac, 7/9/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    ReferenceCountedProfiler::Deinitialize(
        VOID
        )
    {
#ifdef PROFILE_REFCOUNT_OBJECTS
        if(PTask::Runtime::GetRCProfilingEnabled() && m_nRCProfilerInit) {
            DeleteCriticalSection(&m_csRCProfiler);
            m_nRCProfilerEnable = FALSE;
            m_nRCProfilerInit = 0;
        }
#endif
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps the profiler leaks. </summary>
    ///
    /// <remarks>   crossbac, 7/9/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    ReferenceCountedProfiler::Report(
        std::ostream& ss
        )
    {
#ifdef PROFILE_REFCOUNT_OBJECTS
        if(!m_nRCProfilerInit) return;
        std::set<PTask::ReferenceCounted*>::iterator si;
        EnterCriticalSection(&m_csRCProfiler);
        if(m_nRCProfilerEnable) {
            long lDelta = abs(m_nRCAllocations - m_nRCDeletions);
            if(lDelta != 0) {
                ss << "leaks detected!" << std::endl;
                ss << lDelta << " refcount objects leaked!" << std::endl;
            }
            ss << m_nRCAllocations << " refcount objects allocated, "
               << m_nRCDeletions << " deleted." 
               << std::endl;
            for(si=m_vAllAllocations.begin(); si!=m_vAllAllocations.end(); si++) {
                PTask::ReferenceCounted * pBlock = *si;
                ss << pBlock << std::endl;
            }
        }
        LeaveCriticalSection(&m_csRCProfiler);   
#else
        UNREFERENCED_PARAMETER(ss);
#endif
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   profile allocation. </summary>
    ///
    /// <remarks>   crossbac, 7/9/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    ReferenceCountedProfiler::RecordAllocation(
        ReferenceCounted * pBlock
        )
    {
#ifdef PROFILE_REFCOUNT_OBJECTS
        assert(m_nRCProfilerInit || !PTask::Runtime::GetRCProfilingEnabled());
        if(m_nRCProfilerInit && PTask::Runtime::GetRCProfilingEnabled() && m_nRCProfilerEnable) {
            pBlock->m_uiUID = InterlockedIncrement(&m_nRCProfilerIDCount);
            if(m_nRCProfilerEnable) {
                EnterCriticalSection(&m_csRCProfiler);
                m_vAllAllocations.insert(pBlock);
                LeaveCriticalSection(&m_csRCProfiler);
                // printf("alloc RC#%d\n", pBlock->m_uiUID);
                InterlockedIncrement(&m_nRCAllocations);
            }
        }
#else
        UNREFERENCED_PARAMETER(pBlock);
#endif
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   profile deletion. </summary>
    ///
    /// <remarks>   crossbac, 7/9/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    ReferenceCountedProfiler::RecordDeletion(
        ReferenceCounted * pBlock
        )
    {
#ifdef PROFILE_REFCOUNT_OBJECTS
        assert(m_nRCProfilerInit ||!PTask::Runtime::GetRCProfilingEnabled());
        if(m_nRCProfilerInit && PTask::Runtime::GetRCProfilingEnabled() && m_nRCProfilerEnable) {
            EnterCriticalSection(&m_csRCProfiler);
            m_vAllAllocations.erase(pBlock);
            LeaveCriticalSection(&m_csRCProfiler);
            // printf("delete RC#%d\n", pBlock->m_uiUID);
            InterlockedIncrement(&m_nRCDeletions);
        }
#else
        UNREFERENCED_PARAMETER(pBlock);
#endif
    }

};
