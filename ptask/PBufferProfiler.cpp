///-------------------------------------------------------------------------------------------------
// file:	PBufferProfiler.cpp
//
// summary:	Implements the buffer profiler class
///-------------------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "PBufferProfiler.h"
#include "hrperft.h"
#include <iomanip>
#include <assert.h>

using namespace std;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pParentDatablock">         [in,out] If non-null, the parent datablock. </param>
    /// <param name="bufferAccessFlags">        The buffer access flags. </param>
    /// <param name="nChannelIndex">            Zero-based index of the datablock channel this
    ///                                         PBuffer is backing. </param>
    /// <param name="pAccelerator">             (optional) [in,out] If non-null, the accelerator. </param>
    /// <param name="pAllocatingAccelerator">   (optional) [in,out] If non-null, the allocating
    ///                                         accelerator. </param>
    /// <param name="uiUniqueIdentifier">       (optional) unique identifier. </param>
    ///-------------------------------------------------------------------------------------------------

    PBufferProfiler::PBufferProfiler(
        VOID
        )
    {
        m_bAllocProfilerInit = FALSE;
        m_nAllocations = 0;
        m_pAllocationTimer = NULL;
        m_pcsAllocProfiler = NULL;
        Initialize();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    PBufferProfiler::~PBufferProfiler(
        VOID
        )
    {
        Deinitialize();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initialises the allocation profiler. </summary>
    ///
    /// <remarks>   Crossbac, 9/25/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    PBufferProfiler::Initialize(
        VOID
        )
    {
        if(InterlockedCompareExchange(&m_bAllocProfilerInit, 1, 0)) {
            LPCRITICAL_SECTION pcsAllocProfiler = new CRITICAL_SECTION();
            InitializeCriticalSection(pcsAllocProfiler);
            m_nAllocations = 0;
            m_pAllocationTimer = new CHighResolutionTimer(gran_msec);
            m_pAllocationTimer->reset();
            m_pcsAllocProfiler = pcsAllocProfiler;
        } else {
            while(m_pcsAllocProfiler == NULL)
                Sleep(10);
        }
        assert(m_bAllocProfilerInit);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deinit allocation profiler. </summary>
    ///
    /// <remarks>   Crossbac, 9/25/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    PBufferProfiler::Deinitialize(
        VOID
        )
    {
        if(!m_bAllocProfilerInit)
            return;
        if(m_pcsAllocProfiler != NULL) {
            DeleteCriticalSection(m_pcsAllocProfiler);
            delete m_pcsAllocProfiler;
            m_pcsAllocProfiler = NULL;
        }
        m_bAllocProfilerInit = FALSE;
        delete m_pAllocationTimer;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds an allocation data. </summary>
    ///
    /// <remarks>   Crossbac, 9/25/2012. </remarks>
    ///
    /// <param name="uiAllocBytes"> The allocate in bytes. </param>
    /// <param name="uiAllocAccID"> Identifier for the allocate accumulate. </param>
    /// <param name="dLatency">     The latency. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    PBufferProfiler::Record(
        UINT uiAllocBytes, 
        UINT uiAllocAccID,
        double dLatency
        )
    {
        if(!Runtime::GetProfilePlatformBuffers()) return;        
        if(!m_bAllocProfilerInit) return;
        EnterCriticalSection(m_pcsAllocProfiler);
        m_vAllocationLatencies[m_nAllocations] = dLatency;
        m_vAllocationDevices[m_nAllocations] = uiAllocAccID;
        m_vAllocationSizes[m_nAllocations] = uiAllocBytes;
        m_nAllocations++;
        LeaveCriticalSection(m_pcsAllocProfiler);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps an allocation profiler data. </summary>
    ///
    /// <remarks>   Crossbac, 9/25/2012. </remarks>
    ///
    /// <param name="ios">  [in,out] The ios. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    PBufferProfiler::Report(
        std::ostream &ios
        )
    {
        if(!Runtime::GetProfilePlatformBuffers()) return;        
        if(!m_bAllocProfilerInit) return;
        EnterCriticalSection(m_pcsAllocProfiler);
        for(UINT i=0; i<m_nAllocations; i++) {
            ios << "A" << m_vAllocationDevices.find(i)->second 
                << "(" << m_vAllocationSizes.find(i)->second << " bytes)->"
                << m_vAllocationLatencies.find(i)->second << "ms"
                << std::endl;
        }
        LeaveCriticalSection(m_pcsAllocProfiler);
    }
    
};
