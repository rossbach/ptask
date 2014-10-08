//--------------------------------------------------------------------------------------
// File: Accelerator.cpp
// maintaner: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

#include "MemorySpace.h"
#include "accelerator.h"
#include "AsyncContext.h"
#include "datablocktemplate.h"
#include "PTaskRuntime.h"
#include "Scheduler.h"
#include "DeviceMemoryStatus.h"
#include "task.h"
#include <vector>
#include <algorithm>
#include <assert.h>
#include "cuaccelerator.h"
using namespace std;

namespace PTask {

    /// <summary> accelerator uid counter </summary>
    unsigned int		                         Accelerator::m_uiUIDCounter = 0;
    _declspec(thread) LPFNTLSCTXTDEINITIALIZER   Accelerator::s_vTLSDeinitializers[MAXTLSCTXTCALLBACKS];
    _declspec(thread) int                        Accelerator::s_nTLSDeinitializers = 0;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Generate a new unique identifier to be 
    /// 			used as an accelerator id. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name=""> none </param>
    ///
    /// <returns>   UINT, unique </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Accelerator::AssignUniqueAcceleratorIdentifier(
        VOID
        )
    {
        return ::InterlockedIncrement(&m_uiUIDCounter);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Default constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Accelerator::Accelerator(
        VOID
        ) : Lockable(NULL)
    {
        m_nCoreCount = 0;
        m_nRuntimeVersion = 0;
        m_nMemorySize = 0;
        m_nClockRate = 0;
        m_bSupportsConcurrentKernels = 0;
        m_nPlatformIndex = 0;
        m_bInitialized = FALSE;
        m_pDevice = NULL;
        m_lpszDeviceName = NULL;
        m_bAsyncContextsReleased = FALSE;
        m_uiAcceleratorId = AssignUniqueAcceleratorIdentifier();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    Accelerator::~Accelerator(
        VOID
        )
    {
        ReleaseAsyncContexts();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 3/20/2014. </remarks>
    ///
    /// <param name="parameter1">   The first parameter. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    Accelerator::ReleaseAsyncContexts(
        VOID
        )
    {
        if(!m_bAsyncContextsReleased) {
            std::map<ASYNCCONTEXTTYPE, AsyncContext*>::iterator mi;
            std::map<Task*, AsyncContext*>::iterator ti;
            for(ti=m_vTaskAsyncContextMap.begin(); ti!=m_vTaskAsyncContextMap.end(); ti++) {
                AsyncContext * pAsyncContext = ti->second;
                pAsyncContext->Release();
            } 
            m_vTaskAsyncContexts.clear();
            m_vTaskAsyncContextMap.clear();
            for(mi=m_vDeviceAsyncContexts.begin(); mi!=m_vDeviceAsyncContexts.end(); mi++) {
                AsyncContext * pAsyncContext = mi->second;
                pAsyncContext->Release();
            }
            m_vDeviceAsyncContexts.clear();
            m_bAsyncContextsReleased = TRUE;
        }
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
    Accelerator::DeviceToDeviceTransfer(
        __inout PBuffer *       pDstBuffer,
        __in    PBuffer *       pSrcBuffer,
        __in    AsyncContext *  pAsyncContext
        ) 
    {
        UNREFERENCED_PARAMETER(pDstBuffer);
        UNREFERENCED_PARAMETER(pSrcBuffer);
        UNREFERENCED_PARAMETER(pAsyncContext);
        assert(FALSE && "not implemented!");
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Override of less-than operator for accelerator objects. Makes it possible to sort
    ///             lists of accelerators in order of increasing "strength" using std::sort.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/8/2013. </remarks>
    ///
    /// <param name="rhs">  The right hand operand. </param>
    ///
    /// <returns>   true if operator&lt;(const accelerator&amp;rhs)const, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool Accelerator::operator <(const Accelerator &rhs) const {
        // we need a notion of more vs. less powerful when we are 
        // either choosing among available accelerators or
        // limiting the concurrency of the runtime, and 
        // preferring to use the more powerful of those available. 
        // The sort order (somewhat arbitrarily) is:
        // 1. highest runtime version support
        // 2. support for concurrent kernels
        // 3. highest core count
        // 4. fastest core clock
        // 5. biggest memory
        // 6. enumeration order (ensuring we will usually choose the same physical device across multiple back ends)
        if(m_nRuntimeVersion != rhs.m_nRuntimeVersion) return m_nRuntimeVersion < rhs.m_nRuntimeVersion;
        if(m_bSupportsConcurrentKernels != rhs.m_bSupportsConcurrentKernels) return rhs.m_bSupportsConcurrentKernels != 0;
        if(m_nCoreCount != rhs.m_nCoreCount) return m_nCoreCount < rhs.m_nCoreCount;
        if(m_nClockRate != rhs.m_nClockRate) return m_nClockRate < rhs.m_nClockRate;
        if(m_nMemorySize != rhs.m_nMemorySize ) return m_nMemorySize < rhs.m_nMemorySize;
        return m_nPlatformIndex < rhs.m_nPlatformIndex;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Select the best N accelerators from the 
    /// 			given list of candidates.  </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="nConcurrencyLimit">    The concurrency limit. </param>
    /// <param name="candidates">           [in,out] [in,out] If non-null, the candidates. </param>
    /// <param name="selected">             [in,out] [in,out] If non-null, the selected. </param>
    /// <param name="rejected">             [in,out] [in,out] If non-null, the rejected. </param>
    ///-------------------------------------------------------------------------------------------------

    void         
    Accelerator::SelectBestAccelerators(
        UINT nConcurrencyLimit,
        std::vector<Accelerator*> &candidates,
        std::vector<Accelerator*> &selected,
        std::vector<Accelerator*> &rejected
        )
    {
        // Given a list of accelerator objects, 
        // select the best N and reject the remainder,
        // where "better" is defined by the sort order
        // in the less-than operator defined above. 
        std::sort(candidates.begin(), candidates.end());
        BOOL bConcurrencyLimited = (nConcurrencyLimit != 0);
        BOOL bPreferFront = TRUE;
        UINT nInserted = 0;
        
        if(bPreferFront) {
            vector<Accelerator*>::iterator i = candidates.begin();
            vector<Accelerator*>::iterator rlast = candidates.end();
            while(i < rlast) {
                Accelerator * pCandidate = *i;
                if(pCandidate->IsHost()) {
                    selected.push_back(pCandidate);
                } else {
                    if(!bConcurrencyLimited || nInserted < nConcurrencyLimit) {
                        selected.push_back(pCandidate);
                        nInserted++;
                    } else {
                        rejected.push_back(pCandidate);
                    }
                }
                i++;
            }
        } else {
            vector<Accelerator*>::reverse_iterator i = candidates.rbegin();
            vector<Accelerator*>::reverse_iterator rlast = candidates.rend();
            while(i < rlast) {
                Accelerator * pCandidate = *i;
                if(pCandidate->IsHost()) {
                    selected.push_back(pCandidate);
                } else {
                    if(!bConcurrencyLimited || nInserted < nConcurrencyLimit) {
                        selected.push_back(pCandidate);
                        nInserted++;
                    } else {
                        rejected.push_back(pCandidate);
                    }
                }
                i++;
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a PBuffer: platform specific buffer encapsulating buffer management on
    ///             the accelerator side. Each subclass of Accelerator will also implement a subclass
    ///             of PBuffer to provide a uniform interface on buffer management. The buffer is
    ///             considered to be either:
    ///             
    ///             1) A 3-dimensional array of fixed-stride elements, 2) An undimensioned extent of
    ///             byte-addressable memory.
    ///             
    ///             Buffer geometry is inferred from the datablock which is the logical buffer the
    ///             resulting PBuffer will back on this accelerator.
    ///             
    ///             The proxy accelerator is provided to handle interoperation corner between
    ///             different types of accelerator objects, where deferring buffer allocation to the
    ///             proxy accelerator can enable better performance the parent is backed on both
    ///             accelerator types. See documentation for NewPlatformSpecificBuffer for more
    ///             details.
    ///             
    ///             This method is implemented by the PBuffer superclass, and actual allocation of
    ///             the PBuffer subtype is deferred to NewPlatformSpecificBuffer.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <param name="pAsyncContext">            [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="pLogicalBufferParent">     [in] If non-null, the datablock that is the logical
    ///                                         buffer using this 'physical' buffer to back a
    ///                                         particular channel on this accelerator. </param>
    /// <param name="nDatablockChannelIndex">   Access flags determining what views to create. </param>
    /// <param name="eBufferAccessFlags">       Zero-based index of the channel being backed. Must be:
    ///                                         * DBDATA_IDX = 0, OR
    ///                                         * DBMETADATA_IDX = 1, OR
    ///                                         * DBTEMPLATE_IDX = 2. </param>
    /// <param name="pProxyAccelerator">        [in,out] If non-null, the proxy allocator. </param>
    /// <param name="pExtent">                  [in,out] Number of elements in Z dimension. </param>
    /// <param name="strDebugName">             (optional) [in] If non-null, the a name for the
    ///                                         object (helps with debugging). </param>
    /// <param name="bByteAddressable">         (optional) Specifies whether the created buffer
    ///                                         should be geometry-less ("raw") or not. This flag is
    ///                                         not required for all platforms: for example, DirectX
    ///                                         buffers must created with special descriptors if the
    ///                                         HLSL code accessing them is going to use byte-
    ///                                         addressing. This concept is absent from CUDA and
    ///                                         OpenCL. </param>
    ///
    /// <returns>   null if it fails, else a new PBUffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    PBuffer*	
    Accelerator::CreateBuffer( 
        __in AsyncContext * pAsyncContext,
        __in Datablock * pLogicalBufferParent,
        __in UINT nDatablockChannelIndex, 
        __in BUFFERACCESSFLAGS eBufferAccessFlags, 
        __in Accelerator * pProxyAccelerator,
        __in HOSTMEMORYEXTENT * pExtent,
        __in char * strDebugName, 
        __in bool bByteAddressable
        )
    {
        // fix up the byte addressability flags: make sure they actually agree!
        bByteAddressable |= ((eBufferAccessFlags & PT_ACCESS_BYTE_ADDRESSABLE) != 0);

        // in some cases we need a device framework to allocate memory for another accelerator. The
        // most common (and perhaps only, at the moment) example of this derives from the fact that
        // CUDA requires page-locked host memory for asynchronous D <==> H transfers to work. We can
        // ensure that host buffers are page locked either by allocating the host memory through the
        // CUDA API, or using the CUDA API to page-lock it after the fact. Regardless, we need the CUDA
        // API, which we access in PTask through an accelerator object. If the caller has provided
        // an explicit (non-null) "pProxyAccelerator" object, it should be used for this purpose. If
        // it is NULL, we may be able to infer a proxy accelerator by looking at the async context. 
        
        Accelerator * pAsyncCtxtAccelerator = 
            (pAsyncContext == NULL) ? 
                NULL :                                  // there is no async context, so no inference possible
                pAsyncContext->GetDeviceContext();      // get the accelerator from the async context
        Accelerator * pAllocatingAccelerator = 
            (pProxyAccelerator == NULL) ?
                pAsyncCtxtAccelerator :                 // use the accelerator from the async context
                pProxyAccelerator;                      // caller provided a proxy: we must use it. 

        // allocate a new platform specific object, still requires initialization
        PBuffer * pBuffer = NewPlatformSpecificBuffer(pLogicalBufferParent,
                                                      nDatablockChannelIndex,
                                                      eBufferAccessFlags,
                                                      pAllocatingAccelerator);

        PTRESULT pt = PTASK_ERR;
        if(pBuffer != NULL) {

            // initialize the platform specific buffer, causing device buffers to
            // be allocated, and when pInitialBufferContents is non-NULL, populated.
            pt = pBuffer->Initialize(pAsyncContext,
                                     PBUFFER_DEFAULT_SIZE,
                                     pExtent,
                                     strDebugName, 
                                     bByteAddressable);            
        }

        if((pBuffer == NULL) || PTFAILED(pt)) {
            assert(pBuffer != NULL && PTSUCCESS(pt));
            PTask::Runtime::MandatoryInform("Device allocation failed in %s\n", __FUNCTION__);
            delete pBuffer;
            pBuffer = NULL;
        }

        return pBuffer;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a PBuffer: platform specific buffer encapsulating buffer management on
    ///             the accelerator side. Each subclass of Accelerator will also implement a subclass
    ///             of PBuffer to provide a uniform interface on buffer management. The buffer is
    ///             considered to be either:
    ///             
    ///             1) A 3-dimensional array of fixed-stride elements, 2) An undimensioned extent of
    ///             byte-addressable memory.
    ///             
    ///             Buffer geometry is inferred from the datablock which is the logical buffer the
    ///             resulting PBuffer will back on this accelerator.
    ///             
    ///             The proxy accelerator is provided to handle interoperation corner between
    ///             different types of accelerator objects, where deferring buffer allocation to the
    ///             proxy accelerator can enable better performance the parent is backed on both
    ///             accelerator types. See documentation for NewPlatformSpecificBuffer for more
    ///             details.
    ///             
    ///             This method is implemented by the PBuffer superclass, and actual allocation of
    ///             the PBuffer subtype is deferred to NewPlatformSpecificBuffer.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <param name="pAsyncContext">            [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="pLogicalBufferParent">     [in] If non-null, the datablock that is the logical
    ///                                         buffer using this 'physical' buffer to back a
    ///                                         particular channel on this accelerator. </param>
    /// <param name="nDatablockChannelIndex">   Access flags determining what views to create. </param>
    /// <param name="eBufferAccessFlags">       Zero-based index of the channel being backed. Must be:
    ///                                         * DBDATA_IDX = 0, OR
    ///                                         * DBMETADATA_IDX = 1, OR
    ///                                         * DBTEMPLATE_IDX = 2. </param>
    /// <param name="pProxyAccelerator">        [in,out] If non-null, the proxy allocator. </param>
    /// <param name="pExtent">                  [in,out] Number of elements in Z dimension. </param>
    /// <param name="strDebugName">             (optional) [in] If non-null, the a name for the
    ///                                         object (helps with debugging). </param>
    /// <param name="bByteAddressable">         (optional) Specifies whether the created buffer
    ///                                         should be geometry-less ("raw") or not. This flag is
    ///                                         not required for all platforms: for example, DirectX
    ///                                         buffers must created with special descriptors if the
    ///                                         HLSL code accessing them is going to use byte-
    ///                                         addressing. This concept is absent from CUDA and
    ///                                         OpenCL. </param>
    ///
    /// <returns>   null if it fails, else a new PBUffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    PBuffer*	
    Accelerator::CreatePagelockedBuffer( 
        __in AsyncContext * pAsyncContext,
        __in Datablock * pLogicalBufferParent,
        __in UINT nDatablockChannelIndex, 
        __in BUFFERACCESSFLAGS eBufferAccessFlags, 
        __in Accelerator * pProxyAccelerator,
        __in HOSTMEMORYEXTENT * pExtent,
        __in char * strDebugName, 
        __in bool bByteAddressable
        )
    {
        // fix up the byte addressability flags: make sure they actually agree!
        bByteAddressable |= ((eBufferAccessFlags & PT_ACCESS_BYTE_ADDRESSABLE) != 0);
        bool bAttemptPageLocked = true;

        // in some cases we need a device framework to allocate memory for another accelerator. The
        // most common (and perhaps only, at the moment) example of this derives from the fact that
        // CUDA requires page-locked host memory for asynchronous D <==> H transfers to work. We can
        // ensure that host buffers are page locked either by allocating the host memory through the
        // CUDA API, or using the CUDA API to page-lock it after the fact. Regardless, we need the CUDA
        // API, which we access in PTask through an accelerator object. If the caller has provided
        // an explicit (non-null) "pProxyAccelerator" object, it should be used for this purpose. If
        // it is NULL, we may be able to infer a proxy accelerator by looking at the async context. 
        
        Accelerator * pAsyncCtxtAccelerator = 
            (pAsyncContext == NULL) ? 
                NULL :                                  // there is no async context, so no inference possible
                pAsyncContext->GetDeviceContext();      // get the accelerator from the async context
        Accelerator * pAllocatingAccelerator = 
            (pProxyAccelerator == NULL) ?
                pAsyncCtxtAccelerator :                 // use the accelerator from the async context
                pProxyAccelerator;                      // caller provided a proxy: we must use it. 

        // allocate a new platform specific object, still requires initialization
        PBuffer * pBuffer = NewPlatformSpecificBuffer(pLogicalBufferParent,
                                                      nDatablockChannelIndex,
                                                      eBufferAccessFlags,
                                                      pAllocatingAccelerator);

        // initialize the platform specific buffer, causing device buffers to
        // be allocated, and when pInitialBufferContents is non-NULL, populated.        
        PTRESULT pt = pBuffer->Initialize(pAsyncContext,
                                          PBUFFER_DEFAULT_SIZE,
                                          pExtent,
                                          strDebugName, 
                                          bByteAddressable,
                                          bAttemptPageLocked);
            
        if(PTFAILED(pt)) {
            delete pBuffer;
            pBuffer = NULL;
        }
        return pBuffer;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if the accelerator is initialized.  </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> none </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Accelerator::Initialized(
        VOID
        ) 
    {
        return m_bInitialized;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is enabled. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <returns>   true if enabled, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    Accelerator::IsEnabled(
        VOID
        )
    {
        return Scheduler::IsEnabled(this);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is a host accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <returns>   true if enabled, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    Accelerator::IsHost(
        VOID
        )
    {
        return m_class == ACCELERATOR_CLASS_HOST;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enables/disables the accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="bEnable">  true to enable, false to disable. </param>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Accelerator::Enable(
        BOOL bEnable
        )
    {
        return Scheduler::SetAcceleratorEnabled(this, bEnable);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the actual class of the accelerator object 
    /// 			implementing this interface. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> none </param>
    ///
    /// <returns>   The class. </returns>
    ///-------------------------------------------------------------------------------------------------

    ACCELERATOR_CLASS 
    Accelerator::GetClass(
        VOID
        ) 
    {
        return m_class;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the device name. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the device name. </returns>
    ///-------------------------------------------------------------------------------------------------

    char *      
    Accelerator::GetDeviceName(
        VOID
        )
    {
        return m_lpszDeviceName;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the core count. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <returns>   The core count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT        
    Accelerator::GetCoreCount(
        VOID
        ) 
    { 
        return m_nCoreCount; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the core clock rate. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <returns>   The core clock rate. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT        
    Accelerator::GetCoreClockRate(
        VOID
        ) 
    { 
        return m_nClockRate; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the platform specific runtime version. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <returns>   The platform specific runtime version. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT        
    Accelerator::GetPlatformSpecificRuntimeVersion(
        VOID
        ) 
    { 
        return m_nRuntimeVersion; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the global memory size. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <returns>   The global memory size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT        
    Accelerator::GetGlobalMemorySize(
        VOID
        )   
    { 
        return m_nMemorySize; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the supports concurrent kernels. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL        
    Accelerator::SupportsConcurrentKernels(
        VOID
        ) 
    { 
        return m_bSupportsConcurrentKernels; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a core count. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <param name="n">    The n. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    Accelerator::SetCoreCount(
        UINT n
        ) 
    { 
        m_nCoreCount = n; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a core clock rate. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <param name="n">    The n. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    Accelerator::SetCoreClockRate(
        UINT n
        ) 
    { 
        m_nClockRate = n; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a platform specific runtime version. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <param name="n">    The n. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    Accelerator::SetPlatformSpecificRuntimeVersion(
        UINT n
        ) 
    { 
        m_nRuntimeVersion = n; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a global memory size. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <param name="n">    The n. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    Accelerator::SetGlobalMemorySize(
        UINT n
        ) 
    { 
        m_nMemorySize = n; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the supports concurrent kernels. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <param name="b">    true to b. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    Accelerator::SetSupportsConcurrentKernels(
        BOOL b
        ) 
    { 
        m_bSupportsConcurrentKernels = b; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the physical device. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the physical device. </returns>
    ///-------------------------------------------------------------------------------------------------

    PhysicalDevice * 
    Accelerator::GetPhysicalDevice(
        VOID
        ) 
    { 
        return m_pDevice; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a physical device. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <param name="p">    [in,out] If non-null, the p. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    Accelerator::SetPhysicalDevice(
        PhysicalDevice* p
        ) 
    { 
        m_pDevice = p; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the platform index. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <returns>   The platform index. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT        
    Accelerator::GetPlatformIndex(
        VOID
        ) 
    { 
        return m_nPlatformIndex; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a platform index. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <param name="n">    The platform index. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    Accelerator::SetPlatformIndex(
        UINT n
        ) 
    { 
        m_nPlatformIndex = n; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the accelerator id. </summary>
    ///
    /// <remarks>   Crossbac, 12/17/2011. </remarks>
    ///
    /// <returns>   The platform index. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT        
    Accelerator::GetAcceleratorId(
        VOID
        ) 
    { 
        return m_uiAcceleratorId; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the memory space identifier for this accelerator. For now, we do not attempt
    ///             to coalesce memory spaces based on whether they share the same physical device.
    ///             Memory space identifiers must be contiguous starting with zero, where zero is
    ///             reserved for host memory. No lock is required for this call because this is 
    ///             read-only data after the scheduler initializes memory spaces.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <returns>   The memory space identifier. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT        
    Accelerator::GetMemorySpaceId(
        VOID
        )
    {
        return m_uiMemorySpaceId;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the memory space identifier for this accelerator. For now, we do not attempt
    ///             to coalesce memory spaces based on whether they share the same physical device.
    ///             Memory space identifiers must be contiguous starting with zero, where zero is
    ///             reserved for host memory.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pMemorySpace">     The memory space. </param>
    /// <param name="uiMemorySpaceId">  Identifier for the memory space associated with this
    ///                                 accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void        
    Accelerator::SetMemorySpace(
        MemorySpace * pMemorySpace,
        UINT uiMemorySpaceId
        )
    {
        m_pMemorySpace = pMemorySpace;
        m_uiMemorySpaceId = uiMemorySpaceId;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the page locked allocation count. </summary>
    ///
    /// <remarks>   Crossbac, 9/24/2012. </remarks>
    ///
    /// <returns>   The page locked allocation count. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned __int64 
    Accelerator::GetPageLockedAllocationCount(
        VOID
        )
    {
        if(!PTask::Runtime::GetTrackDeviceMemory()) {
            PTask::Runtime::Warning("Device-side memory tracking not enabled--cannot report statistics (e.g. GetPageLockedAllocationCount)");
        }
        m_pMemorySpace->Lock();
        DEVICEMEMORYSTATE * pMemState = m_pMemorySpace->GetMemoryState();
        unsigned __int64 uiRes = pMemState->GetPageLockedMemoryState()->m_uiCurrentlyAllocatedBuffers;
        m_pMemorySpace->Unlock();
        return uiRes;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the page locked allocation bytes. </summary>
    ///
    /// <remarks>   Crossbac, 9/24/2012. </remarks>
    ///
    /// <returns>   The page locked allocation bytes. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned __int64 
    Accelerator::GetPageLockedAllocationBytes(
        VOID
        )
    {
        if(!PTask::Runtime::GetTrackDeviceMemory()) {
            PTask::Runtime::Warning("Device-side memory tracking not enabled--cannot report statistics (e.g. GetPageLockedAllocationBytes)");
        }
        m_pMemorySpace->Lock();
        DEVICEMEMORYSTATE * pMemState = m_pMemorySpace->GetMemoryState();
        unsigned __int64 uiRes = pMemState->GetPageLockedMemoryState()->m_uiCurrentlyAllocatedBytes;
        m_pMemorySpace->Unlock();
        return uiRes;        
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the allocation count. </summary>
    ///
    /// <remarks>   Crossbac, 9/24/2012. </remarks>
    ///
    /// <returns>   The allocation count. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned __int64 
    Accelerator::GetTotalAllocationBuffers(
        VOID
        )
    {
        if(!PTask::Runtime::GetTrackDeviceMemory()) {
            PTask::Runtime::Warning("Device-side memory tracking not enabled--cannot report statistics (e.g. GetTotalAllocationBuffers)");
        }
        m_pMemorySpace->Lock();
        DEVICEMEMORYSTATE * pMemState = m_pMemorySpace->GetMemoryState();
        unsigned __int64 uiRes = pMemState->GetGlobalMemoryState()->m_uiCurrentlyAllocatedBuffers;
        m_pMemorySpace->Unlock();
        return uiRes;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the total allocation bytes. </summary>
    ///
    /// <remarks>   Crossbac, 9/24/2012. </remarks>
    ///
    /// <returns>   The total allocation bytes. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned __int64 
    Accelerator::GetTotalAllocationBytes(
        VOID
        )
    {
        if(!PTask::Runtime::GetTrackDeviceMemory()) {
            PTask::Runtime::Warning("Device-side memory tracking not enabled--cannot report statistics (e.g. GetTotalAllocationBytes)");
        }
        m_pMemorySpace->Lock();
        DEVICEMEMORYSTATE * pMemState = m_pMemorySpace->GetMemoryState();
        unsigned __int64 uiRes = pMemState->GetGlobalMemoryState()->m_uiCurrentlyAllocatedBytes;
        m_pMemorySpace->Unlock();
        return uiRes;        
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record a memory allocation. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///
    /// <param name="pMemoryExtent">        [in,out] If non-null, extent of the memory. </param>
    /// <param name="bPinnedAllocation">    true to pinned allocation. </param>
    /// <param name="uiBytes">              The bytes. </param>
    ///-------------------------------------------------------------------------------------------------

    void                    
    Accelerator::RecordAllocation(
        __in void * pMemoryExtent, 
        __in BOOL bPinnedAllocation, 
        __in size_t uiBytes
        )
    {
        if(PTask::Runtime::GetTrackDeviceMemory()) {
            m_pMemorySpace->RecordAllocation(pMemoryExtent, uiBytes, bPinnedAllocation);
        }
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
    Accelerator::RecordDeallocation(
        __in void * pMemoryExtent
        )
    {
        if(PTask::Runtime::GetTrackDeviceMemory()) {
            m_pMemorySpace->RecordDeallocation(pMemoryExtent);    
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resets the allocation statistics. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Accelerator::ResetAllocationStatistics(
        VOID
        )
    {
        if(PTask::Runtime::GetTrackDeviceMemory()) {
            m_pMemorySpace->Reset();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps the allocation statistics. </summary>
    ///
    /// <remarks>   Crossbac, 3/15/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Accelerator::DumpAllocationStatistics(
        std::ostream &ios
        )
    {
        if(PTask::Runtime::GetTrackDeviceMemory()) {
            m_pMemorySpace->Report(ios);
        } else {
            PTask::Runtime::Warning("Device-side memory tracking not enabled--cannot report statistics");
        }
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
    Accelerator::SupportsExplicitAsyncOperations(
        VOID
        )
    {
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synchronizes all default contexts for accelerators supporting async
    ///             operations. </summary>
    ///
    /// <remarks>   crossbac, 6/26/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Accelerator::SynchronizeDefaultContexts(
        VOID
        )
    {
        for(UINT uiMemSpace=HOST_MEMORY_SPACE_ID+1; 
            uiMemSpace < MemorySpace::GetNumberOfMemorySpaces(); 
            uiMemSpace++) {

            Accelerator * pAccelerator = MemorySpace::GetAcceleratorFromMemorySpaceId(uiMemSpace);
            if(pAccelerator->SupportsExplicitAsyncOperations()) {
                pAccelerator->Lock();
                AsyncContext * pAsyncContext = pAccelerator->GetAsyncContext(ASYNCCTXT_DEFAULT);
                pAsyncContext->Lock();
                pAsyncContext->SynchronizeContext();
                pAsyncContext->Unlock();
                pAccelerator->Unlock();
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the default asynchronous context. </summary>
    ///
    /// <remarks>   Crossbac, 7/13/2012. </remarks>
    ///
    /// <returns>   null if it fails, else the default asynchronous context. </returns>
    ///-------------------------------------------------------------------------------------------------

    AsyncContext * 
    Accelerator::GetAsyncContext( 
        ASYNCCONTEXTTYPE eAsyncContextType
        )
    {
        AsyncContext * pAsyncContext = NULL;
        assert(eAsyncContextType != ASYNCCTXT_TASK);
        std::map<ASYNCCONTEXTTYPE, AsyncContext*>::iterator mi;
        if(!SupportsExplicitAsyncOperations()) {
            mi=m_vDeviceAsyncContexts.find(ASYNCCTXT_DEFAULT);
            assert(mi!=m_vDeviceAsyncContexts.end());
            if(mi!=m_vDeviceAsyncContexts.end())
                pAsyncContext = mi->second;
        } else {
            mi=m_vDeviceAsyncContexts.find(eAsyncContextType);
            assert(mi!=m_vDeviceAsyncContexts.end());            
            if(mi!=m_vDeviceAsyncContexts.end())
                pAsyncContext = mi->second;
        }
        return pAsyncContext;
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
    Accelerator::CreateAsyncContext(
        __in Task * pTask,
        __in ASYNCCONTEXTTYPE eAsyncContextType
        )
    {
        AsyncContext * pAsyncContext = NULL;

        Lock();

        if(!SupportsExplicitAsyncOperations()) {

            // if the underlying device doesn't support explicit async operation management,
            // there is no sense in proliferation of async objects with tasks or memory operations
            // associated with the accelerator. In this case, map all requests to the default
            // async context, which, if it is not yet there, should be created. 
            
            std::map<ASYNCCONTEXTTYPE, AsyncContext*>::iterator di;
            di = m_vDeviceAsyncContexts.find(ASYNCCTXT_DEFAULT);
            if(di == m_vDeviceAsyncContexts.end()) {
                pAsyncContext = PlatformSpecificCreateAsyncContext(pTask, ASYNCCTXT_DEFAULT);
                pAsyncContext->Initialize();
                m_vDeviceAsyncContexts[ASYNCCTXT_DEFAULT] = pAsyncContext;
                pAsyncContext->AddRef();
            } else {
                pAsyncContext = di->second;
            }

        } else {

            if(pTask != NULL) {

                // if we are creating a task execution context, keep track of it in the
                // per-task map, which we will use to make sure individual contexts are aware
                // of sync operations that occur at device level. Note that only one type of
                // async context is valid here: the task-exec context. We want all tasks to share
                // the default contexts for memory operations and transfers. 
            
                assert(eAsyncContextType == ASYNCCTXT_TASK);
                std::map<Task*, AsyncContext*>::iterator mi;
                mi=m_vTaskAsyncContextMap.find(pTask);
                assert(mi == m_vTaskAsyncContextMap.end());
                if(mi == m_vTaskAsyncContextMap.end()) {

                    pAsyncContext = PlatformSpecificCreateAsyncContext(pTask, eAsyncContextType);
                    BOOL bInitSuccess = (pAsyncContext != NULL) && pAsyncContext->Initialize();
                    if(bInitSuccess) {

                        // successfully created and initialized the task async context.
                        // addref it, and stash it in our async context maps so we can
                        // propagate device-level sync operations when necessary. 
                        
                        m_vTaskAsyncContextMap[pTask] = pAsyncContext;
                        m_vTaskAsyncContexts[pAsyncContext] = pTask;
                        pAsyncContext->AddRef();

                    } else {

                        if(pAsyncContext) delete pAsyncContext;
                        PTask::Runtime::HandleError("%s: %s create task async context failed!\n",
                                                    __FUNCTION__,
                                                    pTask->GetTaskName());
                        pAsyncContext = NULL;
                    }

                } else {

                    // creating another task-async context for task/accelerator combination? 
                    PTask::Runtime::MandatoryInform("%s:%s redundant AsyncContext creation!\n", 
                                                    __FUNCTION__,
                                                    pTask->GetTaskName());
                    pAsyncContext = mi->second;
                }

                assert(pAsyncContext != NULL);

            } else {

                // we are creating an async context for something other than task invocations.
                // this means that this is an object that will be shared across all tasks
                // for operations like memory transfers etc. 
            
                assert(eAsyncContextType != ASYNCCTXT_TASK);
                std::map<ASYNCCONTEXTTYPE, AsyncContext*>::iterator ti;
                ti=m_vDeviceAsyncContexts.find(eAsyncContextType);
                assert(ti== m_vDeviceAsyncContexts.end());
                if(ti==m_vDeviceAsyncContexts.end()) {

                    // not present (as expected). Create/init, addref, add to maps.
                    pAsyncContext = PlatformSpecificCreateAsyncContext(pTask, eAsyncContextType);
                    BOOL bInitSuccess = pAsyncContext != NULL && pAsyncContext->Initialize();
                    if(bInitSuccess) {
                        m_vDeviceAsyncContexts[eAsyncContextType] = pAsyncContext;
                        pAsyncContext->AddRef();
                    } else {
                        if(pAsyncContext) delete pAsyncContext;
                        PTask::Runtime::HandleError("%s: create async context (type=%d) failed!\n",
                                                    __FUNCTION__,
                                                    eAsyncContextType);
                        pAsyncContext = NULL;
                    }
                } else {

                    PTask::Runtime::MandatoryInform("%s:redundant default AsyncContext(%d) creation!\n", 
                                                    __FUNCTION__,
                                                    eAsyncContextType);
                    pAsyncContext = ti->second;
                }
            }
        }

        Unlock();
        return pAsyncContext;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   destroys the asynchronous context for the task. Since accelerators keep a reference to
    ///             all async contexts, along with some maps, the demise of a task must be visible to the
    ///             accelerator objects for which it has created task async contexts.
    ///             </summary>
    ///
    /// <param name="pTask">                [in] non-null, the CUDA-capable acclerator to which the
    ///                                     stream is bound. </param>
    /// <param name="eAsyncContextType">    Type of the asynchronous context. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Accelerator::ReleaseTaskAsyncContext(
        __in Task * pTask,
        __in AsyncContext * pAsyncContext
        )
    {
        if(!SupportsExplicitAsyncOperations())
            return;
        Lock();
        assert(pTask != NULL);
        assert(pAsyncContext != NULL);
        assert(pAsyncContext->SupportsExplicitAsyncOperations());
        std::map<AsyncContext*, Task*>::iterator mi;
        std::map<Task*, AsyncContext*>::iterator ti;
        mi = m_vTaskAsyncContexts.find(pAsyncContext);
        ti = m_vTaskAsyncContextMap.find(pTask);
        assert(mi != m_vTaskAsyncContexts.end());
        assert(ti != m_vTaskAsyncContextMap.end());
        m_vTaskAsyncContexts.erase(mi);
        m_vTaskAsyncContextMap.erase(ti);
        pAsyncContext->Release();
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes any thread local context data structures associated with the
    ///             accelerator. Most back-ends do nothing. Subclasses that actually do some TLS-
    ///             based context management (CUDA)
    ///             override.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/20/2014. </remarks>
    ///
    /// <param name="eRole">            the role. </param>
    /// <param name="bMakeDefault">     The make default. </param>
    /// <param name="bPooledThread">    The pooled thread. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Accelerator::InitializeTLSContext(
        __in PTTHREADROLE eRole,
        __in BOOL bMakeDefault,
        __in BOOL bPooledThread
        )
    {
        // do nothing. subclasses override
        UNREFERENCED_PARAMETER(eRole);
        UNREFERENCED_PARAMETER(bMakeDefault);
        UNREFERENCED_PARAMETER(bPooledThread);
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
    Accelerator::DeinitializeTLSContextManagement(
        VOID
        ) 
    {
        // any subclasses that use TLS to manage device contexts are required to install de-init
        // handlers here, so we needn't search for live accelerator objects to figure out how to de-
        // initialize the TLS structures managed by subclasses. Since de-init is *per-thread* rather
        // than per-accelerator, it is safe to uninstall the callbacks once we have called them. 
        
        while(s_nTLSDeinitializers > 0) {
            s_nTLSDeinitializers--;
            LPFNTLSCTXTDEINITIALIZER lpfn = s_vTLSDeinitializers[s_nTLSDeinitializers];
            s_vTLSDeinitializers[s_nTLSDeinitializers] = NULL;
            (*lpfn)();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Determines if we require thread local context initialization. 
    ///             Subclasses must override. </summary>
    ///
    /// <remarks>   Crossbac, 3/20/2014. </remarks>
    ///
    /// <returns>   true if instances of the given Accelerator class (sub-class)
    ///             use TLS data structures to manage device contexts (and there
    ///             fore must have init/deinit functions called to set them up). 
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Accelerator::UsesTLSContextManagement(
        VOID
        )
    {
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Provides an entry point for new threads that may access accelerators to
    ///             initialize any TLS data structures. In general this is heuristic--if, for example
    ///             there is only one accelerator object and one graph runner thread, we prefer to
    ///             just set the device context on that thread and never touch it again. The logic
    ///             here reflects an attempt to capitalize on a few such optimization opportunities.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 3/20/2014. </remarks>
    ///
    /// <param name="eRole">                The role. </param>
    /// <param name="uiThreadRoleIndex">    Zero-based index of the thread role. </param>
    /// <param name="uiThreadRoleCount">    Number of thread roles. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Accelerator::InitializeTLSContextManagement(
        __in PTTHREADROLE eRole,
        __in UINT uiThreadRoleIndex,
        __in UINT uiThreadRoleCount,
        __in BOOL bPooledThread
        )
    {
        vector<Accelerator*> vTLSAccs;
        vector<Accelerator*>::iterator si;
        for(UINT i=HOST_MEMORY_SPACE_ID+1;
            i<MemorySpace::GetNumberOfMemorySpaces();
            i++) {
            Accelerator * pAccelerator = MemorySpace::GetAcceleratorFromMemorySpaceId(i);
            if(Scheduler::IsEnabled(pAccelerator) &&
               pAccelerator->UsesTLSContextManagement())
                vTLSAccs.push_back(pAccelerator);
        }

        int nTLSAccs = static_cast<int>(vTLSAccs.size());
        if(nTLSAccs == 0)
            return; 

        // we have some accelerator objects whose sub-class uses TLS data structures for device context
        // management. Since init for these data structures is *per-thread*, we need only call this
        // function once; the logic below chooses a suitable default from amongst potentially many
        // instances and tells the initializer whether to attempt to install it as a default after
        // setting up the TLS area. 
        
        int nDenom = min(nTLSAccs, static_cast<int>(uiThreadRoleCount));
        int nTIdx = uiThreadRoleIndex % nDenom;
        Accelerator * pDefaultCandidate = vTLSAccs[nTIdx];
        BOOL bIsDefaultCandidate = ShouldHaveDefaultContext(eRole);
        pDefaultCandidate->InitializeTLSContext(eRole, bIsDefaultCandidate, bPooledThread);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Determine if the thread role suggests we might want a default
    ///             device context to remain current at all times on the thread with
    ///             the given role. This is a global policy decision, so we implement
    ///             it in a static method of the super class. </summary>
    ///
    /// <remarks>   crossbac, 6/20/2014. </remarks>
    ///
    /// <param name="eRole">    The role. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Accelerator::ShouldHaveDefaultContext(
        PTTHREADROLE eRole
        )
    {
        switch(eRole) {
        case PTTR_UNKNOWN:     return FALSE;
        case PTTR_GRAPHRUNNER: return TRUE;
        case PTTR_SCHEDULER:   return FALSE;
        case PTTR_GC:          return TRUE;
        case PTTR_APPLICATION: return !PTask::Runtime::GetApplicationThreadsManagePrimaryContext();
        default: return FALSE;
        }
    }


};