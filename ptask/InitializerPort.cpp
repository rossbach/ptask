//--------------------------------------------------------------------------------------
// File: InitializerPort.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "InitializerPort.h"
#include "AsyncContext.h"
#include "task.h"
#include "signalprofiler.h"
#include <vector>
#include <map>
#include "assert.h"
using namespace std;

namespace PTask {

    /*! Constructor
     *  A Initializer port is a port that is 
     *  always full, and when pulled by a downstream 
     *  ptask, allocates a datablock initialized to 
     *  a value specified by its datablock template. 
     *  Initializer ports are a short cut way to produce
     *  ptask parameters that must be initialized before
     *  being used, but whose initialization does not
     *  vary in a way that requires user-code to be involved.
     *  A canonical example:
     *  void sum(inout int val) { 
     *    for(...) { val += ...; }
     *  }
     *  GPU code cannot initialize val to 0 in the same
     *  kernel that performs the summation because there
     *  is no way to ensure that whatever thread initializes
     *  it runs first. Hence the caller of sum must always
     *  set val to 0 beforehand. In ptask without Initializer ports
     *  this structure results in a graph with an extra port and channel
     *  into which user code must always push a datablock with a 
     *  0 in it. The Initializer port provides a way for user code
     *  to build a graph that always knows what value should be
     *  used at a given port. 
     */
    InitializerPort::InitializerPort(
        VOID
        ) 
    {
        m_uiId = NULL;
        m_pTemplate = NULL;
        m_ePortType = INITIALIZER_PORT;
        m_pInOutConsumer = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    InitializerPort::~InitializerPort() {
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Return true if a datablock is available. Since a Initializer port allocates in response
    ///     to a pull, we can simply return TRUE here.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if occupied, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    InitializerPort::IsOccupied(
        VOID
        ) 
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Return a datablock initialized to the initial value specified in this port's template.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *		
    InitializerPort::Pull(
        VOID
        ) 
    {
        Datablock * pResult = NULL;
        Lock();
        if(InputPort::Peek() != NULL) {
            pResult = InputPort::Pull();
        } else {
            Task * pTask = GetTask();
            Accelerator * pDependentAccelerator = NULL;
            Accelerator * pBoundAccelerator     = NULL;
            Accelerator * pDispatchAccelerator = pTask->GetDispatchAccelerator();
            if(HasDependentAcceleratorBinding()) {
                pDependentAccelerator = pTask->GetAssignedDependentAccelerator(this);
            }
            pBoundAccelerator = pDependentAccelerator ? pDependentAccelerator : pDispatchAccelerator;
            AsyncContext * pAsyncContext = pBoundAccelerator ? 
                pTask->GetOperationAsyncContext(pBoundAccelerator, ASYNCCTXT_XFERHTOD) : NULL;
            pResult = AllocateBlock(pAsyncContext, true);
        }
        Unlock();
        ctlpegress(this, pResult);
        return pResult;
    }
   
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Peek always returns null since we allocate only in response to a pull. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The VOID to peek. </param>
    ///
    /// <returns>   null if it fails, else the current top-of-stack object. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *		
    InitializerPort::Peek(
        VOID
        ) 
    {
        Datablock * pBlock = NULL;
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>
    ///     Push a datablock into a port. This should never get called on a Initializer port, since
    ///     it generates its own contents.
    /// </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="p">    [in,out] If non-null, the Datablock* to push. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL		
    InitializerPort::Push(
        Datablock* p
        ) 
    {
        UNREFERENCED_PARAMETER(p);
        assert(FALSE);
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a new InitializerPort. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pTemplate">            [in,out] If non-null, the template. </param>
    /// <param name="uiId">                 The identifier. </param>
    /// <param name="lpszVariableBinding">  [in,out] If non-null, the variable binding. </param>
    /// <param name="nParmIdx">             Zero-based index of the n parm. </param>
    /// <param name="nInOutRouteIdx">       Zero-based index of the n in out route. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port*
    InitializerPort::Create(
        DatablockTemplate * pTemplate,
        UINT uiId, 
        char * lpszVariableBinding,
        int nParmIdx,
        int nInOutRouteIdx
        )
    {
        InitializerPort * pPort = new InitializerPort();
        if(SUCCEEDED(pPort->Initialize(pTemplate, uiId, lpszVariableBinding, nParmIdx, nInOutRouteIdx)))
            return pPort;
        delete pPort;
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate a datablock. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] (optional)  If non-null, the accelerator and task
    ///                                 context in which this block will be used. </param>
    /// <param name="bPooled">          true if the block should be pooled. Currently ignored, as we
    ///                                 only maintain block pools for output ports. 
    ///                                 ------------------------------------------- 
    ///                                 FIXME: TODO: CJR:
    ///                                 implement block pooling on other port types! There is no reason
    ///                                 why we can't have block pooling for all port types! </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *  
    InitializerPort::AllocateBlock(
        __in AsyncContext * pAsyncContext,
        __in BOOL bPooled
        ) 
    {
        UNREFERENCED_PARAMETER(bPooled);
        Datablock * pBlock = NULL;
        Lock();
        DatablockTemplate * pTemplate = GetTemplate();
        BUFFERACCESSFLAGS eFlags = pTemplate->IsByteAddressable() ? PT_ACCESS_BYTE_ADDRESSABLE : 0;
        eFlags |= (PT_ACCESS_ACCELERATOR_READ | PT_ACCESS_ACCELERATOR_WRITE);
        if(PTask::Runtime::GetDebugMode()) {
            // in debug mode we materialize a host view
            // on internal channel so that a debugger can
            // have visibility for intermediate data in a graph. 
            // Consequently, we need host read permissions.
            eFlags |= PT_ACCESS_HOST_READ;
        }
        pBlock = Datablock::CreateInitialValueBlock(pAsyncContext, pTemplate, eFlags);
        Unlock();
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check type-specific semantics. Return true if all the structures are initialized
    ///             for this port in a way that is consistent with a well-formed graph. Called by
    ///             CheckSemantics()
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="pos">      [in,out] output string stream. </param>
    /// <param name="pGraph">   [in,out] non-null, the graph. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    InitializerPort::CheckTypeSpecificSemantics(
        std::ostream * pos,
        PTask::Graph * pGraph
        )
    {
        UNREFERENCED_PARAMETER(pGraph);
        BOOL bResult = TRUE;
        std::ostream& os = *pos;
        if(m_vChannels.size() != 0 || m_vControlChannels.size() != 0) {
            // it's meaningful to have a channel on an initializer
            // port, but it had better be predicated. 
            vector<Channel*>::iterator vi;
            for(vi=m_vChannels.begin(); vi!=m_vChannels.end(); vi++) {
                Channel * pChannel = *vi;
                if(pChannel->GetPredicationType(CE_SRC) == CGATEFN_NONE) {
                    bResult = FALSE;
                    os << this 
                        << " bound to an unpredicated input channel: initializer ports" 
                        << " can use input channels, but they must be predicated."
                        << std::endl;
                }
            }

        }
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has block pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if block pool, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    InitializerPort::HasBlockPool(
        VOID
        )
    {
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Force block pooling for a port that has an up-stream allocator. In general, when
    ///             we have an upstream allocator (meta) port, the runtime will not create a block
    ///             pool for the corresponding output port. This turns out to put device-side
    ///             allocation on the critical path in some cases, so we provide a way to override
    ///             that behavior and allow a port to create a pool based on some size hints. When
    ///             there is a block available with sufficient space in the pool, the meta port can
    ///             avoid the allocation and draw from the pool.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 9/25/2012. </remarks>
    ///
    /// <param name="nPoolSize">                Size of the block pool. </param>
    /// <param name="nStride">                  The stride. </param>
    /// <param name="nDataBytes">               The data in bytes. </param>
    /// <param name="nMetaBytes">               The meta in bytes. </param>
    /// <param name="nTemplateBytes">           The template in bytes. </param>
    /// <param name="bPageLockHostViews">       (optional) the page lock host views. </param>
    /// <param name="bEagerDeviceMaterialize">  (optional) the eager device materialize. </param>
    ///-------------------------------------------------------------------------------------------------

    void            
    InitializerPort::ForceBlockPoolHint(
        __in UINT nPoolSize,
        __in UINT nStride,
        __in UINT nDataBytes,
        __in UINT nMetaBytes,
        __in UINT nTemplateBytes,
        __in BOOL bPageLockHostViews,
        __in BOOL bEagerDeviceMaterialize
        )
    {
        UNREFERENCED_PARAMETER(nPoolSize);
        UNREFERENCED_PARAMETER(nStride);
        UNREFERENCED_PARAMETER(nDataBytes);
        UNREFERENCED_PARAMETER(nMetaBytes);
        UNREFERENCED_PARAMETER(nTemplateBytes);
        UNREFERENCED_PARAMETER(bPageLockHostViews);
        UNREFERENCED_PARAMETER(bEagerDeviceMaterialize);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate block pool. Attempt to preallocate blocks on this port.
    ///             
    ///             Allocation of data-blocks and platform-specific buffers can be a signficant
    ///             latency expense at dispatch time. We can actually preallocate output datablocks
    ///             and create device- side buffers at graph construction time. For each node in the
    ///             graph, allocate data blocks on any output ports, and create device-specific
    ///             buffers for all accelerators capable of executing the node.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/15/2012. </remarks>
    ///
    /// <param name="pAccelerators">    [in] If non-null, the accelerators on which views of blocks
    ///                                 allocated in the pool may be required. </param>
    /// <param name="uiPoolSize">       [in] (optional) Size of the pool. If zero/defaulted,
    /// 								Runtime::GetICBlockPoolSize() will be used to determine the
    /// 								size of the pool. </param>
    /// 								
    /// <returns>   True if it succeeds, false if it fails. If a port type doesn't actually implement
    ///             pooling, return false as well.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    InitializerPort::AllocateBlockPool(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )
    {
        UNREFERENCED_PARAMETER(pAccelerators);
        UNREFERENCED_PARAMETER(uiPoolSize);
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destroys the block pool. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    InitializerPort::DestroyBlockPool(
        VOID
        )
    {

    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate block pool. Attempt to preallocate blocks on this port.
    ///             Asynchronous version. Only allocates device-space buffers
    ///             in the first pass. Second pass queues all the copies.
    ///             This function handles only the first pass.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/15/2012. </remarks>
    ///
    /// <param name="pAccelerators">    [in] If non-null, the accelerators on which views of blocks
    ///                                 allocated in the pool may be required. </param>
    /// <param name="uiPoolSize">       [in] (optional) Size of the pool. If zero/defaulted,
    /// 								Runtime::GetICBlockPoolSize() will be used to determine the
    /// 								size of the pool. </param>
    /// 								
    /// <returns>   True if it succeeds, false if it fails. If a port type doesn't actually implement
    ///             pooling, return false as well.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL            
    InitializerPort::AllocateBlockPoolAsync(
        __in std::vector<Accelerator*>* pAccelerators,
        __in unsigned int               uiPoolSize
        )
    {
        UNREFERENCED_PARAMETER(pAccelerators);
        UNREFERENCED_PARAMETER(uiPoolSize);
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocate block pool. Attempt to preallocate blocks on this port.
    ///             Asynchronous version. Only allocates device-space buffers
    ///             in the first pass. Second pass queues all the copies.
    ///             This function handles the second pass.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/15/2012. </remarks>
    ///
    /// <param name="pAccelerators">    [in] If non-null, the accelerators on which views of blocks
    ///                                 allocated in the pool may be required. </param>
    /// <param name="uiPoolSize">       [in] (optional) Size of the pool. If zero/defaulted,
    /// 								Runtime::GetICBlockPoolSize() will be used to determine the
    /// 								size of the pool. </param>
    /// 								
    /// <returns>   True if it succeeds, false if it fails. If a port type doesn't actually implement
    ///             pooling, return false as well.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    InitializerPort::FinalizeBlockPoolAsync(
        VOID
        )
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds to the pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    InitializerPort::AddNewBlock(
        Datablock * pBlock
        )
    {
        UNREFERENCED_PARAMETER(pBlock);
        assert(FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds to the pool. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    InitializerPort::ReturnToPool(
        Datablock * pBlock
        )
    {
        UNREFERENCED_PARAMETER(pBlock);
        assert(FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   gets the pool size. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    UINT
    InitializerPort::GetPoolSize(
        VOID
        )
    {
        assert(FALSE);
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets request page locked. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="bPageLocked">  true to lock, false to unlock the page. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    InitializerPort::SetRequestsPageLocked(
        BOOL bPageLocked
        )
    {
        assert(FALSE);
        UNREFERENCED_PARAMETER(bPageLocked);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets request page locked. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    InitializerPort::GetRequestsPageLocked(
        VOID
        )
    {
        return FALSE;
    }

};
