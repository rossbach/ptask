//--------------------------------------------------------------------------------------
// File: datablock.cpp
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include <stdio.h>
#include <crtdbg.h>
#include "ptaskutils.h"
#include "accelerator.h"
#include "datablock.h"
#include "datablocktemplate.h"
#include "PCUBuffer.h"
#include "OutputPort.h"
#include <assert.h>
#include "PTaskRuntime.h"
#include "ptgc.h"
#include "MemorySpace.h"
#include "Task.h"
#include "CoherenceProfiler.h"
#include "DatablockProfiler.h"
#include "instrumenter.h"
#include "GlobalPoolManager.h"
#include "Scheduler.h"
#include "nvtxmacros.h"
#include <iomanip>
using namespace std;
using namespace PTask::Runtime;

// does this entry have an up-to-date copy of the data?
#define valid(x) ((x) == BSTATE_SHARED || (x) == BSTATE_EXCLUSIVE)

#ifdef DEBUG
// check invariants on the coherence state machine
// by calling the member function of the same name
#define CHECK_INVARIANTS() CheckInvariants()
#define CHECK_INVARIANTS_LOCK() { Lock(); CheckInvariants(); Unlock(); }
#else
#define CHECK_INVARIANTS()
#define CHECK_INVARIANTS_LOCK()
#endif

#ifdef CHECK_CRITICAL_PATH_ALLOC
#define check_critical_path_clone()                                                \
    if(Runtime::GetCriticalPathAllocMode() && Scheduler::GetRunningGraphCount()) { \
	    PTask::Runtime::MandatoryInform("%s::%s() crit-path alloc: CLONE!\n",      \
                                        __FILE__,                                  \
                                        __FUNCTION__);                             \
    }

#define check_critical_path_alloc(v,t)                                             \
    if(Runtime::GetCriticalPathAllocMode() && Scheduler::GetRunningGraphCount()) { \
	    PTask::Runtime::MandatoryInform("%s::%s(%s,%s) crit-path alloc!\n",        \
                                        __FILE__,                                  \
                                        __FUNCTION__,                              \
                                        v, t);                                     \
    }    

#define break_critical_path_alloc(v)                                               \
    if(Runtime::GetCriticalPathAllocMode() && Scheduler::GetRunningGraphCount()) { \
	    PTask::Runtime::MandatoryInform("%s::%s(%s) crit-path alloc!\n",           \
                                        __FILE__,                                  \
                                        __FUNCTION__,                              \
                                        v);                                        \
        DebugBreak();                                                              \
    }
#else
#define check_critical_path_alloc(v,t) 
#define break_critical_path_alloc(v,t) 
#define check_critical_path_clone() 
#endif

#ifdef PROFILE_MIGRATION
#define ctpenable()                                         PTask::Runtime::GetCoherenceProfileMode()
#define ctprofile_record_vus(a,b,c)                         ((a)?(a)->RecordViewUpdateStart(b,c):NULL)
#define ctprofile_record_vue(a,b,c,d,e)                     {if(a){(a)->RecordViewUpdateEnd(b,c,d,e);}}
#define ctprofile_record_pba(a,b)                           {if(a){(a)->RecordPortBinding(b);}}
#define ctprofile_record_tba(a,b)                           {if(a){(a)->RecordTaskBinding(b);}}
#define ctprofile_record_binda(a,b,c)                       {if(a){(a)->RecordBinding(b,c,NULL);}}
#define ctprofile_record_bindioa(a,b,c,d)                   {if(a){(a)->RecordBinding(b,c,d);}}
#define ctprofile_record_pb(b)                              ctprofile_record_pba(m_pCoherenceProfiler, b)
#define ctprofile_record_tb(b)                              ctprofile_record_tba(m_pCoherenceProfiler, b)
#define ctprofile_record_bind(b,c)                          ctprofile_record_binda(m_pCoherenceProfiler, b, c)
#define ctprofile_record_bindio(b,c,d)                      ctprofile_record_bindioa(m_pCoherenceProfiler,b,c,d)
#define ctprofile_init_instancea(a)                         {if(ctpenable()){(a)=new CoherenceProfiler(this);}}
#define ctprofile_deinit_instancea(a)                       {if(a){ assert(ctpenable()); delete a; a = NULL;}}
#define ctprofile_init_instance()                           ctprofile_init_instancea(m_pCoherenceProfiler)
#define ctprofile_deinit_instance()                         ctprofile_deinit_instancea(m_pCoherenceProfiler)
#define ctprofile_merge_instancea(a)                        {if(a){(a)->MergeHistory();}}
#define ctprofile_merge_instance()                          ctprofile_merge_instancea(m_pCoherenceProfiler)
#define ctprofile_view_update_decl()                        COHERENCETRANSITION * pTx = NULL;
#define ctprofile_view_update_continue_cp(cp,spc,evt)       pTx = ctprofile_record_vus((cp), spc, (evt));
#define ctprofile_view_update_continue_cp_a(cp,a,evt)       pTx = ctprofile_record_vus((cp), (a)->GetMemorySpaceId(), (evt));
#define ctprofile_view_update_start_db(db,spc,evt)          COHERENCETRANSITION * pTx = ctprofile_record_vus((db)->m_pCoherenceProfiler, spc, (evt));
#define ctprofile_view_update_start_cp(cp,spc,evt)          COHERENCETRANSITION * pTx = ctprofile_record_vus((cp), spc, (evt));
#define ctprofile_view_update_end_cp(cp,spc,cstate,xf)      ctprofile_record_vue((cp), spc, (cstate), (xf), pTx);
#define ctprofile_view_update_end_db(db,spc,cstate,xf)      ctprofile_record_vue((db)->m_pCoherenceProfiler, spc, (cstate), (xf), pTx);
#define ctprofile_view_update_start_a_cp(cp,acc,evt)        COHERENCETRANSITION * pTx = ctprofile_record_vus((cp), (acc)->GetMemorySpaceId(), (evt));
#define ctprofile_view_update_start_a_db(db,acc,evt)        COHERENCETRANSITION * pTx = ctprofile_record_vus((db)->m_pCoherenceProfiler, (acc)->GetMemorySpaceId(), (evt));
#define ctprofile_view_update_end_a_db(db,acc,cstate,xf)    ctprofile_record_vue((db)->m_pCoherenceProfiler, (acc)->GetMemorySpaceId(), (cstate), (xf), pTx);
#define ctprofile_view_update_end_a_cp(cp,acc,cstate,xf)    ctprofile_record_vue((cp), (acc)->GetMemorySpaceId(), (cstate), (xf), pTx);
#define ctprofile_view_update_continue(spc,evt)             ctprofile_view_update_continue_cp(m_pCoherenceProfiler,spc,evt)
#define ctprofile_view_update_continue_a(acc,evt)           ctprofile_view_update_continue_cp_a(m_pCoherenceProfiler,acc,evt)
#define ctprofile_view_update_start(spc,evt)                ctprofile_view_update_start_cp(m_pCoherenceProfiler,spc,evt)
#define ctprofile_view_update_end(spc,cstate,xf)            ctprofile_view_update_end_cp(m_pCoherenceProfiler,spc,cstate,xf)
#define ctprofile_view_update_start_a(acc,evt)              ctprofile_view_update_start_a_db(m_pCoherenceProfiler,acc,evt)
#define ctprofile_view_update_end_a(acc,cstate,xf)          ctprofile_view_update_end_a_db(m_pCoherenceProfiler,acc,cstate,xf)
#define ctprofile_view_update_start_nested(spc,evt)         ctprofile_view_update_decl(); BOOL bNest = m_pCoherenceProfiler && m_pCoherenceProfiler->m_bCoherenceProfilerTransitionActive; if(!bNest) {ctprofile_view_update_continue(spc,evt);}
#define ctprofile_view_update_end_nested(spc,cstate,xf)     if(!bNest) { ctprofile_view_update_end(spc,cstate,xf); }
#define ctprofile_view_update_start_nested_a(acc,evt)       ctprofile_view_update_start_nested(acc->GetMemorySpaceId(),evt)
#define ctprofile_view_update_end_nested_a(acc,cstate,xf)   ctprofile_view_update_end_nested(acc->GetMemorySpaceId(),cstate,xf)
#define ctprofile_view_update_discard()                     if(pTx != NULL) delete pTx;
#define ctprofile_view_update_end_cond(b,spc,st,xf)         if(b) { ctprofile_view_update_end(spc,st,xf); } else { ctprofile_view_update_discard(); }
#define ctprofile_view_update_active()                      assert(!Runtime::g_bCoherenceProfile || m_pCoherenceProfiler->m_bCoherenceProfilerTransitionActive);
#define NOXFER FALSE
#define XFER TRUE
#else
#define ctpenable()                                         0
#define ctprofile_record_vus(a,b,c)                         
#define ctprofile_record_vue(a,b,c,d,e)                     
#define ctprofile_record_pba(a,b)                           
#define ctprofile_record_tba(a,b)                           
#define ctprofile_record_binda(a,b,c)                       
#define ctprofile_record_bindioa(a,b,c,d)                   
#define ctprofile_record_pb(b)                              
#define ctprofile_record_tb(b)                              
#define ctprofile_record_bind(b,c)                          
#define ctprofile_record_bindio(b,c,d)                      
#define ctprofile_init_instancea(a)                         
#define ctprofile_deinit_instancea(a)                       
#define ctprofile_init_instance()                           
#define ctprofile_deinit_instance()                         
#define ctprofile_merge_instancea(a)                        
#define ctprofile_merge_instance()                          
#define ctprofile_view_update_decl()                        
#define ctprofile_view_update_continue_cp(cp,spc,evt)       
#define ctprofile_view_update_continue_cp_a(cp,a,evt)       
#define ctprofile_view_update_start_db(db,spc,evt)          
#define ctprofile_view_update_start_cp(cp,spc,evt)          
#define ctprofile_view_update_end_cp(cp,spc,cstate,xf)      
#define ctprofile_view_update_end_db(db,spc,cstate,xf)      
#define ctprofile_view_update_start_a_cp(cp,acc,evt)        
#define ctprofile_view_update_start_a_db(db,acc,evt)        
#define ctprofile_view_update_end_a_db(db,acc,cstate,xf)    
#define ctprofile_view_update_end_a_cp(cp,acc,cstate,xf)    
#define ctprofile_view_update_continue(spc,evt)             
#define ctprofile_view_update_continue_a(acc,evt)           
#define ctprofile_view_update_start(spc,evt)                
#define ctprofile_view_update_end(spc,cstate,xf)            
#define ctprofile_view_update_start_a(acc,evt)              
#define ctprofile_view_update_end_a(acc,cstate,xf)          
#define ctprofile_view_update_start_nested(spc,evt)        
#define ctprofile_view_update_end_nested(spc,cstate,xf)     
#define ctprofile_view_update_start_nested_a(acc,evt)       
#define ctprofile_view_update_end_nested_a(acc,cstate,xf)   
#define ctprofile_view_update_discard()                     
#define ctprofile_view_update_end_cond(b,spc,st,xf)         
#define ctprofile_view_update_active()                      
#endif

#ifdef PROFILE_DATABLOCKS
#define dbprofile_enabled()                 PTask::Runtime::GetDBProfilingEnabled()
#define dbprofile_alloca(a,t)               {if(dbprofile_enabled()){(a) = new DatablockProfiler(t);}}
#define dbprofile_deletea(a)                {if(dbprofile_enabled()){delete (a);}}
#define dbprofile_alloc()                   dbprofile_alloca(m_pDatablockProfiler,this)
#define dbprofile_delete()                  dbprofile_deletea(m_pDatablockProfiler)
#define dbprofile_record_ownera(a)          {if(a&&dbprofile_enabled()){(a)->RecordPoolBinding();}}
#define dbprofile_record_pba(a,p)           {if(a&&dbprofile_enabled()){(a)->RecordBinding(p);}}
#define dbprofile_record_tba(a,t)           {if(a&&dbprofile_enabled()){(a)->RecordBinding(t);}}
#define dbprofile_record_bindioa(a,p,t,io)  {if(a&&dbprofile_enabled()){(a)->RecordBinding(p,t,io);}}
#define dbprofile_record_owner()            dbprofile_record_ownera(m_pDatablockProfiler)
#define dbprofile_record_pb(p)              dbprofile_record_pba(m_pDatablockProfiler,p)
#define dbprofile_record_tb(t)              dbprofile_record_tba(m_pDatablockProfiler,t)
#define dbprofile_record_bindio(p,t,io)     dbprofile_record_bindioa(m_pDatablockProfiler,p,t,io)
#else
#define dbprofile_enabled()                 0
#define dbprofile_alloc()           
#define dbprofile_delete()          
#define dbprofile_record_owner()    
#define dbprofile_record_pb(p)              
#define dbprofile_record_tb(t)              
#define dbprofile_record_bindio(p,t,io)     
#endif


namespace PTask {

    /// <summary> Datablock ID counter </summary>
    static UINT g_uiDBIDctr = 0;

    /// <summary> initialize the channel iterator end sentinal. </summary>
    Datablock::ChannelIterator Datablock::m_iterLastChannel(NULL, NUM_DATABLOCK_CHANNELS);

    DatablockTemplate * Datablock::m_gSizeDescriptorTemplate = NULL;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Datablock has a great many constructors but most initialization is the same while
    ///             the specialization applies to just a handful of members. This method handles the
    ///             intersection of member initializations that are shared by all constuctors.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::DefaultInitialize(
        VOID
        )
    {
        m_uiDBID = ++g_uiDBIDctr;
        m_pApplicationContext = NULL;
        m_bBlockResized = FALSE;
        m_bMarshallable = TRUE;
        m_bDeleting = FALSE;
        m_bLogicallyEmpty = FALSE;
        m_pTemplate = NULL;
        memset(m_cbRequested, 0, sizeof(m_cbRequested));
        memset(m_cbFinalized, 0, sizeof(m_cbFinalized));
        memset(m_cbAllocated, 0, sizeof(m_cbAllocated));
        m_pDestinationPort = NULL;
        m_uiRecordCount = 0;
        m_bRecordStream = FALSE;
        m_pProducerTask = NULL;
        m_bPooledBlock = FALSE;
        m_pPoolOwner = NULL;
        m_bSealed = FALSE;
        m_eBufferAccessFlags = PT_ACCESS_HOST_WRITE | PT_ACCESS_ACCELERATOR_READ;
        m_bByteAddressable = (m_eBufferAccessFlags & PT_ACCESS_BYTE_ADDRESSABLE);
        m_bDeviceViewsMemsettable = FALSE;
        m_bForceRequestedSize = FALSE;
        m_bAttemptPinnedHostBuffers = FALSE;
        m_bPoolRequiresInitialValue = FALSE;
        m_bPoolInitialValueSet = FALSE;
        UINT nMapEntries = MemorySpace::GetNumberOfMemorySpaces();
        m_ppBufferMap = new BUFFER_MAP_ENTRY*[nMapEntries];
        for(UINT i=0; i<nMapEntries; i++) {
            m_ppBufferMap[i] = new BUFFER_MAP_ENTRY(i);
        }
        m_vControlSignalStack.push(DBCTLC_NONE);
        m_pCoherenceProfiler = NULL;
        m_pDatablockProfiler = NULL;
        dbprofile_alloc();
        ctprofile_init_instance();
        ptgc_new(this);        
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Datablock::~Datablock() {

        dbprofile_delete();
        m_bDeleting = TRUE;
        assert(!LockIsHeld());
        Lock();
        ctprofile_merge_instance();
        ctprofile_deinit_instance();
        UINT nChannelIndex;
        UINT nMemSpaceId;
        for(nMemSpaceId=HOST_MEMORY_SPACE_ID; nMemSpaceId<MemorySpace::GetNumberOfMemorySpaces(); nMemSpaceId++) {
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[nMemSpaceId];
            for(nChannelIndex=PTask::DBDATA_IDX;
                nChannelIndex<PTask::NUM_DATABLOCK_CHANNELS;
                nChannelIndex++) {
                if(pEntry->pBuffers[nChannelIndex] != NULL) {
                    PBuffer * pBuffer = pEntry->pBuffers[nChannelIndex];
                    // make sure we dont free the backing memory for operations
                    // we scheduled asynchronously and subseqently freed. 
                    // perform a synchronous wait here to ensure we don't 
                    // pull the rug out from under any in flight device-side work.
                    pBuffer->SynchronousWaitOutstandingOperations();
                    pEntry->pBuffers[nChannelIndex] = NULL;
                    delete pBuffer;
                }
            }
            delete pEntry;
        }
        delete [] m_ppBufferMap;

        // call application context callback if this datablock has a template and
        // the template has a callback associated with it.
        LPFNAPPLICATIONCONTEXTCALLBACK pCallback = 
            m_pTemplate ? m_pTemplate->GetApplicationContextCallback() : NULL;
        if (NULL != pCallback)
        {
            pCallback(
                APPLICATIONCONTEXTCALLBACKPOINT::CALLBACKPOINT_DESTROY,
                this, &this->m_pApplicationContext);
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 12/19/2011. Refactored crossbac 7/9/12 to handle differing init
    ///             data/template dimensions according to caller's needs.
    ///             </remarks>
    ///
    /// <param name="pAsyncContext">        [in] If non-null, an async context, which will wind up
    ///                                     using this block. </param>
    /// <param name="pTemplate">            [in] If non-null, the datablock template. </param>
    /// <param name="eFlags">               [in] buffer access flags. </param>
    /// <param name="luiBlockControlCode">  [in] a block control code. </param>
    /// <param name="pExtent">              [in] If non-null, initial data. </param>
    /// <param name="bForceInitDataSize">   True if the number of bytes in the initial data buffer is
    ///                                     different from that in the template parameter *AND* the
    ///                                     caller would like the allocation to match the init data
    ///                                     size. In this case, the allocation size supercedes dims
    ///                                     in the template. If the this parameter is FALSE and the
    ///                                     dimensions differ, then the allocation is based template
    ///                                     dimensions and init data is striped (or truncated) to fit
    ///                                     the allocation.
    ///                                     . </param>
    ///-------------------------------------------------------------------------------------------------

    Datablock::Datablock(
        __in AsyncContext *         pAsyncContext,
        __in DatablockTemplate *    pTemplate, 
        __in BUFFERACCESSFLAGS      eFlags,
        __in CONTROLSIGNAL          luiBlockControlCode,
        __in HOSTMEMORYEXTENT *     pExtent,
        __in BOOL                   bForceInitDataSize
        ) 
        : ReferenceCounted()
    {
        check_critical_path_alloc("6parm-variant", (pTemplate?pTemplate->GetTemplateName():"notemplate"));
#ifdef PROFILE_DATABLOCKS
        m_bIsClone = FALSE;
        m_pCloneSource = NULL;
#endif
        // this version of the constructor *requires* a template. 
        // do more than complain if the caller fails to respect this requirement. 
        if(pTemplate == NULL) {
            assert(pTemplate != NULL && "non-null template required for this Datablock constructor override");
            PTask::Runtime::HandleError("non-null template required for this Datablock constructor override");
            return;
        }

        DefaultInitialize();
        m_pTemplate = pTemplate;
        m_pTemplate->AddRef();

        // By providing a non-null template, the user has told us that all datablock data dimensions
        // are derived from that template, but there is a caveat. If the bForceInitDataSize is true, *and*
        // that size differs from the size in the template, then we don't want to take the byte count
        // from the template. If the force flag is set, set the requested size to the init data size,
        // otherwise, derive a requested size from the template. 

        BOOL bVariableTemplate = pTemplate->IsVariableDimensioned();
        m_bForceRequestedSize = bForceInitDataSize;
        UINT cbInitData = pExtent ? pExtent->uiSizeBytes : 0;
        m_cbRequested[DBDATA_IDX] = 
            (bForceInitDataSize || bVariableTemplate) ? 
                cbInitData : 
                pTemplate->GetDatablockByteCount(DBDATA_IDX);

        // set the block control code
        __setControlSignal(luiBlockControlCode);

        // most common access case is data block produced by the program
        if(PT_ACCESS_DEFAULT == (m_eBufferAccessFlags = eFlags)) 
            m_eBufferAccessFlags = PT_ACCESS_HOST_WRITE | PT_ACCESS_ACCELERATOR_READ;
        m_bByteAddressable = (m_eBufferAccessFlags & PT_ACCESS_BYTE_ADDRESSABLE);

        MaterializeViews(NULL, pAsyncContext, pExtent);

        // call application context callback if there is one associated with the template.
        LPFNAPPLICATIONCONTEXTCALLBACK pCallback = pTemplate->GetApplicationContextCallback();
        if (NULL != pCallback)
        {
            pCallback(
                APPLICATIONCONTEXTCALLBACKPOINT::CALLBACKPOINT_CREATE,
                this, &this->m_pApplicationContext);
        }

        CHECK_INVARIANTS_LOCK();
    } 

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 12/19/2011. Refactored crossbac 7/9/12 to handle differing init
    ///             data/template dimensions according to caller's needs.
    ///             </remarks>
    ///
    /// <param name="vAsyncContexts">           [in] If non-null, an async context, which will wind
    ///                                         up using this block. </param>
    /// <param name="pTemplate">                [in] If non-null, the datablock template. </param>
    /// <param name="eFlags">                   [in] buffer access flags. </param>
    /// <param name="luiBlockControlCode">      [in] a block control code. </param>
    /// <param name="pExtent">                  [in] If non-null, initial data. </param>
    /// <param name="bForceInitDataSize">       True if the number of bytes in the initial data
    ///                                         buffer is different from that in the template
    ///                                         parameter *AND* the caller would like the allocation
    ///                                         to match the init data size. In this case, the
    ///                                         allocation size supercedes dims in the template. If
    ///                                         the this parameter is FALSE and the dimensions differ,
    ///                                         then the allocation is based template dimensions and
    ///                                         init data is striped (or truncated) to fit the
    ///                                         allocation.
    ///                                         . </param>
    /// <param name="bCreateDeviceBuffers">     The materialize all. </param>
    /// <param name="bMaterializeDeviceViews">  The materialize device views. </param>
    /// <param name="bPageLockHostViews">       The page lock host views. </param>
    ///-------------------------------------------------------------------------------------------------

    Datablock::Datablock(
        __in std::set<AsyncContext*>&    vAsyncContexts,
        __in DatablockTemplate *         pTemplate, 
        __in BUFFERACCESSFLAGS           eFlags,
        __in CONTROLSIGNAL               luiBlockControlCode,
        __in HOSTMEMORYEXTENT *          pExtent,
        __in BOOL                        bForceInitDataSize,
        __in BOOL                        bCreateDeviceBuffers,
        __in BOOL                        bMaterializeDeviceViews,
        __in BOOL                        bPageLockHostViews
        ) 
        : ReferenceCounted()
    {
        check_critical_path_alloc("9parm-variant", (pTemplate?pTemplate->GetTemplateName():"notemplate"));
#ifdef PROFILE_DATABLOCKS
        m_bIsClone = FALSE;
        m_pCloneSource = NULL;
#endif
        // this version of the constructor *requires* a template. 
        // do more than complain if the caller fails to respect this requirement. 
        if(pTemplate == NULL) {
            assert(pTemplate != NULL && "non-null template required for this Datablock constructor override");
            PTask::Runtime::HandleError("non-null template required for this Datablock constructor override");
            return;
        }

        DefaultInitialize();
        m_pTemplate = pTemplate;
        m_pTemplate->AddRef();

        // By providing a non-null template, the user has told us that all datablock data dimensions
        // are derived from that template, but there is a caveat. If the bForceInitDataSize is true, *and*
        // that size differs from the size in the template, then we don't want to take the byte count
        // from the template. If the force flag is set, set the requested size to the init data size,
        // otherwise, derive a requested size from the template. 

        BOOL bVariableTemplate = pTemplate->IsVariableDimensioned();
        m_bForceRequestedSize = bForceInitDataSize;
        UINT cbInitData = pExtent ? pExtent->uiSizeBytes : 0;
        m_cbRequested[DBDATA_IDX] = 
            (bForceInitDataSize || bVariableTemplate) ? 
                cbInitData : 
                pTemplate->GetDatablockByteCount(DBDATA_IDX);

        // set the block control code
        __setControlSignal(luiBlockControlCode);

        // most common access case is data block produced by the program
        if(PT_ACCESS_DEFAULT == (m_eBufferAccessFlags = eFlags)) 
            m_eBufferAccessFlags = PT_ACCESS_HOST_WRITE | PT_ACCESS_ACCELERATOR_READ;
        m_bByteAddressable = (m_eBufferAccessFlags & PT_ACCESS_BYTE_ADDRESSABLE);

        MaterializeViews(vAsyncContexts, pExtent, bCreateDeviceBuffers, bMaterializeDeviceViews, bPageLockHostViews);

        // call application context callback if there is one associated with the template.
        LPFNAPPLICATIONCONTEXTCALLBACK pCallback = pTemplate->GetApplicationContextCallback();
        if (NULL != pCallback)
        {
            pCallback(
                APPLICATIONCONTEXTCALLBACKPOINT::CALLBACKPOINT_CREATE,
                this, &this->m_pApplicationContext);
        }

        CHECK_INVARIANTS_LOCK();
    } 

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Set application context associated with this datablock.
    ///
    ///             The context is initially NULL when a datablock is created, and is
    ///             reinitialized to NULL each time it is returned to a block pool.
    ///
    ///             The context is copied to any clones of the datablock.
    ///             </summary>
    ///
    /// <remarks>   jcurrey, 3/21/2014. </remarks>
    ///
    /// <param name="pApplicationContext">   The context. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::SetApplicationContext(
        __in void * pApplicationContext
        )
    {
        m_pApplicationContext = pApplicationContext;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Get application context that the application previously has associated with 
    ///             this datablock.
    ///
    ///             The context is initially NULL when a datablock is created, and is
    ///             reinitialized to NULL each time it is returned to a block pool.
    ///
    ///             The context is copied to any clones of the datablock.
    ///             </summary>
    ///
    /// <remarks>   jcurrey, 3/21/2014. </remarks>
    ///
    /// <returns>   The context. </returns>
    ///-------------------------------------------------------------------------------------------------

    void * 
    Datablock::GetApplicationContext()
    {
        return m_pApplicationContext;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resets the initial value for the pool. </summary>
    ///
    /// <remarks>   crossbac, 4/30/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Datablock::ResetInitialValueForPool(
        __in Accelerator * pAccelerator
        )
    {
        Lock();
        if(m_bPooledBlock && m_bPoolRequiresInitialValue) {

            assert(m_pTemplate->HasInitialValue());
            void * lpvInitData = (void*)m_pTemplate->GetInitialValue();
            UINT cbDataSize = m_pTemplate->GetInitialValueSizeBytes();
            BOOL bViewMaterialized = FALSE;

            if(pAccelerator != NULL && !pAccelerator->IsHost()) {
                
                // if need the initial value in a device space, we can materialize it either by allocating
                // buffers in that space if they are absent (in which case the template will be used to set the
                // initial value), by memsetting on the device, if appropriate, or in the worst case, by
                // creating a host view of the initial value and invalidating all device views (so that
                // subsequent bind attempts will update the device view based on the valid view in host space).

                UINT uiMemSpaceId = pAccelerator->GetMemorySpaceId();
                BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[uiMemSpaceId];
                if(pEntry->eState == BSTATE_NO_ENTRY) {

                    // allocate the buffers, with populate flag, which will cause the needed init
                    AllocateBuffers(uiMemSpaceId, pAccelerator->GetAsyncContext(ASYNCCTXT_XFERHTOD), TRUE);
                    bViewMaterialized = TRUE;

                } else {

                    if(m_pTemplate->IsInitialValueMemsettable()) {
                        PBuffer * pBuffer = pEntry->pBuffers[DBDATA_IDX];
                        if(pBuffer->SupportsMemset() && 
                           m_pTemplate->IsInitialValueMemsettableD8()) {
                           int nMemsetValue = (int) (*((char*)lpvInitData));
                           pBuffer->FillExtent(nMemsetValue);
                           bViewMaterialized = TRUE;
                        }
                    }
                }

            } 

            if(!bViewMaterialized) {

                // we didn't get a fresh initial view through either of the
                // above methods. So 
                if(m_ppBufferMap[HOST_MEMORY_SPACE_ID]->eState == BSTATE_NO_ENTRY) {
                    AllocateBuffers(HOST_MEMORY_SPACE_ID, NULL, FALSE);
                }
                assert(m_ppBufferMap[HOST_MEMORY_SPACE_ID]->eState != BSTATE_NO_ENTRY);
                if(m_ppBufferMap[HOST_MEMORY_SPACE_ID]->eState != BSTATE_NO_ENTRY) {
                    PBuffer * pHostBuffer = GetPlatformBuffer(HOST_MEMORY_SPACE_ID, DBDATA_IDX);
                    unsigned char * pHostData = (unsigned char*)pHostBuffer->GetBuffer();
                    UINT cbHostData = m_pTemplate->GetDatablockByteCount();
                    UINT cbRemaining = cbHostData;
                    UINT cbCopied = 0;
                    while(cbRemaining) {
                        UINT cbCopy = min(cbRemaining, cbDataSize);
                        memcpy(pHostData+cbCopied, lpvInitData, cbCopy);
                        cbCopied += cbCopy;
                        cbRemaining -= cbCopy;
                    }
                    SetCoherenceState(HOST_MEMORY_SPACE_ID, BSTATE_EXCLUSIVE);
                    m_bPoolInitialValueSet = TRUE;
                }
            }
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets pool initial value valid. </summary>
    ///
    /// <remarks>   crossbac, 5/4/2013. </remarks>
    ///
    /// <param name="bValid">   true to valid. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::SetPoolInitialValueValid(
        BOOL bValid
        )
    {
        assert(LockIsHeld());
        m_bPoolInitialValueSet = bValid;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets pool initial value valid. </summary>
    ///
    /// <remarks>   crossbac, 5/4/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::GetPoolInitialValueValid(
        VOID
        )
    {
        assert(LockIsHeld());
        return m_bPoolInitialValueSet;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize view(s) of the initial data in whatever memory spaces are most
    ///             appropriate / possible. If there are multiple devices, refer to the materialize
    ///             all flag to decide whether to create device views. If we cannot create device-
    ///             side buffers, based on the context (many devices) and the flags (materialize all
    ///             flag is false), then create a pinned host view to ensure that subsequent async
    ///             APIs can use the block. If we can create the device side view, refer to the pin
    ///             flag to decide whether or not to pin a host view.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/10/2012. </remarks>
    ///
    /// <param name="vAsyncContexts">           [in,out] If non-null, target accelerator. </param>
    /// <param name="pExtent">                  [in,out] If non-null, information describing the lpv
    ///                                         initialise. </param>
    /// <param name="bCreateDeviceBuffers">     The materialize all. </param>
    /// <param name="bMaterializeDeviceViews">  The materialize device views. </param>
    /// <param name="bForcePinned">             The force pinned. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::MaterializeViews(
        __in std::set<AsyncContext*>&    vAsyncContexts,
        __in HOSTMEMORYEXTENT *          pExtent,
        __in BOOL                        bCreateAllDeviceBuffers,
        __in BOOL                        bMaterializeDeviceViews,
        __in BOOL                        bForcePinned 
        )
    {
        // if we don't have a target, or no template,
        // or if this is a block that is bound to a scalar,
        // default to a normal host-only view. Technically,
        // we should complain if the programmer uses this API
        // to attempt that kind of allocation, but it's simple
        // enough to accommodate the abuse. 
        if(vAsyncContexts.size() == 0 ||
           m_pTemplate == NULL ||
           m_pTemplate->DescribesScalarParameter())
            return MaterializeViews(NULL, NULL, pExtent);

        void * lpvInitData = pExtent ? pExtent->lpvAddress : NULL;
        UINT cbInitData = pExtent ? pExtent->uiSizeBytes : 0;

        // some platforms allow setting function parms without explicitly creating a device-side
        // buffer. If this block is going to be bound to such a variable, we don't want to create any
        // device-side buffers.
        BOOL bByvalArgSupport = TRUE;
        BOOL bPinnedHostSupport = TRUE;
        BOOL bDeviceTargetExists = FALSE;
        set<Accelerator*> vViewAccelerators;
        set<Accelerator*> vMaterializedBufferSpaces;
        map<AsyncContext*, Accelerator*> vProxyAllocators;
        map<AsyncContext*, Accelerator*> vMatViewAccelerators;
        std::set<AsyncContext*>::iterator asi; 
        map<AsyncContext*, Accelerator*>::iterator si;
        UINT nNonHostAccelerators = 0;
        BOOL bExplicitHostTarget = FALSE;
        for(asi=vAsyncContexts.begin(); asi!=vAsyncContexts.end(); asi++) {
            AsyncContext * pContext = *asi;
            Accelerator * pTargetAccelerator = pContext->GetDeviceContext();
            BOOL bTargetIsHost = pTargetAccelerator->IsHost();
            bExplicitHostTarget |= bTargetIsHost;
            nNonHostAccelerators += bTargetIsHost ? 0 : 1;
            bDeviceTargetExists |= !bTargetIsHost;
            bByvalArgSupport &= pTargetAccelerator->SupportsByvalArguments();
            vViewAccelerators.insert(pTargetAccelerator);
            vMatViewAccelerators[pContext] = pTargetAccelerator;
            if(!pTargetAccelerator->IsHost() &&
                pTargetAccelerator->SupportsPinnedHostMemory()) {
                bPinnedHostSupport = TRUE;
                vProxyAllocators[pContext] = pTargetAccelerator;
            }
        }

        // if the caller gave us no template, or the template describes data with mulitple channels and
        // variable stride, we cannot assume that the initial data is final, even if it's present, so
        // we will not bother to materialize or allocate any device-side buffers. Of course, the exception
        // is when the block is already finalized...  
        BOOL bFinalizeable = (m_pTemplate != NULL || m_bSealed);
        BOOL bWantDeviceBuffers = (vViewAccelerators.size() == 1) ||     // need exactly 1 target, or programmer hint
                                   bCreateAllDeviceBuffers;              // or programmer hint
        BOOL bCreateDeviceBuffers = bWantDeviceBuffers && bFinalizeable; // need final size, and some target accelerators
        BOOL bMaterializeOnDevice =                                      // should we create the device-side buffer eagerly? 
                 (bCreateDeviceBuffers &&                                // we have accelerator(s) to use
                  (lpvInitData != NULL) &&                               // we have data
                  bMaterializeDeviceViews);                              // the caller wants us to

        // if we are going to materialize this block in a non-host memory space, we may want to create
        // a host view first that enables us to create pinned memory to back this buffer. We only want
        // to do this if we are sure we'll require a host-side view later on (a property we can infer
        // from the block permissions) and we know that one of the accelerators we are creating a view
        // on actually supports pinned memory. 
        PBuffer * pHostPBuffer = NULL;
        BOOL bPinnedHostViewCreated = FALSE;
        int nValidViewsMaterialized = 0;
        BOOL bPinnedHostViewWanted = !bWantDeviceBuffers || bForcePinned;
        BOOL bPinnedHostViewPossible = bPinnedHostSupport && bPinnedHostViewWanted;
        m_bAttemptPinnedHostBuffers = bPinnedHostViewPossible;

        Lock();
        BOOL bTrackableTransitionOccurred = FALSE; // used by profiler
        ctprofile_view_update_decl();

        if(bPinnedHostViewPossible) {

            AsyncContext * pAsyncContext = NULL;
            Accelerator * pProxyAllocator = NULL;
            if(vProxyAllocators.size()) {
                map<AsyncContext*, Accelerator*>::iterator mi;
                mi = vProxyAllocators.begin();
                assert(mi->second != NULL);
                assert(mi->first != NULL);
                assert(!(mi->second)->IsHost());
            }
            UINT cbChannelAllocSize = GetChannelAllocationSizeBytes(DBDATA_IDX);
            UINT cbAllocationSize = cbChannelAllocSize == 0 ? cbInitData : cbChannelAllocSize;
            if(m_bForceRequestedSize && cbAllocationSize != cbInitData) {
                cbAllocationSize = cbInitData;
            }
            ctprofile_view_update_start(HOST_MEMORY_SPACE_ID, CET_PINNED_HOST_VIEW_CREATE);
            AllocateBuffer(HOST_MEMORY_SPACE_ID,
                           pProxyAllocator,
                           pAsyncContext,
                           DBDATA_IDX,
                           cbAllocationSize,
                           pExtent);
            pHostPBuffer = GetPlatformBuffer(HOST_MEMORY_SPACE_ID, DBDATA_IDX);
            bPinnedHostViewCreated = (pHostPBuffer != NULL) && pHostPBuffer->IsPhysicalBufferPinned();
            nValidViewsMaterialized += (lpvInitData == NULL) ? 0 : 1;
            ctprofile_view_update_end(HOST_MEMORY_SPACE_ID, BSTATE_EXCLUSIVE, NOXFER);
        }

        BOOL bDeviceMaterialized = FALSE;
        if(bCreateDeviceBuffers) {

            // we've got what we need to create the accelerator-side
            // materialization directly, without any host-side buffering
            // bMaterializeOnDevice
            UINT cbChannelAllocSize = GetChannelAllocationSizeBytes(DBDATA_IDX);
            UINT cbAllocationSize = cbChannelAllocSize == 0 ? cbInitData : cbChannelAllocSize;
            if(m_bForceRequestedSize && cbAllocationSize != cbInitData) {
                cbAllocationSize = cbInitData;
            }

            HOSTMEMORYEXTENT extent;
            HOSTMEMORYEXTENT * pAccInitExtent = NULL;
            if(bMaterializeOnDevice) {
                HOSTMEMORYEXTENT * pAccInitExtent = pExtent;
                if(lpvInitData != NULL && bPinnedHostViewCreated) {
                    assert(pHostPBuffer != NULL);
                    extent.lpvAddress = pHostPBuffer->GetBuffer();
                    extent.uiSizeBytes = pHostPBuffer->GetLogicalExtentBytes();
                    extent.bPinned = pHostPBuffer->IsPhysicalBufferPinned();
                    pAccInitExtent = &extent;
                }
            }
            
            ctprofile_view_update_continue_a((*(vViewAccelerators.begin())), CET_BLOCK_CREATE);
            bTrackableTransitionOccurred = TRUE;
            for(si=vMatViewAccelerators.begin(); si!=vMatViewAccelerators.end(); si++) {
                AsyncContext * pAsyncContext = si->first;
                Accelerator * pAccelerator = si->second;
                if(pAccelerator->IsHost() && pHostPBuffer != NULL) 
                    continue; // already did it!
                AllocateBuffer(pAccelerator,
                               pAsyncContext, 
                               DBDATA_IDX, 
                               cbAllocationSize,
                               pAccInitExtent);
                nValidViewsMaterialized += bMaterializeOnDevice ? 1 : 0;
                bDeviceMaterialized = bMaterializeOnDevice;
                if(bDeviceMaterialized)
                    vMaterializedBufferSpaces.insert(pAccelerator);
            }
        } 

        // should we create the host-side buffer? Ideally, we've done it already!
        // However, we can get to here if we *failed* to create a pinned host view and 
        // we did not create a device view either. 

        if(!bDeviceMaterialized && pHostPBuffer == NULL) {

            if(lpvInitData) {

                ctprofile_view_update_continue(HOST_MEMORY_SPACE_ID, CET_BLOCK_CREATE);
                bTrackableTransitionOccurred = TRUE;
                InstantiateChannel(
                    HOST_MEMORY_SPACE_ID,
                    NULL,
                    NULL,
                    DBDATA_IDX,
                    pExtent,
                    TRUE,
                    FALSE);

                pHostPBuffer = GetPlatformBuffer(HOST_MEMORY_SPACE_ID, DBDATA_IDX);
                nValidViewsMaterialized += (lpvInitData == NULL) ? 0 : 1;
                if(pHostPBuffer == NULL) {
                    assert(FALSE);
                    PTask::Runtime::HandleError("%s: Failed to instantiate channel (idx=%d)\n",
                                                __FUNCTION__,
                                                DBDATA_IDX);
                } 
            }
        }

        BUFFER_COHERENCE_STATE state = BSTATE_INVALID;
        if(nValidViewsMaterialized == 0)
            state = BSTATE_INVALID;
        else if(nValidViewsMaterialized == 1) 
            state = BSTATE_EXCLUSIVE;
        else if(nValidViewsMaterialized > 1)
            state = BSTATE_SHARED;
        if(pHostPBuffer != NULL) { 
            SetCoherenceState(HOST_MEMORY_SPACE_ID, state);
        }
        std::set<Accelerator*>::iterator msi;
        for(msi=vMaterializedBufferSpaces.begin(); msi!=vMaterializedBufferSpaces.end(); msi++) {
            Accelerator * pAccelerator = *msi;
            SetCoherenceState(pAccelerator, state);
        }
        ctprofile_view_update_end_cond(bTrackableTransitionOccurred, 
                                       HOST_MEMORY_SPACE_ID, 
                                       state, 
                                       bDeviceMaterialized);
        Unlock();
        return bDeviceMaterialized;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Materialize remote views. </summary>
    ///
    /// <remarks>   crossbac, 7/10/2012. </remarks>
    ///
    /// <param name="pTargetAccelerator">   [in,out] If non-null, target accelerator. </param>
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="pExtent">              [in,out] If non-null, information describing the lpv
    ///                                     initialise. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::MaterializeViews(
        __in Accelerator * pTargetAccelerator,
        __in AsyncContext * pAsyncContext,
        __in HOSTMEMORYEXTENT * pExtent
        )
    {
        void * lpvInitData = pExtent ? pExtent->lpvAddress : NULL;
        UINT cbInitData = pExtent ? pExtent->uiSizeBytes : 0;

        // some platforms allow setting function parms without explicitly creating a device-side
        // buffer. If this block is going to be bound to such a variable, we don't want to create any
        // device-side buffers.
        BOOL bByvalArgSupport = TRUE;
        BOOL bPinnedHostSupport = FALSE;
        Accelerator * pProxyAllocator = NULL;
        set<Accelerator*> vViewAccelerators;
        set<Accelerator*> vMatViewAccelerators;
        if(pTargetAccelerator != NULL && pTargetAccelerator->GetMemorySpaceId() != HOST_MEMORY_SPACE_ID) { 
            vViewAccelerators.insert(pTargetAccelerator);
            bByvalArgSupport &= pTargetAccelerator->SupportsByvalArguments();
            if(pTargetAccelerator->SupportsPinnedHostMemory()) {
                bPinnedHostSupport = TRUE;
                pProxyAllocator = pTargetAccelerator;
            }
        }
        Accelerator * pCtxAccelerator = pAsyncContext == NULL ? NULL : pAsyncContext->GetDeviceContext();
        if(pCtxAccelerator != NULL && pCtxAccelerator->GetMemorySpaceId() != HOST_MEMORY_SPACE_ID) {
            vViewAccelerators.insert(pCtxAccelerator);
            bByvalArgSupport &= pCtxAccelerator->SupportsByvalArguments();
            if(pCtxAccelerator->SupportsPinnedHostMemory()) {
                bPinnedHostSupport = TRUE;
                pProxyAllocator = pCtxAccelerator; // ok to overwrite setting above--
                                                   // either will do if both support pinning
            }
        }

        BOOL bHostOnlyByvalScalar = 
                (m_pTemplate != NULL && 
                (m_pTemplate->DescribesScalarParameter()) &&   // this template is for a scalar const parm
                (vViewAccelerators.size() != 0) &&             // we actually have a candidate accelerator
                (bByvalArgSupport));                           // all candidate accelarators have byval shader arguments

        // if the caller gave us no template, or the template describes data with mulitple channels and
        // variable stride, we cannot assume that the initial data is final, even if it's present, so
        // we will not bother to materialize or allocate any device-side buffers. Of course, the exception
        // is when the block is already finalized...  
        BOOL bFinalizeable = (m_pTemplate != NULL &&
                              !m_pTemplate->DescribesRecordStream() && 
                              !m_pTemplate->IsByteAddressable()) ||
                              m_bSealed;

        // should we create the device-side buffer eagerly? if we can, and it's appropriate, create it!
        BOOL bMaterializeOnDevice =                  
                 ((vViewAccelerators.size() != 0) &&          // we have an accelerator to use
                  (lpvInitData != NULL) &&                    // we have data
                 !bHostOnlyByvalScalar);                      // the block does not represent a scalar 
                                                              // byval function parm (which needs no dev buffer)

        // if we are going to materialize this block in a non-host memory space, we may want to create
        // a host view first that enables us to create pinned memory to back this buffer. We only want
        // to do this if we are sure we'll require a host-side view later on (a property we can infer
        // from the block permissions) and we know that one of the accelerators we are creating a view
        // on actually supports pinned memory. 
        PBuffer * pHostPBuffer = NULL;
        BOOL bPinnedHostViewCreated = FALSE;
        int nValidViewsMaterialized = 0;
        BOOL bPinnedHostViewNeeded = bPinnedHostSupport &&
                                     bMaterializeOnDevice && 
                                     bFinalizeable &&
                                     NEEDS_HOST_VIEW(m_eBufferAccessFlags) &&
                                     NEEDS_ACCELERATOR_VIEW(m_eBufferAccessFlags);

        Lock();
        BOOL bTrackableTransitionOccurred = FALSE; // used by profiler
        ctprofile_view_update_decl();
        if(bPinnedHostViewNeeded) {
            assert(pProxyAllocator != NULL);
            UINT cbChannelAllocSize = GetChannelAllocationSizeBytes(DBDATA_IDX);
            UINT cbAllocationSize = cbChannelAllocSize == 0 ? cbInitData : cbChannelAllocSize;
            if(m_bForceRequestedSize && cbAllocationSize != cbInitData) {
                cbAllocationSize = cbInitData;
            }
            ctprofile_view_update_start(HOST_MEMORY_SPACE_ID, CET_PINNED_HOST_VIEW_CREATE);
            AllocateBuffer(
                HOST_MEMORY_SPACE_ID,
                pProxyAllocator,
                pAsyncContext,
                DBDATA_IDX,
                cbAllocationSize,
                pExtent);
            pHostPBuffer = GetPlatformBuffer(HOST_MEMORY_SPACE_ID, DBDATA_IDX);
            bPinnedHostViewCreated = (pHostPBuffer != NULL) && pHostPBuffer->IsPhysicalBufferPinned();
            nValidViewsMaterialized += (lpvInitData == NULL) ? 0 : 1;
            ctprofile_view_update_end(HOST_MEMORY_SPACE_ID, BSTATE_EXCLUSIVE, NOXFER);
        }

        BOOL bDeviceMaterialized = FALSE;
        if(bMaterializeOnDevice && bFinalizeable) {

            // we've got what we need to create the accelerator-side
            // materialization directly, without any host-side buffering
            bDeviceMaterialized = TRUE;
            UINT cbChannelAllocSize = GetChannelAllocationSizeBytes(DBDATA_IDX);
            UINT cbAllocationSize = cbChannelAllocSize == 0 ? cbInitData : cbChannelAllocSize;
            if(m_bForceRequestedSize && cbAllocationSize != cbInitData) {
                cbAllocationSize = cbInitData;
            }

            HOSTMEMORYEXTENT extent;
            HOSTMEMORYEXTENT * pAccInitExtent = pExtent;
            if(lpvInitData != NULL && pHostPBuffer != NULL) {
                extent.lpvAddress = pHostPBuffer->GetBuffer();
                extent.uiSizeBytes = pHostPBuffer->GetLogicalExtentBytes();
                extent.bPinned = pHostPBuffer->IsPhysicalBufferPinned();
                pAccInitExtent = &extent;
            }
            set<Accelerator*>::iterator si;
            ctprofile_view_update_continue_a((*(vViewAccelerators.begin())), CET_BLOCK_CREATE);
            bTrackableTransitionOccurred = TRUE;
            for(si=vViewAccelerators.begin(); si!=vViewAccelerators.end(); si++) {
                Accelerator * pAccelerator = *si;
                AllocateBuffer(pAccelerator, 
                               pAsyncContext, 
                               DBDATA_IDX, 
                               cbAllocationSize,
                               pAccInitExtent);
                nValidViewsMaterialized += (lpvInitData == NULL) ? 0 : 1;
                vMatViewAccelerators.insert(pAccelerator);
            }
        } 

        // should we create the host-side buffer? We should if we either haven't created a device
        // buffer, or if we have created one, but we know this block is a scalar value that has a high
        // likelihood of being referenced host side. We need in particular to avoid creating only a
        // constant buffer for something that might be needed by the host. If we do this, we risk
        // making it impossible to recreate the host view. 

        BOOL bMaterializeOnHost = ((m_pTemplate != NULL) && m_pTemplate->DescribesScalarParameter());
        BOOL bMaterializeZeroLength = (m_pTemplate != NULL && m_pTemplate->IsInitialValueEmpty() && !lpvInitData);
        if((!bDeviceMaterialized || bMaterializeOnHost) && (pHostPBuffer == NULL)) {

            // Either we can't assume the initial data is final, or we don't know yet what accelerator this
            // datablock will have materializations on, so if we have init data for the default channel, we
            // need to stash a copy of of the init data for later when we are actually bound to an
            // accelerator for dispatch. We also must handle the case where this block is created from an
            // initializer in the template that has no records (zero-length/empty buffer). We have to
            // actually create buffers to back the block in this case because device-side bindings must
            // involve non-null buffers. 
            
            if(lpvInitData || bMaterializeZeroLength) {

                ctprofile_view_update_continue(HOST_MEMORY_SPACE_ID, CET_BLOCK_CREATE);
                bTrackableTransitionOccurred = TRUE;
                InstantiateChannel(
                    HOST_MEMORY_SPACE_ID,
                    pAsyncContext,
                    NULL,
                    DBDATA_IDX,
                    pExtent,
                    TRUE,
                    FALSE);

                pHostPBuffer = GetPlatformBuffer(HOST_MEMORY_SPACE_ID, DBDATA_IDX);
                nValidViewsMaterialized += (lpvInitData == NULL) ? 0 : 1;
                if(pHostPBuffer == NULL) {
                    assert(FALSE);
                    PTask::Runtime::HandleError("%s: Failed to instantiate channel (idx=%d)",
                                                __FUNCTION__,
                                                DBDATA_IDX);
                } 
            }
        }

        BUFFER_COHERENCE_STATE state = BSTATE_INVALID;
        if(nValidViewsMaterialized == 0)
            state = BSTATE_INVALID;
        else if(nValidViewsMaterialized == 1) 
            state = BSTATE_EXCLUSIVE;
        else if(nValidViewsMaterialized > 1)
            state = BSTATE_SHARED;
        if(pHostPBuffer != NULL) { 
            SetCoherenceState(HOST_MEMORY_SPACE_ID, state);
        }
        set<Accelerator*>::iterator si;
        for(si=vMatViewAccelerators.begin(); si!=vMatViewAccelerators.end(); si++) {
            Accelerator * pAccelerator = *si;
            SetCoherenceState(pAccelerator, state);
        }
        ctprofile_view_update_end_cond(bTrackableTransitionOccurred, 
                                       HOST_MEMORY_SPACE_ID, 
                                       state, 
                                       bDeviceMaterialized);
        Unlock();
        return bDeviceMaterialized;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="pTemplate">            [in] If non-null, the datablock template. </param>
    /// <param name="uiDataBufferSize">     Size of the data buffer. </param>
    /// <param name="metaBufferSize">       Size of the meta data buffer. </param>
    /// <param name="templateBufferSize">   Size of the template buffer. </param>
    /// <param name="eFlags">               The flags. </param>
    /// <param name="luiBlockControlCode">  The control code encodes control signals associated with
    ///                                     this block. For example, if the block has a control code
    ///                                     of DBCTL_EOF, then this is the last block in a stream of
    ///                                     records. </param>
    /// <param name="pExtent">              [in,out] If non-null, the extent. </param>
    /// <param name="uiRecordCount">        Number of records. </param>
    /// <param name="bFinalize">            The finalize. </param>
    ///
    ///-------------------------------------------------------------------------------------------------

    Datablock::Datablock(
        __in AsyncContext *         pAsyncContext,
        __in DatablockTemplate *    pTemplate, 
        __in UINT                   uiDataBufferSize, 
        __in UINT                   metaBufferSize, 
        __in UINT                   templateBufferSize,
        __in BUFFERACCESSFLAGS      eFlags,
        __in CONTROLSIGNAL          luiBlockControlCode,
        __in HOSTMEMORYEXTENT *     pExtent,
        __in UINT                   uiRecordCount,
        __in BOOL                   bFinalize
        )
    {
        check_critical_path_alloc("10parm-variant", (pTemplate?pTemplate->GetTemplateName():"notemplate"));

#ifdef PROFILE_DATABLOCKS
        m_bIsClone = FALSE;
        m_pCloneSource = NULL;
#endif
        DefaultInitialize();
        m_bMarshallable = TRUE;
        m_bDeleting = FALSE;
        if(pTemplate != NULL) {
            m_pTemplate = pTemplate;
            m_pTemplate->AddRef();

            // call application context callback if there is one associated with the template.
            LPFNAPPLICATIONCONTEXTCALLBACK pCallback = pTemplate->GetApplicationContextCallback();
            if (NULL != pCallback)
            {
                pCallback(
                    APPLICATIONCONTEXTCALLBACKPOINT::CALLBACKPOINT_CREATE,
                    this, &this->m_pApplicationContext);
            }
        }
        m_cbRequested[DBDATA_IDX] = uiDataBufferSize;
        m_cbRequested[DBMETADATA_IDX] = metaBufferSize;
        m_cbRequested[DBTEMPLATE_IDX] = templateBufferSize;
        m_eBufferAccessFlags = (eFlags == 0 || eFlags == PT_ACCESS_HOST_WRITE) ?
            (PT_ACCESS_HOST_WRITE | PT_ACCESS_ACCELERATOR_READ) : eFlags;
        m_bByteAddressable = (m_eBufferAccessFlags & PT_ACCESS_BYTE_ADDRESSABLE);
        __setControlSignal(luiBlockControlCode); 
        m_bRecordStream = (metaBufferSize != 0 || templateBufferSize != 0);
        m_pProducerTask = NULL;

        if(pExtent && pExtent->lpvAddress != NULL && bFinalize) {
            Lock();
            Seal(uiRecordCount, uiDataBufferSize, metaBufferSize, templateBufferSize);
            MaterializeViews(NULL, pAsyncContext, pExtent);
            Unlock();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="luiBlockControlCode">  The control code. </param>
    ///-------------------------------------------------------------------------------------------------

    Datablock::Datablock(
        __in CONTROLSIGNAL luiBlockControlCode
        )
    {
        check_critical_path_alloc("ctl-code", "template NA");
#ifdef PROFILE_DATABLOCKS
        m_bIsClone = FALSE;
        m_pCloneSource = NULL;
#endif
        DefaultInitialize();
        __setControlSignal(luiBlockControlCode); 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Makes a deep copy of this object. </summary>
    ///
    /// <remarks>   Crossbac, 5/25/2012. </remarks>
    ///
    /// <param name="pClone">           [in,out] If non-null, the clone. </param>
    /// <param name="pSrcAsyncContext"> [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="pDstAsyncContext"> [in,out] If non-null, context for the destination
    ///                                 asynchronous. </param>
    ///
    /// <returns>   null if it fails, else a copy of this object. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *
    Datablock::Clone(
        __in Datablock * pClone,
        __in AsyncContext * pSrcAsyncContext,
        __in AsyncContext * pDstAsyncContext
        )
    {
		if(pClone->m_bPooledBlock) {
			BlockPoolOwner * pOwner = pClone->m_pPoolOwner;
			Datablock * pPooledBlock = pOwner->GetBlockFromPool();
			if(pPooledBlock != NULL) {
				pPooledBlock->InitializeClone(pClone, pSrcAsyncContext, pDstAsyncContext);
				return pPooledBlock;
			}
		}
        check_critical_path_clone();
        return new Datablock(pClone, pSrcAsyncContext, pDstAsyncContext);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Copy constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pBlockToClone">    [in,out] If non-null, the clone. </param>
    /// <param name="pSrcAsyncContext"> [in,out] If non-null, context for the source asynchronous. </param>
    /// <param name="pDstAsyncContext"> [in,out] If non-null, context for the destination
    ///                                 asynchronous. </param>
    ///-------------------------------------------------------------------------------------------------

    Datablock::Datablock(
        __in Datablock * pBlockToClone,
        __in AsyncContext * pSrcAsyncContext,
        __in AsyncContext * pDstAsyncContext
        ) 
    {
        check_critical_path_alloc("clone", 
            pBlockToClone->GetTemplate() != NULL ? pBlockToClone->GetTemplate()->GetTemplateName() : "no template");
#ifdef PROFILE_DATABLOCKS
        m_bIsClone = FALSE;
        m_pCloneSource = NULL;
#endif
        DefaultInitialize();
		InitializeClone(pBlockToClone, 
						pSrcAsyncContext,
						pDstAsyncContext);
    }

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Initializes the clone. </summary>
	///
	/// <remarks>	crossbac, 8/14/2013. </remarks>
	///
	/// <param name="pClonedBlock">	   	[in,out] If non-null, the cloned block. </param>
	/// <param name="pSrcAsyncContext">	[in,out] If non-null, context for the source asynchronous. </param>
	/// <param name="pDstAsyncContext">	[in,out] If non-null, context for the destination
	/// 								asynchronous. </param>
	///-------------------------------------------------------------------------------------------------
    
	void
	Datablock::InitializeClone(
        __in Datablock * pBlockToClone,
        __in AsyncContext * pSrcAsyncContext,
        __in AsyncContext * pDstAsyncContext
        ) 
    {
        assert(pBlockToClone != NULL);
        m_bDeleting = FALSE;
        m_pDestinationPort = NULL;
        m_bMarshallable = pBlockToClone->m_bMarshallable;
        m_pTemplate = pBlockToClone->GetTemplate();
        if(m_pTemplate)
        {
            m_pTemplate->AddRef();
            
            // call application context callback if there is one associated with the template.
            LPFNAPPLICATIONCONTEXTCALLBACK pCallback = m_pTemplate->GetApplicationContextCallback();
            if (NULL != pCallback)
            {
                pCallback(
                    APPLICATIONCONTEXTCALLBACKPOINT::CALLBACKPOINT_CLONE,
                    pBlockToClone, &this->m_pApplicationContext);
            }
        }

        Lock(); 
        memcpy(m_cbRequested, pBlockToClone->m_cbRequested, NUM_DATABLOCK_CHANNELS * sizeof(UINT));
        memcpy(m_cbAllocated, pBlockToClone->m_cbAllocated, NUM_DATABLOCK_CHANNELS * sizeof(UINT));
        memcpy(m_cbFinalized, pBlockToClone->m_cbFinalized, NUM_DATABLOCK_CHANNELS * sizeof(UINT));
        __setControlSignal(pBlockToClone->m_vControlSignalStack.top());
        m_uiRecordCount = pBlockToClone->m_uiRecordCount;
        m_bRecordStream = pBlockToClone->m_bRecordStream;
        m_pProducerTask = pBlockToClone->m_pProducerTask;
        m_bPooledBlock = FALSE;
        m_pPoolOwner = NULL;
        m_eBufferAccessFlags = pBlockToClone->m_eBufferAccessFlags; 
        m_bByteAddressable = (m_eBufferAccessFlags & PT_ACCESS_BYTE_ADDRESSABLE);
        m_bSealed = pBlockToClone->m_bSealed;
        m_pApplicationContext = pBlockToClone->m_pApplicationContext;

        UINT uiDestMemSpace = HOST_MEMORY_SPACE_ID;   // used by profiler--can tolerate inaccuracy
        UINT uiSrcMemSpace = HOST_MEMORY_SPACE_ID;    // used by profiler--can tolerate inaccuracy
        BOOL bTransferOccurred = FALSE; bTransferOccurred; // <--suppress compiler warnings
        BOOL bSrcAsyncContextUpdateRequired = (pSrcAsyncContext == NULL);
        Accelerator * pLastAcc = pBlockToClone->GetMostRecentAccelerator();
        Accelerator * pSrcAcc = pLastAcc;             
        Accelerator * pDestAcc = NULL;

        if(pDstAsyncContext != NULL) {

            // if a destination async context has been specified, figure
            // out where we are supposed to materialize the data.
            pDestAcc = pDstAsyncContext->GetDeviceContext();
            uiDestMemSpace = pDestAcc->GetMemorySpaceId();
            assert(ASYNCCTXT_ISXFERCTXT(pDstAsyncContext->GetAsyncContextType()) ||
                   !pDstAsyncContext->SupportsExplicitAsyncOperations());
        }

        if(pSrcAsyncContext != NULL) {

            // if a source async context has been specified, figure out if it actually corresponds to a
            // device for which we have a valid buffer--if not, and the block actually *does* have a
            // cloneable value in another memory space, override the source context request. 

            pSrcAcc = pSrcAsyncContext->GetDeviceContext();
            uiSrcMemSpace = pSrcAcc->GetMemorySpaceId();
            if(pLastAcc != NULL && pLastAcc != pSrcAcc) {

                // the device for the source context differs from the device considered to be the "most"
                // recent. This can occur legitimately if the block is in shared state and has valid buffers in
                // both the corresponding memory spaces. When this is *not* the case, we defer to the most
                // recent, since the coherence protocol must be respected to maintain correctness. Otherwise,
                // we respect the caller's request. 
                
                UINT uiLastAccSpace = pLastAcc->GetMemorySpaceId();
                assert(uiLastAccSpace != uiSrcMemSpace);
                BOOL bLastAccValid = valid(m_ppBufferMap[uiLastAccSpace]->eState) && HasBuffers(uiLastAccSpace);
                BOOL bSrcAccValid = valid(m_ppBufferMap[uiSrcMemSpace]->eState) && HasBuffers(uiSrcMemSpace);
                if(bLastAccValid && !bSrcAccValid) {

                    // the most recent accelerator has a clonable version of the data, but
                    // the requested source does not. We must override the caller's request.
                    uiSrcMemSpace = uiLastAccSpace;
                    pSrcAcc = pLastAcc;
                    bSrcAsyncContextUpdateRequired = TRUE;
                }
            }
        } 

        if(bSrcAsyncContextUpdateRequired) {

            // we need to choose a source async context, either because the caller didn't provide
            // one or because the one provided was a for a device that had no valid view of the
            // data we are attempting to clone. If we have a device source, and this is the case,
            // look at the destination to choose an async context appropriate to the operation.
            
            if(uiDestMemSpace == HOST_MEMORY_SPACE_ID && uiSrcMemSpace != HOST_MEMORY_SPACE_ID) {

                // cloning from device space into host space. get the 
                // dedicated D to H context for the source device.                 
                assert(pSrcAcc != NULL);
                assert(!pSrcAcc->IsHost());
                pSrcAsyncContext = pSrcAcc->GetAsyncContext(ASYNCCTXT_XFERDTOH);

            } else if(uiDestMemSpace != HOST_MEMORY_SPACE_ID && uiSrcMemSpace != HOST_MEMORY_SPACE_ID) {

                // device-device materialization. Regardless of which context we choose to queue
                // the operation on, we want to use the D to D async context for the source accelerator
                // (meaning device memcpy and attempted p2p copies will both be attempted on the dedicated
                // DtoD context for the source accelerator--for the former there  is no choice actually,
                // while in the latter case, it's a matter of policy, where the current policy is chosen
                // for convenience: TODO: FIXME: find out if this matters!)
               
                assert(pSrcAcc != NULL);
                assert(!pSrcAcc->IsHost());
                assert(pDestAcc != NULL);
                assert(!pDestAcc->IsHost());
                pSrcAsyncContext = pSrcAcc->GetAsyncContext(ASYNCCTXT_XFERDTOD);
                
            }

        } else {

            // if we have an async context to use, it better be a transfer stream. 
            if(pSrcAsyncContext != NULL && !pSrcAcc->IsHost()) {
                assert(ASYNCCTXT_ISXFERCTXT(pSrcAsyncContext->GetAsyncContextType()) ||
                      !pSrcAsyncContext->SupportsExplicitAsyncOperations());
            }
        }

        // at this point, we know something about where the caller would like to have seen the
        // data materialized and which asynchrounous contexts are candidates for queueing any
        // data transfer or data copy. Decide where to materialize valid views in the new block. 
        // If the source and/or destination are known, we will try to queue operations in the 
        // relevant async contexts. If we have multiple valid views in the block to clone, we have
        // a choice about how to materialize cloned views: where the request is for a device side
        // view, and we have a valid device view in the source, try to use device memcpy or
        // peer to peer transfer. 
        
        BOOL bDeviceMemcpy = (uiDestMemSpace != HOST_MEMORY_SPACE_ID) && 
                             (uiDestMemSpace == uiSrcMemSpace) &&
                             pSrcAcc->SupportsDeviceMemcpy();

        BOOL bP2PXfer = (!bDeviceMemcpy) &&
                        (uiDestMemSpace != HOST_MEMORY_SPACE_ID) && 
                        (uiSrcMemSpace != HOST_MEMORY_SPACE_ID) && 
                        (uiSrcMemSpace != uiDestMemSpace) &&
                        (pSrcAcc->SupportsDeviceToDeviceTransfer(pDestAcc) ||
                         pDestAcc->SupportsDeviceToDeviceTransfer(pSrcAcc));

        BOOL bSourceLock = (pSrcAcc != NULL) && !pSrcAcc->IsHost();
        BOOL bDstLock = (pDestAcc != NULL) && !pDestAcc->IsHost();
        if(bSourceLock) pSrcAcc->Lock();
        if(bDstLock) pDestAcc->Lock();
        pBlockToClone->Lock();

        ctprofile_view_update_start(uiDestMemSpace, CET_BLOCK_CLONE);

        // used to decide coherence state on new block.
        BOOL bViewsMaterialized = FALSE;

        UINT nChannelIndex;
        for(nChannelIndex=PTask::DBDATA_IDX; 
            nChannelIndex<PTask::NUM_DATABLOCK_CHANNELS;
            nChannelIndex++) 
        {
            if(pBlockToClone->HasValidChannel(nChannelIndex)) {

                void * pChannelData = NULL;
                UINT cbChannelData = 0;
                assert(nChannelIndex != DBDATA_IDX || !HasBuffers(uiDestMemSpace) || pBlockToClone->m_bPooledBlock);
                if(!m_ppBufferMap[uiDestMemSpace]->pBuffers[nChannelIndex]) {
                    AllocateBuffer(uiDestMemSpace, pDestAcc, pDstAsyncContext, nChannelIndex, NULL);
                }
                UINT cbChannelLogical = pBlockToClone->GetChannelLogicalSizeBytes(nChannelIndex);
                PBuffer * pDstBuffer = GetPlatformBuffer(uiDestMemSpace, nChannelIndex);
                PBuffer * pSrcBuffer = pBlockToClone->GetPlatformBuffer(uiSrcMemSpace, nChannelIndex);                
                assert((pDstBuffer != NULL && 
                        !pDstBuffer->HasOutstandingAsyncOps(OT_MEMCPY_TARGET)) || 
                         pBlockToClone->m_bPooledBlock); // should be allocation only, unless we got this block from a pool!
                assert(pSrcBuffer != NULL);

                if(uiDestMemSpace == HOST_MEMORY_SPACE_ID) {

                    if(uiSrcMemSpace != HOST_MEMORY_SPACE_ID && 
                       (valid(pBlockToClone->m_ppBufferMap[HOST_MEMORY_SPACE_ID]->eState))) {

                        // we have a valid host copy. let's just use it instead of requiring yet another
                        // D to H copy operation to populate the host buffer.
                        BUFFER_MAP_ENTRY * pEntry = pBlockToClone->m_ppBufferMap[HOST_MEMORY_SPACE_ID];
                        PBuffer * pSharedHostCopy = pEntry->pBuffers[nChannelIndex];
                        assert(pEntry->eState == BSTATE_SHARED);
                        assert(pSharedHostCopy != NULL);
                        if(pSharedHostCopy->ContextRequiresSync(NULL, OT_MEMCPY_TARGET))
                            pSharedHostCopy->WaitOutstandingAsyncOperations(NULL, OT_MEMCPY_TARGET);
                        void * pSrcChannelData = pSharedHostCopy->GetBuffer();
                        pChannelData = pDstBuffer->GetBuffer();
                        cbChannelData = pSharedHostCopy->GetLogicalExtentBytes();
                        memcpy(pChannelData, pSrcChannelData, min(cbChannelData, cbChannelLogical));
                        bViewsMaterialized = TRUE;

                    } else if(uiSrcMemSpace == HOST_MEMORY_SPACE_ID) {

                        // create only a host view from a host view. requires us to wait any outstanding writes on the
                        // source buffer for this channel, but can otherwise be accomplished with host memcpy. 

                        if(pSrcBuffer->ContextRequiresSync(NULL, OT_MEMCPY_SOURCE))
                            pSrcBuffer->WaitOutstandingAsyncOperations(NULL, OT_MEMCPY_SOURCE);
                        void * pSrcChannelData = pSrcBuffer->GetBuffer();
                        pChannelData = pDstBuffer->GetBuffer();
                        cbChannelData = pSrcBuffer->GetLogicalExtentBytes();
                        memcpy(pChannelData, pSrcChannelData, min(cbChannelData, cbChannelLogical));
                        bViewsMaterialized = TRUE;
                        
                    } else {

                        // create a host view from a device view. There are no consumers of this block
                        // that we know about yet, so it suffices to queue the operation on the source
                        // async context and leave it outstanding. We may need a fence on any oustanding
                        // operations for the source buffer, but the copy call below should handle synchronization
                        // on outstanding operations.
                        
                        assert(ASYNCCTXT_ISXFERCTXT(pSrcAsyncContext->GetAsyncContextType()) ||
                               !pSrcAsyncContext->SupportsExplicitAsyncOperations());
                        pSrcBuffer->PopulateHostView(pSrcAsyncContext, pDstBuffer, FALSE);
                        bViewsMaterialized = TRUE;
                    }

                } else {

                    // the destination view is a device view. We are either copying from the host, or we are
                    // dealing with a device-device copy , which can be a simple device memcpy or a P2P transfer
                    // depending on the requested destination async context. 
                    
                    assert(pDestAcc != NULL);
                    assert(!pDestAcc->IsHost());
                    if(uiSrcMemSpace == HOST_MEMORY_SPACE_ID) {
                        
                        // H to D. Copy call will synchronize
                        assert(pDstAsyncContext != NULL);
                        assert(ASYNCCTXT_ISXFERCTXT(pDstAsyncContext->GetAsyncContextType()) ||
                               !pDstAsyncContext->SupportsExplicitAsyncOperations());
                        pDstBuffer->PopulateAcceleratorView(pDstAsyncContext, pSrcBuffer);
                        bViewsMaterialized = TRUE;                        

                    } else {

                        // materializing a device view from a device view. 
                        if(bDeviceMemcpy) {

                            assert(pSrcAcc != NULL);
                            assert(pSrcAsyncContext != NULL);
                            assert(pSrcAcc->SupportsDeviceMemcpy());
                            assert(pDstAsyncContext == pSrcAsyncContext || pDstAsyncContext == NULL);
                            ASYNCCONTEXTTYPE eContextType = pSrcAsyncContext->GetAsyncContextType(); eContextType; // suppress warning
                            assert(ASYNCCTXT_ISXFERCTXT(eContextType) || !pSrcAsyncContext->SupportsExplicitAsyncOperations());
                            if(pSrcAsyncContext->SupportsExplicitAsyncOperations()) {
                                assert(eContextType == ASYNCCTXT_XFERDTOD);
                                pSrcBuffer->Copy(pDstBuffer, pSrcBuffer, pSrcAsyncContext, cbChannelLogical);
                            } else {
                                assert(FALSE);
                            }
                            bViewsMaterialized = TRUE;

                        } else if(bP2PXfer) {

                            // peer to peer transfer!
                            assert(pSrcAcc != NULL);
                            assert(pDestAcc != NULL);
                            assert(pSrcAsyncContext != NULL);
                            assert(pDstAsyncContext != NULL);
                            assert(pDstAsyncContext != pSrcAsyncContext);
                            assert(pSrcAcc->SupportsDeviceToDeviceTransfer(pDestAcc));
                            ASYNCCONTEXTTYPE eSrcContextType = pSrcAsyncContext->GetAsyncContextType(); eSrcContextType; // suppress warnings
                            ASYNCCONTEXTTYPE eDstContextType = pDstAsyncContext->GetAsyncContextType(); eDstContextType; // suppress warnings
                            assert(ASYNCCTXT_ISXFERCTXT(eSrcContextType) || !pSrcAsyncContext->SupportsExplicitAsyncOperations());
                            assert(ASYNCCTXT_ISXFERCTXT(eDstContextType) || !pDstAsyncContext->SupportsExplicitAsyncOperations());
                            assert(eSrcContextType == ASYNCCTXT_XFERDTOD);
                            assert(eDstContextType == ASYNCCTXT_XFERDTOD);
                            pSrcAcc->DeviceToDeviceTransfer(pDstBuffer, pSrcBuffer, pSrcAsyncContext);
                            bViewsMaterialized = TRUE;

                        }

                    }
                }
            }
        }   

        if(bViewsMaterialized)
            SetCoherenceState(uiDestMemSpace, BSTATE_EXCLUSIVE);
        pBlockToClone->Unlock();
        if(bSourceLock) pSrcAcc->Unlock();
        if(bDstLock) pDestAcc->Unlock();
        if(m_pTemplate) {
            m_pTemplate->AddRef();
        }
        ctprofile_view_update_end(uiDestMemSpace, BSTATE_EXCLUSIVE, bTransferOccurred);
        CHECK_INVARIANTS();
        this->Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pushes a new control signal context. </summary>
    ///
    /// <remarks>   Crossbac, 2/14/2013. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    VOID 
    Datablock::PushControlSignalContext(
        CONTROLSIGNAL luiCode
        )
    {
        assert(LockIsHeld());
        m_vControlSignalStack.push(luiCode);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Pops the control signal context. </summary>
    ///
    /// <remarks>   Crossbac, 2/14/2013. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    CONTROLSIGNAL 
    Datablock::PopControlSignalContext(
        VOID
        ) {
        assert(LockIsHeld());
        assert(m_vControlSignalStack.size() > 0);
        if(m_vControlSignalStack.size() == 0) return DBCTLC_NONE;
        CONTROLSIGNAL code = m_vControlSignalStack.top();
        m_vControlSignalStack.pop();
        return code;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the block control code: protected, no lock required </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The block control code. </returns>
    ///-------------------------------------------------------------------------------------------------

    CONTROLSIGNAL 
    Datablock::__getControlSignals(
        VOID
        ) 
    {
        return m_vControlSignalStack.top();
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the block control code. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The block control code. </returns>
    ///-------------------------------------------------------------------------------------------------

    CONTROLSIGNAL 
    Datablock::GetControlSignals(
        VOID
        ) 
    {
        // Call with lock held!
        assert(LockIsHeld());
        return __getControlSignals();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Checks whether the block carries the given signal. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The block control code. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::TestControlSignal(
        CONTROLSIGNAL luiCode
        )
    {
        // Call with lock held!
        assert(LockIsHeld());
        return TESTSIGNAL(m_vControlSignalStack.top(), luiCode);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Checks whether the block carries the given signal. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The block control code. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::HasAnyControlSignal(
        VOID
        )
    {
        // Call with lock held!
        assert(LockIsHeld());
        return HASSIGNAL(m_vControlSignalStack.top());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Checks whether the block carries the BOF signal. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if the block control code carries the signal. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsBOF(
        VOID
        )
    {
        return TestControlSignal(DBCTLC_BOF);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Checks whether the block carries the EOF signal. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if the block control code carries the signal. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsEOF(
        VOID
        )
    {
        return TestControlSignal(DBCTLC_EOF);
    }
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Checks whether the block carries the begin iteration signal. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if the block control code carries the signal. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsBOI(
        VOID
        )
    {
        return TestControlSignal(DBCTLC_BEGINITERATION);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Checks whether the block carries the end iteration signal. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if the block control code carries the signal. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsEOI(
        VOID
        )
    {
        return TestControlSignal(DBCTLC_ENDITERATION);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   adds the signal to the block's control code: private lock-free version.</summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="code"> The code. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    VOID 
    Datablock::__setControlSignal(
        CONTROLSIGNAL luiCode
        )
    {
        assert(m_vControlSignalStack.size() > 0);
        CONTROLSIGNAL luiCurrentSignal = m_vControlSignalStack.top();
        m_vControlSignalStack.pop();
        luiCurrentSignal = (luiCurrentSignal | luiCode);
        m_vControlSignalStack.push(luiCurrentSignal);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   adds the signal to the block's control code. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="code"> The code. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    VOID 
    Datablock::SetControlSignal(
        CONTROLSIGNAL luiCode
        )
    {
        assert(LockIsHeld());
        __setControlSignal(luiCode);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Clears the control signal described by luiCode. </summary>
    ///
    /// <remarks>   Crossbac, 2/14/2013. </remarks>
    ///
    /// <param name="luiCode">  The lui code. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    VOID
    Datablock::__clearControlSignal(
        CONTROLSIGNAL luiCode 
        )         
    {
        assert(m_vControlSignalStack.size() > 0);
        CONTROLSIGNAL luiCurrentSignal = m_vControlSignalStack.top();
        m_vControlSignalStack.pop();
        luiCurrentSignal = (luiCurrentSignal & (~luiCode));
        m_vControlSignalStack.push(luiCurrentSignal);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   removes the signal from the block's control code. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="code"> The code. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    VOID 
    Datablock::ClearControlSignal(
        CONTROLSIGNAL luiCode
        )
    {
        assert(LockIsHeld());
        __clearControlSignal(luiCode);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   removes the signal from the block's control code. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="code"> The code. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    VOID 
    Datablock::ClearAllControlSignals(
        VOID
        )
    {
        assert(LockIsHeld());
        CONTROLSIGNAL allSigs = ~(DBCTLC_NONE);
        __clearControlSignal(allSigs);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this block carries a control token. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if control token, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Datablock::IsControlToken(
        VOID
        )
    {
        assert(LockIsHeld());
        BOOL bDataSizesEmpty = 
            this->m_cbFinalized[DBDATA_IDX] == 0 &&
            this->m_cbRequested[DBDATA_IDX] == 0;
        return 
            __getControlSignals() != DBCTLC_NONE &&
            bDataSizesEmpty;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this block is both a control and data block. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if control and data block, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Datablock::IsControlAndDataBlock(
        VOID
        )
    {
        assert(LockIsHeld());
        if(__getControlSignals() == DBCTLC_NONE)            
            return FALSE;
        UINT idx = DBDATA_IDX;
        BOOL bDataSizesEmpty = 
            this->m_cbFinalized[idx] == 0 &&
            this->m_cbRequested[idx] == 0;
        return HasValidChannels() || !bDataSizesEmpty;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Create all the buffers required for a given block to be referenced on the given
    ///             accelerator. This entails creating the data channel, and if metadata and template
    ///             channels are specified in the datablock template, then those device-specific
    ///             buffers need to be created as well. If the user has supplied initial data,
    ///             populate the accelerator- side buffers with the initial data, and mark the block
    ///             coherent on the accelerator.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pAccelerator">             [in] If non-null, an accelerator that will require a
    ///                                         view of this block. </param>
    /// <param name="pAsyncContext">            [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="bPopulate">                true to populate. </param>
    /// <param name="uiMinChannelBufferIdx">    Zero-based index of the minimum channel buffer. </param>
    /// <param name="uiMaxChannelBufferIdx">    Zero-based index of the maximum channel buffer. </param>
    ///
    /// <returns>   PTRESULT: PTASK_OK on success PTASK_ERR_EXISTS if called on a block with
    ///             materialized buffers.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::AllocateBuffers(
        __in Accelerator * pAccelerator,
        __in AsyncContext * pAsyncContext,
        __in BOOL bPopulate,
        __in UINT uiMinChannelBufferIdx,
        __in UINT uiMaxChannelBufferIdx
        )
    {
        assert(LockIsHeld());
        if(pAccelerator == NULL) {
            // we cannot create an acclerator-specific
            // buffer without known which acclerator!
            assert(pAccelerator != NULL);
            return PTASK_ERR_INVALID_PARAMETER;
        }
        UINT nValidHostChannels = 0;
        UINT nChannelIndex = 0;
        UINT nMemSpaceId = pAccelerator->GetMemorySpaceId();        
        BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[nMemSpaceId];
        BUFFER_MAP_ENTRY * pHostEntry = m_ppBufferMap[HOST_MEMORY_SPACE_ID];
        BOOL bActuallyPopulated = FALSE;
        BOOL bCanPopulate = bPopulate && 
                            nMemSpaceId != HOST_MEMORY_SPACE_ID &&
                            valid(pHostEntry->eState);

        ctprofile_view_update_start_nested_a(pAccelerator, CET_BUFFER_ALLOCATE);
        for(max(nChannelIndex=PTask::DBDATA_IDX, uiMinChannelBufferIdx);
            nChannelIndex<min(PTask::NUM_DATABLOCK_CHANNELS, uiMaxChannelBufferIdx+1);
            nChannelIndex++) {

            UINT uiTargetBufferSize;
            // how big should the allocation be? If the template is not existent
            // or we have a record-stream template, then we need to defer to the
            // explicitly set dimensions: take the 
            // max of the logical size and the allocated size. This is necessary
            // because we want buffers in all memory spaces to be of congruent size,
            // so if we've over-allocated in one memory space, then we must overallocate
            // in all spaces in case the block is part of a conservatively-sized pool
            // for example. If we have a fixed size template, then use it. 
            BOOL bNullTemplate = (m_pTemplate == NULL);
            BOOL bRecordStream = (m_pTemplate != NULL) && (m_pTemplate->DescribesRecordStream());
            BOOL bDataChannel  = (nChannelIndex == PTask::DBDATA_IDX);
            if(bNullTemplate || bRecordStream || !bDataChannel) {
                // the template does not completely describe the allocation
                // so we must look at the explicit size fields. 
                UINT uiLogical = m_bSealed ? m_cbFinalized[nChannelIndex] : m_cbRequested[nChannelIndex];
                UINT uiAllocated = m_cbAllocated[nChannelIndex];
                uiTargetBufferSize = max(uiLogical, uiAllocated);
                assert(uiAllocated == 0 || uiAllocated >= uiLogical);
            } else {
                // the template does describe the allocation. 
                uiTargetBufferSize = m_pTemplate->GetDatablockByteCount(nChannelIndex);
            }
            if(pEntry->pBuffers[nChannelIndex]) {
                PBuffer * pBuffer = pEntry->pBuffers[nChannelIndex];
                pEntry->pBuffers[nChannelIndex] = NULL;
                delete pBuffer;
            }

            // Note that a target buffer size of 0 does not mean we don't want to allocate
            // a buffer. We need to be able to bind objects to device-side resources even if those
            // objects represent datasets that are *logically* empty. Emit a warning if we are creating
            // such an empty buffer, but follow the state machine anyway. Note also that "populating"
            // an empty buffer can succeed as well. 
            // ---------------------------------------------
            // THIS ONLY APPLIES TO the data channel though! 
            
            if(uiTargetBufferSize != 0 || bDataChannel) {
                
                void * lpvInitialData = NULL;
                UINT cbInitialData = 0;
                BOOL bPinned = FALSE;

                if(bCanPopulate && pHostEntry->pBuffers[nChannelIndex]) {
                    assert(valid(pHostEntry->eState));
                    nValidHostChannels++;
                    PBuffer * pHostBuffer = pHostEntry->pBuffers[nChannelIndex];
                    lpvInitialData = pHostBuffer->GetBuffer();
                    cbInitialData = pHostBuffer->GetLogicalExtentBytes();
                    bPinned = pHostBuffer->IsPhysicalBufferPinned();
                    if(lpvInitialData != NULL) {
                        bActuallyPopulated = TRUE;
                    }
                }

                HOSTMEMORYEXTENT extent(lpvInitialData, cbInitialData, bPinned);
                AllocateBuffer(pAccelerator, 
                               pAsyncContext,
                               nChannelIndex,
                               cbInitialData, 
                               &extent);
            }
        }
        BUFFER_COHERENCE_STATE uiTargetState = bActuallyPopulated ? BSTATE_SHARED : BSTATE_INVALID;
        SetCoherenceState(nMemSpaceId, uiTargetState);
        ctprofile_view_update_end_nested(HOST_MEMORY_SPACE_ID, 
                                         uiTargetState, 
                                         ((nMemSpaceId != HOST_MEMORY_SPACE_ID) && bActuallyPopulated));
        return PTASK_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Create all the buffers required for a given block to be referenced on the given
    ///             accelerator. This entails creating the data channel, and if metadata and template
    ///             channels are specified in the datablock template, then those device-specific
    ///             buffers need to be created as well. If the user has supplied initial data,
    ///             populate the accelerator- side buffers with the initial data, and mark the block
    ///             coherent on the accelerator.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pAccelerator">     (optional) [in] If non-null, an accelerator that will require
    ///                                 a view of this block. </param>
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="bPopulate">        true to populate. </param>
    ///
    /// <returns>   PTRESULT: PTASK_OK on success PTASK_ERR_EXISTS if called on a block with
    ///             materialized buffers.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::AllocateBuffers(
        __in UINT nMemSpaceId,
        __in AsyncContext * pAsyncContext,
        __in BOOL bPopulate
        )
    {
        assert(LockIsHeld());
        ctprofile_view_update_active();
        Accelerator * pAccelerator = MemorySpace::GetAcceleratorFromMemorySpaceId(nMemSpaceId);
        return AllocateBuffers(pAccelerator, pAsyncContext, bPopulate);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this datablock has accelerator buffers already created for the given
    ///             accelerator.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. You must hold the lock on the block to call this function.
    ///             </remarks>
    ///
    /// <param name="pAccelerator"> (optional) [in] non-null, an accelerator. </param>
    ///
    /// <returns>   true if accelerator buffers, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::HasBuffers(
        __in Accelerator * pAccelerator
        )
    {
        assert(LockIsHeld());
        if(pAccelerator == NULL)
            return FALSE;
        UINT uiMemSpace = pAccelerator->GetMemorySpaceId();
        return HasBuffers(uiMemSpace);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this datablock has accelerator buffers
    /// 			already created for any memory space other than the host
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   true if accelerator buffers, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::HasDeviceBuffers(
        VOID
        )
    {
        assert(LockIsHeld());
        UINT uiMemSpaceID;
        for(uiMemSpaceID=HOST_MEMORY_SPACE_ID+1;
            uiMemSpaceID<MemorySpace::GetNumberOfMemorySpaces();
            uiMemSpaceID++) {

            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[uiMemSpaceID];
            for(UINT nChannelIndex=0; nChannelIndex<NUM_DATABLOCK_CHANNELS; nChannelIndex++) {
                if(pEntry->pBuffers[nChannelIndex] != NULL)
                    return TRUE;
            }
        }
        return FALSE;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this datablock has accelerator buffers already created for the given
    ///             accelerator.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. You must hold the lock on the block to call this function.
    ///             </remarks>
    ///
    /// <param name="pAccelerator"> (optional) [in] non-null, an accelerator. </param>
    ///
    /// <returns>   true if accelerator buffers, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::HasBuffers(
        __in UINT uiMemorySpaceID
        )
    {
        assert(LockIsHeld());
        assert(uiMemorySpaceID < MemorySpace::GetNumberOfMemorySpaces());
        if(m_ppBufferMap[uiMemorySpaceID]->eState != BSTATE_NO_ENTRY) {
            assert((m_ppBufferMap[uiMemorySpaceID]->pBuffers[DBDATA_IDX] != NULL) ||
                   (m_ppBufferMap[uiMemorySpaceID]->pBuffers[DBMETADATA_IDX] != NULL) ||
                   (m_ppBufferMap[uiMemorySpaceID]->pBuffers[DBTEMPLATE_IDX] != NULL));
            return TRUE;
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if the given block is out of date on the dispatch accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="pAccelerator">             [in,out] If non-null, the dispatch accelerator. </param>
    /// <param name="uiRequiredPermissions">    The required permissions. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL    
    Datablock::RequiresViewUpdate(
        Accelerator * pAccelerator,
        BUFFER_COHERENCE_STATE uiRequiredPermissions
        )
    {
        assert(pAccelerator != NULL);
        return RequiresViewUpdate(pAccelerator->GetMemorySpaceId(), uiRequiredPermissions);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if the given block is out of date on the dispatch accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="uiMemorySpaceID">          If non-null, the dispatch accelerator. </param>
    /// <param name="uiRequiredPermissions">    The required permissions. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL    
    Datablock::RequiresViewUpdate(
        UINT uiMemorySpaceID,
        BUFFER_COHERENCE_STATE uiRequiredPermissions
        )
    {
        assert(LockIsHeld());
        assert(uiMemorySpaceID < MemorySpace::GetNumberOfMemorySpaces());
        if(!HasBuffers(uiMemorySpaceID)) 
            return TRUE;

        BUFFER_MAP_ENTRY * pTargetEntry = m_ppBufferMap[uiMemorySpaceID];
        if(!valid(pTargetEntry->eState))
            return TRUE;

        if((pTargetEntry->eState == BSTATE_SHARED) && 
            (uiRequiredPermissions == BSTATE_EXCLUSIVE))
            return TRUE;
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets prefer page locked host views. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <param name="bPageLock">    true to lock, false to unlock the page. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::SetPreferPageLockedHostViews(
        BOOL bPageLock
        )
    {
        assert(LockIsHeld());
        m_bAttemptPinnedHostBuffers = bPageLock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets prefer page locked host views. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::GetPreferPageLockedHostViews(
        VOID
        )
    {
        assert(LockIsHeld());
        return m_bAttemptPinnedHostBuffers;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this block can generate device views with memset rather than 
    /// 			host-device transfers. </summary>
    ///
    /// <remarks>   crossbac, 7/6/2012. </remarks>
    ///
    /// <returns>   true if device views memsettable, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsDeviceViewsMemsettable(
        VOID
        )
    {
        assert(LockIsHeld());
        return m_bDeviceViewsMemsettable;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Updates the accelerator's view of this block. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pAccelerator">             [in,out] If non-null, the accelerator. </param>
    /// <param name="pAsyncContext">            [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="bPopulateView">            true if the caller wants the accelerator-side buffer
    ///                                         to have the most recent data contents for the block. Many
    ///                                         blocks are bound as outputs where the device-side code
    ///                                         performs only writes to the data, meaning we should not
    ///                                         attempt to transfer a more recent view before binding the
    ///                                         block and dispatching. </param>
    /// <param name="uiRequestedState">         The requested coherence state. This affects whether
    ///                                         other accelerator views require invalidation. </param>
    /// <param name="uiMinChannelBufferIdx">    Zero-based index of the minimum channel buffer. </param>
    /// <param name="uiMaxChannelBufferIdx">    Zero-based index of the maximum channel buffer. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::UpdateView(
        __in Accelerator * pAccelerator,
        __in AsyncContext * pAsyncContext,
        __in BOOL bPopulateView, 
        __in BUFFER_COHERENCE_STATE uiRequestedState,
        __in UINT uiMinChannelBufferIdx,
        __in UINT uiMaxChannelBufferIdx
        )
    {
        // Update the view of this datablock on the given accelerator, moving the coherence state to
        // that implied by the permissions being requested. Create any buffers required in the target
        // memory space if they do not exist, already, and ensure that those buffers contain the most
        // recent view of the Datablock's contents if "bPopulate" is set to true.
        // -----------------------------------------------------------------------------------------
        // When a task dispatches, it first calls "MigrateInputs" to start moving any data whose most
        // recent view is only available on another accelerator to the dispatch accelerator. It
        // subsequently calls Bind* to bind input and outputs, and ultimately dispatches the task. This
        // method should only be called at bind time (just before dispatch) by tasks that are making
        // sure they can safely bind accelerator code variables to device-side buffers that are up-to-
        // date. It should *not* be called during block migration. The reason is that migration works
        // by starting a copy from the source accelerator to the host, and bind operations expect to be
        // able to complete the any migrations implicitly because it expects to perform transfers only
        // from the host.
        // -----------------------
        // XXX:TODO:FIXME: allow arbitrary transfers. 
        assert(LockIsHeld());
        assert(pAccelerator != NULL);
        PTRESULT pr = PTASK_OK;

        BOOL bTransferOccurred = FALSE;
        UINT nMemSpace = pAccelerator->GetMemorySpaceId();
        UINT uiSrcMemSpace = HOST_MEMORY_SPACE_ID;  // a good prediction: will be updated below, but not changed..
        ctprofile_view_update_start(nMemSpace, CET_ACCELERATOR_VIEW_UPDATE);
        if(!HasBuffers(pAccelerator)) {

            // get the target memory space id, and if we don't have buffers
            // backing this block in that memory space, create the buffers,
            // populating the buffers with the most recent data as a side-effect.
            pr = AllocateBuffers(pAccelerator, 
                                 pAsyncContext, 
                                 bPopulateView,
                                 uiMinChannelBufferIdx,
                                 uiMaxChannelBufferIdx);

            // set the source memory space id. This is used only by the 
            // coherence profiler, and this is our best guess as to what source
            // memory space supplied the data if we populated the buffers,
            // which we might not even have done. In short, this is known 
            // to be potentially inaccurate, and may introduce noise in the statistics,
            // but does not affect correct behavior of the coherence state machine!
            uiSrcMemSpace = HOST_MEMORY_SPACE_ID; 

        } else {

            // If we already have buffers backing this block in the target memory space, re-use them. The
            // valid macro checks the state of the buffer map entry--if it is shared or exclusive, then it
            // is up-to-date with the most recent data (by construction), and there is nothing we need to
            // do. If it is not valid, we can go find the PBuffers for the space and populate them from
            // host memory *iff* the bPopulate flag is specified (it might not for write-only bindings).
            // -------------------------------------------------------------
            // NB: we assume that the host entry in the buffer map is valid. 
            BUFFER_MAP_ENTRY * pTargetEntry = m_ppBufferMap[nMemSpace];
            if(!valid(pTargetEntry->eState) && bPopulateView) {

                uiSrcMemSpace = HOST_MEMORY_SPACE_ID;
                BUFFER_MAP_ENTRY * pHostEntry = m_ppBufferMap[HOST_MEMORY_SPACE_ID];
                assert(pTargetEntry->eState != BSTATE_NO_ENTRY);    // that would mean no buffers!
                assert(nMemSpace != HOST_MEMORY_SPACE_ID);          // this method is not for updating the host view!
                assert(valid(pHostEntry->eState));                  // host assumed to be up-to-date
                
                UINT nChannelIndex;
                for(nChannelIndex=max(DBDATA_IDX, uiMinChannelBufferIdx); 
                    nChannelIndex<min(NUM_DATABLOCK_CHANNELS, uiMaxChannelBufferIdx+1);
                    nChannelIndex++) {

                    PBuffer * pHostBuffer = pHostEntry->pBuffers[nChannelIndex];
                    if(pHostBuffer == NULL) {
                        assert(nChannelIndex != DBDATA_IDX);
                        continue;
                    }

                    PBuffer * pTargetBuffer = GetPlatformBuffer(nMemSpace, nChannelIndex, pAccelerator);
                    if(pTargetBuffer != NULL) {
                        UINT uiXfer = pTargetBuffer->PopulateAcceleratorView(pAsyncContext, pHostBuffer);
                        bTransferOccurred = uiXfer > 0;
                        if(!bTransferOccurred)
                            pr = PTASK_ERR;
                    } else if(nChannelIndex == DBDATA_IDX) {
                        assert(false);
                        pr = PTASK_ERR;
                    }
                }
            }
        }

        // Required device buffers exist in the given memory space. now just set the coherence state
        // according the requested permissions. If shared state is requested, then the state for this
        // memory space's buffers will be set to shared. Exclusive request will invalidate all other
        // views, including the host. 
        SetCoherenceState(nMemSpace, uiRequestedState);
        ctprofile_view_update_end(uiSrcMemSpace, uiRequestedState, bTransferOccurred);
        return pr;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Propagate any control information in this datablock if the port is part of the
    ///             control routing network, and the block contains control signals.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. You must hold the lock on the block to call this. </remarks>
    ///
    /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::PropagateControlInformation(
        Port * pPort
        )
    {
        assert(LockIsHeld());
        assert(pPort != NULL);    
        CONTROLSIGNAL luiCurrentSignalWord = __getControlSignals();
        if(luiCurrentSignalWord != DBCTLC_NONE) {
            // if this block is marked with control signals, then  we need to signal any output ports that
            // are gated, and propagate the control code along any control propagation ports. 
            pPort->SignalGatedPorts();
            pPort->PropagateControlSignal(luiCurrentSignalWord);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the allocation size for the given channel based on
    /// 			ambient information about the datablock. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///
    /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
    ///
    /// <returns>   The channel allocation size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::GetChannelAllocationSizeBytes(
        UINT nChannelIndex
        )
    {
        UINT nBytes = 0;

        if(nChannelIndex == DBDATA_IDX && m_pTemplate != NULL) {

            // if we have a template, in the overwhelming majority of cases, the byte size is directly
            // derived from it. There is one exception: when the template carries initialization data that
            // is "memsettable". Consider a scenario where device-side code expects in input initialized to
            // all zeros. Particularly when buffers are large, it makes very little sense to allocate a
            // large zero-ed buffer just to memcpy it across the PCI bus, especially since some GPU
            // frameworks support memset APIs. In such cases we want to be able to have a template that
            // specifies what is needed to do the memset, but we still want the flexibility to deal with
            // variable length, strided inputs. Consequently, we provide a force init data size flag. If it
            // is true, we default to that. Otherwise, take the template dimensions. 
            
            if(m_bForceRequestedSize) {

                // technically the assert below is somewhat over-restrictive. However, I can't think of a
                // meaningful context in which to set up a memsettable template with size 0, and I can think of
                // a lot cases in which this could be accidently mis-set to 0. So consider it an error for now.

                nBytes = m_cbRequested[nChannelIndex];
                // assert(!((nBytes == 0) && m_pTemplate->GetDatablockByteCount(DBDATA_IDX) > 0));
            
            } else {

                // take the size specfied in the template unless it is a variable length template
                if(m_pTemplate->DescribesRecordStream()) {
                    return m_bSealed ? 
                        max(m_cbFinalized[nChannelIndex], m_cbAllocated[nChannelIndex]) :
                        max(m_cbRequested[nChannelIndex], m_cbAllocated[nChannelIndex]);
                } else {
                    UINT uiTemplateBytes = m_pTemplate->GetDatablockByteCount(nChannelIndex);
                    return m_bSealed ? 
                        max(m_cbFinalized[nChannelIndex], m_cbAllocated[nChannelIndex]) :
                        max(uiTemplateBytes, m_cbAllocated[nChannelIndex]);
                }
            }

        } else {

            nBytes = m_bSealed ? 
                        max(m_cbFinalized[nChannelIndex], m_cbAllocated[nChannelIndex]) :
                        max(m_cbRequested[nChannelIndex], m_cbAllocated[nChannelIndex]);
        }

        return nBytes;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the allocation size for the given channel based on
    /// 			ambient information about the datablock. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 1/6/2012. </remarks>
    ///
    /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
    ///
    /// <returns>   The channel allocation size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::GetChannelLogicalSizeBytes(
        UINT nChannelIndex
        )
    {
        UINT nBytes = 0;

        if(nChannelIndex == DBDATA_IDX && m_pTemplate != NULL) {

            // if we have a template, in the overwhelming majority of cases, the byte size is directly
            // derived from it. There is one exception: when the template carries initialization data that
            // is "memsettable". Consider a scenario where device-side code expects in input initialized to
            // all zeros. Particularly when buffers are large, it makes very little sense to allocate a
            // large zero-ed buffer just to memcpy it across the PCI bus, especially since some GPU
            // frameworks support memset APIs. In such cases we want to be able to have a template that
            // specifies what is needed to do the memset, but we still want the flexibility to deal with
            // variable length, strided inputs. Consequently, we provide a force init data size flag. If it
            // is true, we default to that. Otherwise, take the template dimensions. 
            
            if(m_bForceRequestedSize) {

                // technically the assert below is somewhat over-restrictive. However, I can't think of a
                // meaningful context in which to set up a memsettable template with size 0, and I can think of
                // a lot cases in which this could be accidently mis-set to 0. So consider it an error for now.

                nBytes = m_cbRequested[nChannelIndex];
                // assert(!((nBytes == 0) && m_pTemplate->GetDatablockByteCount(DBDATA_IDX) > 0));
            
            } else {

                // take the size specfied in the template unless it is a variable length template
                if(m_pTemplate->DescribesRecordStream()) {
                    return m_bSealed ? 
                        m_cbFinalized[nChannelIndex] :
                        m_cbRequested[nChannelIndex];
                } else {
                    UINT uiTemplateBytes = m_pTemplate->GetDatablockByteCount(nChannelIndex);
                    return m_bSealed ? 
                        m_cbFinalized[nChannelIndex] :
                        uiTemplateBytes;
                }
            }

        } else {

            nBytes = m_bSealed ? 
                        m_cbFinalized[nChannelIndex] :
                        m_cbRequested[nChannelIndex];
        }

        return nBytes;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'nChannelIndex' is template size overridden. </summary>
    ///
    /// <remarks>   crossbac, 7/9/2012. </remarks>
    ///
    /// <param name="nChannelIndex">    Zero-based index of the channel. </param>
    ///
    /// <returns>   true if template size overridden, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsTemplateSizeOverridden(
        UINT nChannelIndex
        )
    {
        if(m_pTemplate != NULL) {
            UINT uiTemplateCount = m_pTemplate->GetDatablockByteCount(nChannelIndex);
            UINT uiRequestCount = m_cbRequested[nChannelIndex];
            BOOL bDivergedSizes = uiRequestCount != uiTemplateCount;
            if(m_bSealed || m_bForceRequestedSize) 
                return bDivergedSizes;
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates an accelerator buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="nMemorySpaceId">       target memory space. </param>
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="nChannelIndex">        Zero-based index of the n channel. </param>
    /// <param name="cbAllocationSize_">    Size of the allocation. If default of 0 is given, use
    ///                                     sizes derived from the parent datablock. </param>
    /// <param name="pExtent">              (optional) [in] If non-null, initial data. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::AllocateBuffer(
        UINT nMemorySpaceId,
        AsyncContext * pAsyncContext,
        UINT nChannelIndex,
        UINT cbAllocationSize_,
        HOSTMEMORYEXTENT * pExtent
        )
    {
        ctprofile_view_update_active();
        Accelerator * pAccelerator = MemorySpace::GetAcceleratorFromMemorySpaceId(nMemorySpaceId);
        return AllocateBuffer(pAccelerator, 
                              NULL,
                              pAsyncContext,
                              nChannelIndex, 
                              cbAllocationSize_, 
                              pExtent);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates an accelerator buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="nMemorySpaceId">       target memory space. </param>
    /// <param name="pProxyAllocator">      [in,out] Identifier for the proxy memory space. </param>
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="nChannelIndex">        Zero-based index of the n channel. </param>
    /// <param name="cbAllocationSize_">    Size of the allocation. If default of 0 is given, use
    ///                                     sizes derived from the parent datablock. </param>
    /// <param name="pExtent">              (optional) [in] If non-null, initial data. </param>
    ///
    /// <returns>   . </returns>
    ///
    /// ### <param name="cbInitDataSize_">  Size of the initial data. If default of 0 is given, use
    ///                                     derived block size. </param>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::AllocateBuffer(
        UINT nMemorySpaceId,
        Accelerator * pProxyAllocator,
        AsyncContext * pAsyncContext,
        UINT nChannelIndex,
        UINT cbAllocationSize_,
        HOSTMEMORYEXTENT * pExtent
        )
    {
        ctprofile_view_update_active();
        Accelerator * pAccelerator = MemorySpace::GetAcceleratorFromMemorySpaceId(nMemorySpaceId);
        return AllocateBuffer(pAccelerator,  
                              pProxyAllocator,
                              pAsyncContext,
                              nChannelIndex, 
                              cbAllocationSize_, 
                              pExtent);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates an accelerator buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pAccelerator">         (optional) [in] If non-null, an accelerator that will
    ///                                     require a view of this block. </param>
    /// <param name="pAsyncContext">        [in,out] Size of the allocation. If default of 0 is given,
    ///                                     use sizes derived from the parent datablock. </param>
    /// <param name="nChannelIndex">        Zero-based index of the n channel. </param>
    /// <param name="cbAllocationSize_">    Size of the allocation. </param>
    /// <param name="pExtent">              (optional) [in] If non-null, initial data. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::AllocateBuffer(
        Accelerator * pAccelerator,
        AsyncContext * pAsyncContext,
        UINT nChannelIndex,
        UINT cbAllocationSize_,
        HOSTMEMORYEXTENT * pExtent
        )
    {
        ctprofile_view_update_active();
        return AllocateBuffer(pAccelerator,
                              NULL,
                              pAsyncContext,
                              nChannelIndex,
                              cbAllocationSize_,
                              pExtent);
    }

#if 0

    // this turns out to be identical to the non-pinned version!

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a pinned backing buffer for this block, if possible. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="uiMemorySpaceId">      destination memory space--required to be HOST!. </param>
    /// <param name="pProxyAllocator">      [in,out] Identifier for the proxy allocator memory space. </param>
    /// <param name="pAsyncContext">        [in,out] Zero-based index of the user interface channel. </param>
    /// <param name="nChannelIndex">        Size of the allocation. </param>
    /// <param name="cbAllocationSize_">    (optional) [in] If non-null, initial data. </param>
    /// <param name="pExtent">              [in,out] If non-null, the extent. </param>
    ///
    /// <returns>   PTRESULT--use PTFAIL/PTSUCCEED macros to interpret. </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::AllocatePinnedBuffer(
        __in UINT uiMemorySpaceId, 
        __in Accelerator * pProxyAllocator,
        __in AsyncContext * pAsyncContext,
        __in UINT nChannelIndex, 
        __in UINT cbAllocationSize_, 
        __in HOSTMEMORYEXTENT * pExtent
        )
    {
        PTRESULT pr;
        assert(m_bAttemptPinnedHostBuffers);
        ctprofile_view_update_active();
        assert(uiMemorySpaceId == HOST_MEMORY_SPACE_ID);
        assert(pProxyAllocator != NULL);
        assert(!pProxyAllocator->IsHost());
        assert(pProxyAllocator->SupportsPinnedHostMemory());
        if(uiMemorySpaceId != HOST_MEMORY_SPACE_ID) return PTASK_ERR_INVALID_PARAMETER;
        if(pProxyAllocator == NULL || pProxyAllocator->IsHost()) return PTASK_ERR_INVALID_PARAMETER;
        if(!pProxyAllocator->SupportsPinnedHostMemory()) return PTASK_ERR_INVALID_PARAMETER;

        void * lpvInitData = pExtent ? pExtent->lpvAddress : NULL;
        UINT cbInitDataSize_ = pExtent ? pExtent->uiSizeBytes : 0;
        BOOL bInitExtentPinned = pExtent ? pExtent->bPinned : FALSE;

        UINT uiDerivedChannelSize = GetChannelAllocationSizeBytes(nChannelIndex);
        UINT uiLogicalChannelSize  = GetChannelLogicalSizeBytes(nChannelIndex);
        UINT cbAllocationSize = cbAllocationSize_ ? cbAllocationSize_ : uiDerivedChannelSize;
        UINT cbInitDataSize = cbInitDataSize_ ? cbInitDataSize_ : uiLogicalChannelSize;
        assert(LockIsHeld());
        char * pActualInitValue = (char*) lpvInitData;
        BOOL bActualInitValuePinned = bInitExtentPinned;

        m_cbAllocated[nChannelIndex] = cbAllocationSize;
        if(uiLogicalChannelSize != cbInitDataSize && lpvInitData != NULL) {                
            pActualInitValue = reinterpret_cast<char*>(
                MemorySpace::AllocateMemoryExtent(HOST_MEMORY_SPACE_ID, 
                                                    uiLogicalChannelSize,
                                                    0));
            if(uiLogicalChannelSize > cbInitDataSize) {
                // copy the initial value into an init buffer
                // as many times as it will fit. 
                assert(!this->m_bForceRequestedSize);
                for(UINT i=0; i<uiLogicalChannelSize; i+=cbInitDataSize) {
                    UINT cbCopy = min(cbInitDataSize, uiLogicalChannelSize-i);
                    memcpy(&pActualInitValue[i], lpvInitData, cbCopy);
                }
            } else {
                // copy as much of the initial value as
                // will fit into the actual init buffer
                memcpy(pActualInitValue, lpvInitData, uiLogicalChannelSize);
            }
        } 

        UINT cbInstantiation = pActualInitValue == NULL ? 0 : uiLogicalChannelSize;      
        HOSTMEMORYEXTENT extent(pActualInitValue, cbInstantiation, bActualInitValuePinned);
        pr = InstantiateChannel(HOST_MEMORY_SPACE_ID, 
                                pAsyncContext, 
                                pProxyAllocator,
                                nChannelIndex, 
                                &extent,
                                true, 
                                false);

        if(uiLogicalChannelSize != cbInitDataSize && lpvInitData != NULL) {
            MemorySpace::DeallocateMemoryExtent(HOST_MEMORY_SPACE_ID, pActualInitValue);
        }
        return pr;
    }
#endif

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates an accelerator buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pAccelerator">         (optional) [in] If non-null, an accelerator that will
    ///                                     require a view of this block. </param>
    /// <param name="pProxyAllocator">      [in,out] If non-null, the proxy allocator. </param>
    /// <param name="pAsyncContext">        [in,out] Size of the allocation. If default of 0 is given,
    ///                                     use sizes derived from the parent datablock. </param>
    /// <param name="nChannelIndex">        Zero-based index of the n channel. </param>
    /// <param name="cbAllocationSize_">    Size of the allocation. </param>
    /// <param name="pExtent">              (optional) [in] If non-null, initial data. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::AllocateBuffer(
        Accelerator * pAccelerator,
        Accelerator * pProxyAllocator,
        AsyncContext * pAsyncContext,
        UINT nChannelIndex,
        UINT cbAllocationSize_,
        HOSTMEMORYEXTENT * pExtent
        )
    {
        PTRESULT pr;
        ctprofile_view_update_active();
        void * lpvInitData = pExtent ? pExtent->lpvAddress : NULL;
        UINT cbInitDataSize_ = pExtent ? pExtent->uiSizeBytes : 0;
        BOOL bInitExtentPinned = pExtent ? pExtent->bPinned : FALSE;

        UINT uiDerivedChannelSize = GetChannelAllocationSizeBytes(nChannelIndex);
        UINT uiLogicalChannelSize  = GetChannelLogicalSizeBytes(nChannelIndex);
        UINT cbAllocationSize = cbAllocationSize_ ? cbAllocationSize_ : uiDerivedChannelSize;
        UINT cbInitDataSize = cbInitDataSize_ ? cbInitDataSize_ : uiLogicalChannelSize;
        assert(LockIsHeld());
        char * pActualInitValue = (char*) lpvInitData;
        BOOL bActualInitValuePinned = bInitExtentPinned;

        m_cbAllocated[nChannelIndex] = cbAllocationSize;
        if(uiLogicalChannelSize != cbInitDataSize && lpvInitData != NULL) {                
            pActualInitValue = reinterpret_cast<char*>(
                MemorySpace::AllocateMemoryExtent(HOST_MEMORY_SPACE_ID, 
                                                    uiLogicalChannelSize,
                                                    0));
            if(uiLogicalChannelSize > cbInitDataSize) {
                // copy the initial value into an init buffer
                // as many times as it will fit. 
                assert(!this->m_bForceRequestedSize);
                for(UINT i=0; i<uiLogicalChannelSize; i+=cbInitDataSize) {
                    UINT cbCopy = min(cbInitDataSize, uiLogicalChannelSize-i);
                    memcpy(&pActualInitValue[i], lpvInitData, cbCopy);
                }
            } else {
                // copy as much of the initial value as
                // will fit into the actual init buffer
                memcpy(pActualInitValue, lpvInitData, uiLogicalChannelSize);
            }
        } 

        UINT cbInstantiation = pActualInitValue == NULL ? 0 : uiLogicalChannelSize;      
        HOSTMEMORYEXTENT extent(pActualInitValue, cbInstantiation, bActualInitValuePinned);
        pr = InstantiateChannel(pAccelerator, 
                                pAsyncContext, 
                                pProxyAllocator,
                                nChannelIndex, 
                                &extent,
                                pActualInitValue != NULL, 
                                false);

        if(uiLogicalChannelSize != cbInitDataSize && lpvInitData != NULL) {
            MemorySpace::DeallocateMemoryExtent(HOST_MEMORY_SPACE_ID, pActualInitValue);
        }
        return pr;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a PBuffer in the accelerator's memory space for the given channel based
    ///             on what we know about the datablock.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pAccelerator">         (optional) [in] If non-null, an accelerator that will
    ///                                     require a view of this block. </param>
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="pProxyAccelerator">    [in,out] If non-null, the proxy accelerator. </param>
    /// <param name="nChannelIndex">        Zero-based index of the n channel. </param>
    /// <param name="pExtent">              (optional) [in] If non-null, initial data. </param>
    /// <param name="bUpdateCoherenceMap">  true to update coherence map. </param>
    /// <param name="bRaw">                 true to raw. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::InstantiateChannel(
        __in Accelerator * pAccelerator,
        __in AsyncContext * pAsyncContext,
        __in Accelerator * pProxyAccelerator,
        __in UINT nChannelIndex,
        __in HOSTMEMORYEXTENT * pExtent,
        __in bool bUpdateCoherenceMap,
        __in bool bRaw
        )
    {
        // call with m_lock held!
        PTRESULT pr = PTASK_ERR;
        assert(LockIsHeld());        
        ctprofile_view_update_active();
        PBuffer * pBuffer = NULL;
        void * lpvInitData = pExtent ? pExtent->lpvAddress : NULL;

        BOOL bBufferlessBlock = (m_pTemplate != NULL &&
                                 m_pTemplate->DescribesScalarParameter() && 
                                 pAccelerator->SupportsByvalArguments());

        if(bBufferlessBlock) {
            // if this is a scalar parameter, we don't want to create a device-side buffer to back it,
            // since we will wind up wasting that storage. this case is for underlying runtimes that
            // support direct manipulation of kernel function parameters (i.e. CUDA, OpenCL). So if we have
            // an initial value for a bufferless block, this better be a request for materialization in
            // host memory. 
            assert(lpvInitData != NULL || pAccelerator->IsHost());
        }

        if(pAsyncContext == NULL || !pAsyncContext->SupportsExplicitAsyncOperations())
            pAsyncContext = pAccelerator->GetAsyncContext(ASYNCCTXT_XFERHTOD);

        if(pAccelerator->IsHost() && m_bAttemptPinnedHostBuffers) {
            pBuffer = pAccelerator->CreatePagelockedBuffer(pAsyncContext,
                                                           this, 
                                                           nChannelIndex, 
                                                           m_eBufferAccessFlags,
                                                           pProxyAccelerator,
                                                           pExtent,
                                                           NULL,  
                                                           bRaw);
        } else {
            pBuffer = pAccelerator->CreateBuffer(pAsyncContext,
                                                 this, 
                                                 nChannelIndex, 
                                                 m_eBufferAccessFlags,
                                                 pProxyAccelerator,
                                                 pExtent,
                                                 NULL,  
                                                 bRaw);
        }

        assert(pBuffer != NULL);
        if(pBuffer != NULL) {
            UINT nMemorySpaceId = pAccelerator->GetMemorySpaceId();
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[nMemorySpaceId];
            assert(pEntry->pBuffers[nChannelIndex] == NULL);
            pEntry->pBuffers[nChannelIndex] = pBuffer;

#if 0
			m_cbAllocated[nChannelIndex] = max(m_cbAllocated[nChannelIndex], pBuffer->GetAllocationExtentBytes());
#endif

            if(!bUpdateCoherenceMap || lpvInitData == NULL) {
                SetCoherenceState(nMemorySpaceId, BSTATE_INVALID);
            } else {
                // Caller is telling us that the initial data represents the most
                // up-to-date view of the data, and wants us to set the coherence 
                // state to reflect that. Of course, if we haven't been given an
                // initial buffer value, that has to mean the state is invalid. Otherwise
                // the new state depends on whether there are other valid copies. 
                // Fortunately, the SetCoherenceState implementation will upgrade
                // our shared request if there are no other copies. 
                SetCoherenceState(nMemorySpaceId, BSTATE_SHARED);
            }
            pr = PTASK_OK;
        }
        CHECK_INVARIANTS();
        return pr;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a PBuffer in the memory space for the given channel based on what we know
    ///             about the datablock.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="nMemorySpace">         [in] target memory space. </param>
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="pProxyAllocator">      [in,out] The proxy memory space. </param>
    /// <param name="nChannelIndex">        Zero-based index of the n channel. </param>
    /// <param name="pExtent">              (optional) [in] If non-null, initial data. </param>
    /// <param name="bUpdateCoherenceMap">  true to update coherence map. </param>
    /// <param name="bRaw">                 true to raw. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::InstantiateChannel(
        __in UINT nMemorySpace,
        __in AsyncContext * pAsyncContext,
        __in Accelerator * pProxyAllocator,
        __in UINT nChannelIndex,
        __in HOSTMEMORYEXTENT * pExtent,
        __in bool bUpdateCoherenceMap,
        __in bool bRaw
        )
    {
        assert(LockIsHeld()); 
        ctprofile_view_update_active();
        if(MemorySpace::HasStaticAllocator(nMemorySpace)) {
            // PTask::Runtime::Inform("static allocator bypased for memory space in InstantiateChannel!");
        }
        Accelerator * pAccelerator = MemorySpace::GetAcceleratorFromMemorySpaceId(nMemorySpace);
        assert(pAccelerator != NULL);
        return InstantiateChannel(pAccelerator, 
                                  pAsyncContext, 
                                  pProxyAllocator,
                                  nChannelIndex, 
                                  pExtent,
                                  bUpdateCoherenceMap, 
                                  bRaw);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this block is up-to-date on the given accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pAccelerator"> (optional) [in] non-null, an accelerator. </param>
    ///
    /// <returns>   true if coherent, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsCoherent(
        Accelerator * pAccelerator
        )
    {
        assert(LockIsHeld());
        UINT nMemSpaceId = pAccelerator->GetMemorySpaceId();
        BUFFER_COHERENCE_STATE state = m_ppBufferMap[nMemSpaceId]->eState;
        return valid(state);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Updates the host memspace view of this block. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="pAsyncContext">        [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="uiRequestedState">     The requested coherence state. This affects whether other
    ///                                     accelerator views require invalidation. </param>
    /// <param name="bForceSynchronous">    The force synchronous. </param>
    /// <param name="bAcquireLocks">        The acquire locks. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::SynchronizeHostView(
        __in AsyncContext * pAsyncContext,
        __in BUFFER_COHERENCE_STATE uiRequestedState,
        __in BOOL bForceSynchronous,
        __in BOOL bAcquireLocks
        )
    {
        MARKRANGEENTER(L"SynchronizeHostView");
        PTRESULT pt = PTASK_OK;
        BOOL bTransferOccurred = FALSE;
        assert(LockIsHeld());
        ctprofile_view_update_start(HOST_MEMORY_SPACE_ID, CET_HOST_VIEW_UPDATE);
        Accelerator * pAccelerator = GetMostRecentAccelerator();
        UINT uiMemSpace = pAccelerator->GetMemorySpaceId();
        if(uiMemSpace == HOST_MEMORY_SPACE_ID) {
            // we should be already sync'd!
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[uiMemSpace];
            if(!valid(pEntry->eState)) {
                pt = PTASK_ERR;
                PTask::Runtime::HandleError("%s called to sync host to host with invalid host view!",
                                            __FUNCTION__);
            } else {
                PTask::Runtime::Warning("Datablock::SynchronizeHostView called to sync host to host!");
                SetCoherenceState(uiMemSpace, uiRequestedState);
            }
        } else {
            // perform the sync 
            pt = SynchronizeViews(HOST_MEMORY_SPACE_ID, 
                                  uiMemSpace, 
                                  pAsyncContext,
                                  uiRequestedState, 
                                  bAcquireLocks,
                                  bForceSynchronous);
            bTransferOccurred = PTSUCCESS(pt);
        }
        ctprofile_view_update_end(uiMemSpace, uiRequestedState, bTransferOccurred);
        CHECK_INVARIANTS();
        MARKRANGEEXIT();
        return pt;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Locks the block and if the lock is being acquired for a view synchronization,
    ///             lock the most recent accelerator and any view sync target provided. This allows
    ///             us to order lock acquire correctly (accelerators before datablocks)
    ///             when deciding whether to produce downstream views after dispatch.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/2/2013. </remarks>
    ///
    /// <param name="bLockForViewSynchronization">  The lock for view synchronization. </param>
    /// <param name="pTargetAccelerator">           (optional) [in,out] If non-null, target
    ///                                             accelerator. </param>
    ///
    /// <returns>   the most recent accelerator, locked, if there is one. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator *
    Datablock::LockForViewSync(
        __in BOOL bLockForViewSynchronization,
        __in Accelerator * pTargetAccelerator
        )
    {
        Accelerator * pMostRecentAcc = NULL;
        if(bLockForViewSynchronization) {

            Lock();
            // figure out if we have a producer to lock...this requires
            // grabbing a lock on the datablock.
            pMostRecentAcc = GetMostRecentAccelerator();            
            assert(pMostRecentAcc == NULL || pMostRecentAcc != pTargetAccelerator);
            Unlock();

            BOOL bMostRecentLockRequired = (pMostRecentAcc != NULL) && !pMostRecentAcc->IsHost();
            BOOL bTargetLockRequired = (pTargetAccelerator != NULL) && !pTargetAccelerator->IsHost();
            BOOL bTargetIsRecent = (pMostRecentAcc == pTargetAccelerator);
            UINT uiRequiredLockCount = (bMostRecentLockRequired?1:0) + (bTargetLockRequired&&!bTargetIsRecent?1:0);
            UNREFERENCED_PARAMETER(uiRequiredLockCount);

            std::deque<Accelerator*> vLockList;
            if(bMostRecentLockRequired) 
                vLockList.push_front(pMostRecentAcc);
            if(bTargetLockRequired) {
                if(bMostRecentLockRequired) {
                    UINT uiTargetSpace = pTargetAccelerator->GetMemorySpaceId();
                    UINT uiRecentSpace = pMostRecentAcc->GetMemorySpaceId();
                    if(uiTargetSpace == uiRecentSpace) {
                        // super weird case, but tolerable. Verify that we 
                        // are dealing with the same non-host accelerator objects. 
                        assert(pTargetAccelerator == pMostRecentAcc);
                        assert(vLockList.size() == 1);
                        assert(uiTargetSpace != HOST_MEMORY_SPACE_ID);
                    } else if(uiTargetSpace < uiRecentSpace) {
                        vLockList.push_front(pTargetAccelerator);
                    } else {
                        assert(uiTargetSpace > uiRecentSpace);
                        vLockList.push_back(pTargetAccelerator);
                    }
                } else {
                    vLockList.push_front(pTargetAccelerator);
                }
            }
            assert(vLockList.size() == uiRequiredLockCount);
            std::deque<Accelerator*>::iterator di;
            for(di=vLockList.begin(); di!=vLockList.end(); di++) {
                assert(!LockIsHeld() || (*di)->LockIsHeld());
                (*di)->Lock();
            }
        }

        // we've got any accelerator locks we need if we are locking for a view sync.
        // so go ahead and lock the block.
        Lock();

        // do we still have the most recent accelerator?
        // If we don't, and this *is* an attempt to acquire view synchronization
        // locks we've got trouble: somehow the block state changed while
        // we were acquiring accelerator locks.
        if(bLockForViewSynchronization && pMostRecentAcc != GetMostRecentAccelerator()) {
            assert(!bLockForViewSynchronization || pMostRecentAcc == GetMostRecentAccelerator());
            Runtime::HandleError("%s: block state changed during accelerator lock acquire!\n",
                                 __FUNCTION__);
        }
        return pMostRecentAcc;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unlocks the block and if the lock was acquired for a view synchronization,
    ///             unlock the most recent accelerator and any view sync target provided. This
    ///             allows us to order lock acquire correctly (accelerators before datablocks)
    ///             when deciding whether to produce downstream views after dispatch.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/2/2013. </remarks>
    ///
    /// <param name="bLockForViewSynchronization">  The lock for view synchronization. </param>
    /// <param name="pTargetAccelerator">           (optional) [in,out] If non-null, target
    ///                                             accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::UnlockForViewSync(
        __in BOOL bLockForViewSynchronization,
        __in Accelerator * pMostRecentAcc,
        __in Accelerator * pTargetAccelerator
        )
    {
        if(bLockForViewSynchronization) {
            if(pMostRecentAcc != NULL && !pMostRecentAcc->IsHost()) {
                assert(pMostRecentAcc->LockIsHeld());
                pMostRecentAcc->Unlock();
            }
            if(pTargetAccelerator != NULL && 
               !pTargetAccelerator->IsHost() && 
                pTargetAccelerator != pMostRecentAcc) {
                assert(pTargetAccelerator->LockIsHeld());
                pTargetAccelerator->Unlock();
            }
        }
        assert(LockIsHeld());
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synchronizes views with the given memory spaces. If locks are not already held,
    ///             then acquire them. Since we order datablock lock acquisition after accelerator,
    ///             we have to release our locks on this datablock and re-acquire them. Hence it is
    ///             *extremely important* to use this member only if no datablock state has been
    ///             modified yet with the lock held. Use only if you know what you're doing! Or think
    ///             you do...
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///
    /// <param name="uiDestMemorySpaceId">      Identifier for the memory space. </param>
    /// <param name="uiSourceMemorySpaceId">    Identifier for the source memory space. </param>
    /// <param name="pAsyncContext">            [in,out] If non-null, context for the source
    ///                                         asynchronous. </param>
    /// <param name="uiRequestedState">         target coherence state for the requested copy. </param>
    /// <param name="bLocksRequired">           false if caller already holds locks. </param>
    /// <param name="bForceSynchronous">        The force synchronous. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::SynchronizeViews(
        __in UINT uiDestMemorySpaceId,
        __in UINT uiSourceMemorySpaceId,
        __in AsyncContext * pOpAsyncContext,
        __in BUFFER_COHERENCE_STATE uiRequestedState,
        __in BOOL bLocksRequired,
        __in BOOL bForceSynchronous
        )
    {
        // Materialize a view of the accelerator-side buffer in pSrc into the memory domain of the
        // pDest accelerator. If neither memory space is the host memory space, then copy through
        // the host memory space before copying to the destination. But first, some sanity checks:
        assert(LockIsHeld());
        assert(valid(uiRequestedState)); 
        ctprofile_view_update_active();
        MARKRANGEENTER(L"SynchronizeViews");

        // find the accelerators involved.
        set<Accelerator*, compare_accelerators_by_memspace_id> * vaccs = NULL;
        Accelerator * pDest = MemorySpace::GetAcceleratorFromMemorySpaceId(uiDestMemorySpaceId);
        Accelerator * pSrc = MemorySpace::GetAcceleratorFromMemorySpaceId(uiSourceMemorySpaceId);        
        BOOL bPeerToPeerTransferSupport = pSrc->SupportsDeviceToDeviceTransfer(pDest);
        assert(pDest != pSrc);

        if(!bLocksRequired) {
            // if the caller said no locks, at least try to verify
            // that the caller possibly knows what they are doing.
            assert(pDest->LockIsHeld() || pDest->IsHost());
            assert(pSrc->LockIsHeld() || pSrc->IsHost());
            assert(pDest != pSrc);
        
        } else {

            if(!pDest->IsHost() || !pSrc->IsHost()) {

                // if the source or destination are not host accelerators then we really do require an ordered
                // lock acquire, where the ordering with the datablock lock is the sticking point. If we do not
                // have the locks we need, to get the locks we will have to drop the datablock lock which risks
                // changes to the state that got us here in the first place. We used to take our chances and
                // release the block so we could acquire the accelerators first, but that is not tenable. If we
                // hit this case without the locks we need, we have a serious problem and will declare a
                // runtime error. 
               
                BOOL bRequireKeepDatablockLock = FALSE;
                if(bRequireKeepDatablockLock) {

                    assert(FALSE);
                    PTask::Runtime::MandatoryInform("%s:locks-required for non-host accelerators: lock ordering violation\n", __FUNCTION__);
                    PTask::Runtime::HandleError("%s: locks-required for non-host accelerators: lock ordering violation\n", __FUNCTION__);
                    MARKRANGEEXIT();
                    return PTASK_ERR;

                } else {

                    set<Accelerator*, compare_accelerators_by_memspace_id>::iterator vi;
                    vaccs = new set<Accelerator*, compare_accelerators_by_memspace_id>();
                    vaccs->insert(pDest);
                    vaccs->insert(pSrc);

                    PTask::Runtime::MandatoryInform("%s: dropping datablock lock to avoid lock ordering violation\n", __FUNCTION__);
                    UINT nLockDepth = GetLockDepth();           
                    for(UINT i=0; i<nLockDepth; i++) Unlock();            
                    for(vi=vaccs->begin(); vi!=vaccs->end(); vi++) {
                        Accelerator * pAccelerator = *vi;
                        if(pAccelerator->GetMemorySpaceId() != HOST_MEMORY_SPACE_ID)
                            pAccelerator->Lock();
                    }
                    for(UINT i=0; i<nLockDepth; i++) Lock();
                }
            }
        }

        BOOL bSourceIsHost = uiSourceMemorySpaceId == PTask::HOST_MEMORY_SPACE_ID;
        BOOL bDestIsHost = uiDestMemorySpaceId == PTask::HOST_MEMORY_SPACE_ID;
        BOOL bP2PTransfer = bPeerToPeerTransferSupport && !(bSourceIsHost || bDestIsHost);
        BOOL bTransferThroughHost = (!(bSourceIsHost || bDestIsHost) && 
                                     !bPeerToPeerTransferSupport);

        BUFFER_MAP_ENTRY * pSourceEntry = m_ppBufferMap[uiSourceMemorySpaceId];
        
        if(!valid(pSourceEntry->eState)) {
            assert(valid(pSourceEntry->eState));
            PTask::Runtime::HandleError("%s: attempt to sync from invalid source (memspace=%d) to dest(mem=%d)!\n",
                                        __FUNCTION__,
                                        uiSourceMemorySpaceId,
                                        uiDestMemorySpaceId);
            MARKRANGEEXIT();
            return PTASK_ERR;
        }

        UINT nChannelIndex;
        for(nChannelIndex=PTask::DBDATA_IDX;
            nChannelIndex!=PTask::NUM_DATABLOCK_CHANNELS;
            nChannelIndex++) {            

            PBuffer * pDstBuffer = NULL;
            PBuffer * pSrcBuffer = pSourceEntry->pBuffers[nChannelIndex];
            if(pSrcBuffer != NULL) {

                if(bP2PTransfer) {

                    // if we are transferring through the host, we copy from the source
                    // to host memory, then from host memory to the destination. All
                    // three versions should now be shared copies. 

                    if(!m_ppBufferMap[uiDestMemorySpaceId]->pBuffers[nChannelIndex]) {
                        AllocateBuffer(uiDestMemorySpaceId, NULL, nChannelIndex, NULL, 0);
                    }
                    pDstBuffer = GetPlatformBuffer(uiDestMemorySpaceId, nChannelIndex, pDest);
                    pSrc->DeviceToDeviceTransfer(pDstBuffer, pSrcBuffer, pOpAsyncContext);

                } else {

                    // we need the host buffer no matter what. If it's not the
                    // source or the destination, then its a waypoint.
                    Accelerator * pProxyAllocator = NULL;
                    if(!bSourceIsHost && pSrc->SupportsPinnedHostMemory())
                        pProxyAllocator = pSrc;
                    else if(!bDestIsHost && pDest->SupportsPinnedHostMemory()) 
                        pProxyAllocator = pDest;
                    assert(pProxyAllocator == NULL || pProxyAllocator->LockIsHeld());

                    if(!m_ppBufferMap[HOST_MEMORY_SPACE_ID]->pBuffers[nChannelIndex]) 
                        AllocateBuffer(HOST_MEMORY_SPACE_ID, pProxyAllocator, NULL, nChannelIndex);

                    PBuffer * pHostBuffer = GetPlatformBuffer(HOST_MEMORY_SPACE_ID, nChannelIndex, pSrc);
                    UINT cbAllocation = GetChannelAllocationSizeBytes(nChannelIndex);
                    HOSTMEMORYEXTENT hostExtent(pHostBuffer->GetBuffer(),
                                                pHostBuffer->GetLogicalExtentBytes(),
                                                pHostBuffer->IsPhysicalBufferPinned());                
                    if(bTransferThroughHost) {

                        // if we are transferring through the host, we copy from the source
                        // to host memory, then from host memory to the destination. All
                        // three versions should now be shared copies. 
                        
                        BOOL bAllocRequired = !m_ppBufferMap[uiDestMemorySpaceId]->pBuffers[nChannelIndex];
                        pSrcBuffer->PopulateHostView(pOpAsyncContext, pHostBuffer, bForceSynchronous);
                        
                        if(bAllocRequired) {

                            // the allocate buffer call is going to try to use the data in the
                            // host extent we just started populating above. If that call is
                            // asynchronous, we need to be sure the copy back is finished
                            // before we allow the allocation to use it as an initial value.
                            // ------------------------
                            // CJR: we are allocating with null async contexts now
                            // pHostBuffer->WaitOutstandingDependences(pDestContext);

                            AllocateBuffer(uiDestMemorySpaceId,  // destination memspace
                                           pProxyAllocator,      // in case source or dest requires special alloc
                                           pOpAsyncContext,      // async context for this operation
                                           nChannelIndex,
                                           cbAllocation,
                                           &hostExtent);

                        } else {

                            pDstBuffer = GetPlatformBuffer(uiDestMemorySpaceId, nChannelIndex, pDest);
                            pDstBuffer->PopulateAcceleratorView(pOpAsyncContext, pHostBuffer);

                        }

                    } else if(bSourceIsHost) {

                        // if we are transferring from the host, perform the copy. 
                        // both versions should be in shared state unless the request
                        // was for exclusive permissions. 
                        pDstBuffer = GetPlatformBuffer(uiDestMemorySpaceId, nChannelIndex, pDest);
                        if(!m_ppBufferMap[uiDestMemorySpaceId]->pBuffers[nChannelIndex]) 
                            AllocateBuffer(uiDestMemorySpaceId, pProxyAllocator, NULL, nChannelIndex);
                        pDstBuffer->PopulateAcceleratorView(pOpAsyncContext, pHostBuffer);     

                    } else if(bDestIsHost) {          

                        // if we are transferring to the host, perform the copy. 
                        // both versions should be in shared state unless the request
                        // was for exclusive permissions. 
                        AsyncContext * pContext = pOpAsyncContext ? 
                            pOpAsyncContext : pSrc->GetAsyncContext(ASYNCCTXT_XFERDTOH);
                        MARKRANGEENTER(L"PopulateHostView");
                        pSrcBuffer->PopulateHostView(pContext, pHostBuffer, bForceSynchronous);
                        MARKRANGEEXIT();

                    } else {
                        // shouldn't be possible to get here.    
                        assert(false);
                    }
                }
            }
        }

        // Release our accelerator locks: no longer needed.
        if(bLocksRequired) {
            set<Accelerator*, compare_accelerators_by_memspace_id>::iterator vi;            
            for(vi=vaccs->begin(); vi!=vaccs->end(); vi++) {
                Accelerator * pAccelerator = *vi;
                if(pAccelerator->GetMemorySpaceId() != HOST_MEMORY_SPACE_ID)
                    pAccelerator->Unlock();
            }
        }

        // now update the coherence state based on the route the 
        // data took and the requested permissions. 
        SetCoherenceState(uiDestMemorySpaceId, uiRequestedState);

        // clean up!
        if(vaccs != NULL) {
            MARKRANGEENTER(L"delete-vaccs");
            delete vaccs;
            MARKRANGEEXIT();
        }

        CHECK_INVARIANTS();
        MARKRANGEEXIT();
        return PTASK_OK;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bring views of this block up-to-date on both accelerators. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pDest">                    [in] non-null, destination accelerator. </param>
    /// <param name="pSrc">                     [in] non-null, source accelerator. </param>
    /// <param name="pAsyncContext">            [in,out] If non-null, context for the asynchronous. </param>
    /// <param name="uiRequestedPermissions">   The requested permissions for the migrated view. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::Migrate(
        __inout Accelerator * pDest, 
        __in    Accelerator * pSrc,
        __inout AsyncContext * pAsyncContext,
        __in    BUFFER_COHERENCE_STATE uiRequestedPermissions
        )
    {
        // materialize a view of the accelerator-side buffer in pSrc into the memory domain of the
        // pDest accelerator. it should be sufficient to materialize a host-side view and then mark the
        // block incoherent, so that the next binding at dispatch time will force a copy to the
        // dispatch accelerator. Locks should already be held for both accelerators. 
        
        assert(LockIsHeld());
        assert(pDest->LockIsHeld());
        assert(pSrc->LockIsHeld());
        assert(pDest != pSrc);

        BOOL bDXContext = pDest->GetClass() == ACCELERATOR_CLASS_DIRECT_X &&
                          pSrc->GetClass() == ACCELERATOR_CLASS_DIRECT_X;

        if(bDXContext && false) {

            // synchrony in the direct x API forces us to handle
            // this case differently to avoid holding locks on all
            // the accelerators in the system for the duration
            // of migration. Sync to the host and then release
            // the source lock so we can make forward progress. 
            // Subsequent call to Bind* will force the second host->dest
            // update.
           
            PTRESULT pt;
            UINT uiSourceMemorySpaceId = pSrc->GetMemorySpaceId();
            ctprofile_view_update_start(HOST_MEMORY_SPACE_ID, CET_ACCELERATOR_VIEW_UPDATE);
            pt = SynchronizeViews(HOST_MEMORY_SPACE_ID, 
                                 uiSourceMemorySpaceId,
                                 pAsyncContext,
                                 uiRequestedPermissions, 
                                 FALSE,
                                 FALSE);
            ctprofile_view_update_end(uiSourceMemorySpaceId, uiRequestedPermissions, PTSUCCESS(pt));
            CHECK_INVARIANTS();
            return pt;

        } else {

            PTRESULT pt;
            UINT uiSourceMemorySpaceId = pSrc->GetMemorySpaceId();
            UINT uiDestMemorySpaceId = pDest->GetMemorySpaceId();
            ctprofile_view_update_start(uiDestMemorySpaceId, CET_ACCELERATOR_VIEW_UPDATE);
            pt = SynchronizeViews(uiDestMemorySpaceId, 
                                 uiSourceMemorySpaceId,
                                 pAsyncContext,
                                 uiRequestedPermissions, 
                                 FALSE,
                                 // FALSE);
                                 bDXContext);
            ctprofile_view_update_end(uiSourceMemorySpaceId, uiRequestedPermissions, PTSUCCESS(pt));
            CHECK_INVARIANTS();
            return pt;

        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   invalidate sharers. </summary>
    ///
    /// <remarks>   Crossbac, 9/20/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::AcquireExclusive(
        Accelerator * pAccelerator
        )
    {
        assert(LockIsHeld());
        assert(pAccelerator != NULL);
        if(pAccelerator == NULL) return FALSE;
        UINT nMemorySpaceId = pAccelerator->GetMemorySpaceId();
        BUFFER_MAP_ENTRY * pOE = m_ppBufferMap[nMemorySpaceId];
        assert(pOE->eState != BSTATE_NO_ENTRY); // no buffer to acquire
        if(pOE->eState == BSTATE_NO_ENTRY) return FALSE;
        ctprofile_view_update_start(nMemorySpaceId, CET_BIND_OUTPUT);
        SetCoherenceState(nMemorySpaceId, BSTATE_EXCLUSIVE);
        ctprofile_view_update_end(nMemorySpaceId, BSTATE_EXCLUSIVE, NOXFER);
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Set the coherence state for the given accelerator's view of this block. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="pAccelerator">     (optional) [in] If non-null, an accelerator that will require
    ///                                 a view of this block. </param>
    /// <param name="uiCoherenceState"> State of the coherence. </param>
    ///-------------------------------------------------------------------------------------------------

    void                    
    Datablock::SetCoherenceState(
        Accelerator * pAccelerator, 
        BUFFER_COHERENCE_STATE uiCoherenceState
        )
    {
        UINT nMemorySpaceId = pAccelerator->GetMemorySpaceId();
        SetCoherenceState(nMemorySpaceId, uiCoherenceState);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Set the coherence state for the given memory space. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="pAccelerator">     (optional) [in] If non-null, an accelerator that will require
    ///                                 a view of this block. </param>
    /// <param name="uiCoherenceState"> State of the coherence. </param>
    ///-------------------------------------------------------------------------------------------------

    void                    
    Datablock::SetCoherenceState(
        UINT nMemorySpaceId, 
        BUFFER_COHERENCE_STATE uiCoherenceState
        )
    {
        BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[nMemorySpaceId];
        switch(uiCoherenceState) {

        case BSTATE_NO_ENTRY:
            // there is never a reason to do this through
            // a public API--no entry is not a real state.
            pEntry->eState = uiCoherenceState;
            assert(false);
            break;

        case BSTATE_INVALID:
            // invalidating a single entry needn't
            // impact any other entries.
            pEntry->eState = uiCoherenceState;
            break;

        case BSTATE_SHARED: {
            // all valid versions need to be marked shared.
            for(UINT i=HOST_MEMORY_SPACE_ID; i<MemorySpace::GetNumberOfMemorySpaces(); i++) {
                BUFFER_MAP_ENTRY * pOE = m_ppBufferMap[i];
                if(i==nMemorySpaceId) 
                    continue;
                if(pOE->eState != BSTATE_INVALID &&
                    pOE->eState != BSTATE_NO_ENTRY)  {
                    pOE->eState = uiCoherenceState;
                }
            }
            if(pEntry->eState != BSTATE_EXCLUSIVE) {
                // avoid a needless permissions downgrade.
                pEntry->eState = uiCoherenceState;
            }
            break;
        }

        case BSTATE_EXCLUSIVE: {
            // all valid versions need to be invalidated except the
            // one which we are giving exclusive permissions to.
            for(UINT i=HOST_MEMORY_SPACE_ID; i<MemorySpace::GetNumberOfMemorySpaces(); i++) {
                if(i==nMemorySpaceId) 
                    continue;
                BUFFER_MAP_ENTRY * pOE = m_ppBufferMap[i];
                if(pOE->eState != BSTATE_NO_ENTRY)  {
                    pOE->eState = BSTATE_INVALID;
                }
            }
            pEntry->eState = uiCoherenceState;
            break;    
        }
        }
        CHECK_INVARIANTS();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   marks any buffers in valid state as invalid. </summary>
    ///
    /// <remarks>   crossbac, 4/29/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::Invalidate(
        VOID
        )
    {
        assert(LockIsHeld());
        UINT nMemSpaces = MemorySpace::GetNumberOfMemorySpaces();
        
        for(UINT i=HOST_MEMORY_SPACE_ID; i<nMemSpaces; i++) {
            
            BUFFER_MAP_ENTRY * pOE = m_ppBufferMap[i];
            BUFFER_COHERENCE_STATE uiCoherenceState = pOE->eState;

            if(uiCoherenceState != BSTATE_NO_ENTRY) {
                assert(pOE->pBuffers[DBDATA_IDX] != NULL);
                pOE->eState = BSTATE_INVALID;
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Check invariants for the coherence state machine. Any number of INVALID and
    ///             NO_ENTRY copies are allows. There can be 0..* SHARED copies and 0 EXCLUSIVE, or 1
    ///             EXCLUSIVE copy and 0 SHARED copies.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::CheckInvariants(
        VOID
        )
    {
#if DEBUG
        assert(LockIsHeld());
        UINT nExclusiveCopies = 0;
        UINT nSharedCopies = 0;
        UINT nChannelIndex = 0;
        UINT nMemSpaces = MemorySpace::GetNumberOfMemorySpaces();
        
        for(UINT i=HOST_MEMORY_SPACE_ID; i<nMemSpaces; i++) {
            
            BUFFER_MAP_ENTRY * pOE = m_ppBufferMap[i];
            BUFFER_COHERENCE_STATE uiCoherenceState = pOE->eState;

            if(uiCoherenceState == BSTATE_NO_ENTRY) {
                // there should be no pbuffer objects on any channel.
                for(nChannelIndex=DBDATA_IDX; 
                    nChannelIndex<NUM_DATABLOCK_CHANNELS; 
                    nChannelIndex++) {
                    assert(pOE->pBuffers[nChannelIndex] == NULL);
                }
            } else {
                // there should be pbuffer objects for at least the data channel,
                // otherwise the state really should be NO_ENTRY. This holds
                // for INVALID entries too--invalid means out-of-date buffers.
                UINT nValidBuffers = 0;
                assert(pOE->pBuffers[DBDATA_IDX] != NULL);
                for(nChannelIndex=DBDATA_IDX; 
                    nChannelIndex<NUM_DATABLOCK_CHANNELS; 
                    nChannelIndex++) {
                    if(pOE->pBuffers[nChannelIndex] != NULL)
                        nValidBuffers++;
                }
                assert(nValidBuffers != 0);
            }

            // count up number of copies in various states.
            switch(uiCoherenceState) {
            case BSTATE_NO_ENTRY:   break;
            case BSTATE_INVALID:    break;
            case BSTATE_SHARED:     nSharedCopies++;    break;
            case BSTATE_EXCLUSIVE:  nExclusiveCopies++; break;
            }
        }
        
        BOOL bCorrectSharedState    = (nSharedCopies >= 0 && nExclusiveCopies == 0);
        BOOL bCorrectExclusiveState = (nSharedCopies == 0 && nExclusiveCopies == 1);
        assert(bCorrectSharedState || bCorrectExclusiveState);
#endif
    }

    // TODO JC Documentation.

    void 
    Datablock::ForceExistingViewsValid(
        VOID
        )
    {
        assert(LockIsHeld());
        UINT uiValidViews = 0;
        for(UINT i=HOST_MEMORY_SPACE_ID; i<MemorySpace::GetNumberOfMemorySpaces(); i++) {
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[i];
            if(pEntry->pBuffers[0] != NULL) 
                uiValidViews++;
        }
        BUFFER_COHERENCE_STATE eTargetState = uiValidViews > 1 ? BSTATE_SHARED : BSTATE_EXCLUSIVE;
        for(UINT i=HOST_MEMORY_SPACE_ID; i<MemorySpace::GetNumberOfMemorySpaces(); i++) {
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[i];
            if(pEntry->pBuffers[0] != NULL) 
                pEntry->eState = eTargetState;
        }
    }

#ifdef FORCE_CORRUPT_NON_AFFINITIZED_VIEWS
    // TODO JC Documentation.

    void 
    Datablock::ForceCorruptNonAffinitizedViews(
        Accelerator * pAcc
        ) 
    {
        if(!pAcc) return;
        assert(LockIsHeld());
        UINT uiValidViews = 0;
        UINT uiAffinSpaceId = pAcc->GetMemorySpaceId();
        for(UINT i=HOST_MEMORY_SPACE_ID; i<MemorySpace::GetNumberOfMemorySpaces(); i++) {
            if(i == uiAffinSpaceId) continue;
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[i];
            if(pEntry->pBuffers[0] != NULL) {
                // LEAK
                pEntry->pBuffers[0] = (PBuffer*) 0xDEADBEEF;
                //pEntry->eState = BSTATE_NO_ENTRY; 
            }
        }
    }
#endif

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the accelerators whose memory space contains the most up-to-date
    /// 			view of this datablock. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <returns>   number of valid view accelerators. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT           
    Datablock::GetValidViewAccelerators(
        std::vector<Accelerator*> &accs
        )
    {
        assert(LockIsHeld());
        for(UINT i=HOST_MEMORY_SPACE_ID; i<MemorySpace::GetNumberOfMemorySpaces(); i++) {
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[i];
            if(valid(pEntry->eState)) {
                accs.push_back(MemorySpace::GetAcceleratorFromMemorySpaceId(i));
            } 
        }
        return (UINT)accs.size();
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the last accelerator writer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the last accelerator writer. </returns>
    ///-------------------------------------------------------------------------------------------------

    Accelerator * 
    Datablock::GetMostRecentAccelerator(
        VOID
        )
    {
        assert(LockIsHeld());
        Accelerator * pBestGuess = NULL;
        for(UINT i=HOST_MEMORY_SPACE_ID; i<MemorySpace::GetNumberOfMemorySpaces(); i++) {
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[i];
            if(pEntry->eState == BSTATE_EXCLUSIVE) {
                return MemorySpace::GetAcceleratorFromMemorySpaceId(i);
            } else if(pEntry->eState == BSTATE_SHARED) {
                pBestGuess = MemorySpace::GetAcceleratorFromMemorySpaceId(i);
            }
        }
        // we found someone with a shared copy. 
        // that's good enough for what the caller wants,
        // which is a way to get a fresh version of the block 
        // contents.
        return pBestGuess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the last async context. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the last accelerator writer. </returns>
    ///-------------------------------------------------------------------------------------------------

    AsyncContext * 
    Datablock::GetMostRecentAsyncContext(
        VOID
        )
    {
        assert(LockIsHeld());
        Accelerator * pAccelerator = GetMostRecentAccelerator();
        Task * pLastTask = m_pProducerTask;
        if(pAccelerator != NULL && pLastTask != NULL) {
            return pLastTask->GetOperationAsyncContext(pAccelerator, ASYNCCTXT_TASK);
        }
        return NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a pointer to host memory containing the most recent version of the specified
    ///             channel buffer for this block.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="uiChannelIndex">               Zero-based index of the channel. </param>
    /// <param name="bWritable">                    true if a writable version is required, which
    ///                                             necessitates invalidating all other views of this
    ///                                             block. </param>
    /// <param name="bSynchronizeBufferContents">   (optional) the update view. </param>
    ///
    /// <returns>   null if it fails, else the data pointer. </returns>
    ///-------------------------------------------------------------------------------------------------

    void * 
    Datablock::GetChannelBufferPointer(
        __in UINT uiChannelIndex,
        __in BOOL bWriteable,
        __in BOOL bSynchronizeBufferContents
        )
    {
        assert(LockIsHeld());
        void * pHostMemory = NULL;
        BOOL bDispatchContext = FALSE;
        BOOL bUpdateCoherence = TRUE;
        BOOL bMaterializeAbsent = TRUE;
        BOOL bSynchronousConsumer = TRUE;
        BUFFER_COHERENCE_STATE uiRequestedState = 
            bWriteable ? BSTATE_EXCLUSIVE : BSTATE_SHARED;

        std::set<Accelerator*>::iterator si;
        std::set<Accelerator*> vLockAccs;
        ASYNCHRONOUS_OPTYPE eOpType = bWriteable ? OT_MEMCPY_TARGET : OT_MEMCPY_SOURCE;        
        GetCurrentOutstandingAsyncAccelerators(HOST_MEMORY_SPACE_ID, eOpType, vLockAccs);
        
        for(si=vLockAccs.begin(); si!=vLockAccs.end(); si++) 
            (*si)->Lock();

        pHostMemory = GetHostChannelBuffer(NULL,
                                           uiChannelIndex,
                                           uiRequestedState,
                                           bMaterializeAbsent,
                                           bSynchronizeBufferContents,
                                           bUpdateCoherence,
                                           bDispatchContext,
                                           bSynchronousConsumer);

        for(si=vLockAccs.begin(); si!=vLockAccs.end(); si++) 
            (*si)->Unlock();

        return pHostMemory;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a pointer to the data channel in this datablock. The contract is that we
    ///             will always return something here, so if the channel is not backed by any buffers
    ///             anywhere yet, we must create a host side view.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/9/2012. </remarks>
    ///
    /// <param name="bWriteable">   true if writeable. </param>
    ///
    /// <returns>   null if it fails, else the data pointer. </returns>
    ///-------------------------------------------------------------------------------------------------

    void *
    Datablock::GetDataPointer(
        __in BOOL bWriteable,
        __in BOOL bSynchronizeBufferContents
        )
    {
        return GetChannelBufferPointer(DBDATA_IDX, bWriteable, bSynchronizeBufferContents);
    }

    /////-------------------------------------------------------------------------------------------------
    ///// <summary>   Gets the data channel host-side buffer. </summary>
    /////
    ///// <remarks>   Crossbac, 12/28/2011. </remarks>
    /////
    ///// <param name=""> The. </param>
    /////
    ///// <returns>   null if it fails, else the data buffer. </returns>
    /////-------------------------------------------------------------------------------------------------

    //void* 
    //Datablock::GetDataChannelBuffer(
    //    BOOL bWriteable                
    //    )
    //{
    //    assert(LockIsHeld());
    //    void* pHostBuffer = NULL;
    //    ASYNCHRONOUS_OPTYPE eOpType = bWriteable ? OT_MEMCPY_TARGET : OT_MEMCPY_SOURCE;
    //    if(HasOutstandingAsyncDependences(NULL, eOpType)) {
    //        std::set<Accelerator*>* pAccs = GetOutstandingAsyncAccelerators(NULL, eOpType);
    //        if(pAccs != NULL) {
    //            std::set<Accelerator*>::iterator si;
    //            for(si=pAccs->begin(); si!=pAccs->end(); si++) {
    //                (*si)->Lock();
    //            }
    //            pHostBuffer = GetHostChannelBuffer(NULL, DBDATA_IDX, bWriteable, TRUE);
    //            for(si=pAccs->begin(); si!=pAccs->end(); si++) {
    //                (*si)->Unlock();
    //            }
    //            return pHostBuffer;
    //        }
    //    }  
    //    return GetHostChannelBuffer(NULL, DBDATA_IDX, bWriteable, TRUE);
    //}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a data buffer size. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The data buffer size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::GetDataBufferLogicalSizeBytes(
        VOID
        )
    {
        assert(LockIsHeld());
        return GetChannelLogicalSizeBytes(PTask::DBDATA_IDX);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a data buffer size. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The data buffer size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::GetDataBufferAllocatedSizeBytes(
        VOID
        )
    {
        assert(LockIsHeld());
        return m_cbAllocated[PTask::DBDATA_IDX];
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Grow datablock channel buffers all at once. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="uiNewDataSizeBytes">       New size of the data buffer. </param>
    /// <param name="uiNewMetaSizeBytes">       The new meta size in bytes. </param>
    /// <param name="uiNewTemplateSizeBytes">   The new template size in bytes. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::GrowBuffers(
        __in UINT uiNewDataSizeBytes, 
        __in UINT uiNewMetaSizeBytes, 
        __in UINT uiNewTemplateSizeBytes
        ) 
    {
        assert(LockIsHeld());
        assert(!m_bSealed);
        UINT nChannelIndex = 0;
        UINT uiChannelSizes[NUM_DATABLOCK_CHANNELS];
        uiChannelSizes[DBDATA_IDX] = uiNewDataSizeBytes;       
        uiChannelSizes[DBMETADATA_IDX] = uiNewMetaSizeBytes;
        uiChannelSizes[DBTEMPLATE_IDX] = uiNewTemplateSizeBytes;
        BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[HOST_MEMORY_SPACE_ID];    

        if(pEntry->eState == BSTATE_NO_ENTRY) {

            // the buffers for this block have not actually 
            // been allocated yet. So it is our job to allocate them.
            // Since this is not a realloc, there is no copying required
            
            assert(NULL == pEntry->pBuffers[DBDATA_IDX]);        
            assert(NULL == pEntry->pBuffers[DBMETADATA_IDX]);        
            assert(NULL == pEntry->pBuffers[DBTEMPLATE_IDX]);
            for(nChannelIndex=DBDATA_IDX;
                nChannelIndex<=DBTEMPLATE_IDX;
                nChannelIndex++) {
                m_cbRequested[nChannelIndex] = uiChannelSizes[nChannelIndex];
                assert(!HasValidChannel(nChannelIndex));
                assert(pEntry->pBuffers[nChannelIndex] == NULL);
                if(pEntry->pBuffers[nChannelIndex] != NULL) {
                    delete pEntry->pBuffers[nChannelIndex];
                    pEntry->pBuffers[nChannelIndex] = NULL;
                }
                ctprofile_view_update_start(HOST_MEMORY_SPACE_ID, CET_GROW_BUFFER);
                InstantiateChannel(HOST_MEMORY_SPACE_ID,
                                    NULL,
                                    NULL,
                                    nChannelIndex,
                                    NULL,
                                    FALSE, 
                                    TRUE);
                ctprofile_view_update_end(HOST_MEMORY_SPACE_ID, BSTATE_EXCLUSIVE, NOXFER);

                GetPlatformBuffer(HOST_MEMORY_SPACE_ID, nChannelIndex);
                assert(!m_bSealed || (m_cbAllocated[nChannelIndex] >= m_cbRequested[nChannelIndex]));
                assert(!m_bSealed || (m_cbAllocated[nChannelIndex] >= uiChannelSizes[nChannelIndex]));
            }

        } else {

            // at least one buffer exists already for this block. 
            // we need to realloc any existing buffers for which the new 
            // size exceeds the old size (including copying previous data).
            // If a buffer already exists which is sufficiently large, leave it.

            BUFFER_COHERENCE_STATE oState = pEntry->eState;
            for(nChannelIndex=DBDATA_IDX;
                nChannelIndex<=DBTEMPLATE_IDX;
                nChannelIndex++) {

                UINT uiNewSize = uiChannelSizes[nChannelIndex];
                UINT uiOldSize = m_cbRequested[nChannelIndex];
                PBuffer * pOldBuffer = pEntry->pBuffers[nChannelIndex];        
                if(pOldBuffer != NULL && uiOldSize >= uiNewSize)
                    continue;  // no action needed for this buffer. 

                pEntry->pBuffers[nChannelIndex] = NULL;
                m_cbRequested[nChannelIndex] = uiNewSize;
                ctprofile_view_update_start(HOST_MEMORY_SPACE_ID, CET_GROW_BUFFER);
                InstantiateChannel(HOST_MEMORY_SPACE_ID,
                                   NULL,
                                   NULL,
                                   nChannelIndex, 
                                   NULL,
                                   FALSE,
                                   TRUE);
                ctprofile_view_update_end(HOST_MEMORY_SPACE_ID, pEntry->eState, NOXFER);
                m_cbAllocated[nChannelIndex] = uiNewSize;

                m_bBlockResized = TRUE;
                if(pOldBuffer != NULL) {
                    PBuffer * pHostBuffer = GetPlatformBuffer(HOST_MEMORY_SPACE_ID, nChannelIndex);
                    void * pOldData = pOldBuffer->GetBuffer();
                    memcpy(pHostBuffer->GetBuffer(), pOldData, min(uiOldSize, uiNewSize));
                    delete pOldBuffer;
                }

                // potentially, buffers exist in other memory spaces already--this is often the 
                // case when a block is drawn from a block pool. There are some interesting issues of
                // policy to decide when this is the case. Should a block leave its pool if it grows? 
                // Probably, since pools are created to meet particular size needs. If we wind up with
                // disparate-size blocks hiding in a pool with a particular size, its not hard to imagine
                // memory pressure getting out of hand in a way that is difficult to diagnose. Second, 
                // when we do resize a block, should those existing device-space buffers be resized too
                // (which will be faster when we do actually need them) or just invalidated and freed
                // (which will be slower but eliminates needless memory allocation activity). 
                
                for(UINT uiOtherMemSpace=1; 
                    uiOtherMemSpace<MemorySpace::GetNumberOfMemorySpaces(); 
                    uiOtherMemSpace++) {
                    BUFFER_MAP_ENTRY * pOtherEntry = m_ppBufferMap[uiOtherMemSpace];
                    assert(pOtherEntry->eState != BSTATE_EXCLUSIVE); // otherwise we need to sync!
                    PBuffer * pBuffer = pOtherEntry->pBuffers[nChannelIndex];
                    if(pBuffer != NULL) {
                        switch(PTask::Runtime::GetBlockResizeMemorySpacePolicy()) {
                        case BRSMSP_RELEASE_DEVICE_BUFFERS: 
                            pOtherEntry->pBuffers[nChannelIndex] = NULL;
                            pOtherEntry->eState = BSTATE_NO_ENTRY;
                            break;
                        case BRSMSP_GROW_DEVICE_BUFFERS: {
                            UINT uiOldAllocSize = pBuffer->GetAllocationExtentBytes();
                            ctprofile_view_update_start(uiOtherMemSpace, CET_GROW_BUFFER);
                            InstantiateChannel(uiOtherMemSpace,
                                               NULL,
                                               NULL,
                                               nChannelIndex, 
                                               NULL,
                                               FALSE,
                                               TRUE);
                            ctprofile_view_update_end(uiOtherMemSpace, pEntry->eState, NOXFER);
                            PBuffer * pNewBuffer = pOtherEntry->pBuffers[nChannelIndex];
                            pBuffer->Copy(pNewBuffer, pBuffer, NULL, uiOldAllocSize);
                            break; }
                        }
                        delete pBuffer;
                    }
                }

                assert(!m_bSealed || (m_cbAllocated[nChannelIndex] >= m_cbRequested[nChannelIndex]));
                assert(!m_bSealed || (m_cbAllocated[nChannelIndex] >= uiNewSize));

            }
            pEntry->eState = oState;
        } 

        BOOL bSuccess = 
            (((pEntry->pBuffers[DBDATA_IDX] != NULL || (m_cbRequested[DBDATA_IDX] == 0)) && (m_cbRequested[DBDATA_IDX] >= uiNewDataSizeBytes)) &&
                ((pEntry->pBuffers[DBMETADATA_IDX] != NULL || (m_cbRequested[DBMETADATA_IDX] == 0)) && (m_cbRequested[DBMETADATA_IDX] >= uiNewMetaSizeBytes)) &&
                ((pEntry->pBuffers[DBTEMPLATE_IDX] != NULL || (m_cbRequested[DBTEMPLATE_IDX] == 0)) && (m_cbRequested[DBTEMPLATE_IDX] >= uiNewTemplateSizeBytes)));

        CHECK_INVARIANTS();

        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Grow a buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="newSize">  New size of the data buffer. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::GrowBuffer(
        UINT nChannelIndex,
        UINT newSize
        )
    {
        assert(LockIsHeld());
        assert(!m_bSealed);
        BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[HOST_MEMORY_SPACE_ID];
        PBuffer * pOldBuffer = pEntry->pBuffers[nChannelIndex];        
        if(pEntry->eState == BSTATE_NO_ENTRY || pOldBuffer == NULL) {
            m_cbRequested[nChannelIndex] = newSize;
            if(!HasValidChannel(nChannelIndex)) {
                ctprofile_view_update_start(HOST_MEMORY_SPACE_ID, CET_GROW_BUFFER);
                InstantiateChannel(HOST_MEMORY_SPACE_ID,
                                   NULL,
                                   NULL,
                                   nChannelIndex,
                                   NULL,
                                   FALSE, 
                                   TRUE);
                ctprofile_view_update_end(HOST_MEMORY_SPACE_ID, BSTATE_EXCLUSIVE, NOXFER);
            }
            GetPlatformBuffer(HOST_MEMORY_SPACE_ID, nChannelIndex);
            assert(!m_bSealed || (m_cbAllocated[nChannelIndex] >= m_cbRequested[nChannelIndex]));
            assert(!m_bSealed || (m_cbAllocated[nChannelIndex] >= newSize));
        } else {
            if(pOldBuffer != NULL) {
                UINT uiOldSize = m_cbRequested[nChannelIndex];
                BUFFER_COHERENCE_STATE oState = pEntry->eState;
                pEntry->pBuffers[nChannelIndex] = NULL;
                m_cbRequested[nChannelIndex] = newSize;
                ctprofile_view_update_start(HOST_MEMORY_SPACE_ID, CET_GROW_BUFFER);
                InstantiateChannel(HOST_MEMORY_SPACE_ID,
                                   NULL,
                                   NULL,
                                   nChannelIndex, 
                                   NULL,
                                   FALSE,
                                   TRUE);
                ctprofile_view_update_end(HOST_MEMORY_SPACE_ID, pEntry->eState, NOXFER);
                PBuffer * pHostBuffer = GetPlatformBuffer(HOST_MEMORY_SPACE_ID, nChannelIndex);
                void * pOldData = pOldBuffer->GetBuffer();
                memcpy(pHostBuffer->GetBuffer(), pOldData, min(uiOldSize, newSize));
                pEntry->eState = oState;
                delete pOldBuffer;
                assert(!m_bSealed || (m_cbAllocated[nChannelIndex] >= m_cbRequested[nChannelIndex]));
                assert(!m_bSealed || (m_cbAllocated[nChannelIndex] >= newSize));
            }
        }
        CHECK_INVARIANTS();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Grow data buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="newSize">  New size of the data buffer. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::GrowDataBuffer(
        UINT newSize
        )
    {
        GrowBuffer(DBDATA_IDX, newSize);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Synthesize a metadata channel based on template information.
    ///             
    ///             This is specialized support for Dandelion:
    ///             ------------------------------------------
    ///             Dandelion expects a per-record object size entry in the meta-data channel. This
    ///             expectation is probably obsolete, since we are currently restricted to fixed-
    ///             stride objects in Dandelion. However, to accommodate that expectation, we provide
    ///             a method to populate the metadata channel of a block from its datablock template.
    ///             We simply compute the record count based on the stride and buffer sizes, and
    ///             write the num_record copies of the stride in a new metadata channel.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 4/26/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    [in,out] If non-null, context for the asynchronous. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    PTRESULT
    Datablock::SynthesizeMetadataFromTemplate(
        __in AsyncContext * pAsyncContext
        )
    {
        Lock();
        PTRESULT pt = PTASK_OK;
        if(!HasValidChannel(DBMETADATA_IDX) || (m_bPooledBlock && !HasValidChannel(DBTEMPLATE_IDX))) {

            // Only do this if there is not already a valid metadata channel. Otherwise synthesizing can
            // overwrite existing valid metadata. Synthesize by writing RecordCount copies of the stride
            // from the template. If there is no template for this block we infer the stride from the
            // actual size of the data channel. If there is already a record count, make sure it matches
            // what we're going to create. In general, we only want to do this for blocks allocated on meta
            // ports. Note that we are also willing to do this for pooled blocks, since the existence of a 
            // meta data buffer channel does not necessarily imply a valid meta-data channel. As a sanity-check,
            // make sure the template channel *is* empty, to avoid clobbering a collaborative allocation result.

            //assert(IsSealed()); 
            UINT uiBytes = 0;
            UINT uiStride = 0;
            if(m_pTemplate != NULL) {
                uiBytes = m_pTemplate->GetDatablockByteCount(DBDATA_IDX);
                uiStride = m_pTemplate->GetStride();
                UINT uiTemplateRecords = uiBytes / uiStride;
				if(m_uiRecordCount != uiTemplateRecords) {
					BOOL bOverrideableDimensions = m_pTemplate->IsVariableDimensioned() || 
												   m_pTemplate->DescribesRecordStream();
					assert(m_uiRecordCount == 0 || bOverrideableDimensions);
					if(!bOverrideableDimensions) {						
						m_uiRecordCount = uiTemplateRecords;
					}
				}
            } else {
                assert(m_uiRecordCount != 0);
                uiBytes = m_cbFinalized[DBDATA_IDX];
                uiStride = uiBytes / m_uiRecordCount;
                assert(uiBytes % m_uiRecordCount == 0);
            }
            UINT uiMetaBytes = m_uiRecordCount * sizeof(UINT);
            m_cbFinalized[DBMETADATA_IDX] = uiMetaBytes;

            if(!HasValidChannel(DBMETADATA_IDX)) {
                // now instantiate a host-side view of the meta-data channel.
                ctprofile_view_update_start(HOST_MEMORY_SPACE_ID, CET_SYNTHESIZE_BLOCK);
                pt = InstantiateChannel(HOST_MEMORY_SPACE_ID, 
                                       pAsyncContext,
                                       NULL,
                                       DBMETADATA_IDX,
                                       NULL,
                                       FALSE, // don't update coherence map!
                                       FALSE);
                ctprofile_view_update_end(HOST_MEMORY_SPACE_ID, BSTATE_EXCLUSIVE, NOXFER);
            }

            // get the platform buffer in the host memory space
            PBuffer * pMetaPBuffer = GetPlatformBuffer(HOST_MEMORY_SPACE_ID, 
                                                       DBMETADATA_IDX,
                                                       NULL);

            // write the stride RecordCount times. 
            void * pMetaBuffer = pMetaPBuffer->GetBuffer();
            UINT * pObjectSizes = reinterpret_cast<UINT*>(pMetaBuffer);
            for(UINT i=0; i<m_uiRecordCount; i++)
                *pObjectSizes++ = uiStride;
        }
        Unlock();
        return pt;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a meta buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the meta buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    //void* 
    //Datablock::GetMetaBuffer(
    //    BOOL bWriteable
    //    )
    //{
    //    assert(LockIsHeld());
    //    if(m_ppBufferMap[HOST_MEMORY_SPACE_ID]->pBuffers[DBMETADATA_IDX] == NULL) {
    //        ctprofile_view_update_start(HOST_MEMORY_SPACE_ID, CET_POINTER_REQUEST);
    //        InstantiateChannel(HOST_MEMORY_SPACE_ID,
    //                           NULL,
    //                           NULL,
    //                           DBMETADATA_IDX, 
    //                           NULL,
    //                           FALSE, 
    //                           TRUE);
    //        ctprofile_view_update_end(HOST_MEMORY_SPACE_ID, BSTATE_EXCLUSIVE, NOXFER);
    //    }
    //    return GetHostChannelBuffer(NULL, DBMETADATA_IDX, bWriteable, TRUE);
    //}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a meta buffer size. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The meta buffer size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::GetMetaBufferLogicalSizeBytes(
        VOID
        )
    {
        assert(LockIsHeld());
        return GetChannelLogicalSizeBytes(PTask::DBMETADATA_IDX);
    }    

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a meta buffer size. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The meta buffer size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::GetMetaBufferAllocatedSizeBytes(
        VOID
        )
    {
        assert(LockIsHeld());
        return m_cbAllocated[DBMETADATA_IDX];
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Grow meta buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="newSize">  New size of the meta data buffer. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::GrowMetaBuffer(
        UINT newSize
        )
    {
        GrowBuffer(DBMETADATA_IDX, newSize);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a pointer to the host memory backing the given channel. If the
    ///             bUpdateIsStale flag is set, force an update from any available shared/exclusive
    ///             copies in other memory spaces. If the request is for a writeable copy, invalidate
    ///             copies in other memory spaces.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pAsyncContext">                [in,out] If non-null, context for the
    ///                                             asynchronous. </param>
    /// <param name="nChannelIndex">                The index of the requested channel. </param>
    /// <param name="uiRequestedState">             true if the caller plans to write to the block. </param>
    /// <param name="bMaterializeAbsentBuffers">    true if the runtime should update a stale host
    ///                                             view before returning. </param>
    /// <param name="bUpdateDataViews">             The allocate if absent. </param>
    /// <param name="bUpdateCoherenceState">        State of the update coherence. </param>
    /// <param name="bDispatchContext">             True if called from task dispatch context. </param>
    /// <param name="bSynchronousConsumer">         The synchronous consumer. </param>
    ///
    /// <returns>   null if it fails, else the channel buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    void* 
    Datablock::GetHostChannelBuffer(
        __in AsyncContext * pAsyncContext,
        __in UINT nChannelIndex, 
        __in BUFFER_COHERENCE_STATE uiRequestedState, 
        __in BOOL bMaterializeAbsentBuffers,
        __in BOOL bUpdateDataViews,
        __in BOOL bUpdateCoherenceState,
        __in BOOL bDispatchContext,
        __in BOOL bSynchronousConsumer
        )
    {
        assert(LockIsHeld());
        PBuffer * pBuffer = NULL;
        ctprofile_view_update_decl();
        BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[HOST_MEMORY_SPACE_ID];
        UINT uiSrcMemorySpaceID = HOST_MEMORY_SPACE_ID;
        void * pChannelBuffer = NULL;
        BOOL bViewUpdateOccurred = FALSE;
        BOOL bAllocationOccurred = FALSE;
        BOOL bCohStateChange = FALSE;
        
        if(!bMaterializeAbsentBuffers) {

            // the caller doesn't want an update. 
            // return whatever is in the map at the given index. 
            // we still need to wait for outstanding dependences
            // and set the coherence state for this buffer accordingly.
            assert(!bUpdateDataViews);
            assert(!bUpdateCoherenceState);
            pBuffer = pEntry->pBuffers[nChannelIndex];

        } else if(!valid(pEntry->eState) || pEntry->pBuffers[nChannelIndex] == NULL) {

            if(bUpdateDataViews) {

                // synchronize request requires us to be able
                // to allocate the buffer if it is not there!
                assert(bMaterializeAbsentBuffers);

                // The entry for the host memory space is not valid. 
                // This means we need to find a valid entry in another memory
                // space and copy it to host memory, if one exists. If none
                // exists, we can just create an empty host side buffer. 
            
                BUFFER_MAP_ENTRY * pValidEntry = NULL;
                UINT nFirstAccMemorySpace = HOST_MEMORY_SPACE_ID + 1;
                UINT nMemSpaces = MemorySpace::GetNumberOfMemorySpaces();
                for(UINT i=nFirstAccMemorySpace; i<nMemSpaces; i++) {                
                    // take the first valid copy we can find. 
                    // TODO: when there are many, take the fastest one!
                    if(valid(m_ppBufferMap[i]->eState)) {
                        pValidEntry = m_ppBufferMap[i];
                        break;
                    }
                }

                // if we found a valid entry in another memory space, synchronize all valid channels from the
                // remote copy. It is tempting to handle this operation at channel granularity (e.g. don't sync
                // if the user is requesting a channel that doesn't exists in the remote copy), but since the
                // block coherence state applies to all three channels, we must synchronize before returning,
                // even if the sync doesn't affect the requested channel state. 
                if(pValidEntry != NULL) {
                
                    // synchronize views
                    ctprofile_view_update_continue(HOST_MEMORY_SPACE_ID, CET_HOST_VIEW_UPDATE);
                    uiSrcMemorySpaceID = pValidEntry->nMemorySpaceId;
                    bViewUpdateOccurred = TRUE;
                    bAllocationOccurred = pEntry->pBuffers[nChannelIndex] == NULL;
                    BOOL bContextAsyncCapable = pAsyncContext != NULL && pAsyncContext->SupportsExplicitAsyncOperations();
                    BOOL bSynchronousUpdate = bSynchronousConsumer || !bContextAsyncCapable;
                    BOOL bLocksRequired = bDispatchContext || 
                        (bSynchronousConsumer && PTask::Runtime::GetDefaultOutputViewMaterializationPolicy() == VIEWMATERIALIZATIONPOLICY_ON_DEMAND); 
                    SynchronizeViews(HOST_MEMORY_SPACE_ID, 
                                     uiSrcMemorySpaceID, 
                                     pAsyncContext, 
                                     uiRequestedState, 
                                     bLocksRequired,
                                     bSynchronousUpdate);

                    // we'd better have a non-null buffer now,
                    // and since we did a view sync with the force-sync flag,
                    // there had better not be any outstanding dependences.
                    pBuffer = pEntry->pBuffers[nChannelIndex];

                } 
            }

            pBuffer = pEntry->pBuffers[nChannelIndex];
            if(pBuffer == NULL && bMaterializeAbsentBuffers) {
                assert(pEntry->pBuffers[nChannelIndex] == NULL);
                if(pEntry->pBuffers[nChannelIndex] == NULL) {
                    InstantiateChannel(HOST_MEMORY_SPACE_ID, NULL, NULL, nChannelIndex, NULL, FALSE, TRUE);
                    pBuffer = pEntry->pBuffers[nChannelIndex];
                    assert(pBuffer != NULL);
                    bAllocationOccurred = TRUE;
                }
            }
        }

        pBuffer = GetPlatformBuffer(HOST_MEMORY_SPACE_ID, nChannelIndex);
        assert(pBuffer != NULL || !bMaterializeAbsentBuffers);

        // We have reached a state where one of two cases must hold. The first: there must be a PBuffer
        // backing the channel, either because it was already valid, or because we couldn't find a
        // remote copy to sync from, so now return the pointer. The second: if there was no valid entry
        // above, or the synchronzation did not involve the requested channel, the call to
        // GetPlatformBuffer will not create an empty one. Only set coherence state according to the
        // request if the we have a non-buller at this index. 

        if(pBuffer != NULL) {
            if(bSynchronousConsumer && 
               pBuffer->ContextRequiresSync(pAsyncContext, OT_MEMCPY_TARGET)) {

                assert(!bViewUpdateOccurred); // view update should have been synchronous
                assert(!bAllocationOccurred); // if we allocated, there was a view update (see above) 
                                              // or no async operation could have been queued.

                // if there are still outstanding operations on this buffer, it means we have asked for the
                // host-backing buffer, but the block is still inflight as a target for another operation. This
                // can occur if we are simply getting an existing buffer, or if a block is recycled through a
                // block pool before the command queues that reference its buffers drain. How we choose to wait
                // for the operations depends on the calling context. If we are not calling from dispatch
                // context, we cannot expect to acquire locks on the async context, so we perform a lockless
                // wait, otherwise, perform the full wait. (The tradeoff is that the lockless wait doesn't
                // clean up the outstanding deps if they have resolved, which means we have to revisit the same
                // dependences again later. 
               
                if(bDispatchContext) {                
                    pBuffer->WaitOutstandingAsyncOperations(pAsyncContext, OT_MEMCPY_TARGET);
                } else {
                    pBuffer->LocklessWaitOutstanding(OT_MEMCPY_TARGET);
                }
            }
            if(bUpdateCoherenceState) {
                SetCoherenceState(HOST_MEMORY_SPACE_ID, uiRequestedState);
                bCohStateChange = TRUE;
            }
            pChannelBuffer = pBuffer->GetBuffer();    
        }

        ctprofile_view_update_end_cond(bViewUpdateOccurred, uiSrcMemorySpaceID, uiRequestedState, bViewUpdateOccurred);
		recordMaterialize();
        return pChannelBuffer;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a template buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   null if it fails, else the template buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    //void* 
    //Datablock::GetTemplateBuffer(
    //    BOOL bWriteable
    //    )
    //{
    //    return GetHostChannelBuffer(NULL, DBTEMPLATE_IDX, bWriteable, TRUE);
    //}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a template buffer size. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The template buffer size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::GetTemplateBufferLogicalSizeBytes(
        VOID
        )
    {
        assert(LockIsHeld());
        return GetChannelLogicalSizeBytes(DBTEMPLATE_IDX);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a template buffer size. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The template buffer size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::GetTemplateBufferAllocatedSizeBytes(
        VOID
        )
    {
        assert(LockIsHeld());
        return m_cbAllocated[DBTEMPLATE_IDX];
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Grow template buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="newSize">  New size of the template buffer. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::GrowTemplateBuffer(
        UINT newSize
        )
    {
        GrowBuffer(DBTEMPLATE_IDX, newSize);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Seals this datablock. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="uiRecordCount">    Number of records. </param>
    /// <param name="uiDataSize">       Size of the data. </param>
    /// <param name="uiMetaDataSize">   Size of the meta data. </param>
    /// <param name="uiTemplateSize">   Size of the template. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::Seal(
        UINT uiRecordCount,
        UINT uiDataSize, 
        UINT uiMetaDataSize, 
        UINT uiTemplateSize
        )
    {
        assert(LockIsHeld());
        m_uiRecordCount = uiRecordCount;
        m_cbFinalized[DBDATA_IDX] = uiDataSize;
        m_cbFinalized[DBMETADATA_IDX] = uiMetaDataSize;
        m_cbFinalized[DBTEMPLATE_IDX]= uiTemplateSize;

        // if the requested sizes are uninitialized, we
        // need to initialize them to match the actual sizes
        if(m_cbRequested[DBDATA_IDX] == 0) 
            m_cbRequested[DBDATA_IDX] = uiDataSize;
        if(m_cbRequested[DBMETADATA_IDX] == 0) 
            m_cbRequested[DBMETADATA_IDX] = uiMetaDataSize;
        if(m_cbRequested[DBTEMPLATE_IDX] == 0) 
            m_cbRequested[DBTEMPLATE_IDX] = uiTemplateSize;

        m_bSealed = TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unseals this datablock. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::Unseal(
		VOID
        )
    {
        assert(LockIsHeld());
		m_bSealed = FALSE;
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' has valid data buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if valid data buffer, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::HasValidDataBuffer(
        VOID
        )
    {
        assert(LockIsHeld());
        BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[HOST_MEMORY_SPACE_ID];
        return (valid(pEntry->eState) && pEntry->pBuffers[DBDATA_IDX] != NULL);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   size estimator. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::SizeEstimator(
        UINT nChannelIndex
        )
    {
        assert(LockIsHeld());
        if(nChannelIndex == DBDATA_IDX && m_pTemplate != NULL) {
            if(m_bForceRequestedSize) {
                return m_cbRequested[nChannelIndex];
            } else {
                return m_pTemplate->GetDatablockByteCount(DBDATA_IDX);
            }
        }
        assert(m_bSealed);
        return m_cbFinalized[nChannelIndex];
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Data size estimator. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::DataSizeEstimator(
        VOID
        )
    {
        assert(LockIsHeld());
        return SizeEstimator(DBDATA_IDX);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Meta size estimator. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::MetaSizeEstimator(
        VOID
        )
    {
        assert(LockIsHeld());
        return SizeEstimator(DBMETADATA_IDX);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Template size estimator. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::TemplateSizeEstimator(
        VOID
        )
    {
        assert(LockIsHeld());
        return SizeEstimator(DBTEMPLATE_IDX);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a metadata pointer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="bWriteable">                   true if writeable. </param>
    /// <param name="bSynchronizeBufferContents">   The synchronize buffer contents. </param>
    ///
    /// <returns>   null if it fails, else the metadata pointer. </returns>
    ///-------------------------------------------------------------------------------------------------

    void *
    Datablock::GetMetadataPointer(
        __in BOOL bWriteable,
        __in BOOL bSynchronizeBufferContents
        ) 
    {
        return GetChannelBufferPointer(DBMETADATA_IDX, bWriteable, bSynchronizeBufferContents);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a template pointer. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="bWriteable">                   true if writeable. </param>
    /// <param name="bSynchronizeBufferContents">   The synchronize buffer contents. </param>
    ///
    /// <returns>   null if it fails, else the template pointer. </returns>
    ///-------------------------------------------------------------------------------------------------

    void *
    Datablock::GetTemplatePointer(
        __in BOOL bWriteable,
        __in BOOL bSynchronizeBufferContents
        ) 
    {
        return GetChannelBufferPointer(DBTEMPLATE_IDX, bWriteable, bSynchronizeBufferContents);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' has metadata channel. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if metadata channel, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::HasMetadataChannel(
        VOID
        ) 
    {
        assert(LockIsHeld());
        return m_cbFinalized[DBMETADATA_IDX] != 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' has template channel. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if template channel, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::HasTemplateChannel(
        VOID
        )
    {
        assert(LockIsHeld());
        return m_cbFinalized[DBTEMPLATE_IDX] != 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the parameter type of the block if it is known. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The parameter type. </returns>
    ///-------------------------------------------------------------------------------------------------

    PTASK_PARM_TYPE 
    Datablock::GetParameterType(
        VOID
        )
    {
        assert(LockIsHeld());
        if(!m_pTemplate)
            return PTPARM_NONE;
        return m_pTemplate->GetParameterBaseType();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' is scalar parameter. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if scalar parameter, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsScalarParameter(
        VOID
        )
    {
        assert(LockIsHeld());
        if(!m_pTemplate)
            return FALSE;
        return m_pTemplate->DescribesScalarParameter();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' is record stream. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if record stream, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsRecordStream(
        VOID
        )
    {
        assert(LockIsHeld());
        if(m_pTemplate) {
            BOOL bResult = m_pTemplate->DescribesRecordStream();
            assert(bResult || !m_bRecordStream);
            return bResult;
        }
        return m_bRecordStream;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a record count. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The record count. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::GetRecordCount(
        VOID
        )
    {
        assert(LockIsHeld());
        return m_uiRecordCount;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a record count. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="uiRecordCount">    Number of records. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    VOID 
    Datablock::SetRecordCount(
        UINT uiRecordCount
        )
    {
        assert(LockIsHeld());
        assert(!m_bSealed);
        m_uiRecordCount = uiRecordCount;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Marks this block as a member of pPort's block pool.</summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::SetPooledBlock(
        BlockPoolOwner * pPort
        ) 
    {
        assert(LockIsHeld());
        assert(pPort == m_pPoolOwner || pPort == NULL || m_pPoolOwner == NULL);
        m_pPoolOwner = pPort;
        m_bPooledBlock = (pPort != NULL);
        if(m_pPoolOwner != NULL) {
            dbprofile_record_owner();
            m_bPoolRequiresInitialValue = m_pTemplate && m_pTemplate->HasInitialValue();
            if(m_bPoolRequiresInitialValue) {
                UINT nMemSpaceId;
                for(nMemSpaceId=HOST_MEMORY_SPACE_ID; 
                    nMemSpaceId<MemorySpace::GetNumberOfMemorySpaces(); 
                    nMemSpaceId++) {
                    BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[nMemSpaceId];
                    if(pEntry->eState != BSTATE_NO_ENTRY && pEntry->eState != PTask::BSTATE_INVALID) {
                        PBuffer * pBuffer = pEntry->pBuffers[0];
#ifdef FORCE_CORRUPT_NON_AFFINITIZED_VIEWS
                        if(pBuffer != (PBuffer*)0xDEADBEEF)
#endif
                            pBuffer->MarkDirty(FALSE);
                        m_vInitialValueCleanList.insert(pBuffer);
                    }
                }
            }
        } 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this datablock is marshallable. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <returns>   true if marshallable, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsMarshallable(
        VOID
        )
    {
        assert(LockIsHeld());
        return m_bMarshallable;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Marks the datablock marshallable or not. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="bMarshallable">    true if marshallable. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::SetMarshallable(
        BOOL bMarshallable
        )
    {
        assert(LockIsHeld());
        m_bMarshallable = bMarshallable;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the datablock template describing this block </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the template. </returns>
    ///-------------------------------------------------------------------------------------------------
        
    DatablockTemplate * 
    Datablock::GetTemplate(
        VOID
        ) 
    { 
        assert(LockIsHeld());
        return m_pTemplate; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the access flags for this block. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   The access flags. </returns>
    ///-------------------------------------------------------------------------------------------------

    BUFFERACCESSFLAGS 
    Datablock::GetAccessFlags(
        VOID
        ) 
    { 
        assert(LockIsHeld());
        return m_eBufferAccessFlags; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets the access flags. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="eFlags">   The access flags. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::SetAccessFlags(
        BUFFERACCESSFLAGS eFlags
        ) 
    { 
        assert(LockIsHeld());
        m_eBufferAccessFlags = eFlags; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the destination port of this block </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the destination port. </returns>
    ///-------------------------------------------------------------------------------------------------

    Port * 
    Datablock::GetDestinationPort(
        VOID
        ) 
    { 
        assert(LockIsHeld());
        return m_pDestinationPort; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a destination port for this block. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pPort">    (optional) [in] non-null, the port in question. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::SetDestinationPort(
        Port*pPort
        ) 
    { 
        assert(LockIsHeld());
        m_pDestinationPort = pPort; 
        dbprofile_record_pb(pPort);
        ctprofile_record_pb(pPort);
    }
     
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record which ptask produced this datablock as output. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pTask">    (optional) [in] non-null, the Task in question. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::SetProducerTask(
        Task * pTask
        ) 
    { 
        assert(LockIsHeld());
        m_pProducerTask = pTask; 
        m_vWriters.insert(pTask);
        dbprofile_record_tb(pTask);
        ctprofile_record_tb(pTask);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this datablock is part of an output port's block pool. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if pooled, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsPooled(
        VOID
        ) 
    { 
        assert(LockIsHeld());
        return m_bPooledBlock; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this datablock is sealed. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if pooled, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsSealed(
        VOID
        ) 
    { 
        assert(LockIsHeld());
        return m_bSealed; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this datablock is supposed by backed by byte-addressable device-side
    /// 			buffers. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   true if pooled, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsByteAddressable(
        VOID
        ) 
    { 
        assert(LockIsHeld());
        return m_bByteAddressable; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the port whose pool owns this datablock. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the pool owner. </returns>
    ///-------------------------------------------------------------------------------------------------

    BlockPoolOwner * 
    Datablock::GetPoolOwner(
        VOID
        ) 
    { 
        assert(LockIsHeld());
        return m_pPoolOwner; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Override of ReferenceCounted::Release(). Release this datablock: if the refcount
    ///             drops to zero and the block is a member of a block pool, return the block to the
    ///             pool. Otherwise delete it.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <returns>   new reference count. </returns>
    ///-------------------------------------------------------------------------------------------------

    LONG
    Datablock::Release() {
        LONG privateCount = 0;
        Lock();
        if(m_uiRefCount == 0) {
            assert(m_uiRefCount != 0); // should have been deleted already!
        }
        if(m_uiRefCount == 1) {
            privateCount = 0;
            m_uiRefCount = 0;
        } else {
            privateCount = InterlockedDecrement(&m_uiRefCount);
            assert(m_uiRefCount >= 0);
        }
        Unlock();
        if(privateCount == 0) {
            if(PTask::Runtime::GetBlockPoolsEnabled() && 
               m_bPooledBlock && 
               m_pPoolOwner && 
               BlockPoolOwner::IsPoolOwnerActive(m_pPoolOwner)) {
                // Reset application context.
                m_pApplicationContext = NULL;

                // Invalidate, but don't release buffers.
                m_bSealed = FALSE;
                UINT nMemSpaceId;
                std::vector<BUFFER_MAP_ENTRY*> vCleanEntries;
                for(nMemSpaceId=HOST_MEMORY_SPACE_ID; 
                    nMemSpaceId<MemorySpace::GetNumberOfMemorySpaces(); 
                    nMemSpaceId++) {
                    BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[nMemSpaceId];
                    if(pEntry->eState != PTask::BSTATE_NO_ENTRY) {
                        PBuffer * pBuffer = pEntry->pBuffers[0];
#ifdef FORCE_CORRUPT_NON_AFFINITIZED_VIEWS
                        PBuffer * pHack = (PBuffer*) 0xDEADBEEF;
                        if(pBuffer == pHack) continue;
#endif
                        if(m_bPoolRequiresInitialValue) {
                            std::set<PBuffer*>::iterator pi = m_vInitialValueCleanList.find(pBuffer);
                            if(pi != m_vInitialValueCleanList.end()) {
                                if(pBuffer->IsDirty()) {
                                    m_vInitialValueCleanList.erase(pi);
                                } else {
                                    vCleanEntries.push_back(pEntry);
                                }
                            }
                        } else {
                            pEntry->eState = PTask::BSTATE_INVALID;
                        }
                        pBuffer->ReleaseRetiredDependences(NULL);
                    }
                }
                size_t nCleanEntries = vCleanEntries.size();
                if(nCleanEntries) {
                    std::vector<BUFFER_MAP_ENTRY*>::iterator vi;
                    for(vi=vCleanEntries.begin(); vi!=vCleanEntries.end(); vi++) {
                        BUFFER_MAP_ENTRY * pEntry = *vi;
                        pEntry->eState = nCleanEntries > 1 ? BSTATE_SHARED : BSTATE_EXCLUSIVE;
                    }
                } else {
                    // TODO JC Commented out for SFMA14 Opt Flow investigation. Anything else to do?
                    // ResetInitialValueForPool(NULL);
                }
                m_pPoolOwner->ReturnToPool(this);
            } else {
                GarbageCollector::QueueForGC(this);
            }
        }
        return privateCount;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the stride of objects in the Datablock. If the block has a template, this
    ///             comes from the stride member of the template's dimensions. If no template is
    ///             present and the block is unsealed, assert. If no template is present, and the
    ///             datablock is sealed, return the data-channel's byte-length by the number of
    ///             records. If no template is present, and the datablock is byte- addressable,
    ///             return 1.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/29/2011. </remarks>
    ///
    /// <returns>   The stride. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::GetStride(
        VOID
        )
    {
        assert(LockIsHeld());
        if(m_pTemplate != NULL) {
            return m_pTemplate->GetStride();
        }
        if(IsByteAddressable()) {
            // each element is 1 byte!
            return 1;
        }
        if(!IsSealed()) {
            // if the block is not sealed, we cannot estimate
            // the stride based on the record count: we can only 
            // assume the stride is a single byte.
            assert(IsSealed());
            return 1;
        }
        if(GetRecordCount() == 0 || m_cbFinalized[DBDATA_IDX] == 0) {
            PTask::Runtime::Inform("Datablock::GetStride cannot infer stride: 0-size block?");
            return 0;
        }
        return (m_cbFinalized[DBDATA_IDX] / GetRecordCount());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the number of objects in X dimension of the Datablock. If the block has a
    ///             template, this comes from the XDIM member of the template's dimensions. If no
    ///             template is present and the block is unsealed, assert. If no template is present,
    ///             and the datablock is sealed, return the record count. If no template is present,
    ///             and the datablock is byte- addressable, return the data-channel size.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/29/2011. </remarks>
    ///
    /// <returns>   The number of elements in X. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::GetXElementCount(
        VOID
        )
    {
        assert(LockIsHeld());
        if(m_pTemplate != NULL) {
            return m_pTemplate->GetXElementCount();
        }
        if(!IsSealed()) {
            // if the block is not sealed, and there is no template,
            // we cannot compute the number of X elements based on
            // the record count and stride
            assert(IsSealed());
            return 0;
        }
        return GetRecordCount();
    }
        
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the number of objects in Y dimension of the Datablock. If the block has a
    ///             template, this comes from the YDIM member of the template's dimensions. If no
    ///             template is present and the block is unsealed, assert. If no template is present,
    ///             return 1.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/29/2011. </remarks>
    ///
    /// <returns>   The number of elements in Y. </returns>
    ///-------------------------------------------------------------------------------------------------
        
    UINT 
    Datablock::GetYElementCount(
        VOID
        )
    {
        assert(LockIsHeld());
        if(m_pTemplate != NULL) {
            return m_pTemplate->GetYElementCount();
        }
        // if the block is not sealed, we still can
        // return a valid Y: it's always 1. But the 
        // user shouldn't be calling this on unsealed blocks! 
        assert(IsSealed());
        return 1;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the number of objects in Z dimension of the Datablock. If the block has a
    ///             template, this comes from the ZDIM member of the template's dimensions. If no
    ///             template is present and the block is unsealed, assert. If no template is present,
    ///             return 1.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/29/2011. </remarks>
    ///
    /// <returns>   The number of elements in Z. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::GetZElementCount(
        VOID
        )
    {
        assert(LockIsHeld());
        if(m_pTemplate != NULL) {
            return m_pTemplate->GetZElementCount();
        }
        // if the block is not sealed, we still can
        // return a valid Z: it's always 1. But the 
        // user shouldn't be calling this on unsealed blocks! 
        assert(IsSealed());
        return 1;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes a new channel iterator which can be used
    /// 			to iterate over all valid channels in the block. </summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock::ChannelIterator& 
    Datablock::FirstChannel(
        VOID
        ) 
    { 
        m_vIterator.m_nChannelIndex = DBDATA_IDX;
        m_vIterator.m_pIteratedBlock = this;
        return m_vIterator;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the end-of-iteration sentinal. </summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock::ChannelIterator& 
    Datablock::LastChannel(
        VOID
        ) 
    { 
        return m_iterLastChannel; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return true if the given block is out of date on the dispatch (or dependent
    ///             target) accelerator *and* up-to-date somewhere other than in host memory.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="pTargetAccelerator">   [in,out] If non-null, the dispatch accelerator. </param>
    ///
    /// <returns>   true if migration is required for this block, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL    
    Datablock::RequiresMigration(
        Accelerator * pTargetAccelerator
        )
    {
        assert(LockIsHeld());
        BUFFER_COHERENCE_STATE dstState = GetCoherenceState(pTargetAccelerator);
        if(valid(dstState)) return FALSE; // up to date already!

        // we are not up to date on the target accelerator...
        // is there a most recent accelerator. Note the GetMostRecentAccelerator
        // member will prefer an accelerator memory space over a host memory space.

        Accelerator * pSourceAccelerator = GetMostRecentAccelerator();
        if((pSourceAccelerator != NULL) && 
           (pSourceAccelerator != pTargetAccelerator) &&
            pSourceAccelerator->GetMemorySpaceId() != HOST_MEMORY_SPACE_ID) {

            // In general we require migration if the source and dest are different, and not the host
            // memory space. Note that this is a somewhat specialized notion of "migration": migration
            // means device->device. Also note: if there is a shared copy in both source and dest, we also
            // do not require migration. All other combinations of coherence states either do not arise by
            // construction, or require migration. 

            BUFFER_COHERENCE_STATE srcState = GetCoherenceState(pSourceAccelerator);
            BOOL bDestViewValid = (srcState == BSTATE_SHARED && dstState == BSTATE_SHARED);
            return !bDestViewValid;
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the block has outstanding async operations on the buffer for the target
    ///             memory space that need to resolve before an operation of the given type
    ///             can be performed.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/25/2013. </remarks>
    ///
    /// <param name="pTargetMemorySpace">   [in,out] If non-null, the accelerator. </param>
    /// <param name="eOpType">              Type of the operation. </param>
    ///
    /// <returns>   true if outstanding dependence, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::HasOutstandingAsyncDependences(
        __in Accelerator * pAccelerator,
        __inout ASYNCHRONOUS_OPTYPE eOpType
        )
    {
        assert(LockIsHeld());
        UINT uiMemSpaceId = pAccelerator ? pAccelerator->GetMemorySpaceId() : HOST_MEMORY_SPACE_ID;
        return HasOutstandingAsyncDependences(uiMemSpaceId, eOpType);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the block has outstanding async operations on the buffer for the target
    ///             memory space that need to resolve before an operation of the given type can be
    ///             performed.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/25/2013. </remarks>
    ///
    /// <param name="uiMemSpaceId"> If non-null, the accelerator. </param>
    /// <param name="eOpType">      Type of the operation. </param>
    ///
    /// <returns>   true if outstanding dependence, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::HasOutstandingAsyncDependences(
        __in UINT uiMemSpaceId,
        __inout ASYNCHRONOUS_OPTYPE eOpType
        )
    {
        assert(LockIsHeld());
        assert(uiMemSpaceId >= 0 && uiMemSpaceId < MemorySpace::GetNumberOfMemorySpaces());
        if(uiMemSpaceId >= 0 && uiMemSpaceId < MemorySpace::GetNumberOfMemorySpaces()) {
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[uiMemSpaceId];
            PBuffer * pBuffer = pEntry->pBuffers[DBDATA_IDX];
            if(pBuffer != NULL) {
                if(pBuffer->HasConflictingOutstandingDependences(eOpType)) {
                    return TRUE;                    
                }
            }
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the set of accelerators for which there are outstanding async operations in
    ///             the target memory space that need to complete before new operations in the target
    ///             memory space can begin. Note that this version returns a pointer to a member data
    ///             structure, so callers should use it with caution. In particular, this is
    ///             *NOT* a good way to manage lock acquire/release lists for accelerators, since the
    ///             list may change between the lock acquire phase and release phase if outstanding
    ///             dependences are resolved while those locks are held. For such a purpose, the
    ///             GetCurrentOutstandingAsyncAccelerators API (which returns a copy) should be used
    ///             instead.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/26/2013. </remarks>
    ///
    /// <param name="pTargetMemorySpace">   [in,out] The first parameter. </param>
    /// <param name="eOpType">              Type of the operation. </param>
    ///
    /// <returns>   null if it fails, else the outstanding asynchronous accelerators. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::set<Accelerator*>*
    Datablock::GetOutstandingAsyncAcceleratorsPointer(
        __in Accelerator * pTargetMemorySpace,
        __in ASYNCHRONOUS_OPTYPE eOpType
        )
    {
        assert(LockIsHeld());
        UINT uiMemSpaceId = (pTargetMemorySpace == NULL) ? 
            HOST_MEMORY_SPACE_ID : 
            pTargetMemorySpace->GetMemorySpaceId();
        return GetOutstandingAsyncAcceleratorsPointer(uiMemSpaceId, eOpType);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the set of accelerators for which there are outstanding async operations in
    ///             the target memory space that need to complete before new operations in the target
    ///             memory space can begin. Note that this version returns a pointer to a member data
    ///             structure, so callers should use it with caution. In particular, this is
    ///             *NOT* a good way to manage lock acquire/release lists for accelerators, since the
    ///             list may change between the lock acquire phase and release phase if outstanding
    ///             dependences are resolved while those locks are held. For such a purpose, the
    ///             GetCurrentOutstandingAsyncAccelerators API (which returns a copy) should be used
    ///             instead.
    ///             </summary>
    ///             
    /// <remarks>   crossbac, 6/26/2013. </remarks>
    ///
    /// <param name="pTargetMemorySpace">   [in,out] The first parameter. </param>
    /// <param name="eOpType">              Type of the operation. </param>
    ///
    /// <returns>   null if it fails, else the outstanding asynchronous accelerators. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::set<Accelerator*>*
    Datablock::GetOutstandingAsyncAcceleratorsPointer(
        __in UINT uiMemSpaceId,
        __in ASYNCHRONOUS_OPTYPE eOpType
        )
    {
        assert(LockIsHeld());
        assert(uiMemSpaceId >= 0 && uiMemSpaceId < MemorySpace::GetNumberOfMemorySpaces());
        if(uiMemSpaceId >= 0 && uiMemSpaceId < MemorySpace::GetNumberOfMemorySpaces()) {
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[uiMemSpaceId];
            PBuffer * pBuffer = pEntry->pBuffers[DBDATA_IDX];
            if(pBuffer != NULL) {
                return pBuffer->GetOutstandingAcceleratorDependences(eOpType);
            }
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the set of accelerators for which there are outstanding async operations in
    ///             the target memory space that need to complete before new operations in the target
    ///             memory space can begin.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/26/2013. </remarks>
    ///
    /// <param name="uiTargetMemorySpaceId">    The target memory space. </param>
    /// <param name="eOpType">                  Type of the operation. </param>
    /// <param name="vAccelerators">            [out] the accelerators. </param>
    ///
    /// <returns>   the number of outstanding accelerators. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    Datablock::GetCurrentOutstandingAsyncAccelerators(
        __in  UINT uiTargetMemorySpaceId,
        __in  ASYNCHRONOUS_OPTYPE eOpType,
        __out std::set<Accelerator*>& vAccelerators
        )
    {
        assert(LockIsHeld());
        assert(uiTargetMemorySpaceId >= 0 && 
               uiTargetMemorySpaceId < MemorySpace::GetNumberOfMemorySpaces());
        size_t uiAccsSizeOnEntry = vAccelerators.size();
        if(uiTargetMemorySpaceId >= 0 && 
           uiTargetMemorySpaceId < MemorySpace::GetNumberOfMemorySpaces()) {
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[uiTargetMemorySpaceId];
            PBuffer * pBuffer = pEntry->pBuffers[DBDATA_IDX];
            if(pBuffer != NULL) {
                std::set<Accelerator*>* pAccs = pBuffer->GetOutstandingAcceleratorDependences(eOpType);
                vAccelerators.insert(pAccs->begin(), pAccs->end());
            }
        }
        return static_cast<UINT>(vAccelerators.size()-uiAccsSizeOnEntry);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Lockless wait outstanding: without acquiring any locks attempt to perform a
    ///             synchronous wait for any outstanding async dependences on this block that
    ///             conflict with an operation of the given type on the given target accelerator to
    ///             complete. This is an experimental API, enable/disable with
    ///             PTask::Runtime::*etTaskDispatchLocklessIncomingDepWait(), attempting to leverage
    ///             the fact that CUDA apis for waiting on events (which appear to be thread-safe and
    ///             decoupled from a particular device context)
    ///             to minimize serialization associated with outstanding dependences on blocks
    ///             consumed by tasks that do not require accelerators for any other reason than to
    ///             wait for such operations to complete.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 7/1/2013. </remarks>
    ///
    /// <param name="pTargetMemorySpace">   [in,out] If non-null, target accelerator. </param>
    /// <param name="eOpType">              Type of the operation. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Datablock::LocklessWaitOutstanding(
        __in Accelerator * pTargetMemorySpace, 
        __in ASYNCHRONOUS_OPTYPE eOpType
        )
    {
        assert(LockIsHeld());
        assert(pTargetMemorySpace != NULL);
        assert(!pTargetMemorySpace->SupportsExplicitAsyncOperations());
        assert(PTask::Runtime::GetTaskDispatchLocklessIncomingDepWait());
        if(pTargetMemorySpace != NULL) {
            UINT uiMemorySpaceId = pTargetMemorySpace->GetMemorySpaceId();
            assert(uiMemorySpaceId == HOST_MEMORY_SPACE_ID); // currently the only relevant case
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[uiMemorySpaceId];
            PBuffer * pBuffer = pEntry->pBuffers[DBDATA_IDX];
            if(pBuffer != NULL) {
                return pBuffer->LocklessWaitOutstanding(eOpType);
            }
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets this block's coherence state for the given accelerator. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="pAccelerator"> (optional) [in] If non-null, an accelerator that will require a
    ///                             view of this block. </param>
    ///
    /// <returns>   The coherence state. </returns>
    ///-------------------------------------------------------------------------------------------------

    BUFFER_COHERENCE_STATE  
    Datablock::GetCoherenceState(
        Accelerator * pAccelerator
        )
    {
        assert(LockIsHeld());
        assert(pAccelerator != NULL);
        UINT nMemSpaceId = pAccelerator->GetMemorySpaceId();
        return m_ppBufferMap[nMemSpaceId]->eState;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a platform-specific accelerator data buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pAccelerator">     (optional) [in] non-null, the accelerator in question. </param>
    /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
    /// <param name="pRequester">       (optional) [in,out] If non-null, the requesting accelerator,
    ///                                 which we may want to use to allocate host-side memory if it has a
    ///                                 specialized allocator. </param>
    ///
    /// <returns>   null if it fails, else the accelerator data buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    PBuffer * 
    Datablock::GetPlatformBuffer(
        Accelerator * pAccelerator, 
        UINT nChannelIndex,
        Accelerator * pRequester
        )
    {
        return GetPlatformBuffer(pAccelerator->GetMemorySpaceId(), nChannelIndex, pRequester);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a platform-specific accelerator data buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pAccelerator">     (optional) [in] non-null, the accelerator in question. </param>
    /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
    /// <param name="pRequester">       (optional) [in,out] If non-null, the requesting accelerator,
    ///                                 which we may want to use to allocate host-side memory if it has a
    ///                                 specialized allocator. </param>
    ///
    /// <returns>   null if it fails, else the accelerator data buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    PBuffer * 
    Datablock::GetPlatformBuffer(
        Accelerator * pAccelerator, 
        ChannelIterator &vChannelIterator,
        Accelerator * pRequester
        )
    {
        return GetPlatformBuffer(pAccelerator->GetMemorySpaceId(), 
                                 (UINT) vChannelIterator.m_nChannelIndex, 
                                 pRequester);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a platform-specific accelerator data buffer. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="nMemorySpaceId">   (optional) [in] non-null, the accelerator in question. </param>
    /// <param name="nChannelIndex">    Zero-based index of the n channel. </param>
    /// <param name="pRequester">       (optional) [in,out] If non-null, the requester. </param>
    ///
    /// <returns>   null if it fails, else the accelerator data buffer. </returns>
    ///-------------------------------------------------------------------------------------------------

    PBuffer * 
    Datablock::GetPlatformBuffer(
        UINT nMemorySpaceId, 
        UINT nChannelIndex,
        Accelerator * pRequester
        )
    {
        UNREFERENCED_PARAMETER(pRequester);
        assert(LockIsHeld());
        assert(nMemorySpaceId >= HOST_MEMORY_SPACE_ID);
        assert(nMemorySpaceId < MemorySpace::GetNumberOfMemorySpaces());
        assert(nChannelIndex >= DBDATA_IDX);
        assert(nChannelIndex < NUM_DATABLOCK_CHANNELS);

        BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[nMemorySpaceId];
        if(pEntry->eState == BSTATE_NO_ENTRY || pEntry->pBuffers[nChannelIndex] == NULL) {
            return NULL;
        }
        PBuffer * pBuffer = pEntry->pBuffers[nChannelIndex];
        assert(pBuffer != NULL);
        return pBuffer;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the block has any valid channels at all.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <returns>   true if valid channel, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::HasValidChannels(
        VOID
        )
    {
        assert(LockIsHeld());
        for(UINT i=HOST_MEMORY_SPACE_ID; i<MemorySpace::GetNumberOfMemorySpaces(); i++) {
            for(UINT n=DBDATA_IDX; n<NUM_DATABLOCK_CHANNELS; n++) {
                if(valid(m_ppBufferMap[i]->eState) && m_ppBufferMap[i]->pBuffers[n] != NULL) {
                    return TRUE;
                }
            }
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the ChannelIterator has a valid channel in *any* memory space for this
    ///             datablock (at the channel corresponding to its current iteration state).
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="vIterator">    [in] channel iterator. </param>
    ///
    /// <returns>   true if valid channel, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::HasValidChannel(
        ChannelIterator &vIterator
        )
    {
        return HasValidChannel((UINT)vIterator.m_nChannelIndex);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'nChannelIndex' has valid channel in *any* memory space
    /// 			for this datablock. </summary>
    ///
    /// <remarks>   Crossbac, 12/30/2011. </remarks>
    ///
    /// <param name="nChannelIndex">    Zero-based index of the channel. </param>
    ///
    /// <returns>   true if valid channel, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::HasValidChannel(
        UINT nChannelIndex
        )
    {
        assert(LockIsHeld());
        for(UINT i=HOST_MEMORY_SPACE_ID; i<MemorySpace::GetNumberOfMemorySpaces(); i++) {
            if(valid(m_ppBufferMap[i]->eState) && m_ppBufferMap[i]->pBuffers[nChannelIndex] != NULL) {
                return TRUE;
            }
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the unique id for the datablock. </summary>
    ///
    /// <remarks>   Crossbac, 2/24/2012. </remarks>
    ///
    /// <returns>   The dbuid. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Datablock::GetDBUID(
        VOID
        ) 
    { 
        return m_uiDBID; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Queries if the datablock logically is empty. This check is required because we
    ///             must often allocate a non-zero size object in a given memory space to represent
    ///             one whose logical size is zero bytes, so that we have an object that can be bound
    ///             to kernel execution resources (parameters, globals). We need to be able to tell
    ///             when a sealed block has non-zero size, but has a record count of zero so that
    ///             descriptor ports and metaports can do the right thing.
    ///             </summary>
    ///
    /// <remarks>   call with lock held.
    /// 			
    /// 			crossbac, 6/19/2012. </remarks>
    ///
    /// <returns>   true if the block is logically is empty, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsLogicallyEmpty(
        VOID
        )
    {
        assert(LockIsHeld());
        return m_bLogicallyEmpty;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Create a new datablock. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pAsyncContext">        (optional) [in] If non-null, an async context, which will
    ///                                     wind up using this block. </param>
    /// <param name="pTemplate">            [in,out] If non-null, the template. </param>
    /// <param name="pExtent">              (optional) [in] If non-null, initial data. </param>
    /// <param name="flags">                (optional) [in] buffer access flags. </param>
    /// <param name="luiBlockControlCode">  (optional) [in] a block control code. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Datablock::CreateDatablock(
        __in AsyncContext *      pAsyncContext,
        __in DatablockTemplate * pTemplate, 
        __in HOSTMEMORYEXTENT *  pExtent,
        __in BUFFERACCESSFLAGS   flags,
        __in CONTROLSIGNAL       luiBlockControlCode
        )
    {
        return new Datablock(pAsyncContext, 
                             pTemplate, 
                             flags, 
                             luiBlockControlCode, 
                             pExtent, 
                             FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Create a new datablock, ensuring that we enable subsequent use with backend async
    ///             APIs. If we can create a device-side view right off, do so. If not, be sure to
    ///             allocate host-side views with pinned memory.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="vAsyncContexts">       (optional) [in] If non-null, an async context, which will
    ///                                     wind up using this block. </param>
    /// <param name="pTemplate">            [in] If non-null, the datablock template. </param>
    /// <param name="pInitialData">         (optional) [in] If non-null, initial data. </param>
    /// <param name="flags">                (optional) [in] buffer access flags. </param>
    /// <param name="luiBlockControlCode">  (optional) [in] a block control code. </param>
    /// <param name="bMaterializeAll">      The materialize all. </param>
    /// <param name="bPageLockHostViews">   The page lock host views. </param>
    ///
    /// <returns>   null if it fails, else the new block. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Datablock::CreateDatablockAsync(
        __in std::set<AsyncContext*>&     vAsyncContexts,
        __in DatablockTemplate *          pTemplate, 
        __in HOSTMEMORYEXTENT *           pInitialData,
        __in BUFFERACCESSFLAGS            flags,
        __in CONTROLSIGNAL                luiBlockControlCode,
        __in BOOL                         bCreateDeviceBuffers,
        __in BOOL                         bMaterializeDeviceViews,
        __in BOOL                         bPageLockHostViews
        )
    {
        return new Datablock(vAsyncContexts, 
                             pTemplate, 
                             flags, 
                             luiBlockControlCode, 
                             pInitialData,
                             FALSE,
                             bCreateDeviceBuffers,
                             bMaterializeDeviceViews,
                             bPageLockHostViews);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Create a new datablock, ensuring that we enable subsequent use with backend async
    ///             APIs. If we can create a device-side view right off, do so. If not, be sure to
    ///             allocate host-side views with pinned memory.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="vAccelerators">            (optional) [in] If non-null, an async context, which
    ///                                         will wind up using this block. </param>
    /// <param name="pTemplate">                [in] If non-null, the datablock template. </param>
    /// <param name="pInitialData">             (optional) [in] If non-null, initial data. </param>
    /// <param name="flags">                    (optional) [in] buffer access flags. </param>
    /// <param name="luiBlockControlCode">      (optional) [in] a block control code. </param>
    /// <param name="bCreateDeviceBuffers">     The materialize all. </param>
    /// <param name="bMaterializeDeviceViews">  The materialize device views. </param>
    /// <param name="bPageLockHostViews">       The page lock host views. </param>
    ///
    /// <returns>   null if it fails, else the new block. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Datablock::CreateDatablockAsync(
        __in std::set<Accelerator*>&      vAccelerators,
        __in DatablockTemplate *          pTemplate, 
        __in HOSTMEMORYEXTENT *           pInitialData,
        __in BUFFERACCESSFLAGS            flags,
        __in CONTROLSIGNAL                luiBlockControlCode,
        __in BOOL                         bCreateDeviceBuffers,
        __in BOOL                         bMaterializeDeviceViews,
        __in BOOL                         bPageLockHostViews
        )
    {
        BOOL bHostFound = FALSE;
        std::set<AsyncContext*> vContexts;
        std::set<Accelerator*>::iterator ai;
        for(ai=vAccelerators.begin(); ai!=vAccelerators.end(); ai++) {
            Accelerator * pAccelerator = *ai;
            AsyncContext * pAsyncContext = pAccelerator->GetAsyncContext(ASYNCCTXT_XFERHTOD);
            bHostFound |= pAccelerator->IsHost();
            assert(pAsyncContext != NULL);
            vContexts.insert(pAsyncContext);
        }
        assert(bHostFound || !bPageLockHostViews);
        return new Datablock(vContexts, 
                             pTemplate, 
                             flags, 
                             luiBlockControlCode, 
                             pInitialData,
                             FALSE,
                             bCreateDeviceBuffers,
                             bMaterializeDeviceViews,
                             bPageLockHostViews);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Create a new datablock. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pTemplate">            [in] If non-null, the datablock template. </param>
    /// <param name="uiDataBufferSize">     Size of the data buffer. </param>
    /// <param name="uiMetaBufferSize">     Size of the meta data buffer. </param>
    /// <param name="uiTemplateBufferSize"> Size of the template buffer. </param>
    /// <param name="eFlags">               The flags. </param>
    /// <param name="luiBlockControlCode">  (optional) a block control code. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Datablock::CreateDatablock(
        __in DatablockTemplate * pTemplate, 
        __in UINT                uiDataBufferSize, 
        __in UINT                uiMetaBufferSize, 
        __in UINT                uiTemplateBufferSize,
        __in BUFFERACCESSFLAGS   eFlags,
        __in CONTROLSIGNAL       luiBlockControlCode
        )
    {
        Datablock * pBlock = new Datablock(NULL,                  // no AsyncContext object available
                                           pTemplate,             // template pointer
                                           uiDataBufferSize,      // initial allocation size for data channel
                                           uiMetaBufferSize,      // initial allocation size for meta data channel
                                           uiTemplateBufferSize,  // initial allocation size for template data channel
                                           eFlags,                // flags
                                           luiBlockControlCode,    // block control code
                                           NULL,                  // no initial data 
                                           0,                     // 0 records in initial data
                                           FALSE);                // do not attempt to finalize/materialize a view
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Create a new datablock. </summary>
    ///
    /// <remarks>   Crossbac, 12/19/2011. </remarks>
    ///
    /// <param name="pAsyncContext">        [in,out] (optional)  If non-null, context for the
    ///                                     asynchronous. </param>
    /// <param name="pTemplate">            [in,out] If non-null, the template. </param>
    /// <param name="uiDataBufferSize">     Size of the data buffer. </param>
    /// <param name="uiMetaBufferSize">     Size of the meta data buffer. </param>
    /// <param name="uiTemplateBufferSize"> Size of the template buffer. </param>
    /// <param name="eFlags">               The flags. </param>
    /// <param name="luiBlockControlCode">  (optional) a block control code. </param>
    /// <param name="pExtent">              [in,out] (optional) buffer access flags. </param>
    /// <param name="uiInitDataRecords">    The initialise data records. </param>
    /// <param name="bFinalize">            The finalize. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Datablock::CreateDatablock(
        __in AsyncContext *         pAsyncContext,
        __in DatablockTemplate *    pTemplate, 
        __in UINT                   uiDataBufferSize, 
        __in UINT                   uiMetaBufferSize, 
        __in UINT                   uiTemplateBufferSize,
        __in BUFFERACCESSFLAGS      eFlags,
        __in CONTROLSIGNAL          luiBlockControlCode,
        __in HOSTMEMORYEXTENT *     pExtent,
        __in UINT                   uiInitDataRecords,
        __in BOOL                   bFinalize
        )
    {
        return new Datablock(pAsyncContext,
                             pTemplate,
                             uiDataBufferSize,
                             uiMetaBufferSize,
                             uiTemplateBufferSize,
                             eFlags,
                             luiBlockControlCode,
                             pExtent,
                             uiInitDataRecords,
                             bFinalize);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a control block. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2012. </remarks>
    ///
    /// <param name="luiBlockControlCode">   (optional) [in] a block control code. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *
    Datablock::CreateControlBlock( 
        __in CONTROLSIGNAL luiBlockControlCode
        )
    {
        return new Datablock(luiBlockControlCode);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a new datablock based on the initial value in the datablock template.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/19/2012. </remarks>
    ///
    /// <param name="pTemplate">        [in,out] If non-null, the template. </param>
    /// <param name="eFlags">           The flags. </param>
    /// <param name="pAsyncContext">    [in,out] If non-null, context for asynchronous ops. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Datablock::CreateInitialValueBlock(
        __in AsyncContext *      pAsyncContext,
        __in DatablockTemplate * pTemplate,
        __in BUFFERACCESSFLAGS   eFlags
        )
    {
        UINT uiTemplateValueSizeBytes = pTemplate->GetDatablockByteCount(DBDATA_IDX);
        UINT uiAllocationSizeBytes = pTemplate->IsInitialValueEmpty() ? 0 : uiTemplateValueSizeBytes;
        return CreateInitialValueBlock(pAsyncContext, pTemplate, uiAllocationSizeBytes, eFlags, FALSE);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a new datablock based on the initial value in the datablock template.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/19/2012. </remarks>
    ///
    /// <param name="pAsyncContext">            [in,out] If non-null, context for asynchronous ops. </param>
    /// <param name="pTemplate">                [in,out] If non-null, the template. </param>
    /// <param name="uiAllocationSizeBytes">    The maximum number of bytes. Used to enforce an upper
    ///                                         bound if the initial value size is different from the
    ///                                         size specified in the template parameter. Zero means use
    ///                                         the template parameter size. </param>
    /// <param name="eFlags">                   The flags. </param>
    /// <param name="bPooledBlock">             The pooled block. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Datablock::CreateInitialValueBlock(
        __in AsyncContext *      pAsyncContext,
        __in DatablockTemplate * pTemplate,
        __in UINT                uiAllocationSizeBytes,
        __in BUFFERACCESSFLAGS   eFlags,
        __in BOOL                bPooledBlock
        )
    {
        BOOL bExplicitEmpty = pTemplate->IsInitialValueEmpty();
        void * lpvInitialValue = const_cast<void*>(pTemplate->GetInitialValue());
        UINT cbTemplateInitialValue = pTemplate->GetInitialValueSizeBytes();
        UINT uiStride = pTemplate->GetStride();
        UINT uiRecords = bExplicitEmpty ? 0 : uiAllocationSizeBytes/uiStride;
        BOOL bLogicallyEmpty = (uiAllocationSizeBytes == 0 || bExplicitEmpty);

        HOSTMEMORYEXTENT extent(lpvInitialValue, cbTemplateInitialValue, FALSE);
        Datablock * pBlock = new Datablock(pAsyncContext,
                                           pTemplate,
                                           uiAllocationSizeBytes,
                                           0, 
                                           0,
                                           eFlags,
                                           DBCTLC_NONE,
                                           &extent,
                                           uiRecords,
                                           TRUE);
        assert(pBlock != NULL);
        if(!bPooledBlock) {
            pBlock->AddRef();
        }
        if(bLogicallyEmpty) {
            pBlock->Lock();
            pBlock->m_bLogicallyEmpty = TRUE;
            assert(pBlock->GetRecordCount() == 0);
            assert(pBlock->m_cbFinalized[XDIM] == 0);
            if(lpvInitialValue == NULL) {
                // in this case we want to be sure to force 
                // creation of a backing channel even though the logical
                // size of the block is 0...
                assert(pBlock->HasValidChannels());
            }
            pBlock->Unlock();
        } else {
            pBlock->Lock();
            pBlock->m_bDeviceViewsMemsettable = pTemplate->IsInitialValueMemsettable();
            pBlock->Unlock();
        }
        return pBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a new datablock for an output port. Uses the initial value if possible.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/19/2012. </remarks>
    ///
    /// <param name="pDispatchAccelerator"> [in,out] If non-null, the dispatch accelerator. </param>
    /// <param name="pAsyncContext">        [in,out] If non-null, context for asynchronous ops. </param>
    /// <param name="pTemplate">            [in,out] If non-null, the template. </param>
    /// <param name="cbDestSize">           Size of the destination. </param>
    /// <param name="eFlags">               The flags. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Datablock::CreateDestinationBlock(
        __in Accelerator * pDispatchAccelerator,
        __in AsyncContext * pAsyncContext,
        __in DatablockTemplate * pTemplate,
        __in UINT cbDestSize,
        __in BUFFERACCESSFLAGS eFlags,
        __in BOOL bPooledBlock
        )
    {
        Datablock * pDestBlock = NULL;

        // If the output port has a template, we want to use its stride information, as well as use any
        // initial value specified in the template to create an output buffer for the dispatch
        // accelerator If there is no template, or no initial data specification in the template, then
        // just materialize an accelerator-side empty buffer. 
        void * lpvInitData = NULL;
        UINT cbInitDataSize = 0;
        UINT cbDestStride = 1;
        BOOL bMemsettable = FALSE;

        if(pTemplate != NULL) {
            bMemsettable = pTemplate->HasInitialValue() && pTemplate->IsInitialValueMemsettable();
            lpvInitData = (void*) pTemplate->GetInitialValue();
            cbInitDataSize = pTemplate->GetInitialValueSizeBytes();
            cbDestStride = pTemplate->GetStride();
        }

        // Create the new datablock. Note that we create it with an empty control code, regardless of
        // whether the output port is part of the control routing network. This is because the output
        // port is responsible for marking the block when the outputs are bound before dispatch. 
        // If the template has an initial value that makes materialization of whatever initial view 
        // is required possible with memset, we will use the CreateInitialValueBlock API. Otherwise,
        // we'll build the block up by hand. 

        UINT cbAllocation = cbDestSize * cbDestStride;
         
        if(bMemsettable) {

            if(pAsyncContext == NULL || !pAsyncContext->SupportsExplicitAsyncOperations())
                pAsyncContext = pDispatchAccelerator->GetAsyncContext(ASYNCCTXT_XFERDTOD);

            pDestBlock = Datablock::CreateInitialValueBlock(pAsyncContext, pTemplate, cbAllocation, eFlags, bPooledBlock);
            pDestBlock->Lock();
            if(pDestBlock->IsPooled()) {
                assert(pDestBlock->RefCount() == 1);
            }
            pDestBlock->m_bDeviceViewsMemsettable = TRUE;
            pDestBlock->Unlock();

        } else {

            pDestBlock = Datablock::CreateDatablock(NULL, 
                                                    cbAllocation, 0, 0, 
                                                    eFlags,
                                                    DBCTLC_NONE);
                                     
            pDestBlock->Lock();
            if(!bPooledBlock) {
                pDestBlock->AddRef();
            } else {
                assert(pDestBlock->RefCount() == 1);
            }

            // Seal the block. If we *don't* have initial data, we can materialize immediately on
            // the device-side, because we don't have any communication to do. If we *do* have initial
            // data, we have a bit of a difficult choice. To do an asynchronous transfer we typically need
            // information beyond what is available in this method (e.g. the cuda stream associated with the
            // task to which the new data will be bound). In such a case
            // install the new block on the allocation port.
            pDestBlock->m_pTemplate = pTemplate;
            pDestBlock->Seal(cbDestSize, cbDestSize*cbDestStride, 0, 0);
            HOSTMEMORYEXTENT extent(lpvInitData, cbInitDataSize, FALSE);
            ctprofile_view_update_start_a_db(pDestBlock, pDispatchAccelerator, CET_BLOCK_CREATE);
            pDestBlock->AllocateBuffer(pDispatchAccelerator,
                                       pAsyncContext,
                                       DBDATA_IDX,
                                       cbAllocation, 
                                       &extent);
                
            ctprofile_view_update_end_db(pDestBlock, 
                                         HOST_MEMORY_SPACE_ID, 
                                         BSTATE_EXCLUSIVE, 
                                         ((lpvInitData != NULL) && !pDispatchAccelerator->IsHost()));
            pDestBlock->Unlock();
        } 

        return pDestBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a new datablock for an output port. Uses the initial value if possible.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/19/2012. </remarks>
    ///
    /// <param name="pDispatchAccelerator">     [in,out] If non-null, the dispatch accelerator. </param>
    /// <param name="pAsyncContext">            [in,out] If non-null, context for asynchronous ops. </param>
    /// <param name="pDataTemplate">            [in,out] If non-null, the template. </param>
    /// <param name="pMetaChannelTemplate">     [in,out] If non-null, the meta channel template. </param>
    /// <param name="pTemplateChannelTemplate"> [in,out] If non-null, the template channel template. </param>
    /// <param name="uiRecordCount">            Number of records in this block. Used only when the
    ///                                         block describes part of a record stream. </param>
    /// <param name="cbDataSizeBytes">          Size of the destination. </param>
    /// <param name="cbMetaSizeBytes">          The meta size in bytes. </param>
    /// <param name="cbTemplateSizeBytes">      The template size in bytes. </param>
    /// <param name="eFlags">                   The flags. </param>
    /// <param name="bPooledBlock">             True if this block is part of a block pool. PTask
    ///                                         pools blocks on output ports to avoid latency for
    ///                                         allocating blocks that required by every dispatch and
    ///                                         have predictable geometry. If a block is pooled, it
    ///                                         returns to its block pool when it is released. If it
    ///                                         is not, it is deleted when its refcount hits zero. </param>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Datablock::CreateDestinationBlock(
        __in Accelerator * pDispatchAccelerator,
        __in AsyncContext * pAsyncContext,
        __in DatablockTemplate * pDataTemplate,
        __in DatablockTemplate * pMetaChannelTemplate,
        __in DatablockTemplate * pTemplateChannelTemplate,
        __in UINT uiRecordCount,
        __in UINT cbDataSizeBytes,
        __in UINT cbMetaSizeBytes,
        __in UINT cbTemplateSizeBytes,
        __in BUFFERACCESSFLAGS eFlags,
        __in BOOL bPooledBlock
        )
    {
        Datablock * pDestBlock = NULL;
        BOOL bTransferOccurred = FALSE;

        // Create the new datablock. Note that we create it with an empty control code, regardless of
        // whether the output port is part of the control routing network. This is because the output
        // port is responsible for marking the block when the outputs are bound before dispatch. 

        pDestBlock = Datablock::CreateDatablock(NULL, 
                                                cbDataSizeBytes,
                                                cbMetaSizeBytes,
                                                cbTemplateSizeBytes,
                                                eFlags,
                                                DBCTLC_NONE);
        if(!bPooledBlock) {
            pDestBlock->AddRef();
        }
        pDestBlock->Lock();

        // Seal the block. If we *don't* have initial data, we can materialize immediately on
        // the device-side, because we don't have any communication to do. If we *do* have initial
        // data, we have a bit of a difficult choice. To do an asynchronous transfer we typically need
        // information beyond what is available in this method (e.g. the cuda stream associated with the
        // task to which the new data will be bound). In such a case
        // install the new block on the allocation port.
        pDestBlock->m_pTemplate = pDataTemplate;
        pDestBlock->Seal(uiRecordCount, cbDataSizeBytes, cbMetaSizeBytes, cbTemplateSizeBytes);

        // If we have templates for the various channels, and initial values are specified,
        // use the initial value specified in the template to create an output buffer for the dispatch
        // accelerator. If there is no template, or no initial data specification in the template, then
        // just materialize an accelerator-side empty buffer. 

        for(UINT uiChannelIndex=DBDATA_IDX; uiChannelIndex<NUM_DATABLOCK_CHANNELS; uiChannelIndex++) {

            DatablockTemplate * pTemplate = NULL;
            void * lpvInitData = NULL;
            UINT cbInitDataSize = 0;
            UINT cbDestStride = 1;
            UINT cbAllocation = 0;
            switch(uiChannelIndex) {
            case DBDATA_IDX:     pTemplate = pDataTemplate;            cbAllocation = cbDataSizeBytes; break;
            case DBMETADATA_IDX: pTemplate = pMetaChannelTemplate;     cbAllocation = cbMetaSizeBytes; break;
            case DBTEMPLATE_IDX: pTemplate = pTemplateChannelTemplate; cbAllocation = cbTemplateSizeBytes; break;
            }

            if(cbAllocation == 0) 
                continue; // no buffer required.

            if(pTemplate != NULL) {
                lpvInitData = (void*) pTemplate->GetInitialValue();
                cbInitDataSize = pTemplate->GetInitialValueSizeBytes();
                cbDestStride = pTemplate->GetStride();
            }
            HOSTMEMORYEXTENT extent(lpvInitData, cbInitDataSize, FALSE);
            ctprofile_view_update_start_a_db(pDestBlock, pDispatchAccelerator, CET_BLOCK_CREATE);
            pDestBlock->AllocateBuffer(pDispatchAccelerator,
                                       pAsyncContext,
                                       uiChannelIndex,
                                       cbAllocation, 
                                       &extent);
            bTransferOccurred = 
                lpvInitData != NULL &&
                pDispatchAccelerator->GetMemorySpaceId() != HOST_MEMORY_SPACE_ID;
            ctprofile_view_update_end_db(pDestBlock, HOST_MEMORY_SPACE_ID, BSTATE_EXCLUSIVE, bTransferOccurred);
        }
        pDestBlock->Unlock();
        return pDestBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a buffer dims descriptor block. </summary>
    ///
    /// <remarks>   crossbac, 6/19/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    If non-null, context for the asynchronous. </param>
    /// <param name="pDescribedBlock">  [in] block whose size we are describing. </param>
    /// <param name="pChannelTemplate"> [in,out] If non-null, the channel template. </param>
    ///
    /// <returns>   null if it fails, else a new block. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Datablock::CreateBufferDimensionsDescriptorBlock(
        __in AsyncContext * pAsyncContext,
        __in Datablock * pDescribedBlock,
        __in DatablockTemplate * pChannelTemplate
        )
    {
        assert(pDescribedBlock != NULL);
        
        pDescribedBlock->Lock();
        DatablockTemplate* pTemplate = pDescribedBlock->GetTemplate();
        if(pTemplate == NULL) {
            assert(pTemplate != NULL);
            PTask::Runtime::HandleError("%s: Cannot use buffer dims descriptor binding when no template is present!",
                                        __FUNCTION__);
            return NULL;
        } 
        BUFFERDIMENSIONS vBlockDims(pTemplate->GetBufferDimensions(DBDATA_IDX));
        pDescribedBlock->Unlock();

        UINT uiValueSize = sizeof(BUFFERDIMENSIONS);
        BUFFERACCESSFLAGS eFlags = PT_ACCESS_ACCELERATOR_READ | PT_ACCESS_IMMUTABLE;
        HOSTMEMORYEXTENT extent(&vBlockDims, uiValueSize, FALSE);
        Datablock * pSizeBlock = new Datablock(pAsyncContext,       // async context object
                                               pChannelTemplate,    // no template required
                                               uiValueSize,         // data channel: size of the integer value
                                               0,                   // metadata channel: nothing
                                               0,                   // template channel: nothing
                                               eFlags,              // need only read the value GPU side
                                               DBCTLC_NONE,          // no control code
                                               &extent,             // address of the record count
                                               1,                   // there is one record in this block 
                                               TRUE);               // materialize this block

        pSizeBlock->AddRef();
        pSizeBlock->Lock();
        pSizeBlock->m_bDeviceViewsMemsettable = FALSE;
        pSizeBlock->Unlock();
        return pSizeBlock;
    }
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a size descriptor block. A size descriptor block is a datablock with a
    ///             single record containing an integer-value record-count/size.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/19/2012. </remarks>
    ///
    /// <param name="pAsyncContext">    If non-null, context for the asynchronous. </param>
    /// <param name="pDescribedBlock">  [in] block whose record count and/or size the new block
    ///                                 should describe. </param>
    /// <param name="pChannelTemplate"> If non-null, the channel template. </param>
    /// <param name="uiValueSize">      Size of the value. </param>
    ///
    /// <returns>   null if it fails, else else a new datablock with an integer-valued buffer whose
    ///             contents is the number of records in the pDescribedBlock parameter. The size of
    ///             the buffer is determined by the uiValueSize parameter.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Datablock::CreateSizeDescriptorBlock(
        __in AsyncContext *      pAsyncContext,
        __in Datablock *         pDescribedBlock,
        __in UINT                uiValueSize
        )
    {
        assert(uiValueSize != 0);
        assert(pDescribedBlock != NULL);
        
        pDescribedBlock->Lock();
        UINT uiTemplateRecordCount = 0;
        UINT uiRecordCount = pDescribedBlock->GetRecordCount();
        DatablockTemplate* pTemplate = pDescribedBlock->GetTemplate();
        if(pTemplate != NULL) {
            UINT uiStride = pTemplate->GetStride();
            UINT cbBlock = pDescribedBlock->GetDataBufferLogicalSizeBytes();
            uiTemplateRecordCount = cbBlock/uiStride;
        } 
        if(uiRecordCount != 0 && uiTemplateRecordCount != 0) 
            assert(uiRecordCount == uiTemplateRecordCount);

        // if we have a record count of 0, it can be the case that the count was never set (in which
        // case we can safely defer to the datablock template if one is present), or it may be the case
        // that the block is logically empty. If it is truly empty, be sure to preserve that fact when
        // creating  a size descriptor block! 
        if(pDescribedBlock->IsLogicallyEmpty()) {
            assert(uiRecordCount == 0);
        } else {
            assert(!((uiTemplateRecordCount > uiRecordCount) && (uiRecordCount > 0)));
            uiRecordCount = max(uiRecordCount, uiTemplateRecordCount);
        }
        pDescribedBlock->Unlock();

        Datablock * pSizeBlock = NULL;
        pSizeBlock = GlobalPoolManager::RequestBlock(m_gSizeDescriptorTemplate, 
                                                     uiValueSize, 
                                                     0, 
                                                     0);
        if(pSizeBlock != NULL) {

            pSizeBlock->AddRef();
            pSizeBlock->Lock();
            pSizeBlock->m_bDeviceViewsMemsettable = TRUE;
            void * pData = pSizeBlock->GetDataPointer(TRUE, FALSE);
            *((UINT*)pData) = uiRecordCount;
            pSizeBlock->Unlock();

        } else {

            BUFFERACCESSFLAGS eFlags = PT_ACCESS_ACCELERATOR_READ | PT_ACCESS_IMMUTABLE;
            HOSTMEMORYEXTENT extent(&uiRecordCount, uiValueSize, FALSE);
            pSizeBlock = new Datablock(pAsyncContext,              // async context object
                                       m_gSizeDescriptorTemplate,  // all int-size blocks use the same template!
                                       uiValueSize,                // data channel: size of the integer value
                                       0,                          // metadata channel: nothing
                                       0,                          // template channel: nothing
                                       eFlags,                     // need only read the value GPU side
                                       DBCTLC_NONE,                 // no control code
                                       &extent,                    // address of the record count
                                       1,                          // there is one record in this block 
                                       TRUE);                      // materialize this block

            pSizeBlock->AddRef();
            pSizeBlock->Lock();
            pSizeBlock->m_bDeviceViewsMemsettable = TRUE;
            pSizeBlock->Unlock();
        }
        return pSizeBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates an empty size descriptor block. </summary>
    ///
    /// <remarks>   crossbac, 8/21/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the new empty size descriptor block. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock * 
    Datablock::CreateEmptySizeDescriptorBlock(
        void
        )
    {
        UINT uiRecordCount = 0;
        UINT uiValueSize = sizeof(uiRecordCount);
        BUFFERACCESSFLAGS eFlags = PT_ACCESS_ACCELERATOR_READ | PT_ACCESS_IMMUTABLE;
        HOSTMEMORYEXTENT extent(&uiRecordCount, uiValueSize, FALSE);
        Datablock * pSizeBlock = new Datablock(NULL,                       // async context object
                                                m_gSizeDescriptorTemplate,  // all int-size blocks use the same template!
                                                uiValueSize,                // data channel: size of the integer value
                                                0,                          // metadata channel: nothing
                                                0,                          // template channel: nothing
                                                eFlags,                     // need only read the value GPU side
                                                DBCTLC_NONE,                 // no control code
                                                &extent,                    // address of the record count
                                                1,                          // there is one record in this block 
                                                TRUE);                      // materialize this block
        pSizeBlock->Lock();
        pSizeBlock->m_bDeviceViewsMemsettable = TRUE;
        pSizeBlock->Unlock();
        return pSizeBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a block containing the unsigned integer-valued uid
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/19/2012. </remarks>
    ///
    /// <param name="pAsyncContext">        If non-null, context for the asynchronous. </param>
    /// <param name="uiBlockIdentifier">    Information describing the control. </param>
    /// <param name="uiValueSize">          Size of the value. </param>
    ///
    /// <returns>   null if it fails, else a new block. </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *
    Datablock::CreateUniqueIdentifierBlock(
        __in AsyncContext * pAsyncContext,
        __in UINT           uiBlockIdentifier,
        __in UINT           uiValueSize
        )
    {
        // next construct a new datablock with the code 
        // in the data channel of the datablock buffer map.
        assert(uiValueSize == 4 && "other sizes unimplemented (un-needed?)");
        HOSTMEMORYEXTENT extent(&uiBlockIdentifier, sizeof(uiBlockIdentifier), FALSE);
        Datablock * pUIDBlock  = new Datablock(pAsyncContext,       // async context object
                                               NULL,                // no template required
                                               uiValueSize,         // data channel: size of the integer value
                                               0,                   // metadata channel: nothing
                                               0,                   // template channel: nothing
                                               PT_ACCESS_DEFAULT,   // need only read the value GPU side
                                               DBCTLC_NONE,         // no control code (it's in the main data channel buffer!)
                                               &extent,             // memory extent describing the control information
                                               1,                   // one record in this block
                                               TRUE);               // materialize this block

        pUIDBlock->AddRef();
        return pUIDBlock;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates a block containing control information, either a control code or
    ///             information about whether a control code matched a given predicate. This function
    ///             replaces "CreateExplicitEOFBlock", which specialized the case where kernel code
    ///             needs access to a TRUE/FALSE value indicating whether the described block is the
    ///             last block in a stream. Since many other cases, such as matching a DBCTL_BOF, or
    ///             the block UID have essentially the same information (wrap a 4 or 8 byte value in
    ///             the data channel of a block), it made sense to generalize it into a single method
    ///             that can handle all the cases where we expose control information to kernel code
    ///             through a datablock. (Frankly, this probably won't handle *all* such cases, but
    ///             it handles all the ones we know about as of 10/26/2012!)
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/19/2012. </remarks>
    ///
    /// <param name="pAsyncContext">            If non-null, context for asynchronous operations. </param>
    /// <param name="luiControlInformation">    Information describing the control. </param>
    /// <param name="uiValueSize">              Size of the value. </param>
    ///
    /// <returns>   null if it fails, else a new datablock with an integer-valued buffer whose
    ///             contents are exactly the control information parameter. The size of the buffer is
    ///             determined by the uiValueSize parameter.
    ///             </returns>
    ///-------------------------------------------------------------------------------------------------

    Datablock *
    Datablock::CreateControlInformationBlock(
        __in AsyncContext * pAsyncContext,
        __in CONTROLSIGNAL  luiControlInformation,
        __in UINT           uiValueSize
        )
    {        
        // next construct a new datablock with the code 
        // in the data channel of the datablock buffer map.
        assert(uiValueSize == 4 && "other sizes unimplemented (un-needed?)");
        HOSTMEMORYEXTENT extent(&luiControlInformation, sizeof(luiControlInformation), FALSE);
        Datablock * pCtlBlock  = new Datablock(pAsyncContext,       // async context object
                                               NULL,                // no template required
                                               uiValueSize,         // data channel: size of the integer value
                                               0,                   // metadata channel: nothing
                                               0,                   // template channel: nothing
                                               PT_ACCESS_DEFAULT,   // need only read the value GPU side
                                               DBCTLC_NONE,         // no control code (it's in the main data channel buffer!)
                                               &extent,             // memory extent describing the control information
                                               1,                   // one record in this block
                                               TRUE);               // materialize this block

        pCtlBlock->AddRef();
        return pCtlBlock;
    }

#ifdef PROFILE_DATABLOCKS

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Datablock toString operator. This is a debug utility. 
    /// 			This version is intended only to be used when the allocation
    /// 			tracker is enabled.</summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="os">   [in,out] The output stream. </param>
    /// <param name="port"> The block. </param>
    ///
    /// <returns>   The string result. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::ostream& operator <<(
        std::ostream &os, 
        Datablock * pBlock
        ) 
    { 
        std::string strPools = "";
        std::string strPortBindings = "";
        std::string strProducerTask = "";
        std::string strClone = "";
        assert(pBlock->m_pDatablockProfiler != NULL);
        if(pBlock->m_pProducerTask || pBlock->m_pDatablockProfiler->m_vTaskBindings.size()) {
            BOOL bAnyTasks = FALSE;
            for(std::set<Task*>::iterator si=pBlock->m_pDatablockProfiler->m_vTaskBindings.begin();
                si!=pBlock->m_pDatablockProfiler->m_vTaskBindings.end(); si++) {
                if(!bAnyTasks) {
                    strProducerTask += "\n\tPT:[";
                    bAnyTasks = TRUE;
                } else {
                    strProducerTask += ", ";
                }
                strProducerTask += pBlock->m_pDatablockProfiler->m_vTaskNames[*si];
            }
            if(bAnyTasks)
                strProducerTask += "]";
        }
        if(pBlock->m_pDatablockProfiler->m_vPortBindings.size()) {
            BOOL bAnyPorts = FALSE;
            for(std::set<Port*>::iterator si=pBlock->m_pDatablockProfiler->m_vPortBindings.begin();
                si!=pBlock->m_pDatablockProfiler->m_vPortBindings.end(); si++) {
                if(!bAnyPorts) {
                    strPortBindings += "\n\tPB:[";
                    bAnyPorts = TRUE;
                } else {
                    strPortBindings += ", ";
                }
                strPortBindings += pBlock->m_pDatablockProfiler->m_vPortNames[*si];
            }
            if(bAnyPorts) {
                strPortBindings += "]";
            }
        }
        if(pBlock->m_pDatablockProfiler->m_vPools.size()) {
            BOOL bAnyPools = FALSE;
            std::map<BlockPoolOwner*, std::string>::iterator si;
            for(si=pBlock->m_pDatablockProfiler->m_vPools.begin(); si!=pBlock->m_pDatablockProfiler->m_vPools.end(); si++) {
                if(!bAnyPools) {
                    strPools += "\n\tPOOLS:[";
                    bAnyPools = TRUE;
                } else {
                    strPools += ", ";
                }
                strPools += si->second;
            }
            if(bAnyPools) {
                strPools += "]";
            }
        }
        pBlock->Lock();
        UINT uiLogicalSize = pBlock->GetChannelAllocationSizeBytes(0);
        size_t szDeviceSize = 0;
        std::map<UINT, size_t> vBlockFreeBytes;
        if(pBlock->GetInstantiatedBufferSizes(vBlockFreeBytes)) {
            std::map<UINT, size_t>::iterator mi;
            for(mi=vBlockFreeBytes.begin(); mi!=vBlockFreeBytes.end(); mi++) {
                if(mi->first == HOST_MEMORY_SPACE_ID) continue;
                szDeviceSize = max(szDeviceSize, mi->second);
            }
        }
        std::string strTemplateName = "";
        if(pBlock->GetTemplate()) {
            char * lpszTemplateName = pBlock->GetTemplate()->GetTemplateName();
            strTemplateName += "(T:";
            strTemplateName += lpszTemplateName;
            strTemplateName += ") ";
        }
        pBlock->Unlock();
        if(pBlock->m_bIsClone)
            strClone += " (clone) ";
        os 
            << "DB#" 
            << pBlock->m_uiDBID
            << "[" << uiLogicalSize << " bytes, (" << szDeviceSize << " dev-bytes), " << strTemplateName
            << " CTL=" << ((int)pBlock->m_vControlSignalStack.top()) <<"]"
            << " rc: " << pBlock->m_uiRefCount
            << strClone
            << strPools
            << strProducerTask
            << strPortBindings;
        return os;
    }
#else
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Datablock toString operator. This is a debug utility. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="os">   [in,out] The output stream. </param>
    /// <param name="port"> The block. </param>
    ///
    /// <returns>   The string result. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::ostream& operator <<(std::ostream &os, Datablock * pBlock) { 
        os 
            << "DB#" 
            << pBlock->m_uiDBID;
        return os;
    }
#endif

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a string describing this refcount object. Allows subclasses to
    ///             provide overrides that make leaks easier to find when detected by the
    ///             rc profiler. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/9/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the rectangle profile descriptor. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::string
    Datablock::GetRCProfileDescriptor(
        VOID
        )
    {
        stringstream ss;
        ss << this;
        return ss.str();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Debug dump. </summary>
    ///
    /// <remarks>   Crossbac, 7/18/2013. </remarks>
    ///
    /// <param name="ss">               [in,out] If non-null, the ss. </param>
    /// <param name="pcsSSLock">        [in,out] If non-null, the pcs ss lock. </param>
    /// <param name="szLabel">          [in,out] If non-null, the label. </param>
    /// <param name="uiChannelIndex">   Zero-based index of the channel. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::DebugDump(
        std::ostream* ss,
        CRITICAL_SECTION* pcsSSLock,
        char * szTaskLabel,
        char * szLabel,
        UINT uiChannelIndex
        )
    {
        Lock();
        Accelerator * pLastAccelerator = GetMostRecentAccelerator();  
        Unlock();
        if(pLastAccelerator != NULL) pLastAccelerator->Lock();
        if(pcsSSLock != NULL) EnterCriticalSection(pcsSSLock);
        if(ss == NULL) ss = &std::cerr;
        Lock();
        PBuffer * pBuffer = GetPlatformBuffer(pLastAccelerator, uiChannelIndex);
        (*ss) << ((szTaskLabel == NULL)?"":szTaskLabel) << " "
              << ((szLabel == NULL)?"":szLabel) << " ("
              << pBuffer->GetAllocationExtentBytes() << " bytes)"           
              << std::endl;   
        UINT uiDumpStartIndex = 0;
        pBuffer->DebugDump(PTask::Runtime::GetDumpType(),
                           uiDumpStartIndex,
                           PTask::Runtime::GetDumpLength(),
                           PTask::Runtime::GetDumpStride()); 
        Unlock();
        if(pcsSSLock) LeaveCriticalSection(pcsSSLock);
        if(pLastAccelerator != NULL) pLastAccelerator->Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record port binding. </summary>
    ///
    /// <remarks>   Crossbac, 9/19/2012. </remarks>
    ///
    /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::RecordBinding(
        __in Port * pPort
        )
    {
        UNREFERENCED_PARAMETER(pPort);
        dbprofile_record_pb(pPort);
        ctprofile_record_pb(pPort);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record task binding. </summary>
    ///
    /// <remarks>   Crossbac, 9/19/2012. </remarks>
    ///
    /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::RecordBinding(
        __in Task * pTask
        )
    {
        UNREFERENCED_PARAMETER(pTask);
        dbprofile_record_tb(pTask);
        ctprofile_record_tb(pTask);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record binding. </summary>
    ///
    /// <remarks>   Crossbac, 9/20/2012. </remarks>
    ///
    /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::RecordBinding(
        __in Port * pPort, 
        __in Task * pTask, 
        __in Port * pIOConsumer
        )
    {
        UNREFERENCED_PARAMETER(pPort);
        UNREFERENCED_PARAMETER(pTask);
        UNREFERENCED_PARAMETER(pIOConsumer);
        dbprofile_record_bindio(pPort, pTask, pIOConsumer);
        ctprofile_record_bindio(pPort, pTask, pIOConsumer);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is resized block. </summary>
    ///
    /// <remarks>   crossbac, 8/30/2013. </remarks>
    ///
    /// <returns>   true if resized block, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::IsResizedBlock(
        VOID
        ) 
    {
        assert(LockIsHeld());
        return m_bBlockResized;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Clears the resize flags. </summary>
    ///
    /// <remarks>   crossbac, 8/30/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Datablock::ClearResizeFlags(
        VOID
        )
    {
        assert(LockIsHeld());
        m_bBlockResized = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets instantiated buffer sizes. Populate a map from memory space id to 
    ///             the number of bytes of PBuffer backing space that is actually allocated
    ///             for this block. This method is a tool to help the runtime (specifically, the
    ///             GC) figure out when a forced GC sweep might actually help free up some
    ///             GPU memory. 
    ///             
    ///             No lock is required of the caller because it is expected to be called by the
    ///             GC--the datablock should therefore only be accessible from one thread context,
    ///             so there is no danger of the results becoming stale after the call completes.
    ///             ***SO, if you decide to repurpose this method and call it  from any other context, 
    ///             be sure to lock the datablock first. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 9/7/2013. </remarks>
    ///
    /// <param name="vBlockFreeBytes">  [in,out] The block free in bytes. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::GetInstantiatedBufferSizes(
        __inout std::map<UINT, size_t>& vBlockFreeBytes
        )
    {
        Lock(); 
        BOOL bResult = FALSE;
        UINT nMemSpaceId = 0;
        UINT nChannelIndex = 0;
        for(nMemSpaceId=HOST_MEMORY_SPACE_ID; 
            nMemSpaceId<MemorySpace::GetNumberOfMemorySpaces(); 
            nMemSpaceId++) {
            size_t szMemSpaceBytes = 0;
            if(vBlockFreeBytes.find(nMemSpaceId) == vBlockFreeBytes.end()) 
                vBlockFreeBytes[nMemSpaceId] = 0;
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[nMemSpaceId];
            for(nChannelIndex=PTask::DBDATA_IDX;
                nChannelIndex<PTask::NUM_DATABLOCK_CHANNELS;
                nChannelIndex++) {
                if(pEntry->pBuffers[nChannelIndex] != NULL) {
                    PBuffer * pBuffer = pEntry->pBuffers[nChannelIndex];
                    szMemSpaceBytes += pBuffer->GetAllocationExtentBytes();
                }
            }
            vBlockFreeBytes[nMemSpaceId] = vBlockFreeBytes[nMemSpaceId] + szMemSpaceBytes;
            bResult |= (szMemSpaceBytes != 0);
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Releases the physical buffers backing datablock channels for the accelerator
    ///             whose memory space has the given ID, uiMemSpaceID. 
    ///             
    ///             No lock is required of the caller because it is expected to be called by the
    ///             GC--the datablock should therefore only be accessible from one thread context,
    ///             so there is no danger of the results becoming stale after the call completes.
    ///             ***SO, if you decide to repurpose this method and call it  from any other context, 
    ///             be sure to lock the datablock first. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 9/7/2013. </remarks>
    ///
    /// <param name="uiMemSpaceID"> Identifier for the memory space. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Datablock::ReleasePhysicalBuffers(
        __in UINT nMemSpaceId
        )
    {
        Lock(); 
        BOOL bResult = FALSE;
        UINT nChannelIndex = 0;
        BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[nMemSpaceId];
        for(nChannelIndex=PTask::DBDATA_IDX;
            nChannelIndex<PTask::NUM_DATABLOCK_CHANNELS;
            nChannelIndex++) {
            if(pEntry->pBuffers[nChannelIndex] != NULL) {
                bResult = TRUE;
                PBuffer * pBuffer = pEntry->pBuffers[nChannelIndex];
                pEntry->pBuffers[nChannelIndex] = NULL;
                delete pBuffer;
            }
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Invalidate any device views, updating the host view and releasing any physical
    ///             buffers, according to the flags. Return the number of bytes freed.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 9/8/2013. Use this method with caution--this is really intended as a GC-
    ///             facing API, but I am making it public as a temporary remedy for what looks like a
    ///             Dandelion memory leak. Once blocks are consumed by the DandelionBinaryReader, we
    ///             know them to be dead, but they do not always have non-zero refcount. Since we
    ///             know that the host view was materialized, we can usually safely just reclaim the
    ///             device buffers and elide the step of syncing the host because the host was synced
    ///             before the deserialization.
    ///             </remarks>
    ///
    /// <param name="bSynchronizeHostView"> true to synchronize host view. </param>
    /// <param name="bRelinquishBuffers">   true to release backing buffers. </param>
    ///
    /// <returns>   the number of bytes released. </returns>
    ///-------------------------------------------------------------------------------------------------

    size_t 
    Datablock::InvalidateDeviceViews(
        __in BOOL bSynchronizeHostView,
        __in BOOL bRelinquishBuffers
        )
    {
        Lock(); 
        UNREFERENCED_PARAMETER(bSynchronizeHostView);
        UNREFERENCED_PARAMETER(!bRelinquishBuffers);
        assert(!bSynchronizeHostView);  // only supported case for now
        assert(bRelinquishBuffers);     // only supported case for now
        size_t szResult = 0;
        UINT nChannelIndex = 0;
        for(UINT nMemSpaceId=HOST_MEMORY_SPACE_ID+1;
            nMemSpaceId < MemorySpace::GetNumberOfMemorySpaces(); 
            nMemSpaceId++) {
            BUFFER_MAP_ENTRY * pEntry = m_ppBufferMap[nMemSpaceId];
            for(nChannelIndex=PTask::DBDATA_IDX;
                nChannelIndex<PTask::NUM_DATABLOCK_CHANNELS;
                nChannelIndex++) {
                if(pEntry->pBuffers[nChannelIndex] != NULL) {
                    PBuffer * pBuffer = pEntry->pBuffers[nChannelIndex];
                    szResult = max(szResult, pBuffer->GetAllocationExtentBytes());
                    pEntry->pBuffers[nChannelIndex] = NULL;
                    pEntry->eState = BSTATE_NO_ENTRY;
                    delete pBuffer;
                }
            }
        }
        Unlock();
        return szResult;
    }

};
