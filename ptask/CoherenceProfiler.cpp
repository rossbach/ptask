///-------------------------------------------------------------------------------------------------
// file:	CoherenceProfiler.cpp
//
// summary:	Implements the coherence profiler class
///-------------------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "ptaskutils.h"
#include "datablock.h"
#include "CoherenceProfiler.h"
#include "Port.h"
#include "task.h"
#include <assert.h>
#include <sstream>
#include <iomanip>
using namespace std;
using namespace PTask::Runtime;

// does this entry have an up-to-date copy of the data?
#define valid(x) ((x) == BSTATE_SHARED || (x) == BSTATE_EXCLUSIVE)


namespace PTask {

    /// <summary>   The dev to dev migrations with invalidation. </summary>
    LONG     CoherenceProfiler::m_nDToDMigrationsExclusive = 0;

    /// <summary>   The dev to dev migrations with shared state. </summary>
    LONG     CoherenceProfiler::m_nDToDMigrationsShared = 0;

    /// <summary>   The host to dev migrations with invalidation. </summary>
    LONG     CoherenceProfiler::m_nHToDMigrationsExclusive = 0;

    /// <summary>   The host to dev migrations without invalidation. </summary>
    LONG     CoherenceProfiler::m_nHToDMigrationsShared = 0;

    /// <summary>   The dev to host migrations with invalidation. </summary>
    LONG     CoherenceProfiler::m_nDToHMigrationsExclusive = 0;

    /// <summary>   The dev to host migrations without invalidation. </summary>
    LONG     CoherenceProfiler::m_nDToHMigrationsShared = 0;

    /// <summary>   The number of times a coherence event caused multiple
    /// 			valid views to be abandoned. </summary>
    LONG     CoherenceProfiler::m_nMultiViewInvalidations = 0;

    /// <summary>   The number of state transitions whose cause was unspecified. </summary>
    LONG     CoherenceProfiler::m_nCETUnspecified = 0; 

    /// <summary>   The number of state transitions triggered by a binding to task input</summary>
    LONG     CoherenceProfiler::m_nCETBindInput = 0;

    /// <summary>   The number of state transitions triggered by a binding to taks output</summary>
    LONG     CoherenceProfiler::m_nCETBindOutput = 0;

    /// <summary>   The number of state transitions triggered by a binding to a task constant port</summary>
    LONG     CoherenceProfiler::m_nCETBindConstant = 0;

    /// <summary>   The number of state transitions triggered by pushing into multiple consumer channels </summary>
    LONG     CoherenceProfiler::m_nCETDownstreamShare = 0;

    /// <summary>   The number of state transitions triggered by a user request for a pointer in host space</summary>
    LONG     CoherenceProfiler::m_nCETPointerRequest = 0;

    /// <summary>   The number of state transitions triggered by the deletion of the block</summary>
    LONG     CoherenceProfiler::m_nCETBlockDelete = 0;

    /// <summary>   The number of state transitions triggered by the cloning of the block </summary>
    LONG     CoherenceProfiler::m_nCETBlockClone = 0;

    /// <summary>   The number of state transitions triggered by block allocation </summary>
    LONG     CoherenceProfiler::m_nCETBlockCreate = 0;

    /// <summary>   The number of state transitions triggered when we are updating the host view of
    ///             the block, but don't actually have access to the information we need to figure
    ///             out what action triggered the view update. Most likely a user request.
    ///             </summary>
    LONG     CoherenceProfiler::m_nCETHostViewUpdate = 0;

    /// <summary>   The number of state transitions triggered when we are updating the device view of
    ///             the block, but don't actually have access to the information we need to figure
    ///             out what action triggered the view update. Most likely a user request.
    ///             </summary>
    LONG     CoherenceProfiler::m_nCETAcceleratorViewUpdate = 0;

    /// <summary>   The number of state transitions triggered when Buffers are being allocated for a
    ///             block.
    ///             </summary>
    LONG     CoherenceProfiler::m_nCETBufferAllocate = 0;

    /// <summary>   The number of state transitions triggered when a request to grow the buffer
    ///             caused some buffer reallocation and potentially view updates as a side effect.
    ///             </summary>
    LONG     CoherenceProfiler::m_nCETGrowBuffer = 0;

    /// <summary>   The number of state transitions triggered when a request to synthesize 
    /// 			a metadata block caused the traffic </summary>
    LONG     CoherenceProfiler::m_nCETSynthesizeBlock = 0;

    /// <summary>   The number of state transitions triggered when 
    /// 			needed a pinned host buffer in addition to a dev buffer </summary>
    LONG     CoherenceProfiler::m_nCETPinnedHostView = 0;

    /// <summary>   Is the profiler initialised? </summary>
    LONG     CoherenceProfiler::m_nCoherenceProfilerInit = 0;

    /// <summary>   Is the profiler initialised? </summary>
    LONG     CoherenceProfiler::m_nCoherenceProfilerEnabled = 0;

    /// <summary>   true if the coherence tracker should emit copious text. </summary>
    BOOL     CoherenceProfiler::m_bCoherenceProfilerVerbose = FALSE;

    /// <summary>   true if the coherence tracker should emit detailed stats. </summary>
    BOOL     CoherenceProfiler::m_bCoherenceStatisticsDetailed = FALSE;

    /// <summary>   The datablock coherence histories: static view. </summary>
    std::map<UINT, COHERENCEHISTORY*> CoherenceProfiler::m_vHistories;

    CHighResolutionTimer * CoherenceProfiler::m_pTimer = NULL;

    /// <summary>   The coherence profiler lock. Protects the data structures
    /// 			collecting data xfer statistics.
    /// 			</summary>
    CRITICAL_SECTION CoherenceProfiler::m_csCoherenceProfiler;

    /// <summary>   List of names task names. Required because we will no longer have
    /// 			valid task pointers when we check for leaks (all tasks *should* be
    /// 			deleted by that point), and we want to be able to find the task
    /// 			that allocated a block if it was leaked and provide it's name as
    /// 			a debug assist.   
    /// 			</summary>
    std::map<PTask::Task*, std::string> CoherenceProfiler::m_vTaskNames;

    /// <summary>   List of port names. Required because we will no longer have
    /// 			valid port pointers when we check for leaks (all ports *should* be
    /// 			deleted by that point), and we want to be able to find the last
    /// 		    port that touched any leaked blocks. 
    /// 			</summary>
    std::map<PTask::Port*, std::string> CoherenceProfiler::m_vPortNames;

#ifdef PROFILE_MIGRATION

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///
    /// <param name="pDatablock">   [in,out] If non-null, the datablock. </param>
    ///-------------------------------------------------------------------------------------------------

    CoherenceProfiler::CoherenceProfiler(
        __in Datablock * pDatablock
        )
    {
        m_pDatablock = pDatablock;
        InitializeInstanceHistory();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    CoherenceProfiler::~CoherenceProfiler(
        VOID
        )
    {
        DeinitializeInstanceHistory();
        m_pDatablock = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the coherence traffic profiler. </summary>
    ///
    /// <remarks>   Crossbac, 2/24/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CoherenceProfiler::Initialize(
        BOOL bEnable,
        BOOL bVerbose
        )
    {
        if(!PTask::Runtime::GetCoherenceProfileMode())
            return FALSE;
        assert(!m_nCoherenceProfilerInit);
        if(!m_nCoherenceProfilerInit) {
            InitializeCriticalSection(&m_csCoherenceProfiler);
            m_pTimer = new CHighResolutionTimer(gran_msec);
            m_bCoherenceProfilerVerbose = bVerbose;
            m_nCoherenceProfilerInit = TRUE;
            m_bCoherenceStatisticsDetailed = TRUE;
            m_nCoherenceProfilerEnabled = bEnable;
            PTask::Runtime::SetCoherenceProfileMode(bEnable);
            m_pTimer->reset();
            return TRUE;
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the coherence traffic profiler. </summary>
    ///
    /// <remarks>   Crossbac, 2/24/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CoherenceProfiler::Deinitialize(
        VOID
        )
    {
        if(!PTask::Runtime::GetCoherenceProfileMode())
            return FALSE;
        assert(m_nCoherenceProfilerInit);
        if(m_nCoherenceProfilerInit && m_nCoherenceProfilerEnabled) {
            EnterCriticalSection(&m_csCoherenceProfiler);
            std::map<UINT, COHERENCEHISTORY*>::iterator mi;
            for(mi=m_vHistories.begin(); mi!=m_vHistories.end(); mi++) {
                COHERENCEHISTORY * pHistory = mi->second;            
                delete pHistory;
            }
            if(m_pTimer) delete m_pTimer;
            m_pTimer = NULL;
            LeaveCriticalSection(&m_csCoherenceProfiler);
            DeleteCriticalSection(&m_csCoherenceProfiler);
            m_nCoherenceProfilerInit = FALSE;
            return TRUE;
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps the coherence traffic statistics. </summary>
    ///
    /// <remarks>   Crossbac, 2/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    CoherenceProfiler::Report(
        std::ostream& ios
        )
    {
        if(!PTask::Runtime::GetCoherenceProfileMode() ||
            !m_nCoherenceProfilerInit || 
            !m_nCoherenceProfilerEnabled) 
            return;

        stringstream * pss = GetReport();
        stringstream& ss = *pss;
        GetDetailedReport(ss);
        ios << pss->str();
        delete pss;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Coherence tracker get statistics. </summary>
    ///
    /// <remarks>   Crossbac, 9/21/2012. </remarks>
    ///
    /// <returns>   null if it fails, else. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::stringstream*
    CoherenceProfiler::GetReport(
        VOID
        )
    {
        stringstream * pss = new stringstream();
        stringstream& ss = *pss;
        if(!PTask::Runtime::GetCoherenceProfileMode()) {
            return pss;
        }

        EnterCriticalSection(&m_csCoherenceProfiler);
        MergeHistories();

        int nFillWidth = 80;
        ss << endl << endl << "COHERENCE STATISTICS:" << endl;
        ss << setfill('-') << setw(nFillWidth) << "-" << std::endl;
        ss << "DToDMigrationsExclusive   " << m_nDToDMigrationsExclusive  << std::endl;
        ss << "DToDMigrationsShared      " << m_nDToDMigrationsShared     << std::endl;
        ss << "HToDMigrationsExclusive   " << m_nHToDMigrationsExclusive  << std::endl;
        ss << "HToDMigrationsShared      " << m_nHToDMigrationsShared     << std::endl;
        ss << "DToHMigrationsExclusive   " << m_nDToHMigrationsExclusive  << std::endl;
        ss << "DToHMigrationsShared      " << m_nDToHMigrationsShared     << std::endl;
        ss << "MultiViewInvalidations    " << m_nMultiViewInvalidations   << std::endl;
        ss << "CETUnspecified            " << m_nCETUnspecified           << std::endl;
        ss << "CETBindInput              " << m_nCETBindInput             << std::endl;
        ss << "CETBindOutput             " << m_nCETBindOutput            << std::endl;
        ss << "CETBindConstant           " << m_nCETBindConstant          << std::endl;
        ss << "CETDownstreamShare        " << m_nCETDownstreamShare       << std::endl;
        ss << "CETPointerRequest         " << m_nCETPointerRequest        << std::endl;
        ss << "CETBlockDelete            " << m_nCETBlockDelete           << std::endl;
        ss << "CETBlockClone             " << m_nCETBlockClone            << std::endl;
        ss << "CETBlockCreate            " << m_nCETBlockCreate           << std::endl;
        ss << "CETHostViewUpdate         " << m_nCETHostViewUpdate        << std::endl;
        ss << "CETAcceleratorViewUpdate  " << m_nCETAcceleratorViewUpdate << std::endl;
        ss << "CETBufferAllocate         " << m_nCETBufferAllocate        << std::endl;
        ss << "CETGrowBuffer             " << m_nCETGrowBuffer            << std::endl;
        ss << "CETSynthesizeBlock        " << m_nCETSynthesizeBlock       << std::endl;
        ss << "CETPinnedHostView         " << m_nCETPinnedHostView        << std::endl;
        ss << setfill('-') << setw(nFillWidth) << "-" << std::endl;
        LeaveCriticalSection(&m_csCoherenceProfiler);        

        return pss;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps the coherence traffic statistics. </summary>
    ///
    /// <remarks>   Crossbac, 2/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    CoherenceProfiler::GetDetailedReport(
        std::ostream& ss
        )
    {
        if(!m_nCoherenceProfilerInit || !PTask::Runtime::GetCoherenceProfileMode()) 
            return;

        EnterCriticalSection(&m_csCoherenceProfiler);

        BOOL bShowBindHistory = TRUE;
        int nFillWidth = 80;
        int nColumnFillWidth = 4;
        int nHistoryWidth = 40;
        ss << endl << endl << "PER-BLOCK STATISTICS:" << endl;
        ss << setfill('-') << setw(nFillWidth) << "-" << std::endl;
        ss.setf(std::ios_base::left, std::ios_base::adjustfield);
        ss.width(nColumnFillWidth+1);  ss.fill(' '); ss << "DBID";        
        ss.width(nColumnFillWidth);  ss.fill(' '); ss << "D-D";   
        ss.width(nColumnFillWidth);  ss.fill(' '); ss << "D-H";   
        ss.width(nColumnFillWidth);  ss.fill(' '); ss << "H-D";   
        ss.width(nColumnFillWidth);  ss.fill(' '); ss << "H-H";   
        ss.width(nColumnFillWidth+4);  ss.fill(' '); ss << "Bxfr";
        ss.width(nHistoryWidth);     ss.fill(' '); ss << "BINDINGS";
        ss << endl;
        ss << setfill('-') << setw(nFillWidth) << "-" << std::endl;

        std::map<UINT, COHERENCEHISTORY*>::iterator mi;
        for(mi=m_vHistories.begin(); mi!=m_vHistories.end(); mi++) {
            COHERENCEHISTORY * pHistory = mi->second;            
            ss.width(nColumnFillWidth+1);  ss.fill(' '); ss << pHistory->uiDBUID;         
            ss.width(nColumnFillWidth);  ss.fill(' '); ss << pHistory->nDToDCopies;     
            ss.width(nColumnFillWidth);  ss.fill(' '); ss << pHistory->nDToHCopies;     
            ss.width(nColumnFillWidth);  ss.fill(' '); ss << pHistory->nHToDCopies;     
            ss.width(nColumnFillWidth);  ss.fill(' '); ss << pHistory->nHToHCopies;     
            ss.width(nColumnFillWidth+4);  ss.fill(' '); ss << pHistory->nTotalSyncBytes; 
            if(bShowBindHistory) {
                map<__int64, Port*>::iterator pi;
                map<__int64, Port*>::iterator ioci;
                map<__int64, Task*>::iterator ti;
                map<__int64, UINT>::iterator ai;
                map<__int64, UINT>::iterator dai;
                BOOL bFirst = TRUE;
                for(pi=pHistory->pvPortBindHistory->begin();
                    pi!=pHistory->pvPortBindHistory->end();
                    pi++) {
                    __int64 uiTimestamp = pi->first;
                    ioci = pHistory->pvIOCPortBindHistory->find(uiTimestamp);
                    ti = pHistory->pvTaskBindHistory->find(uiTimestamp);
                    ai = pHistory->pvAcceleratorBindHistory->find(uiTimestamp);
                    dai = pHistory->pvDepAcceleratorBindHistory->find(uiTimestamp);
                    if(ti != pHistory->pvTaskBindHistory->end() &&
                       ai != pHistory->pvAcceleratorBindHistory->end()) {
                        if(!bFirst) {
                            ss << endl;
                            ss.width(6*nColumnFillWidth+5);  
                            ss.fill(' ');
                            ss << " ";
                        }
                        bFirst = FALSE;
                        ss 
                            << "<" << std::setw(4) << uiTimestamp % 10000 << ">"
                            << m_vTaskNames[ti->second]
                            << "." << m_vPortNames[pi->second]
                            << "(" << ai->second;
                        if(dai != pHistory->pvDepAcceleratorBindHistory->end()) {                           
                            ss << ":[" << dai->second << "]";
                        }
                        ss << ")";
                        if(ioci != pHistory->pvIOCPortBindHistory->end()) {
                            ss << endl;
                            ss.width(6*nColumnFillWidth+5);  
                            ss.fill(' ');
                            ss << " ";
                            ss 
                                << "<" << std::setw(4) << uiTimestamp % 10000 << ">"
                                << m_vTaskNames[ti->second]
                                << "." << m_vPortNames[ioci->second]
                                << "_O(" << ai->second << ":[--])";                 
                        }
                    }
                }
            }
            ss << endl;            
        }
        LeaveCriticalSection(&m_csCoherenceProfiler);        
    }
    
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Coherence tracker record view update start. </summary>
    ///
    /// <remarks>   Crossbac, 9/18/2012. </remarks>
    ///
    /// <param name="pDatablock">           If non-null, the datablock. </param>
    /// <param name="nDestMemorySpaceID">   Identifier for the memory space. </param>
    /// <param name="eEventType">           Type of the event. </param>
    ///
    /// <returns>   new transition object. </returns>
    ///-------------------------------------------------------------------------------------------------

    COHERENCETRANSITION *
    CoherenceProfiler::RecordViewUpdateStart(
        __in UINT nDestMemorySpaceID, 
        __in COHERENCEEVENTTYPE eEventType
        )
    {
        if(!m_nCoherenceProfilerInit || !PTask::Runtime::GetCoherenceProfileMode())
            return NULL;

        assert(m_pDatablock != NULL);
        assert(m_pDatablock->LockIsHeld());
        assert(!m_bCoherenceProfilerTransitionActive);
        m_bCoherenceProfilerTransitionActive = TRUE;
        COHERENCETRANSITION * pTx = new COHERENCETRANSITION(m_pTimer->elapsed(false));
        pTx->eTriggerEvent = eEventType;
        pTx->uiDstMemorySpaceId = nDestMemorySpaceID;
        for(UINT i=0; i<MemorySpace::GetNumberOfMemorySpaces(); i++) {
            pTx->eStartState[i] = m_pDatablock->m_ppBufferMap[i]->eState;
        }
        return pTx;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Coherence tracker record view update end. </summary>
    ///
    /// <remarks>   Crossbac, 9/18/2012. </remarks>
    ///
    /// <param name="nSrcMemorySpaceID">    Identifier for the source memory space. </param>
    /// <param name="uiRequestedState">     The requested coherence state. This affects whether other
    ///                                     accelerator views require invalidation. </param>
    /// <param name="bTransferOccurred">    The transfer occurred. </param>
    /// <param name="pTx">                  non-null, the state transition descriptor. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CoherenceProfiler::RecordViewUpdateEnd(
        __in UINT nSrcMemorySpaceID, 
        __in BUFFER_COHERENCE_STATE uiRequestedState,
        __in BOOL bTransferOccurred,
        __in COHERENCETRANSITION * pTx
        )
    {
        if(!PTask::Runtime::GetCoherenceProfileMode())
            return;

        assert(m_pDatablock != NULL);
        assert(m_pDatablock->LockIsHeld());
        assert(m_bCoherenceProfilerTransitionActive);
        pTx->eTargetState = uiRequestedState;
        pTx->uiSrcMemorySpaceId = nSrcMemorySpaceID;
        for(UINT i=0; i<MemorySpace::GetNumberOfMemorySpaces(); i++) 
            pTx->eEndState[i] = m_pDatablock->m_ppBufferMap[i]->eState;
        COHERENCEHISTORY * pHistory = m_pCoherenceHistory;
        __int64 uiTimestamp = ::GetTickCount64();
        pTx->Finalize(m_pTimer->elapsed(false), bTransferOccurred);
        (*(pHistory->pvStateHistory))[uiTimestamp] = pTx;
        if(pTx->uiDstMemorySpaceId == HOST_MEMORY_SPACE_ID &&
           pTx->uiSrcMemorySpaceId != HOST_MEMORY_SPACE_ID && 
           bTransferOccurred) {
           assert(valid(pTx->GetFinalState()));
           pHistory->nDToHCopies++;
           bTransferOccurred = TRUE;
        }
        if(pTx->uiSrcMemorySpaceId == HOST_MEMORY_SPACE_ID &&
           pTx->uiDstMemorySpaceId != HOST_MEMORY_SPACE_ID && 
           bTransferOccurred) {
           assert(valid(pTx->GetFinalState()));
           pHistory->nHToDCopies++;
           bTransferOccurred = TRUE;
        }
        if(pTx->uiSrcMemorySpaceId != HOST_MEMORY_SPACE_ID &&
           pTx->uiDstMemorySpaceId != HOST_MEMORY_SPACE_ID && 
           bTransferOccurred) {
           assert(valid(pTx->GetFinalState()));
           pHistory->nDToDCopies++;
           bTransferOccurred = TRUE;
        }
        if(pTx->uiSrcMemorySpaceId == HOST_MEMORY_SPACE_ID &&
           pTx->uiDstMemorySpaceId == HOST_MEMORY_SPACE_ID && 
           bTransferOccurred) {
           assert(valid(pTx->GetFinalState()));
           pHistory->nHToHCopies++;
        }
        if(bTransferOccurred) {
            pHistory->nTotalSyncBytes += m_pDatablock->GetDataBufferLogicalSizeBytes();
        }
        m_bCoherenceProfilerTransitionActive = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the coherence history for this block. </summary>
    ///
    /// <remarks>   Crossbac, 9/18/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    CoherenceProfiler::InitializeInstanceHistory(  
        VOID
        )
    {
        m_pCoherenceHistory = NULL;
        m_bCoherenceProfilerTransitionActive = FALSE;
        if(!PTask::Runtime::GetCoherenceProfileMode())
            return;
        m_pCoherenceHistory = new COHERENCEHISTORY(m_pDatablock->m_uiDBID);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deinitializes the coherence history for this block. </summary>
    ///
    /// <remarks>   Crossbac, 9/18/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    CoherenceProfiler::DeinitializeInstanceHistory(  
        VOID
        )
    {
        if(!PTask::Runtime::GetCoherenceProfileMode())
            return;

        assert(m_pDatablock->LockIsHeld());
        assert(!m_bCoherenceProfilerTransitionActive);
        // we do not delete the instance coherence history because 
        // the merge makes the static list the owner.
        // delete m_pCoherenceHistory;
        m_pCoherenceHistory = NULL;
        m_bCoherenceProfilerTransitionActive = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record binding. </summary>
    ///
    /// <remarks>   Crossbac, 9/19/2012. </remarks>
    ///
    /// <param name="pPort">        (optional) [in] If non-null, the port the block will occupy. </param>
    /// <param name="pTask">        If non-null, the task. </param>
    /// <param name="pIOConsumer">  [in,out] If non-null, the i/o consumer. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CoherenceProfiler::RecordBinding(
        Port * pPort,
        Task * pTask,
        Port * pIOConsumer
        )
    {
        if(!PTask::Runtime::GetCoherenceProfileMode())
            return;

        __int64 uiTimestamp = ::GetTickCount64();
        if(pPort) {
            EnterCriticalSection(&m_csCoherenceProfiler);
            m_vPortNames[pPort] = std::string(pPort->GetVariableBinding());
            LeaveCriticalSection(&m_csCoherenceProfiler);
            if(m_pCoherenceHistory->pvPortBindHistory->find(uiTimestamp) != 
                m_pCoherenceHistory->pvPortBindHistory->end()) {
                m_pCoherenceHistory->uiConcurrentPortBindings++;
            }
            (*m_pCoherenceHistory->pvPortBindHistory)[::GetTickCount64()] = pPort;
        }
        if(pIOConsumer) {
            EnterCriticalSection(&m_csCoherenceProfiler);
            m_vPortNames[pIOConsumer] = std::string(pIOConsumer->GetVariableBinding());
            LeaveCriticalSection(&m_csCoherenceProfiler);
            if(m_pCoherenceHistory->pvIOCPortBindHistory->find(uiTimestamp) != 
                m_pCoherenceHistory->pvIOCPortBindHistory->end()) {
                m_pCoherenceHistory->uiConcurrentPortBindings++;
            }
            (*m_pCoherenceHistory->pvIOCPortBindHistory)[::GetTickCount64()] = pIOConsumer;
        }
        if(pTask) {
            EnterCriticalSection(&CoherenceProfiler::m_csCoherenceProfiler);
            m_vTaskNames[pTask] = std::string(pTask->GetTaskName());
            LeaveCriticalSection(&CoherenceProfiler::m_csCoherenceProfiler);
            if(m_pCoherenceHistory->pvTaskBindHistory->find(uiTimestamp) != 
                m_pCoherenceHistory->pvTaskBindHistory->end()) {
                m_pCoherenceHistory->uiConcurrentTaskBindings++;
            }
            (*m_pCoherenceHistory->pvTaskBindHistory)[uiTimestamp] = pTask;
            Accelerator * pDispatchAcc = pTask->GetDispatchAccelerator();
            UINT uiAcceleratorID = pDispatchAcc ? pDispatchAcc->GetAcceleratorId() : 0xFFFFFFFF;
            (*m_pCoherenceHistory->pvAcceleratorBindHistory)[uiTimestamp] = uiAcceleratorID;
            if(pPort && pPort->HasDependentAcceleratorBinding()) {
                Accelerator * pDepAcc = pTask->GetAssignedDependentAccelerator(pPort);
                (*m_pCoherenceHistory->pvDepAcceleratorBindHistory)[uiTimestamp] = pDepAcc->GetAcceleratorId();
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record port binding. </summary>
    ///
    /// <remarks>   Crossbac, 9/19/2012. </remarks>
    ///
    /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CoherenceProfiler::RecordPortBinding(
        Port * pPort
        )
    {
        if(!PTask::Runtime::GetCoherenceProfileMode())
            return;

        if(pPort) {
            EnterCriticalSection(&m_csCoherenceProfiler);
            m_vPortNames[pPort] = std::string(pPort->GetVariableBinding());
            LeaveCriticalSection(&m_csCoherenceProfiler);
            __int64 uiTimestamp = ::GetTickCount64();
            if(m_pCoherenceHistory->pvPortBindHistory->find(uiTimestamp) != 
                m_pCoherenceHistory->pvPortBindHistory->end()) {
                m_pCoherenceHistory->uiConcurrentPortBindings++;
            }
            (*m_pCoherenceHistory->pvPortBindHistory)[::GetTickCount64()] = pPort;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record port binding. </summary>
    ///
    /// <remarks>   Crossbac, 9/19/2012. </remarks>
    ///
    /// <param name="pPort">    (optional) [in] If non-null, the port the block will occupy. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CoherenceProfiler::RecordTaskBinding(
        Task * pTask
        )
    {
        if(!PTask::Runtime::GetCoherenceProfileMode())
            return;

        if(pTask) {
            EnterCriticalSection(&CoherenceProfiler::m_csCoherenceProfiler);
            m_vTaskNames[pTask] = std::string(pTask->GetTaskName());
            LeaveCriticalSection(&CoherenceProfiler::m_csCoherenceProfiler);
            __int64 uiTimestamp = ::GetTickCount64();
            if(m_pCoherenceHistory->pvTaskBindHistory->find(uiTimestamp) != 
                m_pCoherenceHistory->pvTaskBindHistory->end()) {
                m_pCoherenceHistory->uiConcurrentTaskBindings++;
            }
            (*m_pCoherenceHistory->pvTaskBindHistory)[uiTimestamp] = pTask;
            Accelerator * pDispatchAcc = pTask->GetDispatchAccelerator();
            UINT uiAcceleratorID = pDispatchAcc ? pDispatchAcc->GetAcceleratorId() : 0xFFFFFFFF;
            (*m_pCoherenceHistory->pvAcceleratorBindHistory)[uiTimestamp] = uiAcceleratorID;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Merge the coherence history for this block with the static view. </summary>
    ///
    /// <remarks>   Crossbac, 9/18/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    CoherenceProfiler::MergeHistory(
        VOID
        )
    {
        if(!PTask::Runtime::GetCoherenceProfileMode())
            return;

        assert(m_pDatablock->LockIsHeld());
        EnterCriticalSection(&CoherenceProfiler::m_csCoherenceProfiler);
        assert(m_vHistories.find(m_pDatablock->m_uiDBID) == m_vHistories.end());
        m_vHistories[m_pDatablock->m_uiDBID] = m_pCoherenceHistory;
        LeaveCriticalSection(&CoherenceProfiler::m_csCoherenceProfiler);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Merge the coherence histories for all blocks into the static view. </summary>
    ///
    /// <remarks>   Crossbac, 9/18/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    CoherenceProfiler::MergeHistories(
        VOID
        )
    {
        if(!PTask::Runtime::GetCoherenceProfileMode())
            return;

        EnterCriticalSection(&CoherenceProfiler::m_csCoherenceProfiler);

        CoherenceProfiler::m_nDToDMigrationsExclusive = 0;
        CoherenceProfiler::m_nDToDMigrationsShared = 0;
        CoherenceProfiler::m_nHToDMigrationsExclusive = 0;
        CoherenceProfiler::m_nHToDMigrationsShared = 0;
        CoherenceProfiler::m_nHToDMigrationsExclusive = 0;
        CoherenceProfiler::m_nHToDMigrationsShared = 0;
        CoherenceProfiler::m_nMultiViewInvalidations = 0;

        std::map<UINT, COHERENCEHISTORY*>::iterator mi;
        for(mi=m_vHistories.begin(); mi!=m_vHistories.end(); mi++) {
            COHERENCEHISTORY * pHistory = mi->second;   
            map<__int64, COHERENCETRANSITION*>::iterator ti;
            for(ti=pHistory->pvStateHistory->begin();
                ti!=pHistory->pvStateHistory->end();
                ti++) {
                COHERENCETRANSITION * pTx = ti->second;
                BOOL bDD = pTx->IsDToDXfer();
                BOOL bHD = pTx->IsHToDXfer();
                BOOL bDH = pTx->IsDToHXfer();
                BOOL bHH = pTx->bXferOccurred && !(bDD || bHD || bDH);
                UNREFERENCED_PARAMETER(bHH);
                switch(pTx->GetFinalState()) {
                case BSTATE_NO_ENTRY: 
                case BSTATE_INVALID:  
                    if(pTx->GetNumberOfValidCopies(pTx->eStartState) > 1)
                        m_nMultiViewInvalidations++;
                    break;
                case BSTATE_SHARED:
                    m_nDToDMigrationsShared += bDD ? 1 : 0;
                    m_nHToDMigrationsShared += bHD ? 1 : 0;
                    m_nDToHMigrationsShared += bDH ? 1 : 0;
                    break;
                case BSTATE_EXCLUSIVE:
                    m_nDToDMigrationsExclusive += bDD ? 1 : 0;
                    m_nHToDMigrationsExclusive += bHD ? 1 : 0;
                    m_nDToHMigrationsExclusive += bDH ? 1 : 0;
                    break;
                }
                switch(pTx->eTriggerEvent) {
                case CET_UNSPECIFIED            : m_nCETUnspecified++; break; 
                case CET_BIND_INPUT             : m_nCETBindInput++; break;
                case CET_BIND_OUTPUT            : m_nCETBindOutput++; break;
                case CET_BIND_CONSTANT          : m_nCETBindConstant++; break;
                case CET_PUSH_DOWNSTREAM_SHARE  : m_nCETDownstreamShare++; break;
                case CET_POINTER_REQUEST        : m_nCETPointerRequest++; break;
                case CET_BLOCK_DELETE           : m_nCETBlockDelete++; break;
                case CET_BLOCK_CLONE            : m_nCETBlockClone++; break;
                case CET_BLOCK_CREATE           : m_nCETBlockCreate++; break;
                case CET_HOST_VIEW_UPDATE       : m_nCETHostViewUpdate++; break;
                case CET_ACCELERATOR_VIEW_UPDATE: m_nCETAcceleratorViewUpdate++; break;
                case CET_BUFFER_ALLOCATE        : m_nCETBufferAllocate++; break;
                case CET_GROW_BUFFER            : m_nCETGrowBuffer++; break;
                case CET_SYNTHESIZE_BLOCK       : m_nCETSynthesizeBlock++; break;
                case CET_PINNED_HOST_VIEW_CREATE: m_nCETPinnedHostView++; break;                
                }
            }
        }

        LeaveCriticalSection(&CoherenceProfiler::m_csCoherenceProfiler);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Coherence tracker set detailed. </summary>
    ///
    /// <remarks>   Crossbac, 9/21/2012. </remarks>
    ///
    /// <param name="b">    true to b. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CoherenceProfiler::SetDetailed(
        BOOL b
        )
    {
        if(!CoherenceProfiler::m_nCoherenceProfilerInit) return;
        EnterCriticalSection(&CoherenceProfiler::m_csCoherenceProfiler);
        m_bCoherenceStatisticsDetailed = b;
        LeaveCriticalSection(&CoherenceProfiler::m_csCoherenceProfiler);
    }

#else
CoherenceProfiler::CoherenceProfiler(Datablock * pDatablock) { pDatablock; assert(FALSE); }
CoherenceProfiler::~CoherenceProfiler() { assert(FALSE); } 
BOOL CoherenceProfiler::Initialize(BOOL bEnable, BOOL bVerbose) { bEnable; bVerbose; assert(FALSE); return FALSE; }
BOOL CoherenceProfiler::Deinitialize() { assert(FALSE); return FALSE; }
void CoherenceProfiler::Report(std::ostream& ios) { ios; assert(FALSE); }
std::stringstream* CoherenceProfiler::GetReport() { assert(FALSE); return NULL; }
void CoherenceProfiler::GetDetailedReport(std::ostream& ss) { ss; assert(FALSE); }
COHERENCETRANSITION * CoherenceProfiler::RecordViewUpdateStart(UINT nD, COHERENCEEVENTTYPE eEventType) { nD; eEventType; assert(FALSE); return NULL; }
void CoherenceProfiler::RecordViewUpdateEnd(UINT nID, BUFFER_COHERENCE_STATE uiR, BOOL b, COHERENCETRANSITION * pTx) { nID; uiR; b; pTx; assert(FALSE); }
void CoherenceProfiler::InitializeInstanceHistory() { assert(FALSE); }
void CoherenceProfiler::DeinitializeInstanceHistory() { assert(FALSE); }
void CoherenceProfiler::RecordBinding(Port * pP, Task * pT, Port * pIO) { pP; pT; pIO; assert(FALSE); }
void CoherenceProfiler::RecordPortBinding(Port * pP) { pP; }
void CoherenceProfiler::RecordTaskBinding(Task * pT) { pT; }
void CoherenceProfiler::MergeHistory() {}
void CoherenceProfiler::MergeHistories() {}
void CoherenceProfiler::SetDetailed(BOOL b) { b; assert(FALSE); }
#endif

};
