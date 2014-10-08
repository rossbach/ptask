///-------------------------------------------------------------------------------------------------
// file:	DatablockProfiler.cpp
//
// summary:	Implements the datablock profiler class
///-------------------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "ptaskutils.h"
#include "accelerator.h"
#include "datablock.h"
#include "DatablockProfiler.h"
#include "datablocktemplate.h"
#include "PCUBuffer.h"
#include "OutputPort.h"
#include <assert.h>
#include "PTaskRuntime.h"
#include "ptgc.h"
#include "MemorySpace.h"
#include "Task.h"
#include "CoherenceProfiler.h"
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
// don't check invariants!
#define CHECK_INVARIANTS()
#define CHECK_INVARIANTS_LOCK()
#endif

namespace PTask {

    /// <summary>   The number of datablock allocations. </summary>
    LONG DatablockProfiler::m_nDBAllocations = 0;

    /// <summary>   The datablock deletion count. </summary>
    LONG DatablockProfiler::m_nDBDeletions = 0;

    /// <summary>   The number of clone allocations. </summary>
    LONG DatablockProfiler::m_nDBCloneAllocations = 0;

    /// <summary>   The number of clone deletions. </summary>
    LONG DatablockProfiler::m_nDBCloneDeletions = 0;

    /// <summary>   Is the profiler initialised? </summary>
    LONG DatablockProfiler::m_nDBProfilerInit = 0;

    /// <summary>   Is the profiler enabled? </summary>
    LONG DatablockProfiler::m_nDBProfilerEnabled = 0;

    /// <summary>   Verbose console output? </summary>
    BOOL DatablockProfiler::m_bDBProfilerVerbose = 0;

    /// <summary>   The set of datablocks currently allocated but not yet deleted. </summary>
    std::set<PTask::Datablock*> DatablockProfiler::m_vAllAllocations;

    /// <summary>   List of names task names. Required because we will no longer have
    /// 			valid task pointers when we check for leaks (all tasks *should* be
    /// 			deleted by that point), and we want to be able to find the task
    /// 			that allocated a block if it was leaked and provide it's name as
    /// 			a debug assist.   
    /// 			</summary>
    std::map<PTask::Task*, std::string> DatablockProfiler::m_vTaskNames;

    /// <summary>   List of port names. Required because we will no longer have
    /// 			valid port pointers when we check for leaks (all ports *should* be
    /// 			deleted by that point), and we want to be able to find the last
    /// 		    port that touched any leaked blocks. 
    /// 			</summary>
    std::map<PTask::Port*, std::string> DatablockProfiler::m_vPortNames;

    /// <summary>   The profiler lock. Protects the allocation counts,
    /// 			the allocation set, and the port and task maps.
    /// 			</summary>
    CRITICAL_SECTION DatablockProfiler::m_csDBProfiler;

#ifdef PROFILE_DATABLOCKS

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///
    /// <param name="pDatablock">   [in,out] If non-null, the datablock. </param>
    ///-------------------------------------------------------------------------------------------------

    DatablockProfiler::DatablockProfiler(
        __in Datablock * pDatablock
        )
    {
        m_pDatablock = NULL;
        if(PTask::Runtime::GetDBProfilingEnabled()) {
            m_pDatablock = pDatablock;
            RecordAllocation(m_pDatablock);
        }
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    DatablockProfiler::~DatablockProfiler(
        VOID
        )
    {
        if(PTask::Runtime::GetDBProfilingEnabled()) {
            assert(m_pDatablock != NULL);
            RecordDeletion(m_pDatablock);
            m_vPools.clear();
            m_vPortBindings.clear();
            m_vTaskBindings.clear();
            m_pDatablock = NULL;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the allocation tracker. </summary>
    ///
    /// <remarks>   crossbac, 7/10/2012. </remarks>
    ///
    /// <param name="bVerbose"> true to verbose. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    DatablockProfiler::Initialize(
        BOOL bEnable,
        BOOL bVerbose
        ) 
    {
        if(!PTask::Runtime::GetDBProfilingEnabled())
            return FALSE;

        if(InterlockedCompareExchange(&m_nDBProfilerInit, 1, 0) == 0) {
            InitializeCriticalSection(&m_csDBProfiler);
            m_bDBProfilerVerbose = bVerbose;
            m_nDBProfilerEnabled = bEnable;
        } else {
            EnterCriticalSection(&m_csDBProfiler);
            m_nDBProfilerEnabled = bEnable;
            LeaveCriticalSection(&m_csDBProfiler);
        }
        return m_nDBProfilerInit;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the allocation tracker. </summary>
    ///
    /// <remarks>   crossbac, 7/10/2012. </remarks>
    ///
    /// <param name="bVerbose"> true to verbose. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    DatablockProfiler::Deinitialize(
        VOID
        ) 
    {
        if(!PTask::Runtime::GetDBProfilingEnabled())
            return FALSE;

        EnterCriticalSection(&m_csDBProfiler);
        m_vAllAllocations.clear();
        m_vTaskNames.clear();
        m_vPortNames.clear();
        LeaveCriticalSection(&m_csDBProfiler);
        if(InterlockedCompareExchange(&m_nDBProfilerInit, 0, 1) == 1) {
            DeleteCriticalSection(&m_csDBProfiler);
            m_nDBProfilerEnabled = FALSE;
            return TRUE;
        } 
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocation tracker dump leaks. </summary>
    ///
    /// <remarks>   crossbac, 7/10/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    DatablockProfiler::Report(
        __inout std::ostream& ios
        ) 
    {
        if(!PTask::Runtime::GetDBProfilingEnabled())
            return;

        if(m_nDBProfilerInit && m_nDBProfilerEnabled) {
            set<PTask::Datablock*>::iterator si;
            if(m_vAllAllocations.size() != 0) 
                Runtime::MandatoryInform("leaks detected!\n");
            EnterCriticalSection(&m_csDBProfiler);
            Runtime::MandatoryInform("%d blocks allocated, %d blocks deleted\n", m_nDBAllocations, m_nDBDeletions);
            ios << m_nDBCloneAllocations 
                << " clones allocated , " 
                << m_nDBCloneDeletions
                << " clones deleted."
                << std::endl;
            printf("%d clones allocated, %d clones deleted\n", m_nDBCloneAllocations, m_nDBCloneDeletions);
            for(si=m_vAllAllocations.begin(); si!=m_vAllAllocations.end(); si++) {
                PTask::Datablock * pBlock = *si;
                ios << pBlock << std::endl;
            }
            LeaveCriticalSection(&m_csDBProfiler);    
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocation tracker record allocation. </summary>
    ///
    /// <remarks>   crossbac, 7/10/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    DatablockProfiler::RecordAllocation(
        Datablock * pBlock
        )
    {
        if(!PTask::Runtime::GetDBProfilingEnabled())
            return;

        assert(m_nDBProfilerInit);
        if(m_nDBProfilerInit && m_nDBProfilerEnabled) {
            EnterCriticalSection(&m_csDBProfiler);
            m_vAllAllocations.insert(pBlock);
            LeaveCriticalSection(&m_csDBProfiler);
            if(m_bDBProfilerVerbose) 
                Runtime::MandatoryInform("alloc DB#%d\n", pBlock->m_uiDBID);
            InterlockedIncrement(&m_nDBAllocations);
            if(pBlock->m_bIsClone) 
                InterlockedIncrement(&m_nDBCloneAllocations); 
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Allocation tracker record deletion. </summary>
    ///
    /// <remarks>   crossbac, 7/10/2012. </remarks>
    ///
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    DatablockProfiler::RecordDeletion(
        Datablock * pBlock
        )
    {
        if(!PTask::Runtime::GetDBProfilingEnabled())
            return;

        assert(m_nDBProfilerInit);
        if(m_nDBProfilerInit && m_nDBProfilerEnabled) {
            EnterCriticalSection(&m_csDBProfiler);
            m_vAllAllocations.erase(pBlock);
            LeaveCriticalSection(&m_csDBProfiler);
            if(m_bDBProfilerVerbose) 
                Runtime::MandatoryInform("delete DB#%d\n", pBlock->m_uiDBID);
            InterlockedIncrement(&m_nDBDeletions);
            if(pBlock->m_bIsClone) 
                InterlockedIncrement(&m_nDBCloneDeletions);
        }
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
    DatablockProfiler::RecordBinding(
        Port * pPort,
        Task * pTask,
        Port * pIOConsumer
        )
    {
        if(!PTask::Runtime::GetDBProfilingEnabled())
            return;

        if(pPort) {
            EnterCriticalSection(&m_csDBProfiler);
            m_vPortNames[pPort] = std::string(pPort->GetVariableBinding());
            LeaveCriticalSection(&m_csDBProfiler);
            m_vPortBindings.insert(pPort);
        }
        if(pIOConsumer) {
            EnterCriticalSection(&m_csDBProfiler);
            m_vPortNames[pIOConsumer] = std::string(pIOConsumer->GetVariableBinding());
            LeaveCriticalSection(&m_csDBProfiler);
            m_vPortBindings.insert(pIOConsumer);
        }
        if(pTask) {
            EnterCriticalSection(&m_csDBProfiler);
            m_vTaskNames[pTask] = std::string(pTask->GetTaskName());
            LeaveCriticalSection(&m_csDBProfiler);
            m_vTaskBindings.insert(pTask);
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
    DatablockProfiler::RecordBinding(
        Port * pPort
        )
    {
        if(!PTask::Runtime::GetDBProfilingEnabled())
            return;

        if(pPort) {
            EnterCriticalSection(&m_csDBProfiler);
            m_vPortNames[pPort] = std::string(pPort->GetVariableBinding());
            LeaveCriticalSection(&m_csDBProfiler);
            m_vPortBindings.insert(pPort);
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
    DatablockProfiler::RecordBinding(
        Task * pTask
        )
    {
        if(!PTask::Runtime::GetCoherenceProfileMode())
            return;

        if(pTask) {
            EnterCriticalSection(&m_csDBProfiler);
            m_vTaskNames[pTask] = std::string(pTask->GetTaskName());
            LeaveCriticalSection(&m_csDBProfiler);
            m_vTaskBindings.insert(pTask);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record pool binding. </summary>
    ///
    /// <remarks>   Crossbac, 7/19/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    DatablockProfiler::RecordPoolBinding(
        VOID
        )
    {
        if(!PTask::Runtime::GetDBProfilingEnabled())
            return;

        EnterCriticalSection(&m_csDBProfiler);
        BlockPoolOwner * pPoolOwner = m_pDatablock->GetPoolOwner();
        if(pPoolOwner != NULL) {
            m_vPools[pPoolOwner] = pPoolOwner->GetPoolOwnerName();
        }
        LeaveCriticalSection(&m_csDBProfiler);
    }

#else
DatablockProfiler::DatablockProfiler(Datablock * pDatablock) { pDatablock; assert(FALSE); }
DatablockProfiler::~DatablockProfiler() { assert(FALSE); }
BOOL DatablockProfiler::Initialize(BOOL bE, BOOL bV) { bE; bV; assert(!bE); return FALSE; }
BOOL DatablockProfiler::Deinitialize() { assert(!PTask::Runtime::GetDBProfilingEnabled()); return FALSE; }
void DatablockProfiler::Report(std::ostream& ios) { ios; assert(!PTask::Runtime::GetDBProfilingEnabled()); }
void DatablockProfiler::RecordAllocation(Datablock * pBlock) { pBlock; assert(FALSE); }
void DatablockProfiler::RecordDeletion(Datablock * pBlock) { pBlock; assert(FALSE); }
void DatablockProfiler::RecordBinding(Port * pPort, Task * pTask, Port * pIOConsumer) { pPort; pTask; pIOConsumer; }
void DatablockProfiler::RecordBinding(Port * pPort) { pPort; }
void DatablockProfiler::RecordBinding(Task * pTask) { pTask; }
void DatablockProfiler::RecordPoolBinding() { }
#endif

};
