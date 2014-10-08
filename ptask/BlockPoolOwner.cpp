///-------------------------------------------------------------------------------------------------
// file:	BlockPoolOwner.cpp
//
// summary:	Implements the block pool owner class
///-------------------------------------------------------------------------------------------------

#include "AsyncContext.h"
#include "BlockPoolOwner.h"
#include "OutputPort.h"
#include "InputPort.h"
#include "PTaskRuntime.h"
#include "MemorySpace.h"
#include "task.h"
#include "assert.h"

using namespace std;

namespace PTask {

    /// <summary>   The lock for the block pool owners. </summary>
    CRITICAL_SECTION                    BlockPoolOwner::s_csBlockPoolOwners;

    /// <summary>   true if block pool owner managment is initialized. </summary>
    LONG                                BlockPoolOwner::s_bPoolOwnersInit = 0;

    /// <summary>   The active pool owners. </summary>
    std::map<BlockPoolOwner*, Graph*>   BlockPoolOwner::s_vActivePoolOwners;

    /// <summary>   The dead pool owners. </summary>
    std::map<BlockPoolOwner*, Graph*>   BlockPoolOwner::s_vDeadPoolOwners;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the block pool manager. Because PTask objects are 
    ///             reference counted, it is difficult to enforce life-cycle relationships
    ///             that appear to be implied by member containment. For block pools, it
    ///             is entirely possible that user code (or internal code) keeps a reference to a datablock 
    ///             after the block pool from which it came is destroyed or deleted. Consequently,
    ///             the block pool owner pointer is not guaranteed to be valid when a block is released,
    ///             and we must keep a global list of what block pool objects are actually valid and
    ///             active to avoid attempting to return a block to a pool that has been deleted.
    ///             This method creates the data structures pertinent to maintaining that information.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/18/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    BlockPoolOwner::InitializeBlockPoolManager(
        VOID
        )
    {
        if(InterlockedCompareExchange(&s_bPoolOwnersInit, 1, 0)==0) {
            InitializeCriticalSection(&s_csBlockPoolOwners);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destroy the block pool manager. Because PTask objects are 
    ///             reference counted, it is difficult to enforce life-cycle relationships
    ///             that appear to be implied by member containment. For block pools, it
    ///             is entirely possible that user code (or internal code) keeps a reference to a datablock 
    ///             after the block pool from which it came is destroyed or deleted. Consequently,
    ///             the block pool owner pointer is not guaranteed to be valid when a block is released,
    ///             and we must keep a global list of what block pool objects are actually valid and
    ///             active to avoid attempting to return a block to a pool that has been deleted.
    ///             This method cleans up the data structures pertinent to maintaining that information.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/18/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void                                  
    BlockPoolOwner::DestroyBlockPoolManager(
        VOID
        )
    {
        if(InterlockedCompareExchange(&s_bPoolOwnersInit, 0, 1)==1) {
            EnterCriticalSection(&s_csBlockPoolOwners);
            s_vActivePoolOwners.clear();
            s_vDeadPoolOwners.clear();
            LeaveCriticalSection(&s_csBlockPoolOwners);
            DeleteCriticalSection(&s_csBlockPoolOwners);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Is a block pool owner pointer valid? Because PTask objects are reference counted,
    ///             it is difficult to enforce life-cycle relationships that appear to be implied by
    ///             member containment. For block pools, it is entirely possible that user code (or
    ///             internal code) keeps a reference to a datablock after the block pool from which
    ///             it came is destroyed or deleted. Consequently, the block pool owner pointer is
    ///             not guaranteed to be valid when a block is released, and we must keep a global
    ///             list of what block pool objects are actually valid and active to avoid attempting
    ///             to return a block to a pool that has been deleted.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/18/2013. </remarks>
    ///
    /// <param name="pOwner">   [in,out] If non-null, the owner. </param>
    ///
    /// <returns>   true if a pool owner is active, false if not. </returns>
	///-------------------------------------------------------------------------------------------------

    BOOL  
    BlockPoolOwner::IsPoolOwnerActive(
        __in BlockPoolOwner * pOwner
        )
    {
        assert(s_bPoolOwnersInit);
        if(s_bPoolOwnersInit) {
            BOOL bResult = FALSE;
            EnterCriticalSection(&s_csBlockPoolOwners);
            bResult = s_vActivePoolOwners.find(pOwner) != s_vActivePoolOwners.end();
            if(bResult) {
                assert(s_vDeadPoolOwners.find(pOwner) == s_vDeadPoolOwners.end());
                bResult = pOwner->IsBlockPoolActive();
            } else {
                assert(s_vDeadPoolOwners.find(pOwner) != s_vDeadPoolOwners.end());
            }
            LeaveCriticalSection(&s_csBlockPoolOwners);
            return bResult;
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Add a new block pool owner to the global list. Because PTask objects are
    ///             reference counted, it is difficult to enforce life-cycle relationships that
    ///             appear to be implied by member containment. For block pools, it is entirely
    ///             possible that user code (or internal code) keeps a reference to a datablock after
    ///             the block pool from which it came is destroyed or deleted. Consequently, the
    ///             block pool owner pointer is not guaranteed to be valid when a block is released,
    ///             and we must keep a global list of what block pool objects are actually valid and
    ///             active to avoid attempting to return a block to a pool that has been deleted.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/18/2013. </remarks>
    ///
    /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
    /// <param name="pOwner">   [in,out] If non-null, the owner. </param>
    ///-------------------------------------------------------------------------------------------------

    void  
    BlockPoolOwner::RegisterActivePoolOwner(
        __in Graph * pGraph,
        __in BlockPoolOwner * pOwner
        )
    {
        assert(s_bPoolOwnersInit);
        if(s_bPoolOwnersInit) {
            EnterCriticalSection(&s_csBlockPoolOwners);
            assert(s_vActivePoolOwners.find(pOwner) == s_vActivePoolOwners.end());
            assert(s_vDeadPoolOwners.find(pOwner) == s_vDeadPoolOwners.end());
            s_vActivePoolOwners[pOwner] = pGraph;
            LeaveCriticalSection(&s_csBlockPoolOwners);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Retire a block pool owner from the global list. Because PTask objects are
    ///             reference counted, it is difficult to enforce life-cycle relationships that
    ///             appear to be implied by member containment. For block pools, it is entirely
    ///             possible that user code (or internal code) keeps a reference to a datablock after
    ///             the block pool from which it came is destroyed or deleted. Consequently, the
    ///             block pool owner pointer is not guaranteed to be valid when a block is released,
    ///             and we must keep a global list of what block pool objects are actually valid and
    ///             active to avoid attempting to return a block to a pool that has been deleted.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/18/2013. </remarks>
    ///
    /// <param name="pOwner">   [in,out] If non-null, the owner. </param>
    ///-------------------------------------------------------------------------------------------------

    void                             
    BlockPoolOwner::RetirePoolOwner(
        __in BlockPoolOwner * pOwner
        )
    {
        assert(s_bPoolOwnersInit);
        if(s_bPoolOwnersInit) {
            EnterCriticalSection(&s_csBlockPoolOwners);
            std::map<BlockPoolOwner*, Graph*>::iterator mi;
            mi = s_vActivePoolOwners.find(pOwner);
            assert(mi != s_vActivePoolOwners.end());
            assert(s_vDeadPoolOwners.find(pOwner) == s_vDeadPoolOwners.end());
            if(mi != s_vActivePoolOwners.end()) {
                s_vDeadPoolOwners[pOwner] = mi->second;
                s_vActivePoolOwners.erase(mi);
            }
            LeaveCriticalSection(&s_csBlockPoolOwners);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Retire all block pool owner from the given graph. Because PTask objects are
    ///             reference counted, it is difficult to enforce life-cycle relationships that
    ///             appear to be implied by member containment. For block pools, it is entirely
    ///             possible that user code (or internal code) keeps a reference to a datablock after
    ///             the block pool from which it came is destroyed or deleted. Consequently, the
    ///             block pool owner pointer is not guaranteed to be valid when a block is released,
    ///             and we must keep a global list of what block pool objects are actually valid and
    ///             active to avoid attempting to return a block to a pool that has been deleted.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/18/2013. </remarks>
    ///
    /// <param name="pOwner">   [in,out] If non-null, the owner. </param>
    ///-------------------------------------------------------------------------------------------------

    void                                  
    BlockPoolOwner::RetireGraph(
        __in Graph * pGraph
        )
    {
        assert(s_bPoolOwnersInit);
        if(s_bPoolOwnersInit) {
            EnterCriticalSection(&s_csBlockPoolOwners);
            std::map<BlockPoolOwner*, Graph*>::iterator mi;
            std::vector<BlockPoolOwner*>::iterator vi;
            std::vector<BlockPoolOwner*> vactive;
            std::vector<BlockPoolOwner*> vdead;
            for(mi=s_vActivePoolOwners.begin(); mi!=s_vActivePoolOwners.end(); mi++) {
                if(mi->second == pGraph) {
                    vactive.push_back(mi->first);
                }
            }
            assert(vactive.size() == 0 && "active pool owners for retiring graph!");
            for(mi=s_vDeadPoolOwners.begin(); mi!=s_vDeadPoolOwners.end(); mi++) {
                if(mi->second == pGraph) {
                    vdead.push_back(mi->first);
                }
            }
            for(vi=vactive.begin(); vi!=vactive.end(); vi++) 
                s_vActivePoolOwners.erase(*vi);
            for(vi=vdead.begin(); vi!=vdead.end(); vi++) 
                s_vDeadPoolOwners.erase(*vi);
            LeaveCriticalSection(&s_csBlockPoolOwners);
        }
    }

};
