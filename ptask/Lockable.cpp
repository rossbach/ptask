///-------------------------------------------------------------------------------------------------
// file:	Lockable.cpp
//
// summary:	Implements the lockable object class
///-------------------------------------------------------------------------------------------------

#include "Lockable.h"
#include "ptaskutils.h"
#include "ptlock.h"
#include "assert.h"

#ifdef DEBUG
PTask::PTLock g_loglock("lock-logger");
#define UPDATE_OWNER(x)  UpdateOwningThreadId(x)
#define CURRENT_THREAD_IS_OWNER()   (m_dwOwningThreadId == ::GetCurrentThreadId())
#define LOGLOCKACTIVITY(isacquire)  LogLockActivity(isacquire)
#else
#define UPDATE_OWNER(x)  
#define CURRENT_THREAD_IS_OWNER()   TRUE
#define LOGLOCKACTIVITY(isacquire)
#endif


namespace PTask {

    static const BOOL LO_LOCKING = TRUE;
    static const BOOL LO_UNLOCKING = FALSE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Default constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="lpszProtectedObjectName">  [in,out] If non-null, name of the protected object. </param>
    ///-------------------------------------------------------------------------------------------------

    Lockable::Lockable(
        char * lpszProtectedObjectName
        )
    {
        m_bTrack = FALSE;
        m_uiUnnestedAcquires = 0;
        m_uiUnnestedReleases = 0;
        m_nLockDepth = 0;
        m_dwOwningThreadId = NULL;
        m_lpszProtectedObjectName = NULL;
        InitializeCriticalSection(&m_lock);
        if(lpszProtectedObjectName != NULL) {
            size_t nNameLength = strlen(lpszProtectedObjectName);
            m_lpszProtectedObjectName = new char[nNameLength+1];
            strcpy_s(m_lpszProtectedObjectName, nNameLength+1, lpszProtectedObjectName);
        } else {
            m_lpszProtectedObjectName = new char[64];
            UINT uid = ptaskutils::nextuid();
            sprintf_s(m_lpszProtectedObjectName, 64, "unlabeled-object-%d", uid);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Lockable::~Lockable(
        VOID
        )
    {        
        assert(m_nLockDepth == 0 && "attempt to delete object while locked!");
        if(m_lpszProtectedObjectName != NULL) {
            delete [] m_lpszProtectedObjectName;
            m_lpszProtectedObjectName = NULL;
        }
        m_dwOwningThreadId = NULL;
        DeleteCriticalSection(&m_lock);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Lock this object. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <returns>   the new lock depth. </returns>
    ///-------------------------------------------------------------------------------------------------

    int 
    Lockable::Lock(
        VOID
        )
    {
        EnterCriticalSection(&m_lock);
        UPDATE_OWNER(LO_LOCKING);
        m_nLockDepth++;
        LOGLOCKACTIVITY(TRUE);
        return m_nLockDepth;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unlock this object. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <returns>   the new lock depth. </returns>
    ///-------------------------------------------------------------------------------------------------

    int Lockable::Unlock(
        VOID
        )
    {
        assert(m_nLockDepth > 0);
        UPDATE_OWNER(LO_UNLOCKING);
        m_nLockDepth--;
        LOGLOCKACTIVITY(FALSE);
        LeaveCriticalSection(&m_lock);
        return m_nLockDepth;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object is locked. This method is to be used in asserts that the
    ///             current thread holds the lock, and *not* to be used to implement TryLock
    ///             semantics!
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <returns>   true if held, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Lockable::LockIsHeld(
        VOID
        )
    {
        BOOL bHeld = m_nLockDepth > 0;
        bHeld &= CURRENT_THREAD_IS_OWNER();
        return bHeld;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the lock depth. Don't call this unless you've got the lock.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The lock depth. </returns>
    ///-------------------------------------------------------------------------------------------------

    int 
    Lockable::GetLockDepth(
        VOID
        ) {
        assert(LockIsHeld());
        return m_nLockDepth;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Updates the owning thread identifier. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="bLocking"> true if this update is for the lock operation, otherwise this update
    ///                         is for an unlock. </param>
    ///-------------------------------------------------------------------------------------------------

    void                    
    Lockable::UpdateOwningThreadId(
        BOOL bLocking
        )
    {
        if(bLocking && m_nLockDepth == 0) {
            // claim the lock for this thread. the caller will increment
            // the lock depth after making this call. 
            m_dwOwningThreadId = ::GetCurrentThreadId();
        } else if(!bLocking && m_nLockDepth == 1) {
            // release this threads ownership of the lock.
            // caller will decrement the lock depth after this
            // call completes.
            m_dwOwningThreadId = 0;
        } else {
            // this call to lock or unlock should not change the 
            // owning thread. Check that and the lock depth.
            assert(m_dwOwningThreadId == GetCurrentThreadId());
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   In debug mode, enables/disables tracking for a particular object, returns
    ///             true if tracking is enabled after the call. When tracking is enabled,
    ///             all lock/unlock calls are logged to the console. A handy tool for teasing
    ///             apart deadlocks.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 8/29/2013. </remarks>
    ///
    /// <param name="bEnable">  (Optional) the enable. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Lockable::TrackLockActivity(
        BOOL bEnable
        )
    {
#ifdef DEBUG
        m_bTrack = bEnable;
        return m_bTrack;
#else
        UNREFERENCED_PARAMETER(bEnable);
        return FALSE;
#endif
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Logs lock activity. </summary>
    ///
    /// <remarks>   crossbac, 8/29/2013. </remarks>
    ///
    /// <param name="bLocking"> true to locking. </param>
    ///-------------------------------------------------------------------------------------------------

    void                    
    Lockable::LogLockActivity(
        BOOL bLocking
        )
    {
#ifdef DEBUG
        if(m_bTrack) {                                       
            g_loglock.LockRW();
            char * strLockStr = NULL;
            bool bNested = true;
            if(bLocking) {
                strLockStr = "lock  ";
                if(m_nLockDepth == 1) {
                    m_uiUnnestedAcquires++;
                    strLockStr = "LOCK  ";
                    bNested = false;
                }
            } else {
                strLockStr = "unlock";
                if(m_nLockDepth == 0) {
                    strLockStr = "UNLOCK";
                    m_uiUnnestedReleases++;
                    bNested = false;
                }
            }
            std::cerr << strLockStr << ":"
                      << m_lpszProtectedObjectName
                      << " depth: " << m_nLockDepth
                      << " owner: " << m_dwOwningThreadId;
            if(!bNested) 
                std::cerr << " #" << (bLocking?m_uiUnnestedAcquires:m_uiUnnestedReleases);
            std::cerr << std::endl;
            g_loglock.UnlockRW();                            
        }
#else
        UNREFERENCED_PARAMETER(bLocking);
#endif
    }

};