///-------------------------------------------------------------------------------------------------
// file:	ReferenceCounted.cpp
//
// summary:	Implements the reference counted class
///-------------------------------------------------------------------------------------------------

#include "ReferenceCounted.h"
#include "RefCountProfiler.h"
#include "assert.h"

namespace PTask {

#ifdef PROFILE_REFCOUNT_OBJECTS
#define rcprofile_alloc(x)          ReferenceCountedProfiler::RecordAllocation(x)
#define rcprofile_delete(x)         ReferenceCountedProfiler::RecordDeletion(x)
#else
#define rcprofile_alloc(x)
#define rcprofile_delete(x)
#endif


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Default constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    ReferenceCounted::ReferenceCounted(
        VOID
        ) : Lockable(NULL)
    {
        m_uiRefCount = 0;
        rcprofile_alloc(this);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    ReferenceCounted::ReferenceCounted(
        char * szProtectedObjectName
        ) : Lockable(szProtectedObjectName)
    {
        m_uiRefCount = 0;
        rcprofile_alloc(this);
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    ReferenceCounted::~ReferenceCounted(
        VOID
        )
    {
        rcprofile_delete(this);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds reference. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    LONG
    ReferenceCounted::AddRef() {
        return InterlockedIncrement(&m_uiRefCount);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the release. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    LONG
    ReferenceCounted::Release(
        VOID
        ) 
    {
        LONG privateCount = InterlockedDecrement(&m_uiRefCount);
        assert(m_uiRefCount >= 0);
        if(privateCount == 0) 
            delete this;
        return privateCount;
        // OLD IMPLEMENTATION
        ////  #error Release should not take the lock, outside of a gc thread
        ////  LONG privateCount = 0;
        ////  Lock();
        ////  if(m_uiRefCount == 1) {
        ////      privateCount = 0;
        ////      m_uiRefCount = 0;
        ////  } else {
        ////      privateCount = InterlockedDecrement(&m_uiRefCount);
        ////  }
        ////  Unlock();
        ////  if(privateCount == 0) 
        ////      delete this;
        ////  return privateCount;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the current reference count. </summary>
    ///
    /// <remarks>   Crossbac, 12/28/2011. </remarks>
    ///
    /// <param name=""> none </param>
    ///
    /// <returns>   reference count. </returns>
    ///-------------------------------------------------------------------------------------------------

    LONG 
    ReferenceCounted::RefCount(
        VOID
        ) 
    { 
        assert(LockIsHeld());
        return m_uiRefCount; 
    }

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
    ReferenceCounted::GetRCProfileDescriptor(
        VOID
        )
    {
#ifdef PROFILE_REFCOUNT_OBJECTS
        return std::string("unk");
#else
        assert(FALSE);
        return std::string("");
#endif
    }
        
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   ReferenceCounted toString operator. This is a debug utility. </summary>
    ///
    /// <remarks>   Crossbac, 12/27/2011. </remarks>
    ///
    /// <param name="os">   [in,out] The output stream. </param>
    /// <param name="port"> The block. </param>
    ///
    /// <returns>   The string result. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::ostream& operator <<(std::ostream &os, ReferenceCounted * pBlock) { 
#ifdef PROFILE_REFCOUNT_OBJECTS
        os  << "RC#" 
            << pBlock->m_uiUID 
            << "(rc:" 
            << pBlock->m_uiRefCount
            << ", name:"
            << pBlock->GetRCProfileDescriptor()
            << ")";
#else
        os << "RC-obj-" << (unsigned __int64) pBlock;
#endif
        return os;
    }


};
