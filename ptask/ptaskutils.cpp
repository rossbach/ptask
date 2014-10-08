///-------------------------------------------------------------------------------------------------
// file:	ptaskutils.cpp
//
// summary:	Implements the ptaskutils class
///-------------------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "ptaskutils.h"
#include <assert.h>

namespace PTask {

    CRITICAL_SECTION	ptaskutils::m_csUIDLock;
    unsigned int		ptaskutils::m_uiUIDCounter = 0;
    BOOL				ptaskutils::m_bInitialized = FALSE;;


    // Round Up Division function
    size_t 
    ptaskutils::roundup(
        int group_size, 
        int global_size
        ) 
    {
        int r = global_size % group_size;
        if(r == 0) 
            return global_size;
        return global_size + group_size - r;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes ptaskutils. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    ptaskutils::initialize(
        VOID
        ) 
    {
        if(m_bInitialized) return;
        m_uiUIDCounter = 0;
        InitializeCriticalSection(&m_csUIDLock);
        m_bInitialized = TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Cleanup ptask utils. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    ptaskutils::cleanup(
        VOID
        )
    {
        if(!m_bInitialized) return;
        DeleteCriticalSection(&m_csUIDLock);
        m_bInitialized = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Get the next unique identifier. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned int
    ptaskutils::nextuid(
        VOID
        ) 
    {
        if(!m_bInitialized)
            initialize();
        unsigned int res = 0;
        EnterCriticalSection(&m_csUIDLock);
        res = ++m_uiUIDCounter;
        LeaveCriticalSection(&m_csUIDLock);
        return res;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the accelerator class for accelerators that can execute code specified in
    ///             szfile.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="szFile">   The file. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    ACCELERATOR_CLASS
    ptaskutils::SelectAcceleratorClass(
        const char * szFile
        )
    {
        ACCELERATOR_CLASS accTargetClass = ACCELERATOR_CLASS_DIRECT_X;
        const char * ext = strrchr(szFile, '.');
        
        if(!strcmp(szFile, "HostFunction")) {
            // Special case for host functions not defined in a separate DLL.
            // TODO Find a way to avoid this?
            accTargetClass = ACCELERATOR_CLASS_HOST;
        } else if(ext == NULL) {
            accTargetClass = ACCELERATOR_CLASS_UNKNOWN;
        } else if(!_stricmp(ext+1, "cl")) {
            accTargetClass = ACCELERATOR_CLASS_OPEN_CL;
        } else if(!_stricmp(ext+1, "hlsl") || !_stricmp(ext+1, "fx")) {
            accTargetClass = ACCELERATOR_CLASS_DIRECT_X;
        } else if(!_stricmp(ext+1, "ptx")) {
            accTargetClass = ACCELERATOR_CLASS_CUDA;
        } else if(!_stricmp(ext+1, "dll")) {
            accTargetClass = ACCELERATOR_CLASS_HOST;
        } else {
            // XXXX: TODO:
            // unsupported platform/source file type
            // deal with other platform types!
            assert(false);
        }
        return accTargetClass;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Loads file into memory. </summary>
    ///
    /// <remarks>   Crossbac, 1/28/2013. </remarks>
    ///
    /// <param name="szFile">   The file. </param>
    /// <param name="ppMemory"> [in,out] If non-null, the memory. </param>
    /// <param name="puiBytes"> [in,out] If non-null, the pui in bytes. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    ptaskutils::LoadFileIntoMemory(
        const HANDLE hFile,
        void ** ppMemory,
        UINT * puiBytes
        )
    {
        if(hFile == INVALID_HANDLE_VALUE || 
           ppMemory == NULL || 
           puiBytes == NULL)
            return FALSE;

        DWORD dwBytesRead = 0;
        DWORD dwBytesAllocated = (UINT) GetFileSize(hFile, NULL);
        void * pMemory = malloc(dwBytesAllocated+1);
        if(ReadFile(hFile, pMemory, dwBytesAllocated, &dwBytesRead, NULL)) {
            *puiBytes = (UINT) dwBytesRead;
            *ppMemory = pMemory;
            return TRUE;
        } else {
            free(pMemory);
            *ppMemory = NULL;
            *puiBytes = NULL;
            return FALSE;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Loads file into memory. </summary>
    ///
    /// <remarks>   Crossbac, 1/28/2013. </remarks>
    ///
    /// <param name="szFile">   The file. </param>
    /// <param name="ppMemory"> [in,out] If non-null, the memory. </param>
    /// <param name="puiBytes"> [in,out] If non-null, the pui in bytes. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    ptaskutils::LoadFileIntoMemory(
        const char * szFile,
        void ** ppMemory,
        UINT * puiBytes
        )
    {
        if(szFile == NULL || ppMemory == NULL || puiBytes == NULL)
            return FALSE;

        HANDLE hFile = CreateFileA(szFile, 
                                   GENERIC_READ, 
                                   FILE_SHARE_READ, 
                                   NULL, 
                                   OPEN_EXISTING, 
                                   FILE_ATTRIBUTE_NORMAL, 
                                   NULL);

        if(hFile == INVALID_HANDLE_VALUE)
            return FALSE;

        return LoadFileIntoMemory(hFile, ppMemory, puiBytes);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Loads file into memory. </summary>
    ///
    /// <remarks>   Crossbac, 1/28/2013. </remarks>
    ///
    /// <param name="szFile">   The file. </param>
    /// <param name="ppMemory"> [in,out] If non-null, the memory. </param>
    /// <param name="puiBytes"> [in,out] If non-null, the pui in bytes. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    ptaskutils::LoadFileIntoMemory(
        const WCHAR * pwszFile,
        void ** ppMemory,
        UINT * puiBytes
        )
    {
        if(pwszFile == NULL || ppMemory == NULL || puiBytes == NULL)
            return FALSE;

        HANDLE hFile = CreateFileW(pwszFile, 
                                   GENERIC_READ, 
                                   FILE_SHARE_READ, 
                                   NULL, 
                                   OPEN_EXISTING, 
                                   FILE_ATTRIBUTE_NORMAL, 
                                   NULL);

        if(hFile == INVALID_HANDLE_VALUE)
            return FALSE;

        return LoadFileIntoMemory(hFile, ppMemory, puiBytes);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return the number of set signal bits. </summary>
    ///
    /// <remarks>   crossbac, 6/30/2014. </remarks>
    ///
    /// <param name="luiSignalWord">    The lui signal word. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    ptaskutils::SignalCount(
        CONTROLSIGNAL luiSignalWord
        ) 
    {
        UINT uiCount = 0;
        UINT uiPos = 0;
        while(uiPos < sizeof(luiSignalWord)) {
            CONTROLSIGNAL sig = (1i64 << uiPos);
            if(TESTSIGNAL(luiSignalWord, sig))
                uiCount++;
            uiPos++;
        }
        return uiCount;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   get the index of the first set signal if any. </summary>
    ///
    /// <remarks>   Crossbac, 2/14/2013. </remarks>
    ///
    /// <param name="luiSignalWord">    The lui signal word. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    int 
    ptaskutils::GetFirstSignalIndex(
        __in CONTROLSIGNAL luiSignalWord
        )
    {
        UINT uiCount = 0;
        UINT uiPos = 0;
        UINT uiFoundSet = 0;
        while(uiPos < sizeof(luiSignalWord)) {
            CONTROLSIGNAL sig = (1i64 << uiPos);
            if(TESTSIGNAL(luiSignalWord, sig)) {
                uiCount++;
                uiFoundSet = 1;
                break; 
            }
            uiPos++;
        }
        if(uiFoundSet) 
            return static_cast<int>(uiPos); 
        return -1;
    }

};
