//--------------------------------------------------------------------------------------
// File: CompiledKernel.cpp
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#include "CompiledKernel.h"
#include <windows.h>
#include "accelerator.h"
#include "PTaskRuntime.h"
#include "ptlock.h"
#include <assert.h>

namespace PTask {

    std::map<std::string, HMODULE> CompiledKernel::m_vLoadedDlls;
    std::map<std::string, std::map<std::string, FARPROC>> CompiledKernel::m_vEntryPoints;
    PTLock  CompiledKernel::m_vModuleLock("CompiledKernelsLock");

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Constructor. </summary>
    ///
    /// <remarks>	Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="lpszSourceFile">				[in] non-null, source file. </param>
    /// <param name="lpszOperation">				[in] non-null, the operation. </param>
    /// <param name="lpszInitializerBinary">		[in,out] If non-null, the initializer binary. </param>
    /// <param name="lpszInitializerEntryPoint">	[in,out] If non-null, the initializer entry
    /// 											point. </param>
    /// <param name="eInitializerPSObjectClass">	(Optional) the initializer ps object class. </param>
    ///-------------------------------------------------------------------------------------------------

    CompiledKernel::CompiledKernel(
		__in char *            lpszSourceFile, 
		__in char *            lpszOperation,
		__in char *            lpszInitializerBinary,
		__in char *            lpszInitializerEntryPoint,
		__in ACCELERATOR_CLASS eInitializerPSObjectClass
		)
    {
		m_bInitializerInvoked = FALSE;
		m_lpszInitializerBinary = NULL;
		m_lpszInitializerEntryPoint = NULL;
		m_lpvInitializerModule = NULL;
		m_lpvInitializerProcAddress = NULL;
        size_t nSourceBufferLen = strlen(lpszSourceFile)+1;
        size_t nOperationBufferLen = strlen(lpszOperation)+1;
        m_lpszSourceFile = new char[nSourceBufferLen+1];
        m_lpszOperation = new char[nOperationBufferLen+1];
        strcpy_s(m_lpszSourceFile, nSourceBufferLen, lpszSourceFile);
        strcpy_s(m_lpszOperation, nOperationBufferLen, lpszOperation);
		if(lpszInitializerBinary != NULL) {
			size_t nInitBinaryLen = strlen(lpszInitializerBinary)+1;
			m_lpszInitializerBinary = new char[nInitBinaryLen+1];
			strcpy_s(m_lpszInitializerBinary, nInitBinaryLen, lpszInitializerBinary);
		}
		if(lpszInitializerEntryPoint != NULL) {
			size_t nInitEntryLen = strlen(lpszInitializerEntryPoint)+1;
			m_lpszInitializerEntryPoint = new char[nInitEntryLen+1];
			strcpy_s(m_lpszInitializerEntryPoint, nInitEntryLen, lpszInitializerEntryPoint);
		}
		m_eInitializerPSObjectClass = eInitializerPSObjectClass;

		if(m_lpszInitializerBinary && m_lpszInitializerEntryPoint) {

			FARPROC lpfn = NULL;
			HMODULE hModule = NULL;
            std::string strDLL(m_lpszInitializerBinary);
            std::string strFunc(m_lpszInitializerEntryPoint);
            m_vModuleLock.Lock();
            std::map<std::string, HMODULE>::iterator mi = m_vLoadedDlls.find(strDLL);
            if(mi!=m_vLoadedDlls.end()) {
                hModule = mi->second;
                std::map<std::string, std::map<std::string, FARPROC>>::iterator lpfni;
                lpfni = m_vEntryPoints.find(strDLL);
                if(lpfni != m_vEntryPoints.end()) {
                    std::map<std::string, FARPROC>::iterator fi;
                    fi=lpfni->second.find(strFunc);
                    if(fi!=lpfni->second.end()) {
                        lpfn = fi->second;
                    }
                }
            }

            if(hModule == NULL) {
			    if(NULL == (hModule = LoadLibraryExA(m_lpszInitializerBinary, NULL, NULL))) {
				    static const char * szLLError = "LoadLibraryExA failure: cannot load dll in CompiledKernel::ctor()\n";
				    PTask::Runtime::HandleError(szLLError);
				    return;
			    }
                m_vLoadedDlls[strDLL] = hModule;
            }

            if(lpfn == NULL ) {
			    if(NULL == (lpfn = GetProcAddress(hModule, m_lpszInitializerEntryPoint))) {
				    static const char * szGPAError = "GetProcAddress failed in HostAccelerator::Compile\n";
				    PTask::Runtime::HandleError(szGPAError);
                }
                m_vEntryPoints[strDLL][strFunc]= lpfn;
            }
            m_vModuleLock.Unlock();

			m_lpvInitializerModule = hModule;
			m_lpvInitializerProcAddress = lpfn;
		}
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    CompiledKernel::~CompiledKernel(void)
    {
        delete [] m_lpszSourceFile;
        delete [] m_lpszOperation;
		if(m_lpszInitializerBinary) delete m_lpszInitializerBinary;
		if(m_lpszInitializerEntryPoint) delete m_lpszInitializerEntryPoint;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a platform specific binary. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
    ///
    /// <returns>   null if it fails, else the platform specific binary. </returns>
    ///-------------------------------------------------------------------------------------------------

    void * 
    CompiledKernel::GetPlatformSpecificBinary(
        Accelerator * pAccelerator
        ) 
    { 
        if(m_vPlatformSpecificKernels.find(pAccelerator) == m_vPlatformSpecificKernels.end()) {
            assert(false);
            return NULL;
        }
        return m_vPlatformSpecificKernels[pAccelerator]; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a platform specific binary. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
    /// <param name="p">            [in,out] If non-null, the p. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CompiledKernel::SetPlatformSpecificBinary(
        Accelerator * pAccelerator, 
        void * p
        ) 
    { 
        m_vPlatformSpecificKernels[pAccelerator] = p; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a platform specific module. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
    ///
    /// <returns>   null if it fails, else the platform specific module. </returns>
    ///-------------------------------------------------------------------------------------------------

    void * 
    CompiledKernel::GetPlatformSpecificModule(
        Accelerator * pAccelerator
        ) 
    { 
        if(m_vPlatformSpecificModules.find(pAccelerator) == m_vPlatformSpecificModules.end()) {
            assert(false);
            return NULL;
        }
        return m_vPlatformSpecificModules[pAccelerator]; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a platform specific module. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in] non-null, the accelerator. </param>
    /// <param name="p">            [in,out] If non-null, the p. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CompiledKernel::SetPlatformSpecificModule(
        Accelerator * pAccelerator, 
        void * p
        ) 
    { 
        m_vPlatformSpecificModules[pAccelerator] = p; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Gets the source file. </summary>
    ///
    /// <remarks>	Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>	null if it fails, else the source file. </returns>
    ///-------------------------------------------------------------------------------------------------

    const char * 
    CompiledKernel::GetSourceFile(
        VOID
        ) 
    { 
        return m_lpszSourceFile; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Gets the operation. </summary>
    ///
    /// <remarks>	Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>	null if it fails, else the operation. </returns>
    ///-------------------------------------------------------------------------------------------------

    const char * 
    CompiledKernel::GetOperation(
        VOID
        ) 
    { 
        return m_lpszOperation; 
    } 
	 
    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Gets the source binary for init routine. </summary>
    ///
    /// <remarks>	Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>	null if it fails, else the source file. </returns>
    ///-------------------------------------------------------------------------------------------------

    const char * 
	CompiledKernel::GetInitializerBinary(
		VOID
		)
	{
		return m_lpszInitializerBinary;
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Gets the entry point for any initializer routines.
    /// 			</summary>
    ///
    /// <remarks>	Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>	null if it fails, else the operation. </returns>
    ///-------------------------------------------------------------------------------------------------

    const char * 
	CompiledKernel::GetInitializerEntryPoint(
		VOID
		)
	{
		return m_lpszInitializerEntryPoint;
	}


	///-------------------------------------------------------------------------------------------------
	/// <summary>	Sets the initializer binary. </summary>
	///
	/// <remarks>	crossbac, 8/13/2013. </remarks>
	///
	/// <param name="hModule">	The module. </param>
	///-------------------------------------------------------------------------------------------------

	void 
	CompiledKernel::SetInitializerBinary(
		HMODULE hModule
		)
	{
		m_lpvInitializerModule = (void*) hModule;
	}

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Sets the initializer entry point. </summary>
	///
	/// <remarks>	crossbac, 8/13/2013. </remarks>
	///
	/// <param name="lpvProcAddress">	[in,out] If non-null, the lpv proc address. </param>
	///-------------------------------------------------------------------------------------------------

	void 
	CompiledKernel::SetInitializerEntryPoint(
		void * lpvProcAddress
		)
	{
		m_lpvInitializerProcAddress = lpvProcAddress;
	}

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Query if this kernel has a static initializer that should be called as part
	/// 			of putting the graph in the run state. </summary>
	///
	/// <remarks>	crossbac, 8/13/2013. </remarks>
	///
	/// <returns>	true if static initializer, false if not. </returns>
	///-------------------------------------------------------------------------------------------------

	BOOL 
	CompiledKernel::HasStaticInitializer(
		VOID
		)
	{
		return m_lpvInitializerModule != NULL && m_lpvInitializerProcAddress != NULL;
	}

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Determines if any present initializer routines requires platform-specific
	/// 			device objects to provided when called. </summary>
	///
	/// <remarks>	crossbac, 8/13/2013. </remarks>
	///
	/// <returns>	true if it succeeds, false if it fails. </returns>
	///-------------------------------------------------------------------------------------------------

	BOOL 
	CompiledKernel::InitializerRequiresPSObjects(
		VOID
		)
	{
		return HasStaticInitializer() && m_eInitializerPSObjectClass != ACCELERATOR_CLASS_UNKNOWN;
	}

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Gets initializer required ps classes. </summary>
	///
	/// <remarks>	crossbac, 8/13/2013. </remarks>
	///
	/// <returns>	null if it fails, else the initializer required ps classes. </returns>
	///-------------------------------------------------------------------------------------------------

	ACCELERATOR_CLASS 
	CompiledKernel::GetInitializerRequiredPSClass(
		VOID
		)
	{
		return m_eInitializerPSObjectClass;
	}

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Executes the initializer, with a list of platform specific resources.
	/// 			</summary>
	///
	/// <remarks>	crossbac, 8/13/2013. </remarks>
	///
	/// <param name="vPSDeviceObjects">	[in,out] [in,out] If non-null, the ps device objects. </param>
	///
	/// <returns>	true if it succeeds, false if it fails. </returns>
	///-------------------------------------------------------------------------------------------------

	BOOL 
	CompiledKernel::InvokeInitializer(
		__in DWORD dwThreadId,
		__in std::set<Accelerator*>& vPSDeviceObjects
		)
	{
		assert(InitializerRequiresPSObjects());
		assert(m_lpvInitializerProcAddress != NULL);
		LPFNTASKINITIALIZER lpfnInit = (LPFNTASKINITIALIZER) m_lpvInitializerProcAddress;
		if(!lpfnInit) return FALSE;

		int nIndex = 0;
		std::set<Accelerator*>::iterator vi;
		for(vi=vPSDeviceObjects.begin(); vi!=vPSDeviceObjects.end(); vi++, nIndex++) {
            Accelerator * pAccelerator = *vi;
			pAccelerator->Lock();           
            pAccelerator->MakeDeviceContextCurrent();
			int nDeviceId = reinterpret_cast<int>(pAccelerator->GetDevice());
			(*lpfnInit)(dwThreadId, nDeviceId);
            pAccelerator->ReleaseCurrentDeviceContext();
			pAccelerator->Unlock();
		}
		return TRUE;
	}

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Executes the initializer, if present.
	/// 			</summary>
	///
	/// <remarks>	crossbac, 8/13/2013. </remarks>
	///
	/// <returns>	true if it succeeds, false if it fails. </returns>
	///-------------------------------------------------------------------------------------------------

	BOOL 
	CompiledKernel::InvokeInitializer(
		DWORD dwThreadId
		)
	{
		std::vector<Accelerator*>::iterator vi;
		LPFNTASKINITIALIZER lpfnInit = (LPFNTASKINITIALIZER) m_lpvInitializerProcAddress;
		if(lpfnInit) {
			(*lpfnInit)(dwThreadId, 0);
		}
		return TRUE;
	}

};
