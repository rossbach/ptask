//--------------------------------------------------------------------------------------
// File: HostTask.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#include <stdio.h>
#include <crtdbg.h>
#include <assert.h>
#include "HostTask.h"
#include "datablock.h"
#include "PHBuffer.h"
#include "InputPort.h"
#include "OutputPort.h"
#include "MetaPort.h"
#include "Scheduler.h"
#include "PTaskRuntime.h"
#include "hostaccelerator.h"
#include "HostAsyncContext.h"
#include "Recorder.h"
#include "instrumenter.h"
#include <vector>
using namespace std;

#define ALIGN_UP(offset, alignment) \
    (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

#ifdef ADHOC_STATS
#define on_psdispatch_enter()  OnPSDispatchEnter()
#define on_psdispatch_exit()   OnPSDispatchExit()
#else
#define on_psdispatch_enter()  
#define on_psdispatch_exit()   
#endif

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 6/18/2012. </remarks>
    ///
    /// <param name="hRuntimeTerminateEvt"> Handle of the graph terminate event. </param>
    /// <param name="hGraphTeardownEvent">  Handle of the stop event. </param>
    /// <param name="hGraphStopEvent">      Handle of the running event. </param>
    /// <param name="hGraphRunningEvent">   The graph running event. </param>
    /// <param name="pCompiledKernel">  The CompiledKernel associated with this task. </param>
    ///-------------------------------------------------------------------------------------------------

    HostTask::HostTask(
        __in HANDLE hRuntimeTerminateEvt, 
        __in HANDLE hGraphTeardownEvent, 
        __in HANDLE hGraphStopEvent, 
        __in HANDLE hGraphRunningEvent,
        __in CompiledKernel * pCompiledKernel
        ) : Task(hRuntimeTerminateEvt, 
                 hGraphTeardownEvent,
                 hGraphStopEvent,
                 hGraphRunningEvent,
                 pCompiledKernel)
    {
        SetComputeGeometry(1, 1, 1);
        m_bGeometryExplicit = FALSE;
        m_bThreadBlockSizesExplicit = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   crossbac, 6/18/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    HostTask::~HostTask() {}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates this object. </summary>
    ///
    /// <remarks>   crossbac, 6/18/2012. </remarks>
    ///
    /// <param name="pAccelerators">    [in,out] [in,out] If non-null, the accelerators. </param>
    /// <param name="pKernel">          [in,out] If non-null, the kernel. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    HostTask::Create( 
        std::set<Accelerator*>& pAccelerators, 
        CompiledKernel * pKernel
        )
    {
        if(!pKernel) return E_FAIL;
        BOOL bSuccess = FALSE;
        set<Accelerator*>::iterator vi;
        for(vi=pAccelerators.begin(); vi!=pAccelerators.end(); vi++) {
            Accelerator * pAccelerator = *vi;
            m_eAcceleratorClass = pAccelerator->GetClass();
            FARPROC pShader = (FARPROC) pKernel->GetPlatformSpecificBinary(pAccelerator);
            HMODULE pModule = (HMODULE) pKernel->GetPlatformSpecificModule(pAccelerator);
            if(pShader != NULL) {
                assert(pModule != NULL);
                bSuccess = TRUE;
                m_pCSMap[pAccelerator] = pShader;
                m_pModuleMap[pAccelerator] = pModule;
            }
        }
        return bSuccess ? S_OK : E_FAIL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   When the graph is complete, (indicated because Graph.Run was called), this method
    ///             is called on every task to allow tasks to perform and one-time initializations
    ///             that cannot be performed without knowing that the structure of the graph is now
    ///             static. For example, computing parameter offset maps for dispatch.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/5/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    HostTask::PlatformSpecificOnGraphComplete(
        VOID
        )
    {

    }

    //--------------------------------------------------------------------------------------
    // Bind a compute shader (preparing to dispatch it)
    // @param pd3dImmediateContext  device context
    //--------------------------------------------------------------------------------------
    BOOL
    HostTask::BindExecutable(
        VOID
        ) 
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Perform the platform-specific work required to bind an
    /// 			individual input parameter. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="ordinal">  [in,out] The ordinal. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostTask::PlatformSpecificBindInput(
        Port * pPort, 
        int ordinal, 
        UINT uiActualIndex, 
        PBuffer * pBuffer
        )
    {
        UNREFERENCED_PARAMETER(uiActualIndex);
        UNREFERENCED_PARAMETER(ordinal);

        if(pPort->IsFormalParameter()) {
            int nParmIdx = pPort->GetFormalParameterIndex();
            assert(m_pParameters.find(nParmIdx) == m_pParameters.end());
            m_pParameters[nParmIdx] = pBuffer->GetBuffer();
            m_pParameterPorts[nParmIdx] = pPort;
            m_pParameterDatablockMap[nParmIdx] = pBuffer->GetParent();
        } else {
            assert(false && "global scope data not supported for HostTasks");
            PTask::Runtime::HandleError("%s: global scope data not supported for HostTasks", __FUNCTION__);
            return FALSE;
        }
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Perform the platform-specific work required to bind an
    /// 			individual output parameter. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="ordinal">  [in,out] The ordinal. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostTask::PlatformSpecificBindOutput(
        Port * pPort, 
        int ordinal, 
        UINT uiActualIndex, 
        PBuffer * pBuffer
        )
    {
        UNREFERENCED_PARAMETER(uiActualIndex);
        UNREFERENCED_PARAMETER(ordinal);

        if(pPort->IsFormalParameter()) {
            int nParmIdx = pPort->GetFormalParameterIndex();
            assert(m_pParameters.find(nParmIdx) == m_pParameters.end());
            void * pHostBuffer = pBuffer->GetBuffer();
            m_pParameters[nParmIdx] = pHostBuffer;
            m_pParameterPorts[nParmIdx] = pPort;
            m_pParameterDatablockMap[nParmIdx] = pBuffer->GetParent();
        } else {
            assert(false && "global scope data unsupported for HostTasks");
            PTask::Runtime::HandleError("%s: global scope data not supported for HostTasks", __FUNCTION__);
            return FALSE;
        }
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Perform the platform-specific work required to bind an
    /// 			individual input parameter. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="ordinal">  [in,out] The ordinal. </param>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostTask::PlatformSpecificBindConstant(
        Port * pPort, 
        int ordinal, 
        UINT uiActualIndex, 
        PBuffer * pBuffer
        )
    {
        UNREFERENCED_PARAMETER(uiActualIndex);
        UNREFERENCED_PARAMETER(ordinal);

        if(pPort->IsFormalParameter()) {
            // this is actually a formal parameter 
            // for the to-be-invoked kernel function
            int * piValue;
            void * pHostBuffer = pBuffer->GetBuffer();
            assert(pHostBuffer != NULL);
            int nParmIdx = pPort->GetFormalParameterIndex();
            assert(m_pParameters.find(nParmIdx) == m_pParameters.end());
            m_pParameterDatablockMap[nParmIdx] = pBuffer->GetParent();
            switch(pPort->GetParameterType()) {
            case PTPARM_INT:
                piValue = (int*) pHostBuffer;
                m_pParameters[nParmIdx] = (void*) *piValue;
                m_pParameterPorts[nParmIdx] = pPort;
                break;
            case PTPARM_FLOAT:
                // intentionally using int since we
                // have to typecast the raw float value
                // to a void*. 
                piValue = (int*) pHostBuffer;
                m_pParameters[nParmIdx] = (void*) *piValue;
                m_pParameterPorts[nParmIdx] = pPort;
                break;
            case PTPARM_BYVALSTRUCT: 
                // this case is not fundamentally different from
                // the other cases, except that we rely on the caller
                // to know the type information, and because we *don't*
                // have the type information here, we are forced to pass 
                // it by reference. Since this is supposed to be a binding
                // to a constant, we will be violating the programmer's intent
                // by doing this, but presumably the transformation is safe: 
                // a constant should not be modified anyway. 
                m_pParameters[nParmIdx] = pHostBuffer;
                m_pParameterPorts[nParmIdx] = pPort;
                break;
            default: 
                // everything else requires pointers
                m_pParameters[nParmIdx] = pHostBuffer;
                m_pParameterPorts[nParmIdx] = pPort;
                break;
            }
        } else { 
            assert(false && "no global scope support in HostTasks");
            PTask::Runtime::HandleError("%s: global scope data not supported for HostTasks", __FUNCTION__);
            return FALSE;
        }
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the platform specific finalize bindings. </summary>
    ///
    /// <remarks>   Crossbac, 1/5/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostTask::PlatformSpecificFinalizeBindings(
        VOID
        )
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Platform-specific dispatch if the task has no dependences on other accelerators.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pCS">  The function pointer address for dispatch. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostTask::PlatformSpecificDispatchNoDependences(
        __in FARPROC pCS
        )
    {
        if(pCS == NULL)
            return FALSE;

        LPFNHOSTTASK lpfnKernel = (LPFNHOSTTASK) pCS;
        UINT nArguments = (UINT) m_pParameters.size();
        if(nArguments) {
            assert(nArguments <= MAXARGS);
            for(UINT i=0; i<nArguments; i++) {
                std::map<int, void*>::iterator mi = m_pParameters.find(i);
                if(mi==m_pParameters.end()) {
                    assert(mi!=m_pParameters.end());
                    if(PTask::Runtime::HandleError("%s::%s(%d): attempt to bind non-existent parameter %d on %s\n",
                                                   __FILE__,
                                                   __FUNCTION__,
                                                   __LINE__,
                                                   i,
                                                   m_lpszTaskName)) return FALSE;
                } else {                
                    m_ppArgs[i] = mi->second;
                }
            }
        }

        for(UINT x=0; x<m_nPreferredXDim; x++) {
            for(UINT y=0; y<m_nPreferredYDim; y++) {
                for(UINT z=0; z<m_nPreferredZDim; z++) {
                    on_psdispatch_enter();
                    lpfnKernel(nArguments, m_ppArgs);
                    on_psdispatch_exit();
                }
            }
        }
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Platform-specific dispatch if the task has dependences on other accelerators.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pCS">      The function pointer address for dispatch. </param>
    /// <param name="nDeps">    The number dependent assignments. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostTask::PlatformSpecificDispatchWithDependences(
        __in FARPROC pCS,
        __in UINT nDeps
        )
    {
        assert(nDeps != 0);
        if(nDeps == 0) {            
            return FALSE;
        }

        Lock();
        std::set<Accelerator*> vContexts;
        for(UINT i=0; i<nDeps; i++) {
            Accelerator * pDepAcc = GetAssignedDependentAccelerator(i);
            m_ppDeps[i] = pDepAcc->GetDevice();
            vContexts.insert(pDepAcc);
        }
        Unlock();

        BOOL bSuccess = TRUE;
        UINT nArguments = (UINT) m_pParameters.size();
        if(nArguments != 0) {
            if(bSuccess) {
                for(UINT i=0; i<nArguments; i++) {
                    std::map<int, void*>::iterator mi = m_pParameters.find(i);
                    std::map<int, Port*>::iterator pi = m_pParameterPorts.find(i);
                    if(mi==m_pParameters.end() || pi==m_pParameterPorts.end()) {
                        assert(mi!=m_pParameters.end());
                        assert(pi!=m_pParameterPorts.end());
                        if(PTask::Runtime::HandleError("%s::%s(%d): attempt to bind non-existent parameter %d on %s\n",
                                                       __FILE__,
                                                       __FUNCTION__,
                                                       __LINE__,
                                                       i,
                                                       m_lpszTaskName)) return FALSE;
                    } else {                
                        m_ppArgs[i] = mi->second;
                        Port * pPort = pi->second;
                        m_pbIsDependentBinding[i] = FALSE;
                        m_pvDeviceBindings[i] = NULL;
                        if(pPort->HasDependentAcceleratorBinding()) {
                            Accelerator * pDepAcc = GetAssignedDependentAccelerator(pPort);
                            m_pbIsDependentBinding[i] = TRUE;
                            m_pvDeviceBindings[i] = pDepAcc->GetDevice();
                        } 
                    }
                }
            }
        }

        if(bSuccess) {
            BOOL bCtxChange = FALSE;
            if(vContexts.size() == 1) {
                Accelerator * pDepAcc = *(vContexts.begin());
                bCtxChange = !pDepAcc->IsDeviceContextCurrent();
                if(bCtxChange) pDepAcc->MakeDeviceContextCurrent();
            } else if(vContexts.size() > 1) {
                PTask::Runtime::Warning("HostTask with multiple dependent accelerators\n"
                                        "PTask cannot set the device context (which to use?)");
            } 
            LPFNDEPHOSTTASK lpfnKernel = (LPFNDEPHOSTTASK) pCS;
            for(UINT x=0; x<m_nPreferredXDim; x++) {
                for(UINT y=0; y<m_nPreferredYDim; y++) {
                    for(UINT z=0; z<m_nPreferredZDim; z++) {

                        on_psdispatch_enter();
                        lpfnKernel(nArguments,             // number of function args to unpack
                                   m_ppArgs,               // pointer to void* argumments
                                   m_pbIsDependentBinding, // BOOL per arg: true means arg is valid in dep mem space
                                   m_pvDeviceBindings,     // per arg dependent dev binding, valid when above is true
                                   nDeps,                  // number of dependent accelerator ids
                                   m_ppDeps);              // ids of dependent accelerators
                        on_psdispatch_exit();
                    }
                }
            }
            if(vContexts.size() == 1) {
                Accelerator * pDepAcc = *(vContexts.begin());
                if(bCtxChange) pDepAcc->ReleaseCurrentDeviceContext();
            }
        }

        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Platform-specific dispatch if the task has dependences on other accelerators.
    ///             This version extends the PlatformSpecificDispatchWithDependences version
    ///             with the ability to provide other platform-specific objects such as stream
    ///             handles through a struct/descriptor based interface. Currently, this is
    ///             called if m_bRequestDependentPSObjects is true, otherwise, legacy versions
    ///             are called.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="pCS">      The function pointer address for dispatch. </param>
    /// <param name="nDeps">    The number dependent assignments. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostTask::PlatformSpecificDispatchWithDependencesEx(
        __in FARPROC pCS,
        __in UINT nDeps
        )
    {
        assert(nDeps != 0);
        if(nDeps == 0) {            
            return FALSE;
        }

        Lock();
        std::set<Accelerator*> vContexts;
        for(UINT i=0; i<nDeps; i++) {
            Accelerator * pDepAcc = GetAssignedDependentAccelerator(i);
            m_ppDeps[i] = pDepAcc->GetDevice();
            vContexts.insert(pDepAcc);
            m_ppStreams[i] = m_bRequestDependentPSObjects ?
                m_vDispatchAsyncContexts[pDepAcc]->GetPlatformContextObject() :
                NULL;
        }
        Unlock();

        BOOL bSuccess = TRUE;
        UINT nArguments = (UINT) m_pParameters.size();
        if(nArguments != 0) {
            if(bSuccess) {
                for(UINT i=0; i<nArguments; i++) {
                    std::map<int, void*>::iterator mi = m_pParameters.find(i);
                    std::map<int, Port*>::iterator pi = m_pParameterPorts.find(i);
                    std::map<int, Datablock*>::iterator di=m_pParameterDatablockMap.find(i);
                    if(mi==m_pParameters.end() || 
                       pi==m_pParameterPorts.end() || 
                       di==m_pParameterDatablockMap.end()) {

                        assert(mi!=m_pParameters.end());
                        assert(pi!=m_pParameterPorts.end()); 
                        assert(di!=m_pParameterDatablockMap.end());
                        if(PTask::Runtime::HandleError("%s::%s(%d): attempt to bind non-existent parameter %d on %s\n",
                                                       __FILE__,
                                                       __FUNCTION__,
                                                       __LINE__,
                                                       i,
                                                       m_lpszTaskName)) return FALSE;
                    } else {                
                        m_ppArgs[i] = mi->second;
                        m_ppDatablocks[i] = di->second;
                        Port * pPort = pi->second;
                        m_pbIsDependentBinding[i] = FALSE;
                        m_pvDeviceBindings[i] = NULL;
                        if(pPort->HasDependentAcceleratorBinding()) {
                            Accelerator * pDepAcc = GetAssignedDependentAccelerator(pPort);
                            m_pbIsDependentBinding[i] = TRUE;
                            m_pvDeviceBindings[i] = pDepAcc->GetDevice();
                        }
                    }
                }
            }
        }

        if(bSuccess) {

            BOOL bCtxChange = FALSE;
            if(vContexts.size() == 1) {
                Accelerator * pDepAcc = *(vContexts.begin());
                bCtxChange = !pDepAcc->IsDeviceContextCurrent();
                if(bCtxChange) pDepAcc->MakeDeviceContextCurrent();
            } else if(vContexts.size() > 1) {
                PTask::Runtime::Warning("HostTask with multiple dependent accelerators\n"
                                        "PTask cannot set the device context (which to use?)");
            } 

            DEPENDENTCONTEXT vContext;
            LPFNDEPHOSTTASKEX lpfnKernel = (LPFNDEPHOSTTASKEX) pCS;
            memset(&vContext, 0, sizeof(vContext));
            vContext.cbDependentContext = sizeof(vContext);
            vContext.nArguments = nArguments;
            vContext.ppArguments = m_ppArgs;
            vContext.ppDatablocks = m_ppDatablocks;
            vContext.nDependentAccelerators = nDeps;
            vContext.pDependentDevices = m_ppDeps;
            vContext.pbIsDependentBinding = m_pbIsDependentBinding;
            vContext.pvDeviceBindings = m_pvDeviceBindings;
            vContext.pStreams = m_ppStreams;
            vContext.lpszTaskName = m_lpszTaskName;

            for(UINT x=0; x<m_nPreferredXDim; x++) {
                for(UINT y=0; y<m_nPreferredYDim; y++) {
                    for(UINT z=0; z<m_nPreferredZDim; z++) {
                        on_psdispatch_enter();
                        lpfnKernel(&vContext);
                        on_psdispatch_exit();
                    }
                }
            }

            if(vContexts.size() == 1) {
                Accelerator * pDepAcc = *(vContexts.begin());
                if(bCtxChange) pDepAcc->ReleaseCurrentDeviceContext();
            }
        }

        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Platform specific dispatch. </summary>
    ///
    /// <remarks>   crossbac, 4/24/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    HostTask::PlatformSpecificDispatch( 
        VOID
        )
    {
        assert(m_pDispatchAccelerator != NULL);
        assert(m_pDispatchAccelerator->LockIsHeld());

        BOOL bResult = FALSE;
        FARPROC pCS = m_pCSMap[m_pDispatchAccelerator];

        assert(m_bThreadBlockSizesExplicit == TRUE ||
                (m_nPreferredXDim == 1 &&
                 m_nPreferredYDim == 1 &&
                 m_nPreferredZDim == 1));

        Lock();
        UINT nDeps = GetDependentBindingClassCount();
        Unlock();

        if(nDeps == 0) {
            bResult = PlatformSpecificDispatchNoDependences(pCS);
        } else {
            if(m_bRequestDependentPSObjects) {
                bResult = PlatformSpecificDispatchWithDependencesEx(pCS, nDeps);
            } else {
                bResult = PlatformSpecificDispatchWithDependences(pCS, nDeps);
            }
        }

        m_pParameters.clear();
        m_pParameterPorts.clear();
        m_pParameterDatablockMap.clear();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the ps dispatch enter action. </summary>
    ///
    /// <remarks>   crossbac, 8/20/2013. </remarks>
    ///
    /// <param name="hStream">  The stream. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostTask::OnPSDispatchEnter(
        void
        )
    {
        record_psdispatch_entry();
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the ps dispatch exit action. </summary>
    ///
    /// <remarks>   crossbac, 8/20/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    HostTask::OnPSDispatchExit(
        VOID
        )
    {
        record_psdispatch_exit();
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a synchronization timestamp. </summary>
    ///
    /// <remarks>   crossbac, 5/9/2012. </remarks>
    ///
    /// <param name="p">    [in,out] If non-null, the p. </param>
    ///
    /// <returns>   The synchronization timestamp. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    HostTask::GetSynchronizationTimestamp(
        Accelerator * p
        )
    {
        UNREFERENCED_PARAMETER(p);
        UINT ts = 0;
        return ts;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Increment synchronise timestamp. </summary>
    ///
    /// <remarks>   crossbac, 5/9/2012. </remarks>
    ///
    /// <param name="p">    [in,out] If non-null, the p. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    HostTask::IncrementSyncTimestamp(
        Accelerator * p
        )
    {
        UNREFERENCED_PARAMETER(p);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a compute geometry. </summary>
    ///
    /// <remarks>   crossbac, 5/9/2012. </remarks>
    ///
    /// <param name="tgx">  The tgx. </param>
    /// <param name="tgy">  The tgy. </param>
    /// <param name="tgz">  The tgz. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    HostTask::SetComputeGeometry(
        int tgx,
        int tgy,
        int tgz
        )
    {
        // We don't want to record use of this setter from within Task constructors,
        // only subsequent use at the application level.
        if (nullptr != m_lpszTaskName) {
            RECORDACTION4P(SetComputeGeometry, this, tgx, tgy, tgz);
        }
        m_nPreferredXDim = tgx;
        m_nPreferredYDim = tgy;
        m_nPreferredZDim = tgz;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a block and grid size. </summary>
    ///
    /// <remarks>   crossbac, 5/9/2012. </remarks>
    ///
    /// <param name="grid">     The grid. </param>
    /// <param name="block">    The block. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    HostTask::SetBlockAndGridSize(
        PTASKDIM3 grid,
        PTASKDIM3 block
        )
    {
        // We don't want to record use of this setter from within Task constructors,
        // only subsequent use at the application level.
        if (nullptr != m_lpszTaskName) {
            UNREFERENCED_PARAMETER(grid);
            UNREFERENCED_PARAMETER(block);
            RECORDACTION(SetBlockAndGridSize, this, grid, block);
        }

        // 6/12/13: Below code was copied (we believe inadvertently) from CUTask.
        // Commenting out now. All the accelerator-specific geometry stuff should be refactored.
        // m_bThreadBlockSizesExplicit = TRUE;
        // m_pThreadBlockSize = block;
        // assert (grid.z == 1 && "Z dimension of grid must be one");
        // m_pGridSize = grid;

#ifdef DEBUG
        printf("WARNING: Setting block and grid size only possible on CUDA tasks, not Host tasks\n");
#endif

    }

};
