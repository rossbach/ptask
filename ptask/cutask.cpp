//--------------------------------------------------------------------------------------
// File: CUTask.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#ifdef CUDA_SUPPORT

#include <stdio.h>
#include <iostream>
#include <crtdbg.h>
#include <assert.h>
#include "CUTask.h"
#include "datablock.h"
#include "PCUBuffer.h"
#include "InputPort.h"
#include "OutputPort.h"
#include "InitializerPort.h"
#include "MetaPort.h"
#include "Scheduler.h"
#include "PTaskRuntime.h"
#include "CUAsyncContext.h"
#include "ptasklynx.h"
#include "Recorder.h"
#include "CompiledKernel.h"
#include "extremetrace.h"
#include "instrumenter.h"
#include <vector>
using namespace std;

#define PTASSERT(x) assert(x)

#define ALIGN_UP(offset, alignment) \
    (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

#ifdef ADHOC_STATS
#define on_psdispatch_enter(x)  OnPSDispatchEnter(x)
#define on_psdispatch_exit(x)   OnPSDispatchExit(x)
#else
#define on_psdispatch_enter(x)
#define on_psdispatch_exit(x)   
#endif

#define ACQUIRE_CTXNL(acc)                           \
        BOOL bFCC = !acc->IsDeviceContextCurrent();  \
        if(bFCC) acc->MakeDeviceContextCurrent();        
#define RELEASE_CTXNL(acc)                           \
        if(bFCC) acc->ReleaseCurrentDeviceContext(); \

#define ACQUIRE_CTX(acc)                           \
        acc->Lock();                               \
        ACQUIRE_CTXNL(acc);
#define RELEASE_CTX(acc)                           \
        RELEASE_CTXNL(acc);                        \
        acc->Unlock();

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="hRuntimeTerminateEvt"> Handle of the terminate. </param>
    /// <param name="hGraphTeardownEvent">  Handle of the stop event. </param>
    /// <param name="hGraphStopEvent">      Handle of the running event. </param>
    /// <param name="hGraphRunningEvent">   The graph running event. </param>
    /// <param name="pCompiledKernel">  The CompiledKernel associated with this task. </param>
    ///-------------------------------------------------------------------------------------------------

    CUTask::CUTask(
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
        m_uiParameterSize = 0;
        m_bParameterOffsetsInitialized = FALSE;
        m_hPSDispatchStart = NULL;
        m_hPSDispatchEnd = NULL;
        m_bPSDispatchEventsValid = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/20/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    CUTask::~CUTask() {
        if(m_bPSDispatchEventsValid) {
            if(m_hPSDispatchEnd != NULL) cuEventDestroy(m_hPSDispatchEnd);
            if(m_hPSDispatchStart != NULL) cuEventDestroy(m_hPSDispatchStart);
            m_hPSDispatchEnd = NULL;
            m_hPSDispatchStart = NULL;
        }
        map<Accelerator*, CUfunction>::iterator mi;
        for(mi=m_pCSMap.begin(); mi!=m_pCSMap.end(); mi++) {
            // TODO: how to release type CUfunction?
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Compile and create binary cuda kernels for the ptask. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="pAccelerators">    [in] non-null, a list of accelerator objects for which we
    ///                                 should compile. since a system may have multiple CUDA-capable
    ///                                 devices with different device attributes/features, we need a
    ///                                 different binary for each device capable of running the ptask. </param>
    /// <param name="pKernel">          [in,out] pointer to the compiled kernel object encapsulating
    ///                                 the resulting binaries.. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    CUTask::Create( 
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
            CUfunction pShader = (CUfunction) pKernel->GetPlatformSpecificBinary(pAccelerator);
            CUmodule pModule = (CUmodule) pKernel->GetPlatformSpecificModule(pAccelerator);
            if(pShader != NULL) {
                assert(pModule != NULL);
                bSuccess = TRUE;
                m_pCSMap[pAccelerator] = pShader;
                m_pModuleMap[pAccelerator] = pModule;
                CreateDispatchAsyncContext(pAccelerator);
            }
        }
        return bSuccess ? S_OK : E_FAIL;
    }

    //--------------------------------------------------------------------------------------
    // Bind a compute shader (preparing to dispatch it)
    // @param pd3dImmediateContext  device context
    //--------------------------------------------------------------------------------------
    BOOL
    CUTask::BindExecutable(
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
    CUTask::PlatformSpecificBindInput(
        Port * pPort, 
        int ordinal, 
        UINT uiActualIndex, 
        PBuffer * pBuffer
        )
    {
        UNREFERENCED_PARAMETER(uiActualIndex);
        UNREFERENCED_PARAMETER(ordinal);

        if(pPort->IsFormalParameter()) {
            assert(m_pParameterOffsets.find(pPort) != m_pParameterOffsets.end());
            int nOffset = m_pParameterOffsets[pPort];
            CUdeviceptr devBuffer = (CUdeviceptr) pBuffer->GetBuffer();
            CUfunction pCS = m_pCSMap[m_pDispatchAccelerator];
            trace5("cuParamSetv(%16llX, %d, %16llX, %d)\n", pCS, nOffset, devBuffer, sizeof(devBuffer));
            ACQUIRE_CTXNL(m_pDispatchAccelerator);
            CUresult res = cuParamSetv(pCS, nOffset, &devBuffer, sizeof(devBuffer));
            RELEASE_CTXNL(m_pDispatchAccelerator);
            if(res != CUDA_SUCCESS) { 
                PTASSERT(res == CUDA_SUCCESS);
                PTask::Runtime::HandleError("%s:%s: cuParamSetv failed in BindInputs\n", 
                                            __FUNCTION__,
                                            m_lpszTaskName);
            }
        } else {
            // this is actually a buffer that should be 
            // migrated to constant memory on the device
            // CUmodule pModule = m_pModuleMap[m_pDispatchAccelerator];
            PTask::Runtime::HandleError("%s:%s: FAILURE: Unsupported Use of global scope\n",
                                        __FUNCTION__,
                                        m_lpszTaskName);
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
    CUTask::PlatformSpecificBindOutput(
        Port * pPort, 
        int ordinal, 
        UINT uiActualIndex, 
        PBuffer * pBuffer
        )
    {
        UNREFERENCED_PARAMETER(uiActualIndex);
        UNREFERENCED_PARAMETER(ordinal);

        if(pPort->IsFormalParameter()) {
            int nOffset = m_pParameterOffsets[pPort];
            CUdeviceptr pDevBuffer = (CUdeviceptr) pBuffer->GetBuffer();
            CUfunction pCS = m_pCSMap[m_pDispatchAccelerator];
            trace5("cuParamSetv(%16llX, %d, %16llX, %d)\n", pCS, nOffset, pDevBuffer, sizeof(pDevBuffer));
            ACQUIRE_CTXNL(m_pDispatchAccelerator);
            CUresult res = cuParamSetv(pCS, nOffset, &pDevBuffer, sizeof(pDevBuffer));
            RELEASE_CTXNL(m_pDispatchAccelerator);
            if(res != CUDA_SUCCESS) {
                PTASSERT(res == CUDA_SUCCESS);
                PTask::Runtime::HandleError("%s: %s: cuParamSetv failed\n", 
                                            __FUNCTION__,
                                            m_lpszTaskName);
            }
        } else {
            // this is actually a global buffer
            // since it's an output, there is actually nothing
            // to do--it's implicitly bound, and we only care about
            // getting data back from the device after dispatch.
            // CUmodule pModule = m_pModuleMap[m_pDispatchAccelerator];
            PTask::Runtime::HandleError("%s:%s: FAILURE: Use of (unsupported) global scope\n",
                                        __FUNCTION__,
                                        m_lpszTaskName);
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
    CUTask::PlatformSpecificBindConstant(
        Port * pPort, 
        int ordinal, 
        UINT uiActualIndex, 
        PBuffer * pBuffer
        )
    {
        UNREFERENCED_PARAMETER(uiActualIndex);
        UNREFERENCED_PARAMETER(ordinal);

        BOOL bSuccess = TRUE;
        CUresult err = CUDA_SUCCESS;
        CUfunction pCS = m_pCSMap[m_pDispatchAccelerator];
        ACQUIRE_CTXNL(m_pDispatchAccelerator);

        if(pPort->IsFormalParameter()) {
            // this is actually a formal parameter 
            // for the to-be-invoked kernel function
            int * piValue;
            float * pfValue;
            void * pHostBuffer = pBuffer->GetBuffer();
            assert(pHostBuffer != NULL);
            assert(m_pParameterOffsets.find(pPort) != m_pParameterOffsets.end());
            int nOffset = m_pParameterOffsets[pPort];
            switch(pPort->GetParameterType()) {
            case PTPARM_INT:
                piValue = (int*) pHostBuffer;
                trace4("cuParamSeti(%16llX, %d, %d)\n", pCS, nOffset, *piValue);
                err = cuParamSeti(pCS, nOffset, *piValue);
                if(err != CUDA_SUCCESS) {
                    PTASSERT(err == CUDA_SUCCESS);
                    PTask::Runtime::HandleError("%s:%s: cuParamSeti failed\n", __FUNCTION__, m_lpszTaskName);
                    bSuccess = FALSE;
                }
                break;
            case PTPARM_FLOAT:
                pfValue = (float*) pHostBuffer;
                trace("cuParamSetf");
                err = cuParamSetf(pCS, nOffset, *pfValue);
                if(err != CUDA_SUCCESS) {
                    PTASSERT(err == CUDA_SUCCESS);
                    PTask::Runtime::HandleError("%s:%s: cuParamSetf failed\n", __FUNCTION__, m_lpszTaskName);
                    bSuccess = FALSE;
                }
                break;
            case PTPARM_BYVALSTRUCT: {
                UINT uiBytes = pBuffer->GetLogicalExtentBytes();
                CUresult res = cuParamSetv(pCS, nOffset, pHostBuffer, uiBytes);
                if(res != CUDA_SUCCESS) {
                    PTASSERT(res == CUDA_SUCCESS);
                    PTask::Runtime::HandleError("%s:%s: cuParamSetv failed\n", __FUNCTION__, m_lpszTaskName);
                    bSuccess = FALSE;
                }
                break;
            }
            default: {                
                CUdeviceptr pDevBuffer = (CUdeviceptr) pBuffer->GetBuffer();
                CUfunction pCS = m_pCSMap[m_pDispatchAccelerator];
                trace5("cuParamSetv(%16llX, %d, %16llX, %d)\n", pCS, nOffset, pDevBuffer, sizeof(pDevBuffer));
                CUresult res = cuParamSetv(pCS, nOffset, &pDevBuffer, sizeof(pDevBuffer));
                if(res != CUDA_SUCCESS) {
                    PTASSERT(res == CUDA_SUCCESS);
                    PTask::Runtime::HandleError("%s:%s: cuParamSetv failed\n", __FUNCTION__, m_lpszTaskName);
                    bSuccess = FALSE;
                }
                break;
            }
            }
        } else { 
            // this is actually a buffer that should be 
            // in constant memory on the device. 
            // It should have been populated in __materializeImmutableAcceleratorView,
            // so there should be nothing to do here.
        }

        RELEASE_CTXNL(m_pDispatchAccelerator);
        return bSuccess;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the platform specific finalize bindings. </summary>
    ///
    /// <remarks>   Crossbac, 1/5/2012. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CUTask::PlatformSpecificFinalizeBindings(
        VOID
        )
    {
        // nothing to do for CUDA tasks
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds a parameter indeces to 'indexmap'. Binding parameters in CUDA requires
    ///             knowing the offset at which a parameter must be bound in a call to cuParamSet*.
    ///             Because we are not guaranteed to traverse port data structures in the same order
    ///             as arguments appear in the kernel's signature, we pre-compute a map of ports ->
    ///             offsets, allowing us to visit the ports in any order at bind time. This is a two
    ///             step process: the first step (this function) is to create a map from parameter
    ///             index to port object.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="portmap">  [in,out] [in,out] If non-null, the portmap. </param>
    /// <param name="indexmap"> [in,out] [in,out] If non-null, the indexmap. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CUTask::AddParameterIndeces(
        std::map<UINT, Port*>& portmap,
        std::map<UINT, Port*>& indexmap
        ) 
    {
        assert(!m_bParameterOffsetsInitialized);
        map<UINT, Port*>::iterator mi;
        for(mi=portmap.begin(); 
            mi!=portmap.end(); mi++) {				
            Port * pPort = mi->second;
            if(pPort->IsFormalParameter()) {
                if(pPort->GetPortType() == INPUT_PORT) {
                    InputPort * pInputPort = (InputPort*) pPort;
                    Port * pConsumer = pInputPort->GetInOutConsumer();
                    if(pConsumer != NULL) {
                        // if this is the input port half of an in/out
                        // pair then then the binding will be handled
                        // by the output port. 
                        continue;
                    }
                }
                int nParmIdx = pPort->GetFormalParameterIndex();
                if(pPort->GetPortType() == OUTPUT_PORT) {
                    OutputPort * pOutputPort = (OutputPort*) pPort;
                    Port * pProducer = pOutputPort->GetInOutProducer();
                    if(pProducer != NULL) {
                        // if this is the output port half of an in/out
                        // pair then then the parameter index needs to come
                        // from the input port. 
                        nParmIdx = pProducer->GetFormalParameterIndex();
                    }
                }
                indexmap[nParmIdx] = pPort;
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Calculates the parameter offsets. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. Binding parameters in CUDA requires knowing the offset at
    ///             which a parameter must be bound in a call to cuParamSet*. Because we are not
    ///             guaranteed to traverse port data structures in the same order as arguments appear
    ///             in the kernel's signature, we pre-compute a map of ports -> offsets, allowing us
    ///             to visit the ports in any order at bind time. This is a two step process: the
    ///             second step (this function) is to traverse the parameters in order, and use the
    ///             parm-idx->port map created in step one to compute the offset of each variable. We
    ///             need only compute this once over the lifetime of a ptask.
    ///             </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CUTask::ComputeParameterOffsets(
        VOID
        ) 
    {
        if(m_bParameterOffsetsInitialized)
            return;
        trace2("(%s)->ComputeParameterOffsets\n", m_lpszTaskName);

        // JCJC if (!strcmp("SOR_1_Level_0", m_lpszTaskName))
        // JCJC {
        // JCJC     DebugBreak();
        // JCJC }

        map<UINT, Port*> indexmap;
        AddParameterIndeces(m_mapConstantPorts, indexmap);
        AddParameterIndeces(m_mapInputPorts, indexmap);
        AddParameterIndeces(m_mapOutputPorts, indexmap);
        // note that meta ports do not require parameter offsets
        // since those datablocks are consumed by the runtime!
        // now we can go through the map in order and use port type
        // information to compute the offset required
        UINT nOffset = 0;
        UINT nParmIdx = 0;
        UINT nLastParmIdx = 0xFFFFFFFF;
        map<UINT, Port*>::iterator mi;
        int nOffsets = 0;
        for(mi=indexmap.begin(); mi!=indexmap.end(); mi++, nOffsets++) {
            nParmIdx = mi->first;
            Port* pPort = mi->second;
            if(!((nLastParmIdx == 0xFFFFFFFF && nParmIdx == 0) || (nParmIdx == nLastParmIdx + 1))) {
                assert((nLastParmIdx == 0xFFFFFFFF && nParmIdx == 0) || (nParmIdx == nLastParmIdx + 1));
                PTask::Runtime::HandleError("%s:%s: index invariants failed\n", __FUNCTION__, m_lpszTaskName);
            }
            DatablockTemplate * pTemplate = pPort->GetTemplate();
            if(pTemplate == NULL) {
                assert(pTemplate != NULL);
                PTask::Runtime::HandleError("%s:%s: null template\n", __FUNCTION__, m_lpszTaskName);
            }
            if(!pPort->IsFormalParameter()) {
                assert(pPort->IsFormalParameter());
                PTask::Runtime::HandleError("%s:%s: non-formal parameter port input!\n", __FUNCTION__, m_lpszTaskName);
            }
            switch(pTemplate->GetParameterBaseType()) {
            case PTPARM_INT:
                // using cuParamSeti
                ALIGN_UP(nOffset, __alignof(int));
                trace3("PTPARM_INT: at %s->off=%d\n", pPort->GetVariableBinding(), nOffset);
                m_pParameterOffsets[pPort] = nOffset;
                nOffset += sizeof(int);
                break;
            case PTPARM_FLOAT:
                // using cuParamSetv
                ALIGN_UP(nOffset, __alignof(float));
                trace3("PTPARM_FLOAT: at %s->off=%d\n", pPort->GetVariableBinding(), nOffset);
                m_pParameterOffsets[pPort] = nOffset;
                nOffset += sizeof(float);
                break;
            case PTPARM_DOUBLE:
                // unsupported (no cuParamSetd available!)
                trace("PTPARM_DOUBLE: BAD NEWS!");
                assert(FALSE && "no CUDA support for double-precision parameters");
                PTask::Runtime::HandleError("%s:%s: CUDA support for double-precision parameters\n",
                                            __FUNCTION__,
                                            m_lpszTaskName);
                break;
            case PTPARM_BYVALSTRUCT:
                ALIGN_UP(nOffset, __alignof(float));
                trace3("PTPARM_BYVALSTRUCT: at %s->off=%d\n", pPort->GetVariableBinding(), nOffset);
                m_pParameterOffsets[pPort] = nOffset;
                nOffset += pTemplate->GetDatablockByteCount(DBDATA_IDX);
                break;
            case PTPARM_BYREFSTRUCT:
            case PTPARM_BUFFER:
            case PTPARM_NONE:
                // everything else uses cuParamSetv, so the
                // offset will always be the size of a devptr
                // because we have to create a device-side buffer
                // to handle the parameter. 
                ALIGN_UP(nOffset, __alignof(CUdeviceptr));
                trace3("PTPARM_NONE(etc): at %s->off=%d\n", pPort->GetVariableBinding(), nOffset);
                m_pParameterOffsets[pPort] = nOffset;
                nOffset += sizeof(CUdeviceptr);
                break;
            default:
                assert(false);
                PTask::Runtime::HandleError("%s:%s: found unknown parm type!\n", __FUNCTION__, m_lpszTaskName);
                break;
            }
            nLastParmIdx = nParmIdx;
        }
        m_uiParameterSize = nOffset;
        m_bParameterOffsetsInitialized = TRUE;
        trace2("(%s)->ComputeParameterOffsets complete!\n", m_lpszTaskName);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Estimate dispatch dimensions. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    CUTask::EstimateDispatchDimensions() {
        return GeometryEstimator::EstimateCUTaskGeometry(this);
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
    CUTask::PlatformSpecificOnGraphComplete(
        VOID
        )
    {
        ComputeParameterOffsets();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>  Initialize instrumentation (buffers, etc). </summary>
    ///
    /// <remarks>   t-nailaf, 06/10/2013. </remarks>
    ///
    /// <param name=""> </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    void
    CUTask::InitializeInstrumentation(
        VOID
        )
    {
#ifdef PTASK_LYNX_INSTRUMENTATION
        if(Runtime::GetInstrumentationMetric() != Runtime::INSTRUMENTATIONMETRIC::NONE)
		{
            assert(m_pDispatchAccelerator != NULL);
            assert(m_pDispatchAccelerator->LockIsHeld());
            assert(m_bParameterOffsetsInitialized);

            ACQUIRE_CTXNL(m_pDispatchAccelerator);

            lynx::initializeInstrumentation(m_lpszTaskName, Runtime::GetInstrumentationMetric(), 
                (m_pGridSize.x * m_pGridSize.y * m_pGridSize.z), 
                (m_pThreadBlockSize.x * m_pThreadBlockSize.y * m_pThreadBlockSize.z),
                m_pModuleMap[m_pDispatchAccelerator]);

            RELEASE_CTXNL(m_pDispatchAccelerator);
		}
#endif
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>  Finalize instrumentation (buffers, etc). </summary>
    ///
    /// <remarks>   t-nailaf, 06/10/2013. </remarks>
    ///
    /// <param name=""> </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------
    void
    CUTask::FinalizeInstrumentation(
        VOID
        )
    {
#ifdef PTASK_LYNX_INSTRUMENTATION
        if(Runtime::GetInstrumentationMetric() != Runtime::INSTRUMENTATIONMETRIC::NONE)
		{
            assert(m_pDispatchAccelerator != NULL);
            assert(m_pDispatchAccelerator->LockIsHeld());
            assert(m_bParameterOffsetsInitialized);

            ACQUIRE_CTXNL(m_pDispatchAccelerator);

			lynx::finalizeInstrumentation();

            RELEASE_CTXNL(m_pDispatchAccelerator);
		}
#endif
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Platform specific dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    CUTask::PlatformSpecificDispatch( 
        VOID
        )
    {
        assert(m_pDispatchAccelerator != NULL);
        assert(m_pDispatchAccelerator->LockIsHeld());
        assert(m_bParameterOffsetsInitialized);

        CUfunction pCS = m_pCSMap[m_pDispatchAccelerator];
        ACQUIRE_CTXNL(m_pDispatchAccelerator);
        
        trace3("cuParamSetSize(%16llX, %d)\n", pCS, m_uiParameterSize);
        CUresult error = cuParamSetSize(pCS, m_uiParameterSize);
        if(error != CUDA_SUCCESS) {
            PTASSERT(error == CUDA_SUCCESS);
            PTask::Runtime::HandleError("%s:%s: cuParamSetSize failed\n", __FUNCTION__, m_lpszTaskName);
            return FALSE;
        }

        // For now the user needs to tell us the CUDA block and grid sizes
        //assert (m_bThreadBlockSizesExplicit == TRUE);
        //printf("%s.cuFuncSetBlockShape(%d, %d, %d)\n", m_lpszTaskName, m_pThreadBlockSize.x, m_pThreadBlockSize.y, m_pThreadBlockSize.z);
        trace5("cuFuncSetBlockShape(%16llX, %d, %d, %d)\n", pCS, m_pThreadBlockSize.x, m_pThreadBlockSize.y, m_pThreadBlockSize.z);
        error = cuFuncSetBlockShape(pCS, m_pThreadBlockSize.x, m_pThreadBlockSize.y, m_pThreadBlockSize.z);
        if(error != CUDA_SUCCESS) {
            PTASSERT(error == CUDA_SUCCESS);
            PTask::Runtime::HandleError("%s:%s: cuFuncSetBlockShape failed (res=%d)\n",
                                        __FUNCTION__,
                                        m_lpszTaskName,
                                        error);
            return FALSE;
        }

        AsyncContext * pContext = GetOperationAsyncContext(m_pDispatchAccelerator, ASYNCCTXT_TASK);
        assert(pContext->GetAsyncContextType() == ASYNCCTXT_TASK);
        CUstream hStream = (CUstream) pContext->GetPlatformContextObject();
        // printf("%s.cuLaunchGridAsync(%d, %d)\n", m_lpszTaskName, m_pGridSize.x, m_pGridSize.y);
        trace6("%s.cuLaunchGridAsync(%16llX, %d, %d, %16llX)\n", m_lpszTaskName, pCS, m_pGridSize.x, m_pGridSize.y, hStream);

        on_psdispatch_enter(hStream);
        ptasklynx_start_timer();
        error = cuLaunchGridAsync(pCS, m_pGridSize.x, m_pGridSize.y, hStream);
        ptasklynx_stop_timer();
        on_psdispatch_exit(hStream);

        if(error != CUDA_SUCCESS) {
            PTASSERT(error == CUDA_SUCCESS);
            PTask::Runtime::HandleError("%s:%s: cuLaunchGridAsync failed (res=%d)\n",
                                        __FUNCTION__,
                                        m_lpszTaskName,
                                        error);
            return FALSE;
        }
        
        if(PTask::Runtime::GetForceSynchronous()) {
            error = cuStreamSynchronize(hStream);
            PTASSERT(error == CUDA_SUCCESS);
            error = cuCtxSynchronize();
            PTASSERT(error == CUDA_SUCCESS);
        }

        RELEASE_CTXNL(m_pDispatchAccelerator);
        return TRUE;
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
    CUTask::OnPSDispatchEnter(
        CUstream hStream
        )
    {
        if(!PTask::Runtime::GetProfilePSDispatch()) {
            record_psdispatch_entry();
            return FALSE;
        }

        if(!m_bPSDispatchEventsValid) {
            CUresult resStart = cuEventCreate(&m_hPSDispatchStart, CU_EVENT_DEFAULT);
            CUresult resStop = cuEventCreate(&m_hPSDispatchEnd, CU_EVENT_DEFAULT);
            if(resStart != CUDA_SUCCESS || resStop != CUDA_SUCCESS) {
                PTASSERT(resStart == CUDA_SUCCESS);
                PTASSERT(resStop == CUDA_SUCCESS);
                PTask::Runtime::HandleError("%s:%s: cuEventCreate failed (resStart=%d, resStop=%d)\n",
                                            __FUNCTION__,
                                            m_lpszTaskName,
                                            resStart,
                                            resStop);
                return FALSE;
            }
            m_bPSDispatchEventsValid = TRUE;
        }
        CUresult resRecord = cuEventRecord(m_hPSDispatchStart, hStream);
        if(resRecord != CUDA_SUCCESS) {
            PTASSERT(resRecord == CUDA_SUCCESS);
            PTask::Runtime::HandleError("%s:%s: cuEventRecord failed (res%d)\n",
                                        __FUNCTION__,
                                        m_lpszTaskName,
                                        cuEventRecord);
            return FALSE;
        }
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Executes the ps dispatch exit action. </summary>
    ///
    /// <remarks>   crossbac, 8/20/2013. </remarks>
    ///
    /// <param name="parameter1">   The first parameter. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    CUTask::OnPSDispatchExit(
        CUstream hStream
        )
    {
        if(!PTask::Runtime::GetProfilePSDispatch()) {
            record_psdispatch_exit();
            return FALSE;
        }

        assert(m_bPSDispatchEventsValid);
        if(!m_bPSDispatchEventsValid) 
            return FALSE;

        float fTime = 0.0f;
        cuEventRecord(m_hPSDispatchEnd, hStream);
        cuEventSynchronize(m_hPSDispatchEnd);
        cuEventElapsedTime(&fTime, m_hPSDispatchStart, m_hPSDispatchEnd);
        record_psdispatch_latency((double) fTime); 
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a synchronization timestamp. </summary>
    ///
    /// <remarks>   crossbac, 6/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///
    /// <returns>   The synchronization timestamp. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    CUTask::GetSynchronizationTimestamp(
        Accelerator * pAccelerator
        )
    {
        UNREFERENCED_PARAMETER(pAccelerator);
        assert(false && "CUTask::GetSynchronizationTimestamp() unimplemented!");
        return 0;        
        //UINT ts = 0;
        //Lock();
        //map<Accelerator*, UINT>::iterator mi;
        //mi = m_pSyncTimestamps.find(p);
        //if(mi != m_pSyncTimestamps.end()) {
        //    ts = mi->second;
        //}
        //Unlock();
        //return ts;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Increment synchronise timestamp. </summary>
    ///
    /// <remarks>   crossbac, 6/11/2012. </remarks>
    ///
    /// <param name="pAccelerator"> [in,out] If non-null, the accelerator. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CUTask::IncrementSyncTimestamp(
        Accelerator * pAccelerator
        )
    {
        UNREFERENCED_PARAMETER(pAccelerator);
        assert(false && "CUTask::IncrementSyncTimestamp() unimplemented!");
        //Lock();
        //map<Accelerator*, UINT>::iterator mi;
        //mi = m_pSyncTimestamps.find(p);
        //if(mi == m_pSyncTimestamps.end()) {
        //    m_pSyncTimestamps[p] = 1;
        //} else {
        //    UINT ts = mi->second;
        //    mi->second = ts++;
        //}
        //Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a compute geometry. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="tgx">  (optional) the thread group X dimensions. </param>
    /// <param name="tgy">  (optional) the thread group Y dimensions. </param>
    /// <param name="tgz">  (optional) the thread group Z dimensions. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CUTask::SetComputeGeometry(
        int tgx,
        int tgy,
        int tgz
        )
    {
        // We don't want to record use of this setter from within Task constructors,
        // only subsequent use at the application level.
        if (nullptr != m_lpszTaskName)
        {
            RECORDACTION4P(SetComputeGeometry, this, tgx, tgy, tgz);
        }
        m_nPreferredXDim = tgx;
        m_nPreferredYDim = tgy;
        m_nPreferredZDim = tgz;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a block and grid size. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="grid">     The grid. </param>
    /// <param name="block">    The block. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    CUTask::SetBlockAndGridSize(
        PTASKDIM3 grid,
        PTASKDIM3 block
        )
    {
        // We don't want to record use of this setter from within Task constructors,
        // only subsequent use at the application level.
        if (nullptr != m_lpszTaskName)
        {
            RECORDACTION(SetBlockAndGridSize, this, grid, block);
        }
        m_bThreadBlockSizesExplicit = TRUE;
        m_pThreadBlockSize = block;
        assert (grid.z == 1 && "Z dimension of grid must be one");
        m_pGridSize = grid;
    }

};
#endif