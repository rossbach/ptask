//--------------------------------------------------------------------------------------
// File: DXTask.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include <stdio.h>
#include <crtdbg.h>
#include "ptdxhdr.h"
#include <assert.h>
#include "DXTask.h"
#include "datablock.h"
#include "InputPort.h"
#include "PDXBuffer.h"
#include "OutputPort.h"
#include "MetaPort.h"
#include "DXAsyncContext.h"
#include <vector>
#include "Scheduler.h"
#include "PTaskRuntime.h"
#include "Recorder.h"
#include "nvtxmacros.h"
#include "PDXBuffer.h"
#include "Recorder.h"

using namespace std;

#define PLATFORMBUFFERSRV(x)  reinterpret_cast<ID3D11ShaderResourceView*>(x)
#define PLATFORMBUFFERUAV(x)  reinterpret_cast<ID3D11UnorderedAccessView*>(x)
#define PLATFORMBUFFER(x)     reinterpret_cast<ID3D11Buffer*>(x)

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///
    /// <param name="hRuntimeTerminateEvt"> The runtime terminate event. </param>
    /// <param name="hGraphTeardownEvent">  The graph teardown event. </param>
    /// <param name="hGraphStopEvent">      The graph stop event. </param>
    /// <param name="hGraphRunningEvent">   The graph running event. </param>
    /// <param name="pCompiledKernel">      [in,out] If non-null, the compiled kernel. </param>
    ///-------------------------------------------------------------------------------------------------

    DXTask::DXTask(
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
        m_ppInputSRVs = NULL;
        m_ppOutputUAVs = NULL;
        m_ppConstantBuffers = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   crossbac, 6/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    DXTask::~DXTask() {
        map<Accelerator*, ID3D11ComputeShader*>::iterator mi;
        for(mi=m_pCSMap.begin(); mi!=m_pCSMap.end(); mi++) {
            PTSRELEASE(mi->second);
        }
        if(m_ppInputSRVs) delete [] m_ppInputSRVs;
        if(m_ppOutputUAVs) delete [] m_ppOutputUAVs;
        if(m_ppConstantBuffers) delete [] m_ppConstantBuffers;
    }

    //--------------------------------------------------------------------------------------
    // Create the CS
    // @param pAccelerators list of accelerators to compile for
    // @param pKernel	    compiled kernel object
    //--------------------------------------------------------------------------------------
    HRESULT 
    DXTask::Create( 
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
            ID3D11ComputeShader* pShader = (ID3D11ComputeShader*) pKernel->GetPlatformSpecificBinary(pAccelerator);
            if(pShader != NULL) {
                bSuccess = TRUE;
                m_pCSMap[pAccelerator] = pShader;
                pShader->AddRef();
                CreateDispatchAsyncContext(pAccelerator);
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
    DXTask::PlatformSpecificOnGraphComplete(
        VOID
        )
    {
        // our input blocks may have may have a data, meta, and template channel per port, so allocate
        // the biggest array we can possibly need, but keep track of how many inputs actually wind up being used. 
        m_ppInputSRVs = new ID3D11ShaderResourceView*[m_mapInputPorts.size()*NUM_DATABLOCK_CHANNELS];
        m_ppOutputUAVs = new ID3D11UnorderedAccessView*[m_mapOutputPorts.size()*NUM_DATABLOCK_CHANNELS];
        m_ppConstantBuffers = new ID3D11Buffer*[m_mapConstantPorts.size()*NUM_DATABLOCK_CHANNELS];
    }

    //--------------------------------------------------------------------------------------
    // Bind a compute shader (preparing to dispatch it)
    // @param pd3dImmediateContext  device context
    //--------------------------------------------------------------------------------------
    BOOL
    DXTask::BindExecutable(
        VOID
        ) 
    {
        ID3D11ComputeShader * pCS = m_pCSMap[m_pDispatchAccelerator];
        assert(pCS != NULL);
        AsyncContext * pAsyncContext = GetOperationAsyncContext(m_pDispatchAccelerator, ASYNCCTXT_TASK);
        ID3D11DeviceContext* pd3dImmediateContext = (ID3D11DeviceContext*) pAsyncContext->GetPlatformContextObject();
        pd3dImmediateContext->CSSetShader( pCS, NULL, 0 );
        return (pCS != NULL);
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
    DXTask::PlatformSpecificBindInput(
        Port * pPort, 
        int ordinal, 
        UINT uiActualIndex, 
        PBuffer * pBuffer
        )
    {
        // Bind the PBuffer for this channel to the next shader resource view. 
        UNREFERENCED_PARAMETER(pPort);
        UNREFERENCED_PARAMETER(ordinal);
        assert(pBuffer != NULL);
        PDXBuffer * pDXBuffer = reinterpret_cast<PDXBuffer*>(pBuffer);
        PDXBuffer * pLockedBuffer = pDXBuffer->PlatformSpecificAcquireSync(0);
        if(pLockedBuffer != NULL)
            m_vP2PDispatchInputLocks.insert(pLockedBuffer);
        m_ppInputSRVs[uiActualIndex] = PLATFORMBUFFERSRV(pBuffer->GetBindableObject(BVT_ACCELERATOR_READABLE).psrv);
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
    DXTask::PlatformSpecificBindOutput(
        Port * pPort, 
        int ordinal, 
        UINT uiActualIndex, 
        PBuffer * pBuffer
        )
    {
        // Bind the PBuffer for this channel to the next unordered access view. 
        UNREFERENCED_PARAMETER(pPort);
        UNREFERENCED_PARAMETER(ordinal);
        assert(pBuffer != NULL);
        PDXBuffer * pDXBuffer = reinterpret_cast<PDXBuffer*>(pBuffer);
        PDXBuffer * pLockedBuffer = pDXBuffer->PlatformSpecificAcquireSync(0);
        if(pLockedBuffer != NULL)
            m_vP2PDispatchOutputLocks.insert(pLockedBuffer);
        ID3D11UnorderedAccessView * pUAV = PLATFORMBUFFERUAV(pBuffer->GetBindableObject(BVT_ACCELERATOR_WRITEABLE).puav);
        assert(pUAV != NULL);
        m_ppOutputUAVs[uiActualIndex] = pUAV;
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
    DXTask::PlatformSpecificBindConstant(
        Port * pPort, 
        int ordinal, 
        UINT uiActualIndex, 
        PBuffer * pBuffer
        )
    {
        UNREFERENCED_PARAMETER(pPort);
        UNREFERENCED_PARAMETER(ordinal);
        assert(pBuffer != NULL);
        ID3D11Buffer * pConst = PLATFORMBUFFER(pBuffer->GetBindableObject(BVT_ACCELERATOR_IMMUTABLE).pconst);
        assert(pConst != NULL);
        m_ppConstantBuffers[uiActualIndex] = pConst;
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
    DXTask::PlatformSpecificFinalizeBindings(
        VOID
        )
    {
        AsyncContext * pAsyncContext = GetOperationAsyncContext(m_pDispatchAccelerator, ASYNCCTXT_TASK);
        ID3D11DeviceContext* pd3dImmediateContext = (ID3D11DeviceContext*) pAsyncContext->GetPlatformContextObject();
        pd3dImmediateContext->CSSetConstantBuffers( 0, m_nActualConstantCount, m_ppConstantBuffers );
        pd3dImmediateContext->CSSetShaderResources( 0, (UINT) m_nActualInputCount, m_ppInputSRVs);
        pd3dImmediateContext->CSSetUnorderedAccessViews( 0, (UINT) m_nActualOutputCount, m_ppOutputUAVs, NULL );
        return TRUE;
    }
  
    //--------------------------------------------------------------------------------------
    // Estimate the dispatch dimensions for an imminent dispatch.
    //  This is only required if the user has not called SetGeometry
    //  and we are forced to make a guess about how many threads are required.
    // @param Datablock  data block we can use to infer how many threads
    //                   may be needed to compute this shader.
    //--------------------------------------------------------------------------------------
    void 
    DXTask::__estimateDispatchDimensions(
        Datablock * pBlock,
        UINT& x,
        UINT& y,
        UINT& z
        )
    {
        if(pBlock->HasTemplateChannel()) {
            // if we have a template channel, we need to infer
            // the dispatch dimensions from the data in the template
            // buffer. Not implemented yet!
            assert(FALSE);
        } else { 
            UINT nRecords = pBlock->GetRecordCount();
            if(pBlock->IsRecordStream() && nRecords != 0) {
                x = nRecords;
                y = 1;
                z = 1;
                m_bGeometryEstimated;
                m_nRecordCardinality = nRecords;
                return;
            }
            DatablockTemplate * pTemplate = pBlock->GetTemplate();
            if(pTemplate != NULL) {
                UINT uiObjectStride = pTemplate->GetStride();
                if(pTemplate->IsByteAddressable() && !pTemplate->DescribesRecordStream()) {                    
                    assert(uiObjectStride != 1 && uiObjectStride != 0); // eek!
                    UINT nObjects = pTemplate->GetXElementCount() / uiObjectStride;
                    x = nObjects;
                    y = 1;
                    z = 1;
                } else {
                    x = pTemplate->GetXElementCount();
                    y = pTemplate->GetYElementCount();   
                    z = pTemplate->GetZElementCount();   
                }
            } 
        }
    }

    //--------------------------------------------------------------------------------------
    // Estimate the dispatch dimensions for an imminent dispatch.
    //  This is only required if the user has not called SetGeometry
    //  and we are forced to make a guess about how many threads are required.
    // @param Datablock  data block we can use to infer how many threads
    //                   may be needed to compute this shader.
    //--------------------------------------------------------------------------------------
    void 
    DXTask::EstimateDispatchDimensions(
        Datablock * pBlock
        )
    {
        if(m_bGeometryExplicit) 
            return;

        if(m_bGeometryEstimated) {
#ifdef _DEBUG
            // is the established geometry compatible with
            // what this datablock suggests?
            UINT tX, tY, tZ;
            __estimateDispatchDimensions(pBlock, tX, tY, tZ);
            assert(tX == m_nPreferredXDim);
            assert(tY == m_nPreferredYDim);
            assert(tZ == m_nPreferredZDim);
#endif
            return;
        }
        __estimateDispatchDimensions(pBlock,
                                     m_nPreferredXDim,
                                     m_nPreferredYDim,
                                     m_nPreferredZDim);
    }
 
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dispatches a task. 
    ///             1. make sure that any device-resident inputs are actually resident on the
    ///                dispatch accelerator (something we can't know until the dispatch accelerator
    ///                is selected by the scheduler)
    ///             2. bind the actual shader binary, inputs, outputs and constant buffers
    ///             3. dispatch
    ///             4. unbind the actual shader binary, inputs, outputs and constant buffers
    ///             5. release our reference to any datablocks that are in flight, typically causing
    ///                inputs to get freed, unless the caller has kept a reference to them.     
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    DXTask::PlatformSpecificDispatch( 
        VOID
        )
    {
        ID3D11DeviceContext* pd3dImmediateContext = (ID3D11DeviceContext*) m_pDispatchAccelerator->GetContext();
        MARKRANGEENTER(L"DXTask::PSDispatch");
        pd3dImmediateContext->Dispatch( m_nPreferredXDim, 
                                        m_nPreferredYDim, 
                                        m_nPreferredZDim );
        MARKRANGEEXIT();
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind shader binary after ptask dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    DXTask::UnbindExecutable(
        VOID
        ) 
    {
        AsyncContext * pAsyncContext = GetOperationAsyncContext(m_pDispatchAccelerator, ASYNCCTXT_TASK);
        ID3D11DeviceContext* pd3dImmediateContext = (ID3D11DeviceContext*) pAsyncContext->GetPlatformContextObject();
        pd3dImmediateContext->CSSetShader( NULL, NULL, 0 );
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind input variables after ptask dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    DXTask::UnbindInputs(
        VOID
        ) 
    {
        Task::UnbindInputs();
        UINT nInputPorts = (UINT) m_mapInputPorts.size();
        if(nInputPorts) {
            AsyncContext * pAsyncContext = GetOperationAsyncContext(m_pDispatchAccelerator, ASYNCCTXT_TASK);
            ID3D11DeviceContext* pd3dImmediateContext = (ID3D11DeviceContext*) pAsyncContext->GetPlatformContextObject();
            ID3D11ShaderResourceView** ppSRVNULL = new ID3D11ShaderResourceView*[nInputPorts];
            memset(ppSRVNULL, 0, nInputPorts*sizeof(ID3D11ShaderResourceView*));
            pd3dImmediateContext->CSSetShaderResources( 0, nInputPorts, ppSRVNULL );
            delete [] ppSRVNULL;
            std::set<PBuffer*>::iterator si;
            for(si=m_vP2PDispatchInputLocks.begin(); si!=m_vP2PDispatchInputLocks.end(); si++) {
                PDXBuffer * pDXBuffer = reinterpret_cast<PDXBuffer*>(*si);
                pDXBuffer->PlatformSpecificReleaseSync(0);
            }        
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind output variables after ptask dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    DXTask::UnbindOutputs(
        VOID
        ) 
    {
        UINT nOutputPorts = (UINT) m_mapOutputPorts.size();
        if(nOutputPorts) {
            AsyncContext * pAsyncContext = GetOperationAsyncContext(m_pDispatchAccelerator, ASYNCCTXT_TASK);
            ID3D11DeviceContext* pd3dImmediateContext = (ID3D11DeviceContext*) pAsyncContext->GetPlatformContextObject();
            ID3D11UnorderedAccessView** ppUAViewNULL = new ID3D11UnorderedAccessView*[nOutputPorts];
            memset(ppUAViewNULL, 0, nOutputPorts * sizeof(ID3D11UnorderedAccessView*));
            pd3dImmediateContext->CSSetUnorderedAccessViews( 0, nOutputPorts, ppUAViewNULL, NULL );
            delete [] ppUAViewNULL;
            std::set<PBuffer*>::iterator si;
            for(si=m_vP2PDispatchOutputLocks.begin(); si!=m_vP2PDispatchOutputLocks.end(); si++) {
                PDXBuffer * pDXBuffer = reinterpret_cast<PDXBuffer*>(*si);
                pDXBuffer->PlatformSpecificReleaseSync(0);
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Unbind constant buffers after ptask dispatch. </summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///
    /// <param name=""> The. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    DXTask::UnbindConstants(
        VOID
        ) 
    {
        UINT nConstantPorts = (UINT) m_mapConstantPorts.size();
        if(nConstantPorts) {
            AsyncContext * pAsyncContext = GetOperationAsyncContext(m_pDispatchAccelerator, ASYNCCTXT_TASK);
            ID3D11DeviceContext* pd3dImmediateContext = (ID3D11DeviceContext*) pAsyncContext->GetPlatformContextObject();
            ID3D11Buffer** ppCBNULL = new ID3D11Buffer*[nConstantPorts];
            memset(ppCBNULL, 0, nConstantPorts * sizeof(ID3D11Buffer*));
            pd3dImmediateContext->CSSetConstantBuffers( 0, nConstantPorts, ppCBNULL );
            delete [] ppCBNULL;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a compute geometry. </summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///
    /// <param name="tgx">  The tgx. </param>
    /// <param name="tgy">  The tgy. </param>
    /// <param name="tgz">  The tgz. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    DXTask::SetComputeGeometry(
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
        m_bGeometryExplicit = TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a block and grid size. </summary>
    ///
    /// <remarks>   Crossbac, 1/3/2012. </remarks>
    ///
    /// <param name="grid">     The grid. </param>
    /// <param name="block">    The block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    DXTask::SetBlockAndGridSize(
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
        PTask::Runtime::Warning("WARNING: Setting block and grid size only possible on CUDA tasks, not DirectX tasks\n");
    }		

};
