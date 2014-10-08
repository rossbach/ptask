//--------------------------------------------------------------------------------------
// File: CLTask.cpp
// Copyright (c) Microsoft Corporation. All rights reserved.
// maintainer:crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifdef OPENCL_SUPPORT
#include "primitive_types.h"
#include "cltask.h"
#include "oclhdr.h"
#include "InputPort.h"
#include "MetaPort.h"
#include "OutputPort.h"
#include "ptaskutils.h"
#include "PCLBuffer.h"
#include "claccelerator.h"
#include "PTaskRuntime.h"
#include "CLAsyncContext.h"
#include "Recorder.h"
#include <map>
#include <vector>
#include <assert.h>
using namespace std;

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="hTerminateEvt">    Handle of the graph terminate event. </param>
    /// <param name="hStopEvent">       Handle of the stop event. </param>
    /// <param name="hRunningEvent">    Handle of the running event. </param>
    /// <param name="pCompiledKernel">  The CompiledKernel associated with this task. </param>
    ///-------------------------------------------------------------------------------------------------

    CLTask::CLTask(
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
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    CLTask::~CLTask() {
        map<Accelerator*, cl_kernel>::iterator mi;
        map<Accelerator*, cl_program>::iterator pi;
        for(mi=m_pCSMap.begin(); mi!=m_pCSMap.end(); mi++) {
    	    // clReleaseKernel(mi->second);  
        }
        for(pi=m_pModuleMap.begin(); pi!=m_pModuleMap.end(); pi++) {
    	    clReleaseProgram(pi->second);  
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Loads a source. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="cFilename">        Filename of the file. </param>
    /// <param name="cPreamble">        code that is prepended to the loaded file, typically a set of #defines or a header. </param>
    /// <param name="szFinalLength">    [in,out] returned length of the code string. </param>
    ///
    /// <returns>   the source string if succeeded, NULL otherwise. </returns>
    ///-------------------------------------------------------------------------------------------------

    char* 
    CLTask::LoadSource(
	    const char* cFilename, 
	    const char* cPreamble, 
	    size_t* szFinalLength
	    )
    {
        // locals 
        FILE* pFileStream = NULL;
        size_t szSourceLength;

        if(fopen_s(&pFileStream, cFilename, "rb") != 0) 
            return NULL;

        size_t szPreambleLength = strlen(cPreamble);

        // get the length of the source code
        fseek(pFileStream, 0, SEEK_END); 
        szSourceLength = ftell(pFileStream);
        fseek(pFileStream, 0, SEEK_SET); 

        // allocate a buffer for the source code string and read it in
        char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1); 
        memcpy(cSourceString, cPreamble, szPreambleLength);
        if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1){
            fclose(pFileStream);
            free(cSourceString);
            return 0;
        }

        fclose(pFileStream);
        if(szFinalLength != 0)
            *szFinalLength = szSourceLength + szPreambleLength;
        cSourceString[szSourceLength + szPreambleLength] = '\0';
        return cSourceString;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Creates this object. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pAccelerators">    [in] non-null, the accelerators to compile for. </param>
    /// <param name="pKernel">          [in,out] If non-null, the kernel. </param>
    ///
    /// <returns>   HRESULT (use SUCCEEDED/FAILED macros) </returns>
    ///-------------------------------------------------------------------------------------------------

    HRESULT 
    CLTask::Create( 
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
            cl_kernel pShader = (cl_kernel) pKernel->GetPlatformSpecificBinary(pAccelerator);
            cl_program pModule = (cl_program) pKernel->GetPlatformSpecificModule(pAccelerator);
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
    CLTask::PlatformSpecificOnGraphComplete(
        VOID
        )
    {
    }

    //--------------------------------------------------------------------------------------
    // Bind a compute shader (preparing to dispatch it)
    // @param pd3dImmediateContext  device context
    //--------------------------------------------------------------------------------------
    BOOL
    CLTask::BindExecutable(
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
    CLTask::PlatformSpecificBindInput(
        Port * pPort, 
        int ordinal, 
        UINT uiActualIndex, 
        PBuffer * pBuffer
        )
    {
        // bind to OpenCL kernel args!
        UNREFERENCED_PARAMETER(uiActualIndex);
        UNREFERENCED_PARAMETER(pPort);
        cl_int res = CL_SUCCESS;
        cl_mem devBuffer = (cl_mem) pBuffer->GetBuffer();
        cl_kernel pCS = m_pCSMap[m_pDispatchAccelerator];
		if(CL_SUCCESS != (res = clSetKernelArg(pCS, ordinal, sizeof(cl_mem), (void*)&devBuffer))) {
            PTask::Runtime::HandleError("clSetKernelArg failed in CLTask::PlatformSpecificBindInput\n");
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
    CLTask::PlatformSpecificBindOutput(
        Port * pPort, 
        int ordinal, 
        UINT uiActualIndex, 
        PBuffer * pBuffer
        )
    {
        UNREFERENCED_PARAMETER(uiActualIndex);
        UNREFERENCED_PARAMETER(pPort);
        cl_int res = CL_SUCCESS;
        assert(pBuffer != NULL);
		cl_mem pDevBuffer = (cl_mem) pBuffer->GetBuffer();
        cl_kernel pCS = m_pCSMap[m_pDispatchAccelerator];
		if(CL_SUCCESS != (res = clSetKernelArg(pCS, ordinal, sizeof(cl_mem), (void*)&pDevBuffer))) {
            PTask::Runtime::HandleError("clSetKernelArg failed in CLTask::PlatformSpecificBindOutput\n");
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
    CLTask::PlatformSpecificBindConstant(
        Port * pPort, 
        int ordinal, 
        UINT uiActualIndex, 
        PBuffer * pBuffer
        )
    {
        UNREFERENCED_PARAMETER(uiActualIndex);

        cl_kernel pCS = m_pCSMap[m_pDispatchAccelerator];
        if(pPort->IsScalarParameter()) {

            // we don't need a device side buffer to back
            // this data, since it's describing a scalar value
            // that is a formal parameter to the kernel function. 
            BindParameter(pCS, pBuffer, pPort, ordinal);

        } else {

            cl_mem pParam = (cl_mem) pBuffer->GetBuffer();
            cl_int err = clSetKernelArg(pCS, ordinal, sizeof(cl_mem), (void*)&pParam);
            if(err != CL_SUCCESS) {
                PTask::Runtime::HandleError("clSetKernelArg failed in CLTask::PlatformSpecificBindConstant\n");
                return FALSE;
            }
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
    CLTask::PlatformSpecificFinalizeBindings(
        VOID
        )
    {
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Bind a parameter. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="pCS">      The create struct. </param>
    /// <param name="pBlock">   [in,out] If non-null, the block. </param>
    /// <param name="pPort">    [in,out] If non-null, the port. </param>
    /// <param name="ordinal">  [in,out] The ordinal. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    CLTask::BindParameter(
        cl_kernel pCS, 
        PBuffer * pBuffer,
        Port * pPort,
	    int& ordinal
	    ) 
    {
        // note: call with pBlock AddRef'ed
        // caller is responsible for incrementing the ordinal
	    cl_int err;
        Datablock * pBlock = pBuffer->GetParent();
        assert(pBlock->LockIsHeld());
        assert(pBlock->IsScalarParameter() || pBlock->GetTemplate() == NULL);
        assert(pBlock->HasValidDataBuffer());
	    void * pHostBuffer = pBuffer->GetBuffer();
        PTASK_PARM_TYPE pType = (pBlock->GetTemplate() != NULL) ? 
            pBlock->GetParameterType() : pPort->GetParameterType();
	    switch(pType) {
	    case PTPARM_INT:
		    err = clSetKernelArg(pCS, ordinal, sizeof(cl_int), (void*)pHostBuffer);
		    assert(err == CL_SUCCESS);
		    break;
	    case PTPARM_FLOAT:
		    err = clSetKernelArg(pCS, ordinal, sizeof(cl_float), (void*)pHostBuffer);
		    assert(err == CL_SUCCESS);
		    break;
	    default:
		    assert(false && "byval struct support not present in PTask OpenCL backend");
		    break;
	    }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Estimate global size of the thread groups. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT
    CLTask::EstimateGlobalSize() {
	    UINT sz = m_nPreferredXDim * m_nPreferredYDim * m_nPreferredZDim;
	    return sz; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Runs this CLTask. All inputs/outputs are assumed to be bound already. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name=""> none </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    CLTask::PlatformSpecificDispatch( 
	    VOID
	    )
    {
        assert(m_pDispatchAccelerator != NULL);
        assert(m_pDispatchAccelerator->LockIsHeld());

        // no relative ordering guarantees are given between kernel launches on a command queue, and
        // memory object manipulations. (other than that they start in order). In particular, producers
        // of buffers consumed by this kernel may not be complete, unless we explicitly wait for them,
        // or wait until all commands on this queue are complete. For now, just make sure the queue is
        // complete.
	    cl_command_queue pQueue = ((CLAccelerator*)m_pDispatchAccelerator)->GetQueue();
        clFinish(pQueue);

        cl_kernel pCS = m_pCSMap[m_pDispatchAccelerator];
	    size_t szGlobal = (size_t) EstimateGlobalSize();
	    size_t szLocal = DEFAULT_GROUP_SIZE; // eek. We need a way to deal with this.
	    szGlobal = ptaskutils::roundup((int) szLocal, (int) szGlobal);
        
	    cl_int ciErr = clEnqueueNDRangeKernel(pQueue, pCS, 1, NULL, &szGlobal, NULL, 0, NULL, NULL);
		
        if(ciErr != CL_SUCCESS) {
            PTask::Runtime::HandleError("%s:%s failed\n", __FUNCTION__, m_lpszTaskName);
            return FALSE;
        }
        
        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a compute geometry. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="tgx">  (optional) the thread group x dimensions. </param>
    /// <param name="tgy">  (optional) the thread group y dimensions. </param>
    /// <param name="tgz">  (optional) the thread group z dimensions. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CLTask::SetComputeGeometry(
	    int tgx,
	    int tgy,
	    int tgz
	    )
    {
        // We don't want to record use of this setter from within Task constructors,
        // only subsequent use at the application level.
        if (nullptr != m_lpszTaskName)
        {
            Recorder::Record(new PTask::SetComputeGeometry(this, tgx, tgy, tgz));
        }
	    m_nPreferredXDim = tgx;
	    m_nPreferredYDim = tgy;
	    m_nPreferredZDim = tgz;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a block and grid size. </summary>
    ///
    /// <remarks>   Crossbac, 12/22/2011. </remarks>
    ///
    /// <param name="grid">     The grid. </param>
    /// <param name="block">    The block. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    CLTask::SetBlockAndGridSize(
	    PTASKDIM3 grid, 
	    PTASKDIM3 block
	    )
    {
        // We don't want to record use of this setter from within Task constructors,
        // only subsequent use at the application level.
        if (nullptr != m_lpszTaskName)
        {
            Recorder::Record(new PTask::SetBlockAndGridSize(this, grid, block));
        }

    #ifdef DEBUG
	    PTask::Runtime::Warning("WARNING: Setting block and grid size unimplemented for Open CL tasks.\n\n");
    #endif
    }
};

#endif

