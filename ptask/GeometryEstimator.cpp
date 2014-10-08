///-------------------------------------------------------------------------------------------------
// file:	GeometryEstimator.cpp
//
// summary:	Implements the geometry estimator class
///-------------------------------------------------------------------------------------------------

#include "GeometryEstimator.h"
#include "datablock.h"
#include "port.h"
#include "MetaPort.h"
#include "OutputPort.h"
#include "InputPort.h"
#include "task.h"
#include "cutask.h"

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Basic Input size geometry estimator. Accepts as input all the datablocks that
    /// 			will be bound to inputs for a given task, but examines only the block bound to
    /// 			parameter 0. This is a legacy function: achtung!
    /// 			</summary>
    ///
    /// <remarks>	crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="nArguments">		 	The number of arguments. </param>
    /// <param name="ppArguments">		 	[in] non-null, a vector of input data blocks. </param>
    /// <param name="pBlockDims">		 	[out] non-null, the thread block dimensions. </param>
    /// <param name="pGridDims">		 	[out] non-null, the grid dimensions . </param>
    /// <param name="nElementsPerThread">	(optional) The elements assumed by kernel code to be
    /// 									assigned to each thread. Default is 1. </param>
    /// <param name="nBasicGroupSizeX">  	(optional) size of the basic group. Default is 512. </param>
    /// <param name="nBasicGroupSizeY">  	The basic group size y coordinate. </param>
    /// <param name="nBasicGroupSizeZ">  	The basic group size z coordinate. </param>
    ///-------------------------------------------------------------------------------------------------
        
    void 
    GeometryEstimator::BasicInputSizeGeometryEstimator(
        __in   UINT nArguments, 
        __in   PTask::PTASKARGDESC ** ppArguments, 
        __out  PTask::PTASKDIM3 * pBlockDims, 
        __out  PTask::PTASKDIM3 * pGridDims,
        __in   int nElementsPerThread,
        __in   int nBasicGroupSizeX,
		__in   int nBasicGroupSizeY,
		__in   int nBasicGroupSizeZ
        )
    {
        UNREFERENCED_PARAMETER(nArguments);
        assert(ppArguments[0] != NULL);
        assert(ppArguments[0]->pBlock != NULL);
        if((ppArguments[0] == NULL) || (ppArguments[0]->pBlock == NULL)) {
            PTask::Runtime::HandleError("%s: null argument!", __FUNCTION__);
            return;
        }
        ppArguments[0]->pBlock->Lock();
        PTask::DatablockTemplate * pTemplate = ppArguments[0]->pBlock->GetTemplate();
        int n = pTemplate == NULL ?
            ppArguments[0]->pBlock->GetRecordCount() :
            ppArguments[0]->pBlock->GetDataBufferLogicalSizeBytes()/pTemplate->GetStride();
        ppArguments[0]->pBlock->Unlock();
        int nNumItemsX = (n + (nElementsPerThread - 1))/nElementsPerThread;
		int nNumItemsY = 1;
		int nNumItemsZ = 1;
        int gridDimX = (nNumItemsX + (nBasicGroupSizeX - 1))/nBasicGroupSizeX;
        int gridDimY = (nNumItemsY + (nBasicGroupSizeY - 1))/nBasicGroupSizeY;
        int gridDimZ = (nNumItemsZ + (nBasicGroupSizeZ - 1))/nBasicGroupSizeZ;
        *pGridDims  = PTask::PTASKDIM3(max(gridDimX,1), max(gridDimY,1), max(gridDimZ,1));
        *pBlockDims = PTask::PTASKDIM3(nBasicGroupSizeX, nBasicGroupSizeY, nBasicGroupSizeZ);
		assert(nBasicGroupSizeY == 1 && "current limitation on warp sizing in geometry estimator exceeded"); 
		assert(nBasicGroupSizeZ == 1 && "current limitation on warp sizing in geometry estimator exceeded");
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Max Input size geometry estimator. Accepts as input all the datablocks that will
    /// 			be bound to inputs for a given task, and takes the max over all the record counts
    /// 			to find the conservative maximum number of thread blocks that will be required to
    /// 			ensure each input element is processed.
    /// 			</summary>
    ///
    /// <remarks>	crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="nArguments">		 	The number of arguments. </param>
    /// <param name="ppArguments">		 	[in] non-null, a vector of input data blocks. </param>
    /// <param name="pBlockDims">		 	[out] non-null, the thread block dimensions. </param>
    /// <param name="pGridDims">		 	[out] non-null, the grid dimensions . </param>
    /// <param name="nElementsPerThread">	(optional) The elements assumed by kernel code to be
    /// 									assigned to each thread. Default is 1. </param>
    /// <param name="nBasicGroupSizeX">  	(optional) size of the basic group. Default is 512. </param>
    /// <param name="nBasicGroupSizeY">  	The basic group size y coordinate. </param>
    /// <param name="nBasicGroupSizeZ">  	The basic group size z coordinate. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    GeometryEstimator::MaxInputSizeGeometryEstimator(
        __in   UINT nArguments, 
        __in   PTask::PTASKARGDESC ** ppArguments, 
        __out  PTask::PTASKDIM3 * pBlockDims, 
        __out  PTask::PTASKDIM3 * pGridDims,
        __in   int nElementsPerThread,
        __in   int nBasicGroupSizeX,
		__in   int nBasicGroupSizeY,
		__in   int nBasicGroupSizeZ
        )
    {
        int nMaxNumThreads = 1;
        for(UINT i=0; i<nArguments; i++) {

            // null arguments and null blocks are OK here, since not all port/channel combinations can be
            // peeked. In particular, initializer channels and ports will claim to be ready but return null
            // on a peek. The user should not be setting up to estimate thread dims based on an initializer
            // anyway--such a situation indicates statically known sizes (on the template used to create
            // the initializer): dynamic estimation should not be used to handle such cases. 
            if(ppArguments[i] == NULL || ppArguments[i]->pBlock == NULL) 
                continue;

            // update the max based on whatever information we can gain about the dimensions of this
            // argument. In general, if there is no template, figure out the record count for the block,
            // otherwise, rely on the template to tell us the size. Note that this strategy can be
            // unreliable if there is a template but the block is dynamically sized. The original PTask
            // design was intended to have the absence of a template mean "dynamically-sized" unambiguously,
            // but there is a legacy of Dandelion application code that creates templates as place holders.
            // If the block is sealed with a record count, *and* there is a template, the value provided by
            // the block supercedes the value provided by the template. 
            Datablock * pBlock = ppArguments[i]->pBlock;
            pBlock->Lock();
            UINT nItemEstimate;
            PTask::DatablockTemplate * pTemplate = ppArguments[i]->pBlock->GetTemplate();
            UINT nRecords = pBlock->GetRecordCount();
            UINT nTemplateItems = pTemplate ? pBlock->GetDataBufferLogicalSizeBytes()/pTemplate->GetStride() : 0;
            if(pTemplate == NULL) {
                nItemEstimate = nRecords; 
            } else {
                if(nRecords == 0) {
                    nItemEstimate = nTemplateItems;
                } else {
                    nItemEstimate = min(nRecords, nTemplateItems);
                }
            }
            ppArguments[i]->pBlock->Unlock();
            int numThreads = (nItemEstimate + (nElementsPerThread - 1))/nElementsPerThread;
            nMaxNumThreads = max(nMaxNumThreads, numThreads);
        }

        int nNumItemsX = nMaxNumThreads;
		int nNumItemsY = 1;
		int nNumItemsZ = 1;
        int gridDimX = (nNumItemsX + (nBasicGroupSizeX - 1))/nBasicGroupSizeX;
        int gridDimY = (nNumItemsY + (nBasicGroupSizeY - 1))/nBasicGroupSizeY;
        int gridDimZ = (nNumItemsZ + (nBasicGroupSizeZ - 1))/nBasicGroupSizeZ;
        *pGridDims  = PTask::PTASKDIM3(max(gridDimX,1), max(gridDimY,1), max(gridDimZ,1));
        *pBlockDims = PTask::PTASKDIM3(nBasicGroupSizeX, nBasicGroupSizeY, nBasicGroupSizeZ);
		assert(nBasicGroupSizeY == 1 && "current limitation on warp sizing in geometry estimator exceeded"); 
		assert(nBasicGroupSizeZ == 1 && "current limitation on warp sizing in geometry estimator exceeded");
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Max output size geometry estimator. Accepts as input all the datablocks that will
    /// 			be bound to outputs for a given task, and takes the max over all the record
    /// 			counts to find the conservative maximum number of thread blocks that will be
    /// 			required to ensure each input element is processed. Note that this is a somewhat
    /// 			more subtle task than examining input blocks because output blocks with MetaPorts
    /// 			serving as input allocator will not be allocated yet.
    /// 			</summary>
    ///
    /// <remarks>	crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="nArguments">		 	The number of arguments. </param>
    /// <param name="ppArguments">		 	[in] non-null, a vector of input data blocks. </param>
    /// <param name="pBlockDims">		 	[out] non-null, the thread block dimensions. </param>
    /// <param name="pGridDims">		 	[out] non-null, the grid dimensions . </param>
    /// <param name="nElementsPerThread">	(optional) The elements assumed by kernel code to be
    /// 									assigned to each thread. Default is 1. </param>
    /// <param name="nBasicGroupSizeX">  	(optional) size of the basic group. Default is 512. </param>
    /// <param name="nBasicGroupSizeY">  	The basic group size y coordinate. </param>
    /// <param name="nBasicGroupSizeZ">  	The basic group size z coordinate. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    GeometryEstimator::MaxOutputSizeGeometryEstimator(
        __in   UINT nArguments, 
        __in   PTask::PTASKARGDESC ** ppArguments, 
        __out  PTask::PTASKDIM3 * pBlockDims, 
        __out  PTask::PTASKDIM3 * pGridDims,
        __in   int nElementsPerThread,
        __in   int nBasicGroupSizeX,
		__in   int nBasicGroupSizeY,
		__in   int nBasicGroupSizeZ
        )
    {
        // check whether the source port is a meta-port. If it is, consume
        // the value that the allocator will consume. Otherwise, take the max
        // of the template and the record count.
        UINT nMaxNumThreads = 1;
        for(UINT i=0; i<nArguments; i++) {
            PORTTYPE ePortType = ppArguments[i]->pSourcePort->GetPortType();
            assert(ePortType == OUTPUT_PORT || ePortType == META_PORT);
            ppArguments[i]->pBlock->Lock();
            if(ePortType == META_PORT) {
                UINT cbResult = 0;
                Datablock * pBlock = ppArguments[i]->pBlock;
                assert(pBlock != NULL);
                UINT * pInteger = (UINT*) pBlock->GetDataPointer(FALSE);
                cbResult = *pInteger;
                nMaxNumThreads = max(cbResult, nMaxNumThreads);
            } else {
                PTask::DatablockTemplate * pTemplate = ppArguments[i]->pBlock->GetTemplate();
                UINT n = pTemplate == NULL ?
                    ppArguments[i]->pBlock->GetRecordCount() :
                    ppArguments[i]->pBlock->GetDataBufferLogicalSizeBytes()/pTemplate->GetStride();
                UINT numThreads = (n + (nElementsPerThread - 1))/nElementsPerThread;
                nMaxNumThreads = max(nMaxNumThreads, numThreads);
            }
            ppArguments[i]->pBlock->Unlock();
        }
        int nNumItemsX = nMaxNumThreads;
		int nNumItemsY = 1;
		int nNumItemsZ = 1;
        int gridDimX = (nNumItemsX + (nBasicGroupSizeX - 1))/nBasicGroupSizeX;
        int gridDimY = (nNumItemsY + (nBasicGroupSizeY - 1))/nBasicGroupSizeY;
        int gridDimZ = (nNumItemsZ + (nBasicGroupSizeZ - 1))/nBasicGroupSizeZ;
        *pGridDims  = PTask::PTASKDIM3(max(gridDimX,1), max(gridDimY,1), max(gridDimZ,1));
        *pBlockDims = PTask::PTASKDIM3(nBasicGroupSizeX, nBasicGroupSizeY, nBasicGroupSizeZ);
		assert(nBasicGroupSizeY == 1 && "current limitation on warp sizing in geometry estimator exceeded"); 
		assert(nBasicGroupSizeZ == 1 && "current limitation on warp sizing in geometry estimator exceeded");    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Ports are bound to dimensions of the iteration space such that the datablock size
    /// 			maps directly to one dimension of space. Accept all port/block pairs and use those
    /// 			with an explicit binding to assemble the iteration space. 
    ///             </summary>
    ///
    /// <remarks>   crossbac, 12/20/2011. </remarks>
    ///
    /// <param name="nArguments">           The number of arguments. </param>
    /// <param name="ppArguments">          [in] non-null, a vector of input data blocks. </param>
    /// <param name="pBlockDims">           [out] non-null, the thread block dimensions. </param>
    /// <param name="pGridDims">            [out] non-null, the grid dimensions . </param>
    /// <param name="nElementsPerThread">   (optional) The elements assumed by kernel code to be
    ///                                     assigned to each thread. Default is 1. </param>
    /// <param name="nBasicGroupSizeX">     (optional) size of the basic group. Default is 32. </param>
    /// <param name="nBasicGroupSizeY">     (optional) the basic group size y coordinate. </param>
    /// <param name="nBasicGroupSizeZ">     (optional) the basic group size z coordinate. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    GeometryEstimator::ExplicitDimensionEstimator(
        UINT nArguments, 
        PTask::PTASKARGDESC ** ppArguments, 
        PTask::PTASKDIM3 * pBlockDims, 
        PTask::PTASKDIM3 * pGridDims,
        int nElementsPerThread,
        int nBasicGroupSizeX,
        int nBasicGroupSizeY,
        int nBasicGroupSizeZ
        )
    {
        UINT nNumItemsX = 1;
        UINT nNumItemsY = 1;
        UINT nNumItemsZ = 1;
        for(UINT i=0; i<nArguments; i++) {
            Port * pPort = ppArguments[i]->pSourcePort;
            PORTTYPE ePortType = pPort->GetPortType();
            Datablock * pBlock = ppArguments[i]->pBlock;
            GEOMETRYESTIMATORDIMENSION gdim = pPort->GetEstimatorDimensionBinding();
            if(gdim == GD_NONE || pBlock == NULL) continue;
            ppArguments[i]->pBlock->Lock();
            int nItems = 0;
            if(ePortType == META_PORT) {
                assert(pBlock != NULL);
                UINT * pInteger = (UINT*) pBlock->GetDataPointer(FALSE);
                nItems = *pInteger;
            } else if(ePortType == OUTPUT_PORT) {
                PTask::DatablockTemplate * pTemplate = ppArguments[i]->pPortTemplate;
                if(pTemplate != NULL) {
                    nItems = pTemplate->GetDatablockByteCount(DBDATA_IDX)/pTemplate->GetStride();
                }
            } else {
                PTask::DatablockTemplate * pTemplate = pBlock->GetTemplate();
                nItems = pTemplate == NULL ?
                    pBlock->GetRecordCount() :
                    pBlock->GetDataBufferLogicalSizeBytes()/pTemplate->GetStride();
            }
            ppArguments[i]->pBlock->Unlock();
            int numThreads = max(((nItems + (nElementsPerThread - 1))/nElementsPerThread), 1);
            switch(gdim) {
            case GD_NONE: assert(false); break;
            case GD_X: nNumItemsX = numThreads; break;
            case GD_Y: nNumItemsY = numThreads; break;
            case GD_Z: nNumItemsZ = numThreads; break;
            }
        }
        int gridDimX = (nNumItemsX + (nBasicGroupSizeX - 1))/nBasicGroupSizeX;
        int gridDimY = (nNumItemsY + (nBasicGroupSizeY - 1))/nBasicGroupSizeY;
        int gridDimZ = (nNumItemsZ + (nBasicGroupSizeZ - 1))/nBasicGroupSizeZ;
        *pGridDims  = PTask::PTASKDIM3(max(gridDimX,1), max(gridDimY,1), max(gridDimZ,1));
        *pBlockDims = PTask::PTASKDIM3(nBasicGroupSizeX, nBasicGroupSizeY, nBasicGroupSizeZ);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds peeked blocks from all the ports in the given map to the argument list. 
    /// 			Helps assemble the argument list input for an estimator </summary>
    ///
    /// <remarks>   crossbac, 5/1/2012. </remarks>
    ///
    /// <param name="pPortMap">     [in,out] If non-null, the port map. </param>
    /// <param name="ppArgs">       [in,out] If non-null, the arguments. </param>
    /// <param name="nPortIndex">   [in,out] Zero-based index of the n port. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    GeometryEstimator::AddToArgumentList(
        std::map<UINT, Port*>* pPortMap,
        PTASKARGDESC ** ppArgs,
        int &nPortIndex,
        int nMaxPortsToAdd
        )
    {
        std::map<UINT, Port*>::iterator mi;
        for(mi=pPortMap->begin(); 
            mi!=pPortMap->end() && ((nPortIndex < nMaxPortsToAdd) || nMaxPortsToAdd < 0); 
            mi++) {

            Port * pPort = mi->second;
            MetaPort * pAllocatorPort = NULL;
            OutputPort * pOPort = NULL;
            InputPort * pInOutProducer = NULL;
            PORTTYPE ePortType = pPort->GetPortType();
            if(!pPort->IsDispatchDimensionsHint())
                continue;

            switch(ePortType) {
            case META_PORT: 
                
                // do nothing--metaports are relevant only for output size estimators, and we peek them when we
                // examine the corresponding output port. The estimator function may expect an entry for every
                // port though. Provide null here. 
                ppArgs[nPortIndex++] = NULL;
                break; 

            case OUTPUT_PORT: {

                // if there is an allocator for this port, peek it. if not, look for an inout producer. If
                // there is one, peek its block. otherwise provide only the template on the port, since it is
                // the only possible allocation source. 
                pOPort = static_cast<OutputPort*>(pPort);
                pAllocatorPort = static_cast<MetaPort*>(pOPort->GetAllocatorPort());
                pInOutProducer = static_cast<InputPort*>(pOPort->GetInOutProducer());
                PTASKARGDESC* pArg = static_cast<PTASKARGDESC*>(malloc(sizeof(PTASKARGDESC)));
                if(pAllocatorPort != NULL) {
                    assert(pInOutProducer == NULL);
                    pArg->pSourcePort = pOPort;
                    pArg->pAllocator = pAllocatorPort;
                    pArg->pPortTemplate = NULL;
                    pArg->pBlock = pAllocatorPort->Peek();
                } else if(pInOutProducer != NULL) {
                    pArg->pSourcePort = pOPort;
                    pArg->pAllocator = NULL;
                    pArg->pPortTemplate = NULL;
                    pArg->pBlock = pInOutProducer->Peek();
                } else {
                    pArg->pSourcePort = pOPort;
                    pArg->pAllocator = NULL;
                    pArg->pPortTemplate = pOPort->GetTemplate();
                    pArg->pBlock = NULL;
                }
                ppArgs[nPortIndex] = pArg;
                nPortIndex++;
                break;
            }

            case INPUT_PORT:
            case STICKY_PORT:        
            case INITIALIZER_PORT:
            default: {

                // for input port types we peek the available value
                // create the appropriate descriptor and continue.
                
                PTASKARGDESC* pArg = static_cast<PTASKARGDESC*>(malloc(sizeof(PTASKARGDESC)));
                pArg->pBlock = pPort->Peek();
                pArg->pSourcePort = pPort;
                pArg->pAllocator = NULL;
                pArg->pPortTemplate = pPort->GetTemplate();
                ppArgs[nPortIndex] = pArg;
                nPortIndex++;
                break;
            }
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Adds peeked blocks from all the ports for all relevant port maps to the argument
    ///             list. Helps assemble the argument list input for an estimator.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/1/2012. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the port map. </param>
    /// <param name="pppArgs">  [in,out] If non-null, the ppp arguments. </param>
    ///
    /// <returns>   the number of arguments in the given list. </returns>
    ///-------------------------------------------------------------------------------------------------
    
    int
    GeometryEstimator::CreateEstimatorArgumentList(
        Task * pTask,
        PTASKARGDESC *** pppArgs
        )
    {
        int nTotalPorts;
        int nArgumentsAdded = 0;
        GEOMETRYESTIMATORTYPE tEstimatorType = pTask->m_tEstimatorType;
        
        switch(tEstimatorType) {

        case USER_DEFINED_ESTIMATOR:
        case EXPLICIT_DIMENSION_ESTIMATOR:

            // user defined and explicit estimators accept *all* port/block pairs as input
            nTotalPorts = (int) (pTask->m_mapInputPorts.size() + 
                                 pTask->m_mapConstantPorts.size() + 
                                 pTask->m_mapOutputPorts.size());
            *pppArgs = (PTASKARGDESC**) calloc(nTotalPorts, sizeof(PTASKARGDESC*));
            AddToArgumentList(&pTask->m_mapInputPorts, *pppArgs, nArgumentsAdded);
            AddToArgumentList(&pTask->m_mapConstantPorts, *pppArgs, nArgumentsAdded);
            AddToArgumentList(&pTask->m_mapOutputPorts, *pppArgs, nArgumentsAdded);
            break;

        case BASIC_INPUT_SIZE_ESTIMATOR:

            // basic size estimator only cares about the first *input* port
            nTotalPorts = 1;
            *pppArgs = (PTASKARGDESC**) calloc(nTotalPorts, sizeof(PTASKARGDESC*));
            AddToArgumentList(&pTask->m_mapInputPorts, *pppArgs, nArgumentsAdded, nTotalPorts);
            break;

        case MAX_INPUT_SIZE_ESTIMATOR:

            // take the max over all input blocks
            nTotalPorts = (int) (pTask->m_mapInputPorts.size() + 
                                 pTask->m_mapConstantPorts.size());
            *pppArgs = (PTASKARGDESC**) calloc(nTotalPorts, sizeof(PTASKARGDESC*));
            AddToArgumentList(&pTask->m_mapInputPorts, *pppArgs, nArgumentsAdded);
            AddToArgumentList(&pTask->m_mapConstantPorts, *pppArgs, nArgumentsAdded);
            break;

        case MAX_OUTPUT_SIZE_ESTIMATOR:

            // take the max over all output blocks
            nTotalPorts = (int) (pTask->m_mapOutputPorts.size());
            *pppArgs = (PTASKARGDESC**) calloc(nTotalPorts, sizeof(PTASKARGDESC*));
            AddToArgumentList(&pTask->m_mapOutputPorts, *pppArgs, nArgumentsAdded);
            break;
        }

        return nArgumentsAdded;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Estimate task geometry for a cuda task. This implementation is 
    /// 			platform specific because the interface for specifying launch dimensions 
    /// 			is specific to cuda.
    /// 			</summary>
    ///
    /// <remarks>   crossbac, 5/1/2012. </remarks>
    ///
    /// <param name="pTask">    [in,out] If non-null, the task. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    GeometryEstimator::EstimateCUTaskGeometry(
        Task * pTTask
        )
    {
        if(pTTask->m_tEstimatorType == NO_SIZE_ESTIMATOR)
            return;
        CUTask * pTask = static_cast<CUTask*>(pTTask);
        PTASKARGDESC ** ppArgs = NULL;
		LPFNGEOMETRYESTIMATOR lpfnEstimator = NULL;
        int nArguments = GeometryEstimator::CreateEstimatorArgumentList(pTask, &ppArgs);     
        assert(nArguments != 0);
        switch(pTask->m_tEstimatorType) {
        case USER_DEFINED_ESTIMATOR:		
			lpfnEstimator = pTask->m_lpfnEstimator;                              
			break;
		case MAX_INPUT_SIZE_ESTIMATOR:		
			lpfnEstimator = GeometryEstimator::MaxInputSizeGeometryEstimator;   
			break;
		case MAX_OUTPUT_SIZE_ESTIMATOR:		
			lpfnEstimator = GeometryEstimator::MaxOutputSizeGeometryEstimator;	 
			break;
		case EXPLICIT_DIMENSION_ESTIMATOR:  
			lpfnEstimator = GeometryEstimator::ExplicitDimensionEstimator;		 
			break;
		case BASIC_INPUT_SIZE_ESTIMATOR:	
			lpfnEstimator = GeometryEstimator::BasicInputSizeGeometryEstimator; 
			break;
		default:                            
			lpfnEstimator = GeometryEstimator::BasicInputSizeGeometryEstimator; 
			break;
		}
		lpfnEstimator(nArguments,
					  ppArgs,
                      &pTask->m_pThreadBlockSize, 
                      &pTask->m_pGridSize, 
                      pTask->m_nEstimatorElemsPerThread, 
				      pTask->m_nEstimatorGroupSizeX,
				      pTask->m_nEstimatorGroupSizeY,
				      pTask->m_nEstimatorGroupSizeZ);
        if(ppArgs) {
            while(nArguments > 0) {
                PTASKARGDESC * pArg = ppArgs[nArguments-1];
                assert(pArg != NULL);
                if(pArg != NULL) {
                    free(pArg);
                } 
                nArguments--;
            }
            free(ppArgs);
        }
    }

};