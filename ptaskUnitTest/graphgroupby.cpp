#include <stdio.h>
#include <iostream>
#include <crtdbg.h>
#include <ctime>
#include <sstream>
#include "accelerator.h"
#include "assert.h"
#include <vector>
#include <map>
#include <algorithm>
#include <set>
#include "graphgroupby.h"
#include "SimpleVector.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

using namespace std;
using namespace PTask;

#define HASH_TABLE_SIZE 256
#define BLOCK_SIZE 512
#define ELEMS_PER_THREAD_GROUPBY 1
typedef int KeyType;
typedef int InputType;
struct HashTableNode;

extern BOOL g_bSingleThreaded;

struct GroupbyGraphParams
{
	Graph *g;

	int inputLength;
	int numKeys;
	int numIterations;
	DatablockTemplate *hashTableTemplate;
	DatablockTemplate *inputTemplate;
	DatablockTemplate *keycountTemplate;
	DatablockTemplate *ptrTemplate;
	DatablockTemplate *intTemplate;

    UINT accelerator;
	UINT uiUidCounter;
	std::string dirName;

	std::vector<GraphInputChannel*> inputChannels;
	std::vector<GraphInputChannel*> inputLengthChannels;
	std::vector<GraphInputChannel*> hashtableChannels;
	GraphInputChannel *keyCountInChannel;

	GraphOutputChannel *outputChannel;

	GroupbyGraphParams(int l, int n, int iters, char *dName);
};

GroupbyGraphParams::GroupbyGraphParams(int l, int n, int iters, char *dName)
	:inputLength(l), numKeys(n), uiUidCounter(0), dirName(dName), numIterations(iters)
{
	hashTableTemplate = PTask::Runtime::GetDatablockTemplate("HashTableSize", sizeof(HashTableNode*), HASH_TABLE_SIZE, 1, 1);
	inputTemplate     = PTask::Runtime::GetDatablockTemplate("InputSize", sizeof(InputType), inputLength, 1, 1);
	keycountTemplate  = PTask::Runtime::GetDatablockTemplate("KeyCountTemplate", sizeof(int), 10, 1, 1);
	ptrTemplate       = PTask::Runtime::GetDatablockTemplate("PtrTemplate", sizeof(int*), 1, 1, 1);
	intTemplate       = PTask::Runtime::GetDatablockTemplate("IntTemplate", sizeof(int), PTPARM_INT);
}

void computeGridAndBlockSize(PTASKDIM3 &grid, PTASKDIM3 &block, int n)
{
	int numThreads = (n + (ELEMS_PER_THREAD_GROUPBY-1))/ELEMS_PER_THREAD_GROUPBY;
	int blockDimX  = BLOCK_SIZE;
	int gridDimX   = (numThreads + (blockDimX-1))/blockDimX;
	block = PTASKDIM3(blockDimX, 1, 1);	
	grid =  PTASKDIM3(gridDimX, 1, 1);
}

void constructCountDeallocator(Port *globalCountPort, GroupbyGraphParams& params)
{
	std::string fileName   = params.dirName + "\\groupby.sm_20.ptx";
	std::string kernelName = "deallocateCountArray";
	CompiledKernel * pKernel = PTask::Runtime::GetCompiledKernel(const_cast<char*>(fileName.c_str()), const_cast<char*>(kernelName.c_str()));
	
	int numInputPorts = 1; 
	int numOutputPorts = 0; 

	Port **pInputPorts = new Port*[numInputPorts];
	Port **pOutputPorts = NULL; 
	pInputPorts[0]  = PTask::Runtime::CreatePort(INPUT_PORT, params.ptrTemplate, params.uiUidCounter++, "globalCount(in)", 0);
	
	Task * pKernelTask = params.g->AddTask(pKernel, 
									  numInputPorts,
									  pInputPorts,
									  numOutputPorts,
									  pOutputPorts,
									  const_cast<char*>(kernelName.c_str()));
    pKernelTask->SetPriority(10);
    PTask::Runtime::SetTaskAffinity(pKernelTask, params.accelerator, PTask::AFFINITYTYPE_MANDATORY);

	PTASKDIM3 grid(1, 1, 1), block(1, 1, 1);
	pKernelTask->SetBlockAndGridSize(grid, block);

	params.g->AddInternalChannel(globalCountPort, pInputPorts[0], "globalCount_deallocateGlobalCount");
    delete [] pInputPorts;
	return;
}

void constructCleanUpNodes(Port *hashTablePort, Port *globalCountPort, GroupbyGraphParams& params)
{

    std::string fileName   = params.dirName + "\\groupby.sm_20.ptx";
	std::string kernelName = "destroyHashTableInt";
	CompiledKernel * pKernel = COMPILE_KERNEL(const_cast<char*>(fileName.c_str()), const_cast<char*>(kernelName.c_str()));
	
	int numInputPorts = 1; 
	int numOutputPorts = 0; 

	Port **pInputPorts = new Port*[numInputPorts];
	Port **pOutputPorts = NULL; 
	pInputPorts[0]  = PTask::Runtime::CreatePort(INPUT_PORT, params.hashTableTemplate, params.uiUidCounter++, "hashTable(in)", 0);
    // the hash-table is not marshallable because it is an
    // array of pointers to device-side malloc'ed items. 
    // consequently, any attempt to migrate that data will produce
    // garbage data, and we lack a facility to flatten the data
    // structure to make it possible to migrate it.
    pInputPorts[0]->SetMarshallable(FALSE);
	
	Task * pKernelTask = params.g->AddTask(pKernel, 
									  numInputPorts,
									  pInputPorts,
									  numOutputPorts,
									  pOutputPorts,
									  const_cast<char*>(kernelName.c_str()));
    pKernelTask->SetPriority(10);
    PTask::Runtime::SetTaskAffinity(pKernelTask, params.accelerator, PTask::AFFINITYTYPE_MANDATORY);

	PTASKDIM3 grid(1, 1, 1), block(HASH_TABLE_SIZE, 1, 1);
	pKernelTask->SetBlockAndGridSize(grid, block);

	params.g->AddInternalChannel(hashTablePort, pInputPorts[0], "hashTable_hashTableDestroy");
	
	constructCountDeallocator(globalCountPort, params);
    delete [] pInputPorts;
	return;

}

//reshuffleIntInt(int *arr, int n, int *out, HashTableNode<int>** hashTable, int **globalCount)
void constructShuffleNode(Port *globalCountPort, Port *hashTablePort, GroupbyGraphParams& params)
{
	std::string fileName   = params.dirName + "\\groupby.sm_20.ptx";
	std::string kernelName = "reshuffleIntInt";
	CompiledKernel * pKernel = COMPILE_KERNEL(const_cast<char*>(fileName.c_str()), const_cast<char*>(kernelName.c_str()));

	int numInputPorts = 4; 
	int numOutputPorts = 3; 

	Port **pInputPorts = new Port*[numInputPorts];
	Port **pOutputPorts = new Port*[numOutputPorts];
	pInputPorts[0]  = PTask::Runtime::CreatePort(INPUT_PORT, params.inputTemplate, params.uiUidCounter++, "in", 0); // , 0);
	pInputPorts[1]  = PTask::Runtime::CreatePort(STICKY_PORT, params.intTemplate, params.uiUidCounter++, "n", 1);
	pInputPorts[2]  = PTask::Runtime::CreatePort(INPUT_PORT, params.hashTableTemplate, params.uiUidCounter++, "hashTable(in)", 3, 1);
	pInputPorts[3]  = PTask::Runtime::CreatePort(INPUT_PORT, params.ptrTemplate, params.uiUidCounter++, "globalCount(in)", 4, 2);

	pOutputPorts[0] = PTask::Runtime::CreatePort(OUTPUT_PORT, params.inputTemplate, params.uiUidCounter++, "out", 2);
	pOutputPorts[1] = PTask::Runtime::CreatePort(OUTPUT_PORT, params.hashTableTemplate, params.uiUidCounter++, "hashTable(out)", 3);
	pOutputPorts[2] = PTask::Runtime::CreatePort(OUTPUT_PORT, params.ptrTemplate, params.uiUidCounter++, "globalCount(out)", 4);
    
    // the hash-table is not marshallable because it is an
    // array of pointers to device-side malloc'ed items. 
    // consequently, any attempt to migrate that data will produce
    // garbage data, and we lack a facility to flatten the data
    // structure to make it possible to migrate it.
    pOutputPorts[1]->SetMarshallable(FALSE);
    pInputPorts[2]->SetMarshallable(FALSE);

	Task * pKernelTask = params.g->AddTask(pKernel, 
									  numInputPorts,
									  pInputPorts,
									  numOutputPorts,
									  pOutputPorts,
									  const_cast<char*>(kernelName.c_str()));
    PTask::Runtime::SetTaskAffinity(pKernelTask, params.accelerator, PTask::AFFINITYTYPE_MANDATORY);

	PTASKDIM3 grid, block;
	computeGridAndBlockSize(grid, block, params.inputLength);
	pKernelTask->SetBlockAndGridSize(grid, block);

	params.g->AddInternalChannel(globalCountPort, pInputPorts[3], "globalCount_shuffle");
	GraphInputChannel *inChannel = params.g->AddInputChannel(pInputPorts[0], "inChannel_shuffle");
	params.inputChannels.push_back(inChannel);
	GraphInputChannel *lenChannel = params.g->AddInputChannel(pInputPorts[1], "nChannel_shuffle");
	params.inputLengthChannels.push_back(lenChannel);
	//GraphInputChannel *hashTableChannel = params.g->AddInputChannel(pInputPorts[2], "hashTableChannel_keyCount");
	//params.hashtableChannels.push_back(hashTableChannel);
	params.g->AddInternalChannel(hashTablePort, pInputPorts[2], "hashTableChannel_shuffle");

	params.outputChannel = params.g->AddOutputChannel(pOutputPorts[0], "OutChannel_shuffle");
	constructCleanUpNodes(pOutputPorts[1], pOutputPorts[2], params);

	delete [] pInputPorts;
	delete [] pOutputPorts;
}

//singleBlockPrefixSumIntAdd(int **arr, int *n)
void constructPrefixSum(Port *globalCountPort, Port *keyCountPort, Port *hashTablePort, GroupbyGraphParams &params)
{
	std::string fileName   = params.dirName + "\\groupby.sm_20.ptx";
	std::string kernelName = "singleBlockPrefixSumIntAdd";
	CompiledKernel * pKernel = COMPILE_KERNEL(const_cast<char*>(fileName.c_str()), const_cast<char*>(kernelName.c_str()));

	int numInputPorts = 2; 
	int numOutputPorts = 1; 

	Port **pInputPorts = new Port*[numInputPorts];
	Port **pOutputPorts = new Port*[numOutputPorts];

	pInputPorts[0]  = PTask::Runtime::CreatePort(INPUT_PORT, params.ptrTemplate, params.uiUidCounter++, "globalCount(in)", 0, 0);
	pInputPorts[1]  = PTask::Runtime::CreatePort(INPUT_PORT, params.keycountTemplate, params.uiUidCounter++, "keyCount(in)", 1);
	pOutputPorts[0] = PTask::Runtime::CreatePort(OUTPUT_PORT, params.ptrTemplate, params.uiUidCounter++, "globalCount(out)", 0);

	Task * pKernelTask = params.g->AddTask(pKernel, 
									  numInputPorts,
									  pInputPorts,
									  numOutputPorts,
									  pOutputPorts,
									  const_cast<char*>(kernelName.c_str()));
    PTask::Runtime::SetTaskAffinity(pKernelTask, params.accelerator, PTask::AFFINITYTYPE_MANDATORY);

	PTASKDIM3 grid, block;
	computeGridAndBlockSize(grid, block, 512);
	pKernelTask->SetBlockAndGridSize(grid, block);


	params.g->AddInternalChannel(keyCountPort, pInputPorts[1], "keyCountChannel_prefixSum");
	params.g->AddInternalChannel(globalCountPort, pInputPorts[0], "globalCountChanel_prefixSum");

	//params.outputChannel = params.g->AddOutputChannel(pOutputPorts[0]);

	constructShuffleNode(pOutputPorts[0], hashTablePort, params);
	
	delete [] pInputPorts;
	delete [] pOutputPorts;
}

//keyCountIntInt(int *arr, int n, HashTableNode<int>** hashTable, int** globalCount)
void constructKeyCountKernel(Port *countArray, Port *keyCountPort, Port *hashTablePort, GroupbyGraphParams& params)
{
	std::string fileName   = params.dirName + "\\groupby.sm_20.ptx";
	std::string kernelName = "keyCountIntInt";
	CompiledKernel * pKernel = COMPILE_KERNEL(const_cast<char*>(fileName.c_str()), const_cast<char*>(kernelName.c_str()));

	int numInputPorts = 4; 
	int numOutputPorts = 1; 

	Port **pInputPorts = new Port*[numInputPorts];
	Port **pOutputPorts = new Port*[numOutputPorts];

	pInputPorts[0]  = PTask::Runtime::CreatePort(INPUT_PORT, params.inputTemplate, params.uiUidCounter++, "in", 0);
	pInputPorts[1]  = PTask::Runtime::CreatePort(STICKY_PORT, params.intTemplate, params.uiUidCounter++, "n", 1);
	pInputPorts[2]  = PTask::Runtime::CreatePort(INPUT_PORT, params.hashTableTemplate, params.uiUidCounter++, "hashTable", 2);
	pInputPorts[3]  = PTask::Runtime::CreatePort(INPUT_PORT, params.ptrTemplate, params.uiUidCounter++, "globalCount(in)", 3, 0);
    // the hash-table is not marshallable because it is an
    // array of pointers to device-side malloc'ed items. 
    // consequently, any attempt to migrate that data will produce
    // garbage data, and we lack a facility to flatten the data
    // structure to make it possible to migrate it.
    pInputPorts[2]->SetMarshallable(FALSE);

	pOutputPorts[0] = PTask::Runtime::CreatePort(OUTPUT_PORT, params.ptrTemplate, params.uiUidCounter++, "globalCount(out)", 3);

	Task * pKernelTask = params.g->AddTask(pKernel, 
									  numInputPorts,
									  pInputPorts,
									  numOutputPorts,
									  pOutputPorts,
									  const_cast<char*>(kernelName.c_str()));
    PTask::Runtime::SetTaskAffinity(pKernelTask, params.accelerator, PTask::AFFINITYTYPE_MANDATORY);

	PTASKDIM3 grid, block;
	computeGridAndBlockSize(grid, block, params.inputLength);
	pKernelTask->SetBlockAndGridSize(grid, block);

	params.g->AddInternalChannel(countArray, pInputPorts[3], "globalCountChannel");
	GraphInputChannel *inChannel = params.g->AddInputChannel(pInputPorts[0], "inChannel_keyCount");
	params.inputChannels.push_back(inChannel);
	GraphInputChannel *lenChannel = params.g->AddInputChannel(pInputPorts[1], "nChannel_keyCount");
	params.inputLengthChannels.push_back(lenChannel);
	//GraphInputChannel *hashTableChannel = params.g->AddInputChannel(pInputPorts[2], "hashTableChannel_keyCount");
	//params.hashtableChannels.push_back(hashTableChannel);
	params.g->AddInternalChannel(hashTablePort, pInputPorts[2], "hashTableChannel_keyCount");

	constructPrefixSum(pOutputPorts[0], keyCountPort, hashTablePort, params);

	delete [] pInputPorts;
	delete [] pOutputPorts;
}

void constructAllocKernel(Port *keyCountPort, Port *hashTablePort, GroupbyGraphParams& params)
{
	std::string fileName   = params.dirName + "\\groupby.sm_20.ptx";
	std::string kernelName = "allocateCountArray";
	CompiledKernel * pKernel = COMPILE_KERNEL(const_cast<char*>(fileName.c_str()), const_cast<char*>(kernelName.c_str()));

	int numInputPorts = 1; 
	int numOutputPorts = 1; 

	Port **pInputPorts = new Port*[numInputPorts];
	Port **pOutputPorts = new Port*[numOutputPorts];

	pInputPorts[0]  = PTask::Runtime::CreatePort(INPUT_PORT, params.keycountTemplate, params.uiUidCounter++, "keyCount(in)", 1);
	pOutputPorts[0] = PTask::Runtime::CreatePort(OUTPUT_PORT, params.ptrTemplate, params.uiUidCounter++, "countArray(out)", 0);

	Task * pKernelTask = params.g->AddTask(pKernel, 
									  numInputPorts,
									  pInputPorts,
									  numOutputPorts,
									  pOutputPorts,
									  const_cast<char*>(kernelName.c_str()));
    PTask::Runtime::SetTaskAffinity(pKernelTask, params.accelerator, PTask::AFFINITYTYPE_MANDATORY);

	PTASKDIM3 grid, block;
	computeGridAndBlockSize(grid, block, 512);
	pKernelTask->SetBlockAndGridSize(grid, block);
	
	params.g->AddInternalChannel(keyCountPort, pInputPorts[0], "KeyCountChannel");

	constructKeyCountKernel(pOutputPorts[0], keyCountPort, hashTablePort, params);
	
	delete [] pInputPorts;
	delete [] pOutputPorts;
}

void contructUniqueKeyCountNode(GroupbyGraphParams& params)
{
	std::string fileName   = params.dirName + "\\groupby.sm_20.ptx";
	std::string kernelName = "countUniqueKeysIntInt";
	CompiledKernel * pKernel = COMPILE_KERNEL(const_cast<char*>(fileName.c_str()), const_cast<char*>(kernelName.c_str()));

	int numInputPorts = 4; //input array, len, hashtable, numKeys
	int numOutputPorts = 2; //numKeys, hashtable

	Port **pInputPorts = new Port*[numInputPorts];
	Port **pOutputPorts = new Port*[numOutputPorts];

	//pInputPorts[0] = PTask::Runtime::CreatePort(INPUT_PORT, params.inputTemplate, params.uiUidCounter++, "in", 0);
	//pInputPorts[1] = PTask::Runtime::CreatePort(STICKY_PORT, params.intTemplate, params.uiUidCounter++, "n", 1);
	//pInputPorts[2] = PTask::Runtime::CreatePort(INPUT_PORT, params.hashTableTemplate, params.uiUidCounter++, "hashTable", 2);
	//pInputPorts[3] = PTask::Runtime::CreatePort(INPUT_PORT, params.keycountTemplate, params.uiUidCounter++, "keyCount(in)", 3, 0);
	//
	//pOutputPorts[0] = PTask::Runtime::CreatePort(OUTPUT_PORT, params.keycountTemplate, params.uiUidCounter++, "keyCount(out)", 3);

	pInputPorts[0] = PTask::Runtime::CreatePort(INPUT_PORT, params.inputTemplate, params.uiUidCounter++, "in", 1);
	pInputPorts[1] = PTask::Runtime::CreatePort(STICKY_PORT, params.intTemplate, params.uiUidCounter++, "n", 3);
	pInputPorts[2] = PTask::Runtime::CreatePort(INPUT_PORT, params.hashTableTemplate, params.uiUidCounter++, "hashTable(in)", 2, 1);
	pInputPorts[3] = PTask::Runtime::CreatePort(INPUT_PORT, params.keycountTemplate, params.uiUidCounter++, "keyCount(in)", 0, 0);
	
	pOutputPorts[0] = PTask::Runtime::CreatePort(OUTPUT_PORT, params.keycountTemplate, params.uiUidCounter++, "keyCount(out)", 0);
	pOutputPorts[1] = PTask::Runtime::CreatePort(OUTPUT_PORT, params.hashTableTemplate, params.uiUidCounter++, "hashTable(out)", 2);
    // the hash-table is not marshallable because it is an
    // array of pointers to device-side malloc'ed items. 
    // consequently, any attempt to migrate that data will produce
    // garbage data, and we lack a facility to flatten the data
    // structure to make it possible to migrate it.
    pOutputPorts[1]->SetMarshallable(FALSE);
    pInputPorts[2]->SetMarshallable(FALSE);

	Task * pKernelTask = params.g->AddTask(pKernel, 
									  numInputPorts,
									  pInputPorts,
									  numOutputPorts,
									  pOutputPorts,
									  const_cast<char*>(kernelName.c_str()));
    PTask::Runtime::SetTaskAffinity(pKernelTask, params.accelerator, PTask::AFFINITYTYPE_MANDATORY);

	PTASKDIM3 grid, block;
	computeGridAndBlockSize(grid, block, params.inputLength);
	pKernelTask->SetBlockAndGridSize(grid, block);

	params.inputChannels.push_back(params.g->AddInputChannel(pInputPorts[0], "countUniqueKeysIn"));
	params.inputLengthChannels.push_back(params.g->AddInputChannel(pInputPorts[1], "countUniqueKeysn"));
	params.hashtableChannels.push_back(params.g->AddInputChannel(pInputPorts[2], "countUniqueKeysHashTable"));
	params.keyCountInChannel = params.g->AddInputChannel(pInputPorts[3], "countUniqueKeysKeyCountIn");
	
	constructAllocKernel(pOutputPorts[0], pOutputPorts[1], params);

	delete [] pInputPorts;
	delete [] pOutputPorts;
}

void constructGraph(GroupbyGraphParams& params)
{
	contructUniqueKeyCountNode(params);
}

void testResults(int *in, int *out, int len)
{
	std::set<int> inputSet;
	for (int i=0; i<len ; ++i)
		inputSet.insert(in[i]);

	std::set<int> intSet;
	int failures = 0;
	for(int i=0; i<len-1 ; ++i)
	{
		if(intSet.find(out[i]) != intSet.end())
			++failures; 
		if(out[i] != out[i+1])
			intSet.insert(out[i]);
	}
	if (intSet.find(out[len-1]) != intSet.end())
	{
		++failures;
	}
	intSet.insert(out[len-1]);
	std::cout << "Failures : " << failures << std::endl;
	if (failures == 0  && inputSet==intSet)
		std::cout << "groupby test succeeded\n";
	else
		std::cout << "groupby test failed\n";
}

void runGraph(GroupbyGraphParams &params)
{
    printf("remember--you should only use PTask::Runtime::SetUseCUDA for this workload!\n");
	CSimpleVector<int> *input = new CSimpleVector<int>(params.inputLength);
	CSimpleVector<int> *output = new CSimpleVector<int>(params.inputLength);
	CSimpleVector<HashTableNode*> *hashTableCPU = new CSimpleVector<HashTableNode*>(HASH_TABLE_SIZE);
	for(int i=0; i<input->N() ; ++i)
		input->v(i) = ((int)(randfloat() * params.numKeys));// * 10 + 1;
		
	for(int i=0; i<hashTableCPU->N(); ++i)
		hashTableCPU->v(i) = NULL;
	int keyCountCPU[10] = {0, };
	
	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();
	for(int iters=0; iters<params.numIterations ; ++iters)
	{
		Datablock *inputBlock = PTask::Runtime::AllocateDatablock(params.inputTemplate, input->cells(), input->arraysize(), params.inputChannels[0]);
		Datablock *hashTableBlock = PTask::Runtime::AllocateDatablock(params.hashTableTemplate, hashTableCPU->cells(), hashTableCPU->arraysize(), params.hashtableChannels[0]);
		Datablock *keyCountBlock = PTask::Runtime::AllocateDatablock(params.keycountTemplate, keyCountCPU, sizeof(keyCountCPU), params.keyCountInChannel);
		Datablock *inputLenBlock = PTask::Runtime::AllocateDatablock(params.intTemplate, &params.inputLength, sizeof(params.inputLength), params.inputLengthChannels[0]);

        if(!inputBlock || !hashTableBlock || !keyCountBlock || !inputLenBlock) {
            printf("XXX allocation failure in graphgroupby::runGraph\n");
            exit(1);
        }

		for(size_t i=0; i<params.inputChannels.size() ; ++i)
			params.inputChannels[i]->Push(inputBlock);

		for(size_t i=0; i<params.hashtableChannels.size() ; ++i)
			params.hashtableChannels[i]->Push(hashTableBlock);

		for(size_t i=0; i<params.inputLengthChannels.size() ; ++i)
			params.inputLengthChannels[i]->Push(inputLenBlock);

		params.keyCountInChannel->Push(keyCountBlock);
		inputBlock->Release();
		hashTableBlock->Release();
		inputLenBlock->Release();
		keyCountBlock->Release();
// 	}

//	for(int iters=0; iters<params.numIterations ; ++iters)
//	{
		Datablock * pResultBlock = params.outputChannel->Pull();
		pResultBlock->Lock();
		int* psrc = (int*) pResultBlock->GetDataPointer(FALSE);
		int* pdst = output->cells();
		int size = sizeof(int) * params.inputLength;
		memcpy(pdst, psrc, size);
		pResultBlock->Unlock();
		pResultBlock->Release();
	}
	
	double dGPUTime = pTimer->elapsed(false);

	std::cout << "PTask execution time : " << dGPUTime << std::endl;
	delete pTimer;
	
	testResults(input->cells(), output->cells(), params.inputLength);

	delete input;
	delete output;
	delete hashTableCPU;
}

UINT find_preferred_accelerator_id(
    VOID
    )
{
    BOOL bPreferredIdSet = FALSE;
    UINT uiPreferredId = 0;
    UINT uiIndex = 0;
    ACCELERATOR_DESCRIPTOR * pDescriptor;
    ACCELERATOR_DESCRIPTOR * pPreferredDescriptor;
    std::vector<ACCELERATOR_DESCRIPTOR*> descriptors;
    while(PTASK_ERR_NOT_FOUND != PTask::Runtime::EnumerateAccelerators(ACCELERATOR_CLASS_CUDA, uiIndex, &pDescriptor)) {
        descriptors.push_back(pDescriptor);
        uiIndex++;
    }
    std::vector<ACCELERATOR_DESCRIPTOR*>::iterator vi;
    for(vi=descriptors.begin(); vi!=descriptors.end(); vi++) {
        if(!bPreferredIdSet && (*vi)->bEnabled) {
            pPreferredDescriptor = (*vi);
            uiPreferredId = pPreferredDescriptor->uiAcceleratorId;
            bPreferredIdSet = TRUE;
        } else {
            pDescriptor = (*vi);
            if(!pDescriptor->bEnabled)
                continue;
            if(pDescriptor->nCoreCount > pPreferredDescriptor->nCoreCount) {
                uiPreferredId = pPreferredDescriptor->uiAcceleratorId;
                pPreferredDescriptor = (*vi);
                continue;
            }
            if(pDescriptor->bSupportsConcurrentKernels &&  !pPreferredDescriptor->bSupportsConcurrentKernels) {
                uiPreferredId = pPreferredDescriptor->uiAcceleratorId;
                pPreferredDescriptor = (*vi);
                continue;
            }
            if(pDescriptor->nMemorySize > pPreferredDescriptor->nMemorySize) {
                uiPreferredId = pPreferredDescriptor->uiAcceleratorId;
                pPreferredDescriptor = (*vi);
                continue;
            }
            if(pDescriptor->nClockRate > pPreferredDescriptor->nClockRate) {
                uiPreferredId = pPreferredDescriptor->uiAcceleratorId;
                pPreferredDescriptor = (*vi);
                continue;
            }
            if(pDescriptor->nPlatformIndex < pPreferredDescriptor->nPlatformIndex) {
                uiPreferredId = pPreferredDescriptor->uiAcceleratorId;
                pPreferredDescriptor = (*vi);
                continue;
            }
        }
    }
    for(vi=descriptors.begin(); vi!=descriptors.end(); vi++) {
        free(*vi);
    }
    return uiPreferredId;
}


int run_graph_groupby_task(char *szdir, char *szshader, int len, int numKeys, int numIterations)
{
    CONFIGUREPTASKU(ICBlockPoolSize, 2);
    CONFIGUREPTASKU(UseCUDA,TRUE);
    CONFIGUREPTASKU(UseDirectX,FALSE);
    CONFIGUREPTASKU(UseOpenCL,FALSE);
	PTask::Runtime::Initialize();
    CheckPlatformSupport("quack.ptx", szshader);

	int seed = 0x4fd478;
	srand( seed );
	GroupbyGraphParams params(len, numKeys, numIterations, szdir);

    UINT uiPreferredAcceleratorId = 
    params.accelerator = ::find_preferred_accelerator_id();
	params.g = new Graph();

	constructGraph(params);
	params.g->WriteDOTFile("c:\\temp\\groupby.dot", TRUE);
	
    if(g_bSingleThreaded)
        printf("single-threaded!\n");
	params.g->Run(g_bSingleThreaded);
	runGraph(params);
	// runGraph(params);
	params.g->Stop();

	//Do all the graph destruction stuff
	params.g->Teardown();
	Graph::DestroyGraph(params.g);
	PTask::Runtime::Terminate();
	return 0;
}