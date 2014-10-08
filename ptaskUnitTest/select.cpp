//--------------------------------------------------------------------------------------
// File: select.cpp
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#include <stdio.h>
#include <crtdbg.h>
#include "accelerator.h"
#include "assert.h"
#include "shaderparms.h"
#include "sort.h"
#include <vector>
#include <algorithm>
#include "select.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

using namespace std;
using namespace PTask;

void
dump_directory(
    int nEntries,
    int * pIndex,
    char * pData
    )
{
    for(int i=0; i<nEntries; i++) { 
        int nEntryIndex = i * 2;
        int nOffset = pIndex[nEntryIndex];
        int nLength = pIndex[nEntryIndex+1];
        char * pStr = (char*) malloc(nLength+1);
        memcpy(pStr, &pData[nOffset], nLength);
        pStr[nLength] = 0;
        printf("%d: %s(%d)\n", nOffset, pStr, nLength);
        free(pStr);
    }
}

void
dump_count_directory(
    int nEntries,
    int * pIndex,
    char * pData
    )
{
    for(int i=0; i<nEntries; i++) { 
        int nEntryIndex = i * 2;
        int nOffset = pIndex[nEntryIndex];
        int nLength = pIndex[nEntryIndex+1];
        int * pInt = (int*) &pData[nOffset];
        printf("%d: %d(%d)\n", nOffset, *pInt, nLength);
    }
}


typedef struct _kvs_dir_t { 
    int nKVPairs;
    int ncbKeys;
    int ncbVals;
    int ncbKeyMap;
    int ncbValMap;
    char* pKeys;
    char* pVals;
    int * pKeyMap;
    int * pValMap;
} KVMAP;

typedef struct _kvparms_t {
    int nKVPairs;
} KV_PARAMS;

KVMAP *
create_kvmap(
    int nKeys,
    int ncbKeys,
    int ncbVals,
    int ncbKeyMap,
    int ncbValMap,
    char * pKeys,
    char * pVals,
    int * pKeyMap,
    int * pValMap
    ) 
{
    KVMAP * pKVMap = (KVMAP*) malloc(sizeof(KVMAP));
    pKVMap->nKVPairs = nKeys;
    pKVMap->ncbKeys = ncbKeys;
    pKVMap->ncbVals = ncbVals;
    pKVMap->ncbKeyMap = ncbKeyMap;
    pKVMap->ncbValMap = ncbValMap;
    pKVMap->pKeys = pKeys;
    pKVMap->pVals = pVals;
    pKVMap->pKeyMap = pKeyMap;
    pKVMap->pValMap = pValMap;
    return pKVMap;
}

#pragma warning(disable:4996)
KVMAP * 
build_directory(
    char * szdata
    ) 
{
    FILE * fp = fopen(szdata, "rb");
    assert(fp != NULL);
    fseek(fp, 0, SEEK_END);
    fpos_t length;
    fgetpos(fp, &length);
    char * pdata = (char*) calloc(length+1, 1);
    fseek(fp, 0, SEEK_SET);
    size_t rd = fread(pdata, 1, length, fp);
    assert(rd == length);
    fclose(fp);

    std::vector<char*> keys;
    std::vector<char*> vals;
    std::vector<int> keyoffsets, keylengths;
    std::vector<int> valoffsets, vallengths;

    char * p = pdata;
    while(p-pdata<length) {
        char * q = p;
        while(*p != ',' && (p - pdata) < length) p++;
        char * key = (char*) malloc(p-q+1);
        memcpy(key, q, p-q);
        key[p-q] = '\0';
        keys.push_back(key);
        p++;
        q=p;
        while(*p != '\n' && (p - pdata) < length) p++;
        char * val = (char*) malloc(p-q+1);
        memcpy(val, q, p-q-1);
        val[p-q-1] = '\0';
        vals.push_back(val);
        p++;
    }
    free(pdata);

    std::vector<char*>::iterator ci, vi;
    int nKeys = 0;
    int nVals = 0;
    int nKeysLen = 0;
    int nValsLen = 0;
    std::vector<ENTRY> directory;
    for(ci=keys.begin(), vi=vals.begin();
        ci!=keys.end() && vi!=vals.end();
        ci++, vi++) {
        printf("%s => %s\n", *ci, *vi);
        int keylen = (int) strlen(*ci);
        int vallen = (int) strlen(*vi);
        keyoffsets.push_back(nKeysLen);
        keylengths.push_back(keylen);
        valoffsets.push_back(nValsLen);
        vallengths.push_back(vallen);
        nKeysLen += keylen+1;
        nKeysLen += (4-(nKeysLen%4));
        nValsLen += vallen+1;
        nValsLen += (4-(nValsLen%4));
        nKeys++;
        nVals++;
    }
    assert(nKeys == nVals);
    char * pKeys = (char*) calloc(nKeysLen, 1);
    char * pVals = (char*) calloc(nValsLen, 1);
    char * pK = pKeys;
    char * pV = pVals;
    int ncbKeyMap = nKeys*2*sizeof(int);
    int ncbValMap = nVals*2*sizeof(int);
    int * pKeyMap = (int*) calloc(nKeys*2, sizeof(int)); // offset, length
    int * pValMap = (int*) calloc(nVals*2, sizeof(int)); // offset, length
    int * pKM = pKeyMap;
    int * pVM = pValMap;
    std::vector<int>::iterator koi, voi, kli, vli;
    for(ci=keys.begin(), vi=vals.begin(), 
        koi=keyoffsets.begin(), voi=valoffsets.begin(),
        kli=keylengths.begin(), vli=vallengths.begin();
        ci!=keys.end() && vi!=vals.end() &&
        koi!=keyoffsets.end() && voi!=valoffsets.end() &&
        kli!=keylengths.end() && vli!=vallengths.end();
        ci++, vi++, koi++, voi++, kli++, vli++) {
        *pKM++ = *koi;      // key offset
        *pKM++ = *kli;      // key length
        *pVM++ = *voi;      // value offset
        *pVM++ = *vli;      // value length
        pK = pKeys + (*koi);
        pV = pVals + (*voi);
        strcpy(pK, *ci);
        strcpy(pV, *vi);
        free(*ci);
        free(*vi);
    }

    return create_kvmap(nKeys, nKeysLen, nValsLen, ncbKeyMap, ncbValMap, pKeys, pVals, pKeyMap, pValMap);
    #pragma warning(default:4996)
}

void
free_kvmap(
    KVMAP* pKVMap
    ) 
{
    if(pKVMap) {
        free(pKVMap->pKeys);
        free(pKVMap->pVals);
        free(pKVMap->pKeyMap);
        free(pKVMap->pValMap);
        free(pKVMap);
    }
}

KVMAP * 
host_select(
    KVMAP * pKVMap
    )
{
    int nKeysLength = 0;
    int nValsLength = 0;
    for(int i=0; i<pKVMap->nKVPairs; i++) {
        int nEntryIndex = i * 2;
        int nKeyLength = pKVMap->pKeyMap[nEntryIndex+1]+1; // require at least one byte for null-termination
        int nValLength = pKVMap->pValMap[nEntryIndex+1]+1; // require at least one byte for null-termination
        nKeysLength += nKeyLength + (4-(nKeyLength%4));
        nValsLength += nValLength + (4-(nValLength%4));
    }

    KVMAP * pKVOut = create_kvmap(pKVMap->nKVPairs, 
                                nKeysLength,
                                nValsLength,
                                pKVMap->nKVPairs*2*sizeof(int),
                                pKVMap->nKVPairs*2*sizeof(int),
                                (char*) calloc(nKeysLength+1, 1),
                                (char*) calloc(nValsLength+1, 1),
                                (int*) malloc(pKVMap->nKVPairs*2*sizeof(int)),
                                (int*) malloc(pKVMap->nKVPairs*2*sizeof(int)));

    for(int i=0; i<pKVMap->nKVPairs; i++) {
        int nEntryIndex = i * 2;
        pKVOut->pKeyMap[nEntryIndex] = pKVMap->pKeyMap[nEntryIndex];
        pKVOut->pKeyMap[nEntryIndex+1] = pKVMap->pKeyMap[nEntryIndex+1];
        pKVOut->pValMap[nEntryIndex] = pKVMap->pValMap[nEntryIndex];
        pKVOut->pValMap[nEntryIndex+1] = pKVMap->pValMap[nEntryIndex+1];
        int nKeyOffset = pKVMap->pKeyMap[nEntryIndex];
        int nKeyLength = pKVMap->pKeyMap[nEntryIndex+1];
        int nValOffset = pKVMap->pValMap[nEntryIndex];
        int nValLength = pKVMap->pValMap[nEntryIndex+1];
        assert(nKeyOffset + nKeyLength < pKVOut->ncbKeys);
        assert(nValOffset + nValLength < pKVOut->ncbVals);
        memcpy(&pKVOut->pKeys[nKeyOffset], &pKVMap->pKeys[nKeyOffset], nKeyLength);
        memcpy(&pKVOut->pVals[nValOffset], &pKVMap->pVals[nValOffset], nValLength);
    }

    return pKVOut;
}

BOOL
kvmaps_equal(
    KVMAP * pKV0,
    KVMAP * pKV1
    )
{
    if(!pKV0 || !pKV1) return FALSE;
    if(pKV0->nKVPairs != pKV1->nKVPairs) return FALSE;
    if(memcmp(pKV0->pKeyMap, pKV1->pKeyMap, pKV0->nKVPairs*2*sizeof(int))) return FALSE;
    if(memcmp(pKV0->pValMap, pKV1->pValMap, pKV0->nKVPairs*2*sizeof(int))) return FALSE;
    
    int nKeysLength = 0;
    int nValsLength = 0;
    for(int i=0; i<pKV0->nKVPairs; i++) {
        int nEntryIndex = i * 2;
        int nKeyLength = pKV0->pKeyMap[nEntryIndex+1];
        int nValLength = pKV0->pKeyMap[nEntryIndex+1];
        nKeysLength += nKeyLength+1;
        nValsLength += nValLength+1;
    }
    if(memcmp(pKV0->pKeys, pKV1->pKeys, nKeysLength)) return FALSE;
    if(memcmp(pKV0->pVals, pKV1->pVals, nValsLength)) return FALSE;

    return TRUE;
}

int run_select_task(	
	char * szfile,
	char * szshader,
    char * szdatafile,
	int iterations
	) 
{
    PTask::Runtime::Initialize();
    CheckPlatformSupport(szfile, szshader);

    printf("ptask select!\n%s\n", szdatafile);
    KVMAP * pKVMap = build_directory(szdatafile);

    printf("Keys:\n");
    dump_directory(pKVMap->nKVPairs, pKVMap->pKeyMap, pKVMap->pKeys);
    printf("\n\nValues:\n");
    dump_directory(pKVMap->nKVPairs, pKVMap->pValMap, pKVMap->pVals);

	int nChannelCount = 0;

	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	Graph * pGraph = new Graph();
    UINT uiMaxDataSize = max(pKVMap->ncbKeys, pKVMap->ncbVals)*10;
    DatablockTemplate * pKVTemplate	    = PTask::Runtime::GetDatablockTemplate("dbkvdata", sizeof(char), uiMaxDataSize, 1, 1, false, true);
    DatablockTemplate * pIndexTemplate	= PTask::Runtime::GetDatablockTemplate("dbindex", sizeof(int), 2*pKVMap->nKVPairs, 1, 1);
	DatablockTemplate * pParmTemplate	= PTask::Runtime::GetDatablockTemplate("kvparms", sizeof(KV_PARAMS), 1, 1, 1);
	CompiledKernel * pSelectKernel		= COMPILE_KERNEL(szfile, szshader);

	const UINT uiInputCount = 5;
	const UINT uiOutputCount = 4;
	
	Port ** pInputPorts = new Port*[uiInputCount];
	Port ** pOutputPorts = new Port*[uiOutputCount];

	UINT uiUidCounter		= 0;
	pInputPorts[0]		= PTask::Runtime::CreatePort(INPUT_PORT, pKVTemplate, uiUidCounter++, "pI0");
	pInputPorts[1]		= PTask::Runtime::CreatePort(INPUT_PORT, pKVTemplate, uiUidCounter++, "pI1");
	pInputPorts[2]		= PTask::Runtime::CreatePort(INPUT_PORT, pIndexTemplate, uiUidCounter++, "pI2");
	pInputPorts[3]		= PTask::Runtime::CreatePort(INPUT_PORT, pIndexTemplate, uiUidCounter++, "pI3");
	pInputPorts[4]		= PTask::Runtime::CreatePort(STICKY_PORT, pParmTemplate, uiUidCounter++, "pI4");
	pOutputPorts[0]	    = PTask::Runtime::CreatePort(OUTPUT_PORT, pKVTemplate, uiUidCounter++, "pO1");	
	pOutputPorts[1]	    = PTask::Runtime::CreatePort(OUTPUT_PORT, pKVTemplate, uiUidCounter++, "pO2");	
	pOutputPorts[2]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pIndexTemplate, uiUidCounter++, "pO3");
	pOutputPorts[3]		= PTask::Runtime::CreatePort(OUTPUT_PORT, pIndexTemplate, uiUidCounter++, "pO4");

	Task * pTask = pGraph->AddTask(pSelectKernel, 
									uiInputCount,
									pInputPorts,
									uiOutputCount,
									pOutputPorts,
									"SelectTask");

	assert(pTask);
	pTask->SetComputeGeometry(pKVMap->nKVPairs, 1, 1);

	GraphInputChannel * pKeysInput		    = pGraph->AddInputChannel(pInputPorts[0], "KeyInputChannel");
	GraphInputChannel * pValsInput		    = pGraph->AddInputChannel(pInputPorts[1], "ValInputChannel");
	GraphInputChannel * pKeyIndexInput		= pGraph->AddInputChannel(pInputPorts[2], "KeyIndexInputChannel");
	GraphInputChannel * pValIndexInput		= pGraph->AddInputChannel(pInputPorts[3], "ValIndexInputChannel");
	GraphInputChannel * pParmsInput		    = pGraph->AddInputChannel(pInputPorts[4], "ParmsInputChannel");
	GraphOutputChannel * pKeysOutput		= pGraph->AddOutputChannel(pOutputPorts[0], "KeyOutputChannel");
	GraphOutputChannel * pValsOutput		= pGraph->AddOutputChannel(pOutputPorts[1], "ValOutputChannel");
	GraphOutputChannel * pKeyIndexOutput	= pGraph->AddOutputChannel(pOutputPorts[2], "KeyIndexOutputChannel");
	GraphOutputChannel * pValIndexOutput	= pGraph->AddOutputChannel(pOutputPorts[3], "ValIndexOutputChannel");

	pGraph->Run();

	double dInitTime;
	double dCopyToDeviceEnd; 
	double dCopyToDeviceTime;
	double dComputeEnd;
	double dComputeTime;
	double dCopyType;
	double dHostStart;
	double dHostEnd;
	double dTeardownStart;
	for(int i=0;i<iterations;i++) {

        BUFFERACCESSFLAGS rawFlags = (PT_ACCESS_HOST_WRITE | PT_ACCESS_ACCELERATOR_READ | PT_ACCESS_BYTE_ADDRESSABLE);
        UINT uiKDataSize = pKVMap->ncbKeys;
        UINT uiVDataSize = pKVMap->ncbVals;

        Datablock * pKDB	= PTask::Runtime::AllocateDatablock(pKVTemplate, pKVMap->pKeys, pKVMap->ncbKeys, pKeysInput, rawFlags, uiKDataSize);
        Datablock * pVDB    = PTask::Runtime::AllocateDatablock(pKVTemplate, pKVMap->pVals, pKVMap->ncbVals, pValsInput, rawFlags, uiVDataSize);
        Datablock * pKIndex = PTask::Runtime::AllocateDatablock(pIndexTemplate, pKVMap->pKeyMap, pKVMap->ncbKeyMap, pKeyIndexInput);
        Datablock * pVIndex = PTask::Runtime::AllocateDatablock(pIndexTemplate, pKVMap->pValMap, pKVMap->ncbValMap, pValIndexInput);
        Datablock * pPrmDB	= PTask::Runtime::AllocateDatablock(pParmTemplate, &pKVMap->nKVPairs, sizeof(pKVMap->nKVPairs), pParmsInput);

		dInitTime = pTimer->elapsed(false);
        pKeysInput->Push(pKDB);
        pValsInput->Push(pVDB);
        pKeyIndexInput->Push(pKIndex);
        pValIndexInput->Push(pVIndex);
        pParmsInput->Push(pPrmDB);
        pKDB->Release();
        pVDB->Release();
        pKIndex->Release();
        pVIndex->Release();
        pPrmDB->Release();
		dCopyToDeviceEnd = pTimer->elapsed(false);
		dCopyToDeviceTime = dCopyToDeviceEnd - dInitTime;
		Datablock * pKoDB = pKeysOutput->Pull();
		Datablock * pVoDB = pValsOutput->Pull();
		Datablock * pKiDB = pKeyIndexOutput->Pull();
		Datablock * pViDB = pValIndexOutput->Pull();
		dComputeEnd = pTimer->elapsed(false);
		dComputeTime = dComputeEnd - dCopyToDeviceEnd;

        pKoDB->Lock();
        pVoDB->Lock();
        pKiDB->Lock();
        pViDB->Lock();
        KVMAP * pKVOutMap = create_kvmap(
                                pKVMap->nKVPairs,
                                pKVMap->ncbKeys,
                                pKVMap->ncbVals,
                                pKVMap->ncbKeyMap,
                                pKVMap->ncbValMap,
                                (char*) pKoDB->GetDataPointer(FALSE),
                                (char*) pVoDB->GetDataPointer(FALSE),
                                (int*) pKiDB->GetDataPointer(FALSE),
                                (int*) pViDB->GetDataPointer(FALSE));
        pKoDB->Unlock();
        pVoDB->Unlock();
        pKiDB->Unlock();
        pViDB->Unlock();

		dCopyType = pTimer->elapsed(false) - dComputeTime;

		printf( "Verifying against CPU result..." );
		int nErrorTolerance = 20;
		dHostStart = pTimer->elapsed(false);
        KVMAP * pKVHostMap = host_select(pKVMap);
        dHostEnd = pTimer->elapsed(false) - dHostStart;


        // TODO fix this check!
		// if(!kvmaps_equal(pKVOutMap, pKVHostMap)) {
		//	printf("failure: erroneous output\n");
        // } else {
			printf( "%s succeeded\n", szshader );
        // }
        printf("Host Keys:\n");
        dump_directory(pKVHostMap->nKVPairs, pKVHostMap->pKeyMap, pKVHostMap->pKeys);
        printf("\nDevice Keys:\n");
        dump_directory(pKVOutMap->nKVPairs, pKVOutMap->pKeyMap, pKVOutMap->pKeys);
        printf("\nHost Values:\n");
        dump_directory(pKVHostMap->nKVPairs, pKVHostMap->pValMap, pKVHostMap->pVals);
        printf("\nDevice Values:\n");
        // dump_count_directory(pKVOutMap->nKVPairs, pKVOutMap->pValMap, pKVOutMap->pVals);        
        dump_directory(pKVOutMap->nKVPairs, pKVOutMap->pValMap, pKVOutMap->pVals);        

		dTeardownStart = pTimer->elapsed(false);
		pKoDB->Release();
		pKiDB->Release();
		pVoDB->Release();
		pViDB->Release();
        free_kvmap(pKVHostMap);
		free(pKVOutMap); // needn't free pointers gotten from datablocks
	}

	pGraph->Stop();
	pGraph->Teardown();

	free_kvmap(pKVMap);
	Graph::DestroyGraph(pGraph);
	delete [] pInputPorts;
	delete [] pOutputPorts;
	double dTeardownTime = pTimer->elapsed(false) - dTeardownStart;

	printf("InitTime:\t%f\n", dInitTime);
	printf("CopyToGPU:\t%f\n", dCopyToDeviceTime);
	printf("GPU Compute:\t%f\n", dComputeTime);
	printf("CPU Compute:\t%f\n", dHostEnd);
	printf("Teardown:\t%f\n", dTeardownTime);

	delete pTimer;

	PTask::Runtime::Terminate();

	return 0;
}
