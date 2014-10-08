//--------------------------------------------------------------------------------------
// File: dxcodecache.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include <stdio.h>
#include <crtdbg.h>
#include "ptdxhdr.h"
#include <assert.h>
using namespace std;
#include "dxcodecache.h"

namespace PTask {

    DXCodeCache::DXCodeCache() {}

    DXCodeCache::~DXCodeCache() {
        map<string, ID3D11ComputeShader*, ltstr>::iterator mi;
        for(mi = m_cache.begin(); mi!=m_cache.end(); mi++) {
            ID3D11ComputeShader* pCS = mi->second;
            if(pCS) {
                LONG refCount = (LONG) pCS->Release();
                while(refCount > 0)
                    refCount = pCS->Release();
            }
        }
    }

    ID3D11ComputeShader*
        DXCodeCache::CacheGet(
        char * szFile, 
        char * szFunc
        ) 
    {
        ID3D11ComputeShader * pCS = NULL;
        std::string key = szFile;
        key += "+";
        key += szFunc;
        std::map<string, ID3D11ComputeShader*, ltstr>::iterator mi = m_cache.find(key);
        if(mi != m_cache.end())
            pCS = mi->second;
        if(pCS) {
            ULONG refCount = pCS->AddRef();
            refCount = refCount;
        }
        return pCS;
    }

    void
        DXCodeCache::CachePut(
        char * szFile, 
        char * szFunc, 
        ID3D11ComputeShader* p
        ) 
    {
        string key = szFile;
        key += "+";
        key += szFunc;
        m_cache[key] = p;
        p->AddRef();
    }

};
