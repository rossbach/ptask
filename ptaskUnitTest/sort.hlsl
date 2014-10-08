//--------------------------------------------------------------------------------------
// File: sort.hlsl
// bitonic sort based on compute shader sort sample
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#define SORT_BLOCK_SIZE 512
#define TRANSPOSE_SIZE 16

cbuffer CB : register( b0 )
{
    unsigned int g_iLevel;
    unsigned int g_iLevelMask;
    unsigned int g_iWidth;
    unsigned int g_iHeight;
};

StructuredBuffer<unsigned int> g_vData : register( t0 );
RWStructuredBuffer<unsigned int> g_vSorted : register( u0 );
groupshared unsigned int g_vShared[SORT_BLOCK_SIZE];

[numthreads(SORT_BLOCK_SIZE, 1, 1)]
void sort( 
	uint3 Gid : SV_GroupID, 
	uint3 DTid : SV_DispatchThreadID, 
	uint3 GTid : SV_GroupThreadID, 
	uint GI : SV_GroupIndex 
	)
{
    g_vShared[GI] = g_vData[DTid.x];
    GroupMemoryBarrierWithGroupSync();
    for (unsigned int j = g_iLevel >> 1 ; j > 0 ; j >>= 1) {
        unsigned int result = ((g_vShared[GI & ~j] <= g_vShared[GI | j]) == (bool)(g_iLevelMask & DTid.x))? g_vShared[GI ^ j] : g_vShared[GI];
        GroupMemoryBarrierWithGroupSync();
        g_vShared[GI] = result;
        GroupMemoryBarrierWithGroupSync();
    }
    g_vSorted[DTid.x] = g_vShared[GI];
}

groupshared unsigned int g_vXposeShared[TRANSPOSE_SIZE * TRANSPOSE_SIZE];

[numthreads(TRANSPOSE_SIZE, TRANSPOSE_SIZE, 1)]
void transpose( 
	uint3 Gid : SV_GroupID, 
	uint3 DTid : SV_DispatchThreadID, 
	uint3 GTid : SV_GroupThreadID, 
	uint GI : SV_GroupIndex 
	)
{
    g_vXposeShared[GI] = g_vData[DTid.y * g_iWidth + DTid.x];
    GroupMemoryBarrierWithGroupSync();
    uint2 XY = DTid.yx - GTid.yx + GTid.xy;
    g_vSorted[XY.y * g_iHeight + XY.x] = g_vXposeShared[GTid.x * TRANSPOSE_SIZE + GTid.y];
}
