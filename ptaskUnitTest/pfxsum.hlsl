//--------------------------------------------------------------------------------------
// File: pfxsum.hlsl
// This file contains a Compute Shader to perform prefix sum
// 1D array B = prefix_sum(A)
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

#define PFXSUM_BLOCK_SIZE  512
StructuredBuffer<int> g_pin : register(t0);
RWStructuredBuffer<int> g_pout : register(u0);
groupshared int g_shared[PFXSUM_BLOCK_SIZE];
cbuffer cbCS : register( b0 )
{
    uint g_N;
};

/*! Compute the prefix sum given an 
 *  array of integers as input
 *  \param int idx -- thread id
 *  \param int N -- number of elements in input
 *  \param StructuredBuffer<int> pin -- input array
 *  \param RWStructuredBuffer<int> pout -- output array
 *  \return void (output written to pout parameter)
 */
void pfxsum( 
    int idx,
    int N,
    StructuredBuffer<int> pin,
    RWStructuredBuffer<int> pout
    ) 
{
    int d, e;
    int offset = 1;
    g_shared[idx*2] = pin[2*idx];
    g_shared[2*idx+1] = pin[2*idx+1];
    for(d = N >> 1; d > 0; d >>= 1) {
        GroupMemoryBarrierWithGroupSync();
        if(idx < d) {
            int ai = offset*(2*idx+1)-1;
            int bi = offset*(2*idx+2)-1;
            g_shared[bi] += g_shared[ai];
        }
        offset *= 2;
    }
    if(idx == 0) {
        g_shared[N-1] = 0;
    }
    for(e=1; e<N; e*=2) {
        offset >>= 1;
        GroupMemoryBarrierWithGroupSync();
        if(idx < e) {
            int ai = offset*(2*idx+1)-1;   
            int bi = offset*(2*idx+2)-1;  
            int t = g_shared[ai];   
            g_shared[ai] = g_shared[bi];   
            g_shared[bi] += t; 
        }
    }
    GroupMemoryBarrierWithGroupSync();
    pout[2*idx] = g_shared[2*idx];
    pout[2*idx+1] = g_shared[2*idx+1];
}


#define tgsX 64
#define tgsY 1
#define tgsZ 1

[numthreads(tgsX, tgsY, tgsZ)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	int idx	= DTid.x;
    pfxsum(idx, g_N, g_pin, g_pout); 
    GroupMemoryBarrierWithGroupSync();
}

