//--------------------------------------------------------------------------------------
// File: metaports.hlsl
// This file contains a Compute Shader to support testing of 
// meta ports for the DirectX back end
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

StructuredBuffer<float> A : register(t0);
RWStructuredBuffer<float> B : register(u0);

cbuffer cbScalar : register( b0 ) { float scalar; }
cbuffer cbSize   : register( b1 ) { uint N; }

// #define thread_group_size_x 1
// #define thread_group_size_y 1
// #define thread_group_size_z 1
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void op( uint3 DTid : SV_DispatchThreadID )
{
	int i	= DTid.x;
    if (i < N) {
        B[i] = A[i] + scalar;
    }
}
