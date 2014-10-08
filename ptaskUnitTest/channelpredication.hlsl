//--------------------------------------------------------------------------------------
// File: channelpredication.hlsl
// This file contains a Compute Shader to support testing of 
// channel predication for the DirectX back end
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

RWStructuredBuffer<float> A : register(u0);
RWStructuredBuffer<float> B : register(u1);

cbuffer cbScalar : register( b0 ) { float scalar; }
cbuffer cbSize   : register( b1 ) { uint N; }

// #define thread_group_size_x 1
// #define thread_group_size_y 1
// #define thread_group_size_z 1
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void scale( uint3 DTid : SV_DispatchThreadID )
{
	int i	= DTid.x;
    if (i < N) {
        A[i] = A[i]*scalar;
        B[i] = B[i]*scalar;
    }
}
