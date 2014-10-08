//--------------------------------------------------------------------------------------
// File: inout.hlsl
//
// This file contains a Compute Shader to add a constant
// to every member of an array, part of testing ptask in/out
// support for the DirectX back-end.
// 
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

cbuffer cbCS : register( b0 )
{
    uint g_n;
    uint g_scalar;
};

RWStructuredBuffer<float> A : register(u0);

//#define thread_group_size_x 1
//#define thread_group_size_y 1
//#define thread_group_size_z 1
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void op( uint3 DTid : SV_DispatchThreadID )
{
	int idx	= DTid.x;
    if(idx < g_n) {
        A[idx] = g_scalar + A[idx];
    }
}


