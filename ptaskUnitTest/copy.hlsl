//--------------------------------------------------------------------------------------
// File: copy.hlsl
// This file contains a Compute Shader to perform
// minimal work. Given input arrays A, B, 
// 1D array C = copy(A)
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);

#define thread_group_size_x 1
#define thread_group_size_y 1
#define thread_group_size_z 1
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void op( uint3 DTid : SV_DispatchThreadID )
{
	int idx	= DTid.x;
	C[idx] = A[idx];
}


