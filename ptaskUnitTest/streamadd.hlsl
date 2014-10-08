//--------------------------------------------------------------------------------------
// File: streamadd.hlsl
// This file contains the Compute Shader to perform:
// 1D array C = array A + array B
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);

#define thread_group_size_x 1
#define thread_group_size_y 1
#define thread_group_size_z 1
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	int idx	= DTid.x;
	C[idx] = A[idx] + B[idx];
}

