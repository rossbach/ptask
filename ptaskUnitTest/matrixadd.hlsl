//--------------------------------------------------------------------------------------
// File: matrixadd.hlsl
//
// This file contains the Compute Shader to perform:
// array C = array A + array B
// 
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

struct ELEMTYPE
{
    int i;
    float f;
};

cbuffer cbCS : register( b0 )
{
    uint g_tex_cols;
	uint g_tex_rows;
};

StructuredBuffer<ELEMTYPE> A : register(t0);
StructuredBuffer<ELEMTYPE> B : register(t1);
RWStructuredBuffer<ELEMTYPE> C : register(u0);

[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void op( uint3 DTid : SV_DispatchThreadID )
{
	int idx	= DTid.y * g_tex_cols + DTid.x;
    C[idx].i = A[idx].i + B[idx].i;
    C[idx].f = A[idx].f + B[idx].f;
}


