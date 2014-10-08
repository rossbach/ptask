//--------------------------------------------------------------------------------------
// File: matrixadd_4ch.hlsl
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

StructuredBuffer<ELEMTYPE> A : register(t0);
StructuredBuffer<int> A_metadata : register(t1);
StructuredBuffer<ELEMTYPE> B : register(t2);
StructuredBuffer<int> B_metadata : register(t3);
RWStructuredBuffer<ELEMTYPE> C : register(u0);
RWStructuredBuffer<int> C_metadata : register(u1);


//#define thread_group_size_x 1
//#define thread_group_size_y 1
//#define thread_group_size_z 1
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void main( uint3 DTid : SV_DispatchThreadID )
{
    int rows = A_metadata[0];
    int cols = A_metadata[1];
    C_metadata[0] = rows;
    C_metadata[1] = cols;
    int idx	= DTid.y * cols + DTid.x;
    C[idx].i = A[idx].i + B[idx].i;
    C[idx].f = A[idx].f + B[idx].f;
}


