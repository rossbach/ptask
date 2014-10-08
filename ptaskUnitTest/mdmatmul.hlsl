//--------------------------------------------------------------------------------------
// File: matrixmul.hlsl
//
// This file contains a Compute Shader for matrix multiplication
// 
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
// #ifndef thread_group_size_x
// #define thread_group_size_x 1
// #define thread_group_size_y 1
// #define thread_group_size_z 1
// #endif

struct ELEMTYPE
{
	int i;
    float f;
};

struct MATRIX_METADATA 
{
    int offset;
    int N;
};

StructuredBuffer<ELEMTYPE> A : register(t0);
StructuredBuffer<MATRIX_METADATA> mdA : register(t1);
StructuredBuffer<ELEMTYPE> B : register(t2);
StructuredBuffer<MATRIX_METADATA> mdB : register(t3);
RWStructuredBuffer<ELEMTYPE> C : register(u0);
RWStructuredBuffer<MATRIX_METADATA> mdC : register(u1);

cbuffer cbCS : register( b0 )
{
    uint g_object_cols;
	uint g_object_rows;
};

[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	int idx = ( DTid.y * g_object_cols + DTid.x );
    MATRIX_METADATA amd = mdA[idx];
    MATRIX_METADATA bmd = mdB[idx];
    int offset = amd.offset;
    int N = amd.N;
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            int nt = 0;
	        float t = 0;
            for(int k=0; k<N; k++) {
		        int aidx = ( i * N + k ) + offset;
		        int bidx = ( k * N + j ) + offset;
                nt+=A[aidx].i*B[bidx].i;
		        t+=A[aidx].f*B[bidx].f;
            }
            int cidx = ((i * N) + j) + offset;
            C[cidx].i = nt;
            C[cidx].f = t;
        }
    }
    mdC[idx].offset = offset;
    mdC[idx].N = N;
}


