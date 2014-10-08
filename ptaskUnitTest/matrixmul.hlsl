//--------------------------------------------------------------------------------------
// File: matrixmul.hlsl
//
// This file contains a Compute Shader for matrix multiplication
// 
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

struct ELEMTYPE
{
	int i;
    float f;
};

StructuredBuffer<ELEMTYPE> A : register(t0);
StructuredBuffer<ELEMTYPE> B : register(t1);
RWStructuredBuffer<ELEMTYPE> C : register(u0);

cbuffer cbCS : register( b0 )
{
    uint g_tex_cols;
	uint g_tex_rows;
};

[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void op( uint3 DTid : SV_DispatchThreadID )
{
	int nt = 0;
	float t = 0;
	int index = 0;
	for(int i=0; i<g_tex_cols; i++) {
		int aidx = ( DTid.y * g_tex_cols + i );
		int bidx = ( i * g_tex_cols + DTid.x );
		nt+=A[aidx].i*B[bidx].i;
		t+=A[aidx].f*B[bidx].f;
	}
	int cidx = ( DTid.y * g_tex_cols + DTid.x );
    C[cidx].i = nt;
    C[cidx].f = t;
}




