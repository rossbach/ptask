//--------------------------------------------------------------------------------------
// File: matrixaddraw.hlsl
//
// This file contains the Compute Shader to perform:
// array C = array A + array B
// use raw byte addressable buffers (instead of structure buffer primitives)
// 
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

ByteAddressBuffer A : register(t0);
ByteAddressBuffer B : register(t1);
RWByteAddressBuffer C : register(u0);

cbuffer cbCS : register( b0 )
{
    uint g_tex_cols;
	uint g_tex_rows;
};

[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void op( uint3 DTid : SV_DispatchThreadID )
{
	int idx = 8*( DTid.y * g_tex_cols + DTid.x );
    int i0 = asint( A.Load( idx ) );
    float f0 = asfloat( A.Load( idx+4 ) );
    int i1 = asint( B.Load( idx ) );
    float f1 = asfloat( B.Load( idx+4 ) );    
    C.Store( idx, asuint(i0 + i1) );
    C.Store( idx+4, asuint(f0 + f1) );
}



