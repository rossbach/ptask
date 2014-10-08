//--------------------------------------------------------------------------------------
// File: matrixmul_4ch.hlsl
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

ByteAddressBuffer A : register(t0);
ByteAddressBuffer A_metadata : register(t1);
ByteAddressBuffer B : register(t2);
ByteAddressBuffer B_metadata : register(t3);
RWByteAddressBuffer C : register(u0);
RWByteAddressBuffer C_metadata : register(u1);


//#define thread_group_size_x 1
//#define thread_group_size_y 1
//#define thread_group_size_z 1
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void op( uint3 DTid : SV_DispatchThreadID )
{
    int rows = asint(A_metadata.Load(0));
    int cols = asint(A_metadata.Load(4));
    C_metadata.Store(0, asint(rows));
    C_metadata.Store(4, asint(cols));
	int nt = 0;
	float t = 0;
	int index = 0;
	for(int i=0; i<cols; i++) {
		int aidx = (( DTid.y * cols + i )*8);
		int bidx = (( i * cols + DTid.x )*8);
        int aelem_i = asint(A.Load(aidx));
        float aelem_f = asfloat(A.Load(aidx+4));
        int belem_i = asint(B.Load(bidx));
        float belem_f = asfloat(B.Load(bidx+4));
		nt+=aelem_i*belem_i;
		t+=aelem_f*belem_f;
	}
	int cidx = ( DTid.y * cols + DTid.x )*8;
    C.Store(cidx, asint(nt));
    C.Store(cidx+4, asuint(t));
}





