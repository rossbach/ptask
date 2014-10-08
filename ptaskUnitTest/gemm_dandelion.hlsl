//--------------------------------------------------------------------------------------
// File: gemm_dandelion.hlsl
//
// This file contains a Compute Shader for matrix multiplication 
// that assumes the fully general dandelion interface, which
// includes a meta-data and template-data along with data per
// datablock/channel
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
	int metaidx = DTid.x*8;
    int rows = asint(A_metadata.Load(metaidx));
    int cols = asint(A_metadata.Load(metaidx+4));
	int index = 0;
    int matbase = DTid.x * 8 * rows * cols;
    for(int r=0; r<rows; r++) {
        for(int c=0; c<cols; c++) {
	        int nt = 0;
	        float t = 0;
            for(int k=0; k<rows; k++) {
	            int aidx = ((r * cols + k)*8) + matbase;
	            int bidx = ((k * cols + c)*8) + matbase;
                int aelem_i = asint(A.Load(aidx));
                float aelem_f = asfloat(A.Load(aidx+4));
                int belem_i = asint(B.Load(bidx));
                float belem_f = asfloat(B.Load(bidx+4));
	            nt+=aelem_i*belem_i;
	            t+=aelem_f*belem_f;
            }
            int elemidx = ((r * cols + c) * 8)+matbase;
            C.Store(elemidx, asint(nt));
            C.Store(elemidx+4, asuint(t));
        }
	}
    C_metadata.Store(metaidx, asint(rows));
    C_metadata.Store(metaidx+4, asint(cols));
}





