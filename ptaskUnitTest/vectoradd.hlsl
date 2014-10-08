ByteAddressBuffer A : register(t0);
ByteAddressBuffer A_metadata : register(t1);
ByteAddressBuffer B : register(t2);
ByteAddressBuffer B_metadata : register(t3);
RWByteAddressBuffer C : register(u0);
RWByteAddressBuffer C_metadata : register(u1);
/*
#define thread_group_size_x 1
#define thread_group_size_y 1
#define thread_group_size_z 1
*/
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void op( uint3 DTid : SV_DispatchThreadID )
{
	int metaidx = DTid.x*4;
    int len = asint(A_metadata.Load(metaidx));
    //int cols = asint(A_metadata.Load(metaidx));
	int index = 0;
    int matbase = DTid.x * 4 * len;
    for(int r=0; r<len; r++) {
        int aidx = (r * 4) + matbase;
	    int bidx = (r * 4) + matbase;
         
        float aelem_f = asfloat(A.Load(aidx));
        float belem_f = asfloat(B.Load(bidx));
			    
	    float t = aelem_f + belem_f;
        int elemidx = (r * 4) + matbase;
        
		C.Store(elemidx, asint(t));
    }
	
    //C_metadata.Store(metaidx, asint(rows));
    C_metadata.Store(metaidx, asint(len));
}





