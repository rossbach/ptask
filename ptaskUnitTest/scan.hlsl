//--------------------------------------------------------------------------------------
// File: scan.hlsl
//
// compute shader implementation of scan
// 
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

struct ENTRY {
    int key_offset;
    int key_length;
    int val_offset;
    int val_length;
};


StructuredBuffer<ENTRY> g_directory : register(t0);
ByteAddressBuffer g_keys : register(t1);
ByteAddressBuffer g_values : register(t2);
RWStructuredBuffer<int> g_outputsizes : register(u0);
RWByteAddressBuffer g_result : register(u1);

cbuffer cbCS : register( b0 )
{
    uint g_entries;
};

int rup(int siz) {
    return ((siz % 4 == 0) ? siz : (((siz/4)+1)*4));
}

#define thread_group_size_x 1
#define thread_group_size_y 1
#define thread_group_size_z 1
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
// select * from input table
void select_count( uint3 DTid : SV_DispatchThreadID )
{
	int idx = DTid.x;
    ENTRY entry = g_directory[idx];
    // since we are doing select *, we 
    // don't actually have to compute on the 
    // input, we can sum up the offset and 
    // key lengths. However, since we have
    // no interface to write a byte at a time,
    // we cannot avoid sync issues with neighboring 
    // threads unless we round up to 4-byte boundary.
    int nOutputSize = rup(entry.key_length) + rup(entry.val_length); // assume length includes terminator
    g_outputsizes[idx] = nOutputSize;
}

[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void main(uint3 DTid : SV_DispatchThreadID )
{
	int idx = DTid.x;
    ENTRY entry = g_directory[idx];
    asint(g_keys.Load(entry.key_offset))

    // since we are doing select *, we 
    // don't actually have to compute on the 
    // input, we can sum up the offset and 
    // key lengths. 
    int nOutputSize = entry.key_length + entry.val_length; // assume length includes terminator
    g_outputsizes[idx] = nOutputSize;
}





