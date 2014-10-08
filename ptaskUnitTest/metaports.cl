/*
 * metaports.cl
 * OpenCL single dimension B[] = A[] + scalar
 */
 
__kernel void op(
	__global float* a, 
	__global float* b, 
	float scalar, 
	int N
    )
{
    int iGID = get_global_id(0);
    if (iGID >= N)
        return;     
    b[iGID] = a[iGID] + scalar;
}

