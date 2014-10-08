/*
 * gatedports.cl
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
    a[iGID] = a[iGID] * scalar;
    b[iGID] = b[iGID] * scalar;
}

