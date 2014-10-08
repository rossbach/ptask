/*
 * deferredports.cl
 */
 
__kernel void op(
    __global int * pulse,
	__global float* a, 
	float scalar, 
	int N
    )
{
    int iGID = get_global_id(0);
    if (iGID >= N)
        return;     
    a[iGID] = a[iGID] * scalar;
}

