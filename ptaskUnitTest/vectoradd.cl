/*
 * vectoradd.cl
 * OpenCL single dimension array add.
 */
 
__kernel void vadd(
	__global const float* a, 
	__global const float* b, 
	__global float* c, 
	int count)
{
    int iGID = get_global_id(0);
    if (iGID >= count)
        return;     
    c[iGID] = a[iGID] + b[iGID];
}
