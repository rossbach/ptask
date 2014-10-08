// Device code
// A is assumed to be initialized by an
// initializer port to be uniformly 0. 
// the length of the output is determined
// by a metaport.
// Given an input matrix A of length N'>N
// the ptask runtime code for this will
// allocate an output of size N. Output
// should be uniform scalar of size N
extern "C" __global__ void op(float* A, float * B, float scalar, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        B[i] = A[i]+scalar;
}
