// Device code
// A is assumed to be initialized by an
// initializer port to be uniformly 0. 
// output should be uniformly scalar.
extern "C" __global__ void scale(float* A, float scalar, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        A[i] = A[i]+scalar;
}
