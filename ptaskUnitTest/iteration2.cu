// Device code
extern "C" __global__ void op(float* A, float scalar, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        A[i] = A[i]*scalar;
    }
}
