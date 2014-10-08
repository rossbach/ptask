// Device code
extern "C" __global__ void op(float* A, float * B, float scalar, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        A[i] = A[i]*scalar;
        B[i] = B[i] + scalar;
    }
}

extern "C" __global__ void op2(float* A, float * B, float scalar, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        B[i] = B[i] + scalar;
        A[i] = A[i]*scalar + B[i];
    }
}

