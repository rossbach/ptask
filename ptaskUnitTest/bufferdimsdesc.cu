// Device code
typedef struct BUFFERDIMS_t {
    unsigned int X;
    unsigned int Y;
    unsigned int Z;
    unsigned int stride;
    unsigned int pitch;
} BUFFERDIMS;

extern "C" __global__ void 
scale(
    float * A, 
    float * B, 
    float scalar, 
    BUFFERDIMS dims
    )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < dims.X * dims.Y)
        B[i] = A[i]*scalar;
}
