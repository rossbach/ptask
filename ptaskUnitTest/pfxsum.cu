///-------------------------------------------------------------------------------------------------
// file:	pfxsum.cu
//
// summary:	simple cuda prefix sum
///-------------------------------------------------------------------------------------------------

#define PFXSUM_BLOCK_SIZE  256

__device__ int nextpowerof2(int n) {
int k;
	if (n > 1) 
	{
	  float f = (float) n;
	  unsigned int const t = 1U << ((*(unsigned int *)&f >> 23) - 0x7f);
	  k = t << (t < n);
	}
	else k = 1;
	return k;
}

extern "C" __global__ void 
pfxsum( 
    int * pin,
    int * pout,
	int N
    ) 
{
	__shared__ int g_shared[PFXSUM_BLOCK_SIZE];
	unsigned int t;
	int idx	= threadIdx.x;
    int d, e;
    int offset = 1;
	int nbIdx = idx*2;
	int nbIdx1 = 2*idx+1;
	g_shared[nbIdx] = nbIdx < N ? pin[nbIdx] : 0;
	g_shared[nbIdx1] = nbIdx1 < N ? pin[nbIdx1] : 0;
	float f = (float) N;
	t = 1U << ((*(unsigned int *)&f >> 23) - 0x7f);
	int nUpper = t << (t < N);
	for(d = nUpper >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if(idx < d) {
            int ai = offset*(2*idx+1)-1;
            int bi = offset*(2*idx+2)-1;
            g_shared[bi] += g_shared[ai];
        }
        offset *= 2;
    }
    if(idx == 0) {
        g_shared[nUpper-1] = 0;
    }
    for(e=1; e<nUpper; e*=2) {
        offset >>= 1;
        __syncthreads();
        if(idx < e) {
            int ai = offset*(2*idx+1)-1;   
            int bi = offset*(2*idx+2)-1;  
			int t = g_shared[ai];   
			g_shared[ai] = g_shared[bi];   
			g_shared[bi] += t; 
        }
    }
    __syncthreads();
    if(2*idx<N) pout[2*idx] = g_shared[2*idx];
    if(2*idx+1<N) pout[2*idx+1] = g_shared[2*idx+1];
	__syncthreads();
}

template<typename T,                // type of input (assumed integral)
         int nBlockSize>            // shared memory block size 
__device__ void 	
tpfxsum( 
	T * pin,
	T * pout,
	int N
	) 
{
	__shared__ int g_shared[nBlockSize];
	int idx	= threadIdx.x;
	T d, e;
	int offset = 1;
	g_shared[idx*2] = pin[2*idx];
	g_shared[2*idx+1] = pin[2*idx+1];
	for(d = N >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if(idx < d) {
			T ai = offset*(2*idx+1)-1;
			T bi = offset*(2*idx+2)-1;
			g_shared[bi] += g_shared[ai];
		}
		offset *= 2;
	}
	if(idx == 0) {
		g_shared[N-1] = 0;
	}
	for(e=1; e<N; e*=2) {
		offset >>= 1;
		__syncthreads();
		if(idx < e) {
			int ai = offset*(2*idx+1)-1;   
			int bi = offset*(2*idx+2)-1;  
			int t = g_shared[ai];   
			g_shared[ai] = g_shared[bi];   
			g_shared[bi] += t; 
		}
	}
	__syncthreads();
	pout[2*idx] = g_shared[2*idx];
	pout[2*idx+1] = g_shared[2*idx+1];
	__syncthreads();
}

extern "C" __global__ void 
cpfxsum( 
    int * pin,
    int * pout,
	int N
    ) 
{
	tpfxsum<int, PFXSUM_BLOCK_SIZE>(pin,
									pout,
									N);
}