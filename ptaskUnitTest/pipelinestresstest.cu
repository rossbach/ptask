///-------------------------------------------------------------------------------------------------
// file:	pipelinestresstest.cu
//
// summary:	long-latency compute kernels to help stress PTask
//          pipeline parallelism / asynchrony.
///-------------------------------------------------------------------------------------------------

typedef struct _pstress_params_t {
	int g_tex_cols;
	int g_tex_rows;
	int g_tex_halfwin;
	int g_pad1;
} PSTRESSPARMS;

extern "C" __global__ void 
op(
    float*  A, 
    float * B, 
    float * C,
    PSTRESSPARMS parms
    )
{
	float t = 0;
    int nCells = 0;
    int xidx = blockDim.x * blockIdx.x + threadIdx.x;
    int yidx = blockDim.y * blockIdx.y + threadIdx.y;
    for(int di=-parms.g_tex_halfwin; di<parms.g_tex_halfwin; di++) {
        for(int dj=-parms.g_tex_halfwin; dj<parms.g_tex_halfwin; dj++) {
            if(yidx+di < 0 || yidx+di >= parms.g_tex_rows) continue;
            if(xidx+dj < 0 || xidx+dj >= parms.g_tex_cols) continue;
            int idx = ((yidx+di) * parms.g_tex_cols) + (xidx+dj);
            float aval = A[idx];
            float bval = B[idx];
            float abprod = aval*bval;
            float sina = sin(aval);
            float cosb = cos(bval);
            float tanab = tan(abprod);
            float inc = tanab/(sina*cosb);
		    t+=inc;
            nCells++;
        }
	}
	int cidx = ( yidx * parms.g_tex_cols + xidx );
    float v = nCells > 0 ? t/nCells : 0.0f;
    C[cidx] = v;
}




