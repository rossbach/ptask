///-------------------------------------------------------------------------------------------------
// file:	pipelinestresstest.hlsl
//
// summary:	long-latency compute kernels to help stress PTask
//          pipeline parallelism / asynchrony.
///-------------------------------------------------------------------------------------------------

StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);

cbuffer cbCS : register( b0 )
{
    int g_tex_cols;
	int g_tex_rows;
    int g_tex_halfwin;
};

//[numthreads(1, 1, 1)]
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void op( uint3 DTid : SV_DispatchThreadID )
{
	float t = 0;
    int nCells = 0;
    int yidx = (int)DTid.y;
    int xidx = (int)DTid.x;
    for(int di=-g_tex_halfwin; di<g_tex_halfwin; di++) {
        for(int dj=-g_tex_halfwin; dj<g_tex_halfwin; dj++) {
            if(yidx+di < 0 || yidx+di >= g_tex_rows) continue;
            if(xidx+dj < 0 || xidx+dj >= g_tex_cols) continue;
            int idx = ((yidx+di) * g_tex_cols) + (xidx+dj);
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
	int cidx = ( DTid.y * g_tex_cols + DTid.x );
    float v = nCells > 0 ? t/nCells : 0.0f;
    C[cidx] = v;
}




