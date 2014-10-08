///-------------------------------------------------------------------------------------------------
// file:	descportsout.cu
//
// summary:	test kernal for output descriptor ports test case:
//          The test does a normal vector scale, but the output data
//          block should also have 'N' in the metadata channel and
//          the entire contents of the pMetaData array in the template channel.
///-------------------------------------------------------------------------------------------------

extern "C" __global__ void 
scale(
    float* A,             // matrix
    float scalar,         // scalar for A * scalar
    int N,                // size of A, pOut
    float * pOut,         // output result
    int * pMDOut,         // meta data channel out (should have N)
    float * pTmplDataIn,  // input data destined for output template channel
    int nTmpl,            // size of pTmplDataIn
    float * pTmplDataOut  // output buffer for tmpl channel
    )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N)
        pOut[i] = A[i]*scalar;
    if(i<nTmpl)
        pTmplDataOut[i] = pTmplDataIn[i];    
    if(i==0)
        pMDOut[i] = N;    
}
