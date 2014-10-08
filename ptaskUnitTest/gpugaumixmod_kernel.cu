// Kernel code for Gaussian Mixture Model Expectation Maximization.
//
// Andrew Harp (andrew.harp@gmail.com)
// http://andrewharp.com/gmmcuda

// This effects the maximum dimensionality of your data.
// Has to be hardcoded because it affects memory allocation.
// Due to the way I clear out the lower triangle of the cholesky (and
// possibly other) places MAX_DIM * MAX_DIM needs to be less than
// BLOCK_SIZE.
#define MAX_DIM 16

// You can change this, but only to a power of 2.
#define BLOCK_SIZE 256

// Estep-normalize is broken down into an arbitrary number of chunks.
#define NUM_CHUNKS 32

#include "cuda_runtime.h"

///////////////////////////////////////////////////////////////////////////

__device__ void cholesky(float* el, const unsigned int ndim) {
  const unsigned int tid = threadIdx.x;
  
  // Dunno how to parallelize this part...
  if (tid == 0) {
    float sum = 0;
    
    int i, j, k;
    //if (el.ncols() != n) throw("need square matrix");
    for (i=0; i<ndim; i++) {
      for (j=i; j<ndim; j++) {
        sum = el[__umul24(i, ndim)+j];
        
        for (k=i-1; k >= 0; k--) {
          sum -= el[__umul24(i, ndim)+k] * el[__umul24(j, ndim)+k];
        }
        
        if (i == j) {
          //if (sum <= 0.0)
          //	throw("Cholesky failed");
          el[__umul24(i, ndim)+i] = sqrt(sum);
        } else {
          el[__umul24(j, ndim)+i] = sum/el[__umul24(i, ndim)+i];
        }
      }
    }
  }
  
  __syncthreads();
  
  // Clear lower triangular part.
  if ((tid/ndim) < (tid%ndim)) {
    el[__umul24((tid/ndim), ndim) + (tid%ndim)] = 0.0f;
  }
}

///////////////////////////////////////////////////////////////////////////

__device__ float logdet(float* el, const unsigned int ndim) {
  float sum = 0.0f;
  for (unsigned int i=0; i<ndim; ++i) {
    sum += __logf(el[(i*ndim)+i]);
  }
  return 2.*sum;
}

///////////////////////////////////////////////////////////////////////////

__device__ float parallelSum(float* data, const unsigned int ndata) {
  const unsigned int tid = threadIdx.x;
  float t;

  __syncthreads();

  // Butterfly sum.  ndata MUST be a power of 2.
  for(unsigned int bit = ndata >> 1; bit > 0; bit >>= 1) {
    t = data[tid] + data[tid^bit];  __syncthreads();
    data[tid] = t;                  __syncthreads();
  }
  return data[tid];
}

///////////////////////////////////////////////////////////////////////////

__device__ void copyArray(float* fromArr, float* toArr, unsigned int ndata=BLOCK_SIZE) {
  unsigned int n;
  unsigned int base_off;
  
  for (base_off=0; base_off < ndata; base_off += blockDim.x) {
    n = base_off + threadIdx.x;
    
    if (n < ndata) {
      toArr[n] = fromArr[n];
    }
  }
}

///////////////////////////////////////////////////////////////////////////

// Parallel reduction, for when all you want is the sum of a certain
// quantity computed for every 1 to N.  CODE should be something in terms
// of n.  The resulting sum will be placed in RESULT.
// tmp_buff, base_off, RESULT, and n must be previously defined, however 
// they will be overwritten during the execution of the macro.
#define REDUCE(N, CODE, RESULT)                                \
base_off = 0;                                                  \
RESULT = 0.0f;                                                 \
while (base_off + BLOCK_SIZE < N) {                            \
  n = base_off + tid;                                          \
  tmp_buff[tid] = CODE;                                        \
  RESULT += parallelSum(tmp_buff, BLOCK_SIZE);                 \
  base_off += BLOCK_SIZE;                                      \
}                                                              \
n = base_off + tid;                                            \
if (n < N) {tmp_buff[tid] = CODE;}                             \
else {tmp_buff[tid] = 0.0f;}                                   \
RESULT += parallelSum(tmp_buff, BLOCK_SIZE);
          
///////////////////////////////////////////////////////////////////////////

// This function computes for a single cluster k.
__global__ void estep_kernel(float* _resp_, float* _frac_, 
                             float* _data_, float* _means_, 
                             float* _sig_,  float* _lndets_,
                             const unsigned int num_clusts, 
                             const unsigned int num_dims, 
                             const unsigned int num_data) {
  const unsigned int try_num = blockIdx.y;
  
  const unsigned int tid = threadIdx.x;
  const unsigned int k = blockIdx.x;
  const unsigned int sigsize = __umul24(num_dims, num_dims);
  
  // Base offsets.
  const unsigned int rb   = __umul24(try_num, num_clusts) * num_data + (k*num_data); 
  const unsigned int mb   = __umul24(try_num, num_clusts) * num_dims;
  const unsigned int sb   = __umul24(try_num, num_clusts) * sigsize;
  const unsigned int fb   = __umul24(try_num, num_clusts);
  const unsigned int lndb = __umul24(try_num, num_clusts);
  
  unsigned int n, base_off;
  float tmp, sum;
  float v[MAX_DIM];
  
  __shared__ float lndet_k;
  __shared__ float frac_k;
  
  __shared__ float chol[MAX_DIM * MAX_DIM];

  __syncthreads();
  
  copyArray(_sig_ + sb + __umul24(k, sigsize), chol, sigsize);
  
  cholesky(chol, num_dims);
  
  __syncthreads();
  
  if (tid == 0) {
    frac_k = _frac_[fb + k];
    lndet_k = logdet(chol, num_dims);
    _lndets_[lndb + k] = lndet_k;
  }
  
  __syncthreads();
  
  // Loop through data.
  for (base_off=0; base_off < num_data; base_off += BLOCK_SIZE) {
    n = base_off + tid;
    sum=0.0f;
    
    //if (b.size() != n || y.size() != n) throw("bad lengths");
    if (n < num_data) {
      for (unsigned int i=0; i<num_dims; ++i) {
        tmp = _data_[__umul24(i, num_data) + n] - _means_[mb + __umul24(k, num_dims)+i];
        
        for (unsigned int j=0; j<i; j++) {
          tmp -= chol[__umul24(i, num_dims)+j] * v[j];
        }
        
        v[i] = tmp/chol[__umul24(i, num_dims)+i];
        sum += v[i] * v[i];
      }

      // Assign likelihood of this data being in this cluster.
      _resp_[rb + n] = -0.5f*(sum + lndet_k) + __logf(frac_k);
    }
  } // (n < num_data)
}

///////////////////////////////////////////////////////////////////////////

// Loop through data again, normalizing probabilities.
// We are looping across clusters here as well as data, since every data
// point needs to know its potential parents.
__global__ void estep_normalize_kernel(float* _resp_, float* _frac_, 
                                       float* _data_, float* _means_, 
                                       float* _sig_,  float* _lndets_,
                                       float* _loglike_,
                                       const unsigned int num_clusts, 
                                       const unsigned int num_dims, 
                                       const unsigned int num_data) {
  const unsigned int try_num = blockIdx.y;
  const unsigned int num_chunks = gridDim.x;
  const unsigned int chunk_num = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  
  // We're only handling so many data points per block in this kernel, since
  // data is independant of other data here.
  const unsigned int n_per_block = ceil((float)num_data / (float)num_chunks);
  const unsigned int start_off = __umul24(n_per_block, chunk_num);
  const unsigned int end_off = min(start_off + n_per_block, num_data);
  
  // Base offsets.
  const unsigned int rb = __umul24(try_num, num_clusts) * num_data; 
  const unsigned int lb = __umul24(try_num, num_chunks);
  
  unsigned int n, base_off, k;
  float sum, max, tmp;
  
  __shared__ float loglike[BLOCK_SIZE];
  loglike[tid] = 0.0f;
  
  __syncthreads();
  
  // Loop through data.
  for (base_off = start_off; base_off < end_off; base_off += BLOCK_SIZE) {
    n = base_off + tid;
    
    if (n < end_off) {
      max = -99.9e30f;
      
      // Find cluster with maximum likelihood for this data point.
      for (k=0; k<num_clusts; ++k) {
        tmp = _resp_[rb + (k*num_data) + n];
        if (tmp > max) {
          max = tmp;
        }
      }
      
      // Sum marginal probabilities.
      sum = 0.0f;
      for (k=0; k<num_clusts; ++k) {
        sum += __expf(_resp_[rb + (k*num_data) + n] - max);
      }
      // Assign probabilities of point belonging to each cluster.
      tmp = max + __logf(sum);
      for (k = 0; k < num_clusts; ++k) {
        _resp_[rb + (k*num_data) + n] = 
                __expf(_resp_[rb + (k*num_data) + n] - tmp);
      }
      loglike[tid] += tmp;
    }
  }
  
  tmp = parallelSum(loglike, BLOCK_SIZE);
  if (tid == 0) {
    _loglike_[lb + chunk_num] = tmp;
  }
}

///////////////////////////////////////////////////////////////////////////

__global__ void mstep_kernel(float* _resp_, float* _frac_, 
                             float* _data_, float* _means_, 
                             float* _sig_,
                             const unsigned int num_clusts, 
                             const unsigned int num_dims, 
                             const unsigned int num_data) {
  const unsigned int try_num = blockIdx.x / num_clusts;
  
  const unsigned int tid = threadIdx.x;
  
  // Every block is mapped to cluster and dimension.
  const unsigned int k = blockIdx.x % num_clusts;
  const unsigned int m = blockIdx.y;
  
  // Base offsets.
  const unsigned int rb = __umul24(try_num, num_clusts) * num_data + (k*num_data); 
  const unsigned int mb = __umul24(try_num, num_clusts) * num_dims;
  const unsigned int fb = __umul24(try_num, num_clusts);
  
  unsigned int n, base_off;
  float wgt_k, sum_k_m;
  
  __shared__ float tmp_buff[BLOCK_SIZE];
  
  // Sum all weight assigned to cluster k.
  REDUCE(num_data, _resp_[rb + n], wgt_k);
  
  __syncthreads();
  
  // Update fractional prior.
  if (tid == 0 && m == 0) {
    _frac_[fb + k] = wgt_k / (float)num_data;
  }
  
  __syncthreads();
  
  // Only concerned about dimension m in this block.
  // Sum will become the sum of movement in that direction for this cluster.
  REDUCE(num_data, _resp_[rb + n] * _data_[(m*num_data)+n], sum_k_m);
  
  __syncthreads();
  
  if (tid == 0) {
    _means_[mb + __umul24(k, num_dims) + m] = sum_k_m / wgt_k;
  }
}

///////////////////////////////////////////////////////////////////////////

__global__ void mstep_sigma_kernel(float* _resp_, float* _frac_, 
                                   float* _data_, float* _means_, 
                                   float* _sig_,
                                   const unsigned int num_clusts, 
                                   const unsigned int num_dims, 
                                   const unsigned int num_data) {
  // Every block is mapped to cluster and dimension pair.
  const unsigned int try_num = blockIdx.x / num_clusts;
  
  const unsigned int k = blockIdx.x % num_clusts;
  const unsigned int m = blockIdx.y / num_dims;
  const unsigned int j = blockIdx.y % num_dims;
  const unsigned int tid = threadIdx.x;
  
  const unsigned int sigsize = __umul24(num_dims, num_dims);
  
  // Base offsets.
  const unsigned int rb = __umul24(try_num, num_clusts) * num_data + (k*num_data); 
  const unsigned int mb = __umul24(try_num, num_clusts) * num_dims;
  const unsigned int sb = __umul24(try_num, num_clusts) * sigsize;
  const unsigned int fb = __umul24(try_num, num_clusts);
  const unsigned int db_m = (m*num_data);
  const unsigned int db_j = (j*num_data);
  
  unsigned int n, base_off;
  __shared__ float tmp_buff[BLOCK_SIZE];
    
  __shared__ float wgt_k;
  __shared__ float mean_k_m;
  
  if (tid == 0) {
    wgt_k = _frac_[fb + k] * num_data;
    mean_k_m = _means_[mb + __umul24(k, num_dims) + m];
  } 
  
  __syncthreads();
  
  float sum;
  REDUCE(num_data, 
         _resp_[rb + n] * 
         (_data_[db_m + n] - mean_k_m) * 
         (_data_[db_j + n]), 
         sum);
  
  // Set this block's Sigma val.
  if (tid == 0) {
    _sig_[sb + 
          (__umul24(k, sigsize)) + 
          (__umul24(m, num_dims)) + j] = sum / wgt_k;
  }
}
