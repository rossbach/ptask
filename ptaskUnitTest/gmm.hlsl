//--------------------------------------------------------------------------------------
// File: gmm.hlsl
//
// Adapted from:
// Kernel code for Gaussian Mixture Model Expectation Maximization.
// Andrew Harp (andrew.harp@gmail.com)
// http://andrewharp.com/gmmcuda
// 
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

cbuffer cbCS : register(b0)
{
	uint g_num_clusts;
	uint g_num_dims;
	uint g_num_data;
	uint g_num_chunks;
}

StructuredBuffer<float> _frac_ : register(t0);
StructuredBuffer<float> _means_ : register(t1);
StructuredBuffer<float> _sig_ : register(t2);
RWStructuredBuffer<float> _resp_ : register(u0);
RWStructuredBuffer<float> _data_ : register(u1);
RWStructuredBuffer<float> _lndets_ : register(u2);
RWStructuredBuffer<float> _loglike_ : register(u3);

#define MAX_DIM 16
#define BLOCK_SIZE 256
#define NUM_CHUNKS 32

groupshared float lndet_k;
groupshared float frac_k;  
groupshared float chol[MAX_DIM * MAX_DIM];
groupshared float loglike[BLOCK_SIZE];

float 
logdet(
	uint3 blockDim : SV_GroupID,
	uint3 DTid : SV_DispatchThreadID,
	float el[MAX_DIM*MAX_DIM], 
	const unsigned int ndim
	) 
{
  float sum = 0.0f;
  for (unsigned int i=0; i<ndim; ++i) {
    sum += log(el[(i*ndim)+i]);
  }
  return 2.*sum;
}

void 
copyArray(
	uint3 blockDim : SV_GroupID,
	uint3 DTid : SV_DispatchThreadID,
	StructuredBuffer<float> fromArr, 
	uint index,
	float toArr[MAX_DIM * MAX_DIM], 
	unsigned int ndata=BLOCK_SIZE
	) 
{
	unsigned int n;
	unsigned int base_off;  
	for (base_off=0; base_off < ndata; base_off += blockDim.x) {
		n = base_off + DTid.x;    
		if (n < ndata) {
			toArr[n] = fromArr[n+index];
		}
	}
}

float 
parallelSum(
	uint3 blockDim : SV_GroupID,
	uint3 DTid : SV_DispatchThreadID,
	float data[BLOCK_SIZE], 
	const uint ndata
	) 
{
  const uint tid = DTid.x;
  float t;

  GroupMemoryBarrierWithGroupSync();

  // Butterfly sum.  ndata MUST be a power of 2.
  for(unsigned int bit = ndata >> 1; bit > 0; bit >>= 1) {
    t = data[tid] + data[tid^bit];  GroupMemoryBarrierWithGroupSync();
    data[tid] = t;                  GroupMemoryBarrierWithGroupSync();
  }
  return data[tid];
}

void 
cholesky(
	uint3 blockDim : SV_GroupID,
	uint3 DTid : SV_DispatchThreadID,
	float el[MAX_DIM*MAX_DIM], 
	const unsigned int ndim
	) 
{
	const uint tid = DTid.x;
  
	// Dunno how to parallelize this part...
	if (tid == 0) {
		float sum = 0;
    
		uint i, j, k;
		//if (el.ncols() != n) throw("need square matrix");
		for (i=0; i<ndim; i++) {
			for (j=i; j<ndim; j++) {
				sum = el[(i * ndim)+j];
        
				for (k=i-1; k >= 0; k--) {
					sum -= el[(i * ndim)+k] * el[(j * ndim)+k];
				}
        
				if (i == j) {
					//if (sum <= 0.0)
					//	throw("Cholesky failed");
					el[(i * ndim)+i] = sqrt(sum);
				} else {
					el[(j * ndim)+i] = sum/el[(i * ndim)+i];
				}
			}
		}
	}
  
	GroupMemoryBarrierWithGroupSync();
  
	// Clear lower triangular part.
	if ((tid/ndim) < (tid%ndim)) {
		el[((tid/ndim) * ndim) + (tid%ndim)] = 0.0f;
	}
}

void estep_kernel(
	uint3 Gid  : SV_GroupID,
	uint3 DTid : SV_DispatchThreadID
	) 
{
	const uint try_num = Gid.y;
  
	const uint tid = DTid.x;
	const uint k = Gid.x;
	const uint sigsize = g_num_dims * g_num_dims;
  
	// Base offsets.
	const uint rb   = try_num * g_num_clusts * g_num_data + (k*g_num_data); 
	const uint mb   = try_num * g_num_clusts * g_num_dims;
	const uint sb   = try_num * g_num_clusts * sigsize;
	const uint fb   = try_num * g_num_clusts;
	const uint lndb = try_num * g_num_clusts;
  
	uint n, base_off;
	float tmp, sum;
	float v[MAX_DIM];
  

	GroupMemoryBarrierWithGroupSync();
  
	copyArray(Gid, DTid, _sig_, sb + (k * sigsize), chol, sigsize);
  
	cholesky(Gid, DTid, chol, g_num_dims);
  
	GroupMemoryBarrierWithGroupSync();
  
	if (tid == 0) {
		frac_k = _frac_[fb + k];
		lndet_k = logdet(Gid, DTid, chol, g_num_dims);
		_lndets_[lndb + k] = lndet_k;
	}
  
	 GroupMemoryBarrierWithGroupSync();
  
	// Loop through data.
	for (base_off=0; base_off < g_num_data; base_off += BLOCK_SIZE) {
		n = base_off + tid;
		sum=0.0f;
    
		//if (b.size() != n || y.size() != n) throw("bad lengths");
		if (n < g_num_data) {
		  for (unsigned int i=0; i<g_num_dims; ++i) {
			tmp = _data_[(i * g_num_data) + n] - _means_[mb + (k * g_num_dims)+i];
        
			for (unsigned int j=0; j<i; j++) {
			  tmp -= chol[(i * g_num_dims)+j] * v[j];
			}
        
			v[i] = tmp/chol[(i * g_num_dims)+i];
			sum += v[i] * v[i];
		  }

		  // Assign likelihood of this data being in this cluster.
		  _resp_[rb + n] = -0.5f*(sum + lndet_k) + log(frac_k);
		}
	} // (n < g_num_data)
}


void 
estep_normalize_kernel(
	uint3 Gid  : SV_GroupID,
	uint3 DTid : SV_DispatchThreadID
	) 
{
  const unsigned int try_num = Gid.y;
  const unsigned int chunk_num = Gid.x;
  const unsigned int tid = DTid.x;
  
  // We're only handling so many data points per block in this kernel, since
  // data is independant of other data here.
  const unsigned int n_per_block = ceil((float)g_num_data / (float)g_num_chunks);
  const unsigned int start_off = (n_per_block * chunk_num);
  const unsigned int end_off = min(start_off + n_per_block, g_num_data);
  
  // Base offsets.
  const unsigned int rb = (try_num * g_num_clusts) * g_num_data; 
  const unsigned int lb = (try_num * g_num_chunks);  
  unsigned int n, base_off, k;
  float sum, max, tmp;  
  loglike[tid] = 0.0f;
  
  GroupMemoryBarrierWithGroupSync();
  
  // Loop through data.
  for (base_off = start_off; base_off < end_off; base_off += BLOCK_SIZE) {
    n = base_off + tid;
    
    if (n < end_off) {
      max = -99.9e30f;
      
      // Find cluster with maximum likelihood for this data point.
      for (k=0; k<g_num_clusts; ++k) {
        tmp = _resp_[rb + (k*g_num_data) + n];
        if (tmp > max) {
          max = tmp;
        }
      }
      
      // Sum marginal probabilities.
      sum = 0.0f;
      for (k=0; k<g_num_clusts; ++k) {
        sum += exp(_resp_[rb + (k*g_num_data) + n] - max);
      }
      // Assign probabilities of point belonging to each cluster.
      tmp = max + log(sum);
      for (k = 0; k < g_num_clusts; ++k) {
        _resp_[rb + (k*g_num_data) + n] = 
                exp(_resp_[rb + (k*g_num_data) + n] - tmp);
      }
      loglike[tid] += tmp;
    }
  }
  
  tmp = parallelSum(Gid, DTid, loglike, BLOCK_SIZE);
  if (tid == 0) {
    _loglike_[lb + chunk_num] = tmp;
  }
}