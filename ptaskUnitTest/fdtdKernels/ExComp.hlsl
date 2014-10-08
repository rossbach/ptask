// #include "indexing.hlsl"


#define HX_INDEX(Arr, i, j, k) Arr[i + j*(Nx+1) + k*((Nx+1)*Ny)]
#define HY_INDEX(Arr, i, j, k) Arr[i + j*Nx + k*Nx*(Ny+1)]
#define HZ_INDEX(Arr, i, j, k) Arr[i + j*Nx + k*Nx*Ny]

#define EX_INDEX(Arr, i, j, k) Arr[i + j*Nx + k*Nx*(Ny+1)]
#define EY_INDEX(Arr, i, j, k) Arr[i + j*(Nx+1) + k*((Nx+1)*Ny)]
#define EZ_INDEX(Arr, i, j, k) Arr[i + j*(Nx+1) + k*((Nx+1)*(Ny+1))]

typedef float ELEMTYPE;

cbuffer cbCS : register( b0 )
{
	int Nx;
	int Ny;
	int Nz;
	
	float Cx;
	float Cy;
	float Cz;

	float eps0; 
	float mu0;  
	float c0;   
	float Dt;   
};


//Buffers for HxComputation
StructuredBuffer<ELEMTYPE> Ex : register(t0);
StructuredBuffer<ELEMTYPE> Hy : register(t1);
StructuredBuffer<ELEMTYPE> Hz : register(t2);
RWStructuredBuffer<ELEMTYPE> ExOut : register(u0);

// Ex(:, 2:Ny, 2:Nz) = Ex(:, 2:Ny, 2:Nz)+ (Dt/eps0) * ( (Hz(:, 2:Ny, 2:Nz)-Hz(:, 1:Ny-1, 2:Nz))*Cy - (Hy(:, 2:Ny, 2:Nz)-Hy(:, 2:Ny, 1:Nz-1))*Cz );
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void ExComputation( uint3 DTid : SV_DispatchThreadID )
{
	int i = DTid.x;
	int j = DTid.y;
	int k = DTid.z;
	
	if (j<1 || k<1 || j>Ny-1 || k>Nz-1) {
		//Just copy the input value to the output value
		EX_INDEX(ExOut, i, j, k) = EX_INDEX(Ex, i, j, k);
	}
	else
	{
		int HzIdx1y = j-1;
		int HyIdx1z = k-1;
		EX_INDEX(ExOut, i, j, k) = EX_INDEX(Ex, i, j, k) + (Dt/eps0)*( (HZ_INDEX(Hz, i, j, k)-HZ_INDEX(Hz, i, HzIdx1y, k))*Cy - 
																	   (HY_INDEX(Hy, i, j, k)-HY_INDEX(Hy, i, j, HyIdx1z))*Cz );
	}
}