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


//Buffers for HyComputation
StructuredBuffer<ELEMTYPE> Hy : register(t0);
StructuredBuffer<ELEMTYPE> Ex : register(t1);
StructuredBuffer<ELEMTYPE> Ez : register(t2);
RWStructuredBuffer<ELEMTYPE> HyOut : register(u0);

//  Hy = Hy+(Dt/mu0)*( (Ez(2:Nx+1, :, :)-Ez(1:Nx, :, :))*Cx - (Ex(:, :, 2:Nz+1)-Ex(:, :, 1:Nz))*Cz );
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void HyComputation( uint3 DTid : SV_DispatchThreadID )
{
	int i = DTid.x;
	int j = DTid.y;
	int k = DTid.z;
	
	int EzIdx1x = 1+i;
	int EzIdx2x = i;

	int ExIdx1z = 1+k;
	int ExIdx2z = k;

	HY_INDEX(HyOut, i, j, k) = HY_INDEX(Hy, i, j, k) + (Dt/mu0) * ( (EZ_INDEX(Ez, EzIdx1x, j, k)-EZ_INDEX(Ez, EzIdx2x, j, k))*Cx - (EX_INDEX(Ex, i, j, ExIdx1z)-EX_INDEX(Ex, i, j, ExIdx2z))*Cz );
}
