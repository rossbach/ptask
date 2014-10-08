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


//Buffers for HzComputation
StructuredBuffer<ELEMTYPE> Hz : register(t0);
StructuredBuffer<ELEMTYPE> Ex : register(t1);
StructuredBuffer<ELEMTYPE> Ey : register(t2);
RWStructuredBuffer<ELEMTYPE> HzOut : register(u0);

// Hz = Hz+(Dt/mu0)*( (Ex(:, 2:Ny+1, :)-Ex(:, 1:Ny, :))*Cy - (Ey(2:Nx+1, :, :)-Ey(1:Nx, :, :))*Cx );
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void HzComputation( uint3 DTid : SV_DispatchThreadID )
{
	int i = DTid.x;
	int j = DTid.y;
	int k = DTid.z;

	int ExIdx1y = 1+j;
	int ExIdx2y = j;

	int EyIdx1x = 1+i;
	int EyIdx2x = i;

	HZ_INDEX(HzOut, i, j, k) = HZ_INDEX(Hz, i, j, k) + (Dt/mu0)*( (EX_INDEX(Ex, i, ExIdx1y, k)-EX_INDEX(Ex, i, ExIdx2y, k))*Cy - (EY_INDEX(Ey, EyIdx1x, j, k)-EY_INDEX(Ey, EyIdx2x, j, k))*Cx );
}

