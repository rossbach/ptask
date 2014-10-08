//fxc /T cs_5_0 fdtd.hlsl

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
StructuredBuffer<ELEMTYPE> Hx : register(t0);
StructuredBuffer<ELEMTYPE> Ey : register(t1);
StructuredBuffer<ELEMTYPE> Ez : register(t2);
RWStructuredBuffer<ELEMTYPE> HxOut : register(u0);

//  Hx = Hx + (Dt/mu0)*( (Ey(:, :, 2:Nz+1)-Ey(:, :, 1:Nz))*Cz - (Ez(:, 2:Ny+1, :)-Ez(:, 1:Ny, :))*Cy );
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void HxComputation( uint3 DTid : SV_DispatchThreadID )
{
	int i = DTid.x;
	int j = DTid.y;
	int k = DTid.z;

	int EyIdx1z = 1+k;
	int EyIdx2z = k;

	int EzIdx1y = 1+j;
	int EzIdx2y = j;

	HX_INDEX(HxOut, i, j, k) = HX_INDEX(Hx, i, j, k) + (Dt/mu0) * ( (EY_INDEX(Ey, i, j, EyIdx1z)-EY_INDEX(Ey, i, j, EyIdx2z))*Cz - (EZ_INDEX(Ez, i, EzIdx1y, k)-EZ_INDEX(Ez, i, EzIdx2y, k))*Cy );
}

