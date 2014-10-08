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
StructuredBuffer<ELEMTYPE> Ez : register(t0);
StructuredBuffer<ELEMTYPE> Hx : register(t1);
StructuredBuffer<ELEMTYPE> Hy : register(t2);
RWStructuredBuffer<ELEMTYPE> EzOut : register(u0);


// Ez(2:Nx, 2:Ny, :) = Ez(2:Nx, 2:Ny, :)+(Dt/eps0)*((Hy(2:Nx, 2:Ny, :)-Hy(1:Nx-1, 2:Ny, :))*Cx - (Hx(2:Nx, 2:Ny, :)-Hx(2:Nx, 1:Ny-1, :))*Cy);
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void EzComputation( uint3 DTid : SV_DispatchThreadID )
{
	int i = DTid.x;
	int j = DTid.y;
	int k = DTid.z;
	
	if (i<1 || j<1 || i>Nx-1 || j>Ny-1) {
		//Just copy the input value to the output value
		EZ_INDEX(EzOut, i, j, k) = EZ_INDEX(Ez, i, j, k);
	}
	else
	{
		int HyIdx1x = i-1;
		int HxIdx1y = j-1;
		EZ_INDEX(EzOut, i, j, k) = EZ_INDEX(Ez, i, j, k) + (Dt/eps0)*((HY_INDEX(Hy, i, j, k)-HY_INDEX(Hy, HyIdx1x, j, k))*Cx - 
																	  (HX_INDEX(Hx, i, j, k)-HX_INDEX(Hx, i, HxIdx1y, k))*Cy);
	}
}
