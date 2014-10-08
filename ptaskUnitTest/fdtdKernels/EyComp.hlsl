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
StructuredBuffer<ELEMTYPE> Ey : register(t0);
StructuredBuffer<ELEMTYPE> Hx : register(t1);
StructuredBuffer<ELEMTYPE> Hz : register(t2);
RWStructuredBuffer<ELEMTYPE> EyOut : register(u0);

// Ey(2:Nx, :, 2:Nz) = Ey(2:Nx, :, 2:Nz)+(Dt/eps0)* ( (Hx(2:Nx, :, 2:Nz)-Hx(2:Nx, :, 1:Nz-1))*Cz - (Hz(2:Nx, :, 2:Nz)-Hz(1:Nx-1, :, 2:Nz))*Cx );
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void EyComputation( uint3 DTid : SV_DispatchThreadID )
{
	int i = DTid.x;
	int j = DTid.y;
	int k = DTid.z;
	
	if (i<1 || k<1 || i>Nx-1 || k>Nz-1) {
		//Just copy the input value to the output value
		EY_INDEX(EyOut, i, j, k) = EY_INDEX(Ey, i, j, k);
	}
	else
	{
		int HxIdx1z = k-1;
		int HzIdx1x = i-1;
		EY_INDEX(EyOut, i, j, k) = EY_INDEX(Ey, i, j, k) + (Dt/eps0)* ((HX_INDEX(Hx, i, j, k)-HX_INDEX(Hx, i, j, HxIdx1z))*Cz - 
																	   (HZ_INDEX(Hz, i, j, k)-HZ_INDEX(Hz, HzIdx1x, j, k))*Cx );
	}
}
