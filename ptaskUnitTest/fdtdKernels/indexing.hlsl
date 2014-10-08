
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
