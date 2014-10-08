#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <crtdbg.h>
#include <ctime>
#include <sstream>
#include "accelerator.h"
#include "assert.h"
#include "shaderparms.h"
#include "sort.h"
#include <vector>
#include <map>
#include <algorithm>
#include <set>
#include "matrixtask.h"
#include "SimpleMatrix.h"
#include "Simple3DArray.h"
#include "matmul.h"
#include "graphmatmul.h"
#include "graphfdtd.h"
#include "elemtype.h"
#include "platformcheck.h"
#include "ptaskapi.h"
#include "confighelpers.h"

using namespace std;
using namespace PTask;

extern BOOL g_bSingleThreaded;
extern BOOL g_bEmulateModular;
extern int g_nChannelCapacity;
extern BOOL g_bSetChannelCapacity; 

struct FDTDGraphParams
{
	Graph *g;

	DatablockTemplate *pConstParamTemplate;
	std::map<std::string, DatablockTemplate*> varToTemplateMap;
	std::map<std::string, IterationSpaceSize> varToSizeMap;
		
	//The ports which we need to use as inputs to connect the next iteration of the loop body
	std::map<std::string, Port*> varToPortMap;
	
	//The channels that connect the graph to the outside world
	std::map<std::string, std::vector<GraphInputChannel*> > inputChannels; //Each input may be needed by more than one kernel
	std::map<std::string, GraphOutputChannel*> outputChannels;
	std::vector< GraphInputChannel*> paramInputs; //Channels for the constant configuration input (one per node)

	int Nx;
	int Ny;
	int Nz;

	UINT uiUidCounter;

	FDTDGraphParams(int x, int y, int z, std::string dir, KernelType ktype);
	
	DatablockTemplate* findTemplate(const std::string& varName);
	IterationSpaceSize findVarSize (const std::string& varName);

	std::string dirName;
	KernelType kernelType;
};

FDTDGraphParams::FDTDGraphParams(int x, int y, int z, std::string dir, KernelType ktype)
	:Nx(x), Ny(y), Nz(z), uiUidCounter(0), dirName(dir), kernelType(ktype)
{
	int stride = sizeof(float);

	varToTemplateMap["Hx"] = PTask::Runtime::GetDatablockTemplate("HxSize", stride, Nx+1, Ny, Nz);
	varToSizeMap["Hx"] = IterationSpaceSize(Nx+1, Ny, Nz);

	varToTemplateMap["Hy"] = PTask::Runtime::GetDatablockTemplate("HySize", stride, Nx, Ny+1, Nz);
	varToSizeMap["Hy"] = IterationSpaceSize(Nx, Ny+1, Nz);

	varToTemplateMap["Hz"] = PTask::Runtime::GetDatablockTemplate("HzSize", stride, Nx, Ny, Nz+1);
	varToSizeMap["Hz"] = IterationSpaceSize(Nx, Ny, Nz+1);

	varToTemplateMap["Ex"] = PTask::Runtime::GetDatablockTemplate("ExSize", stride, Nx, Ny+1, Nz+1);
	varToSizeMap["Ex"] = IterationSpaceSize(Nx, Ny+1, Nz+1);

	varToTemplateMap["Ey"] = PTask::Runtime::GetDatablockTemplate("EySize", stride, Nx+1, Ny, Nz+1);
	varToSizeMap["Ey"] = IterationSpaceSize(Nx+1, Ny, Nz+1);

	varToTemplateMap["Ez"] = PTask::Runtime::GetDatablockTemplate("EzSize", stride, Nx+1, Ny+1, Nz);
	varToSizeMap["Ez"] = IterationSpaceSize(Nx+1, Ny+1, Nz);

	pConstParamTemplate = PTask::Runtime::GetDatablockTemplate("fdtdparms", sizeof(FDTD_PARAMS), PTPARM_BYVALSTRUCT);
}

DatablockTemplate* FDTDGraphParams::findTemplate(const std::string& varName)
{
	std::map<std::string, DatablockTemplate*>::iterator iter = varToTemplateMap.find(varName);
	assert (iter!=varToTemplateMap.end());
	return iter->second;
}

IterationSpaceSize FDTDGraphParams::findVarSize(const std::string& varName)
{
	std::map<std::string, IterationSpaceSize>::iterator iter = varToSizeMap.find(varName);
	assert (iter!=varToSizeMap.end());
	return iter->second;
}

#pragma region ARRAY_FUNCTIONS
void configure_raw_array(
	int x,
	int y,
	int z,
	CSimple3DArray<float> **arr,
	int v)
{
	srand ( static_cast<int>(time(NULL)) );
	CSimple3DArray<float> *pA = new CSimple3DArray<float>(x, y, z);
	for(int i=0; i<x ; ++i) {
		for(int j=0; j<y ; ++j) {
			for(int k=0; k<z ; ++k) {
				if( v == -1 )
					pA->v(i, j, k) = (float)rand()/(RAND_MAX*float(10000.0));
				else
					pA->v(i, j, k) = (float)v;
			}
		}
	}
	*arr = pA;
}

void copyArray(CSimple3DArray<float>*& dst, CSimple3DArray<float>* src)
{
	int x=src->dimension1(), y=src->dimension2(), z=src->dimension3();
	dst = new CSimple3DArray<float>(x, y, z);
	memcpy(dst->cells(), src->cells(), x*y*z*sizeof(float));
}

void print_array(CSimple3DArray<float>* arr,
				 char *arrName)
{
	std::cout << arrName << " :\n";
	for (int i=0; i<arr->dimension1() ; ++i) {
		for (int j=0; j<arr->dimension2() ; ++j) {
			for (int k=0; k<arr->dimension3() ; ++k) {
				std::cout << arr->v(i, j, k) << " ";	
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

bool check_array_result(CSimple3DArray<float> *a, 
						CSimple3DArray<float> *b,
						int *pErrorTolerance,
                        double * pTotalSquareError=NULL)
{
	//assert (pErrorTolerance == NULL);
	int numErrors=0;
	const double allowedErrorPercentage = 0.2;
	double totalSqError=0;
	for (int i=0; i<a->dimension1() ; ++i) {
		for (int j=0; j<a->dimension2() ; ++j) {
			for (int k=0; k<a->dimension3() ; ++k) {
				totalSqError += (a->v(i, j, k) - b->v(i, j, k)) * (a->v(i, j, k) - b->v(i, j, k));
				double percentError = fabs(a->v(i, j, k) - b->v(i, j, k))/a->v(i, j, k) ;
				if (percentError > allowedErrorPercentage)
				{
					// std::cout << percentError << " : " << a->v(i, j, k) << " " <<  b->v(i, j, k) << std::endl;
					++numErrors;
				}
			}
		}
	}

	std::cout << "Total square error : " << totalSqError << std::endl;
    if(pTotalSquareError != NULL)
        *pTotalSquareError += totalSqError;
	return numErrors==0;
}
#pragma endregion ARRAY_FUNCTIONS

#pragma region FDTD_CPU

typedef CSimple3DArray<float>* (*CPUFunctionType)(CSimple3DArray<float>*, CSimple3DArray<float>*, CSimple3DArray<float>*, FDTD_PARAMS);

//  Hx = Hx + (Dt/mu0)*( (Ey(:, :, 2:Nz+1)-Ey(:, :, 1:Nz))*Cz - (Ez(:, 2:Ny+1, :)-Ez(:, 1:Ny, :))*Cy );
CSimple3DArray<float>* fdtd_HxComputation_cpu(CSimple3DArray<float>* Hx, CSimple3DArray<float>* Ey, CSimple3DArray<float>* Ez, FDTD_PARAMS params)
{
	CSimple3DArray<float> *Hx_out = new CSimple3DArray<float>(params.Nx+1, params.Ny, params.Nz);
	for (int i=0; i<Hx->dimension1() ; ++i) {
		for (int j=0; j<Hx->dimension2() ; ++j) {
			for (int k=0 ; k<Hx->dimension3() ; ++k) {
				int EyIdx1z = 1+k;
				int EyIdx2z = k;

				int EzIdx1y = 1+j;
				int EzIdx2y = j;
				Hx_out->v(i, j, k) = Hx->v(i, j, k) + (params.Dt/params.mu0)*((Ey->v(i, j, EyIdx1z)-Ey->v(i, j, EyIdx2z))*params.Cz - (Ez->v(i, EzIdx1y, k)-Ez->v(i, EzIdx2y, k))*params.Cy);
			}
		}
	}
	return Hx_out;
}

//  Hy = Hy+(Dt/mu0)*( (Ez(2:Nx+1, :, :)-Ez(1:Nx, :, :))*Cx - (Ex(:, :, 2:Nz+1)-Ex(:, :, 1:Nz))*Cz );
CSimple3DArray<float>* fdtd_HyComputation_cpu(CSimple3DArray<float>* Hy, CSimple3DArray<float>* Ex, CSimple3DArray<float>* Ez, FDTD_PARAMS params)
{
	CSimple3DArray<float> *HyOut = new CSimple3DArray<float>(params.Nx, params.Ny+1, params.Nz);
	for (int i=0; i<Hy->dimension1() ; ++i) {
		for (int j=0; j<Hy->dimension2() ; ++j) {
			for (int k=0 ; k<Hy->dimension3() ; ++k) {
				int EzIdx1x = 1+i;
				int EzIdx2x = i;

				int ExIdx1z = 1+k;
				int ExIdx2z = k;

				HyOut->v(i, j, k) = Hy->v(i, j, k) + (params.Dt/params.mu0) * ( (Ez->v(EzIdx1x, j, k)-Ez->v(EzIdx2x, j, k))*params.Cx - (Ex->v(i, j, ExIdx1z)-Ex->v(i, j, ExIdx2z))*params.Cz);
			}
		}
	}
	return HyOut;
}

CSimple3DArray<float>* fdtd_HzComputation_cpu(CSimple3DArray<float>* Hz, CSimple3DArray<float>* Ex, CSimple3DArray<float>* Ey, FDTD_PARAMS params)
{
	CSimple3DArray<float> *HzOut = new CSimple3DArray<float>(params.Nx, params.Ny, params.Nz+1);
	for (int i=0; i<Hz->dimension1() ; ++i) {
		for (int j=0; j<Hz->dimension2() ; ++j) {
			for (int k=0 ; k<Hz->dimension3() ; ++k) {
				int ExIdx1y = 1+j;
				int ExIdx2y = j;

				int EyIdx1x = 1+i;
				int EyIdx2x = i;

				HzOut->v(i, j, k) = Hz->v(i, j, k) + (params.Dt/params.mu0)*( (Ex->v(i, ExIdx1y, k)-Ex->v(i, ExIdx2y, k))*params.Cy - (Ey->v(EyIdx1x, j, k)-Ey->v(EyIdx2x, j, k))*params.Cx );
			}
		}
	}
	return HzOut;
}

// Ex(:, 2:Ny, 2:Nz) = Ex(:, 2:Ny, 2:Nz)+ (Dt/eps0) * ( (Hz(:, 2:Ny, 2:Nz)-Hz(:, 1:Ny-1, 2:Nz))*Cy - (Hy(:, 2:Ny, 2:Nz)-Hy(:, 2:Ny, 1:Nz-1))*Cz );
CSimple3DArray<float>* fdtd_ExComputation_cpu(CSimple3DArray<float>* Ex, CSimple3DArray<float>* Hy, CSimple3DArray<float>* Hz, FDTD_PARAMS params)
{
	CSimple3DArray<float> *ExOut = new CSimple3DArray<float>(params.Nx, params.Ny+1, params.Nz+1);
	for (int i=0; i<Ex->dimension1() ; ++i) {
		for (int j=0; j<Ex->dimension2() ; ++j) {
			for (int k=0 ; k<Ex->dimension3() ; ++k) {
				if (j<1 || k<1 || j>params.Ny-1 || k>params.Nz-1) {
					//Just copy the input value to the output value
					ExOut->v(i, j, k) = Ex->v(i, j, k);
				}
				else
				{
					int HzIdx1y = j-1;
					int HyIdx1z = k-1;
					ExOut->v(i, j, k) = Ex->v(i, j, k) + (params.Dt/params.eps0)*( (Hz->v(i, j, k)-Hz->v(i, HzIdx1y, k))*params.Cy - 
																				   (Hy->v(i, j, k)-Hy->v(i, j, HyIdx1z))*params.Cz );
				}
			}
		}
	}
	return ExOut;
}

// Ey(2:Nx, :, 2:Nz) = Ey(2:Nx, :, 2:Nz)+(Dt/eps0)* ( (Hx(2:Nx, :, 2:Nz)-Hx(2:Nx, :, 1:Nz-1))*Cz - (Hz(2:Nx, :, 2:Nz)-Hz(1:Nx-1, :, 2:Nz))*Cx );
CSimple3DArray<float>* fdtd_EyComputation_cpu(CSimple3DArray<float>* Ey, CSimple3DArray<float>* Hx, CSimple3DArray<float>* Hz, FDTD_PARAMS params)
{
	CSimple3DArray<float> *EyOut = new CSimple3DArray<float>(params.Nx+1, params.Ny, params.Nz+1);
	for (int i=0; i<Ey->dimension1() ; ++i) {
		for (int j=0; j<Ey->dimension2() ; ++j) {
			for (int k=0 ; k<Ey->dimension3() ; ++k) {
				if (i<1 || k<1 || i>params.Nx-1 || k>params.Nz-1) {
					//Just copy the input value to the output value
					EyOut->v(i, j, k) = Ey->v(i, j, k);
				}
				else
				{
					int HxIdx1z = k-1;
					int HzIdx1x = i-1;
					EyOut->v(i, j, k) = Ey->v(i, j, k) + (params.Dt/params.eps0)* ((Hx->v(i, j, k) - Hx->v(i, j, HxIdx1z))*params.Cz - 
																				   (Hz->v(i, j, k) - Hz->v(HzIdx1x, j, k))*params.Cx );
				}
			}
		}
	}
	return EyOut;
}

// Ez(2:Nx, 2:Ny, :) = Ez(2:Nx, 2:Ny, :)+(Dt/eps0)*((Hy(2:Nx, 2:Ny, :)-Hy(1:Nx-1, 2:Ny, :))*Cx - (Hx(2:Nx, 2:Ny, :)-Hx(2:Nx, 1:Ny-1, :))*Cy);
CSimple3DArray<float>* fdtd_EzComputation_cpu(CSimple3DArray<float>* Ez, CSimple3DArray<float>* Hx, CSimple3DArray<float>* Hy, FDTD_PARAMS params)
{
	CSimple3DArray<float> *EzOut = new CSimple3DArray<float>(params.Nx+1, params.Ny+1, params.Nz);
	for (int i=0; i<Ez->dimension1() ; ++i) {
		for (int j=0; j<Ez->dimension2() ; ++j) {
			for (int k=0 ; k<Ez->dimension3() ; ++k) {
				if (i<1 || j<1 || i>params.Nx-1 || j>params.Ny-1) {
					//Just copy the input value to the output value
					EzOut->v(i, j, k) = Ez->v(i, j, k);
				}
				else
				{
					int HyIdx1x = i-1;
					int HxIdx1y = j-1;
					EzOut->v(i, j, k) = Ez->v(i, j, k) + (params.Dt/params.eps0)*((Hy->v(i, j, k) - Hy->v(HyIdx1x, j, k))*params.Cx - 
																				  (Hx->v(i, j, k) - Hx->v(i, HxIdx1y, k))*params.Cy);
				}
			}
		}
	}
	return EzOut;
}

void fdtd_cpu(CSimple3DArray<float>*& Hx, CSimple3DArray<float>*& Hy, CSimple3DArray<float>*& Hz, 
	  		  CSimple3DArray<float>*& Ex, CSimple3DArray<float>*& Ey, CSimple3DArray<float>*& Ez, 
			  int numIterations, FDTD_PARAMS params)
{
	for(int i=0; i<numIterations ; ++i)
	{
		CSimple3DArray<float>* HxOut = fdtd_HxComputation_cpu(Hx, Ey, Ez, params);
		delete Hx;
		Hx = HxOut;
		CSimple3DArray<float>* HyOut = fdtd_HyComputation_cpu(Hy, Ex, Ez, params);
		delete Hy;
		Hy = HyOut;
		CSimple3DArray<float>* HzOut = fdtd_HzComputation_cpu(Hz, Ex, Ey, params);
		delete Hz;
		Hz = HzOut;

		CSimple3DArray<float>* ExOut = fdtd_ExComputation_cpu(Ex, Hy, Hz, params);
		delete Ex;
		Ex = ExOut;
		CSimple3DArray<float>* EyOut = fdtd_EyComputation_cpu(Ey, Hx, Hz, params);
		delete Ey;
		Ey = EyOut;
		CSimple3DArray<float>* EzOut = fdtd_EzComputation_cpu(Ez, Hx, Hy, params);
		delete Ez;
		Ez = EzOut;
	}
}

inline void sdelete(CSimple3DArray<float>*& p) { if(p != NULL) {  delete p; p = NULL; } }

void fdtd_cpu_multiple_inputs(CSimple3DArray<float>*& Hx, CSimple3DArray<float>*& Hy, CSimple3DArray<float>*& Hz, 
	  						  CSimple3DArray<float>*& Ex, CSimple3DArray<float>*& Ey, CSimple3DArray<float>*& Ez, 
							  int numIterations, int numBlocks, FDTD_PARAMS params)
{
	CSimple3DArray<float>* HxIn = NULL;
	CSimple3DArray<float>* HyIn = NULL;
	CSimple3DArray<float>* HzIn = NULL;
	CSimple3DArray<float>* ExIn = NULL;
	CSimple3DArray<float>* EyIn = NULL;
	CSimple3DArray<float>* EzIn = NULL;

	copyArray(HxIn, Hx);   sdelete(Hx);
	copyArray(HyIn, Hy);   sdelete(Hy);
	copyArray(HzIn, Hz);   sdelete(Hz);

	copyArray(ExIn, Ex);   sdelete(Ex);
	copyArray(EyIn, Ey);   sdelete(Ey);
	copyArray(EzIn, Ez);   sdelete(Ez);

	for(int i=1; i<=numBlocks ; ++i)
	{
		copyArray(Hx, HxIn);
		copyArray(Hy, HyIn);
		copyArray(Hz, HzIn);

		copyArray(Ex, ExIn);
		copyArray(Ey, EyIn);
		copyArray(Ez, EzIn);

		fdtd_cpu(Hx, Hy, Hz, Ex, Ey, Ez, numIterations, params);

		if(i!=numBlocks)
		{
			sdelete(Hx); 
			sdelete(Hy);
			sdelete(Hz);
			sdelete(Ex);
			sdelete(Ey);
			sdelete(Ez);
		}
	}
    sdelete(HxIn); 
    sdelete(HyIn);
    sdelete(HzIn);
    sdelete(ExIn);
    sdelete(EyIn);
    sdelete(EzIn);
}
#pragma endregion FDTD_CPU



#pragma region GRAPH_CONSTRUCTION
//Construct a task to compute one of the field components (Ex, Ey, Ez etc)
Task* contructComputationTask(const std::string& fileName, 
							  const std::string& kernelName,
							  const std::string& taskName,
							  IterationSpaceSize iterationSpace,
							  std::vector<std::string>& inputVars,
							  std::vector<std::string>& outputVars,
							  Port** pInputPorts, int numInputPorts,
							  Port** pOutputPorts, int numOutputPorts,
							  std::vector<DatablockTemplate*>& inputTemplates, DatablockTemplate* &outputTemplate,
							  FDTDGraphParams& graphParams)
{
	CompiledKernel * pKernel = COMPILE_KERNEL(const_cast<char*>(fileName.c_str()), const_cast<char*>(kernelName.c_str()));

	assert (numInputPorts  == inputVars.size() + 1); //One configuration input is needed 
	assert (outputVars.size() == 1);
	assert (numOutputPorts == 1);
	
	for(int i=0; i<numInputPorts-1 ; ++i)
	{
		DatablockTemplate *inputTemplate = graphParams.findTemplate(inputVars.at(i));
		inputTemplates.push_back(inputTemplate);
		pInputPorts[i] = PTask::Runtime::CreatePort(INPUT_PORT, inputTemplate, graphParams.uiUidCounter++, const_cast<char*>(inputVars.at(i).c_str()), i); 
	}
	pInputPorts[numInputPorts-1] = PTask::Runtime::CreatePort(STICKY_PORT, graphParams.pConstParamTemplate, graphParams.uiUidCounter++, "Config", numInputPorts-1);

	outputTemplate = graphParams.findTemplate(outputVars.at(0));
	pOutputPorts[0]	= PTask::Runtime::CreatePort(OUTPUT_PORT, outputTemplate, graphParams.uiUidCounter++, const_cast<char*>(outputVars.at(0).c_str()), numInputPorts);

	Task * pKernelTask = graphParams.g->AddTask(pKernel, 
									  numInputPorts,
									  pInputPorts,
									  numOutputPorts,
									  pOutputPorts,
									  const_cast<char*>(taskName.c_str()));
	assert (pKernelTask);

	pKernelTask->SetComputeGeometry(iterationSpace.dim1, iterationSpace.dim2, iterationSpace.dim3);
	return pKernelTask;
}


inline void computeGridAndBlockSizes(const std::string& varName, FDTDGraphParams& probParams, PTASKDIM3& threadBlockSize, PTASKDIM3& gridSize)
{
	int numThreadsPerBlock = 32;
	IterationSpaceSize iterSpace = probParams.findVarSize(varName);
	threadBlockSize = PTASKDIM3(numThreadsPerBlock, 1, 1);
	gridSize = PTASKDIM3((iterSpace.dim1+numThreadsPerBlock-1)/numThreadsPerBlock, iterSpace.dim2*iterSpace.dim3, 1);
	return;
}

void constructSingleIterationGraph(FDTDGraphParams& graphParams, int i, bool lastIteration)
{
	//First construct a new iteration
	std::string dirName = graphParams.dirName+"\\";
	IterationSpaceSize iterSpace;
	std::vector<std::string> inputVars, outputVars;
	int numInputPorts=4, numOutputPorts=1;
	std::string fileName;
	PTASKDIM3 gridSize, blockSize;
	
	std::vector<std::string> inputVarsHx, outputVarsHx;
	inputVarsHx.push_back("Hx"); inputVarsHx.push_back("Ey"); inputVarsHx.push_back("Ez");
	outputVarsHx.push_back("Hx");
	Port **inputPortsHx = new Port*[numInputPorts];
	Port **outputPortsHx = new Port*[numOutputPorts];
	std::vector<DatablockTemplate*> inputTemplatesHx;
	DatablockTemplate *outputTemplatesHx;
	
	if (graphParams.kernelType == HLSL)
		fileName = "HxComp.hlsl";
	else if(graphParams.kernelType == CUDA)
		fileName = "fdtdMain.compute_10.ptx";
	else
		assert (0);

	Task *HxTask = contructComputationTask(dirName+fileName, "HxComputation", "HxComTask", graphParams.findVarSize("Hx"),
										   inputVarsHx, outputVarsHx, inputPortsHx, numInputPorts, outputPortsHx, numOutputPorts, inputTemplatesHx, outputTemplatesHx, graphParams);
	
	if(graphParams.kernelType == CUDA)
	{
		computeGridAndBlockSizes("Hx", graphParams, blockSize, gridSize);
		HxTask->SetBlockAndGridSize(gridSize, blockSize);
	}

	std::vector<std::string> inputVarsHy, outputVarsHy;
	inputVarsHy.push_back("Hy"); inputVarsHy.push_back("Ex"); inputVarsHy.push_back("Ez");
	outputVarsHy.push_back("Hy");
	Port **inputPortsHy = new Port*[numInputPorts];
	Port **outputPortsHy = new Port*[numOutputPorts];
	std::vector<DatablockTemplate*> inputTemplatesHy;
	DatablockTemplate *outputTemplatesHy;

	if (graphParams.kernelType == HLSL)
		fileName = "HyComp.hlsl";
	else if(graphParams.kernelType == CUDA)
		fileName = "fdtdMain.compute_10.ptx";
	else
		assert (0);

	Task *HyTask = contructComputationTask(dirName+fileName, "HyComputation", "HyComTask", graphParams.findVarSize("Hy"),
										   inputVarsHy, outputVarsHy, inputPortsHy, numInputPorts, outputPortsHy, numOutputPorts, inputTemplatesHy, outputTemplatesHy, graphParams);

	if(graphParams.kernelType == CUDA)
	{
		computeGridAndBlockSizes("Hy", graphParams, blockSize, gridSize);
		HyTask->SetBlockAndGridSize(gridSize, blockSize);
	}

	std::vector<std::string> inputVarsHz, outputVarsHz;
	inputVarsHz.push_back("Hz"); inputVarsHz.push_back("Ex"); inputVarsHz.push_back("Ey");
	outputVarsHz.push_back("Hz");
	Port **inputPortsHz = new Port*[numInputPorts];
	Port **outputPortsHz = new Port*[numOutputPorts];
	std::vector<DatablockTemplate*> inputTemplatesHz;
	DatablockTemplate *outputTemplatesHz;
	if (graphParams.kernelType == HLSL)
		fileName = "HzComp.hlsl";
	else if(graphParams.kernelType == CUDA)
		fileName = "fdtdMain.compute_10.ptx";
	else
		assert (0);

	Task *HzTask = contructComputationTask(dirName+fileName, "HzComputation", "HzComTask", graphParams.findVarSize("Hz"),
										   inputVarsHz, outputVarsHz, inputPortsHz, numInputPorts, outputPortsHz, numOutputPorts, inputTemplatesHz, outputTemplatesHz, graphParams);


	if(graphParams.kernelType == CUDA)
	{
		computeGridAndBlockSizes("Hz", graphParams, blockSize, gridSize);
		HzTask->SetBlockAndGridSize(gridSize, blockSize);
	}

	std::vector<std::string> inputVarsEx, outputVarsEx;
	inputVarsEx.push_back("Ex"); inputVarsEx.push_back("Hy"); inputVarsEx.push_back("Hz");
	outputVarsEx.push_back("Ex");
	Port **inputPortsEx = new Port*[numInputPorts];
	Port **outputPortsEx = new Port*[numOutputPorts];
	std::vector<DatablockTemplate*> inputTemplatesEx;
	DatablockTemplate *outputTemplatesEx;
	if (graphParams.kernelType == HLSL)
		fileName = "ExComp.hlsl";
	else if(graphParams.kernelType == CUDA)
		fileName = "fdtdMain.compute_10.ptx";
	else
		assert (0);

	Task *ExTask = contructComputationTask(dirName+fileName, "ExComputation", "ExComTask", graphParams.findVarSize("Ex"),
										   inputVarsEx, outputVarsEx, inputPortsEx, numInputPorts, outputPortsEx, numOutputPorts, inputTemplatesEx, outputTemplatesEx, graphParams);

	if(graphParams.kernelType == CUDA)
	{
		computeGridAndBlockSizes("Ex", graphParams, blockSize, gridSize);
		ExTask->SetBlockAndGridSize(gridSize, blockSize);
	}

	std::vector<std::string> inputVarsEy, outputVarsEy;
	inputVarsEy.push_back("Ey"); inputVarsEy.push_back("Hx"); inputVarsEy.push_back("Hz");
	outputVarsEy.push_back("Ey");
	Port **inputPortsEy = new Port*[numInputPorts];
	Port **outputPortsEy = new Port*[numOutputPorts];
	std::vector<DatablockTemplate*> inputTemplatesEy;
	DatablockTemplate *outputTemplatesEy;
	if (graphParams.kernelType == HLSL)
		fileName = "EyComp.hlsl";
	else if(graphParams.kernelType == CUDA)
		fileName = "fdtdMain.compute_10.ptx";
	else
		assert (0);

	Task *EyTask = contructComputationTask(dirName+fileName, "EyComputation", "EyComTask", graphParams.findVarSize("Ey"),
										   inputVarsEy, outputVarsEy, inputPortsEy, numInputPorts, outputPortsEy, numOutputPorts, inputTemplatesEy, outputTemplatesEy, graphParams);

	if(graphParams.kernelType == CUDA)
	{
		computeGridAndBlockSizes("Ey", graphParams, blockSize, gridSize);
		EyTask->SetBlockAndGridSize(gridSize, blockSize);
	}

	std::vector<std::string> inputVarsEz, outputVarsEz;
	inputVarsEz.push_back("Ez"); inputVarsEz.push_back("Hx"); inputVarsEz.push_back("Hy");
	outputVarsEz.push_back("Ez");
	Port **inputPortsEz = new Port*[numInputPorts];
	Port **outputPortsEz = new Port*[numOutputPorts];
	std::vector<DatablockTemplate*> inputTemplatesEz;
	DatablockTemplate *outputTemplatesEz;
	if (graphParams.kernelType == HLSL)
		fileName = "EzComp.hlsl";
	else if(graphParams.kernelType == CUDA)
		fileName = "fdtdMain.compute_10.ptx";
	else
		assert (0);

	Task *EzTask = contructComputationTask(dirName+fileName, "EzComputation", "EzComTask", graphParams.findVarSize("Ez"),
										   inputVarsEz, outputVarsEz, inputPortsEz, numInputPorts, outputPortsEz, numOutputPorts, inputTemplatesEz, outputTemplatesEz, graphParams);

	if(graphParams.kernelType == CUDA)
	{
		computeGridAndBlockSizes("Ez", graphParams, blockSize, gridSize);
		EzTask->SetBlockAndGridSize(gridSize, blockSize);
	}

    std::string unq("");
    if(i != 0) {
        char sz[10];
        sprintf_s(sz, 10, "_%d", i);
        unq += sz;
    }

	//Connect the output of HxComp to Ey and Ez
	graphParams.g->AddInternalChannel(outputPortsHx[0], inputPortsEy[1], ((char*)std::string("HxToEy").append(unq).c_str()));
	graphParams.g->AddInternalChannel(outputPortsHx[0], inputPortsEz[1], ((char*)std::string("HxToEz").append(unq).c_str()));

    //Connect the output of HyComp to Ex and Ez
	graphParams.g->AddInternalChannel(outputPortsHy[0], inputPortsEx[1], ((char*)std::string("HyToEx").append(unq).c_str()));
	graphParams.g->AddInternalChannel(outputPortsHy[0], inputPortsEz[2], ((char*)std::string("HyToEz").append(unq).c_str()));
	//Connect the output of HzComp to Ex and Ey
	graphParams.g->AddInternalChannel(outputPortsHz[0], inputPortsEx[2], ((char*)std::string("HzToEx").append(unq).c_str()));
	graphParams.g->AddInternalChannel(outputPortsHz[0], inputPortsEy[2], ((char*)std::string("HzToEy").append(unq).c_str()));

	graphParams.paramInputs.push_back(graphParams.g->AddInputChannel(inputPortsHx[3], ((char*)std::string("ConstParamInputHx").append(unq).c_str())));
	graphParams.paramInputs.push_back(graphParams.g->AddInputChannel(inputPortsHy[3], ((char*)std::string("ConstParamInputHy").append(unq).c_str())));
	graphParams.paramInputs.push_back(graphParams.g->AddInputChannel(inputPortsHz[3], ((char*)std::string("ConstParamInputHz").append(unq).c_str())));
	graphParams.paramInputs.push_back(graphParams.g->AddInputChannel(inputPortsEx[3], ((char*)std::string("ConstParamInputEx").append(unq).c_str())));
	graphParams.paramInputs.push_back(graphParams.g->AddInputChannel(inputPortsEy[3], ((char*)std::string("ConstParamInputEy").append(unq).c_str())));
	graphParams.paramInputs.push_back(graphParams.g->AddInputChannel(inputPortsEz[3], ((char*)std::string("ConstParamInputEz").append(unq).c_str())));

	//Connect it up correctly using the map
	if(graphParams.varToPortMap.size() == 0)
	{
		GraphInputChannel *HxCompHxInputChannel = graphParams.g->AddInputChannel(inputPortsHx[0], ((char*)std::string("HxCompHxInput").append(unq).c_str()));
		GraphInputChannel *HyCompHyInputChannel = graphParams.g->AddInputChannel(inputPortsHy[0], ((char*)std::string("HyCompHyInput").append(unq).c_str()));
		GraphInputChannel *HzCompHzInputChannel = graphParams.g->AddInputChannel(inputPortsHz[0], ((char*)std::string("HzCompHzInput").append(unq).c_str()));

		graphParams.inputChannels["Hx"].push_back(HxCompHxInputChannel);
		graphParams.inputChannels["Hy"].push_back(HyCompHyInputChannel);
		graphParams.inputChannels["Hz"].push_back(HzCompHzInputChannel);

		GraphInputChannel *HxCompEyInputChannel = graphParams.g->AddInputChannel(inputPortsHx[1], ((char*)std::string("HxCompEyInput").append(unq).c_str()));
		GraphInputChannel *HxCompEzInputChannel = graphParams.g->AddInputChannel(inputPortsHx[2], ((char*)std::string("HxCompEzInput").append(unq).c_str()));

		GraphInputChannel *HyCompExInputChannel = graphParams.g->AddInputChannel(inputPortsHy[1], ((char*)std::string("HyCompExInput").append(unq).c_str()));
		GraphInputChannel *HyCompEzInputChannel = graphParams.g->AddInputChannel(inputPortsHy[2], ((char*)std::string("HyCompEzInput").append(unq).c_str()));

		GraphInputChannel *HzCompExInputChannel = graphParams.g->AddInputChannel(inputPortsHz[1], ((char*)std::string("HzCompExInput").append(unq).c_str()));
		GraphInputChannel *HzCompEyInputChannel = graphParams.g->AddInputChannel(inputPortsHz[2], ((char*)std::string("HzCompEyInput").append(unq).c_str()));

		graphParams.inputChannels["Ex"].push_back(HyCompExInputChannel); graphParams.inputChannels["Ex"].push_back(HzCompExInputChannel);
		graphParams.inputChannels["Ey"].push_back(HxCompEyInputChannel); graphParams.inputChannels["Ey"].push_back(HzCompEyInputChannel);
		graphParams.inputChannels["Ez"].push_back(HxCompEzInputChannel); graphParams.inputChannels["Ez"].push_back(HyCompEzInputChannel);

		GraphInputChannel *ExCompExInputChannel = graphParams.g->AddInputChannel(inputPortsEx[0], ((char*)std::string("ExCompExInput").append(unq).c_str()));
		GraphInputChannel *EyCompEyInputChannel = graphParams.g->AddInputChannel(inputPortsEy[0], ((char*)std::string("EyCompEyInput").append(unq).c_str()));
		GraphInputChannel *EzCompEzInputChannel = graphParams.g->AddInputChannel(inputPortsEz[0], ((char*)std::string("EzCompEzInput").append(unq).c_str()));

		graphParams.inputChannels["Ex"].push_back(ExCompExInputChannel);
		graphParams.inputChannels["Ey"].push_back(EyCompEyInputChannel);
		graphParams.inputChannels["Ez"].push_back(EzCompEzInputChannel);
	}
	else
	{
		Port *HxPort = graphParams.varToPortMap["Hx"];
		Port *HyPort = graphParams.varToPortMap["Hy"];
		Port *HzPort = graphParams.varToPortMap["Hz"];

		Port *ExPort = graphParams.varToPortMap["Ex"];
		Port *EyPort = graphParams.varToPortMap["Ey"];
		Port *EzPort = graphParams.varToPortMap["Ez"];

		InternalChannel *HxCompHxChannel = graphParams.g->AddInternalChannel(HxPort, inputPortsHx[0], ((char*)std::string("HxToHx").append(unq).c_str()));
		InternalChannel *HyCompHyChannel = graphParams.g->AddInternalChannel(HyPort, inputPortsHy[0], ((char*)std::string("HyToHy").append(unq).c_str()));
		InternalChannel *HzCompHzChannel = graphParams.g->AddInternalChannel(HzPort, inputPortsHz[0], ((char*)std::string("HzToHz").append(unq).c_str()));

		InternalChannel *HxCompEyInputChannel = graphParams.g->AddInternalChannel(EyPort, inputPortsHx[1], ((char*)std::string("HxCompEyInput").append(unq).c_str()));
		InternalChannel *HxCompEzInputChannel = graphParams.g->AddInternalChannel(EzPort, inputPortsHx[2], ((char*)std::string("HxCompEzInput").append(unq).c_str()));

		InternalChannel *HyCompExInputChannel = graphParams.g->AddInternalChannel(ExPort, inputPortsHy[1], ((char*)std::string("HyCompExInput").append(unq).c_str()));
		InternalChannel *HyCompEzInputChannel = graphParams.g->AddInternalChannel(EzPort, inputPortsHy[2], ((char*)std::string("HyCompEzInput").append(unq).c_str()));

		InternalChannel *HzCompExInputChannel = graphParams.g->AddInternalChannel(ExPort, inputPortsHz[1], ((char*)std::string("HzCompExInput").append(unq).c_str()));
		InternalChannel *HzCompEyInputChannel = graphParams.g->AddInternalChannel(EyPort, inputPortsHz[2], ((char*)std::string("HzCompEyInput").append(unq).c_str()));

		InternalChannel  *ExCompExInputChannel = graphParams.g->AddInternalChannel(ExPort, inputPortsEx[0], ((char*)std::string("ExCompExInput").append(unq).c_str()));
		InternalChannel  *EyCompEyInputChannel = graphParams.g->AddInternalChannel(EyPort, inputPortsEy[0], ((char*)std::string("EyCompEyInput").append(unq).c_str()));
		InternalChannel  *EzCompEzInputChannel = graphParams.g->AddInternalChannel(EzPort, inputPortsEz[0], ((char*)std::string("EzCompEzInput").append(unq).c_str()));
	}
	//Update the map
	graphParams.varToPortMap["Hx"] = outputPortsHx[0];
	graphParams.varToPortMap["Hy"] = outputPortsHy[0];
	graphParams.varToPortMap["Hz"] = outputPortsHz[0];

	graphParams.varToPortMap["Ex"] = outputPortsEx[0];
	graphParams.varToPortMap["Ey"] = outputPortsEy[0];
	graphParams.varToPortMap["Ez"] = outputPortsEz[0];

	if(lastIteration)
	{
		GraphOutputChannel *HxOutChannel = graphParams.g->AddOutputChannel(outputPortsHx[0], ((char*)std::string("HxOutput").append(unq).c_str()));
		GraphOutputChannel *HyOutChannel = graphParams.g->AddOutputChannel(outputPortsHy[0], ((char*)std::string("HyOutput").append(unq).c_str()));
		GraphOutputChannel *HzOutChannel = graphParams.g->AddOutputChannel(outputPortsHz[0], ((char*)std::string("HzOutput").append(unq).c_str()));

		GraphOutputChannel *ExOutChannel = graphParams.g->AddOutputChannel(outputPortsEx[0], ((char*)std::string("ExOutput").append(unq).c_str()));
		GraphOutputChannel *EyOutChannel = graphParams.g->AddOutputChannel(outputPortsEy[0], ((char*)std::string("EyOutput").append(unq).c_str()));
		GraphOutputChannel *EzOutChannel = graphParams.g->AddOutputChannel(outputPortsEz[0], ((char*)std::string("EzOutput").append(unq).c_str()));

		graphParams.outputChannels["Hx"] = HxOutChannel;
		graphParams.outputChannels["Hy"] = HyOutChannel;
		graphParams.outputChannels["Hz"] = HzOutChannel;
		graphParams.outputChannels["Ex"] = ExOutChannel;
		graphParams.outputChannels["Ey"] = EyOutChannel;
		graphParams.outputChannels["Ez"] = EzOutChannel;

        graphParams.outputChannels["Hx"]->SetViewMaterializationPolicy(VIEWMATERIALIZATIONPOLICY_EAGER);
		graphParams.outputChannels["Hy"]->SetViewMaterializationPolicy(VIEWMATERIALIZATIONPOLICY_EAGER);
		graphParams.outputChannels["Hz"]->SetViewMaterializationPolicy(VIEWMATERIALIZATIONPOLICY_EAGER);
		graphParams.outputChannels["Ex"]->SetViewMaterializationPolicy(VIEWMATERIALIZATIONPOLICY_EAGER);
		graphParams.outputChannels["Ey"]->SetViewMaterializationPolicy(VIEWMATERIALIZATIONPOLICY_EAGER);
		graphParams.outputChannels["Ez"]->SetViewMaterializationPolicy(VIEWMATERIALIZATIONPOLICY_EAGER);

	}

	delete [] inputPortsHx;
	delete [] inputPortsHy;
	delete [] inputPortsHz;

	delete [] inputPortsEx;
	delete [] inputPortsEy;
	delete [] inputPortsEz;

	delete [] outputPortsHx;
	delete [] outputPortsHy;
	delete [] outputPortsHz;

	delete [] outputPortsEx;
	delete [] outputPortsEy;
	delete [] outputPortsEz;
}

void constructUnrolledGraph(FDTDGraphParams& params, int numIterations)
{
	for(int i=1; i<=numIterations ; ++i)
		constructSingleIterationGraph(params, i, i==numIterations);
}
#pragma endregion GRAPH_CONSTRUCTION

bool testSingleKernel(std::string compVar,
	  				  FDTDGraphParams& graphParams,
					  FDTD_PARAMS params)
{
	std::string fileName, kernelName, taskName;
	IterationSpaceSize iterSpace;
	std::vector<std::string> inputVars, outputVars;
	Port **inputPorts, **outputPorts;
	int numInputPorts, numOutputPorts;

	numInputPorts = 4; //3 inputs + 1 config
	inputPorts = new Port*[numInputPorts];

	numOutputPorts = 1;
	outputPorts = new Port*[numOutputPorts];
    std::vector<CSimple3DArray<float>**> vArrays;

	CSimple3DArray<float> *input1 = NULL;   vArrays.push_back(&input1);
    CSimple3DArray<float>* input2 = NULL;   vArrays.push_back(&input2);
    CSimple3DArray<float> *input3 = NULL;   vArrays.push_back(&input3);
	CSimple3DArray<float> *output = NULL;   vArrays.push_back(&output);

	std::vector<DatablockTemplate*> inputTemplates;
	DatablockTemplate *outputTemplate;

	CPUFunctionType cpuFunc = NULL;

	CSimple3DArray<float> *Ex = NULL;       vArrays.push_back(&Ex);
	CSimple3DArray<float> *Ey = NULL;       vArrays.push_back(&Ey);
	CSimple3DArray<float> *Ez = NULL;       vArrays.push_back(&Ez);

    CSimple3DArray<float> *Hx = NULL;       vArrays.push_back(&Hx);
	CSimple3DArray<float> *Hy = NULL;       vArrays.push_back(&Hy);
	CSimple3DArray<float> *Hz = NULL;       vArrays.push_back(&Hz);

    CSimple3DArray<float> *ExOut = NULL;    vArrays.push_back(&ExOut);
	CSimple3DArray<float> *EyOut = NULL;    vArrays.push_back(&EyOut);
	CSimple3DArray<float> *EzOut = NULL;    vArrays.push_back(&EzOut);
	CSimple3DArray<float> *HxOut = NULL;    vArrays.push_back(&HxOut);
	CSimple3DArray<float> *HyOut = NULL;    vArrays.push_back(&HyOut);
	CSimple3DArray<float> *HzOut = NULL;    vArrays.push_back(&HzOut);

	IterationSpaceSize HxSize = graphParams.findVarSize("Hx");
	configure_raw_array(HxSize.dim1, HxSize.dim2, HxSize.dim3, &Hx, -1);
	configure_raw_array(HxSize.dim1, HxSize.dim2, HxSize.dim3, &HxOut, 0);

	IterationSpaceSize HySize = graphParams.findVarSize("Hy");
	configure_raw_array(HySize.dim1, HySize.dim2, HySize.dim3, &Hy, -1);
	configure_raw_array(HySize.dim1, HySize.dim2, HySize.dim3, &HyOut, 0);

	IterationSpaceSize HzSize = graphParams.findVarSize("Hz");
	configure_raw_array(HzSize.dim1, HzSize.dim2, HzSize.dim3, &Hz, -1);
	configure_raw_array(HzSize.dim1, HzSize.dim2, HzSize.dim3, &HzOut, 0);

	IterationSpaceSize ExSize = graphParams.findVarSize("Ex");
	configure_raw_array(ExSize.dim1, ExSize.dim2, ExSize.dim3, &Ex, -1);
	configure_raw_array(ExSize.dim1, ExSize.dim2, ExSize.dim3, &ExOut, 0);
	
	IterationSpaceSize EySize = graphParams.findVarSize("Ey");
	configure_raw_array(EySize.dim1, EySize.dim2, EySize.dim3, &Ey, -1);
	configure_raw_array(EySize.dim1, EySize.dim2, EySize.dim3, &EyOut, 0);
		
	IterationSpaceSize EzSize = graphParams.findVarSize("Ez");
	configure_raw_array(EzSize.dim1, EzSize.dim2, EzSize.dim3, &Ez, -1);
	configure_raw_array(EzSize.dim1, EzSize.dim2, EzSize.dim3, &EzOut, 0);

	if(compVar == "Hx")
	{
		fileName = graphParams.dirName + "\\HxComp.hlsl";
		kernelName = "HxComputation";
		taskName = "HxCompTask";
		inputVars.push_back("Hx");
		inputVars.push_back("Ey");
		inputVars.push_back("Ez");
		outputVars.push_back("Hx");
		iterSpace = graphParams.findVarSize("Hx");

		input1 = Hx;
		input2 = Ey;
		input3 = Ez;

		output = HxOut;

		cpuFunc = fdtd_HxComputation_cpu;
	}
	else if(compVar == "Hy")
	{
		fileName = graphParams.dirName + "\\HyComp.hlsl";
		kernelName = "HyComputation";
		taskName = "HyCompTask";
		inputVars.push_back("Hy");
		inputVars.push_back("Ex");
		inputVars.push_back("Ez");
		outputVars.push_back("Hy");
		iterSpace = graphParams.findVarSize("Hy");

		input1 = Hy;
		input2 = Ex;
		input3 = Ez;

		output = HyOut;

		cpuFunc = fdtd_HyComputation_cpu;
	}
	else if(compVar == "Hz")
	{
		fileName = graphParams.dirName + "\\HzComp.hlsl";
		kernelName = "HzComputation";
		taskName = "HzCompTask";
		inputVars.push_back("Hz");
		inputVars.push_back("Ex");
		inputVars.push_back("Ey");
		outputVars.push_back("Hz");
		iterSpace = graphParams.findVarSize("Hz");

		input1 = Hz;
		input2 = Ex;
		input3 = Ey;

		output = HzOut;

		cpuFunc = fdtd_HzComputation_cpu;
	}
	else if(compVar == "Ex")
	{
		fileName = graphParams.dirName + "\\ExComp.hlsl";
		kernelName = "ExComputation";
		taskName = "ExCompTask";
		inputVars.push_back("Ex");
		inputVars.push_back("Hy");
		inputVars.push_back("Hz");
		outputVars.push_back("Ex");
		iterSpace = graphParams.findVarSize("Ex");

		input1 = Ex;
		input2 = Hy;
		input3 = Hz;

		output = ExOut;

		cpuFunc = fdtd_ExComputation_cpu;
	}
	else if(compVar == "Ey")
	{
		fileName = graphParams.dirName + "\\EyComp.hlsl";
		kernelName = "EyComputation";
		taskName = "EyCompTask";
		inputVars.push_back("Ey");
		inputVars.push_back("Hx");
		inputVars.push_back("Hz");
		outputVars.push_back("Ey");
		iterSpace = graphParams.findVarSize("Ey");

		input1 = Ey;
		input2 = Hx;
		input3 = Hz;

		output = EyOut;

		cpuFunc = fdtd_EyComputation_cpu;
	}
	else if(compVar == "Ez")
	{
		fileName = graphParams.dirName + "\\EzComp.hlsl";
		kernelName = "EzComputation";
		taskName = "EzCompTask";
		inputVars.push_back("Ez");
		inputVars.push_back("Hx");
		inputVars.push_back("Hy");
		outputVars.push_back("Ez");
		iterSpace = graphParams.findVarSize("Ez");

		input1 = Ez;
		input2 = Hx;
		input3 = Hy;

		output = EzOut;

		cpuFunc = fdtd_EzComputation_cpu;
	}
	else
		assert (0 && "Unknown variable specified");
	
	//HACK To make the CUDA option work. Will fix this later.
	if (graphParams.kernelType == CUDA)
		fileName = graphParams.dirName + "fdtdMain.compute_10.ptx";

	Task *kernelTask = contructComputationTask(fileName, kernelName, taskName, iterSpace, inputVars, outputVars, inputPorts, numInputPorts, outputPorts, numOutputPorts, inputTemplates, outputTemplate, graphParams); 
	GraphInputChannel * pInput1	= graphParams.g->AddInputChannel(inputPorts[0], "InputChannel1");
	GraphInputChannel * pInput2	= graphParams.g->AddInputChannel(inputPorts[1], "InputChannel2");
	GraphInputChannel * pInput3	= graphParams.g->AddInputChannel(inputPorts[2], "InputChannel3");
	GraphInputChannel * pParmsInput	= graphParams.g->AddInputChannel(inputPorts[3], "ConstChannel");
	GraphOutputChannel * pOutput    = graphParams.g->AddOutputChannel(outputPorts[0], "outputChannel");

    pOutput->SetViewMaterializationPolicy(VIEWMATERIALIZATIONPOLICY_EAGER);

	graphParams.g->Run();

    Datablock * pInputBlock1 = PTask::Runtime::AllocateDatablock(inputTemplates.at(0), input1->cells(), input1->arraysize(), pInput1);
	Datablock * pInputBlock2 = PTask::Runtime::AllocateDatablock(inputTemplates.at(1), input2->cells(), input2->arraysize(), pInput2);
	Datablock * pInputBlock3 = PTask::Runtime::AllocateDatablock(inputTemplates.at(2), input3->cells(), input3->arraysize(), pInput3);
	Datablock * pConstPrm    = PTask::Runtime::AllocateDatablock(graphParams.pConstParamTemplate, &params, sizeof(params), pParmsInput);

	pInput1->Push(pInputBlock1);
	pInput2->Push(pInputBlock2);
	pInput3->Push(pInputBlock3);
	pParmsInput->Push(pConstPrm);
	
	pInputBlock1->Release();
	pInputBlock2->Release();
	pInputBlock3->Release();
	pConstPrm->Release();

	Datablock * pResultBlock = pOutput->Pull();

	pResultBlock->Lock();
	int stride = sizeof(float);
	int elements = iterSpace.dim1 * iterSpace.dim2 * iterSpace.dim3;
	float * psrc = (float*) pResultBlock->GetDataPointer(FALSE);
	float * pdst = output->cells();
	memcpy(pdst, psrc, elements*stride);
	pResultBlock->Unlock();
	pResultBlock->Release();
	
	printf( "Verifying against CPU result..." );
	int nErrorTolerance = 0;
	CSimple3DArray<float>* cpuOutput = cpuFunc(input1, input2, input3, params); // on CPU
	bool ret = false;
	if(!check_array_result(output, cpuOutput, &nErrorTolerance)) {
		printf("failure: (%d of %d) erroneous cells\n", nErrorTolerance, params.Nx*params.Ny*params.Nz);
		print_array(output, "GPU");
		print_array(cpuOutput, "CPU");
		ret = false;
    } else {
		//print_array(output, "GPU");
		printf( "%s succeeded\n", compVar.c_str() );
		ret = true;
    }

	delete [] inputPorts;
	delete [] outputPorts;

	//delete Ex;
	//delete Ey;
	//delete Ez;
	//delete Hx;
	//delete Hy;
	//delete Hz;

	//delete ExOut;
	//delete EyOut;
	//delete EzOut;
	//delete HxOut;
	//delete HyOut;
	//delete HzOut;

	delete cpuOutput;
    // if(output) delete output;
    
    vector<CSimple3DArray<float>**>::iterator vi;
    for(vi=vArrays.begin(); vi!=vArrays.end(); vi++) 
        if(*(*vi)) delete *(*vi);

	return false;
}

bool testUnrolledGraph(FDTDGraphParams& graphParams,
					   FDTD_PARAMS params,
					   int numIterations, //Number of iterations of the field computation to perform
					   int numBlocks) //Number of distinct input blocks to push into the input
{
	CSimple3DArray<float> *Ex;
	CSimple3DArray<float> *Ey;
	CSimple3DArray<float> *Ez;

	CSimple3DArray<float> *Hx;
	CSimple3DArray<float> *Hy;
	CSimple3DArray<float> *Hz;

	CSimple3DArray<float> *ExOut;
	CSimple3DArray<float> *EyOut;
	CSimple3DArray<float> *EzOut;

	CSimple3DArray<float> *HxOut;
	CSimple3DArray<float> *HyOut;
	CSimple3DArray<float> *HzOut;

	std::map< std::string, CSimple3DArray<float>* > inputAddresses, outputAddresses;

	IterationSpaceSize HxSize = graphParams.findVarSize("Hx");
	configure_raw_array(HxSize.dim1, HxSize.dim2, HxSize.dim3, &Hx, -1);
	inputAddresses["Hx"] = Hx;
	configure_raw_array(HxSize.dim1, HxSize.dim2, HxSize.dim3, &HxOut, 0);
	outputAddresses["Hx"] = HxOut;

	IterationSpaceSize HySize = graphParams.findVarSize("Hy");
	configure_raw_array(HySize.dim1, HySize.dim2, HySize.dim3, &Hy, -1);
	inputAddresses["Hy"] = Hy;
	configure_raw_array(HySize.dim1, HySize.dim2, HySize.dim3, &HyOut, 0);
	outputAddresses["Hy"] = HyOut;

	IterationSpaceSize HzSize = graphParams.findVarSize("Hz");
	configure_raw_array(HzSize.dim1, HzSize.dim2, HzSize.dim3, &Hz, -1);
	inputAddresses["Hz"] = Hz;
	configure_raw_array(HzSize.dim1, HzSize.dim2, HzSize.dim3, &HzOut, 0);
	outputAddresses["Hz"] = HzOut;

	IterationSpaceSize ExSize = graphParams.findVarSize("Ex");
	configure_raw_array(ExSize.dim1, ExSize.dim2, ExSize.dim3, &Ex, -1);
	inputAddresses["Ex"] = Ex;
	configure_raw_array(ExSize.dim1, ExSize.dim2, ExSize.dim3, &ExOut, 0);
	outputAddresses["Ex"] = ExOut;
	
	IterationSpaceSize EySize = graphParams.findVarSize("Ey");
	configure_raw_array(EySize.dim1, EySize.dim2, EySize.dim3, &Ey, -1);
	inputAddresses["Ey"] = Ey;
	configure_raw_array(EySize.dim1, EySize.dim2, EySize.dim3, &EyOut, 0);
	outputAddresses["Ey"] = EyOut;
		
	IterationSpaceSize EzSize = graphParams.findVarSize("Ez");
	configure_raw_array(EzSize.dim1, EzSize.dim2, EzSize.dim3, &Ez, -1);
	inputAddresses["Ez"] = Ez;
	configure_raw_array(EzSize.dim1, EzSize.dim2, EzSize.dim3, &EzOut, 0);
	outputAddresses["Ez"] = EzOut;

	constructUnrolledGraph(graphParams, numIterations);
	for(size_t i=0; i<graphParams.paramInputs.size() ; ++i)
		graphParams.paramInputs.at(i)->SetNoDraw();
	graphParams.g->WriteDOTFile("graphfdtd.dot", false);
	graphParams.g->Run(g_bSingleThreaded);

	// Construct the required data blocks and push them into their channels
	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();
	for(int inputNum=0; inputNum<numBlocks ; ++inputNum)
	{
		for(std::map<std::string, std::vector< GraphInputChannel* > >::iterator iter  = graphParams.inputChannels.begin() ; 
			iter != graphParams.inputChannels.end(); ++iter)
		{
			const std::string& varName = iter->first;
			const std::vector< GraphInputChannel* >& channels = iter->second;
			assert (inputAddresses.find(varName) != inputAddresses.end());
			Datablock *dataBlock = PTask::Runtime::AllocateDatablock(graphParams.findTemplate(varName), 
                                                                     inputAddresses[varName]->cells(), 
                                                                     inputAddresses[varName]->arraysize(),
                                                                     channels.at(0));
			for(size_t i=0; i<channels.size() ; ++i)
			{
				channels.at(i)->Push(dataBlock);
			}
			// We hold a reference implicitly to any blocks returned from AllocateDatablock
			// We have transferred ownership to the input channels we need to
			// release our references.
			dataBlock->Release();
		}
	}

	Datablock * constPrm = PTask::Runtime::AllocateDatablock(graphParams.pConstParamTemplate, &params, sizeof(params), NULL); //graphParams.paramInputs.at(0));
	for(size_t i=0; i<graphParams.paramInputs.size() ; ++i)
	{
		graphParams.paramInputs.at(i)->Push(constPrm);
	}
    constPrm->Release();
	cout << "Finished Pushing inputs\n";
	//Get the outputs of the graph
	for(int outputNum=0; outputNum<numBlocks ; ++outputNum)
	{
		for(std::map<std::string, GraphOutputChannel*>::iterator iter=graphParams.outputChannels.begin();
			iter!=graphParams.outputChannels.end(); ++iter)
		{
			const std::string& varName = iter->first;
			// std::cout << varName << std::endl;
			GraphOutputChannel *outputChannel = iter->second;
			Datablock * pResultBlock = outputChannel->Pull();
			assert (outputAddresses.find(varName) != outputAddresses.end());
			CSimple3DArray<float> *outputArray = outputAddresses[varName];
			pResultBlock->Lock();
			float* psrc = (float*) pResultBlock->GetDataPointer(FALSE);
			float* pdst = outputArray->cells();
			IterationSpaceSize varSize = graphParams.findVarSize(varName);
			int size = sizeof(float) * varSize.dim1 * varSize.dim2 * varSize.dim3;
			memcpy(pdst, psrc, size);
			pResultBlock->Unlock();
			pResultBlock->Release();
		}
	}
    double dGPUTime = pTimer->elapsed(false);

    double dCPUStart = pTimer->elapsed(false);
	fdtd_cpu_multiple_inputs(Hx, Hy, Hz, Ex, Ey, Ez, numIterations, numBlocks, params);
    double dCPUEnd = pTimer->elapsed(false);

	int numErrors = 20;
	// print_array(HxOut, "GPU:");
	// print_array(Hx,    "CPU:");
    bool bSuccess = true;
    double dTotalSqError = 0.0;
	printf("Hx: "); bSuccess &= check_array_result(Hx, HxOut, &numErrors, &dTotalSqError);
	printf("Hy: "); bSuccess &= check_array_result(Hy, HyOut, &numErrors, &dTotalSqError);
	printf("Hz: "); bSuccess &= check_array_result(Hz, HzOut, &numErrors, &dTotalSqError);
	printf("Ex: "); bSuccess &= check_array_result(Ex, ExOut, &numErrors, &dTotalSqError);
	printf("Ey: "); bSuccess &= check_array_result(Ey, EyOut, &numErrors, &dTotalSqError);
	printf("Ez: "); bSuccess &= check_array_result(Ez, EzOut, &numErrors, &dTotalSqError);
    if(!bSuccess && dTotalSqError > 1.0) {
        printf("failed\n");
    } else {
        printf("succeeded\n");
    }
    printf("GPU exec:\t%.1f\nCPU exec:\t%.1f\n", dGPUTime, dCPUEnd-dCPUStart);

    delete pTimer;
	delete Ex;
	delete Ey;
	delete Ez;
	delete Hx;
	delete Hy;
	delete Hz;

	delete ExOut;
	delete EyOut;
	delete EzOut;
	delete HxOut;
	delete HyOut;
	delete HzOut;
	return false;
}


bool testUnrolledGraph_ST(FDTDGraphParams& graphParams,
					   FDTD_PARAMS params,
					   int numIterations, //Number of iterations of the field computation to perform
					   int numBlocks) //Number of distinct input blocks to push into the input
{
	CSimple3DArray<float> *Ex;
	CSimple3DArray<float> *Ey;
	CSimple3DArray<float> *Ez;

	CSimple3DArray<float> *Hx;
	CSimple3DArray<float> *Hy;
	CSimple3DArray<float> *Hz;

	CSimple3DArray<float> *ExOut;
	CSimple3DArray<float> *EyOut;
	CSimple3DArray<float> *EzOut;

	CSimple3DArray<float> *HxOut;
	CSimple3DArray<float> *HyOut;
	CSimple3DArray<float> *HzOut;

	std::map< std::string, CSimple3DArray<float>* > inputAddresses, outputAddresses;

	IterationSpaceSize HxSize = graphParams.findVarSize("Hx");
	configure_raw_array(HxSize.dim1, HxSize.dim2, HxSize.dim3, &Hx, -1);
	inputAddresses["Hx"] = Hx;
	configure_raw_array(HxSize.dim1, HxSize.dim2, HxSize.dim3, &HxOut, 0);
	outputAddresses["Hx"] = HxOut;

	IterationSpaceSize HySize = graphParams.findVarSize("Hy");
	configure_raw_array(HySize.dim1, HySize.dim2, HySize.dim3, &Hy, -1);
	inputAddresses["Hy"] = Hy;
	configure_raw_array(HySize.dim1, HySize.dim2, HySize.dim3, &HyOut, 0);
	outputAddresses["Hy"] = HyOut;

	IterationSpaceSize HzSize = graphParams.findVarSize("Hz");
	configure_raw_array(HzSize.dim1, HzSize.dim2, HzSize.dim3, &Hz, -1);
	inputAddresses["Hz"] = Hz;
	configure_raw_array(HzSize.dim1, HzSize.dim2, HzSize.dim3, &HzOut, 0);
	outputAddresses["Hz"] = HzOut;

	IterationSpaceSize ExSize = graphParams.findVarSize("Ex");
	configure_raw_array(ExSize.dim1, ExSize.dim2, ExSize.dim3, &Ex, -1);
	inputAddresses["Ex"] = Ex;
	configure_raw_array(ExSize.dim1, ExSize.dim2, ExSize.dim3, &ExOut, 0);
	outputAddresses["Ex"] = ExOut;
	
	IterationSpaceSize EySize = graphParams.findVarSize("Ey");
	configure_raw_array(EySize.dim1, EySize.dim2, EySize.dim3, &Ey, -1);
	inputAddresses["Ey"] = Ey;
	configure_raw_array(EySize.dim1, EySize.dim2, EySize.dim3, &EyOut, 0);
	outputAddresses["Ey"] = EyOut;
		
	IterationSpaceSize EzSize = graphParams.findVarSize("Ez");
	configure_raw_array(EzSize.dim1, EzSize.dim2, EzSize.dim3, &Ez, -1);
	inputAddresses["Ez"] = Ez;
	configure_raw_array(EzSize.dim1, EzSize.dim2, EzSize.dim3, &EzOut, 0);
	outputAddresses["Ez"] = EzOut;

	constructUnrolledGraph(graphParams, numIterations);
	for(size_t i=0; i<graphParams.paramInputs.size() ; ++i)
		graphParams.paramInputs.at(i)->SetNoDraw();
	graphParams.g->WriteDOTFile("graphfdtd.dot", false);
	graphParams.g->Run(g_bSingleThreaded);

	// Construct the required data blocks and push them into their channels
	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();

	Datablock * constPrm = PTask::Runtime::AllocateDatablock(graphParams.pConstParamTemplate, &params, sizeof(params), NULL); //graphParams.paramInputs.at(0));
	for(size_t i=0; i<graphParams.paramInputs.size() ; ++i)
	{
		graphParams.paramInputs.at(i)->Push(constPrm);
	}
    constPrm->Release();
	cout << "Finished Pushing inputs\n";
    
    for(int num=0; num<numBlocks; ++num) 
	{
		for(std::map<std::string, std::vector< GraphInputChannel* > >::iterator iter  = graphParams.inputChannels.begin() ; 
			iter != graphParams.inputChannels.end(); ++iter)
		{
			const std::string& varName = iter->first;
			const std::vector< GraphInputChannel* >& channels = iter->second;
			assert (inputAddresses.find(varName) != inputAddresses.end());
			Datablock *dataBlock = PTask::Runtime::AllocateDatablock(graphParams.findTemplate(varName), 
                                                                     inputAddresses[varName]->cells(), 
                                                                     inputAddresses[varName]->arraysize(), 
                                                                     channels.at(0));
			for(size_t i=0; i<channels.size() ; ++i)
			{
				channels.at(i)->Push(dataBlock);
			}
			// We hold a reference implicitly to any blocks returned from AllocateDatablock
			// We have transferred ownership to the input channels we need to
			// release our references.
			dataBlock->Release();
		}

        for(std::map<std::string, GraphOutputChannel*>::iterator iter=graphParams.outputChannels.begin();
			iter!=graphParams.outputChannels.end(); ++iter)
		{
			const std::string& varName = iter->first;
			// std::cout << varName << std::endl;
			GraphOutputChannel *outputChannel = iter->second;
			Datablock * pResultBlock = outputChannel->Pull();
			assert (outputAddresses.find(varName) != outputAddresses.end());
			CSimple3DArray<float> *outputArray = outputAddresses[varName];
			pResultBlock->Lock();
			float* psrc = (float*) pResultBlock->GetDataPointer(FALSE);
			float* pdst = outputArray->cells();
			IterationSpaceSize varSize = graphParams.findVarSize(varName);
			int size = sizeof(float) * varSize.dim1 * varSize.dim2 * varSize.dim3;
			memcpy(pdst, psrc, size);
			pResultBlock->Unlock();
			pResultBlock->Release();
		}
	}
    double dGPUTime = pTimer->elapsed(false);

    double dCPUStart = pTimer->elapsed(false);
	fdtd_cpu_multiple_inputs(Hx, Hy, Hz, Ex, Ey, Ez, numIterations, numBlocks, params);
    double dCPUEnd = pTimer->elapsed(false);

	int numErrors = 20;
	// print_array(HxOut, "GPU:");
	// print_array(Hx,    "CPU:");
    bool bSuccess = true;
    double dTotalSqError = 0.0;
	printf("Hx: "); bSuccess &= check_array_result(Hx, HxOut, &numErrors, &dTotalSqError);
	printf("Hy: "); bSuccess &= check_array_result(Hy, HyOut, &numErrors, &dTotalSqError);
	printf("Hz: "); bSuccess &= check_array_result(Hz, HzOut, &numErrors, &dTotalSqError);
	printf("Ex: "); bSuccess &= check_array_result(Ex, ExOut, &numErrors, &dTotalSqError);
	printf("Ey: "); bSuccess &= check_array_result(Ey, EyOut, &numErrors, &dTotalSqError);
	printf("Ez: "); bSuccess &= check_array_result(Ez, EzOut, &numErrors, &dTotalSqError);
    if(!bSuccess && dTotalSqError > 1.0) {
        printf("failed\n");
    } else {
        printf("succeeded\n");
    }
    printf("GPU exec:\t%.1f\nCPU exec:\t%.1f\n", dGPUTime, dCPUEnd-dCPUStart);

    delete pTimer;
	delete Ex;
	delete Ey;
	delete Ez;
	delete Hx;
	delete Hy;
	delete Hz;

	delete ExOut;
	delete EyOut;
	delete EzOut;
	delete HxOut;
	delete HyOut;
	delete HzOut;
	return false;
}

int run_graph_fdtd_task(
	char * szdir,
	char * szshader,
	int Nx,
	int Ny,
	int Nz,
	int numBlocks,
	int iterations,
	KernelType ktype
	)
{
	//Nx = 32;
    //Ny = 32;
	//Nz = 4;
    
    CONFIGUREPTASKU(UseOpenCL, FALSE);
    CONFIGUREPTASKU(UseDirectX, (ktype == HLSL));
    CONFIGUREPTASKU(UseCUDA, (ktype == CUDA));
	PTask::Runtime::Initialize(); 
    CheckPlatformSupport((ktype == CUDA ? "quack.ptx":"quack.hlsl"), NULL);

    std::set<Datablock*> myset;
    std::set<Datablock*>::iterator vi;
    for(vi=myset.begin(); vi!=myset.end(); vi++) {
        printf("QUACK\n");
    }

	//Hard coding the lengths of the cavities for now
	float Lx = float(0.05), Ly = float(0.04), Lz = float(0.03);
	float nrm = sqrt((Nx/Lx*Nx/Lx) + (Ny/Ly*Ny/Ly) + (Nz/Lz*Nz/Lz))*100;
	float pi = float(3.1415);

    printf("%d %d %d (%d iterations)\n",
           Nx,
           Ny,
           Nz,
           iterations);

    FDTDGraphParams graphParams(Nx, Ny, Nz, std::string(szdir), ktype);
	FDTD_PARAMS params;
	
	params.Nx = Nx;
	params.Ny = Ny;
	params.Nz = Nz;
	
	params.Cx = Nx/Lx;
	params.Cy = Ny/Ly;
	params.Cz = Nz/Lz;

	params.eps0 = float(8.8541878e-12); // Permittivity of vacuum.
	params.mu0 = float(4e-7)*pi; //Permeability of vacuum.
	params.c0 = float(299792458); //Speed of light in vacuum.
	params.Dt = float(1.0)/(params.c0*nrm); //Time step.
	
	graphParams.g = new Graph();
	
	//testSingleKernel("Ez", graphParams, params);
    if(g_bSingleThreaded || g_nChannelCapacity < numBlocks) {
        testUnrolledGraph_ST(graphParams, params, iterations, numBlocks);
    } else {
        testUnrolledGraph(graphParams, params, iterations, numBlocks);
    }

	//Do all the graph destruction stuff
	graphParams.g->Stop();
	graphParams.g->Teardown();
	Graph::DestroyGraph(graphParams.g);
	PTask::Runtime::Terminate();

	return 0;
}

