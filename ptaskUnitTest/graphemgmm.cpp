//--------------------------------------------------------------------------------------
// File: graphemgmm.cpp
//--------------------------------------------------------------------------------------
#include <stdio.h>
#include <crtdbg.h>

#include <iostream>
#include <fstream>

#include "accelerator.h"
#include "assert.h"
#include "shaderparms.h"
#include "sort.h"
#include <vector>
#include <algorithm>

#include "matrixtask.h"
#include "SimpleMatrix.h"
#include "matmul.h"
#include "graphmatmul.h"
#include "elemtype.h"

// graph API stuff
#include "graph.h"
#include "datablock.h"
#include "datablocktemplate.h"
#include "CompiledKernel.h"
#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "hrperft.h"
#include "platformcheck.h"

using namespace std;
using namespace PTask;

// CPU impl
#include "nr3\nr3.h"
#include "nr3\cholesky.h"
#include "nr3\gaumixmod.h"

class ModelComponent {
public:
	ModelComponent(float mean, float variance, float mixingCoefficient)
		: _mean(mean), _variance(variance), _mixingCoefficient(mixingCoefficient)
	{ }

private:
	float _mean;
	float _variance;
	float _mixingCoefficient;
};

typedef vector<ModelComponent> MixtureModel;

//typedef vector<float> Distribution;
typedef vector<double> Distribution;

/* Load a distribution from a CSV file. Expected format is (as output by R's write.csv command):
"","x"
"1",5.49771313251235
"2",4.71954244851374
...
 */
#pragma warning(disable:4996)
Distribution * LoadDistriutionFromCSVFile(const char * filename)
{
	fprintf(stdout, "Loading distribution from %s\n", filename);

	// Skip first line, which contains column schema.
	string line;
	ifstream file(filename) ;
	getline(file, line);

	// Parse remaining lines of the file.
	Distribution * d = new Distribution();
    const char * valueStr;
	float value;
    while(getline(file, line))
	{
		valueStr = strrchr(line.data(), ',');
		valueStr++;
        sscanf(valueStr, "%f", &value);
		d->push_back(value) ;
		// cout << valueStr << " -> " << value << endl;
	}
	cout << d->size() << " values loaded."<< endl;

	return d;
#pragma warning(default:4996)
}

void CPU_EM(MatDoub& data, MatDoub& means, int iterations)
{
    Gaumixmod* gmmcpu = new Gaumixmod(data, means);

    cout << endl << "Data stats:" << endl;
    cout << "  num data points (nn) = " << gmmcpu->nn << endl;
    cout << "  num components (kk)  = " << gmmcpu->kk << endl;
    cout << "  num dimensions (mm)  = " << gmmcpu->mm << endl;

    cout << "Initial means = ";
    for (int k=0; k<means.nrows(); k++)
    {
        cout << means[k][0] << " ";
    }
    cout << endl;

    double delta;
    for (int i=0; i<iterations; i++)
    {
        delta = gmmcpu->estep();
        cout << "Iteration " << i + 1 << " : delta = " << delta << endl;
        gmmcpu->mstep();

        cout << "  means = ";
        for (int k=0; k<means.nrows(); k++)
        {
            cout << gmmcpu->means[k][0] << " ";
        }
        cout << endl;
    }
}

Graph * BuildEMGraph(
	DatablockTemplate * distributionTemplate,
	DatablockTemplate * modelTemplate,
	int iterations)
{
	Graph * pGraph = new Graph();

	// Compile the kernels.
	char * KernelSourceFile = "..\\..\\..\\accelerators\\PTaskUnitTest\\gpugaumixmod_kernel.ptx";
	char * EStepKernelOpName = "estep_kernel";
   	char * EStepNormalizeKernelOpName = "estep_normalize_kernel";
	char * MStepKernelOpName = "mstep_kernel";
	char * MStepSigmaKernelOpName = "mstep_sigma_kernel";
	CompiledKernel * EStepKernel = COMPILE_KERNEL(KernelSourceFile, EStepKernelOpName);
//	CompiledKernel * MStepKernel = PTask::Runtime::GetCompiledKernel(KernelSourceFile, MStepKernelOpName);

/*
	Doub estep() {
		Int k,m,n;
		Doub tmp,sum,max,oldloglike;
		VecDoub u(mm),v(mm);
		oldloglike = loglike;
		for (k=0;k<kk;k++) {
			Cholesky choltmp(sig[k]);
			lndets[k] = choltmp.logdet();
			for (n=0;n<nn;n++) {
				for (m=0;m<mm;m++) u[m] = data[n][m]-means[k][m];
				choltmp.elsolve(u,v);
				for (sum=0.,m=0; m<mm; m++) sum += SQR(v[m]);
				resp[n][k] = -0.5*(sum + lndets[k]) + log(frac[k]);
			}
		}
		loglike = 0;
		for (n=0;n<nn;n++) {
			max = -99.9e99;
			for (k=0;k<kk;k++) if (resp[n][k] > max) max = resp[n][k];
			for (sum=0.,k=0; k<kk; k++) sum += exp(resp[n][k]-max);
			tmp = max + log(sum);
			for (k=0;k<kk;k++) resp[n][k] = exp(resp[n][k] - tmp);
			loglike +=tmp;
		}
		return loglike - oldloglike;
	}
*/


/*
EStepKernel
__global__ void estep_kernel(float* _resp_, float* _frac_, 
                             float* _data_, float* _means_, 
                             float* _sig_,  float* _lndets_,
                             const unsigned int num_clusts, 
                             const unsigned int num_dims, 
                             const unsigned int num_data)

GpuGaumixmod(MatSing& ddata, Mat3dSing& mmeans)

** n_tries = mmeans.dim1()  ?? 

_resp_
_frac_
_data_      
_means_
_sig_
_lndets_
num_clusts
num_dims
num_data

*/

    
    for (int i = 0; i < iterations ; i++)
	{
		// Create Ports and Tasks for the E and M steps for this iteration.

		// Connect the output of the E step to the input of the M step.

		// If this is the first iteration, create a graph input channel and connect the 
		// E step's input to it.
		// Else, connect the E step's input to the output of the M step from the previous iteration.

		// If this is the last iteration, create a graph output channel and connect the 
		// M step's input to it.
	}

	return pGraph;
}

void GPU_EM(MatDoub& data, MatDoub& means, int iterations)
{


/*
    // Parameters. Hardcoded for now.
	int modelComponents = 1;
	const int MAX_SAMPLES_PER_DISTRIBUTION = 10000;
	const int MAX_COMPONENTS_PER_MODEL = 10;



	// Get Datablock templates for the kernel parameters.
	DatablockTemplate * distributionTemplate = PTask::Runtime::GetDatablockTemplate(
		"db_distribution", sizeof(float), MAX_SAMPLES_PER_DISTRIBUTION, 1, 1);
	DatablockTemplate * modelTemplate = PTask::Runtime::GetDatablockTemplate(
		"db_model", sizeof(ModelComponent), MAX_COMPONENTS_PER_MODEL, 1, 1);


	// Build graph for the specified number of iterations of EM.
	// Returns pointers to the graph inputs and output channels.

	Graph * pGraph = BuildEMGraph(
		EStepKernel, MStepKernel, 
		distributionTemplate, modelTemplate,
		iterations);

*/

/*
    Gaumixmod* gmmcpu = new Gaumixmod(data, means);

    cout << endl << "Data stats:" << endl;
    cout << "  num data points (nn) = " << gmmcpu->nn << endl;
    cout << "  num components (kk)  = " << gmmcpu->kk << endl;
    cout << "  num dimensions (mm)  = " << gmmcpu->mm << endl;

    cout << "Initial means = ";
    for (int k=0; k<means.nrows(); k++)
    {
        cout << means[k][0] << " ";
    }
    cout << endl;

    double delta;
    for (int i=0; i<iterations; i++)
    {
        delta = gmmcpu->estep();
        cout << "Iteration " << i + 1 << " : delta = " << delta << endl;
        gmmcpu->mstep();

        cout << "  means = ";
        for (int k=0; k<means.nrows(); k++)
        {
            cout << gmmcpu->means[k][0] << " ";
        }
        cout << endl;
    }
*/

}



int run_graph_emgmm_task(
    char * szfile, 
	char * szshader)
{
    
	// Load distributions from data files.
	const char * distribution1DataFile = "..\\..\\..\\accelerators\\PTaskUnitTest\\distributions\\dist_1.csv"; // One Gaussian
	const char * distribution2DataFile = "..\\..\\..\\accelerators\\PTaskUnitTest\\distributions\\dist_2.csv"; // Two Guassians
	const char * distribution3DataFile = "..\\..\\..\\accelerators\\PTaskUnitTest\\distributions\\dist_3.csv"; // Three Gaussians
	Distribution * d1 = LoadDistriutionFromCSVFile(distribution1DataFile); // rnorm(n=600,mean = 5,sd =0.5)
	Distribution * d2 = LoadDistriutionFromCSVFile(distribution2DataFile); // d1 + rnorm(n=300, mean = 9, sd = 0.5)
	Distribution * d3 = LoadDistriutionFromCSVFile(distribution3DataFile); // d2 + rnorm(n=100, mean = 15, sd = 1.0)

    MatDoub data1((int)d1->size(), 1, &(*d1)[0]);
    MatDoub data2((int)d2->size(), 1, &(*d2)[0]);
    MatDoub data3((int)d3->size(), 1, &(*d3)[0]);
/*    cout << "Data begins:" << endl;
    for (int i=0; i<10; i++)
    {
        cout << "data[" << i << "][0] = " << data[i][0] << endl;
    }
*/
    MatDoub means1(1, 1);
    means1[0][0] = 7.0;

    MatDoub means2(2, 1);
    means2[0][0] = 7.0;
    means2[1][0] = 12.0;

    MatDoub means3(3, 1);
// 7, 12, 25 -> means = -1.#IND -1.#IND -1.#IND  (with data3)
// 5, 9, 25 -> ERROR: Cholesky failed cholesky.h:13
// 5, 9, 15 (i.e. the actual values) works
// 5, 9, 20 works
    means3[0][0] = 5.0;
    means3[1][0] = 9.0;
    means3[2][0] = 20.0;

    CPU_EM(data3, means1, 20);
    CPU_EM(data3, means2, 20);
    CPU_EM(data3, means3, 20);

	return 0;

}
