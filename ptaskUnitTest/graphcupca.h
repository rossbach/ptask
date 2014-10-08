//--------------------------------------------------------------------------------------
// File: graphcupca.h
// test CUDA Principal Component Analysis *using task graph*
//--------------------------------------------------------------------------------------
#ifndef _GRAPH_CUDA_PCA_H_
#define _GRAPH_CUDA_PCA_H_

#include "ptaskapi.h"

using namespace std;
using namespace PTask;

int run_graph_cuda_pca_task(
	char * szfile, 
	char * szshader, 
	int rows, 
	int cols,
	int iterations,
	int innerIterations,
	int components,
	BOOL pipelineInputs
	);

class PCAGraph {
public:

    PCAGraph();
    ~PCAGraph();

    bool BuildGraph(char * szfile, int M, int N, int K, int innerIterations, bool preNorm, bool shortNorm);
    double RunTest(double* X, unsigned int M, unsigned int N, unsigned int K, 
	               double* T, double* P, double* R, int iterations, bool pipelineInputs, bool preNorm);

private:
    void AddParamsChannel(PTask::GraphInputChannel* channel, PTask::Datablock* datablock);
	void PullOutput(unsigned int M, unsigned int N, unsigned int K, bool lastIteration, double* T, double* P, double* R);

    PTask::Graph *              m_ptaskGraph;

	PTask::GraphInputChannel *  m_TInputChannel;
	PTask::GraphInputChannel *  m_PInputChannel;
	PTask::GraphInputChannel *  m_RInputChannel;
	PTask::GraphInputChannel *  m_UInputChannel;
	PTask::GraphInputChannel *  m_LInputChannel;

	PTask::GraphOutputChannel * m_TOutputChannel;
	PTask::GraphOutputChannel * m_POutputChannel;
	PTask::GraphOutputChannel * m_ROutputChannel;
	PTask::GraphOutputChannel * m_UOutputChannel;

	PTask::DatablockTemplate *  m_TTemplate;
	PTask::DatablockTemplate *  m_PTemplate;
	PTask::DatablockTemplate *  m_RTemplate;
	PTask::DatablockTemplate *  m_UTemplate;
	PTask::DatablockTemplate *  m_LTemplate;

	PTask::DatablockTemplate *  m_DcopyParamsTemplate;
	PTask::DatablockTemplate *  m_DaxpyParamsTemplate;
	PTask::DatablockTemplate *  m_DgemvParamsTemplate;
	PTask::DatablockTemplate *  m_Dnrm2ParamsTemplate;
	PTask::DatablockTemplate *  m_DscalParamsTemplate;
	PTask::DatablockTemplate *  m_DgerParamsTemplate;

    int                         m_numParamsChannels;
	PTask::GraphInputChannel ** m_ParamsInputChannels;
	PTask::Datablock **         m_ParamsDatablocks;

};

#endif
