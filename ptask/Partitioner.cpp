///-------------------------------------------------------------------------------------------------
// file:	Partitioner.cpp
//
// summary:	Implements the partitioner class
///-------------------------------------------------------------------------------------------------

#include "graph.h"
#include "Partitioner.h"
#include "ptaskutils.h"
#include "PTaskRuntime.h"

#include <stdio.h>
#include <math.h>
#include "assert.h"

// The partitioner files produce lots of warnings. Suppress them for now...
#pragma warning(push)
// Suppress warning C4100: 'x' : unreferenced formal parameter
#pragma warning( disable : 4100 )
// Suppress warning C4127: conditional expression is constant
#pragma warning( disable : 4127 )
// Suppress warning C4189: 'verbose' : local variable is initialized but not referenced
#pragma warning( disable : 4189 )
// Suppress warning warning C4239: nonstandard extension used
#pragma warning( disable : 4239 )
// Suppress warning warning C4996: 'fopen': This function or variable may be unsafe.
#pragma warning( disable : 4996 )

#include "bisolver.h"
#include "graph_daniel.h"

// Stop warning suppression specific to those files.
#pragma warning(pop)

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   jcurrey, 2/1/2014. </remarks>
    ///
    /// TODO JC params
    ///-------------------------------------------------------------------------------------------------

    Partitioner::Partitioner(
        Graph * graph,
        int numPartitions, 
        const char * workingDir, 
        const char * fileNamePrefix
        )
        : m_graph(graph), m_numPartitions(numPartitions), m_workingDir(workingDir), m_fileNamePrefix(fileNamePrefix)
    { }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   jcurrey, 2/1/2014. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Partitioner::~Partitioner(
        void
        )
    {
        m_graph = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Partition the ptask graph into the specified number of partitions. 
    ///             If successful, return true.
    ///
    ///             Currently only 2 partitions are supported.
    ///
    /// <remarks>   jcurrey, 2/1/2014. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    Partitioner::Partition(
        void
        )
    {
//#define HARD_CODE_DANDELION_ROOT
#ifdef HARD_CODE_DANDELION_ROOT 
        char * dandelionRootStr = "D:\\src\\svc\\Dandelion";
#else
        size_t len;
        char * dandelionRootStr = NULL;
        errno_t err = _dupenv_s(&dandelionRootStr, &len, "DANDELION_ROOT"); // Safe version of getenv().
        if (err || dandelionRootStr == NULL)
        {
            PTask::Runtime::HandleError(
                "%s::%s: DANDELION_ROOT must be set in environment, to determine the path to the partitioner executable!\n",
                 __FILE__, __FUNCTION__);
            return FALSE;
        }
#endif
        std::string partitionerExeDir = dandelionRootStr;
        partitionerExeDir += "\\accelerators\\ptask\\partitioning\\standalone";

		// Make sure the working dir exists.
		if (!CreateDirectoryA(m_workingDir.c_str(), NULL) &&
			ERROR_ALREADY_EXISTS != GetLastError())
		{
            PTask::Runtime::HandleError(
                "%s::%s: Error trying to create working dir for partitioner: %s!\n",
                 __FILE__, __FUNCTION__, m_workingDir.c_str());
			return FALSE;
		}

        // Write file containing input to the partitioner.
        std::string inputFileName = m_workingDir + "\\" + m_fileNamePrefix + ".partition_input.txt";
        int numTasks = m_graph->WriteWeightedModel((char*)inputFileName.c_str());

        // Run the partitioner.
        // Do this via a script which takes the .exe's output (fixed name solution.txt in same dir) 
        // and copies it to the desired dir and filename.
        char szCommandLine[2048];
		sprintf_s(szCommandLine, 2048, 
			"%s\\partitioner.bat %s %s %s %d", 
			(char*)partitionerExeDir.c_str(), (char*)partitionerExeDir.c_str(),
			(char*)m_workingDir.c_str(), (char*)m_fileNamePrefix.c_str(), m_numPartitions);
        printf("Running partitioner script:\n%s\n", szCommandLine);
        system(szCommandLine);

        // Read the solution into an array. Checks that the number of values read matches the expected.
        int * partitionArray = new int[numTasks];
        std::string outputFileName = m_workingDir + "\\" + m_fileNamePrefix + ".partition_output.txt";
        if (!this->ReadSolutionFile((char*)outputFileName.c_str(), numTasks, partitionArray))
        {        
            return FALSE;
        }

        // Set the explicit partitioning for this graph.
        m_graph->SetExplicitPartition(partitionArray, numTasks);

#ifndef HARD_CODE_DANDELION_ROOT 
        free(dandelionRootStr);
#endif

        return TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Read the partitioner's solution from a file into an array. </summary>
    ///
    /// <remarks>   jcurrey, 2/1/2014. </remarks>
    ///
    /// TODO JC params
    ///-------------------------------------------------------------------------------------------------
    BOOL
    Partitioner::ReadSolutionFile(
        const char * fileName,
        int expectedNumValues,
        int * valueArray
        )
    {
    	vector<int> values;
    
    	ifstream in(fileName, ios::in);
    	int value;
        while (in >> value)
        {
		    values.push_back(value);
    	}
    	in.close();

        if (values.size() != expectedNumValues)
        {
            PTask::Runtime::HandleError(
                "%s::%s: Expected partitioner solution to contain %d values but it contained %d!\n",
                 __FILE__, __FUNCTION__, expectedNumValues, values.size());
            return FALSE;
        }
    
    	for (int i=0; i<values.size(); i++) {
    		valueArray[i] = values[i];
    	}
        return TRUE;
    }

#ifdef USE_GRAPH_PARTITIONER_DLL
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Default constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/10/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Partitioner::Partitioner(
        Graph * pGraph
        )
    {
        m_pGraph = pGraph;
        m_pSolution = NULL;
        m_bSolutionValid = FALSE;
        m_nSolutionValue = -1;
        m_nSolutionEvaluation = -1;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/10/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Partitioner::~Partitioner(
        void
        )
    {
        if(m_pSolution) 
            delete [] m_pSolution;
        m_pGraph = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Partition the ptask graph into nPartition. If successful, return true, and set
    ///             nSolutionValue and nSolutionEvaluation, which are (somewhat obscure)
    ///             metrics of the quality of the solution.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 12/10/2013. </remarks>
    ///
    /// <param name="nPartitions">          The partitions. </param>
    /// <param name="nSolutionValue">       [out] The solution value. </param>
    /// <param name="nSolutionEvaluation">  [out] The solution evaluation. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Partitioner::Partition(
        __in    int nPartitions, 
        __inout int& nSolutionValue, 
        __inout int& nSolutionEvaluation
        )
    {
        nSolutionValue = m_nSolutionValue;
        nSolutionEvaluation = m_nSolutionEvaluation;
        if(m_pGraph == NULL || nPartitions > 2 || nPartitions < 0) {
            PTask::Runtime::HandleError("%s::%s(%d): requested %d partitions--only 2 are currently supported!\n",
                                        __FILE__,
                                        __FUNCTION__,
                                        nPartitions);
            return FALSE;
        }

        if(m_pSolution != NULL && m_bSolutionValid) 
            return TRUE;

	    config_daniel conf;
	    BisectionSolver solver;
	    BisectionGraph g; 
        {
            UINT uiEdgeWeight = 10;
            UINT uiVertexWeight = 5;

            int nodeId=0;
            std::map<Task*, int> nodeNumberMap;    
            std::map<int, Task*> nodeNumberBackMap;    
            std::map<int, Task*>::iterator ti;
            std::map<std::string, Task*>::iterator iter;
            for(iter=m_pGraph->m_vTasks.begin(); iter!=m_pGraph->m_vTasks.end() ; ++iter) {
                ++nodeId; // index from 1, not 0!
                Task * pTask = iter->second;
                assert(pTask != NULL);
                nodeNumberMap[pTask] = nodeId;
                nodeNumberBackMap[nodeId] = pTask;
            }

            // The first line just specifies the graph size and the format (weights on vertices and costs
            // on edges): <n> <m> 11. n == number of tasks, m = number of channels. If we are ignoring 
            // back-edges, finding m involves traversing the graph to look for cycles unfortunately.
        
            size_t nVertices = m_pGraph->m_vTasks.size();
            std::map<Task*, std::set<Task*>*>* pMap = NULL;
            UINT nEdges = m_pGraph->GetAdjacencyMap(pMap, TRUE);
        
            g.set_n((int)nVertices);

            // This is followed by n lines, each describing the adjacency list of one vertex 
            // (in order). Each line is of the form:
            // <vertex_weight> <v1> <ew1> <v2> <ew2> … <vk,ewk>

            for(ti=nodeNumberBackMap.begin(); ti!=nodeNumberBackMap.end(); ti++) {

                int nTaskId = ti->first;
                Task * pTaskNode = ti->second;
                assert(pTaskNode != NULL);
                std::map<Task*, std::set<Task*>*>::iterator mi = pMap->find(pTaskNode);

                // vertex weight
                g.set_weight(nTaskId, uiVertexWeight);

                // adjacencies and per-adjacency weights
                std::set<Task*>::iterator si;
                std::set<Task*>* pTaskSet = mi->second;
                for(si=pTaskSet->begin(); si!=pTaskSet->end(); si++) {

                    Task * pAdjacentTask = *si;
                    assert(pAdjacentTask != NULL);
                    #ifdef DEBUG
                        // if this task is adjacent, then pTaskNode should also be in
                        // the adjacency list for the task: pAdjacentNode
                        std::map<Task*, std::set<Task*>*>::iterator ni = pMap->find(pAdjacentTask);
                        assert(ni != pMap->end());
                        assert(ni->second != NULL);
                        assert(ni->second->find(pTaskNode) != ni->second->end());
                    #endif
                    int nAdjacentId = nodeNumberMap[pAdjacentTask];
                    g.add_edge(nTaskId, nAdjacentId, uiEdgeWeight);
                }
            }

            m_pGraph->FreeAdjacencyMap(pMap);
        }

	    PTask::Runtime::MandatoryInform("Modeled PTask graph %s as BisectionGraph with %d edges.\n", 
                                        m_pGraph->GetName(),
                                        g.EdgeCount());

	    int solvalue;
	    vector<int> solution = solver.FindBisection(g, conf, m_nSolutionValue);
	    PTask::Runtime::Inform("Found a solution with value %d.\n", m_nSolutionValue);
	    m_nSolutionEvaluation = BisectionSolver::EvaluateSolution(g, solution);
	    PTask::Runtime::Inform("\nSolution: %d : %d.\n", m_nSolutionValue, m_nSolutionEvaluation);

        int i;
        vector<int>::iterator vi;
        m_pSolution = new int[solution.size()];
        for(i=0, vi=solution.begin(); vi!=solution.end(); vi++, i++)
            m_pSolution[i] = *vi;
        m_bSolutionValid = TRUE;
        nSolutionValue = m_nSolutionValue;
        nSolutionEvaluation = m_nSolutionEvaluation;
        return m_bSolutionValid;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Assign the partition created by a successful call to Partition to the 
    ///             underlying PTask graph. </summary>
    ///
    /// <remarks>   Crossbac, 12/10/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Partitioner::AssignPartition(
        void
        )
    {
        if(!m_bSolutionValid || m_pGraph == NULL || m_pSolution == NULL)
            return FALSE;
        m_pGraph->SetExplicitPartition(m_pSolution, (UINT)m_pGraph->m_vTasks.size());
        return TRUE;
    }
#endif // USE_GRAPH_PARTITIONER_DLL
};
