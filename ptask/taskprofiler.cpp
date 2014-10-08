//--------------------------------------------------------------------------------------
// File: Task.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "TaskProfiler.h"
#include "task.h"
#include "hrperft.h"
#include "shrperft.h"
#include <map>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <assert.h>
using namespace std;

namespace PTask {

    CSharedPerformanceTimer *          TaskProfile::m_pGlobalProfileTimer = NULL;
    ULONG                              TaskProfile::m_nInputBindEvents = 0;
    ULONG                              TaskProfile::m_nInputMigrations = 0;
    UINT                               TaskProfile::m_nMetrics;
    std::map<UINT, std::string>        TaskProfile::m_vMetricOrder;
    std::map<std::string, std::string> TaskProfile::m_vMetricNickNames;
    std::stringstream                  TaskProfile::m_ssTaskStats;
    std::stringstream                  TaskProfile::m_ssTaskDispatchHistory;
    CRITICAL_SECTION                   TaskProfile::m_csTaskProfiler;
    BOOL                               TaskProfile::m_bProfilerOutputTabular;
    BOOL                               TaskProfile::m_bTaskProfilerInit = FALSE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/17/2013. </remarks>
    ///
    /// <param name="pTask">    The task. </param>
    ///-------------------------------------------------------------------------------------------------

    TaskProfile::TaskProfile(
        Task * pTask
        ) 
    {
        m_pTask = pTask;
        InitializeInstanceProfile();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/17/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    TaskProfile::~TaskProfile(
        VOID
        ) 
    {
        DeinitializeInstanceProfile();
    }
            
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Print migration statistics. </summary>
    ///
    /// <remarks>   crossbac, 5/21/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    TaskProfile::MigrationReport(
        std::ostream& ss
        )
    {
        double dAvg = 0;
        if(m_nInputBindEvents != 0) {
            dAvg = ((double) m_nInputMigrations/(double) m_nInputBindEvents) * 100.0;
        }
        if(PTask::Runtime::IsVerbose() || PTask::Runtime::GetCoherenceProfileMode()) {            
            ss << "migration-data: "  
               << m_nInputBindEvents << ", "
               << m_nInputMigrations << ", avg=" 
               << dAvg << std::endl;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the task profiling. </summary>
    ///
    /// <remarks>   Crossbac, 9/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    TaskProfile::Initialize(
        BOOL bTabularOutput
        )
    {
        if(!PTask::Runtime::GetTaskProfileMode()) return;
        if(m_bTaskProfilerInit)  return;
        if(!m_pGlobalProfileTimer) {
            m_pGlobalProfileTimer = new CSharedPerformanceTimer(gran_msec, true);
        }
        InitializeCriticalSection(&m_csTaskProfiler);
        m_bProfilerOutputTabular = bTabularOutput;
        EnterCriticalSection(&m_csTaskProfiler);
        m_ssTaskStats << "";
        m_nMetrics = 0;

        #define tpprofile_init_map_nickname(a,x,y) {                                   \
            m_vMetricNickNames.insert(std::pair<std::string, std::string>(x, #y));     \
            m_vMetricOrder.insert(std::pair<UINT, std::string>(a,x));                  \
            a++; }

        tpprofile_init_map_nickname(m_nMetrics, "disp", Dispatch);
        tpprofile_init_map_nickname(m_nMetrics, "psdisp", PSDispatch);
        tpprofile_init_map_nickname(m_nMetrics, "mig", MigrateInputs);
        tpprofile_init_map_nickname(m_nMetrics, "bndI", BindInputs);
        tpprofile_init_map_nickname(m_nMetrics, "bndO", BindOutputs);
        tpprofile_init_map_nickname(m_nMetrics, "bndM", BindMetaPorts);
        tpprofile_init_map_nickname(m_nMetrics, "bndC", BindConstants);
        tpprofile_init_map_nickname(m_nMetrics, "blkR", BlockedOnReadyQ);
        tpprofile_init_map_nickname(m_nMetrics, "lck", AcquireDispatchResourceLocks);
        tpprofile_init_map_nickname(m_nMetrics, "unlck", ReleaseDispatchResourceLocks);
        tpprofile_init_map_nickname(m_nMetrics, "prDF", PropagateDataflow);
        tpprofile_init_map_nickname(m_nMetrics, "relIB", ReleaseInflightDatablocks);
        tpprofile_init_map_nickname(m_nMetrics, "RBmv", RIBMaterializeViews);
        tpprofile_init_map_nickname(m_nMetrics, "RBsv", RIBSyncHost);
        tpprofile_init_map_nickname(m_nMetrics, "iolck", AssembleIOLockList);
        tpprofile_init_map_nickname(m_nMetrics, "dep", AssignDependentAccelerator);
        tpprofile_init_map_nickname(m_nMetrics, "td", DispatchTeardown);
        tpprofile_init_map_nickname(m_nMetrics, "blkNR", BlockedNotReady);
        tpprofile_init_map_nickname(m_nMetrics, "schd", Schedule);

        m_bTaskProfilerInit = TRUE;
        LeaveCriticalSection(&m_csTaskProfiler);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deinitialize task profiling. </summary>
    ///
    /// <remarks>   Crossbac, 9/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    TaskProfile::Deinitialize(
        VOID
        )
    {
        if(!m_bTaskProfilerInit) return;
        if(!PTask::Runtime::GetTaskProfileMode()) return;
        EnterCriticalSection(&m_csTaskProfiler);
        if(m_pGlobalProfileTimer) {
            delete m_pGlobalProfileTimer;
            m_pGlobalProfileTimer = NULL;
        }
        m_vMetricNickNames.clear();
        m_vMetricOrder.clear();
        LeaveCriticalSection(&m_csTaskProfiler);
        DeleteCriticalSection(&m_csTaskProfiler);
        m_bTaskProfilerInit = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes the task profiling. </summary>
    ///
    /// <remarks>   Crossbac, 9/21/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    TaskProfile::InitializeInstanceProfile(
        VOID
        )
    {
        if(!PTask::Runtime::GetTaskProfileMode()) return;
        InitializeCriticalSection(&m_csTiming);

        #define tpprofile_init_map(x) {                                                \
            m_vEnterProfileMap.insert(                                                 \
                std::pair<std::string,                                                 \
                            std::map<int, std::vector<double>*>&>(#x, m_vEnter##x));   \
            m_vExitProfileMap.insert(                                                  \
                std::pair<std::string,                                                 \
                            std::map<int, std::vector<double>*>&>(#x, m_vExit##x)); }

        tpprofile_init_map(AcquireDispatchResourceLocks);
        tpprofile_init_map(ReleaseDispatchResourceLocks);
        tpprofile_init_map(MigrateInputs);
        tpprofile_init_map(AssembleIOLockList);
        tpprofile_init_map(Schedule);
        tpprofile_init_map(BlockedOnReadyQ);
        tpprofile_init_map(BlockedNotReady);
        tpprofile_init_map(PropagateDataflow);
        tpprofile_init_map(ReleaseInflightDatablocks);
        tpprofile_init_map(RIBMaterializeViews);
        tpprofile_init_map(RIBSyncHost);
        tpprofile_init_map(BindMetaPorts);
        tpprofile_init_map(Dispatch);
        tpprofile_init_map(PSDispatch);
        tpprofile_init_map(BindConstants);
        tpprofile_init_map(BindOutputs);
        tpprofile_init_map(BindInputs);
        tpprofile_init_map(AssignDependentAccelerator);
        tpprofile_init_map(DispatchTeardown);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deinitialize task profiling. </summary>
    ///
    /// <remarks>   Crossbac, 9/21/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    TaskProfile::DeinitializeInstanceProfile(
        VOID
        )
    {
        if(!PTask::Runtime::GetTaskProfileMode()) return;
        tpprofile_destroy(AcquireDispatchResourceLocks);
        tpprofile_destroy(ReleaseDispatchResourceLocks);
        tpprofile_destroy(MigrateInputs);
        tpprofile_destroy(AssembleIOLockList);
        tpprofile_destroy(Schedule);
        tpprofile_destroy(BlockedOnReadyQ);
        tpprofile_destroy(BlockedNotReady);
        tpprofile_destroy(PropagateDataflow);
        tpprofile_destroy(ReleaseInflightDatablocks);
        tpprofile_destroy(RIBMaterializeViews);
        tpprofile_destroy(RIBSyncHost);
        tpprofile_destroy(BindMetaPorts);
        tpprofile_destroy(Dispatch);
        tpprofile_destroy(PSDispatch);
        tpprofile_destroy(BindConstants);
        tpprofile_destroy(BindOutputs);
        tpprofile_destroy(BindInputs);
        tpprofile_destroy(AssignDependentAccelerator);
        tpprofile_destroy(DispatchTeardown);
        m_vEnterProfileMap.clear();
        m_vExitProfileMap.clear();
        DeleteCriticalSection(&m_csTiming);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps a task profile statistics. </summary>
    ///
    /// <remarks>   Crossbac, 9/21/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    TaskProfile::DumpTaskProfile(
        std::ostream &ios
        )
    {
        if(!PTask::Runtime::GetTaskProfileMode()) return;
        if(!m_bTaskProfilerInit) return;
        std::stringstream * pss = GetTaskProfile();
        ios << pss->str();
        delete pss;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return a string stream object with stats for the task. </summary>
    ///
    /// <remarks>   Crossbac, 9/21/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    std::stringstream * 
    TaskProfile::GetTaskProfile(
        VOID
        )
    {
        if(!PTask::Runtime::GetTaskProfileMode()) return NULL;
        if(!m_bTaskProfilerInit) return NULL;
        return (m_bProfilerOutputTabular) ? 
            GetTaskProfileTabular() :
            GetTaskProfileColumnar();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return a string stream object with stats for the task. </summary>
    ///
    /// <remarks>   Crossbac, 9/21/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    std::stringstream * 
    TaskProfile::GetTaskProfileColumnar(
        VOID
        )
    {
        std::stringstream * pss = new std::stringstream();
        std::stringstream &ss = *pss;
        if(!PTask::Runtime::GetTaskProfileMode() || !m_bTaskProfilerInit) 
            return pss;
        
        int nColumnFillWidth = 30;
        int nAverageFillWidth = 10;
        EnterCriticalSection(&m_csTiming);
        ss << m_pTask->m_lpszTaskName << " stats: " << endl;

        vector<double>::iterator vi;
        vector<double>::iterator evi;
        map<int, vector<double>*>::iterator ti;
        map<int, vector<double>*>::iterator eti;
        map<std::string, map<int,vector<double>*>&>::iterator mi;
        map<std::string, map<int,vector<double>*>&>::iterator exiti;
        for(mi=m_vEnterProfileMap.begin(); mi!=m_vEnterProfileMap.end(); mi++)  {
            std::string strMetric = mi->first;
            exiti = m_vExitProfileMap.find(strMetric);
            assert(exiti != m_vExitProfileMap.end());
            std::map<int, vector<double>*>& vEnterData = mi->second;
            std::map<int, vector<double>*>& vExitData = exiti->second;
            int nEntries = 0;
            double dCum = 0.0;
            for(ti=vEnterData.begin(); ti!=vEnterData.end(); ti++) {
                nEntries++;
                int nDispatchId = ti->first;
                eti = vExitData.find(nDispatchId);
                assert(eti != vExitData.end());
                assert(eti->second->size() == ti->second->size());
                for(vi=ti->second->begin(), evi=eti->second->begin(); 
                    vi!=ti->second->end() && evi!=eti->second->end(); 
                    vi++, evi++) {
                    dCum+=((*evi) - (*vi));
                }
            }
            double dAvg = nEntries ? dCum / nEntries : 0.0;
            ss.setf(std::ios_base::right, std::ios_base::adjustfield);
            ss.width(nColumnFillWidth); ss.fill(' '); ss  << strMetric << ": ";            
            ss.setf(std::ios_base::right, std::ios_base::adjustfield);
            ss.width(nAverageFillWidth); ss.fill(' '); ss << std::setprecision(2) << dAvg;
            ss << " (" << dCum << "/" << nEntries << ")" << endl;
        }

        LeaveCriticalSection(&m_csTiming);
        return pss;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return a string stream object with stats for the task. </summary>
    ///
    /// <remarks>   Crossbac, 9/21/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    std::stringstream * 
    TaskProfile::GetTaskProfileTabular(
        VOID
        )
    {
        stringstream * pss = new stringstream();
        stringstream &ss = *pss;
        if(!PTask::Runtime::GetTaskProfileMode() || !m_bTaskProfilerInit) 
            return pss;
        
        int nTaskNameFillWidth = 30;
        int nAverageFillWidth = 6;
        EnterCriticalSection(&m_csTiming);


        vector<double>::iterator vi;
        vector<double>::iterator evi;
        map<int, vector<double>*>::iterator ti;
        map<int, vector<double>*>::iterator eti;
        map<UINT, std::string>::iterator oni;
        map<std::string, map<int,vector<double>*>&>::iterator mi;
        map<std::string, map<int,vector<double>*>&>::iterator exiti;
        map<std::string, std::string>::iterator nni;
        ss.setf(std::ios_base::right, std::ios_base::adjustfield);
        char * szShortName = new char[nTaskNameFillWidth];
        memset(szShortName, 0, nTaskNameFillWidth);
        memcpy(szShortName, m_pTask->m_lpszTaskName, min(strlen(m_pTask->m_lpszTaskName), nTaskNameFillWidth-1));
        ss.width(nTaskNameFillWidth); ss.fill(' '); ss  << szShortName << ": ";
        delete [] szShortName;
        for(UINT nID=0; nID<m_nMetrics; nID++) {
            oni=m_vMetricOrder.find(nID);
            nni=m_vMetricNickNames.find(oni->second);
            mi=m_vEnterProfileMap.find(nni->second);

            std::string strMetric = mi->first;
            exiti = m_vExitProfileMap.find(strMetric);
            assert(exiti != m_vExitProfileMap.end());
            std::map<int, vector<double>*>& vEnterData = mi->second;
            std::map<int, vector<double>*>& vExitData = exiti->second;
            int nEntries = 0;
            double dCum = 0.0;
            for(ti=vEnterData.begin(); ti!=vEnterData.end(); ti++) {
                nEntries++;
                int nDispatchId = ti->first;
                eti = vExitData.find(nDispatchId);
                if(eti != vExitData.end()) {
                    // assert(eti->second->size() == ti->second->size());
                    for(vi=ti->second->begin(), evi=eti->second->begin(); 
                        vi!=ti->second->end() && evi!=eti->second->end(); 
                        vi++, evi++) {
                        dCum+=((*evi) - (*vi));
                    }
                }
            }
            double dAvg = nEntries ? dCum / nEntries : 0.0;
            ss.setf(std::ios_base::right, std::ios_base::adjustfield);
            ss.width(nAverageFillWidth); ss.fill(' '); 
            ss << std::setprecision(2) << std::fixed << dAvg << " ";
        }
        ss << std::endl;

        LeaveCriticalSection(&m_csTiming);
        return pss;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return a string stream object with dispatch history log info. </summary>
    ///
    /// <remarks>   Crossbac, 9/21/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    std::stringstream * 
    TaskProfile::GetDispatchHistory(
        VOID
        )
    {
        stringstream * pss = new stringstream();
        stringstream &ss = *pss;
        if(!PTask::Runtime::GetTaskProfileMode() || !m_bTaskProfilerInit) 
            return pss;
        
        EnterCriticalSection(&m_csTiming);

        vector<double>::iterator vi;
        vector<double>::iterator evi;
        map<int, vector<double>*>::iterator ti;
        map<int, vector<double>*>::iterator eti;
        map<UINT, std::string>::iterator oni;
        map<std::string, map<int,vector<double>*>&>::iterator mi;
        map<std::string, map<int,vector<double>*>&>::iterator exiti;
        map<std::string, std::string>::iterator nni;
        for(UINT nID=0; nID<m_nMetrics; nID++) {
            oni=m_vMetricOrder.find(nID);
            nni=m_vMetricNickNames.find(oni->second);
            mi=m_vEnterProfileMap.find(nni->second);

            std::string strMetric = mi->first;
            exiti = m_vExitProfileMap.find(strMetric);
            assert(exiti != m_vExitProfileMap.end());
            std::map<UINT, UINT>::iterator fi;
            std::map<int, vector<double>*>& vEnterData = mi->second;
            std::map<int, vector<double>*>& vExitData = exiti->second;
            int nEntries = 0;
            for(ti=vEnterData.begin(); ti!=vEnterData.end(); ti++) {
                nEntries++;
                int nDispatchId = ti->first;
                int nAccDispatchId = (nDispatchId == 0) ? nDispatchId+1:nDispatchId;
                assert(m_vDispatchAcceleratorHistory.find(nAccDispatchId) != m_vDispatchAcceleratorHistory.end());
                UINT nDispatchAccelerator = m_vDispatchAcceleratorHistory[nAccDispatchId];
                fi = m_vDependentAcceleratorHistory.find(nAccDispatchId);
                eti = vExitData.find(nDispatchId);
                if(eti != vExitData.end()) {
                    // assert(eti->second->size() == ti->second->size());
                    for(vi=ti->second->begin(), evi=eti->second->begin(); 
                        vi!=ti->second->end() && evi!=eti->second->end(); 
                        vi++, evi++) {
                        ss << strMetric << " "
                           << m_pTask->m_lpszTaskName 
                           << " Disp=" << nDispatchId
                           << " Acc=" << nDispatchAccelerator;
                        if(fi!=m_vDependentAcceleratorHistory.end()) {
                            ss << " [dep:"
                               << fi->second
                               << "]";
                        }
                        ss << " Ts=" << std::setprecision(6) << std::fixed << *vi 
                           << " Te=" << *evi 
                           << endl;
                    }
                }
            }
        }
        ss << std::endl;

        LeaveCriticalSection(&m_csTiming);
        return pss;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Merge task instance statistics. </summary>
    ///
    /// <remarks>   Crossbac, 9/24/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    TaskProfile::MergeTaskInstanceStatistics(
        VOID
        )
    {
        if(!PTask::Runtime::GetTaskProfileMode()) return;
        if(!m_bTaskProfilerInit) return;
        stringstream * pss = GetTaskProfile();
        stringstream * hss = GetDispatchHistory();
        EnterCriticalSection(&m_csTaskProfiler);
        if(pss != NULL) m_ssTaskStats << pss->str();
        if(hss != NULL) m_ssTaskDispatchHistory << hss->str();
        LeaveCriticalSection(&m_csTaskProfiler);
        if(pss != NULL) delete pss;
        if(hss != NULL) delete hss;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps a task profile statistics. </summary>
    ///
    /// <remarks>   Crossbac, 9/24/2012. </remarks>
    ///
    /// <param name="ss">   [in,out] The ss. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    TaskProfile::Report(
        std::ostream& ss
        )
    {
        if(!PTask::Runtime::GetTaskProfileMode()) return;
        if(!m_bTaskProfilerInit) return;
        int nTaskNameFillWidth = 30;
        int nAverageFillWidth = 6;
        map<UINT, std::string>::iterator oni;
        ss.setf(std::ios_base::right, std::ios_base::adjustfield);
        ss.width(nTaskNameFillWidth); ss.fill(' '); ss  << "TASK: ";
        for(UINT nID=0; nID<m_nMetrics; nID++) {
            oni=m_vMetricOrder.find(nID);
            ss.setf(std::ios_base::right, std::ios_base::adjustfield);
            ss.width(nAverageFillWidth); ss.fill(' '); ss << oni->second << " ";
        }
        ss << std::endl;
        ss << setfill('-') << setw(80) << "-" << std::endl;

        EnterCriticalSection(&m_csTaskProfiler);
        ss << m_ssTaskStats.str();
        if(PTask::Runtime::g_bTaskProfileVerbose)
            ss << m_ssTaskDispatchHistory.str();
        LeaveCriticalSection(&m_csTaskProfiler);
    }

};

