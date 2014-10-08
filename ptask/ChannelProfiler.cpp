///-------------------------------------------------------------------------------------------------
// file:	ChannelProfiler.cpp
//
// summary:	Implements the channel profiler class
///-------------------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "channel.h"
#include "channelprofiler.h"
#include "graph.h"
#include <assert.h>
#include <sstream>
#include <iomanip>
using namespace PTask::Runtime;
using namespace std;

namespace PTask {

    BOOL                ChannelProfiler::m_bChannelProfile = FALSE;
    BOOL                ChannelProfiler::m_bChannelProfileInit = FALSE;
    CRITICAL_SECTION    ChannelProfiler::m_csChannelStats;
    std::map<std::string, std::map<std::string, CHANNELSTATISTICS*>*>  ChannelProfiler::m_vChannelStats;

#ifdef PROFILE_CHANNELS
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Default constructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/18/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    ChannelProfiler::ChannelProfiler(
        Channel * pChannel
        )
    {
        m_pChannel = pChannel;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/18/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    ChannelProfiler::~ChannelProfiler(
        VOID
        )
    {
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes this object. </summary>
    ///
    /// <remarks>   Crossbac, 7/18/2013. </remarks>
    ///
    /// <param name="bEnable">  true to enable, false to disable. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    ChannelProfiler::Initialize(
        BOOL bEnable
        ) 
    { 
        m_bChannelProfileInit = FALSE;
        m_bChannelProfile = FALSE; 
        if(!m_bChannelProfileInit) {
            InitializeCriticalSection(&m_csChannelStats);
            m_bChannelProfileInit = TRUE;
        }
        m_bChannelProfile = bEnable; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   De-initialises this object and frees any resources it is using. </summary>
    ///
    /// <remarks>   Crossbac, 7/18/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    ChannelProfiler::Deinitialize(
        VOID
        ) 
    { 
        if(m_bChannelProfileInit && PTask::Runtime::GetChannelProfilingEnabled()) {

            EnterCriticalSection(&m_csChannelStats);
            std::map<std::string, std::map<std::string, CHANNELSTATISTICS*>*>::iterator mi;

            for(mi=m_vChannelStats.begin(); mi!=m_vChannelStats.end(); mi++) {

                std::map<std::string, CHANNELSTATISTICS*>* pGraphMap = mi->second;
                if(pGraphMap == NULL) continue;
                std::map<std::string, CHANNELSTATISTICS*>::iterator pi;
                for(pi=pGraphMap->begin(); pi!=pGraphMap->end(); pi++) {
                    std::string strChannelName = pi->first;
                    CHANNELSTATISTICS * pStats = pi->second;
                    if(pStats != NULL)
                        delete pStats;
                }
                pGraphMap->clear();
                delete pGraphMap;
            }

            m_vChannelStats.clear();
            LeaveCriticalSection(&m_csChannelStats);
            DeleteCriticalSection(&m_csChannelStats);
        }

        m_bChannelProfile = FALSE; 
        m_bChannelProfileInit = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reports the given ss. </summary>
    ///
    /// <remarks>   Crossbac, 7/18/2013. </remarks>
    ///
    /// <param name="ss">   [in,out] The ss. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    ChannelProfiler::Report(
        std::ostream& ss
        )
    {
        if(PTask::Runtime::GetChannelProfilingEnabled() && 
           m_bChannelProfileInit &&
           m_bChannelProfile) {

            int nChannelNameFillWidth = 30;
            int nMetricFillWidth = 6;
            int nFlagFillWidth = 5;

            EnterCriticalSection(&m_csChannelStats);
            std::map<std::string, std::map<std::string, CHANNELSTATISTICS*>*>::iterator mi;
            for(mi=m_vChannelStats.begin(); mi!=m_vChannelStats.end(); mi++) {

                std::string strGraphName = mi->first;
                std::map<std::string, CHANNELSTATISTICS*>* pGraphMap = mi->second;
                ss << "Channel stats for " << strGraphName << ": " << std::endl;
                ss << setfill('-') << setw(80) << "-" << std::endl;
                ss.setf(std::ios_base::right, std::ios_base::adjustfield);
                ss.width(nChannelNameFillWidth); ss.fill(' '); ss  << "CHANNEL";
                ss.width(nMetricFillWidth); ss.fill(' '); ss  << "XMIT";
                ss.width(nMetricFillWidth); ss.fill(' '); ss  << "CAP";
                ss.width(nMetricFillWidth); ss.fill(' '); ss  << "MAX";
                ss.width(nMetricFillWidth); ss.fill(' '); ss  << "AVG";
                ss.width(nFlagFillWidth); ss.fill(' '); ss  << "XLIM";
                ss.width(nFlagFillWidth); ss.fill(' '); ss  << "POOL";

                ss << setfill('-') << setw(80) << "-" << std::endl;

                std::map<std::string, CHANNELSTATISTICS*>::iterator pi;
                for(pi=pGraphMap->begin(); pi!=pGraphMap->end(); pi++) {
                    std::string strChannelName = pi->first;
                    CHANNELSTATISTICS * pStats = pi->second;

                    double dAvgOcc = 
                        (pStats->uiBlocksDelivered ? 
                          (((double)pStats->uiCumulativeOccupancy/(double)pStats->uiBlocksDelivered)) :
                          (0.0));
                    std::stringstream ssTransitLimit;
                    ssTransitLimit << (pStats->uiBlockTransitLimit > 0);
                    std::string strTransitLimit = (pStats->uiBlockTransitLimit > 0)? ssTransitLimit.str() : "--";
                    std::string strBlookPool = pStats->bPoolOwner ? "**":"";
                    ss.setf(std::ios_base::right, std::ios_base::adjustfield);
                    ss.width(nChannelNameFillWidth); ss.fill(' '); ss  << mi->first;
                    ss.width(nMetricFillWidth); ss.fill(' '); ss  << pStats->uiBlocksDelivered;
                    ss.width(nMetricFillWidth); ss.fill(' '); ss  << pStats->uiCapacity;
                    ss.width(nMetricFillWidth); ss.fill(' '); ss  << pStats->uiMaxOccupancy;
                    ss.width(nMetricFillWidth); ss.fill(' '); ss << std::setprecision(2) << std::fixed << dAvgOcc;
                    ss.width(nFlagFillWidth); ss.fill(' '); ss  << strTransitLimit;
                    ss.width(nFlagFillWidth); ss.fill(' '); ss  << strBlookPool;
                }
            }
            LeaveCriticalSection(&m_csChannelStats);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Merge instance statistics. </summary>
    ///
    /// <remarks>   Crossbac, 7/18/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    ChannelProfiler::MergeInstanceStatistics(
        VOID
        )
    {
        if(PTask::Runtime::GetChannelProfilingEnabled() && 
           m_bChannelProfileInit &&
           m_bChannelProfile) {

            EnterCriticalSection(&m_csChannelStats);
            
            assert(m_pChannel->GetGraph()!= NULL);
            Graph * pGraph = m_pChannel->GetGraph();
            std::string strGraphName(pGraph->GetName());
            std::map<std::string, CHANNELSTATISTICS*>* pGraphMap = NULL;
            std::map<std::string, std::map<std::string, CHANNELSTATISTICS*>*>::iterator mi;
            mi=m_vChannelStats.find(strGraphName);
            if(mi==m_vChannelStats.end()) {
                pGraphMap = new std::map<std::string, CHANNELSTATISTICS*>();
                m_vChannelStats[strGraphName] = pGraphMap;
            } else {
                pGraphMap = mi->second;
            }
            assert(pGraphMap != NULL);

            CHANNELSTATISTICS * pStats = NULL;
            std::string strChannelName(m_pChannel->GetName());
            std::map<std::string, CHANNELSTATISTICS*>::iterator pi;
            pi=pGraphMap->find(strChannelName);
            if(pi==pGraphMap->end()) {
                pStats = new CHANNELSTATISTICS();
                (*pGraphMap)[strChannelName] = pStats;
            } else {
                pStats = pi->second;
            }
            m_pChannel->Lock();
            pStats->Update(m_pChannel);
            m_pChannel->Unlock();

            LeaveCriticalSection(&m_csChannelStats);
        }
    }

#else
// profiling mode not built in to ptask. Do nothing.
ChannelProfiler::ChannelProfiler(Channel * pChannel) { pChannel; m_pChannel = NULL; assert(FALSE); }
ChannelProfiler::~ChannelProfiler() {}
void ChannelProfiler::Initialize(BOOL bEnable) { bEnable; assert(!bEnable); }
void ChannelProfiler::Deinitialize() { assert(!PTask::Runtime::GetChannelProfilingEnabled()); }
void ChannelProfiler::Report(std::ostream& ss) { ss; assert(!PTask::Runtime::GetDBProfilingEnabled()); }
void ChannelProfiler::MergeInstanceStatistics() { assert(FALSE); }
#endif

};
