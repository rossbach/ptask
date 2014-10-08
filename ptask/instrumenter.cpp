///-------------------------------------------------------------------------------------------------
// file:	instrumenter.cpp
//
// summary:	Implements the instrumenter class
///-------------------------------------------------------------------------------------------------

#include "instrumenter.h"
#include "shrperft.h"
#include "PTaskRuntime.h"
#include "ptprofsupport.h"
#include <assert.h>

namespace PTask {

    UINT           Instrumenter::m_bInitialized = FALSE;
    Instrumenter * Instrumenter::g_pInstrumenter = NULL;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Default constructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Instrumenter::Instrumenter(
        VOID
        ) :Lockable("AdHocInstrumenter")
    {
        m_bEnabled = PTask::Runtime::GetAdhocInstrumentationEnabled();
        m_pRTTimer = new CSharedPerformanceTimer(gran_msec, true);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Default constructor. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    Instrumenter::~Instrumenter(
        VOID
        ) 
    {
        Lock();
        m_bEnabled = FALSE;
        if(m_pRTTimer != NULL) {
            delete m_pRTTimer;
            m_pRTTimer = NULL;
        }
        Unlock();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initialize an the ad hoc instrumentation framework. Creates a singleton
    ///             instrumenter object.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Instrumenter::Initialize(
        VOID
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return FALSE;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return FALSE;
        if(!m_bInitialized && g_pInstrumenter == NULL) {
            g_pInstrumenter = new Instrumenter();
            g_pInstrumenter->Enable(Runtime::g_bAdhocInstrumentationEnabled);
            m_bInitialized = TRUE;
        }
        return m_bInitialized;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Shutdown the ad hoc instrumentation framework, destroys the singleton
    ///             instrumenter object.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Instrumenter::Destroy(
        VOID
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return FALSE;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return FALSE;
        if(m_bInitialized || g_pInstrumenter != NULL) {
            Instrumenter * pInstrumenter = g_pInstrumenter;            
            g_pInstrumenter = NULL;
            m_bInitialized = FALSE;
            delete pInstrumenter;
        }
        return !m_bInitialized;
    }

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Collect data point. </summary>
	///
	/// <remarks>	crossbac, 8/12/2013. </remarks>
	///
	/// <param name="strEventName">	   	[in,out] Name of the event. </param>
	///
	/// <returns>	. </returns>
	///-------------------------------------------------------------------------------------------------

	double 
	Instrumenter::CollectDataPoint(
		std::string& strEventName
		)
	{
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0.0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0.0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            return g_pInstrumenter->__CollectDataPoint(strEventName);
        }
		return 0.0;
	}

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Collect data point. </summary>
	///
	/// <remarks>	crossbac, 8/12/2013. </remarks>
	///
	/// <param name="strEventName">	   	[in,out] Name of the event. </param>
	///
	/// <returns>	. </returns>
	///-------------------------------------------------------------------------------------------------

	double 
	Instrumenter::CollectDataPoint(
		__in  std::string& strEventName,
        __out UINT& nSamples,
        __out double& dMin,
        __out double& dMax
		)
	{
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0.0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0.0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            return g_pInstrumenter->__CollectDataPoint(strEventName, nSamples, dMin, dMax);
        }
		return 0.0;
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reports all measured latencies and acknowledges any outstanding
    ///             (incomplete) measurments . </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="ss">   [in,out] The ss. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Instrumenter::Report(
        __inout std::ostream& ss
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            g_pInstrumenter->__Report(ss);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enables the instrumentation framework. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="bEnable">  true to enable, false to disable. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Instrumenter::Enable(
        __in BOOL bEnable
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return FALSE;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return FALSE;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            return g_pInstrumenter->__Enable(bEnable);
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the adhoc instrumentation framework is enabled. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <returns>   true if enabled, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Instrumenter::IsEnabled(
        VOID
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return FALSE;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return FALSE;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            return g_pInstrumenter->__IsEnabled();
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if a measurement matching 'strEventName' is in flight. In flight
    ///             means that a start sentinal has been pushed onto the outstanding stack
    ///             that has not been matched yet by a corresponding completion. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   true if in flight, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Instrumenter::IsInFlight(
        __in std::string& strEventName
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return FALSE;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return FALSE;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            return g_pInstrumenter->__IsInFlight(strEventName);
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if a measurement matching 'strEventName' is complete. Note that
    ///             because multiple measurements matching a given name can be tracked, it is 
    ///             possible for an event name to be both "in flight" and complete. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   true if complete, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Instrumenter::IsComplete(
        __in std::string& strEventName
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return FALSE;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return FALSE;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            return g_pInstrumenter->__IsComplete(strEventName);
        }
        return FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets nesting depth for the given event name. If the nest depth is 0 it means
    ///             there are no measurements with the given name in flight. A depth greater than 1
    ///             means there is a nested measurement with the same name. This idiom is likely best
    ///             avoided in potentially concurrent code, since the instrumenter handles nesting
    ///             with a stack, which makes it difficult to disambiguate end sentinels if they are
    ///             not ordered explicitly by the program.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   The nesting depth. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::GetNestingDepth(
        __in std::string& strEventName
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            return g_pInstrumenter->__GetNestingDepth(strEventName);
        }
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event start. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::RecordEventStart(
        std::string& strEventName
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            return g_pInstrumenter->__RecordEventStart(strEventName);
        }
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event complete. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::RecordEventComplete(
        __in std::string& strEventName
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            return g_pInstrumenter->__RecordEventComplete(strEventName);
        }
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event start for an event that should have only one start sentinel,
    ///             but for which concurrency implies non-determinism, so many threads may attempt
    ///             to record the same event start. The primary example of this scenario is
    ///             start of data processing in PTask, which occurs as soon as the first block
    ///             is pushed by the user. It is simplest to record this by calling the instrumenter
    ///             on every exposed call to Channel::Push, with all calls after the first ignored. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::RecordSingletonEventStart(
        __in std::string& strEventName
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            return g_pInstrumenter->__RecordSingletonEventStart(strEventName);
        }
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Record event complete an event that should have only one start sentinel, but for
    /// 			which concurrency implies non-determinism, so many threads may attempt to record
    /// 			the same event start. The primary example of this scenario is start of data
    /// 			processing in PTask, which occurs as soon as the first block is pushed by the
    /// 			user. It is simplest to record this by calling the instrumenter on every exposed
    /// 			call to Channel::Push, with all calls after the first ignored.
    /// 			</summary>
    ///
    /// <remarks>	Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName">		  	[in,out] Name of the event. </param>
    /// <param name="bRequireOutstanding">	(Optional) true to require an outstanding entry. Some
    /// 									stats (like first return-value materialization)
    /// 									are very difficult to capture unambiguously, because
    /// 									calls to record the event must be placed in common code
    /// 									paths. Calling with this parameter set to true allows the
    /// 									record call to fail without protest if the caller knows
    /// 									this to be such an event. </param>
    ///
    /// <returns>	the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------
    UINT 
    Instrumenter::RecordSingletonEventComplete(
        __in std::string& strEventName,
		__in BOOL bRequireOutstanding
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            return g_pInstrumenter->__RecordSingletonEventComplete(strEventName, bRequireOutstanding);
        }
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event start. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::RecordEventStart(
        __in char * lpszEventName
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            std::string strEventName(lpszEventName);
            return g_pInstrumenter->__RecordEventStart(strEventName);
        }
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event complete. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::RecordEventComplete(
        __in char * lpszEventName
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            std::string strEventName(lpszEventName);
            return g_pInstrumenter->__RecordEventComplete(strEventName);
        }
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event start. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::AccumulateEventLatency(
        __in char * lpszEventName,
        __in double dIncrement
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            std::string strEventName(lpszEventName);
            return g_pInstrumenter->__AccumulateEventLatency(strEventName, dIncrement);
        }
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event start. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::RecordCumulativeEventStart(
        __in char * lpszEventName
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            std::string strEventName(lpszEventName);
            return g_pInstrumenter->__RecordCumulativeEventStart(strEventName);
        }
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event complete. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::RecordCumulativeEventComplete(
        __in char * lpszEventName
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            std::string strEventName(lpszEventName);
            return g_pInstrumenter->__RecordCumulativeEventComplete(strEventName);
        }
        return 0;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event start for an event that should have only one start sentinel,
    ///             but for which concurrency implies non-determinism, so many threads may attempt
    ///             to record the same event start. The primary example of this scenario is
    ///             start of data processing in PTask, which occurs as soon as the first block
    ///             is pushed by the user. It is simplest to record this by calling the instrumenter
    ///             on every exposed call to Channel::Push, with all calls after the first ignored. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::RecordSingletonEventStart(
        __in char * lpszEventName
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            std::string strEventName(lpszEventName);
            return g_pInstrumenter->__RecordSingletonEventStart(strEventName);
        }
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Record event complete an event that should have only one start sentinel, but for
    /// 			which concurrency implies non-determinism, so many threads may attempt to record
    /// 			the same event start. The primary example of this scenario is start of data
    /// 			processing in PTask, which occurs as soon as the first block is pushed by the
    /// 			user. It is simplest to record this by calling the instrumenter on every exposed
    /// 			call to Channel::Push, with all calls after the first ignored.
    /// 			</summary>
    ///
    /// <remarks>	Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="lpszEventName">	    [in] Name of the event. </param>
    /// <param name="bRequireOutstanding">	(Optional) true to require an outstanding entry. Some
    /// 									stats (like first return-value materialization)
    /// 									are very difficult to capture unambiguously, because
    /// 									calls to record the event must be placed in common code
    /// 									paths. Calling with this parameter set to true allows the
    /// 									record call to fail without protest if the caller knows
    /// 									this to be such an event. </param>
    ///
    /// <returns>	the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::RecordSingletonEventComplete(
        __in char * lpszEventName,
		__in BOOL bRequireOutstanding
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            std::string strEventName(lpszEventName);
            return g_pInstrumenter->__RecordSingletonEventComplete(strEventName, bRequireOutstanding);
        }
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resets this object. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Instrumenter::Reset(
        VOID
        )
    {
        if(!Runtime::g_bAdhocInstrumentationSupported) return;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return;
        if(m_bInitialized && g_pInstrumenter != NULL) {
            g_pInstrumenter->__Reset();
        }
    }

#ifdef ADHOC_STATS

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Collect data point. </summary>
	///
	/// <remarks>	crossbac, 8/12/2013. </remarks>
	///
	/// <param name="strEventName">	   	[in,out] Name of the event. </param>
	///
	/// <returns>	. </returns>
	///-------------------------------------------------------------------------------------------------

	double 
	Instrumenter::__CollectDataPoint(
		std::string& strEventName
		)
    {
        UINT nSamples = 0;
        double dMin = 0.0;
        double dMax = 0.0;
        return __CollectDataPoint(strEventName, nSamples, dMin, dMax);
    }

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Collect data point. </summary>
	///
	/// <remarks>	crossbac, 8/12/2013. </remarks>
	///
	/// <param name="strEventName">	   	[in,out] Name of the event. </param>
	///
	/// <returns>	. </returns>
	///-------------------------------------------------------------------------------------------------

	double 
	Instrumenter::__CollectDataPoint(
		__in  std::string& strEventName,
        __out UINT& nSamples,
        __out double& dMin,
        __out double& dMax
		)
	{
        if(!Runtime::g_bAdhocInstrumentationSupported) return 0.0;
        if(!Runtime::g_bAdhocInstrumentationEnabled) return 0.0;

		double dResult = 0.0;
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {

            Lock();

            CumulativeEventMap::iterator dmi;
            dmi = m_vCumulativeEvents.find(strEventName);
            if(dmi != m_vCumulativeEvents.end()) {

                nSamples = std::get<0>(dmi->second);
                dResult = std::get<1>(dmi->second);
                dMin = std::get<2>(dmi->second);
                dMax = std::get<3>(dmi->second);

            } else {

                __FinalizeSingletons();
                std::map<std::string, std::vector<double>>::iterator oi;
			    oi = m_vCompleted.find(strEventName);
			    if(oi != m_vCompleted.end()) {

				    UINT uiSamples = static_cast<UINT>(oi->second.size());
				    if(uiSamples > 0) {
					    double dAccumulator = 0.0;
					    double dMinimum = DBL_MAX;
					    double dMaximum = -DBL_MAX;
					    std::vector<double>::iterator vi;
					    for(vi=oi->second.begin(); vi!=oi->second.end(); vi++) {
						    double dSample = *vi;
						    dAccumulator += dSample;
						    dMinimum = min(dMinimum, dSample);
						    dMaximum = max(dMaximum, dSample);
					    }
					    dResult = dAccumulator / uiSamples;
				    }
			    }
            }

            Unlock();
        }
		return dResult;
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reports all measured latencies and acknowledges any outstanding
    ///             (incomplete) measurments . </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="ss">   [in,out] The ss. </param>
    ///-------------------------------------------------------------------------------------------------

    void Instrumenter::__Report(
        __inout std::ostream& ss
        )
    {
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {

            Lock();
            __FinalizeSingletons();

            BOOL bAnyOutstanding = FALSE;
            BOOL bAnyCompleted = FALSE;
            std::map<std::string, std::stack<double>>::iterator mi;
            std::map<std::string, std::vector<double>>::iterator oi;
            for(mi=m_vOutstanding.begin(); mi!=m_vOutstanding.end() && !bAnyOutstanding; mi++) 
                bAnyOutstanding |= (mi->second.size() > 0);
            for(oi=m_vCompleted.begin(); oi!=m_vCompleted.end() && !bAnyCompleted; oi++)
                bAnyCompleted |= (oi->second.size() > 0);

            if(bAnyOutstanding) {
                ss << "Incomplete instrumentation metrics (ADHOC_STATS):" << std::endl;
                for(mi=m_vOutstanding.begin(); mi!=m_vOutstanding.end(); mi++) {
                    if(mi->second.size()) {
                        std::string strEventName = mi->first;
                        __ReportOutstanding(ss, strEventName);
                    }
                }
            }

            if(bAnyCompleted) {
                for(oi=m_vCompleted.begin(); oi!=m_vCompleted.end(); oi++) {
                    if(oi->second.size()) {
                        std::string strEventName = oi->first;
                        __ReportComplete(ss, strEventName);
                    }
                }
            }
            Unlock();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reports all measured latencies matching the given event name.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="ss">                   [in,out] The ss. </param>
    /// <param name="strEventName">         [in,out] Name of the event. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Instrumenter::__ReportComplete(
        __inout std::ostream& ss, 
        __in    std::string&  strEventName
        )
    {
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            assert(LockIsHeld());
            std::map<std::string, std::vector<double>>::iterator oi = m_vCompleted.find(strEventName);
            UINT uiSamples = static_cast<UINT>(oi->second.size());
            if(uiSamples > 0) {
                double dAccumulator = 0.0;
                double dMinimum = DBL_MAX;
                double dMaximum = -DBL_MAX;
                std::vector<double>::iterator vi;
                for(vi=oi->second.begin(); vi!=oi->second.end(); vi++) {
                    double dSample = *vi;
                    dAccumulator += dSample;
                    dMinimum = min(dMinimum, dSample);
                    dMaximum = max(dMaximum, dSample);
                }
                double dAverage = dAccumulator / uiSamples;
                ss << strEventName << "(mn,mx,avg)->["
                   << dMinimum << ", "
                   << dMaximum << ", "
                   << dAverage << "] ms over "
                   << uiSamples << " samples." 
                   << std::endl;
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reports all measured latencies and acknowledges any outstanding (incomplete)
    ///             measurments matching the given event name
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="ss">           [in,out] The ss. </param>
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    Instrumenter::__ReportOutstanding(
        __inout std::ostream& ss, 
        __in    std::string& strEventName
        )
    {
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            assert(LockIsHeld());
            std::map<std::string, std::stack<double>>::iterator mi = m_vOutstanding.find(strEventName);
            UINT uiDepth = static_cast<UINT>(mi->second.size());
            if(uiDepth> 0) {
                ss << strEventName << " INCOMPLETE MEASUREMENT->["
                   << uiDepth << " start sentinels, 0 end]"
                   << std::endl;
            }
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Enables the instrumentation framework. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="bEnable">  true to enable, false to disable. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Instrumenter::__Enable(
        __in BOOL bEnable
        )
    {
        BOOL bResult = FALSE;
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized) {
            Lock();
            m_bEnabled = bEnable;
            bResult = m_bEnabled;
            Unlock();
        }
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if the adhoc instrumentation framework is enabled. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <returns>   true if enabled, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Instrumenter::__IsEnabled(
        VOID
        )
    {
        BOOL bResult = FALSE;
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized) {
            assert(LockIsHeld());
            bResult = m_bEnabled;
        }
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if a measurement matching 'strEventName' is in flight. In flight
    ///             means that a start sentinal has been pushed onto the outstanding stack
    ///             that has not been matched yet by a corresponding completion. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   true if in flight, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Instrumenter::__IsInFlight(
        __in std::string& strEventName
        )
    {
        BOOL bResult = FALSE;
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            Lock();
            std::map<std::string, std::stack<double>>::iterator mi;
            mi = m_vOutstanding.find(strEventName);
            if(mi!=m_vOutstanding.end()) {
                bResult = mi->second.size() != 0;
            }
            Unlock();
        }
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if a measurement matching 'strEventName' is complete. Note that
    ///             because multiple measurements matching a given name can be tracked, it is 
    ///             possible for an event name to be both "in flight" and complete. 
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   true if complete, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    Instrumenter::__IsComplete(
        __in std::string& strEventName
        )
    {
        BOOL bResult = FALSE;
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            Lock();
            std::map<std::string, std::vector<double>>::iterator mi;
            mi = m_vCompleted.find(strEventName);
            if(mi!=m_vCompleted.end()) {
                bResult = mi->second.size() != 0;
            }
            Unlock();
        }
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resets this object. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    Instrumenter::__Reset(
        VOID
        )
    {
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            Lock();
            m_vCompleted.clear();
            m_vOutstanding.clear();
            Unlock();
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets nesting depth for the given event name. If the nest depth is 0 it means
    ///             there are no measurements with the given name in flight. A depth greater than 1
    ///             means there is a nested measurement with the same name. This idiom is likely best
    ///             avoided in potentially concurrent code, since the instrumenter handles nesting
    ///             with a stack, which makes it difficult to disambiguate end sentinels if they are
    ///             not ordered explicitly by the program.
    ///             </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   The nesting depth. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::__GetNestingDepth(
        __in std::string& strEventName
        )
    {
        UINT uiResult = 0;
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            Lock();
            std::map<std::string, std::vector<double>>::iterator mi;
            mi = m_vCompleted.find(strEventName);
            if(mi!=m_vCompleted.end()) {
                uiResult = static_cast<UINT>(mi->second.size());
            }
            Unlock();
        }
        return uiResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event start. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    /// <param name="dIncrement">   Amount to increment by. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::__AccumulateEventLatency(
        __in std::string& strEventName, 
        __in double dIncrement
        ) 
    {
        UINT uiResult = 0;
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            Lock();
            double dOldValue = 0.0;
            double dOldMin = DBL_MAX;
            double dOldMax = DBL_MIN;
            UINT uiOldSamples = 0;
            CumulativeEventMap::iterator mi;
            mi = m_vCumulativeEvents.find(strEventName);
            if(mi != m_vCumulativeEvents.end()) {
                uiOldSamples = std::get<0>(mi->second);
                dOldValue = std::get<1>(mi->second);
                dOldMin = std::get<2>(mi->second);
                dOldMax = std::get<3>(mi->second);
            }
            double dNewValue = dOldValue + dIncrement;
            double dNewMin = min(dOldMin, dIncrement);
            double dNewMax = max(dOldMax, dIncrement);
            UINT uiNewSamples = uiOldSamples + 1;
            m_vCumulativeEvents[strEventName] = std::make_tuple(uiNewSamples, dNewValue, dNewMin, dNewMax);
            uiResult = static_cast<UINT>(m_vOutstanding[strEventName].size());
            Unlock();
        }
        return uiResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event start. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::__RecordCumulativeEventStart(
        __in std::string& strEventName
        )
    {
        UINT uiResult = 0;
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            Lock();
            m_vOutstanding[strEventName].push(m_pRTTimer->elapsed());
            if(m_vCumulativeEvents.find(strEventName) == m_vCumulativeEvents.end()) {
                m_vCumulativeEvents[strEventName] = std::make_tuple(0, 0.0, DBL_MAX, DBL_MIN);
            }
            uiResult = static_cast<UINT>(m_vOutstanding[strEventName].size());
            Unlock();
        }
        return uiResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event complete. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::__RecordCumulativeEventComplete(
        __in std::string& strEventName
        )
    {
        UINT uiResult = 0;
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            Lock();
            std::map<std::string, std::stack<double>>::iterator mi;
            CumulativeEventMap::iterator dmi;
            mi = m_vOutstanding.find(strEventName);
            dmi = m_vCumulativeEvents.find(strEventName);
            if(dmi == m_vCumulativeEvents.end()) {
                PTask::Runtime::MandatoryInform("%s::%s(%s): ERROR: attempt to record completion "
                                                "of non-inflight cumulative event...recovering...\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                strEventName.c_str());
            } else {
                if(mi!=m_vOutstanding.end()) {
                    double dEventEnd = m_pRTTimer->elapsed();
                    double dEventStart = m_vOutstanding[strEventName].top();
                    double dLatency = dEventEnd - dEventStart;
                    m_vOutstanding[strEventName].pop();
                    if(m_vOutstanding[strEventName].size() == 0) {
                        // accumulate only once nesting depth is 0
                        m_vCumulativeEvents[strEventName] = std::make_tuple((std::get<0>(dmi->second)+1),
                                                                            (std::get<1>(dmi->second)+dLatency),
                                                                            min(std::get<2>(dmi->second),dLatency),
                                                                            max(std::get<3>(dmi->second),dLatency));
                    }
                    uiResult = 0;
                }
            }
            Unlock();
        }
        return uiResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event start. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::__RecordEventStart(
        __in std::string& strEventName
        )
    {
        UINT uiResult = 0;
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            Lock();
            m_vOutstanding[strEventName].push(m_pRTTimer->elapsed());
            uiResult = static_cast<UINT>(m_vOutstanding[strEventName].size());
            Unlock();
        }
        return uiResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event complete. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::__RecordEventComplete(
        __in std::string& strEventName
        )
    {
        UINT uiResult = 0;
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            Lock();
            std::map<std::string, std::stack<double>>::iterator mi;
            mi = m_vOutstanding.find(strEventName);
            if(mi==m_vOutstanding.end()) {
                assert(FALSE);
                PTask::Runtime::MandatoryInform("%s::%s(%s): ERROR: attempt to record completion "
                                                "of non-inflight event...recovering...\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                strEventName.c_str());
            } else {
                double dEventEnd = m_pRTTimer->elapsed();
                double dEventStart = m_vOutstanding[strEventName].top();
                double dLatency = dEventEnd - dEventStart;
                m_vOutstanding[strEventName].pop();
                m_vCompleted[strEventName].push_back(dLatency);
                uiResult = static_cast<UINT>(m_vCompleted[strEventName].size());
            }
            Unlock();
        }
        return uiResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event start. Ignore duplicate calls. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName"> [in,out] Name of the event. </param>
    ///
    /// <returns>   the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::__RecordSingletonEventStart(
        __in std::string& strEventName
        )
    {
        UINT uiResult = 0;
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            Lock();
            std::map<std::string, std::stack<double>>::iterator mi;
            mi = m_vOutstanding.find(strEventName);
            if(mi == m_vOutstanding.end() || mi->second.size() == 0) {
                m_vOutstanding[strEventName].push(m_pRTTimer->elapsed());
            }
            uiResult = static_cast<UINT>(m_vOutstanding[strEventName].size());
            Unlock();
        }
        assert(uiResult == 0 || uiResult == 1);
        return uiResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Record event complete an event that should have only one start sentinel, but for
    /// 			which concurrency implies non-determinism, so many threads may attempt to record
    /// 			the same event start. The primary example of this scenario is start of data
    /// 			processing in PTask, which occurs as soon as the first block is pushed by the
    /// 			user. It is simplest to record this by calling the instrumenter on every exposed
    /// 			call to Channel::Push, with all calls after the first ignored.
    /// 			</summary>
    ///
    /// <remarks>	Crossbac, 7/23/2013. </remarks>
    ///
    /// <param name="strEventName">		  	[in,out] Name of the event. </param>
    /// <param name="bRequireOutstanding">	(Optional) true to require an outstanding entry. Some
    /// 									stats (like first return-value materialization)
    /// 									are very difficult to capture unambiguously, because
    /// 									calls to record the event must be placed in common code
    /// 									paths. Calling with this parameter set to true allows the
    /// 									record call to fail without protest if the caller knows
    /// 									this to be such an event. </param>
    ///
    /// <returns>	the new nesting depth for events matching this name. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    Instrumenter::__RecordSingletonEventComplete(
        __in std::string& strEventName,
		__in BOOL bRequireOutstanding
        )
    {
        UINT uiResult = 0;
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            Lock();
            std::map<std::string, std::stack<double>>::iterator mi;
            mi = m_vOutstanding.find(strEventName);
            if(mi==m_vOutstanding.end()) {
				if(bRequireOutstanding) {
					assert(FALSE);
					PTask::Runtime::MandatoryInform("%s::%s(%s): ERROR: attempt to record completion "
													"of non-inflight event...recovering...\n",
													__FILE__,
													__FUNCTION__,
													strEventName.c_str());
				}
            } else if(mi->second.size() != 1) {
                assert(FALSE);
                PTask::Runtime::MandatoryInform("%s::%s(%s): ERROR: attempt to record singleton completion "
                                                "for non-singleton inflight event...\n",
                                                __FILE__,
                                                __FUNCTION__,
                                                strEventName.c_str());
            } else {
                double dEventEnd = m_pRTTimer->elapsed();
                double dEventStart = m_vOutstanding[strEventName].top();
                double dLatency = dEventEnd - dEventStart;
                std::map<std::string, double>::iterator si = m_vSingletonCompleted.find(strEventName);
				if(si != m_vSingletonCompleted.end()) {
					dLatency = max(dLatency, si->second);
				}
                m_vSingletonCompleted[strEventName] = dLatency;
                uiResult = 1;
            }
            Unlock();
        }
        return uiResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record event complete. </summary>
    ///
    /// <remarks>   Crossbac, 7/23/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void
    Instrumenter::__FinalizeSingletons(
        VOID
        )
    {
        assert(m_bInitialized);
        assert(PTask::Runtime::g_bAdhocInstrumentationSupported);
        if(PTask::Runtime::g_bAdhocInstrumentationSupported && m_bInitialized && m_bEnabled) {
            Lock();
            std::map<std::string, double>::iterator si;
            std::map<std::string, std::stack<double>>::iterator mi;
            for(si=m_vSingletonCompleted.begin(); si!=m_vSingletonCompleted.end(); si++) {
                std::string strEventName = si->first;
                double dLatency = si->second;
                mi = m_vOutstanding.find(strEventName);
                if(mi==m_vOutstanding.end()) {
                    assert(FALSE);
                    PTask::Runtime::MandatoryInform("%s::%s(%s): ERROR: attempt to record completion "
                                                    "of non-inflight event...recovering...\n",
                                                    __FILE__,
                                                    __FUNCTION__,
                                                    strEventName.c_str());
                } else if(mi->second.size() != 1) {
                    assert(FALSE);
                    PTask::Runtime::MandatoryInform("%s::%s(%s): ERROR: attempt to finalize singleton completion "
                                                    "for non-singleton inflight event...\n",
                                                    __FILE__,
                                                    __FUNCTION__,
                                                    strEventName.c_str());
                } else {
                    // assert(m_vCompleted.find(strEventName) == m_vCompleted.end());
                    m_vCompleted[strEventName].push_back(dLatency);
                    m_vOutstanding.erase(mi);
                }
            }
            m_vSingletonCompleted.clear();
            Unlock();
        }
    }


#else
#define _U(x) (x)
#define _Uxy(x,y) {(x); (y);}
#define _Uxyz(x,y,z) {(x); (y); (z);}
#define _Uxyzw(x,y,z,w) {(x); (y); (z); (w);}
#define _Uxyzwu(x,y,z,w,u) {(x); (y); (z); (w); (u); }
double Instrumenter::__CollectDataPoint(std::string& strEventName) { _U(strEventName); return 0.0; }
double 	Instrumenter::__CollectDataPoint(std::string& strEventName, UINT& nSamples, double& dMin, double& dMax) { _Uxyzw(strEventName, nSamples, dMin, dMax); return 0.0; }
void Instrumenter::__Report(std::ostream& ss) { _U(ss); assert(FALSE); }
void Instrumenter::__ReportComplete(std::ostream& ss, std::string& strEventName) { _Uxy(ss, strEventName); assert(FALSE); }
void Instrumenter::__ReportOutstanding(std::ostream& ss, std::string& strEventName) { _Uxy(ss, strEventName); assert(FALSE); }
BOOL Instrumenter::__Enable(BOOL bEnable) { _U(bEnable); assert(FALSE); return FALSE; } 
BOOL Instrumenter::__IsEnabled() { assert(FALSE); return FALSE; }
BOOL Instrumenter::__IsInFlight(std::string& strEventName) { _U(strEventName); assert(FALSE); return FALSE; }
BOOL Instrumenter::__IsComplete(std::string& strEventName) { _U(strEventName); assert(FALSE); return FALSE; }
UINT Instrumenter::__GetNestingDepth(std::string& strEventName) { _U(strEventName); assert(FALSE); return 0; }
UINT Instrumenter::__RecordEventStart(std::string& strEventName) { _U(strEventName); assert(FALSE); return 0; }
UINT Instrumenter::__RecordEventComplete(std::string& strEventName) { _U(strEventName); assert(FALSE); return 0; }
UINT Instrumenter::__RecordSingletonEventStart(std::string& strEventName) { _U(strEventName); assert(FALSE); return 0; }
UINT Instrumenter::__RecordSingletonEventComplete(std::string& strEventName, BOOL bRequireOutstanding) { _Uxy(bRequireOutstanding, strEventName);assert(FALSE); return 0; }
UINT Instrumenter::__AccumulateEventLatency(std::string& strEventName, double dIncrement) { _Uxy(dIncrement, strEventName); assert(FALSE); return 0; }
UINT Instrumenter::__RecordCumulativeEventStart(std::string& strEventName) { _U(strEventName); assert(FALSE); return 0; }
UINT Instrumenter::__RecordCumulativeEventComplete(std::string& strEventName) { _U(strEventName); assert(FALSE); return 0; }
void Instrumenter::__Reset() { assert(FALSE); }
void Instrumenter::__FinalizeSingletons() { assert(FALSE); } 
#endif

};