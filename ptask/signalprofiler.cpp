///-------------------------------------------------------------------------------------------------
// file:	signalprofiler.cpp
//
// summary:	Implements the signalprofiler class
///-------------------------------------------------------------------------------------------------

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "SignalProfiler.h"
#include "task.h"
#include "port.h"
#include "channel.h"
#include "datablock.h"
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

    /// <summary>   true to if a signal profiler exists and is initialised. </summary>
    BOOL                       SignalProfiler::s_bSignalProfilerInit = FALSE;

#ifndef PROFILE_CONTROLSIGNALS
#define f() { assert(!SignalProfiler::s_bSignalProfilerInit); assert(FALSE); } 
#define up(x) UNREFERENCED_PARAMETER(x)

    void SignalProfiler::Initialize() {  }
    void SignalProfiler::Deinitialize() {  }
    void SignalProfiler::Report(std::ostream& ss) { up(ss); } 
    std::stringstream* SignalProfiler::GetHistory() { return NULL; }
    BOOL SignalProfiler::IsUnderProfile(CONTROLSIGNAL lui) { up(lui); f(); return FALSE; }
    BOOL SignalProfiler::IsUnderProfile(Datablock * pB) { up(pB); f(); return FALSE; }
    void SignalProfiler::RegisterSignal(CONTROLSIGNAL l, BOOL b) { up(l); up(b); f(); }
    void SignalProfiler::RecordSignalTransit(Lockable * p, Datablock * b, SIGEVTTYPE t) { up(p); up(b); up(t); f(); }        
    BOOL SignalProfiler::SignalTrafficOccurred(Lockable * l, CONTROLSIGNAL c, SIGEVTTYPE t) { up(l); up(c); up(t); f(); return FALSE; }
    BOOL SignalProfiler::BalancedSignalTrafficOccurred(Lockable * l, CONTROLSIGNAL c) {up(l); up(c); f(); return FALSE; }
    BOOL SignalProfiler::SuppressedSignalTrafficOccurred(Lockable * l, CONTROLSIGNAL c) {up(l); up(c); f(); return FALSE; }
    BOOL SignalProfiler::ProfiledSignalTrafficOccurred(Lockable * l, SIGEVTTYPE t) { up(l); up(t); f(); return FALSE;}
    BOOL SignalProfiler::AnyProfiledSignalTrafficOccurred(Lockable * l) { up(l); f(); return FALSE; }
    BOOL SignalProfiler::SignalIngressOccurred(Lockable * l, CONTROLSIGNAL c) { up(l); up(c); f(); return FALSE; }
    BOOL SignalProfiler::SignalEgressOccurred(Lockable * l, CONTROLSIGNAL c) {up(l); up(c); f(); return FALSE; }
    BOOL SignalProfiler::ProfiledSignalIngressOccurred(Lockable * l) { up(l); f(); return FALSE;}
    BOOL SignalProfiler::ProfiledSignalEgressOccurred(Lockable * l) { up(l); f(); return FALSE;}
    BOOL SignalProfiler::BalancedProfiledSignalTrafficOccurred(Lockable * l) { up(l); f(); return FALSE;}
    BOOL SignalProfiler::SuppressedProfiledSignalTrafficOccurred(Lockable * l) { up(l); f(); return FALSE;}
    BOOL SignalProfiler::HasRelevantPredicate(Lockable * l) { up(l); f(); return FALSE;}
    BOOL SignalProfiler::IsRelevantPredicate(CHANNELPREDICATE ePredicate) { up(ePredicate); f(); return FALSE; } 
    CHANNELACTIVITYSTATE SignalProfiler::GetSignalActivityState(Lockable * p) { up(p); f(); return cas_none; } 
    CHANNELPREDICATIONSTATE SignalProfiler::GetChannelSignalPredicationState(Lockable * p) { up(p); f(); return cps_na; }
    char * SignalProfiler::GetChannelCodedColor(CHANNELACTIVITYSTATE a, CHANNELPREDICATIONSTATE p) { up(a); up(p); f(); return NULL; }
    std::string SignalProfiler::GetChannelCodedName(Lockable * p, BOOL b) { up(p); up(b); f(); return std::string(""); }

#else 

    CRITICAL_SECTION                                                     SignalProfiler::s_csSignalProfiler;
    BOOL                                                                 SignalProfiler::s_bFilterProfiledSignals = FALSE;
    CONTROLSIGNAL                                                        SignalProfiler::s_luiSignalsOfInterest = DBCTLC_NONE;
    std::map<Lockable*, std::set<SIGOBSERVATION*>>                       SignalProfiler::s_vWitnessToSignalMap;
    std::map<CONTROLSIGNAL, std::set<SIGOBSERVATION*>>                   SignalProfiler::s_vSignalToWitnessMap;
    std::map<double, std::set<SIGOBSERVATION*>>                          SignalProfiler::s_vSignalHistory;
    CSharedPerformanceTimer *                                            SignalProfiler::s_pGlobalProfileTimer;
    char * SignalProfiler::s_lpszChannelColors[3][3] = {
        { "gray60", "gray60", "gray60" },       // cas_none        { cps_na, cps_open, cps_closed }
        { "gray60", "darkgreen", "firebrick" }, // cas_unexercised { cps_na, cps_open, cps_closed }
        { "violet", "green", "red" }            // cas_exercised   { cps_na, cps_open, cps_closed }
    };


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initializes control signal profiling. </summary>
    ///
    /// <remarks>   crossbac, 6/27/2014. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void SignalProfiler::Initialize(
        VOID
        )
    {
        if(!PTask::Runtime::GetSignalProfilingEnabled()) return;
        assert(!s_bSignalProfilerInit || s_pGlobalProfileTimer == NULL);
        if(s_bSignalProfilerInit)  return;
        InitializeCriticalSection(&s_csSignalProfiler);
        s_pGlobalProfileTimer = new CSharedPerformanceTimer(gran_msec, true);
        s_bSignalProfilerInit = TRUE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Deinitialize control signal profiling. </summary>
    ///
    /// <remarks>   crossbac, 6/27/2014. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void 
    SignalProfiler::Deinitialize(
        VOID
        )
    {
        if(!PTask::Runtime::GetSignalProfilingEnabled() || !s_bSignalProfilerInit) return;
        EnterCriticalSection(&s_csSignalProfiler);
        assert(s_pGlobalProfileTimer != NULL);
        delete s_pGlobalProfileTimer;
        s_pGlobalProfileTimer = NULL;
        std::map<double, std::set<SIGOBSERVATION*>>::iterator mi;
        for(mi=s_vSignalHistory.begin(); mi!=s_vSignalHistory.end(); mi++) {
            std::set<SIGOBSERVATION*>::iterator si;
            for(si=mi->second.begin(); si!=mi->second.end(); si++)
                delete *si;
            mi->second.clear();
        }
        s_vWitnessToSignalMap.clear();
        s_vSignalToWitnessMap.clear();
        s_vSignalHistory.clear();
        LeaveCriticalSection(&s_csSignalProfiler);
        DeleteCriticalSection(&s_csSignalProfiler);
        s_bSignalProfilerInit = FALSE;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'luiControlSignal' is under profile. </summary>
    ///
    /// <remarks>   crossbac, 6/27/2014. </remarks>
    ///
    /// <param name="luiControlSignal"> The lui control signal. </param>
    ///
    /// <returns>   true if under profile, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    SignalProfiler::IsUnderProfile(
        __in CONTROLSIGNAL luiControlSignal
        )
    {
        if(!PTask::Runtime::GetSignalProfilingEnabled() || !s_bSignalProfilerInit) 
            return FALSE;
        // no lock required, as we do not allow modification after runtime init
        return !s_bFilterProfiledSignals || 
                ((s_luiSignalsOfInterest & luiControlSignal) != 0);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if control signals on this block are under profile. </summary>
    ///
    /// <remarks>   crossbac, 6/27/2014. </remarks>
    ///
    /// <param name="luiControlSignal"> The lui control signal. </param>
    ///
    /// <returns>   true if under profile, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    SignalProfiler::IsUnderProfile(
        __in Datablock * pBlock
        )
    {
        if(!pBlock || !PTask::Runtime::GetSignalProfilingEnabled() || !s_bSignalProfilerInit) 
            return FALSE;
        return IsUnderProfile(pBlock->__getControlSignals());
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Registers the signal as being one "of interest" to the profiler. </summary>
    ///
    /// <remarks>   crossbac, 6/27/2014. </remarks>
    ///
    /// <param name="luiControlSignal"> The lui control signal. </param>
    /// <param name="bEnable">          (Optional) the enable. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    SignalProfiler::RegisterSignal(
        __in CONTROLSIGNAL luiControlSignal,
        __in BOOL bEnable
        )
    {
        assert(!PTask::Runtime::IsInitialized()); // PTaskRuntime.cpp should disallow this call!
        // no lock required, since signals of interest is immutable after runtime init.
        if(!PTask::Runtime::GetSignalProfilingEnabled()) return;
        if(PTask::Runtime::IsInitialized()) return;
        if(bEnable) {
            s_luiSignalsOfInterest |= luiControlSignal;
            s_bFilterProfiledSignals = s_luiSignalsOfInterest != DBCTLC_NONE;
        } else {
            s_luiSignalsOfInterest &= ~luiControlSignal;
            s_bFilterProfiledSignals = s_luiSignalsOfInterest != DBCTLC_NONE;
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Dumps profile statistics. </summary>
    ///
    /// <remarks>   Crossbac, 7/17/2013. </remarks>
    ///
    /// <param name="ss">   [in,out] The ss. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    SignalProfiler::Report(
        __in std::ostream& ss
        )
    {
        ss << __FUNCTION__ << " is unimplemented" << std::endl;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets signal history for a particular graph object. </summary>
    ///
    /// <remarks>   Crossbac, 7/17/2013. </remarks>
    ///
    /// <returns>   null if it fails, else the task dispatch history. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::stringstream* 
    SignalProfiler::GetHistory(
        VOID
        )
    {
        BOOL bOldApproach = FALSE;
        std::stringstream* pss = new std::stringstream();
        std::stringstream& ss = *pss;
        std::set<SIGOBSERVATION*>::iterator si;
        std::map<double, std::set<SIGOBSERVATION*>>::iterator hi;        
        for(hi=s_vSignalHistory.begin(); hi!=s_vSignalHistory.end(); hi++) {
            for(si=hi->second.begin(); si!=hi->second.end(); si++) {
                SIGOBSERVATION * pObservation = *si;
                if(bOldApproach) {
                    Lockable * pWitness = pObservation->pWitness;
                    std::string strWitnessName("UNKNOWN");
                    Channel * pCWitness = dynamic_cast<Channel*>(pWitness);
                    Task * pTWitness = dynamic_cast<Task*>(pWitness);
                    Port * pPWitness = dynamic_cast<Port*>(pWitness);
                    int nVPtrs = 0;
                    nVPtrs += pCWitness ? 1 : 0;
                    nVPtrs += pPWitness ? 1 : 0;
                    nVPtrs += pTWitness ? 1 : 0;
                    assert(nVPtrs == 1); 
                    if(pCWitness != NULL) {
                        strWitnessName += "CH: ";
                        strWitnessName += pCWitness->GetName();
                    } else if(pPWitness != NULL) {
                        strWitnessName += "PT: ";
                        strWitnessName += pPWitness->GetVariableBinding();
                    } else if(pTWitness != NULL) {
                        strWitnessName += "T: ";
                        strWitnessName += pTWitness->GetTaskName();
                    } 
                    ss << strWitnessName << std::endl;
                } else {
                    ss << pObservation << std::endl;
                }
            }
        }
        return pss;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Record signal transit. </summary>
    ///
    /// <remarks>   crossbac, 6/27/2014. </remarks>
    ///
    /// <param name="pWitness">         [in,out] If non-null, the witness. </param>
    /// <param name="pBlock">           [in,out] The lui control signal. </param>
    /// <param name="eSigEventType">    Type of the signal event. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    SignalProfiler::RecordSignalTransit(
        __in Lockable * pWitness, 
        __in Datablock * pBlock,
        __in SIGEVTTYPE eSigEventType
        )
    {
        if(!PTask::Runtime::GetSignalProfilingEnabled() || !s_bSignalProfilerInit) return;
        if(!PTask::Runtime::IsInitialized()) return;
        if(!IsUnderProfile(pBlock)) return;
        Lock();

        // record the raw signal at the current time
        double dTimestamp = s_pGlobalProfileTimer->elapsed(false);
        SIGOBSERVATION * pObservation = new SIGOBSERVATION(eSigEventType, dTimestamp, pWitness, pBlock);
        s_vSignalHistory[dTimestamp].insert(pObservation); 
        CONTROLSIGNAL luiControlSignal = pObservation->luiRawSignal;
        assert(IsUnderProfile(luiControlSignal)); 

        // decompose the signal into components and record
        for(CONTROLSIGNAL ctlui = 1; ctlui > 0; ctlui <<= 1) {
            if(luiControlSignal & ctlui && IsUnderProfile(ctlui)) {
                s_vWitnessToSignalMap[pWitness].insert(pObservation);
                s_vSignalToWitnessMap[ctlui].insert(pObservation); 
            }
        }
        Unlock(); 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return true if the given graph object ever bore witness to the given control
    ///             signal.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/30/2014. </remarks>
    ///
    /// <param name="pWitness">         [in,out] If non-null, the witness. </param>
    /// <param name="luiControlSignal"> The lui control signal. </param>
    /// <param name="eType">            The type. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    SignalProfiler::SignalTrafficOccurred(
        __in Lockable * pWitness, 
        __in CONTROLSIGNAL luiControlSignal,
        __in SIGEVTTYPE eEventType
        )
    {
        if(!PTask::Runtime::GetSignalProfilingEnabled() || !s_bSignalProfilerInit) return FALSE;
        if(!PTask::Runtime::IsInitialized()) return FALSE;
        Lock(); 
        BOOL bResult = FALSE; 
        std::map<Lockable*, std::set<SIGOBSERVATION*>>::iterator mi;
        mi=s_vWitnessToSignalMap.find(pWitness); 
        if(mi!=s_vWitnessToSignalMap.end()) {
            std::set<SIGOBSERVATION*>::iterator si;
            for(si=mi->second.begin(); si!=mi->second.end() && !bResult; si++) {
                SIGOBSERVATION * pObservation = *si;
                if((luiControlSignal & pObservation->luiRawSignal) &&
                    (pObservation->eType == eEventType)) {
                    bResult = TRUE;
                }
            }
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return true if the given graph object ever bore witness to the given control
    ///             signal, and the ingress/egress was balanced
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/30/2014. </remarks>
    ///
    /// <param name="pWitness">         [in,out] If non-null, the witness. </param>
    /// <param name="luiControlSignal"> The lui control signal. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    SignalProfiler::BalancedSignalTrafficOccurred(
        __in Lockable * pWitness, 
        __in CONTROLSIGNAL luiControlSignal
        )
    {
        if(!PTask::Runtime::GetSignalProfilingEnabled() || !s_bSignalProfilerInit) return FALSE;
        if(!PTask::Runtime::IsInitialized()) return FALSE;
        Lock(); 
        BOOL bResult = FALSE; 
        std::map<Lockable*, std::set<SIGOBSERVATION*>>::iterator mi;
        mi=s_vWitnessToSignalMap.find(pWitness); 
        if(mi!=s_vWitnessToSignalMap.end()) {
            int nIngressEvents = 0;
            int nEgressEvents = 0;
            std::set<SIGOBSERVATION*>::iterator si;
            for(si=mi->second.begin(); si!=mi->second.end() && !bResult; si++) {
                SIGOBSERVATION * pObservation = *si;
                if((luiControlSignal & pObservation->luiRawSignal)) {
                    nIngressEvents += (pObservation->eType == SIGEVTTYPE::SIGEVT_INGRESS) ? 1 : 0;
                    nEgressEvents += (pObservation->eType == SIGEVTTYPE::SIGEVT_EGRESS) ? 1 : 0;
                }
            }
            bResult = nIngressEvents > 0 && nIngressEvents == nEgressEvents;
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return true if the given graph object ever bore witness to the given control
    ///             signal, and the ingress count exceeds the egress count
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/30/2014. </remarks>
    ///
    /// <param name="pWitness">         [in,out] If non-null, the witness. </param>
    /// <param name="luiControlSignal"> The lui control signal. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    SignalProfiler::SuppressedSignalTrafficOccurred(
        __in Lockable * pWitness, 
        __in CONTROLSIGNAL luiControlSignal
        )
    {
        if(!PTask::Runtime::GetSignalProfilingEnabled() || !s_bSignalProfilerInit) return FALSE;
        if(!PTask::Runtime::IsInitialized()) return FALSE;
        Lock(); 
        BOOL bResult = FALSE; 
        std::map<Lockable*, std::set<SIGOBSERVATION*>>::iterator mi;
        mi=s_vWitnessToSignalMap.find(pWitness); 
        if(mi!=s_vWitnessToSignalMap.end()) {
            int nIngressEvents = 0;
            int nEgressEvents = 0;
            std::set<SIGOBSERVATION*>::iterator si;
            for(si=mi->second.begin(); si!=mi->second.end() && !bResult; si++) {
                SIGOBSERVATION * pObservation = *si;
                if((luiControlSignal & pObservation->luiRawSignal)) {
                    nIngressEvents += (pObservation->eType == SIGEVTTYPE::SIGEVT_INGRESS) ? 1 : 0;
                    nEgressEvents += (pObservation->eType == SIGEVTTYPE::SIGEVT_EGRESS) ? 1 : 0;
                }
            }
            assert(nEgressEvents <= nIngressEvents);
            bResult = nIngressEvents > 0 && nIngressEvents > nEgressEvents;
        }
        Unlock();
        return bResult;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return true if the given graph object ever bore witness to the given control
    ///             signal.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 6/30/2014. </remarks>
    ///
    /// <param name="pWitness">         [in,out] If non-null, the witness. </param>
    /// <param name="luiControlSignal"> The lui control signal. </param>
    /// <param name="eType">            The type. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    SignalProfiler::ProfiledSignalTrafficOccurred(
        __in Lockable * pWitness, 
        __in SIGEVTTYPE eEventType
        )
    {
        if(!PTask::Runtime::GetSignalProfilingEnabled() || !s_bSignalProfilerInit) return FALSE;
        if(!PTask::Runtime::IsInitialized()) return FALSE;
        return SignalTrafficOccurred(pWitness, s_luiSignalsOfInterest, eEventType);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return true if the given graph object ever bore witness to
    ///             the given control signal. </summary>
    ///-------------------------------------------------------------------------------------------------

    BOOL SignalProfiler::SignalIngressOccurred(Lockable * pW, CONTROLSIGNAL lui) {  return SignalTrafficOccurred(pW, lui, SIGEVTTYPE::SIGEVT_INGRESS); }
    BOOL SignalProfiler::SignalEgressOccurred(Lockable * pW, CONTROLSIGNAL lui) {  return SignalTrafficOccurred(pW, lui, SIGEVTTYPE::SIGEVT_EGRESS); }
    BOOL SignalProfiler::ProfiledSignalIngressOccurred(Lockable * pW) {  return ProfiledSignalTrafficOccurred(pW, SIGEVTTYPE::SIGEVT_INGRESS); }
    BOOL SignalProfiler::ProfiledSignalEgressOccurred(Lockable * pW) {  return ProfiledSignalTrafficOccurred(pW, SIGEVTTYPE::SIGEVT_EGRESS); }
    BOOL SignalProfiler::AnyProfiledSignalTrafficOccurred(Lockable * pW) { return ProfiledSignalIngressOccurred(pW) || ProfiledSignalEgressOccurred(pW); }
    BOOL SignalProfiler::BalancedProfiledSignalTrafficOccurred(Lockable * pW) { return BalancedSignalTrafficOccurred(pW, s_luiSignalsOfInterest); }
    BOOL SignalProfiler::SuppressedProfiledSignalTrafficOccurred(Lockable * pW) { return SuppressedSignalTrafficOccurred(pW, s_luiSignalsOfInterest); }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   return true if the given graph object ever bore witness to
    ///             the given control signal. </summary>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    SignalProfiler::HasRelevantPredicate(
        __in Lockable * pWitness
        ) 
    {
        // no lock required since signals of interest
        // are immutable after runtime initialization
        if(!PTask::Runtime::GetSignalProfilingEnabled() || !s_bSignalProfilerInit) return FALSE;
        if(!PTask::Runtime::IsInitialized()) return FALSE;
        Channel * pChannel = dynamic_cast<Channel*>(pWitness); 
        if(pChannel == NULL) return FALSE;
        CHANNELPREDICATE eSrcPredicate = pChannel->GetPredicationType(CE_SRC);
        CHANNELPREDICATE eDstPredicate = pChannel->GetPredicationType(CE_DST);
        return IsRelevantPredicate(eSrcPredicate) || IsRelevantPredicate(eDstPredicate); 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if 'ePredicate' is relevant predicate. </summary>
    ///
    /// <remarks>   crossbac, 6/27/2014. </remarks>
    ///
    /// <param name="ePredicate">   The predicate. </param>
    ///
    /// <returns>   true if relevant predicate, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL
    SignalProfiler::IsRelevantPredicate(
        __in CHANNELPREDICATE ePredicate
        )
    {
        if(!s_bFilterProfiledSignals)
            return ePredicate != CGATEFN_NONE && 
                   ePredicate != CGATEFN_DEVNULL;

        switch(ePredicate) {
        case CGATEFN_NONE:  return FALSE;
        case CGATEFN_CLOSE_ON_EOF:  return (s_luiSignalsOfInterest & DBCTLC_EOF) != 0; 
        case CGATEFN_OPEN_ON_EOF:   return (s_luiSignalsOfInterest & DBCTLC_EOF) != 0; 
        case CGATEFN_OPEN_ON_BEGINITERATION:  return (s_luiSignalsOfInterest & DBCTLC_BEGINITERATION) != 0;
        case CGATEFN_CLOSE_ON_BEGINITERATION: return (s_luiSignalsOfInterest & DBCTLC_BEGINITERATION) != 0;
        case CGATEFN_OPEN_ON_ENDITERATION:    return (s_luiSignalsOfInterest & DBCTLC_ENDITERATION) != 0;
        case CGATEFN_CLOSE_ON_ENDITERATION:   return (s_luiSignalsOfInterest & DBCTLC_ENDITERATION) != 0;
        case CGATEFN_DEVNULL: return FALSE;
        case CGATEFN_CLOSE_ON_BOF: return (s_luiSignalsOfInterest & DBCTLC_BOF) != 0; 
        case CGATEFN_OPEN_ON_BOF:  return (s_luiSignalsOfInterest & DBCTLC_BOF) != 0; 
        case CGATEFN_USER_DEFINED: return TRUE; // conservative!
        default:
            PTask::Runtime::MandatoryInform("%s::%s(%d) unknown predicate type %d!\n", 
                                            __FILE__,
                                            __FUNCTION__,
                                            __LINE__,
                                            ePredicate);
            return FALSE;
        }
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets signal activity state. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2014. </remarks>
    ///
    /// <param name="pLockable">    [in,out] If non-null, the lockable. </param>
    ///
    /// <returns>   The signal activity state. </returns>
    ///-------------------------------------------------------------------------------------------------

    CHANNELACTIVITYSTATE 
    SignalProfiler::GetSignalActivityState(
        __in Lockable * pLockable
        )
    {
        Channel * pChannel = dynamic_cast<Channel*>(pLockable); 
        assert(pChannel != NULL); 
        if(pChannel != NULL) {
            BOOL bSigTrafficOccurred = AnyProfiledSignalTrafficOccurred(pChannel); 
            if(bSigTrafficOccurred) 
                return cas_exercised;
            return HasRelevantPredicate(pChannel) ? cas_unexercised : cas_none; 
        }
        return cas_none;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets channel signal predication state. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2014. </remarks>
    ///
    /// <param name="pLockable">    [in,out] If non-null, the lockable. </param>
    ///
    /// <returns>   The channel signal predication state. </returns>
    ///-------------------------------------------------------------------------------------------------

    CHANNELPREDICATIONSTATE
    SignalProfiler::GetChannelSignalPredicationState(
        __in Lockable * pLockable
        )
    {
        Channel * pChannel = dynamic_cast<Channel*>(pLockable); 
        assert(pChannel != NULL); 
        if(pChannel != NULL) {
            if(!HasRelevantPredicate(pChannel))
                return cps_na;
            CHANNELPREDICATE eSrcPred = pChannel->GetPredicationType(CE_SRC); 
            CHANNELPREDICATE eDstPred = pChannel->GetPredicationType(CE_DST); 
            assert(eSrcPred != CGATEFN_NONE || eDstPred != CGATEFN_NONE);
            if(eSrcPred == CGATEFN_DEVNULL || eDstPred == CGATEFN_DEVNULL) 
                return cps_closed;
            CONTROLSIGNAL luiCtlCode = pChannel->GetPropagatedControlSignals();
            BOOL bPasses = FALSE; 
            switch(eSrcPred) {
            case CGATEFN_NONE:                      bPasses = TRUE;
            case CGATEFN_DEVNULL:                   return cps_closed;
            case CGATEFN_CLOSE_ON_EOF:              bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_EOF); break; 
            case CGATEFN_OPEN_ON_EOF:               bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_EOF); break;
            case CGATEFN_CLOSE_ON_BOF:              bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_BOF); break;
            case CGATEFN_OPEN_ON_BOF:               bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_BOF); break;
            case CGATEFN_CLOSE_ON_BEGINITERATION:   bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_BEGINITERATION); break;
            case CGATEFN_OPEN_ON_BEGINITERATION:    bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_BEGINITERATION); break;
            case CGATEFN_CLOSE_ON_ENDITERATION:     bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_ENDITERATION); break;
            case CGATEFN_OPEN_ON_ENDITERATION:      bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_ENDITERATION); break;
            case CGATEFN_USER_DEFINED:              assert(FALSE); return cps_open; 
            }
            if(!bPasses) 
                return cps_closed;
            switch(eDstPred) {
            case CGATEFN_NONE:                      bPasses = TRUE;
            case CGATEFN_DEVNULL:                   return cps_closed;
            case CGATEFN_CLOSE_ON_EOF:              bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_EOF); break; 
            case CGATEFN_OPEN_ON_EOF:               bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_EOF); break;
            case CGATEFN_CLOSE_ON_BOF:              bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_BOF); break;
            case CGATEFN_OPEN_ON_BOF:               bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_BOF); break;
            case CGATEFN_CLOSE_ON_BEGINITERATION:   bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_BEGINITERATION); break;
            case CGATEFN_OPEN_ON_BEGINITERATION:    bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_BEGINITERATION); break;
            case CGATEFN_CLOSE_ON_ENDITERATION:     bPasses = !TESTSIGNAL(luiCtlCode, DBCTLC_ENDITERATION); break;
            case CGATEFN_OPEN_ON_ENDITERATION:      bPasses =  TESTSIGNAL(luiCtlCode, DBCTLC_ENDITERATION); break;
            case CGATEFN_USER_DEFINED:              assert(FALSE); return cps_open; 
            }
            return bPasses ? cps_open : cps_closed; 
        }
        return cps_na;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets channel coded color. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2014. </remarks>
    ///
    /// <param name="eActivityState">       State of the activity. </param>
    /// <param name="ePredicationState">    State of the predication. </param>
    ///
    /// <returns>   null if it fails, else the channel coded color. </returns>
    ///-------------------------------------------------------------------------------------------------

    char * 
    SignalProfiler::GetChannelCodedColor(
        __in CHANNELACTIVITYSTATE eActivityState, 
        __in CHANNELPREDICATIONSTATE ePredicationState
        ) 
    {
        int nActivityIdx = static_cast<int>(eActivityState);
        int nPredicationIdx = static_cast<int>(ePredicationState); 
        assert(nActivityIdx >= 0 && nActivityIdx < 3);
        assert(nPredicationIdx >= 0 && nPredicationIdx < 3); 
        return s_lpszChannelColors[nActivityIdx][nPredicationIdx];
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Null to empty. </summary>
    ///
    /// <remarks>   Crossbac, 1/11/2012. </remarks>
    ///
    /// <param name="a">    a. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    inline std::string SAFENAME(const char *a) {
        return (a) ? std::string(a) : std::string("");
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets channel coded name. </summary>
    ///
    /// <remarks>   crossbac, 7/1/2014. </remarks>
    ///
    /// <param name="pLockable">    [in,out] If non-null, the lockable. </param>
    /// <param name="bBlocked">     The blocked. </param>
    ///
    /// <returns>   The channel coded name. </returns>
    ///-------------------------------------------------------------------------------------------------

    std::string
    SignalProfiler::GetChannelCodedName(
        __in Lockable * pLockable,
        __in BOOL bBlocked
        )
    {
        Channel * pChannel = dynamic_cast<Channel*>(pLockable); 
        assert(pChannel != NULL); 
        if(pChannel != NULL) {
            BOOL bUserSpecifiedName = pChannel->HasUserSpecifiedName();
            char * lpszChannelName = pChannel->GetName();
            pChannel->Lock();
            UINT uiQueueDepth = static_cast<UINT>(pChannel->GetQueueDepth());
            UINT uiCapacity = pChannel->GetCapacity();
            pChannel->Unlock();
            std::ostringstream capacitysuffixss;
            if(bBlocked) {
                capacitysuffixss << "[" << uiQueueDepth << "/" << uiCapacity << "]";
            }
            std::string strTypePrefix;
            Port * pSrc = pChannel->GetBoundPort(CE_SRC);
            Port * pDst = pChannel->GetBoundPort(CE_DST);
            CHANNELTYPE eType = pChannel->GetType(); 
            BOOL bMultiChannel = eType == CT_MULTI;
            switch(eType) {
            case CT_GRAPH_INPUT:  strTypePrefix = "in-"; break; 
            case CT_GRAPH_OUTPUT: strTypePrefix = "out-"; break; 
            case CT_INTERNAL:     strTypePrefix = "int-"; break; 
            case CT_MULTI:        strTypePrefix = "mlt-"; break; 
            case CT_INITIALIZER:  strTypePrefix = "init-"; break; 
            }   
            std::string channelName;

            if(bUserSpecifiedName) {

                // use the name the programmer gave us.
                // in the name of readability, try to 
                // look for "->" and replace it with "->\n"                
                std::string strUserName = lpszChannelName;
                size_t szPos = strUserName.find("->");
                if(szPos != std::string::npos) 
                    strUserName.replace(szPos, 2, "->\\n", 0, 4);
                channelName = strTypePrefix + strUserName + capacitysuffixss.str();

            } else {

                // the channel name was made up by ptask. 
                // which means it will be a (semantically) meaningless GUID.
                // try to generate a name based on src/dst bindings.

                const char * pDstBinding = pDst == NULL ? NULL : pDst->GetVariableBinding();
                const char * pSrcBinding = pSrc == NULL ? NULL : pSrc->GetVariableBinding(); 
                std::string strBindingBasedName;
                strBindingBasedName += pSrcBinding ? pSrcBinding : "";
                strBindingBasedName += "-";
                strBindingBasedName += pDstBinding ? pDstBinding : "";
                channelName = strTypePrefix + strBindingBasedName + capacitysuffixss.str();
            }
            return channelName;
        }
        return std::string("");
    }

    void SignalProfiler::Lock() { EnterCriticalSection(&s_csSignalProfiler); }
    void SignalProfiler::Unlock() { LeaveCriticalSection(&s_csSignalProfiler); }



#endif
};
