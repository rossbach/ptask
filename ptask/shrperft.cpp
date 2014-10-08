///-------------------------------------------------------------------------------------------------
// file:	shrperft.cpp
//
// summary:	Implements the shrperft class
///-------------------------------------------------------------------------------------------------

#include "shrperft.h"
#include <memory.h>

///-------------------------------------------------------------------------------------------------
/// <summary>   Constructor. </summary>
///
/// <remarks>   Crossbac, 1/24/2013. </remarks>
///
/// <param name="gran"> The granularity of the timer, seconds, milliseconds, microseconds. </param>
///-------------------------------------------------------------------------------------------------

CSharedPerformanceTimer::CSharedPerformanceTimer(
    hpf_granularity gran,
    bool bStart
    )
{
    InitializeCriticalSection(&m_cs);
	m_start = 0;
	m_gran = gran;
	init_query_system_time();
	m_freq = tickfreq();
    if(bStart)
        reset();
}

///-------------------------------------------------------------------------------------------------
/// <summary>    dtor
///             </summary>
///
/// <remarks>   Crossbac, 1/24/2013. </remarks>
///-------------------------------------------------------------------------------------------------

CSharedPerformanceTimer::~CSharedPerformanceTimer(
    VOID
    ) 
{
    EnterCriticalSection(&m_cs);
	free_query_system_time();
    LeaveCriticalSection(&m_cs);
    DeleteCriticalSection(&m_cs);
}

///-------------------------------------------------------------------------------------------------
/// <summary>   tickfreq Synopsis: return the tick frequency. </summary>
///
/// <remarks>   Crossbac, 1/24/2013. </remarks>
///
/// <returns>   . </returns>
///-------------------------------------------------------------------------------------------------

double 
CSharedPerformanceTimer::tickfreq(
    VOID
    ) 
{  
	LARGE_INTEGER tps;
	query_freq(&tps); 
	return (double)hpfresult(tps);
}

///-------------------------------------------------------------------------------------------------
/// <summary>   tickcnt Synopsis: return the current tick count. </summary>
///
/// <remarks>   Crossbac, 1/24/2013. </remarks>
///-------------------------------------------------------------------------------------------------

__int64 
CSharedPerformanceTimer::tickcnt(
    VOID
    ) 
{  
	LARGE_INTEGER t;
	query_hpc(&t); 
	return hpfresult(t);
}

///-------------------------------------------------------------------------------------------------
/// <summary>   reset Synopsis: reset the timer. </summary>
///
/// <remarks>   Crossbac, 1/24/2013. </remarks>
///-------------------------------------------------------------------------------------------------

void 
CSharedPerformanceTimer::reset(
    VOID
    ) 
{
    EnterCriticalSection(&m_cs);
	if(!m_freq) 
		m_freq = tickfreq();
	m_start = tickcnt();
    LeaveCriticalSection(&m_cs);
}

//=============================================================================
// elapsed
// Synopsis: return the elapsed time
//=============================================================================
ctrtype 
CSharedPerformanceTimer::elapsed(
    bool reset
    ) 
{
    EnterCriticalSection(&m_cs);
	__int64 end = tickcnt();
	if(!m_freq) return -1.0;
	double res = ((double)(end-m_start))/m_freq;
	if(reset)
		m_start = end;
	ctrtype result = 0.0;
    switch(m_gran) {
    case gran_sec: result = res; break;
    case gran_msec: result = res * 1000; break;
    case gran_usec: result = res * 1000000; break;
    case gran_nanosec: result = res * 1000000000; break;
    } 
    LeaveCriticalSection(&m_cs);
    return result;
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Initialises the query system time. </summary>
///
/// <remarks>   Crossbac, 1/24/2013. </remarks>
///
/// <returns>   . </returns>
///-------------------------------------------------------------------------------------------------

LPFNtQuerySystemTime 
CSharedPerformanceTimer::init_query_system_time(
	VOID
	)
{
	m_hModule = LoadLibraryW(L"NTDLL.DLL");
	FARPROC x = GetProcAddress(m_hModule, "NtQuerySystemTime");
	m_lpfnQuerySystemTime = (LPFNtQuerySystemTime) x;
	return m_lpfnQuerySystemTime;
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Queries system time. </summary>
///
/// <remarks>   Crossbac, 1/24/2013. </remarks>
///
/// <param name="li">   The li. </param>
///
/// <returns>   true if it succeeds, false if it fails. </returns>
///-------------------------------------------------------------------------------------------------

BOOL
CSharedPerformanceTimer::query_system_time(
	PLARGE_INTEGER li
	)
{
	if(!m_lpfnQuerySystemTime) 
		return FALSE;
	(*m_lpfnQuerySystemTime)(li);
	return TRUE;
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Delta milliseconds. </summary>
///
/// <remarks>   Crossbac, 1/24/2013. </remarks>
///
/// <param name="lEarly">   The early. </param>
/// <param name="lLate">    The late. </param>
///
/// <returns>   . </returns>
///-------------------------------------------------------------------------------------------------

DWORD 
CSharedPerformanceTimer::delta_milliseconds(
	LARGE_INTEGER lEarly,
	LARGE_INTEGER lLate
	)
{
	LONGLONG ll1 = lEarly.QuadPart;
	LONGLONG ll2 = lLate.QuadPart;
	LONGLONG ll = ll2 - ll1;
	ll = (ll * 100000) / 1000000000;
	return (DWORD) ll;
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Free query system time. </summary>
///
/// <remarks>   Crossbac, 1/24/2013. </remarks>
///
/// <returns>   . </returns>
///-------------------------------------------------------------------------------------------------

VOID
CSharedPerformanceTimer::free_query_system_time(
	VOID
	)
{
	if(m_hModule) {
		FreeLibrary(m_hModule);
	}
}
