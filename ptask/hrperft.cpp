/********************************************************
* hrperft.cpp
**********************************************************/
#include "hrperft.h"
#include <memory.h>
#include <assert.h>

//=============================================================================
// CHighResolutionTimer
// Synopsis: ctor
//=============================================================================
CHighResolutionTimer::CHighResolutionTimer(hpf_granularity gran) {
    assert(m_gran != gran_usec && "CHighResolutionTimer doesn't support micro-second granularity");
    assert(m_gran != gran_nanosec && "CHighResolutionTimer doesn't support nanosecond granularity");
	m_freq = 0;
	m_start = 0;
	m_gran = gran;
	init_query_system_time();
}

//=============================================================================
// ~CHighResolutionTimer
// Synopsis: dtor
//=============================================================================
CHighResolutionTimer::~CHighResolutionTimer(void) {
	free_query_system_time();
}


//=============================================================================
// tickfreq
// Synopsis: return the tick frequency
//=============================================================================
double CHighResolutionTimer::tickfreq() {  
	LARGE_INTEGER tps;
	query_freq(&tps); 
	return (double)hpfresult(tps);
}

//=============================================================================
// tickcnt
// Synopsis: return the current tick count
//=============================================================================
__int64 CHighResolutionTimer::tickcnt() {  
	LARGE_INTEGER t;
	query_hpc(&t); 
	return (__int64)hpfresult(t);
}

//=============================================================================
// reset
// Synopsis: reset the timer
//=============================================================================
void CHighResolutionTimer::reset() {
	if(!m_freq) 
		m_freq = tickfreq();
	m_start = tickcnt();
}

//=============================================================================
// elapsed
// Synopsis: return the elapsed time
//=============================================================================
ctrtype 
CHighResolutionTimer::elapsed(bool reset) {
	__int64 end = tickcnt();
	double res = ((double)(end-m_start))/m_freq;
	assert(res >= 0.0);
	if(reset)
		m_start = end;
	return m_gran == gran_sec ? res : res * 1000;
}

LPFNtQuerySystemTime 
CHighResolutionTimer::init_query_system_time(
	VOID
	)
{
	m_hModule = LoadLibraryW(L"NTDLL.DLL");
	FARPROC x = GetProcAddress(m_hModule, "NtQuerySystemTime");
	m_lpfnQuerySystemTime = (LPFNtQuerySystemTime) x;
	return m_lpfnQuerySystemTime;
}

BOOL
CHighResolutionTimer::query_system_time(
	PLARGE_INTEGER li
	)
{
	if(!m_lpfnQuerySystemTime) 
		return FALSE;
	(*m_lpfnQuerySystemTime)(li);
	return TRUE;
}

DWORD 
CHighResolutionTimer::delta_milliseconds(
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

VOID
CHighResolutionTimer::free_query_system_time(
	VOID
	)
{
	if(m_hModule) {
		FreeLibrary(m_hModule);
	}
}
