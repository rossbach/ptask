///-------------------------------------------------------------------------------------------------
// file:	Tracer.cpp
//
// summary:	Implements the tracer class
///-------------------------------------------------------------------------------------------------

#include "Tracer.h"
#include <strsafe.h>
#include <Wmistr.h>
#include <assert.h>

namespace PTask {
namespace Runtime {

    Tracer::Tracer(void)
    {
    }

    Tracer::~Tracer(void)
    {
    }

    ULONG 
    Tracer::LogDispatchEvent(
        char * lpszTaskName, 
        BOOL bStart, 
        UINT uiAcceleratorId,
        UINT uiDispatchNumber
        )
    {
        if(!PTask::Runtime::GetDispatchTracingEnabled())
            return 0;
        ULONG ulStatus = ERROR_SUCCESS;
        char msg[TRACER_MAX_MSG_LEN];
        _snprintf_s(
            msg,
            TRACER_MAX_MSG_LEN,
            _TRUNCATE, // Truncate. -1 return indicates truncation occurred, if anyone cares.
            "%s%s D:%u A:%u",
            bStart ? "[" : "]",
            lpszTaskName,
            uiAcceleratorId,
            uiDispatchNumber);
        EtwSetMarkA(msg);
        return ulStatus;
    }

    ULONG 
    Tracer::LogBufferSyncEvent(
        void * pbufferInstance,
        BOOL bStart,
        void * parentDatablock,
        UINT uiAcceleratorId)
    {
        if(!PTask::Runtime::GetDispatchTracingEnabled())
            return 0;
        ULONG ulStatus = ERROR_SUCCESS;
        char msg[TRACER_MAX_MSG_LEN];
        _snprintf_s(
            msg,
            TRACER_MAX_MSG_LEN,
            _TRUNCATE, // Truncate. -1 return indicates truncation occurred, if anyone cares.
            "%sPB:%p DB:%p A:%u",
            bStart ? "(" : ")",
            pbufferInstance,
            parentDatablock,
            uiAcceleratorId);
        EtwSetMarkA(msg);
        return ulStatus;
    }

    VOID
    Tracer::InitializeETW()
    {
        // printf("\n*** Tracer::InitializeETW()\n");
        HMODULE hNTDll = GetModuleHandleW( L"NTDLL" );
        gs_pEtwSetMark = (LPETWSETMARK)GetProcAddress( hNTDll, "EtwSetMark" );
    }

    VOID
    Tracer::EtwSetMarkA(char *msg)
    {
        // printf("*** Tracer::EtwSetMarkA: %s\n", msg);

        ETW_SET_MARK_INFORMATION smi;
        if (gs_pEtwSetMark == NULL) InitializeETW();
 
        size_t msglen = strlen(msg);
        if (msglen > TRACER_MAX_MSG_LEN) return;
        smi.Flag = 0;

        sprintf_s(smi.Mark, TRACER_MAX_MSG_LEN, msg);
        // Paul's orig C++ version:
        //   gs_pEtwSetMark(NULL, &smi, sizeof(ULONG)+(int)msglen);
        // c.f. this from C# version: 
        //   var ret = EtwSetMark(0, ref smi, 8 + smi.Mark.Length * 2);
        gs_pEtwSetMark(0, &smi, sizeof(ULONG)+(int)msglen);

    }

};
};