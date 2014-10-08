// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the HOSTTASKS_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// HOSTTASKS_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef HOSTTASKS_EXPORTS
#define HOSTTASKS_API __declspec(dllexport)
#else
#define HOSTTASKS_API __declspec(dllimport)
#endif

// This class is exported from the HostTasks.dll
class HOSTTASKS_API CHostTasks {
public:
	CHostTasks(void);
	// TODO: add your methods here.
};

extern HOSTTASKS_API int nHostTasks;

HOSTTASKS_API int fnHostTasks(void);

extern "C" {
HOSTTASKS_API void __stdcall
htmatmul(
    UINT nArguments,
    void **ppArguments
    );
}

extern "C" HOSTTASKS_API void __stdcall
htvadd(
    UINT nArguments,
    void **ppArguments
    );
