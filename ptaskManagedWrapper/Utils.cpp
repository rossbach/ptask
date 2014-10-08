///-------------------------------------------------------------------------------------------------
// file:	Utils.h
//
// summary:	Declares the utilities class
///-------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "PTaskManagedWrapper.h"

namespace Microsoft {
namespace Research {
namespace PTask {
namespace Utils {

    char * 
    MarshalString(
        __in String^ pString
        ) 
    {
        IntPtr pMemoryExtent = Marshal::StringToHGlobalAnsi(pString);
        return static_cast<char*>(pMemoryExtent.ToPointer());
    }

    void
    FreeUnmanagedString(
        __inout char * lpszString
        )
    {
        if(lpszString != nullptr) {
            Marshal::FreeHGlobal(IntPtr(lpszString));
        }
    }
}}}}