///-------------------------------------------------------------------------------------------------
// file:	Utils.h
//
// summary:	Declares the utilities class
///-------------------------------------------------------------------------------------------------

#pragma once
#pragma managed
using namespace System;
using namespace System::Runtime::InteropServices; // For marshalling helpers

namespace Microsoft {
namespace Research {
namespace PTask {
namespace Utils {

    char * 
    MarshalString(
        __in String^ pString
        );

    void
    FreeUnmanagedString(
        __inout char * lpszString
        );    
}}}}