///-------------------------------------------------------------------------------------------------
// file:	confighelpers.h
//
// summary:	Declares macros to help configure and report PTask runtime settings 
///-------------------------------------------------------------------------------------------------

#ifndef __PTASK_CONFIG_HELPERS_H__
#define __PTASK_CONFIG_HELPERS_H__
#include "PTaskRuntime.h"
using namespace PTask;
static char * vptostr(VIEWMATERIALIZATIONPOLICY p) { return (char*)ViewPolicyString(p); }
static char * tpptostr(THREADPOOLPOLICY p) { return (p == TPP_AUTOMATIC) ? "AUTOMATIC" : ((p == TPP_EXPLICIT) ? "EXPLICIT":"THREADPERTASK"); }
#ifdef _DEBUG 
#define cfgverboseflag TRUE
#else
extern BOOL g_verbose;
#define cfgverboseflag g_verbose
#endif
#define CONFIGUREPTASK(setting, bsetflag, breqstate) {  \
    if(bsetflag) { Runtime::Set##setting(breqstate); }  \
    if(cfgverboseflag) {                                \
        std::cout << "PTask::Runtime::" << #setting     \
                  << "=" << Runtime::Get##setting()     \
                  << std::endl; } }
#define CONFIGUREPTASKU(setting, breqstate) {           \
    Runtime::Set##setting(breqstate);                   \
    if(cfgverboseflag) {                                \
        std::cout << "PTask::Runtime::" << #setting     \
                  << "=" << Runtime::Get##setting()     \
                  << std::endl; } }
#define CONFIGUREPTASKI(setting, bsetflag, breqstate) {  \
    if(bsetflag) { Runtime::Set##setting(breqstate); }  \
    if(cfgverboseflag) {                                \
        std::cout << "PTask::Runtime::" << #setting     \
                  << "=" << Runtime::Is##setting()      \
                  << std::endl; } }
#define CONFIGUREPTASKE(setting, bsetflag, breqstate, strfun) {   \
    if(bsetflag) { Runtime::Set##setting(breqstate); }            \
    if(cfgverboseflag) {                                          \
        std::cout << "PTask::Runtime::" << #setting               \
        << "=" << strfun(Runtime::Get##setting())                 \
                  << std::endl; } }
#endif