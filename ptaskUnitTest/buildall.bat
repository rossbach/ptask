@echo off

sed --help > NUL:
if errorlevel 1 goto needcygwin

devenv /rebuild "Release|Win32" ..\..\Dandelion.sln
set x86ReleaseRes=%ERRORLEVEL%
devenv /rebuild "Debug|Win32" ..\..\Dandelion.sln
set x86DebugRes=%ERRORLEVEL%
devenv /rebuild "Debug|x64" ..\..\Dandelion.sln
set x64DebugRes=%ERRORLEVEL%
devenv /rebuild "Release|x64" ..\..\Dandelion.sln
set x64ReleaseRes=%ERRORLEVEL%

if not %x64DebugRes%.==0. echo x64 debug build failed!
if not %x64ReleaseRes%.==0. echo x64 release build failed!
if not %x86DebugRes%.==0. echo Win32 debug build failed!
if not %x86ReleaseRes%.==0. echo Win32 release build failed!

if not %x64DebugRes%.==0. goto errExit
if not %x64ReleaseRes%.==0. goto errExit
if not %x86DebugRes%.==0. goto errExit
if not %x86ReleaseRes%.==0. goto errExit

echo All configurations built successfully!
exit /B 0 

:needcygwin
echo ERROR!
echo This script requires that Cygwin is in your path.
exit /B 1

:errExit
echo ERROR!
echo At least one build configuration failed!
exit /B 1

