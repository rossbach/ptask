@echo off

set VC10=C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC
call "%VC10%\vcvarsall.bat"

echo %temp%

set BUILDLIST=
set BUILDLIST=%BUILDLIST% initializerchannels
set BUILDLIST=%BUILDLIST% pipelinestresstest
set BUILDLIST=%BUILDLIST% iteration
set BUILDLIST=%BUILDLIST% channelpredication
set BUILDLIST=%BUILDLIST% controlpropagation
set BUILDLIST=%BUILDLIST% deferredports
set BUILDLIST=%BUILDLIST% gatedports
set BUILDLIST=%BUILDLIST% inout
set BUILDLIST=%BUILDLIST% initports
set BUILDLIST=%BUILDLIST% metaports
set BUILDLIST=%BUILDLIST% switchports
set BUILDLIST=%BUILDLIST% vectorAdd
set BUILDLIST=%BUILDLIST% descportsout
set BUILDLIST=%BUILDLIST% permanentblocks
set BUILDLIST=%BUILDLIST% bufferdimsdesc

for %%x in (%BUILDLIST%) do call buildPTXfile.bat %%x

call makeCUFDTDPTX.bat


