REM -----------------------------------------------
REM set up shared variables for all the CUDA PTX 
REM builds. Note that we still use CUDA 3.2 here
REM and require VC9 to be installed. Presumably 
REM that requirement should go away soon.
REM -----------------------------------------------
@echo off

set NVCC=%CUDA_PATH%\bin\nvcc.exe
set CUDAINC=%CUDA_PATH%\include
set VCBIN=%VCINSTALLDIR%\bin
set COMMONINC=..\..\common\inc;..\common\inc
set SHAREDINC=..\..\..\shared\inc;..\..\shared\inc
set INCPATH="./";"%CUDAINC%";"%COMMONINC%";"%SHAREDINC%";"%PTASK%"
set PTASK=%DANDELION_ROOT%\accelerators\ptask
set OPTS= -arch sm_20 -maxrregcount=32
set PTUROOT=%DANDELION_ROOT%\accelerators\PTaskUnitTest\
