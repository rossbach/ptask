@echo off
call ptxenv.bat

set FDTDSRC=%DANDELION_ROOT%\accelerators\ReferenceFDTD\fdtdKernels.cu
NVCC -m64 %OPTS% -ccbin "%VC9BIN%" -I%INCPATH% -ptx -o "fdtdKernels\fdtdMain.compute_10.ptx" "%FDTDSRC%"
NVCC -m32 %OPTS% -ccbin "%VC9BIN%" -I%INCPATH% -ptx -o "fdtdKernels32\fdtdMain.compute_10.ptx" "%FDTDSRC%"

