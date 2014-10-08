@echo off
call ptxenv.bat

set CUSRC=%PTUROOT%\initports.cu
set CUDST=%PTUROOT%\initports.ptx
set CUDST32=%PTUROOT%\initports32.ptx
NVCC -m64 %OPTS% -ccbin "%VC9BIN%" -I%INCPATH% -ptx -o "%CUDST%" "%CUSRC%"
NVCC -m32 %OPTS% -ccbin "%VC9BIN%" -I%INCPATH% -ptx -o "%CUDST32%" "%CUSRC%"
