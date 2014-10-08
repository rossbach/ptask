@echo off
call ptxenv.bat
set ITEM=%1
set CUSRC=%PTUROOT%\%ITEM%.cu
set CUDST=%PTUROOT%\%ITEM%.ptx
set CUDST32=%PTUROOT%\%ITEM%32.ptx
NVCC -m64 %OPTS% -ccbin "%VCBIN%" -I%INCPATH% -ptx -o "%CUDST%" "%CUSRC%"
NVCC -m32 %OPTS% -ccbin "%VCBIN%" -I%INCPATH% -ptx -o "%CUDST32%" "%CUSRC%"
