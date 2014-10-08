@echo off

set F=%DANDELION_ROOT%\accelerators\PTaskUnitTest\pca.ptx
set I=10
set J=10
set K=3

rem set I=1
rem set J=1
rem set K=1

echo.
echo Iterations=%I%, Inner iterations=%J%, Components(K)=%K%
echo.
echo			PTask		CPU

set L=1
set S=preNorm
echo %S% Pipelining enabled
rem echo PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r  128 -c  128 -n 1 -i %I% -j %J% -K %K%
rem PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r  4 -c  4 -n 1 -i %I% -j %J% -K %K%

rem goto :end
PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r  128 -c  128 -n 1 -i %I% -j %J% -K %K% -L %L%
PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r  256 -c  256 -n 1 -i %I% -j %J% -K %K% -L %L%
PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r  512 -c  512 -n 1 -i %I% -j %J% -K %K% -L %L%
PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r 1024 -c 1024 -n 1 -i %I% -j %J% -K %K% -L %L%
PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r 2048 -c 2048 -n 1 -i %I% -j %J% -K %K% -L %L%


set L=0
set S=preNorm
echo %S% Pipelining disabled
PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r  128 -c  128 -n 1 -i %I% -j %J% -K %K% -L %L%
PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r  256 -c  256 -n 1 -i %I% -j %J% -K %K% -L %L%
PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r  512 -c  512 -n 1 -i %I% -j %J% -K %K% -L %L%
PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r 1024 -c 1024 -n 1 -i %I% -j %J% -K %K% -L %L%
PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r 2048 -c 2048 -n 1 -i %I% -j %J% -K %K% -L %L%

goto :end

set S=loopNorm
rem set S=preNorm
echo.
echo %S%
rem echo PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r   8 -c   8 -n 1 -i %I%  -j %J% -K %K%
rem PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r   8 -c   8 -n 1 -i %I%  -j %J% -K %K%
PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r   64 -c   64 -n 1 -i %I%  -j %J% -K %K%
PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r  256 -c  256 -n 1 -i %I%  -j %J% -K %K%
PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r  512 -c  512 -n 1 -i %I%  -j %J% -K %K%
rem PTaskUnitTest.exe -G -t cupca -f %F% -s %S% -r 1024 -c 1024 -n 1 -i %I%  -j %J% -K %K%

echo.

:end