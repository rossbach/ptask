#!/bin/bash
ITERATIONS=3
SUCCEEDED=0
FAILURES=0

#for schedmode in 0 1 2 3; do
dataaware=2
priority=1
maxconc=0
echo N,Z,I,d,cu-gpu-c1,cu-gpu-c0,cu-gpu-c0-da,hlsl-gpu-c0-da,ref-gpu,cpu,ref-cputime,CU-E,HLSL-E

for d in 1 2 4 6; do
for I in 5 10; do
for Z in 4; do
for N in 64 128 256; do
for i in `seq 1 $ITERATIONS`; do
  ../../bin/x64/Release/PTaskUnitTest.exe -C 1 -m $priority -G -t cufdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > cutemp-C1.txt
  ../../bin/x64/Release/PTaskUnitTest.exe -C 0 -m $priority -G -t cufdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > cutemp-prio-C0.txt
  ../../bin/x64/Release/PTaskUnitTest.exe -C 0 -m $dataaware -G -t cufdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > cutemp-da-C0.txt
  ../../bin/x64/Release/PTaskUnitTest.exe -C 0 -m $dataaware -G -t hlslfdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > hlsltemp.txt
  ../../bin/x64/Release/ReferenceFDTD.exe -x $N -y $N -z $Z -n $d -i $I > reftemp.txt
  HE=""
  CUE=""
  if egrep "failed" cutemp.txt > /dev/null; then
    CUE="*"
  fi
  if egrep "failed" hlsltemp.txt > /dev/null; then
    HE="*"
  fi
  SUCCEEDED=`echo "$SUCCEEDED + 1" | bc`
  cugputime_c1=`egrep GPU cutemp-C1.txt | awk '{ print $3 }'`
  cugputime_c0=`egrep GPU cutemp-prio-C0.txt | awk '{ print $3 }'`
  cugputime_c0_da=`egrep GPU cutemp-da-C0.txt | awk '{ print $3 }'`
  hlgputime=`egrep GPU hlsltemp.txt | awk '{ print $3 }'`
  cputime=`egrep CPU cutemp-C1.txt | awk '{ print $3 }'`
  refcputime=`egrep CPU reftemp.txt | awk '{ print $3 }'`
  refgputime=`egrep GPU reftemp.txt | awk '{ print $3 }'`
  if [ "$i" = "1" ]; then
    echo $N,$Z,$I,$d,$cugputime_c1,$cugputime_c0,$cugputime_c0_da,$hlgputime,$refgputime,$cputime,$refcputime,$CUE,$HE
  else
    echo "  , , , ,$cugputime_c1,$cugputime_c0,$cugputime_c0_da,$hlgputime,$refgputime,$cputime,$refcputime,$CUE,$HE  "
  fi
done
done
done
done
done
