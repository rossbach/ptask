#!/bin/bash
ITERATIONS=2
SUCCEEDED=0
FAILURES=0

echo N,Z,I,d,c1-mt,c1-st,c0-mt,c0-st,c0-mt-da,c0-st-da,cuda-sync,cuda,cpu

for d in 4 8 16; do
for I in 10; do
for Z in 4; do
for N in 128 256; do
for i in `seq 1 $ITERATIONS`; do
  ../../bin/x64/Release/PTaskUnitTest.exe -C 1 -m 1 -G -t cufdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > cutemp-C1-mt.txt
  ../../bin/x64/Release/PTaskUnitTest.exe -j -C 1 -m 1 -G -t cufdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > cutemp-C1-st.txt
  ../../bin/x64/Release/PTaskUnitTest.exe -C 0 -m 1 -G -t cufdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > cutemp-C0-mt.txt
  ../../bin/x64/Release/PTaskUnitTest.exe -j -C 0 -m 1 -G -t cufdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > cutemp-C0-st.txt
  ../../bin/x64/Release/PTaskUnitTest.exe -C 0 -m 2 -G -t cufdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > cutemp-C0-mt-DA.txt
  ../../bin/x64/Release/PTaskUnitTest.exe -j -C 0 -m 2 -G -t cufdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > cutemp-C0-st-DA.txt
  ../../bin/x64/Release/ReferenceFDTD.exe -x $N -y $N -z $Z -n $d -i $I > reftemp.txt
  ../../bin/x64/Release/ReferenceFDTD.exe -x $N -y $N -z $Z -n $d -i $I -s > sreftemp.txt
  HE=""
  CUE=""
  if egrep "failed" cutemp.txt > /dev/null; then
    CUE="*"
  fi
  if egrep "failed" hlsltemp.txt > /dev/null; then
    HE="*"
  fi
  SUCCEEDED=`echo "$SUCCEEDED + 1" | bc`
  c1mt=`egrep GPU cutemp-C1-mt.txt | awk '{ print $3 }'`
  c1st=`egrep GPU cutemp-C1-st.txt | awk '{ print $3 }'`
  c0mt=`egrep GPU cutemp-C0-mt.txt | awk '{ print $3 }'`
  c0st=`egrep GPU cutemp-C0-st.txt | awk '{ print $3 }'`
  c0mtda=`egrep GPU cutemp-C0-mt-DA.txt | awk '{ print $3 }'`
  c0stda=`egrep GPU cutemp-C0-st-DA.txt | awk '{ print $3 }'`
  cputime=`egrep CPU cutemp-C1-mt.txt | awk '{ print $3 }'`
  refgputime=`egrep GPU reftemp.txt | awk '{ print $3 }'`
  srefgputime=`egrep GPU sreftemp.txt | awk '{ print $3 }'`
  if [ "$i" = "1" ]; then
    echo $N,$Z,$I,$d,$c1mt,$c1st,$c0mt,$c0st,$c0mtda,$c0stda,$refgputime,$srefgputime,$cputime
  else
    echo "  , , , ,$c1mt,$c1st,$c0mt,$c0st,$c0mtda,$c0stda,$refgputime,$srefgputime,$cputime"
  fi
done
done
done
done
done
