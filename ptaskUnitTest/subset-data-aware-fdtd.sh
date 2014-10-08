#!/bin/bash
ITERATIONS=2
SUCCEEDED=0
FAILURES=0

echo N,Z,I,d,c1-st,c0-st,c0-mt-da,c0-st-da,cuda

for d in 8 16 32; do
for I in 10; do
for Z in 4; do
for N in 128; do
for i in `seq 1 $ITERATIONS`; do
  ../../bin/x64/Release/PTaskUnitTest.exe -J -C 1 -m 1 -G -t cufdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > cutemp-C1-st.txt
  ../../bin/x64/Release/PTaskUnitTest.exe -J -C 0 -m 1 -G -t cufdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > cutemp-C0-st.txt
  ../../bin/x64/Release/PTaskUnitTest.exe -C 0 -m 2 -G -t cufdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > cutemp-C0-mt-DA.txt
  ../../bin/x64/Release/PTaskUnitTest.exe -J -C 0 -m 2 -G -t cufdtd -d fdtdKernels -r $N -c $N -z $Z -n $d -i $I > cutemp-C0-st-DA.txt
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
  c1st=`egrep GPU cutemp-C1-st.txt | awk '{ print $3 }'`
  c0st=`egrep GPU cutemp-C0-st.txt | awk '{ print $3 }'`
  c0mtda=`egrep GPU cutemp-C0-mt-DA.txt | awk '{ print $3 }'`
  c0stda=`egrep GPU cutemp-C0-st-DA.txt | awk '{ print $3 }'`
  srefgputime=`egrep GPU sreftemp.txt | awk '{ print $3 }'`
  if [ "$i" = "1" ]; then
    echo $N,$Z,$I,$d,$c1st,$c0st,$c0mtda,$c0stda,$srefgputime
  else
    echo "  , , , ,$c1st,$c0st,$c0mtda,$c0stda,$srefgputime"
  fi
done
done
done
done
done
