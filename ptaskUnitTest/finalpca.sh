#!/bin/bash
ITERATIONS=3
SUCCEEDED=0
FAILURES=0
S=preNorm

#for schedmode in 0 1 2 3; do
dataaware=2
priority=1
maxconc=0
echo N,J,stmod,mod,handcode,cublas,ptask,cpu

outpca=outpca
if [ ! -e $outpca ]; then
	mkdir $outpca
fi
rm -rf $outpca/*

for I in 10; do
for J in 10 54; do
for K in 3; do
for N in 128 256 512; do
for i in `seq 1 $ITERATIONS`; do
  ../../bin/x64/Release/PTaskUnitTest.exe -J -R -C 1 -m 3 -G -t cupca -f pca.ptx -s $S -r  $N -c  $N -n 1 -i $I -j $J -K $K > $outpca/pca-ptask-$N-i$I-k$K-j$J.txt
  ../../scratch/jcurrey/matrixMul/bin/win64/Release/pca_cuda.exe -M -r $N -c $N -i $I -j $J -s $S -K $K > $outpca/pca-stmod-$N-i$I-k$K-j$J.txt
  ../../bin/x64/Release/PTaskUnitTest.exe -J -E -R -C 1 -m 3 -L 0 -G -t cupca -f pca.ptx -s $S -r $N -c  $N -n 1 -i $I -j $J -K $K > $outpca/pca-mod-$N-i$I-k$K-j$J.txt
  ../../scratch/jcurrey/matrixMul/bin/win64/Release/pca_cuda.exe -r $N -c $N -i $I -j $J -s $S -K $K > $outpca/pca-handcode-$N-i$I-k$K-j$J.txt
  ptasktime=`egrep $Nx$N $outpca/pca-ptask-$N-i$I-k$K-j$J.txt | awk '{ print $2 }'`
  stmodtime=`egrep $Nx$N $outpca/pca-stmod-$N-i$I-k$K-j$J.txt | awk '{ print $3 }'`
  modtime=`egrep $Nx$N $outpca/pca-mod-$N-i$I-k$K-j$J.txt | awk '{ print $2 }'`
  handcodetime=`egrep $Nx$N $outpca/pca-handcode-$N-i$I-k$K-j$J.txt | awk '{ print $3 }'`
  cputime=`egrep $Nx$N $outpca/pca-handcode-$N-i$I-k$K-j$J.txt | awk '{ print $4 }'`
  cublastime=`egrep $Nx$N $outpca/pca-handcode-$N-i$I-k$K-j$J.txt | awk '{ print $5 }'`
  if [ "$i" = "1" ]; then
    echo $N,$J,$stmodtime,$modtime,$handcodetime,$cublastime,$ptasktime,$cputime
  else
    echo " , ,$stmodtime,$modtime,$handcodetime,$cublastime,$ptasktime,$cputime"
  fi
done
done
done
done
done
