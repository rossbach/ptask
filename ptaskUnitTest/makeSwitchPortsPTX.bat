 "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v3.2\\bin\nvcc.exe"   -m64 -arch sm_10 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v3.2\/include" -I"./" -I"../../common/inc" -I"../../../shared/inc" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v3.2\\include" -maxrregcount=32  -ptx -o "switchports.ptx" "c:\SVC\Dandelion\accelerators\PTaskUnitTest\switchports.cu"
 "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v3.2\\bin\nvcc.exe"   -m32 -arch sm_10 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v3.2\/include" -I"./" -I"../../common/inc" -I"../../../shared/inc" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v3.2\\include" -maxrregcount=32  -ptx -o "switchports32.ptx" "c:\SVC\Dandelion\accelerators\PTaskUnitTest\switchports.cu"