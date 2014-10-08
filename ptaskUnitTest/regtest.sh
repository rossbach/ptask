#!/bin/bash
(set -o igncr) 2>/dev/null && set -o igncr; # this comment is required
# Above line makes this script ignore Windows line endings, to avoid this error without having to run dos2unix:
#   $'\r': command not found
# As per http://stackoverflow.com/questions/14598753/running-bash-script-in-cygwin-on-windows-7

# regtest.sh [--iter i] [--verbose] [--task t] [--runtime r] [--platform p] [--target T] [--schedmode s] [--conc c] [--rebuild] [--buildshaders] [--help]
let SUCCEEDED=0
let FAILURES=0
let UNSUPPORTED=0

usage() {
  echo "ptask regression test usage:"
  echo "regtest.sh [--iter i] [--verbose] [--task t] [--platform p] [--runtime r] [--target T] [--schedmode s] [--conc c] [--serial s] [--rebuild] [--buildshaders] [--inviewmatpol P] [--outviewmatpol P] [--help]"
  echo "--iter i:        run each case i times"
  echo "--runtime r:     run only cases with the given back-end runtime r in [DirectX|CUDA|OpenCL]"
  echo "--verbose:       extra output"
  echo "--task t:        runs specific task t, otherwise runs all known tasks"
  echo "--platform p:    runs specific platform, otherwise all. p in [Win32|x64]"
  echo "--target T:      runs specific target only, otherwise all. T in [Debug|Release]"
  echo "--schedmode s:   tests only in given schedmode, s in [0..3]"
  echo "--threadpool t:  specifies the number of threads for graphrunnerprocs: 0->(1:1-Task:Thread), 1->single-thread, >1->TPP_EXPLICIT"
  echo "--exclbattery b: exclude a battery of tests, b in [mem, rtt, func]"
  echo "--inviewmatpol p: include only tests using input view materialization policy p, p in [EAGER, DEMAND]"
  echo "--outviewmatpol p: include only tests using output view materialization policy p, p in [EAGER, DEMAND]"
  echo "--appthrdctx a:  do/don't allow user/application threads to have default device context (b in [0,1])"
  echo "--battery b:     run only a selected battery of tests, b in [mem, rtt, func]"
  echo "                 0 = SCHEDMODE_COOPERATIVE"
  echo "                 1 = SCHEDMODE_PRIORITY"
  echo "                 2 = SCHEDMODE_DATADRIVEN"
  echo "                 3 = SCHEDMODE_FIFO"
  echo "--conc c:      tests only with -C c, c in [0..n]"
  echo "                 0 means do not limit number of accelerators"
  echo "                 n means use a max of n physical devices"
  echo "--serial s:    tests only in given graph serialization mode, s in [0..2]"
  echo "                 0 = no graph serialization (default)"
  echo "                 1 = serialize graph and exit"
  echo "                 2 = load serialized graph and run test"
  echo "--rebuild:      rebuild all binaries before running"
  echo "--buildshaders: rebuild all PTX files"
  echo "--bindir dir:   look for binaries in dir"
  echo "--rootdir dir:  root of src tree in dir (rather than DANDELION_ROOT\accelerators)"
  echo "--showoutput:   dump program output to console"
  echo "--forcesync:    use PTask force synchronous mode"
  echo "--xtrace:       use PTask extreme trace mode"
  echo "--help:         prints this message"
}

allplatforms=( "x64" "Win32" )
alltargets=( "Debug" "Release" "DebugLynx" "ReleaseLynx" )
BINDIR=`cygpath -w "$DANDELION_ROOT/DistributedDandelionTests/bin"`
# echo BINDDIR=${BINDIR}
ROOTDIR=`cygpath -w "$DANDELION_ROOT/ptask"`
ITERATIONS=1
specificappthrdctx=""
specificinviewmatpolicy=""
specificoutvewmatpolicy=""
specificbattery=""
excludebattery=""
specificthreadpool=""
specificruntime=""
specifictask=""
specificplatform=""
specifictarget=""
specificschedmode=""
specificconc=""
specificserialmode=""
forcesyncflags=" "
xtraceflags=" "
verbose="FALSE"
showoutput=0
rebuild=0
buildshaders=0
readmore=1
help=0
echo "skipping OpenCL test cases...NVIDIA OpenCL with CUDA 5.0 is known broken..."
until [ -z $readmore ]; do
  readmore=
  if [ "x$1" == "x--bindir" ]; then
    BINDIR=`cygpath -u "${2}"`
    echo "run binaries in $BINDIR"
	shift
	shift
	readmore=1
  fi      
  if [ "x$1" == "x--rootdir" ]; then
    ROOTDIR=`cygpath -u "${2}"`
    echo "changing PTask source tree root to $ROOTDIR"
	shift
	shift
	readmore=1
  fi      
  if [ "x$1" == "x--help" ]; then
    help=1
	shift
	readmore=1
  fi      
  if [ "x$1" == "x--showoutput" ]; then
    showoutput=1
	shift
	readmore=1
  fi      
  if [ "x$1" == "x--forcesync" ]; then
    forcesyncflags=" -F "
	shift
	readmore=1
  fi      
  if [ "x$1" == "x--xtrace" ]; then
    xtraceflags=" -A "
	shift
	readmore=1
  fi      
  if [ "x$1" == "x--rebuild" ]; then
    rebuild=1
    echo "rebuild binaries first"
	shift
	readmore=1
  fi      
  if [ "x$1" == "x--buildshaders" ]; then
    buildshaders=1
    echo "rebuild PTX files first"
	shift
	readmore=1
  fi         
  if [ "x$1" == "x--appthrdctx" ]; then
    specificappthrdctx=$2
    echo "run only appthrdctx = $specificappthrdctx  test cases"
	shift
	shift
	readmore=1
  fi        
  if [ "x$1" == "x--battery" ]; then
    specificbattery=$2
    echo "run only $specificbattery test cases"
	shift
	shift
	readmore=1
  fi        
  if [ "x$1" == "x--inviewmatpol" ]; then
    specificinviewmatpolicy=$2
    echo "run only inmatpol=$specificinviewmatpolicy test cases"
	shift
	shift
	readmore=1
  fi        
  if [ "x$1" == "x--outviewmatpol" ]; then
    specificoutviewmatpolicy=$2
    echo "run only outmatpol=$specificoutviewmatpolicy test cases"
	shift
	shift
	readmore=1
  fi        
  if [ "x$1" == "x--exclbattery" ]; then
    excludebattery=$2
    echo "exclude $excludebattery test cases"
	shift
	shift
	readmore=1
  fi          
  if [ "x$1" == "x--threadpool" ]; then
    specificthreadpool=$2
    echo "run only $specificthreadpool test cases"
	shift
	shift
	readmore=1
  fi            
  if [ "x$1" == "x--runtime" ]; then
    specificruntime=$2
    echo "run only $specificruntime test cases"
	shift
	shift
	readmore=1
  fi      
  if [ "x$1" == "x--conc" ]; then
    specificconc=$2
    echo "run only specific concurrency level $specificconc"
	shift
	shift
	readmore=1
  fi      
  if [ "x$1" == "x--schedmode" ]; then
    specificschedmode=$2
    echo "run only specific schedmode $specificschedmode"
	shift
	shift
	readmore=1
  fi      
  if [ "x$1" == "x--serial" ]; then
    specificserialmode=$2
    echo "run only specific graph serialization mode $specificserialmode"
	shift
	shift
	readmore=1
  fi      
  if [ "x$1" == "x--target" ]; then
    specifictarget=$2
    echo "run only specific target $specifictarget"
	shift
	shift
	alltargets=( $specifictarget ) 
	readmore=1
  fi      
  if [ "x$1" == "x--platform" ]; then
    specificplatform=$2
    echo "run only specific platform $specificplatform"
	shift
	shift
	allplatforms=( $specificplatform ) 
	readmore=1
  fi      
  if [ "x$1" == "x--iter" ]; then
    ITERATIONS=$2
    echo "task iterations: $ITERATIONS"
	shift
	shift
	readmore=1
  fi    
  if [ "x$1" == "x--verbose" ]; then
    verbose=TRUE
    echo "verbose mode"
	shift
	readmore=1
  fi    
  if [ "x$1" == "x--task" ]; then
    specifictask=$2
    echo "run specific task $specifictask"
	shift
	shift
	readmore=1
  fi  
done  

if [ "$help" == "1" ]; then 
  usage
  exit 0
fi

testCase() {
  outdir=$1
  platform=$2
  target=$3
  concurrency=$4
  schedmode=$5
  threadpool=$6
  threadpoolpolicy=$7
  icmatpolicy=$8
  ocmatpolicy=$9
  appthrdctx=${10}
  schedthreads=${11}
  serialmode=${12}
  iters=${13}
  task=${14}
  shaderfile=${15}
  shaderop=${16}
  args=${17}
  
  # echo "outdir=$outdir"
  # echo "threadpool=$threadpool"
  # echo "conc=$concurrency"
  # echo "schedmode=$schedmode"
  # echo "schedthreads=$schedthreads"
  # echo "serialmode=$serialmode"
  # echo "task=$task"
  # echo "iters=$iters"
  # echo "shaderop=$shaderop"
  # echo "shaderfile=$shaderfile"
  # echo "testCase: iters=$iters, task=$task, sfile=$shaderfile, shaderop=$shaderop, args=$args"  

  if [ "$specificappthrdctx" != "" ]; then 
	if [ "$specificappthrdctx" != "$appthrdctx" ]; then 
	  return
	fi  
  fi  
  if [ "$specificinviewmatpolicy" != "" ]; then 
	if [ "$specificinviewmatpolicy" != "$icmatpolicy" ]; then 
	  return
	fi  
  fi
  
  if [ "$specificoutviewmatpolicy" != "" ]; then 
	if [ "$specificoutviewmatpolicy" != "$ocmatpolicy" ]; then 
	  return
	fi  
  fi  
  
  if [ "$specificbattery" != "" ]; then 
	if [ "$specificbattery" != "func" ]; then 
	  return
	fi  
  fi
  
  if [ "$excludebattery" != "" ]; then 
	if [ "$excludebattery" == "func" ]; then 
	  return
	fi  
  fi  
  
  shaderplatform=""
  case "$shaderfile" in
    *.hlsl ) 
	  shaderplatform="DX"
	  ;;
	*.ptx )
	  shaderplatform="cu"
	  ;;
	*.cl )
	  shaderplatform="cl"
	  ;;
	*.dll )
	  shaderplatform="host"
	  ;;
	"" )
	  case "$task" in 
	    hlsl* )
	      shaderplatform="DX"
		  ;;
		cu* )
		  shaderplatform="cu"
		  ;;
		* )
	      echo "Unknown backend runtime for shader file=$shaderfile, task=$task"
	      return
	      ;;
      esac
	  ;;
	* )
	  echo "Unknown backend runtime for shader file=$shaderfile, task=$task"
	  return
	  ;;
  esac

  #echo Considering $platform $target Conc $concurrency $schedmode $serialmode $iters $task $shaderfile $shaderop $args
  if [ "$shaderplatform" == "cl" ]; then 
	return
  fi
  
  if [ "$specificruntime" != "" ]; then
    case "$specificruntime" in
	  "DirectX" )  
		if [[ $shaderfile != *.hlsl ]]; then
		   return
		fi
		;;
	  "CUDA" )    
		if [[ $shaderfile != *.ptx && $shaderplatform != cu ]]; then
		   return
		fi
		;;
	  "OpenCL" )   
		if [[ $shaderfile != *.cl ]]; then
		   return
		fi
		;;
	  "host" )
		if [[ $shaderfile != *.dll ]]; then
		   return
		fi
		;;
	  *)           
	    echo "Unknown runtime restriction!"
	    exit 1
		;;
	esac			 
  fi
  if [ "$specifictask" != "" ]; then
    if [ "$task" != "$specifictask" ]; then
      return
	fi
  fi
  if [ "$specificplatform" != "" ]; then
    if [ "$platform" != "$specificplatform" ]; then
      return
	fi
  fi
  if [ "$specifictarget" != "" ]; then
    if [ "$target" != "$specifictarget" ]; then
      return
	fi
  fi
  if [ "$specificschedmode" != "" ]; then
    if [ "$schedmode" != "$specificschedmode" ]; then
      return
	fi
  fi
  if [ "$specificconc" != "" ]; then
    if [ "$concurrency" != "$specificconc" ]; then
      return
	fi
  fi
  if [ "$specificserialmode" != "" ]; then
    if [ "$serialmode" != "$specificserialmode" ]; then
      return
	fi
  fi
  
  schedstr="UNKNOWN"
  schedstr="UNKNOWN"  
  if [ "$schedmode" == "0" ]; then
    schedstr="coop-st$schedthreads"
  elif [ "$schedmode" == "1" ]; then 
    schedstr="prio-st$schedthreads"
  elif [ "$schedmode" == "2" ]; then 
    schedstr="DA-st$schedthreads"
  elif [ "$schedmode" == "3" ]; then 
    schedstr="fifo-st$schedthreads"
  fi;
  tpsigpolicydesc=""
  tpsignalpolicyopts=" -k $threadpoolpolicy "
  if [ "$threadpoolpolicy" == "1" ]; then
	tpsigpolicydesc="sigALL"
  else
	tpsigpolicydesc="sigSHR"
  fi;  
  viewmatopts=" -g $icmatpolicy -b $ocmatpolicy "
  viewmatdesc="ic$icmatpolicy-oc$ocmatpolicy"  
  threadpoolopts=""
  threadpooldescstr=""
  appthrdctxdescstr="ATCXok"
  if [ "$appthrdctx" == "0" ]; then
	appthrdctxdescstr="noATCX"  
  fi
  if [ "$threadpool" == "0" ]; then
	threadpoolopts=" -Q 2 $tpsignalpolicyopts "
	threadpooldescstr="MT-$tpsigpolicydesc-$appthrdctxdescstr"
  elif [ "$threadpool" == "1" ]; then
	threadpoolopts=" -J $tpsignalpolicyopts "
	threadpooldescstr="ST-$tpsigpolicydesc-$appthrdctxdescstr"
  else
	threadpoolopts=" -Q 1 -q $threadpool $tpsignalpolicyopts "
	threadpooldescstr="mtq$threadpool-$tpsigpolicydesc-$appthrdctxdescstr"
  fi  
  
  concstr="max=$concurrency"
  if [ "$concurrency" == "0" ]; then
    concstr="inf"
  fi;
  xflags="$forcesyncflags $xtraceflags"
  
  for iter in `seq 1 $iters`; do
    outfile=$outdir/$task-$platform-$target-C$concurrency-m$schedstr-tp$threadpooldescstr-$shaderplatform-$viewmatdesc-SM$serialmode-run$iter.txt
	shaderfileargs="-f $shaderfile"
	shaderopargs="-s $shaderop"
	if [ "$shaderfile" == "" ]; then
	  shaderfileargs=""
	fi	
	if [ "$shaderop" == "" ]; then
	  shaderopargs=""
	fi
    cmd="$BINDIR/$platform/$target/PTaskUnitTest.exe $xflags -M $appthrdctx -C $concurrency -m $schedmode $threadpoolopts -H $schedthreads -S $serialmode $viewmatopts -G -t $task $shaderfileargs $shaderopargs $args"
    if [ "$verbose" = "TRUE" ]; then
      echo "testCase($outdir, $platform, $target, C$concurrency, m$schedmode, tp$threadpooldesc, schdthrds$schedthreads, vm$viewmatdesc, S$serialmode, run$iter, $task)"
      echo "cmd: $cmd"
      echo "outfile: $outfile"
    fi
    echo "cmd: $cmd" > $outfile 
	echo "" >> $outfile
    $cmd >> $outfile 2>&1
	resstr="----"
	leakstr="----"
    if ! egrep "succeeded" $outfile > /dev/null; then
      if egrep "found no accelerators that can execute" $outfile > /dev/null; then
        resstr="***NO PLATFORM SUPPORT***"
        UNSUPPORTED=$[ $UNSUPPORTED+1 ]
      else
        resstr="***FAILED***..."
        FAILURES=$[ $FAILURES+1 ]
      fi
	else
	  resstr="OK..."
	fi
    if egrep "Detected memory leaks" $outfile > /dev/null; then
      leakstr="*LEAKS*"
	else
	  leakstr=""
    fi  
	printf "%24s:%7s%8s %6s %8s %18s %18s S%1s %4s: %26s %7s\n" $task $platform $target $concstr $schedstr $threadpooldescstr $viewmatdesc $serialmode $shaderplatform $resstr $leakstr		
	
    # if ! egrep "succeeded" $outfile > /dev/null; then
      # if egrep "found no accelerators that can execute" $outfile > /dev/null; then
        # printf "%24s:%7s%8s %6s %8s %14s %18s S%1s %4s:\t***NO PLATFORM SUPPORT***...\n" $task $platform $target $concstr $schedstr $threadpooldescstr $viewmatdesc $serialmode $shaderplatform
        # UNSUPPORTED=$[ $UNSUPPORTED+1 ]
      # else
        # printf "%24s:%7s%8s %6s %8s %14s %18s S%1s %4s:\t***FAILED***...\n" $task $platform $target $concstr $schedstr $threadpooldescstr $viewmatdesc $serialmode $shaderplatform
        # FAILURES=$[ $FAILURES+1 ]
      # fi
    # else
      # printf "%24s:%7s%8s %6s %8s %14s %18s S%1s %4s:\tOK...\n" $task $platform $target $concstr $schedstr $threadpooldescstr $viewmatdesc $serialmode $shaderplatform
      # SUCCEEDED=$[ $SUCCEEDED+1 ]
    # fi  
    if [ "$verbose" = "TRUE" ]; then
	  echo "current failure count: $FAILURES"
	  echo "current success count: $SUCCEEDED"
	  echo "current unsupported test case count: $UNSUPPORTED"
    fi	
    if [ "$showoutput" == "1" ]; then
       cat $outfile
    fi
  done
  sleep 2
}

memTestCase() {
  outdir=$1
  platform=$2
  target=$3
  concurrency=$4
  schedmode=$5
  threadpool=$6
  threadpoolpolicy=$7
  icmatpolicy=$8
  ocmatpolicy=$9  
  appthrdctx=${10}
  schedthreads=${11}
  iters=${12}
  task=${13}
  shaderfile=${14}
  shaderop=${15}
  args=${16}  
  
  # echo "memTestCase: iters=$iters, task=$task, sfile=$shaderfile, shaderop=$shaderop, args=$args"
  
  if [ "$specificappthrdctx" != "" ]; then 
	if [ "$specificappthrdctx" != "$appthrdctx" ]; then 
	  return
	fi  
  fi    
  if [ "$specificinviewmatpolicy" != "" ]; then 
	if [ "$specificinviewmatpolicy" != "$icmatpolicy" ]; then 
	  return
	fi  
  fi
  
  if [ "$specificoutviewmatpolicy" != "" ]; then 
	if [ "$specificoutviewmatpolicy" != "$ocmatpolicy" ]; then 
	  return
	fi  
  fi  
  
  if [ "$specificbattery" != "" ]; then 
	if [ "$specificbattery" != "mem" ]; then 
	  return
	fi  
  fi
  
  if [ "$excludebattery" != "" ]; then 
	if [ "$excludebattery" == "mem" ]; then 
	  return
	fi  
  fi  

  shaderplatform=""
  case "$shaderfile" in
    *.hlsl ) 
	  shaderplatform="DX"
	  ;;
	*.ptx )
	  shaderplatform="cu"
	  ;;
	*.cl )
	  shaderplatform="cl"
	  ;;
	*.dll )
	  shaderplatform="host"
	  ;;
	"" )
	  case "$task" in 
	    hlsl* )
	      shaderplatform="DX"
		  ;;
		cu* )
		  shaderplatform="cu"
		  ;;
		* )
	      echo "Unknown backend runtime for shader file=$shaderfile, task=$task"
	      return
	      ;;
      esac
	  ;;
	* )
	  echo "Unknown backend runtime for shader file=$shaderfile, task=$task"
	  return
	  ;;
  esac
  
  if [ "$shaderplatform" == "cl" ]; then 
	return
  fi  
  
  if [ "$specificruntime" != "" ]; then
    case "$specificruntime" in
	  "DirectX" )  
		if [[ $shaderfile != *.hlsl ]]; then
		   return
		fi
		;;
	  "CUDA" )    
		if [[ $shaderfile != *.ptx ]]; then
		   return
		fi
		;;
	  "OpenCL" )   
		if [[ $shaderfile != *.cl ]]; then
		   return
		fi
		;;
	  "host" )
		if [[ $shaderfile != *.dll ]]; then
		   return
		fi
		;;
	  *)           
	    echo "Unknown runtime restriction!"
	    exit 1
		;;
	esac			 
  fi
  if [ "$specifictask" != "" ]; then
    if [ "$task" != "$specifictask" ]; then
      return
	fi
  fi
  if [ "$specificplatform" != "" ]; then
    if [ "$platform" != "$specificplatform" ]; then
      return
	fi
  fi
  if [ "$specifictarget" != "" ]; then
    if [ "$target" != "$specifictarget" ]; then
      return
	fi
  fi
  if [ "$specificschedmode" != "" ]; then
    if [ "$schedmode" != "$specificschedmode" ]; then
      return
	fi
  fi
  if [ "$specificconc" != "" ]; then
    if [ "$concurrency" != "$specificconc" ]; then
      return
	fi
  fi
  
  schedstr="UNKNOWN"
  schedstr="UNKNOWN"
  if [ "$schedmode" == "0" ]; then
    schedstr="coop-st$schedthreads"
  elif [ "$schedmode" == "1" ]; then 
    schedstr="prio-st$schedthreads"
  elif [ "$schedmode" == "2" ]; then 
    schedstr="DA-st$schedthreads"
  elif [ "$schedmode" == "3" ]; then 
    schedstr="fifo-st$schedthreads"
  fi;
  tpsigpolicydesc=""
  tpsignalpolicyopts=" -k $threadpoolpolicy "
  if [ "$threadpoolpolicy" == "1" ]; then
	tpsigpolicydesc="sigALL"
  else
	tpsigpolicydesc="sigSHR"
  fi;  
  viewmatopts=" -g $icmatpolicy -b $ocmatpolicy "
  viewmatdesc="ic$icmatpolicy-oc$ocmatpolicy"  
  threadpoolopts=""
  threadpooldescstr=""
  appthrdctxdescstr="ATCXok"
  if [ "$appthrdctx" == "0" ]; then
	appthrdctxdescstr="noATCX"  
  fi  
  if [ "$threadpool" == "0" ]; then
	threadpoolopts=" -Q 2 "
	threadpooldescstr="MT-$tpsigpolicydesc-$appthrdctxdescstr"
  elif [ "$threadpool" == "1" ]; then
	threadpoolopts=" -J "
	threadpooldescstr="ST-$tpsigpolicydesc$appthrdctxdescstr"
  else
	threadpoolopts=" -Q 1 -q $threadpool "
	threadpooldescstr="mtq$threadpool-$tpsigpolicydesc-$appthrdctxdescstr"
  fi  
  concstr="max=$concurrency"
  if [ "$concurrency" == "0" ]; then
    concstr="inf"
  fi;
  xflags="$forcesyncflags $xtraceflags"
  
  # echo "memtestcase:"
  # echo "  outdir            = $outdir"
  # echo "  platform          = $platform"
  # echo "  target            = $target"
  # echo "  concurrency       = $concurrency"
  # echo "  schedmode         = $schedmode"
  # echo "  threadpool        = $threadpool"
  # echo "  threadpoolpolicy  = $threadpoolpolicy"
  # echo "  icmatpolicy       = $icmatpolicy"
  # echo "  ocmatpolicy       = $ocmatpolicy"
  # echo "  appthrdctx        = $appthrdctx"
  # echo "  schedthreads      = $schedthreads"
  # echo "  iters             = $iters"
  # echo "  task              = $task"
  # echo "  shaderfile        = $shaderfile"
  # echo "  shaderop          = $shaderop"
  # echo "  args              = $args"    
  
  for iter in `seq 1 $iters`; do
    outfile=$outdir/$task-$platform-$target-C$concurrency-m$schedmode-tp$threadpooldescstr-schdthrd$schedthreads-vm$viewmatdesc-$shaderplatform-run$iter.txt
	shaderfileargs="-f $shaderfile"
	shaderopargs="-s $shaderop"
	if [ "$shaderfile" == "" ]; then
	  shaderfileargs=""
	fi	
	if [ "$shaderop" == "" ]; then
	  shaderopargs=""
	fi
    cmd="$BINDIR/$platform/$target/PTaskUnitTest.exe $xflags -M $appthrdctx -C $concurrency -m $schedmode $threadpoolopts $viewmatopts -H $schedthreads -G -t $task $shaderfileargs $shaderopargs $args"
    if [ "$verbose" = "TRUE" ]; then
      echo "testCase($outdir, $platform, $target, C$concurrency, m$schedmode, tp$threadpoolopts, vm$viewmatdesc, schdthreads$schedthreads, run$iter, $task)"
      echo "cmd: $cmd"
      echo "outfile: $outfile"
    fi
	echo "cmd: $cmd" > $outfile 
    $cmd >> $outfile 2>&1
	resstr="XXXX"
	leakstr="YYYY"
    if egrep "FAIL" $outfile > /dev/null; then
      resstr="***FAILED***..."
      FAILURES=$[ $FAILURES+1 ]
    else
      resstr="OK..."
      SUCCEEDED=$[ $SUCCEEDED+1 ]
    fi  
    if egrep "Detected memory leaks" $outfile > /dev/null; then
      leakstr="*LEAKS*"
	else
	  leakstr=""
    fi  
	printf "%24s:%7s%8s %6s %8s %18s %18s %4s: %15s %7s\n" $task $platform $target $concstr $schedstr $threadpooldescstr $viewmatdesc $shaderplatform $resstr $leakstr	
    if [ "$verbose" = "TRUE" ]; then
	  echo "current failure count: $FAILURES"
	  echo "current success count: $SUCCEEDED"
    fi	
  done
  sleep 2
}

rttTestCase() {
  outdir=$1
  platform=$2
  target=$3
  concurrency=$4
  schedmode=$5
  threadpool=$6
  threadpoolpolicy=$7
  icmatpolicy=$8
  ocmatpolicy=$9    
  appthrdctx=${10}
  schedthreads=${11}
  iters=${12}
  task=${13}
  shaderfile=${14}
  shaderop=${15}
  args=${16}
  
  # echo "rttTestCase: iters=$iters, task=$task, sfile=$shaderfile, shaderop=$shaderop, args=$args"
  
  # echo "  outdir            = $outdir"
  # echo "  platform          = $platform"
  # echo "  target            = $target"
  # echo "  concurrency       = $concurrency"
  # echo "  schedmode         = $schedmode"
  # echo "  threadpool        = $threadpool"
  # echo "  threadpoolpolicy  = $threadpoolpolicy"
  # echo "  icmatpolicy       = $icmatpolicy"
  # echo "  ocmatpolicy       = $ocmatpolicy"
  # echo "  appthrdctx        = $appthrdctx"
  # echo "  schedthreads      = $schedthreads"
  # echo "  iters             = $iters"
  # echo "  task              = $task"
  # echo "  shaderfile        = $shaderfile"
  # echo "  shaderop          = $shaderop"
  # echo "  args              = $args"
  
  if [ "$specificappthrdctx" != "" ]; then 
	if [ "$specificappthrdctx" != "$appthrdctx" ]; then 
	  return
	fi  
  fi    
  if [ "$specificinviewmatpolicy" != "" ]; then 
	if [ "$specificinviewmatpolicy" != "$icmatpolicy" ]; then 
	  return
	fi  
  fi
  
  if [ "$specificoutviewmatpolicy" != "" ]; then 
	if [ "$specificoutviewmatpolicy" != "$ocmatpolicy" ]; then 
	  return
	fi  
  fi  
  
  if [ "$specificbattery" != "" ]; then 
	if [ "$specificbattery" != "rtt" ]; then 
	  return
	fi  
  fi
  
  if [ "$excludebattery" != "" ]; then 
	if [ "$excludebattery" == "rtt" ]; then 
	  return
	fi  
  fi  

  shaderplatform=""
  case "$shaderfile" in
    *.hlsl ) 
	  shaderplatform="DX"
	  ;;
	*.ptx )
	  shaderplatform="cu"
	  ;;
	*.cl )
	  shaderplatform="cl"
	  ;;
	*.dll )
	  shaderplatform="host"
	  ;;
	"" )
	  case "$task" in 
	    hlsl* )
	      shaderplatform="DX"
		  ;;
		cu* )
		  shaderplatform="cu"
		  ;;
		* )
	      echo "Unknown backend runtime for shader file=$shaderfile, task=$task"
	      return
	      ;;
      esac
	  ;;
	* )
	  echo "Unknown backend runtime for shader file=$shaderfile, task=$task"
	  return
	  ;;
  esac
  
  if [ "$specificruntime" != "" ]; then
    case "$specificruntime" in
	  "DirectX" )  
		if [[ $shaderfile != *.hlsl ]]; then
		   return
		fi
		;;
	  "CUDA" )    
		if [[ $shaderfile != *.ptx ]]; then
		   return
		fi
		;;
	  "OpenCL" )   
		if [[ $shaderfile != *.cl ]]; then
		   return
		fi
		;;
	  "host" )
		if [[ $shaderfile != *.dll ]]; then
		   return
		fi
		;;
	  *)           
	    echo "Unknown runtime restriction!"
	    exit 1
		;;
	esac			 
  fi
  if [ "$specifictask" != "" ]; then
    if [ "$task" != "$specifictask" ]; then
      return
	fi
  fi
  if [ "$specificplatform" != "" ]; then
    if [ "$platform" != "$specificplatform" ]; then
      return
	fi
  fi
  if [ "$specifictarget" != "" ]; then
    if [ "$target" != "$specifictarget" ]; then
      return
	fi
  fi
  if [ "$specificschedmode" != "" ]; then
    if [ "$schedmode" != "$specificschedmode" ]; then
      return
	fi
  fi
  if [ "$specificconc" != "" ]; then
    if [ "$concurrency" != "$specificconc" ]; then
      return
	fi
  fi
  
  schedstr="UNKNOWN"
  if [ "$schedmode" == "0" ]; then
    schedstr="coop-st$schedthreads"
  elif [ "$schedmode" == "1" ]; then 
    schedstr="prio-st$schedthreads"
  elif [ "$schedmode" == "2" ]; then 
    schedstr="DA-st$schedthreads"
  elif [ "$schedmode" == "3" ]; then 
    schedstr="fifo-st$schedthreads"
  fi;
  tpsigpolicydesc=""
  tpsignalpolicyopts=" -k $threadpoolpolicy "
  if [ "$threadpoolpolicy" == "1" ]; then
	tpsigpolicydesc="sigALL"
  else
	tpsigpolicydesc="sigSHR"
  fi;  
  viewmatopts=" -g $icmatpolicy -b $ocmatpolicy "
  viewmatdesc="ic$icmatpolicy-oc$ocmatpolicy"  
  threadpoolopts=""
  threadpooldescstr=""
  appthrdctxdescstr="ATCXok"
  if [ "$appthrdctx" == "0" ]; then
	appthrdctxdescstr="noATCX"  
  fi
  if [ "$threadpool" == "0" ]; then
	threadpoolopts=" -Q 2 "
	threadpooldescstr="MT-$tpsigpolicydesc-$appthrdctxdescstr"
  elif [ "$threadpool" == "1" ]; then
	threadpoolopts=" -J "
	threadpooldescstr="ST-$tpsigpolicydesc-$appthrdctxdescstr"
  else
	threadpoolopts=" -Q 1 -q $threadpool "
	threadpooldescstr="mtq$threadpool-$tpsigpolicydesc-$appthrdctxdescstr"
  fi  
  
  concstr="max=$concurrency"
  if [ "$concurrency" == "0" ]; then
    concstr="inf"
  fi;
  xflags="$forcesyncflags $xtraceflags" 
   
  for iter in `seq 1 $iters`; do
    outfile=$outdir/$task-$platform-$target-C$concurrency-m$schedmode-tp$threadpooldescstr-vm$viewmatdesc-schthrd$schedthreads-$shaderplatform-run$iter.txt
	shaderfileargs="-f $shaderfile"
	shaderopargs="-s $shaderop"
	if [ "$shaderfile" == "" ]; then
	  shaderfileargs=""
	fi	
	if [ "$shaderop" == "" ]; then
	  shaderopargs=""
	fi
    cmd="$BINDIR/$platform/$target/PTaskUnitTest.exe $xflags -M $appthrdctx -H $schedthreads $threadpoolopts $viewmatopts $tpsignalpolicyopts -C $concurrency -m $schedmode -G -t $task $shaderfileargs $shaderopargs $args"
    if [ "$verbose" = "TRUE" ]; then
      echo "rttTestCase($outdir, $platform, $target, C$concurrency, m$schedmode, tp$threadpooldescstr, vm$viewmatdesc, schthrd$schedthreads, run$iter, $task)"
      echo "cmd: $cmd"
      echo "outfile: $outfile"
    fi
    echo "cmd: $cmd" > $outfile 
    $cmd >> $outfile 2>&1
	resstr="XXXX"
	leakstr="YYYY"
    if egrep "FAIL" $outfile > /dev/null; then
      resstr="***FAILED***..."
      FAILURES=$[ $FAILURES+1 ]
    else
      resstr="OK..."
      SUCCEEDED=$[ $SUCCEEDED+1 ]
    fi  
    if egrep "Detected memory leaks" $outfile > /dev/null; then
      leakstr="*LEAKS*"
	else
	  leakstr=""
    fi  
	printf "%24s:%7s%8s %6s %8s %18s %18s S%4s %4s: %15s %7s\n" $task $platform $target $concstr $schedstr $threadpooldescstr $viewmatdesc $serialmode $shaderplatform $resstr $leakstr
    if [ "$verbose" = "TRUE" ]; then
	  echo "current failure count: $FAILURES"
	  echo "current success count: $SUCCEEDED"
    fi	
  done
  sleep 2
}

DATETIME="`date +%Y-%m-%d`"  
echo -e "\nPTask regression test $DATETIME"
echo -e "-----------------------------------"

# use the --rebuild switch to control
# whether the script rebuilds all the 
# ptask and related binaries. 
# by default, do not rebuild
if [ "$rebuild" == "1" ]; then
  echo "rebuild first..."
  cmd /c buildall.bat || { 
    echo "FAILED: build failed"  
    exit 1; 
  }
else
  echo "running on existing binaries (skipping rebuild)..."
fi  

if [ "$buildshaders" == "1" ]; then
  echo "rebuilding PTX files..."
  cmd /c buildPTX.bat || { 
    echo "FAILED: build PTX failed"  
    exit 1; 
  }
fi  

outdir=out-$DATETIME
if [ ! -e $outdir ]; then
  mkdir $outdir
else
  rm -f $outdir/*
fi

for Xplatform in "${allplatforms[@]}"; do
for Xtarget in "${alltargets[@]}"; do
for Xserialmode in 0 1 2; do
for Xschedmode in 2 1 3; do
for Xthreadpool in 4 1 0; do
for Xschedthreads in 2 1; do
for Xtpsigpolicy in 1 0; do
for Xicmatpolicy in DEMAND EAGER; do
for Xocmatpolicy in EAGER DEMAND; do
for Xconcurrency in 0 1; do
for Xappthrdctx in 0 1; do
# for the record:
# SCHEDMODE_COOPERATIVE = 0,
# SCHEDMODE_PRIORITY = 1,
# SCHEDMODE_DATADRIVEN = 2, (default)
# SCHEDMODE_FIFO = 3


bits=""
if [ "$platform" == "Win32" ]; then
  bits=32
fi

if [ "$specificthreadpool" != "" ]; then
  if [ "$specificthreadpool" != "$threadpool" ]; then
	return
  fi
fi

# memTestCase() {
  # outdir=$1
  # platform=$2
  # target=$3
  # concurrency=$4
  # schedmode=$5
  # threadpool=$6
  # threadpoolpolicy=$7
  # icmatpolicy=$8
  # ocmatpolicy=$9  
  # appthrdctx=${10}
  # schedthreads=${11}
  # iters=${12}
  # task=${13}
  # shaderfile=${14}
  # shaderop=${15}
  # args=${16}  
  
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest1 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest1 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest1 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest2 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest2 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest2 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest3 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest3 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest3 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest4 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest4 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest4 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest5 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest5 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest5 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest6 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest6 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest6 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest7 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest7 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest7 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest8 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest8 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
memTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest8 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest9 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest9 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest9 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest10 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest10 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest10 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest11 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest11 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest11 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest12 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest12 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest12 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest13 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest13 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest13 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest14 iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest14 iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
rttTestCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $ITERATIONS rttest14 iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS initializerchannels initializerchannels.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS initializerchannels initializerchannels.ptx op "-r 50000 -c 1 -n 1 -i 10" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS initializerchannelsbof initializerchannels.hlsl op2 "-r 50000 -c 1 -n 1 -i 10" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS initializerchannelsbof initializerchannels.ptx op2 "-r 50000 -c 1 -n 1 -i 10" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS select select.hlsl main "-D selectdata.txt"
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS pfxsum pfxsum.hlsl main ""
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS sort sort.hlsl sort "-r 512 -c 512 -n 1 -i 1"
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS matmul matrixmul.hlsl op "-r 64 -c 64 -n 1 -i 1"
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS matmulraw matrixmul_4ch.hlsl op "-r 16 -c 16 -n 1 -i 1"
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS clvecadd vectoradd.cl vadd "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS hlslfdtd "" "" "-d fdtdKernels -r 64 -c 64 -z 4 -n 4 -i 4" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS cufdtd "" "" "-d fdtdKernels$bits -r 64 -c 64 -z 4 -n 4 -i 4" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS cugroupby "" "" "-d $ROOTDIR\\PTaskUnitTest\\groupbyKernels$bits -r 1024 -c 32 -i 1"
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS channelpredication channelpredication$bits.ptx scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS channelpredication channelpredication.hlsl scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS channelpredication channelpredication.cl op "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS controlpropagation controlpropagation$bits.ptx scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS controlpropagation channelpredication.hlsl scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS controlpropagation controlpropagation.cl op "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS tee tee$bits.ptx scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS tee tee.hlsl scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS permanentblocks permanentblocks$bits.ptx scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS bufferdimsdesc bufferdimsdesc$bits.ptx scale "-r 100 -c 100 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS gatedports gatedports$bits.ptx scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS gatedports channelpredication.hlsl scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS gatedports gatedports.cl op "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS deferredports deferredports$bits.ptx scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS descportsout descportsout$bits.ptx scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS deferredports tee.hlsl scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS deferredports deferredports.cl op "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS switchports switchports$bits.ptx scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS switchports channelpredication.hlsl scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS switchports switchports.cl op "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS hostmetaports "$BINDIR\\$platform\\$Xtarget\\HostTasks.dll" htvadd "-r 20000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS hostmatmul "$BINDIR\\$platform\\$Xtarget\\HostTasks.dll" htmatmul "-r 128 -c 128 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS ptaskcublas "$BINDIR\\$platform\\$Xtarget\\HostTasks.dll" SGemmTrA "-r 64 -c 128 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS hostfuncmatmul "$BINDIR\\$platform\\$Xtarget\\HostTasks.dll" htmatmul "-r 128 -c 128 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS hostfunccublasmatmul "$BINDIR\\$platform\\$Xtarget\\HostTasks.dll" SGemmTrA "-r 64 -c 128 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS ptaskcublasnonsq "$BINDIR\\$platform\\$Xtarget\\HostTasks.dll" SGemm "-r 64 -c 128 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS ptaskcublassq "$BINDIR\\$platform\\$Xtarget\\HostTasks.dll" SGemmSq "-r 128 -c 128 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS ptaskcublasnoinout "$BINDIR\\$platform\\$Xtarget\\HostTasks.dll" SGemmSq "-r 128 -c 128 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS cuvecadd vectorAdd$bits.ptx VecAdd "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS dxinout inout.hlsl op "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS inout inout$bits.ptx scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS inout inout.cl op "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS initports initports$bits.ptx scale "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS initports initports.hlsl op "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS initports initports.cl op "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS generaliteration iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS generaliteration iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS generaliteration iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS simpleiteration iteration.hlsl op "-r 50000 -c 1 -n 1 -i 10" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS simpleiteration iteration$bits.ptx op "-r 50000 -c 1 -n 1 -i 10" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS simpleiteration iteration.cl op "-r 50000 -c 1 -n 1 -i 10" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS metaports metaports.hlsl op "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS metaports metaports$bits.ptx op "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS metaports metaports.cl op "-r 50000 -c 1 -n 1 -i 1" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS pipelinestresstest pipelinestresstest.hlsl op "-r 256 -c 256 -n 1 -i 4" 
testCase $outdir $Xplatform $Xtarget $Xconcurrency $Xschedmode $Xthreadpool $Xtpsigpolicy $Xicmatpolicy $Xocmatpolicy $Xappthrdctx $Xschedthreads $Xserialmode $ITERATIONS pipelinestresstestmulti pipelinestresstest.hlsl op "-r 256 -c 256 -n 1 -i 4" 

done
done
done 
done
done
done 
done
done
done 
done
done

echo successes: $SUCCEEDED
echo failures: $FAILURES
echo unsupported cases: $UNSUPPORTED
if [ "$FAILURES" != "0" ]; then 
  exit 1
fi

