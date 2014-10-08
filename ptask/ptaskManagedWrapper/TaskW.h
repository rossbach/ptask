///-------------------------------------------------------------------------------------------------
// file:	TaskW.h
//
// summary:	Declares the task wrapper class
///-------------------------------------------------------------------------------------------------

#pragma once
#pragma unmanaged
#include "ptaskapi.h"
#pragma managed

using namespace System;
using namespace System::Runtime::InteropServices; // For marshalling helpers

// Follows the C++/CLI IDisposable pattern described at 
// http://msdn.microsoft.com/en-us/library/system.idisposable.aspx

namespace Microsoft {
namespace Research {
namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   3-D dimensions object </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    public ref class DIM3 {
    public:
        int x;
        int y;
        int z;

        DIM3(int xx, int yy, int zz) 
        {
            x = xx;
            y = yy;
            z = zz;
        }
    };

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Wrapper class for PTask native Task class. </summary>
    ///
    /// <remarks>   refactored from code by jcurrey by crossbac, 4/13/2012. 
    ///             Follows the C++/CLI IDisposable pattern described at 
    ///             http://msdn.microsoft.com/en-us/library/system.idisposable.aspx
    /// 			</remarks>
    ///-------------------------------------------------------------------------------------------------

    public ref class Task
    {
    public:

        enum class GEOMETRYESTIMATORTYPE {

            /// <summary>   No size estimator function has been provided. 
            /// 			</summary>
            NO_SIZE_ESTIMATOR = 0,

            /// <summary>   Estimate the geometry based on the size of the
            /// 			datablock bound to the first port. 
            /// 			</summary>
            BASIC_INPUT_SIZE_ESTIMATOR = 1, //

            /// <summary>   Estimate the geometry based on the max of the
            /// 			record counts over all input datablocks.
            /// 			</summary>
            MAX_INPUT_SIZE_ESTIMATOR = 2, 

            /// <summary>   Estimate the geometry based on the max of the
            /// 			record counts over all output datablocks.
            /// 			</summary>
            MAX_OUTPUT_SIZE_ESTIMATOR = 3,

            /// <summary>   Ports are bound to a particular dimension
            /// 			of the iteration space. This estimator
            /// 			looks for explicit port bindings and assembles
            /// 			the iteration space accordingly. </summary>
            EXPLICIT_DIMENSION_ESTIMATOR = 4,

            /// <summary>   The user commits to provide a callback to
            /// 			estimate the dispatch dimensions. 
            /// 			</summary>
            USER_DEFINED_ESTIMATOR = 5

            // ....

        };


        Task(::PTask::Task* nativeTask);
        ~Task(); // IDisposable
        !Task(); // finalizer
        ::PTask::Task* GetNativeTask();
        void SetComputeGeometry(Int32 tgx, Int32 tgy, Int32 tgz);
        void SetBlockAndGridSize(DIM3^ gridSize, DIM3^ blockSize);
        void SetCanonicalGeometryEstimator(GEOMETRYESTIMATORTYPE t, int nElemsPerThread, int nX, int nY, int nZ);
        GEOMETRYESTIMATORTYPE GetCanonicalGeometryEstimator(int ^ nElemsPerThread, int ^ pnX, int ^ pnY, int ^ pnZ);
        void BindDependentAcceleratorClass(ACCELERATOR_CLASS cls, int nInstancesRequired);
        void BindDependentAcceleratorClass(ACCELERATOR_CLASS cls, int nInstancesRequired, BOOL bRequestPSObjects);
        void ReleaseStickyBlocks();
        void SetUserCodeAllocatesMemory();

    private:
        ::PTask::Task* m_nativeTask;
        bool           m_disposed;
    };

}}}
