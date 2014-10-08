///-------------------------------------------------------------------------------------------------
// file:	TaskW.cpp
//
// summary:	Implements the task wrapper class
///-------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "PTaskManagedWrapper.h"

namespace Microsoft {
namespace Research {
namespace PTask {

    Task::Task(::PTask::Task* nativeTask)
    {
        m_nativeTask = nativeTask;
        m_disposed = false;
    }

    Task::~Task()
    {
        this->!Task();
        m_disposed = true;
    }

    Task::!Task()
    {
        // Do NOT delete m_nativeTask here, since the native Task instances can only be obtained
        // from a call to (native) Graph::AddTask, and the Graph will delete all Tasks that
        // it owns when Graph::Teardown is called.
    }

    ::PTask::Task* 
        Task::GetNativeTask()
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Task already disposed");

        return m_nativeTask;
    }

    void 
        Task::SetComputeGeometry(Int32 tgx, Int32 tgy, Int32 tgz)
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Task already disposed");

        m_nativeTask->SetComputeGeometry(tgx, tgy, tgz);
    }

	void 
	Task::SetBlockAndGridSize(DIM3^ gridSize, DIM3^ blockSize)
	{
		if (m_disposed)
            throw gcnew ObjectDisposedException("Task already disposed");
		::PTask::PTASKDIM3 nativeGridSize(gridSize->x, gridSize->y, gridSize->z);
		::PTask::PTASKDIM3 nativeBlockSize(blockSize->x, blockSize->y, blockSize->z);

		m_nativeTask->SetBlockAndGridSize(nativeGridSize, nativeBlockSize);
	}

	///-------------------------------------------------------------------------------------------------
	/// <summary>	Sets a canonical geometry estimator. </summary>
	///
	/// <remarks>	Crossbac, 10/1/2012. </remarks>
	///
	/// <exception cref="ObjectDisposedException">	Thrown when a supplied object has been disposed. </exception>
	///
	/// <param name="t">			  	The GEOMETRYESTIMATORTYPE to process. </param>
	/// <param name="nElemsPerThread">	The elems per thread. </param>
	/// <param name="nGroupSizeX">	  	The group size x coordinate. </param>
	/// <param name="nGroupSizeY">	  	The group size y coordinate. </param>
	/// <param name="nGroupSizeZ">	  	The group size z coordinate. </param>
	///-------------------------------------------------------------------------------------------------

	void 
	Task::SetCanonicalGeometryEstimator(
		GEOMETRYESTIMATORTYPE t, 
		int nElemsPerThread,
		int nGroupSizeX,
		int nGroupSizeY,
		int nGroupSizeZ
		)
	{
        if (m_disposed)
            throw gcnew ObjectDisposedException("Task already disposed");
		::PTask::GEOMETRYESTIMATORTYPE nativeType = static_cast<::PTask::GEOMETRYESTIMATORTYPE>(t);
		m_nativeTask->SetCanonicalGeometryEstimator(nativeType, 
												    nElemsPerThread,
													nGroupSizeX,
													nGroupSizeY,
													nGroupSizeZ);
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>	Gets a canonical geometry estimator. </summary>
    ///
    /// <remarks>	Crossbac, 10/1/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">	Thrown when a supplied object has been disposed. </exception>
    ///
    /// <param name="nElemsPerThread">	[in,out] If non-null, the elems per thread. </param>
    /// <param name="nGroupSizeX">	  	[in,out] If non-null, the group size x coordinate. </param>
    /// <param name="nGroupSizeY">	  	[in,out] If non-null, the group size y coordinate. </param>
    /// <param name="nGroupSizeZ">	  	[in,out] If non-null, the group size z coordinate. </param>
    ///
    /// <returns>	The canonical geometry estimator. </returns>
    ///-------------------------------------------------------------------------------------------------

    Task::GEOMETRYESTIMATORTYPE 
	Task::GetCanonicalGeometryEstimator(
		int^ nElemsPerThread,
		int^ nGroupSizeX,
		int^ nGroupSizeY,
		int^ nGroupSizeZ
		)
	{
        if (m_disposed)
           throw gcnew ObjectDisposedException("Task already disposed");
        int pnElemsPerThread = 0;
		int pnX = 0;
		int pnY = 0;
		int pnZ = 0;
		::PTask::GEOMETRYESTIMATORTYPE nativeType = 
			m_nativeTask->GetCanonicalGeometryEstimator(&pnElemsPerThread,
														&pnX,
														&pnY,
														&pnZ);
        *nElemsPerThread = pnElemsPerThread;
        *nGroupSizeX     = pnX;
		*nGroupSizeY     = pnY;
		*nGroupSizeZ     = pnZ;
		return static_cast<GEOMETRYESTIMATORTYPE>(nativeType);
	}

    void
    Task::BindDependentAcceleratorClass(
        ACCELERATOR_CLASS cls,
        int nInstancesRequired
        )
    {
		BindDependentAcceleratorClass(cls, nInstancesRequired, FALSE);
    }

    void
    Task::BindDependentAcceleratorClass(
        ACCELERATOR_CLASS cls,
        int nInstancesRequired,
        BOOL bRequestPSObjects
        )
    {
        if (m_disposed) throw gcnew ObjectDisposedException("Task already disposed");
        ::PTask::ACCELERATOR_CLASS pcls = static_cast<::PTask::ACCELERATOR_CLASS>(cls);
		m_nativeTask->BindDependentAcceleratorClass(pcls, nInstancesRequired, bRequestPSObjects);
    }

    void
    Task::ReleaseStickyBlocks(
        void
        )
    {
        if (m_disposed) throw gcnew ObjectDisposedException("Task already disposed");
		m_nativeTask->ReleaseStickyBlocks();
    }

    void
    Task::SetUserCodeAllocatesMemory(
        void
        )
    {
        if (m_disposed) throw gcnew ObjectDisposedException("Task already disposed");
		m_nativeTask->SetUserCodeAllocatesMemory(1);
    }

}}}
