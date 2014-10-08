///-------------------------------------------------------------------------------------------------
// file:	DataTemplate.cpp
//
// summary:	Implements the data template class
///-------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "PTaskManagedWrapper.h"

namespace Microsoft {
namespace Research {
namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   refactored from code by jcurrey by crossbac, 4/13/2012. </remarks>
    ///
    /// <param name="datablockTemplate">    [in,out] If non-null, the datablock template. </param>
    ///-------------------------------------------------------------------------------------------------

    DataTemplate::DataTemplate(
        ::PTask::DatablockTemplate* datablockTemplate
        )
    {
        m_nativeDatablockTemplate = datablockTemplate;
        m_nativeDatablockTemplate->AddRef();
        m_disposed = false;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   refactored from code by jcurrey by crossbac, 4/13/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    DataTemplate::~DataTemplate()
    {
        this->!DataTemplate();
        m_disposed = true;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Finaliser. </summary>
    ///
    /// <remarks>   refactored from code by jcurrey by crossbac, 4/13/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    DataTemplate::!DataTemplate()
    {
        // m_nativeDatablockTemplate->Release();
        // PTask runtime keeps a list of these, and will delete them,
        // so there is no need to explicitly release here.  
        m_nativeDatablockTemplate = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the datablock byte count. </summary>
    ///
    /// <remarks>   refactored from code by jcurrey by crossbac, 4/13/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   The datablock byte count. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned int
    DataTemplate::GetDatablockByteCount() 
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("DataTemplate already disposed");

        return m_nativeDatablockTemplate->GetDatablockByteCount(0);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the native datablock template. </summary>
    ///
    /// <remarks>   refactored from code by jcurrey by crossbac, 4/13/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   null if it fails, else the native datablock template. </returns>
    ///-------------------------------------------------------------------------------------------------

    ::PTask::DatablockTemplate* 
    DataTemplate::GetNativeDatablockTemplate()
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("DataTemplate already disposed");

        return m_nativeDatablockTemplate;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an initial value. </summary>
    ///
    /// <remarks>   refactored from code by jcurrey by crossbac, 4/13/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <param name="buffer">   [in,out] If non-null, the buffer. </param>
    /// <param name="count">    Number of. </param>
    ///-------------------------------------------------------------------------------------------------

	void 
    DataTemplate::SetInitialValue(
        byte *buffer, 
        unsigned int count,
        unsigned int records,
        bool bExplicitlyEmpty
        )
	{
		if (m_disposed)
			throw gcnew ObjectDisposedException("DataTemplate already disposed");
		
		m_nativeDatablockTemplate->SetInitialValue(buffer, count, records, (bExplicitlyEmpty?1:0));
	}

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the initial value. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   null if it fails, else the initial value. </returns>
    ///-------------------------------------------------------------------------------------------------

    byte * 
    DataTemplate::GetInitialValue(
        void
        )
    {
		if (m_disposed)
			throw gcnew ObjectDisposedException("DataTemplate already disposed");
		
		return (byte*)m_nativeDatablockTemplate->GetInitialValue();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the initial value size in bytes. </summary>
    ///
    /// <remarks>   crossbac, 4/13/2012. </remarks>
    ///
    /// <exception cref="ObjectDisposedException">  Thrown when a supplied object has been disposed. </exception>
    ///
    /// <returns>   null if it fails, else the initial value. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned int
    DataTemplate::GetInitialValueSize(
        void
        )
    {
		if (m_disposed)
			throw gcnew ObjectDisposedException("DataTemplate already disposed");
		
		return m_nativeDatablockTemplate->GetInitialValueSizeBytes();
    }
}}}

