///-------------------------------------------------------------------------------------------------
// file:	PortW.cpp
//
// summary:	Implements the port wrapper class
///-------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "PTaskManagedWrapper.h"

namespace Microsoft {
namespace Research {
namespace PTask {  

    Port::Port(
        ::PTask::Port* nativePort,
        DataTemplate^ managedTemplate
        )
    {
        m_nativePort = nativePort;
        m_boundToTask = false;
        m_disposed = false;
        m_managedTemplate = managedTemplate;
    }

    Port::~Port()
    {
        this->!Port();
        m_disposed = true;
    }

    Port::!Port()
    {
        // Only delete native port if has not yet been bound to a Task (as part of a call to Graph::AddTask).
        // After that, the runtime owns it and will delete it as part of Graph::Teardown.
        if (!m_boundToTask)
            delete m_nativePort;
        m_nativePort = NULL;
    }

    ::PTask::Port* 
    Port::GetNativePort()
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        return m_nativePort;
    }

    void
    Port::SetBoundToTask()
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        m_boundToTask = true;
    }

    void
    Port::SetSticky(
        bool b
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        GetNativePort()->SetSticky(b);
    }

    void 
    Port::SetStickyReleaseSignal(
        unsigned __int64 luiControlSignal
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        GetNativePort()->SetStickyReleaseSignal(luiControlSignal);
    }

    bool
    Port::IsSticky(
        void
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        return (GetNativePort()->IsSticky() != 0);
    }

    void
    Port::SetDestructive(
        bool b
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        GetNativePort()->SetDestructive(b);
    }

    bool
    Port::IsDestructive(
        void
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        return (GetNativePort()->IsDestructive() != 0);
    }

    void
    Port::SetMarshallable(
        bool b
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        GetNativePort()->SetMarshallable(b);
    }

    bool
    Port::IsMarshallable(
        void
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        return (GetNativePort()->IsMarshallable() != 0);
    }

	void 
	Port::SetUpstreamChannelPool(
		unsigned int uiPoolSize,
        bool bGrowable,
        unsigned int uiGrowIncrement
		)
	{
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        GetNativePort()->SetUpstreamChannelPool(uiPoolSize, bGrowable?1:0, uiGrowIncrement);
	}

	void 
	Port::SetUpstreamChannelPool(
		unsigned int uiPoolSize
		)
	{
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        GetNativePort()->SetUpstreamChannelPool(uiPoolSize, FALSE, 0);
	}
	
	void 
	Port::SetUpstreamChannelPool(
		void
		)
	{
		SetUpstreamChannelPool(0, FALSE, 0);
	}


    void
    Port::SetSuppressClones(
        bool b
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        GetNativePort()->SetSuppressClones(b);
    }

    bool
    Port::GetSuppressClones(
        void
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        return (GetNativePort()->GetSuppressClones() != 0);
    }

    void 
    Port::BindToEstimatorDimension(
        int nGeoDimension
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");
        GEOMETRYESTIMATORDIMENSION eGeoDimension = (GEOMETRYESTIMATORDIMENSION) nGeoDimension;
        GetNativePort()->BindToEstimatorDimension((::PTask::GEOMETRYESTIMATORDIMENSION)eGeoDimension);
    }

    int 
    Port::GetEstimatorDimensionBinding(
        void
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");
        return (int)GetNativePort()->GetEstimatorDimensionBinding();
    }


    DataTemplate^ 
    Port::GetTemplate(
        void
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        return m_managedTemplate;
    }

    void
    Port::SetTemplate(
        DataTemplate^ pTemplate
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");

        m_managedTemplate = pTemplate;
    }

    void            
    Port::BindDependentAccelerator(
        ACCELERATOR_CLASS maccClass, 
        int nIndex
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");
        
        ::PTask::ACCELERATOR_CLASS accClass = (::PTask::ACCELERATOR_CLASS) maccClass;
        GetNativePort()->BindDependentAccelerator(accClass, nIndex);
    }

    bool             
    Port::HasDependentAcceleratorBinding(
        VOID
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");
                
        return GetNativePort()->HasDependentAcceleratorBinding() != 0;
    }

    void 
    Port::SetTriggerPort(
        Graph^ pGraph
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");
                
        GetNativePort()->SetTriggerPort(pGraph->GetNativeGraph(), 1);
    }

    bool 
    Port::IsTriggerPort() {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");                
        return GetNativePort()->IsTriggerPort() != 0;
    }

    void            
    Port::ForceBlockPoolHint(
        __in UINT nPoolSize,
        __in UINT nStride,
        __in UINT nDataBytes,
        __in UINT nMetaBytes,
        __in UINT nTemplateBytes
        ) 
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");                
        GetNativePort()->ForceBlockPoolHint(nPoolSize, nStride, nDataBytes, nMetaBytes, nTemplateBytes);
    }

    bool 
    Port::SetScopeTerminus(
        __in unsigned __int64 luiTriggerCode,
        __in bool bIsTerminus
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");                
        return GetNativePort()->SetScopeTerminus(luiTriggerCode, bIsTerminus) != 0;
    }
    
    bool 
    Port::IsScopeTerminus(
        __in unsigned __int64 luiTriggerCode
        )
    {    
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");                
        return GetNativePort()->IsScopeTerminus(luiTriggerCode) != 0;
    }

    bool 
    Port::IsScopeTerminus(
        void
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");                
        return GetNativePort()->IsScopeTerminus() != 0;
    }

    void 
    Port::SetCanStream(
        bool bCanStream
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");                
        GetNativePort()->SetCanStream(bCanStream?1:0);
    }

    bool 
    Port::CanStream(
        void
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");                
        return GetNativePort()->CanStream() != 0;
    }

    void
    Port::SetMetaPortAllocationHint(
        __in UINT uiAllocationHint,
        __in bool bForceAllocationHint
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");                
        if(m_nativePort->GetPortType() == ::PTask::META_PORT) {
            ((::PTask::MetaPort*)m_nativePort)->SetAllocationHint(uiAllocationHint, bForceAllocationHint);
        } else if(m_nativePort->GetPortType() == ::PTask::OUTPUT_PORT) {
            ::PTask::OutputPort* pOPort = reinterpret_cast<::PTask::OutputPort*>(m_nativePort);
            ::PTask::MetaPort* pMetaPort = reinterpret_cast<::PTask::MetaPort*>(pOPort->GetAllocatorPort());
            if(pMetaPort != NULL) {
                pMetaPort->SetAllocationHint(uiAllocationHint, bForceAllocationHint);
            }
        }
    }

    bool 
    Port::IsDispatchDimensionsHint(
        void
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");                
        return m_nativePort->IsDispatchDimensionsHint() != 0;
    }
    
    void 
    Port::SetDispatchDimensionsHint(
        bool bIsHint
        )
    {
        if (m_disposed)
            throw gcnew ObjectDisposedException("Port already disposed");                
        m_nativePort->SetDispatchDimensionsHint(bIsHint?1:0);
    }


    //ACCELERATOR_CLASS 
    //Port::GetDependentAcceleratorClass(
    //    int nIndex
    //    )
    //{
    //    if (m_disposed)
    //        throw gcnew ObjectDisposedException("Port already disposed");
    //            
    //    ::PTask::ACCELERATOR_CLASS accClass = GetNativePort()->GetDependentAcceleratorClass(0);
    //    return (ACCELERATOR_CLASS) accClass;
    //}

    //int               
    //Port::GetDependentAcceleratorIndex(
    //    VOID
    //    )
    //{
    //    if (m_disposed)
    //        throw gcnew ObjectDisposedException("Port already disposed");
    //            
    //    return GetNativePort()->GetDependentAcceleratorIndex();
    //}

}}}
