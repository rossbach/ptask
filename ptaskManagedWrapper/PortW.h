///-------------------------------------------------------------------------------------------------
// file:	PortW.h
//
// summary:	Declares the port wrapper classes
///-------------------------------------------------------------------------------------------------

#pragma once
#pragma unmanaged
#include "ptaskapi.h"
#pragma managed

namespace Microsoft {
namespace Research {
namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Port. </summary>
    ///
    /// <remarks>   refactored from code by jcurrey by crossbac, 4/13/2012. 
    ///             Follows the C++/CLI IDisposable pattern described at 
    ///             http://msdn.microsoft.com/en-us/library/system.idisposable.aspx
    /// 			</remarks>
    ///-------------------------------------------------------------------------------------------------

    public ref class Port
    {
    public:
        enum class PORTTYPE {
            INPUT_PORT,
            OUTPUT_PORT,
            STICKY_PORT,
            META_PORT,
            INITIALIZER_PORT
        };
        enum class GEOMETRYESTIMATORDIMENSION {
            GD_NONE=-1,
            GD_X = 0,
            GD_Y = 1,
            GD_Z = 2
        };

        Port(::PTask::Port* nativePort, DataTemplate^ pTemplate);
        ~Port(); // IDisposable
        !Port(); // finalizer

        ::PTask::Port* GetNativePort();

        void SetBoundToTask();
        void SetSticky(bool b);
        void SetStickyReleaseSignal(unsigned __int64 luiControlSignal);
        bool IsSticky();
        void SetDestructive(bool b);
        bool IsDestructive();
        void SetMarshallable(bool b);
        bool IsMarshallable();
        void SetSuppressClones(bool b);
        bool GetSuppressClones();
        DataTemplate^ GetTemplate();
        void BindToEstimatorDimension(int nGeoDimension);
        int GetEstimatorDimensionBinding();
        void BindDependentAccelerator(ACCELERATOR_CLASS accClass, int nIndex);        
        bool HasDependentAcceleratorBinding();
        void SetTriggerPort(Graph^ pGraph);
        bool IsTriggerPort();
        bool SetScopeTerminus(unsigned __int64 luiTriggerCode, bool bIsTerminus);
        bool IsScopeTerminus(unsigned __int64 luiTriggerCode);
        bool IsScopeTerminus();
        void SetCanStream(bool bCanStream);
        bool CanStream();
        bool IsDispatchDimensionsHint();
        void SetDispatchDimensionsHint(bool bIsHint);
		void SetUpstreamChannelPool(unsigned int uiPoolSize, bool bGrowable, unsigned int uiGrowIncrement);
		void SetUpstreamChannelPool(unsigned int uiPoolSize);
		void SetUpstreamChannelPool();

        void            
        ForceBlockPoolHint(
            __in UINT nPoolSize,
            __in UINT nStride,
            __in UINT nDataBytes,
            __in UINT nMetaBytes,
            __in UINT nTemplateBytes
            );

        void
        SetMetaPortAllocationHint(
            __in UINT uiAllocationHint,
            __in bool bForceAllocationHint
            );

    private:
        ::PTask::Port*  m_nativePort;
        DataTemplate^   m_managedTemplate;
        bool            m_boundToTask;
        bool            m_disposed;

        void SetTemplate(DataTemplate^ pTemplate);
    };

}}}
