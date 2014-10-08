//--------------------------------------------------------------------------------------
// File: datablocktemplate.cpp
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include <stdio.h>
#include <crtdbg.h>
#include "datablocktemplate.h"
#include <string.h>
#include <assert.h>

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Default initialize. </summary>
    ///
    /// <remarks>   crossbac, 7/9/2012. </remarks>
    ///
    /// <param name="lpszTemplateName"> [in,out] If non-null, name of the lspz template. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    DatablockTemplate::DefaultInitialize(
        char * lpszTemplateName
        )
    {
        m_nInitialRecordCount = 0;
        m_bExplicitlyEmptyInitialValue = FALSE;
        m_bByteAddressable = false;
        m_bRecordStream = false;
        m_bScalarParameter = false;
        m_bParameterBaseType = PTPARM_NONE;
        size_t len = strlen(lpszTemplateName)+1;
        m_lpszTemplateName = new char[len];
        strcpy_s(m_lpszTemplateName, len, lpszTemplateName);
        m_lpvInitialValue = NULL;
        m_cbInitialValue = 0;
        m_bMemsetCheckComplete = FALSE;
        m_bMemsettableInitialValue = FALSE;
        m_bMemsetInitialValueStride = 0;
        m_bMemsettableInitialValueD8 = FALSE;
        m_ucMemsettableInitialValueD8 = 0;
        m_vChannelDimensions[DBMETADATA_IDX].Initialize(1,1,1,1,1);
        m_vChannelDimensions[DBTEMPLATE_IDX].Initialize(1,1,1,1,1);
        m_pApplicationContextCallback = NULL;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="lpszTemplateName">     [in,out] If non-null, name of the template. </param>
    /// <param name="uiElementStride">      The element stride in bytes. </param>
    /// <param name="uiElementsX">          Number of elements in X dimension. </param>
    /// <param name="uiElementsY">          Number of elements in Y dimension. </param>
    /// <param name="uiElementsZ">          Number of elements in Z dimension. </param>
    /// <param name="bIsRecordStream">      true if this object is record stream. </param>
    /// <param name="bIsByteAddressable">   true if this object is byte addressable. </param>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate::DatablockTemplate(
        char * lpszTemplateName, 
        unsigned int uiElementStride, 
        unsigned int uiElementsX, 
        unsigned int uiElementsY, 
        unsigned int uiElementsZ,
        bool bIsRecordStream,
        bool bIsByteAddressable
        )
    {
        DefaultInitialize(lpszTemplateName);
        m_vChannelDimensions[DBDATA_IDX].Initialize(uiElementsX, uiElementsY, uiElementsZ, uiElementStride);
        m_bByteAddressable = bIsByteAddressable;
        m_bRecordStream = bIsRecordStream;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="lpszTemplateName">     [in] If non-null, name of the template. </param>
    /// <param name="uiElementStride">      [in] The element stride in bytes. </param>
    /// <param name="uiElementsX">          [in] Number of elements in X dimension. </param>
    /// <param name="uiElementsY">          [in] Number of elements in Y dimension. </param>
    /// <param name="uiElementsZ">          [in] Number of elements in Z dimension. </param>
    /// <param name="uiPitch">              [in] The row pitch. </param>
    /// <param name="bIsRecordStream">      [in] true if this object is record stream. </param>
    /// <param name="bIsByteAddressable">   [in] true if this object is byte addressable. </param>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate::DatablockTemplate(
        __in char *       lpszTemplateName, 
        __in unsigned int uiElementStride, 
        __in unsigned int uiElementsX, 
        __in unsigned int uiElementsY, 
        __in unsigned int uiElementsZ,
        __in unsigned int uiPitch,
        __in bool         bIsRecordStream,
        __in bool         bIsByteAddressable
        )
    {
        DefaultInitialize(lpszTemplateName);
        m_bByteAddressable = bIsByteAddressable;
        m_bRecordStream = bIsRecordStream;
        m_vChannelDimensions[DBDATA_IDX].Initialize(uiElementsX, 
                                                    uiElementsY, 
                                                    uiElementsZ, 
                                                    uiElementStride, 
                                                    uiPitch);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="lpszTemplateName">     [in] If non-null, name of the template. </param>
    /// <param name="pBufferDims">          [in] The element stride in bytes. </param>
    /// <param name="uiNumBufferDims">      [in] Number of elements in X dimension. </param>
    /// <param name="bIsRecordStream">      [in] true if this object is record stream. </param>
    /// <param name="bIsByteAddressable">   [in] true if this object is byte addressable. </param>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate::DatablockTemplate(
        __in char *             lpszTemplateName, 
        __in BUFFERDIMENSIONS * pBufferDims, 
        __in unsigned int       uiNumBufferDims, 
        __in bool               bIsRecordStream,
        __in bool               bIsByteAddressable
        )
    {
        DefaultInitialize(lpszTemplateName);
        m_bByteAddressable = bIsByteAddressable;
        m_bRecordStream = bIsRecordStream;
        for(UINT i=0; i<uiNumBufferDims && i<NUM_DATABLOCK_CHANNELS; i++) {
            m_vChannelDimensions[i].Initialize(pBufferDims[i]);
        }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="lpszType"> [in,out] If non-null, the type. </param>
    /// <param name="uiStride"> The stride. </param>
    /// <param name="pttype">   The pttype. </param>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate::DatablockTemplate(
        char * lpszType, 
        unsigned int uiStride, 
        PTASK_PARM_TYPE pttype
        ) 
    {
        DefaultInitialize(lpszType);
        m_bByteAddressable = false;
        m_bRecordStream = false;
        m_bScalarParameter = true;
        m_bParameterBaseType = pttype;
        m_vChannelDimensions[DBDATA_IDX].Initialize(1, 1, 1, uiStride);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    DatablockTemplate::~DatablockTemplate() {
        // printf("deleting DatablockTemplate %s\n", m_lpszType);
        // assert(m_uiRefCount == 0);
        if(m_lpszTemplateName) delete [] m_lpszTemplateName;
        if(m_lpvInitialValue) delete [] m_lpvInitialValue;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a datablock byte count. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The datablock byte count. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned int 
    DatablockTemplate::GetDatablockByteCount(
        UINT nChannelIndex
        ) {

        if(nChannelIndex >= NUM_DATABLOCK_CHANNELS) {
            assert(false);
            return 0;
        }

        return m_vChannelDimensions[nChannelIndex].AllocationSizeBytes();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' is variable dimensioned. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   true if variable dimensioned, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool 
    DatablockTemplate::IsVariableDimensioned(
        VOID
        ) {
        return m_vChannelDimensions[DBDATA_IDX].AllocationSizeBytes() == 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the stride. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   The stride. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned int 
    DatablockTemplate::GetStride(
        UINT nChannelIndex
        ) 
    {
        return m_vChannelDimensions[nChannelIndex].cbElementStride;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the number of elements in X. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   The stride. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned int 
    DatablockTemplate::GetXElementCount(
        UINT uiChannelIndex
        )
    {
        return m_vChannelDimensions[uiChannelIndex].uiXElements;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the number of elements in Y. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   The stride. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned int 
    DatablockTemplate::GetYElementCount(
        UINT uiChannelIndex
        )
    {
        return m_vChannelDimensions[uiChannelIndex].uiYElements;
    }


    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the number of elements in Z. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   The stride. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned int 
    DatablockTemplate::GetZElementCount(
        UINT uiChannelIndex
        )
    {
        return m_vChannelDimensions[uiChannelIndex].uiZElements;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the number of elements in Z. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   The stride. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned int 
    DatablockTemplate::GetTotalElementCount(
        UINT uiChannelIndex
        )
    {
        return 
            m_vChannelDimensions[uiChannelIndex].uiXElements *
            m_vChannelDimensions[uiChannelIndex].uiZElements *
            m_vChannelDimensions[uiChannelIndex].uiZElements;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the number of elements in Z. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   The stride. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned int 
    DatablockTemplate::GetDimensionElementCount(
        UINT uiDim, 
        UINT uiChannelIndex
        )
    {
        switch(uiDim) {
        case XDIM: return m_vChannelDimensions[uiChannelIndex].uiXElements;
        case YDIM: return m_vChannelDimensions[uiChannelIndex].uiZElements;
        case ZDIM: return m_vChannelDimensions[uiChannelIndex].uiZElements;
        }
        assert(FALSE);
        return 0;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the pitch. </summary>
    ///
    /// <remarks>   Crossbac, 2/18/2013. </remarks>
    ///
    /// <param name="uiChannelIndex">   (optional) zero-based index of the channel. </param>
    ///
    /// <returns>   The pitch. </returns>
    ///-------------------------------------------------------------------------------------------------

    unsigned int 
    DatablockTemplate::GetPitch(
        UINT uiChannelIndex
        )
    {
        return m_vChannelDimensions[uiChannelIndex].cbPitch;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets buffer dimensions. </summary>
    ///
    /// <remarks>   Crossbac, 2/18/2013. </remarks>
    ///
    /// <param name="uiChannelIndex">   (optional) zero-based index of the channel. </param>
    ///
    /// <returns>   The buffer dimensions. </returns>
    ///-------------------------------------------------------------------------------------------------

    BUFFERDIMENSIONS 
    DatablockTemplate::GetBufferDimensions(
        UINT uiChannelIndex
        )
    {
        return m_vChannelDimensions[uiChannelIndex];
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets buffer dimensions. </summary>
    ///
    /// <remarks>   Crossbac, 2/18/2013. </remarks>
    ///
    /// <param name="uiChannelIndex">   (optional) zero-based index of the channel. </param>
    ///
    /// <returns>   The buffer dimensions. </returns>
    ///-------------------------------------------------------------------------------------------------

    void 
    DatablockTemplate::SetBufferDimensions(
        BUFFERDIMENSIONS &dims, 
        UINT uiChannelIndex
        )
    {
        m_vChannelDimensions[uiChannelIndex].Initialize(dims);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if '' is byte addressable. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if byte addressable, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    bool 
    DatablockTemplate::IsByteAddressable(
        VOID
        )
    {
        return m_bByteAddressable;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Describes record stream. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    DatablockTemplate::DescribesRecordStream(
        VOID
        )
    {
        return m_bRecordStream;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Describes scalar parameter. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    DatablockTemplate::DescribesScalarParameter(
        VOID
        )
    {
        return m_bScalarParameter;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets a byte addressable. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="b">    true to b. </param>
    ///-------------------------------------------------------------------------------------------------

    void 
    DatablockTemplate::SetByteAddressable(
        bool b
        )
    {
        m_bByteAddressable = b;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets a parameter base type. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name=""> The. </param>
    ///
    /// <returns>   The parameter base type. </returns>
    ///-------------------------------------------------------------------------------------------------

    PTASK_PARM_TYPE
    DatablockTemplate::GetParameterBaseType(
        VOID
        )
    {
        return m_bParameterBaseType;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Sets an initial value. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="lpvInitData">  [in,out] If non-null, information describing the lpv initialise. </param>
    /// <param name="cbData">       The data. </param>
    ///-------------------------------------------------------------------------------------------------

    void
    DatablockTemplate::SetInitialValue(
        void * lpvInitData, 
        UINT cbData,
        UINT nRecordCount,
        BOOL bExplicitlyEmpty
        )
    {
        assert(lpvInitData != NULL || bExplicitlyEmpty);
        assert(cbData != 0 || bExplicitlyEmpty);
        assert(nRecordCount != 0 || bExplicitlyEmpty);
        m_lpvInitialValue = (cbData > 0 && lpvInitData) ? (void*) new unsigned char[cbData] : NULL;
        m_cbInitialValue = cbData;
        m_nInitialRecordCount = nRecordCount;
        m_bExplicitlyEmptyInitialValue = bExplicitlyEmpty;
        if(cbData > 0 && lpvInitData) {
            memcpy(m_lpvInitialValue, lpvInitData, cbData);
        }
        assert(cbData <= this->GetDatablockByteCount(DBDATA_IDX));
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has an initial value. </summary>
    ///
    /// <remarks>   crossbac, 6/15/2012. </remarks>
    ///
    /// <returns>   true if initial value, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    DatablockTemplate::HasInitialValue(
        VOID
        )
    {
        return ((m_lpvInitialValue != NULL && m_cbInitialValue > 0) || m_bExplicitlyEmptyInitialValue);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Queries if the initial value for this template is empty. </summary>
    ///
    /// <remarks>   crossbac, 6/15/2012. </remarks>
    ///
    /// <returns>   true if an initial value is empty, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    DatablockTemplate::IsInitialValueEmpty(
        VOID
        )
    {
        return m_bExplicitlyEmptyInitialValue;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the initial value size. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   The initial value size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    DatablockTemplate::GetInitialValueSizeBytes(
        VOID
        ) 
    { 
        return m_cbInitialValue; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the initial value element count. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   The initial value size. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    DatablockTemplate::GetInitialValueElements(
        VOID
        ) 
    { 
        return m_nInitialRecordCount;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the initial value. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the initial value. </returns>
    ///-------------------------------------------------------------------------------------------------

    const void * 
    DatablockTemplate::GetInitialValue(
        VOID
        ) 
    { 
        return m_lpvInitialValue; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the type. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   null if it fails, else the type. </returns>
    ///-------------------------------------------------------------------------------------------------

    char * 
    DatablockTemplate::GetTemplateName(
        VOID
        ) 
    { 
        return m_lpszTemplateName; 
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Set the application context callback function associated with this 
    ///             datablock template. </summary>
    ///
    /// <remarks>   jcurrey, 5/1/2014. </remarks>
    ///
    /// <param name="pCallback"> [in] The callback function to associate with this template. </param>
    ///-------------------------------------------------------------------------------------------------
 
    void
    DatablockTemplate::SetApplicationContextCallback(LPFNAPPLICATIONCONTEXTCALLBACK pCallback)
    {
        m_pApplicationContextCallback = pCallback;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Get the application context callback function associated with this 
    ///             datablock template. </summary>
    ///
    /// <remarks>   jcurrey, 5/1/2014. </remarks>
    ///
    /// <returns>   The callback function associated with this template. </param>
    ///-------------------------------------------------------------------------------------------------

    LPFNAPPLICATIONCONTEXTCALLBACK
    DatablockTemplate::GetApplicationContextCallback()
    {
        return m_pApplicationContextCallback;
    }
    
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has an initial value that can be recreated easily
    /// 			using a memset (rather than a memcpy). The object is memsettable if it has
    /// 			an initial value whose size is less than 4 bytes, or whose initial value
    /// 			is identical for all elements when the value is interpreted as either a 4-byte
    /// 			int or an unsigned char. 
    /// 			</summary>
    ///
    /// <remarks>   crossbac, 6/15/2012. </remarks>
    ///
    /// <returns>   true if initial value, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    DatablockTemplate::IsInitialValueMemsettableD8(
        VOID
        )
    {
        return IsInitialValueMemsettable(1);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Query if this object has an initial value that can be recreated easily
    /// 			using a memset (rather than a memcpy). The object is memsettable if it has
    /// 			an initial value whose size is less than 4 bytes, or whose initial value
    /// 			is identical for all elements when the value is interpreted as either a 4-byte
    /// 			int or an unsigned char. 
    /// 			</summary>
    ///
    /// <remarks>   crossbac, 6/15/2012. </remarks>
    ///
    /// <returns>   true if initial value, false if not. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL 
    DatablockTemplate::IsInitialValueMemsettable(
        UINT uiPrimitiveSize
        )
    {
        if(m_bMemsetCheckComplete) {
            if(uiPrimitiveSize == 1) return m_bMemsettableInitialValueD8;
            return m_bMemsettableInitialValue;
        }

        Lock();
        m_bMemsettableInitialValue = FALSE;
        m_bMemsettableInitialValueD8 = FALSE;
        if(HasInitialValue()) {
          
            if(m_cbInitialValue <= sizeof(UINT)) {
                m_bMemsettableInitialValue = TRUE;
                m_bMemsetInitialValueStride = m_cbInitialValue == sizeof(UINT) ? sizeof(UINT) : 1;
                UINT8 * pValue = (UINT8*)m_lpvInitialValue;
                if(m_bMemsetInitialValueStride == 1) {
                    m_bMemsettableInitialValueD8 = TRUE;
                } else {
                    m_bMemsettableInitialValueD8 = TRUE;
                    for(UINT i=0; i<m_cbInitialValue-1; i++)
                        m_bMemsettableInitialValueD8 &= (pValue[i]==pValue[i+1]);
                }
            } else {
                switch(GetStride()) {
                case 1:
                    {
                        BYTE * pucValue = reinterpret_cast<BYTE*>(m_lpvInitialValue);
                        BYTE ucFirst = *pucValue;
                        m_bMemsettableInitialValue = TRUE;
                        m_bMemsetInitialValueStride = 1;
                        for(UINT i=1; i<m_cbInitialValue; i++) {
                            m_bMemsettableInitialValue &= (ucFirst == pucValue[i]);
                            if(!m_bMemsettableInitialValue)
                                break;
                        }
                    }
                default:
                    if(m_cbInitialValue % sizeof(UINT) == 0) {
                        UINT * puiValue = reinterpret_cast<UINT*>(m_lpvInitialValue);
                        UINT uiFirst = *puiValue;
                        m_bMemsetInitialValueStride = sizeof(UINT);
                        m_bMemsettableInitialValue = TRUE;
                        for(UINT i=1; i<m_cbInitialValue/sizeof(UINT); i++) {
                            m_bMemsettableInitialValue &= (uiFirst == puiValue[i]);
                            if(!m_bMemsettableInitialValue)
                                break;
                        }
                    }
                    break;
                }
            }

        }
        m_bMemsetCheckComplete = TRUE;
        Unlock();

        return uiPrimitiveSize == 1 ?
            m_bMemsettableInitialValueD8 :
            m_bMemsettableInitialValue;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the initial value memset stride. </summary>
    ///
    /// <remarks>   crossbac, 7/6/2012. </remarks>
    ///
    /// <returns>   The initial value memset stride. </returns>
    ///-------------------------------------------------------------------------------------------------

    UINT 
    DatablockTemplate::GetInitialValueMemsetStride(
        VOID
        )
    {
        return m_bMemsetInitialValueStride;
    }
};
