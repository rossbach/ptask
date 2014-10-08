//--------------------------------------------------------------------------------------
// File: de.fx
// depth-engine (geometry, XY inference, segmentation)
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Constant Buffer Variables
//--------------------------------------------------------------------------------------
SamplerState samLinear
{
    Filter = MIN_MAG_MIP_POINT;
    // Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};


Texture2D g_ABPhaseTexture;			
Texture2D g_CobraRawTexture;		
Texture2D g_XYZCalibration;			// xyz calibration dat
float	  g_fgwidth = 320;
float	  g_fgheight = 200;
float	  g_abthresh = 50;
float4    g_LUT[256];
float3	  g_vR[3];
float	  g_vT[3];
bool	  g_LUTinit = false;
bool	  g_bABFilter = true;
bool	  g_bSegment = true;
matrix<float, 4, 4> g_R;

SamplerState XYZTextureSampler
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD0;
};


struct PS_OUTPUT
{
    float4 XYZB : SV_Target;  // Pixel color
};

PS_OUTPUT RenderSceneRaw( PS_INPUT In ) 
{ 
    float4 Output;
    PS_OUTPUT res;
    
	int texw, texh, elements;
	float2 vTexCoordFGSpace;
	g_CobraRawTexture.GetDimensions(0, texw, texh, elements);
	vTexCoordFGSpace[0] = (g_fgwidth / texw) * In.Tex[0];
	vTexCoordFGSpace[1] = (g_fgheight / texh) * In.Tex[1];

	int3 idx;
	idx[0] = vTexCoordFGSpace[0] * texw;
	idx[1] = vTexCoordFGSpace[1] * texh;
	idx[2] = 0;
	
	float sample = (float) g_CobraRawTexture.Load(idx);
	int cobraPel24bit = sample;
	unsigned int zValueRaw = (cobraPel24bit >> 10) & 0x3fff;
	float4 xyzEntry = g_XYZCalibration.Sample(XYZTextureSampler, In.Tex);
	float zCalib = xyzEntry[2];
	float zAdjust = zValueRaw / 16384.0;
	zAdjust *= zCalib;	
		
	int zval = zAdjust;
	if(zval > 255)
		zval = 255;
	res.XYZB = g_LUT[zval];
	return res;
}

PS_OUTPUT RenderSceneAB( PS_INPUT In ) 
{ 
    float4 Output;
    PS_OUTPUT res;
    
	int texw, texh, elements;
	float2 vTexCoordFGSpace;
	g_ABPhaseTexture.GetDimensions(0, texw, texh, elements);
	vTexCoordFGSpace[0] = (g_fgwidth / texw) * In.Tex[0];
	vTexCoordFGSpace[1] = (g_fgheight / texh) * In.Tex[1];

	int3 idx;
	idx[0] = vTexCoordFGSpace[0] * texw;
	idx[1] = vTexCoordFGSpace[1] * texh;
	idx[2] = 0;
	
	float4 sample = g_ABPhaseTexture.Sample(XYZTextureSampler, In.Tex);
	float abval = sample[0]/1024.0; 
	res.XYZB = abval;
	return res;
}

float3 Project(float3 input, matrix<float, 4, 4> RT) {
	float xWorld = 
		RT[0][0]*input[0] + 
		RT[0][1]*input[1] +
		RT[0][2]*input[2] +
		RT[3][0];
	float yWorld = 
		RT[1][0]*input[0] + 
		RT[1][1]*input[1] +
		RT[1][2]*input[2] +
		RT[3][1];
	float zWorld = 
		RT[2][0]*input[0] + 
		RT[2][1]*input[1] +
		RT[2][2]*input[2] +
		RT[3][2];
	float3 res;
	res[0] = xWorld;
	res[1] = yWorld;
	res[2] = zWorld;
	return res;					
}

PS_OUTPUT RenderSceneDE( PS_INPUT In ) 
{ 
    float4 Output;
    PS_OUTPUT res;
    
	int texw, texh, elements;
	float2 vTexCoordFGSpace;
	g_ABPhaseTexture.GetDimensions(0, texw, texh, elements);
	vTexCoordFGSpace[0] = (g_fgwidth / texw) * In.Tex[0];
	vTexCoordFGSpace[1] = (g_fgheight / texh) * In.Tex[1];

	int3 idx;
	idx[0] = vTexCoordFGSpace[0] * texw;
	idx[1] = vTexCoordFGSpace[1] * texh;
	idx[2] = 0;
	
	float4 sample = g_ABPhaseTexture.Sample(XYZTextureSampler, In.Tex);
	float4 xyzEntry = g_XYZCalibration.Sample(XYZTextureSampler, In.Tex);
	float abValue = sample[0];
	float zValueRaw = sample[1];
	float zCalib = xyzEntry[2];
	float zAdjust = zValueRaw / 16384.0;
	zAdjust *= zCalib;	
	float xValue = xyzEntry[0] * zAdjust;
	float yValue = xyzEntry[1] * zAdjust; 
	
	if(g_bSegment) {
		if(g_bABFilter && abValue < g_abthresh) {
			res.XYZB = 0;
			return res;
		}
		float3 pt;
		pt[0] = xValue;
		pt[1] = yValue;
		pt[2] = zAdjust;
		float3 ptWorld = Project(pt, g_R);
		if(ptWorld[2] > 0) {
			res.XYZB = 0;
			return res;
		}	
	}
	
	int zval = zAdjust;
	if(zval > 255)
		zval = 255;
 	res.XYZB = g_LUT[zval];
	return res;
}

PS_OUTPUT RenderSceneXYZB( PS_INPUT In ) 
{ 
    float4 Output;
    PS_OUTPUT res;
    
	int texw, texh, elements;
	float2 vTexCoordFGSpace;
	g_ABPhaseTexture.GetDimensions(0, texw, texh, elements);
	vTexCoordFGSpace[0] = (g_fgwidth / texw) * In.Tex[0];
	vTexCoordFGSpace[1] = (g_fgheight / texh) * In.Tex[1];

	int3 idx;
	idx[0] = vTexCoordFGSpace[0] * texw;
	idx[1] = vTexCoordFGSpace[1] * texh;
	idx[2] = 0;
	
	float4 sample = g_ABPhaseTexture.Sample(XYZTextureSampler, In.Tex);
	float4 xyzEntry = g_XYZCalibration.Sample(XYZTextureSampler, In.Tex);
	float abval = sample[0];
	float zValueRaw = sample[1];

	float zCalib = xyzEntry[2];
	float zAdjust = zValueRaw / 16384.0;
	zAdjust *= zCalib;	
	float xValue = xyzEntry[0] * zAdjust;
	float yValue = xyzEntry[1] * zAdjust;
	
	res.XYZB[0] = abval;
	res.XYZB[1] = xValue;
	res.XYZB[2] = yValue;
	res.XYZB[3] = zAdjust;
	return res;
}


//--------------------------------------------------------------------------------------
// cvtfp2int
// unpack the cobra AB 10-bit float format
// format :E-E-E-E-M-M-M-M-M-M 
// no sign bit needed!
//--------------------------------------------------------------------------------------
unsigned int cvtfp2int(unsigned int fpab)
{
	unsigned int mantissa, exponent, hiddenbit;
	mantissa = fpab & 0x003F;
	exponent = (fpab>>6) &	0x000f;
	hiddenbit = exponent?1:0;
	unsigned int luiab = mantissa | (hiddenbit << 6);
	int lsh = exponent + hiddenbit - 2;
	unsigned int uiab = luiab << lsh;
	return uiab;
}


//--------------------------------------------------------------------------------------
// SegmentXYZB
// kernel function that performs 3-byte Cobra raw format -> phase/AB, 
// segments the image based on AB threshold and volume in interest in
// world coordinate space, and returns a 4-tuple of X, Y, Z, B in world
// coordinates, with [0,0,0,0] encoding a pixel that has been eliminated
// due to low AB or outside VOI.
// 
// Inputs:
//  g_CobraRawTexture: float[1] 320x200: each float is really a 24-bit cobra pixel
//  g_XYZCalibration:  float[3] 320x200: x, y, z calibration entries generated on host
//  g_vR:              float[3][3]: rotation matrix for the current camera
//  g_vT:              float[3]:    translation matrix for the current camera
//  g_abthresh         float[1]:    ab threshold for segmentation.

// maintainer: rossbach
//--------------------------------------------------------------------------------------
PS_OUTPUT SegmentXYZB( PS_INPUT In ) 
{ 
    float4 Output;
    PS_OUTPUT res;
    
	// ------------------------------------------------------------
	// compute normlized coordinates for the current pixel based
	// on what we actually know about the image (200x320).
	// ------------------------------------------------------------
	float2 vTexCoordCobraRaw;
	int texw, texh, elements;
	g_CobraRawTexture.GetDimensions(0, texw, texh, elements);
	vTexCoordCobraRaw[0] = (g_fgwidth / texw) * In.Tex[0];
	vTexCoordCobraRaw[1] = (g_fgheight / texh) * In.Tex[1];

	// ------------------------------------------------------------
	// pixel shaders use normalized (u,v) coordinates. We really
	// need a specific value at an integral offset in the texture
	// (rather than an interpolated value). So compute the index
	// of the pixel we are interested in. 
	// ------------------------------------------------------------
	int3 idx;
	idx[0] = vTexCoordCobraRaw[0] * texw;
	idx[1] = vTexCoordCobraRaw[1] * texh;
	idx[2] = 0;
	
	// ------------------------------------------------------------
	// Load the pixel at the index, and unpack it. 
	// shader input is float[1]--inputs are guaranteed to
	// be integer-valued, and can be converted to a 24-bit
	// integer in the native Cobra 3-byte pixel format,
	// which is: 
	// depth (bits 23:10)
	// active brightness (bits 0:9) in EEEE-mmmmmm format
	// ------------------------------------------------------------	
	float sample = (float) g_CobraRawTexture.Load(idx);
	int cobraPel24bit = sample;
	unsigned int zValueRaw = (cobraPel24bit >> 10) & 0x3fff;
	unsigned int abValueRaw = cvtfp2int(cobraPel24bit & 0x3ff);
	
	// ------------------------------------------------------------
	// XYZ calibration table is stored in a texture of float[3]:
	// index 0 -> x calibration entry
	// index 1 -> y calibration entry
	// index 2 -> z calibration entry
	// The table is precomputed in advance on the host 
	// using the NormalizedCoordinates technique outlined by
	// Arrigo Benedetti. 
	// Z values need to be scaled down by 16384, and mulitplied
	// by the Z entry in the calibration table. The resulting
	// camera-coordinates Z value can then be multiplied by X,Y
	// calibration entries to produce the X and Y (camera space).
	// ------------------------------------------------------------
	float4 xyzCalibrationEntry = g_XYZCalibration.Load(idx);
	float xCalib = xyzCalibrationEntry[0];
	float yCalib = xyzCalibrationEntry[1];
	float zCalib = xyzCalibrationEntry[2];	
	float zCameraSpace = (zValueRaw / 16384.0) * zCalib;
	float xCameraSpace = xCalib * zCameraSpace;
	float yCameraSpace = yCalib * zCameraSpace;
	
	// ------------------------------------------------------------
	// Now segment the image. 
	// We discard pixels based on two criteria:
	// 1. Under the AB threshold (meaning high uncertainty)
	// 2. Outside the volume of interest. Currently this 
	//    just means eliminate anything with X,Y < 0 and 
	//    any Z that is "behind" the screen (reflection). 
	// *** Note that we work in a left-handed coordinate system,
	// *** so valid Z values are negative. Anything with 
	// *** z > 0 is behind the screen.
	// ------------------------------------------------------------
	if(abValueRaw < (unsigned int) g_abthresh) {
		res.XYZB = 0;  // vector set
		return res;		   // discarded pixel
	}
	
	float xWorld  = g_vR[0][0]*xCameraSpace + 
					g_vR[0][1]*yCameraSpace +
					g_vR[0][2]*zCameraSpace +
					g_vT[0];				   
	float yWorld  = g_vR[1][0]*xCameraSpace + 
					g_vR[1][1]*yCameraSpace +
					g_vR[1][2]*zCameraSpace +
					g_vT[1];
	float zWorld  = g_vR[2][0]*xCameraSpace + 
					g_vR[2][1]*yCameraSpace +
					g_vR[2][2]*zCameraSpace +
					g_vT[2];
					
	if(xWorld < 0 || yWorld < 0 || zWorld > 0) {
		res.XYZB = 0; 
		return res;		// discard: outside VOI
	}					
	
	// ------------------------------------------------------------
	// output is X, Y, Z, AB in world coordinates. Note this
	// result is for a single camera.
	// ------------------------------------------------------------
	res.XYZB[0] = xWorld;
	res.XYZB[1] = yWorld;
	res.XYZB[2] = zWorld;
	res.XYZB[3] = abValueRaw;
	return res;
}

