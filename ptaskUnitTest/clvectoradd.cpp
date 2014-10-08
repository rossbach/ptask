//--------------------------------------------------------------------------------------
// File: clvectoradd.cpp
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#include <stdio.h>
#include <crtdbg.h>
#include "accelerator.h"
#include "ptask.h"
#include "assert.h"
#include "streamadd.h"
#include "claccelerator.h"
#include "clptask.h"

#if defined(DEBUG) || defined(PROFILE)
static char * szBuf0Name = "StreamA";
static char * szBuf1Name = "StreamB";
static char * szResultName = "Result";
static char * szAChannel = "A Input Channel";
static char * szBChannel = "B Input Channel";
static char * szCChannel = "C Output Channel";
#else
static char * szBuf0Name = NULL;
static char * szBuf1Name = NULL;
static char * szResultName = NULL;
static char * szAChannel = NULL;
static char * szBChannel = NULL;
static char * szCChannel = NULL;
#endif

extern void 
configure_stream_add(
	UINT elems,
	float** A,
	float** B
	);

extern float *
stream_add(
	UINT elems, 
	float * pA,
	float * pB
	);

extern bool 
check_vector_result(
	UINT elems,
	float* pA,
	float* pReferenceResult
	);

extern void
print_vector(
	char * label,
	UINT elems,
	float* p
	);

int run_cl_add_task(
	char * szfile,
	char * szshader,
	int elems
	) 
{
	float* vA;
	float* vB;
	configure_stream_add(elems, &vA, &vB);

	CLAccelerator * pAccelerator = new CLAccelerator();

	UINT stride = sizeof(float);
	Endpoint * pA	= pAccelerator->CreateEndpoint( INPUT_ENDPOINT, stride, elems, vA, szBuf0Name );
    Endpoint * pB	= pAccelerator->CreateEndpoint( INPUT_ENDPOINT, stride, elems, vB, szBuf1Name );
    Endpoint * pC	= pAccelerator->CreateEndpoint( OUTPUT_ENDPOINT, stride, elems, NULL, szResultName );
	Endpoint * pI   = pAccelerator->CreateEndpoint( CONSTANT_ENDPOINT, sizeof(elems), 1, &elems, NULL);

	// the UID for each endpoint needs to match the ordinal position 
	// for the parameter in the cl function signature. In our case:
	// __kernel void vadd(__global const float* a, __global const float* b, __global float* c, int iNumElements)
	pA->SetUID(0);
	pB->SetUID(1);
	pC->SetUID(2);
	pI->SetUID(3);

	printf( "Creating PTask(%s), mapping endpoints...", szshader );
    PTask * ptask = pAccelerator->CreatePTask(szfile, szshader);
	ptask->MapEndpoint(pA);
	ptask->MapEndpoint(pB);
	ptask->MapEndpoint(pC);
	ptask->MapEndpoint(pI);
    printf( "done\n" );

    printf( "Running ptask..." );
    ptask->Run( elems, 1, 1 );
    printf( "done\n" );


	// do we get the result we expect?
	float* vC = new float[elems];
	size_t datasize = elems*sizeof(float);
	UINT collected = pC->Pull(vC, (UINT) datasize);
	assert(datasize == collected);
	UNREFERENCED_PARAMETER(collected); // for release build
	printf( "Verifying against CPU result..." );
	float* pReference = stream_add(elems, vA, vB); // on CPU
	if(!check_vector_result(elems, vC, pReference)) 
		printf("failure\n");
	else 
        printf( "%s succeeded\n", szshader );

	bool bVerbose = true;
	if(bVerbose) {
		print_vector("GPU version: ", elems, vC);
		print_vector("Host verion: ", elems, pReference);
	}

	delete pA;
	delete pB;
	delete pC;
	delete ptask;
	delete pAccelerator;
	delete vA;
	delete vB;
	delete vC;
	delete pReference;
    
	return 0;
}

