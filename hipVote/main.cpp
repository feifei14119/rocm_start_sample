#include "../common/utils.h"

using namespace std;

//#define ASM_KERNEL
//#define HIP_KERNEL

#define	VECTOR_LEN			(64)
#define ITERATION_TIMES		(1000)

// ==========================================================================================
uint32_t VectorLen;
int *h_A, *h_B, *h_C;				// cpu memory handle
int *d_A, *d_B, *d_C;				// gpu memory handle

void InitHostMem()
{
	PrintStep1("Init Host Memory");

	h_A = (int*)malloc(VectorLen * sizeof(int));
	h_B = (int*)malloc(VectorLen * sizeof(int));
	h_C = (int*)malloc(VectorLen * sizeof(int));

	for (unsigned int i = 0; i < VectorLen; i++)
	{
		h_A[i] = i;
		h_B[i] = 100;
		h_C[i] = 0;
	}

	//printf("\nHost Vector A\n");	PrintHostData(h_A);
	//printf("\nHost Vector B\n");	PrintHostData(h_B);
}
void FreeHostMem()
{
	PrintStep1("Free Host Memory");

	free(h_A);
	free(h_B);
	free(h_C);
}

void InitDeviceMem()
{
	PrintStep1("Init Device Memory");

	PrintStep2("Malloc Device Memory");
	HIP_ASSERT(hipMalloc((void**)&d_A, VectorLen * sizeof(int)));
	HIP_ASSERT(hipMalloc((void**)&d_B, VectorLen * sizeof(int)));
	HIP_ASSERT(hipMalloc((void**)&d_C, VectorLen * sizeof(int)));

	PrintStep2("Copy Host Memory To Device Memory");
	HIP_ASSERT(hipMemcpy(d_A, h_A, VectorLen * sizeof(int), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(d_B, h_B, VectorLen * sizeof(int), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemset(d_C, 0, VectorLen * sizeof(int)));
}
void FreeDeviceMem()
{
	PrintStep1("Free Device Memory");

	hipFree(d_A);
	hipFree(d_B);
	hipFree(d_C);
}

// ==========================================================================================
void SetKernelArgs()
{
	PrintStep2("Setup Kernel Arguments");

	AddArg(d_A);
	AddArg(d_B);
	AddArg(d_C);
	AddArg(VectorLen);
}
void SetKernelWorkload()
{
	PrintStep2("Setup Kernel Workload");

	SetGroupSize(WAVE_SIZE);
	SetGroupNum((VectorLen + GroupSize.x - 1) / GroupSize.x);
}
void RunGpuCalculation()
{
	PrintStep1("Run Gpu Kernel");

	SetKernelArgs(); //PrintKernelArgs();
	SetKernelWorkload(); //PrintWorkload();
	LaunchKernel();
	FreeKernelArgs();
}

// ==========================================================================================
void PrintDeviceShuffleCapbility()
{
	printf("\nDevice Atomic Capbility:\n");
	printf("    - Warp size: %d.\n", HipDeviceProp.warpSize);
	printf("    - Architectural Feature Flags: %X.\n", HipDeviceProp.arch);
	printf("        - Warp cross-lane operations.\n");
	printf("        - Warp vote instructions (__any, __all): %s.\n", HipDeviceProp.arch.hasWarpVote ? "TRUE" : "FALSE");
	printf("        - Warp ballot instructions (__ballot): %s.\n", HipDeviceProp.arch.hasWarpBallot ? "TRUE" : "FALSE");
}
void RunTest()
{
	PrintDeviceShuffleCapbility();

#ifdef ASM_KERNEL
	printf("assembly kernel not support for this sample.\n");
#endif

	CreateHipKernel("hipVote", "hipVote.cpp");
	RunGpuCalculation();

	PrintStep2("Copy Device Result To Host");
	HIP_ASSERT(hipMemcpy(h_B, d_B, VectorLen * sizeof(int), hipMemcpyDeviceToHost));
	printf("    - Any Result = %d.\n", h_B[0]);
	printf("    - All Result = %d.\n", h_B[1]);
	printf("    - Ballot Result = 0x%08X.\n", h_B[2]);
}
int main(int argc, char *argv[])
{
	printf("\nHello ROCM.\n\n");
	InitHipRuntime();

	VectorLen = VECTOR_LEN;

	InitHostMem();
	InitDeviceMem();

	RunTest();

	FreeDeviceMem();
	FreeHostMem();

	ReleaseRuntime();
	printf("\nByeBye ROCM.\n\n");

	return 0;
}
