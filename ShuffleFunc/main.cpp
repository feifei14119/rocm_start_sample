#include "utils.h"

using namespace std;

//#define ASM_KERNEL			
//#define HIP_KERNEL

#define	VECTOR_LEN			(64)
#define ITERATION_TIMES		(1000)

// ==========================================================================================
uint32_t VectorLen;
float *h_A, *h_B, *h_C;				// cpu memory handle
float *d_A, *d_B, *d_C;				// gpu memory handle

void InitHostMem()
{
	PrintStep1("Init Host Memory");

	h_A = (float*)malloc(VectorLen * sizeof(float));
	h_B = (float*)malloc(VectorLen * sizeof(float));
	h_C = (float*)malloc(VectorLen * sizeof(float));

	for (unsigned int i = 0; i < VectorLen; i++)
	{
		h_A[i] = i * 1.0f;
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
	HIP_ASSERT(hipMalloc((void**)&d_A, VectorLen * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)&d_B, VectorLen * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)&d_C, VectorLen * sizeof(float)));

	PrintStep2("Copy Host Memory To Device Memory");
	HIP_ASSERT(hipMemcpy(d_A, h_A, VectorLen * sizeof(float), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(d_B, h_B, VectorLen * sizeof(float), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemset(d_C, 0, VectorLen * sizeof(float)));
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
	printf("        - Warp shuffle operations. (__shfl_*): %s.\n", HipDeviceProp.arch.hasWarpShuffle ? "TRUE" : "FALSE");
}
void RunTest()
{
	PrintDeviceShuffleCapbility();

	// ---------------------------------------------------
	printf("\n---------------------------------------\n");
	printf("__shfl.(support int32, float):\n");
#ifdef ASM_KERNEL
	CreateAsmKernel("VectorAdd");
#else
	CreateHipKernel("Shfl", "ShuffleFunc.cpp");
#endif
	RunGpuCalculation();
	printf("device A:\n");
	PrintDeviceData(d_A, VectorLen);
	printf("device B:\n");
	PrintDeviceData(d_B, VectorLen);

	// ---------------------------------------------------
	printf("\n---------------------------------------\n");
	printf("__shfl_up.(support int32, float):\n");
#ifdef ASM_KERNEL
	CreateAsmKernel("AtomicMax");
#else
	CreateHipKernel("ShflUp", "ShuffleFunc.cpp");
#endif
	RunGpuCalculation();
	printf("device A:\n");
	PrintDeviceData(d_A, VectorLen);
	printf("device B:\n");
	PrintDeviceData(d_B, VectorLen);

	// ---------------------------------------------------
	printf("\n---------------------------------------\n");
	printf("__shfl_down.(support int32, float):\n");
#ifdef ASM_KERNEL
	CreateAsmKernel("AtomicInc");
#else
	CreateHipKernel("ShflDown", "ShuffleFunc.cpp");
#endif
	RunGpuCalculation();
	printf("device A:\n");
	PrintDeviceData(d_A, VectorLen);
	printf("device B:\n");
	PrintDeviceData(d_B, VectorLen);

	// ---------------------------------------------------
	printf("\n---------------------------------------\n");
	printf("__shfl_xor.(support int32, float):\n");
#ifdef ASM_KERNEL
	CreateAsmKernel("AtomicOr");
#else
	CreateHipKernel("ShflXor", "ShuffleFunc.cpp");
#endif
	RunGpuCalculation();
	PrintDeviceData(d_A, VectorLen);
	printf("device B:\n");
	PrintDeviceData(d_B, VectorLen);

	printf("\n---------------------------------------\n");
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
