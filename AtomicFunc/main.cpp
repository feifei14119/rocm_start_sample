#include "utils.h"

using namespace std;

//#define ASM_KERNEL			
//#define HIP_KERNEL

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
		h_A[i] = i + 1;
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
int main(int argc, char *argv[])
{
	printf("\nHello ROCM.\n\n");
	InitHipRuntime();

	VectorLen = 32;

	InitHostMem();
	InitDeviceMem();

	// ---------------------------------------------------
	printf("\n---------------------------------------\n");
	printf("Atomic Add/Sub.(support int32, uint32, uint64, float):\n");
#ifdef ASM_KERNEL
	CreateAsmKernel("VectorAdd");
#else
	CreateHipKernel("AtomicAdd", "AtomicFunc.cpp");
#endif
	RunGpuCalculation();
	printf("device B:\n");
	PrintDeviceData(d_B, 1);
	printf("device C:\n");
	PrintDeviceData(d_C, VectorLen);

	// ---------------------------------------------------
	printf("\n---------------------------------------\n");
	printf("Atomic Max/Min.(support int32, uint32, uint64):\n");
#ifdef ASM_KERNEL
	CreateAsmKernel("AtomicMax");
#else
	CreateHipKernel("AtomicMax", "AtomicFunc.cpp");
#endif
	RunGpuCalculation();
	printf("device B:\n");
	PrintDeviceData(d_B, 1);
	printf("device C:\n");
	PrintDeviceData(d_C, VectorLen);

	// ---------------------------------------------------
	printf("\n---------------------------------------\n");
	printf("Atomic Inc/Dec.(support uint32):\n");
#ifdef ASM_KERNEL
	CreateAsmKernel("AtomicInc");
#else
	CreateHipKernel("AtomicInc", "AtomicFunc.cpp");
#endif
	RunGpuCalculation();
	printf("device B:\n");
	PrintDeviceData(d_B, 1);
	printf("device C:\n");
	PrintDeviceData(d_C, VectorLen);

	// ---------------------------------------------------
	printf("\n---------------------------------------\n");
	printf("Atomic And/Or/Xor.(support uint32):\n");
#ifdef ASM_KERNEL
	CreateAsmKernel("AtomicOr");
#else
	CreateHipKernel("AtomicOr", "AtomicFunc.cpp");
#endif
	RunGpuCalculation();
	printf("device B:\n");
	PrintDeviceData(d_B, 1);
	printf("device C:\n");
	PrintDeviceData(d_C, VectorLen);

	// ---------------------------------------------------
	printf("\n---------------------------------------\n");
	printf("Atomic Exch.(support int32, uint32, uint64, float):\n");
#ifdef ASM_KERNEL
	CreateAsmKernel("AtomicExch");
#else
	CreateHipKernel("AtomicExch", "AtomicFunc.cpp");
#endif
	RunGpuCalculation();
	printf("device B:\n");
	PrintDeviceData(d_B, 1);
	printf("device C:\n");
	PrintDeviceData(d_C, VectorLen);

	// ---------------------------------------------------
	printf("\n---------------------------------------\n");
	printf("Atomic CAS.(support int32, uint32, uint64):\n");
#ifdef ASM_KERNEL
	CreateAsmKernel("AtomicCAS");
#else
	CreateHipKernel("AtomicCAS", "AtomicFunc.cpp");
#endif
	RunGpuCalculation();
	printf("device B:\n");
	PrintDeviceData(d_B, 1);
	printf("device C:\n");
	PrintDeviceData(d_C, VectorLen);

	// ---------------------------------------------------
	printf("\n---------------------------------------\n");
	FreeDeviceMem();
	FreeHostMem();

	ReleaseRuntime();
	printf("\nByeBye ROCM.\n\n");

	return 0;
}
