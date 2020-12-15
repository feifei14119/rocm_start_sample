#include "../common/utils.h"

using namespace std;

//#define ASM_KERNEL
//#define HIP_KERNEL
//#define OBJ_V2
//#define OBJ_V3
//#define CMP_LLVM
//#define CMP_HCC

#define GROUP_NUM			(1)
#define WAVE_NUM			(1) // SIMD_PER_CU
#define VECTOR_LEN			(WAVE_SIZE * WAVE_NUM * GROUP_NUM)
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
		h_B[i] = i * 0.01f;
		h_C[i] = 0;
	}

	//printf("\nHost Vector A\n");	PrintHostData(h_A, VectorLen);
	//printf("\nHost Vector B\n");	PrintHostData(h_B, VectorLen);
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

	SetGroupSize(WAVE_SIZE * WAVE_NUM);
	SetGroupNum((VectorLen + GroupSize.x - 1) / GroupSize.x);
}

void RunGpuCalculation()
{
	PrintStep1("Run Gpu Kernel");

	SetKernelArgs(); PrintKernelArgs();
	SetKernelWorkload(); PrintWorkload();
	LaunchKernel();
	FreeKernelArgs();

	printf("\nDevice Vector C\n"); PrintDeviceData(d_C, 1);
}
void RunCpuCalculation()
{
	PrintStep1("Do Cpu Calculation");

	h_C[0] = 0;
	for (unsigned int i = 0; i < VectorLen; i++)
	{
		h_C[0] += h_A[i];
	}

	printf("\nHost Vector C\n");	PrintHostData(h_C, 1);
}

// ==========================================================================================
void TestEfficiency()
{
	PrintStep1("Test Gpu Kernel Efficiency");

	SetKernelArgs();
	SetKernelWorkload();

	PrintStep2("Warmup");
	LaunchKernelGetElapsedMs();

	PrintStep2("Run GpuKernel for " + to_string(ITERATION_TIMES) + " times");
	double elapsed_ms = 0;
	for (unsigned int i = 0; i < ITERATION_TIMES; i++)
	{
		elapsed_ms += LaunchKernelGetElapsedMs();
	}
	printf("    - Kernel Elapsed Time = %.3f(ms).\n", elapsed_ms / ITERATION_TIMES);

	FreeKernelArgs();
}

// ==========================================================================================
void RunTest()
{
	printf("\n---------------------------------------\n");

#ifdef ASM_KERNEL
#ifdef OBJ_V3
	CreateAsmKernel("isaDpp", "isaDpp_v3.s");
#else
	CreateAsmKernel("isaDpp", "isaDpp_v2.s");
#endif
#else
	CreateHipKernel("isaDpp");
#endif

	RunGpuCalculation();
	RunCpuCalculation();
	CompareData(h_C, d_C, 1);
	TestEfficiency();

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
