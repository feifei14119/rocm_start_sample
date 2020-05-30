#include "../common/utils.h"

using namespace std;

//#define ASM_KERNEL			
//#define HIP_KERNEL

#define GROUP_NUM			(2)
#define VECTOR_LEN			(WAVE_SIZE * SIMD_PER_CU * GROUP_NUM)
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
		h_A[i] = i * 10;
		h_B[i] = i * 0.01;
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

	SetGroupSize(WAVE_SIZE * SIMD_PER_CU);
	SetGroupNum((VectorLen + GroupSize.x - 1) / GroupSize.x);
}

void RunGpuCalculation()
{
	PrintStep1("Run Gpu Kernel");

	SetKernelArgs(); PrintKernelArgs();
	SetKernelWorkload(); PrintWorkload();
	LaunchKernel();
	FreeKernelArgs();
}
void RunCpuCalculation()
{
	PrintStep1("Do Cpu Calculation");

	for (unsigned int i = 0; i < VectorLen; i++)
	{
		h_C[i] = h_A[i] + h_B[i];
	}

	//printf("\nHost Vector C\n");	PrintHostData(h_C);
}
void VerifyResult()
{
	PrintStep1("Verify GPU Result");

	float * dev_rslt = (float*)malloc(VectorLen * sizeof(float));

	PrintStep2("Copy Device Result To Host");
	HIP_ASSERT(hipMemcpy(dev_rslt, d_C, VectorLen * sizeof(float), hipMemcpyDeviceToHost));

	PrintStep2("Compare Device Result With Cpu Result");
	for (unsigned int i = 0; i < VectorLen; i++)
	{
		if (fabs(h_C[i] - dev_rslt[i]) > FLT_MIN)
		{
			printf("    - First Error:\n");
			printf("    - Host  : [%d] = %.2f.\n", i, h_C[i]);
			printf("    - Device: [%d] = %.2f.\n", i, dev_rslt[i]);
			break;
		}

		if (i == VectorLen - 1)
		{
			printf("    - Verify Success.\n");
		}
	}
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
int main(int argc, char *argv[])
{
	printf("\nHello ROCM.\n\n");
	InitHipRuntime();

	VectorLen = VECTOR_LEN;
	InitHostMem();
	InitDeviceMem();

#ifdef ASM_KERNEL
	CreateAsmKernel("VectorAdd");
#else
	CreateHipKernel("VectorAdd");
#endif

	RunGpuCalculation();
	RunCpuCalculation();
	VerifyResult();
	TestEfficiency();

	FreeDeviceMem();
	FreeHostMem();

	ReleaseRuntime();
	printf("\nByeBye ROCM.\n\n");

	return 0;
}
