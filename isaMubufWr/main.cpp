﻿#include "../common/utils.h"

using namespace std;

//#define ASM_KERNEL			
//#define HIP_KERNEL

#define WAVE_NUM			(1)
#define VECTOR_LEN			(128)

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
		h_B[i] = i * 0.01;
		h_C[i] = 0;
	}
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

	unsigned int vec_len = VECTOR_LEN;

	AddArg(d_A);
	AddArg(d_B);
	AddArg(d_C);
	AddArg(vec_len);
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
}

// ==========================================================================================
void RunTest()
{
	printf("\n---------------------------------------\n");

#ifdef HIP_KERNEL
	printf("hip kernel not support for this sample.\n");
#endif

#ifdef OBJ_V3
	CreateAsmKernel("isaMubufWr", "isaMubufWr_v3.s");
#else
	CreateAsmKernel("isaMubufWr", "isaMubufWr_v2.s");
#endif
	RunGpuCalculation();

	printf("device C:\n");
	PrintDeviceData(d_C, VectorLen);

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
