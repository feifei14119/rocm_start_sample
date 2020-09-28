#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <hip/hip_runtime.h>
#include <math.h>
#include <float.h>

using namespace std;

#define WAVE_SIZE			(64)
#define SIMD_PER_CU			(4)
#define GROUP_SIZE			(WAVE_SIZE * SIMD_PER_CU)
#define GROUP_NUM			(2)
#define VECTOR_LEN			(GROUP_SIZE * GROUP_NUM)
#define ITERATION_TIMES		(1000)
#define HIP_ASSERT(x) (assert((x)==hipSuccess))

extern "C" __global__ void VectorAdd(const float * a, 	const float * b, float * c,	const uint32_t len) 
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (x >= len)
		return;

	c[x] = a[x] + b[x];
}

uint32_t VectorLen;
float *h_A, *h_B, *h_C;				// cpu memory handle
float *d_A, *d_B, *d_C;				// gpu memory handle

// ==========================================================================================
void InitHostMem()
{
	printf("Init Host Memory.\n");

	h_A = (float*)malloc(VectorLen * sizeof(float));
	h_B = (float*)malloc(VectorLen * sizeof(float));
	h_C = (float*)malloc(VectorLen * sizeof(float));

	for (unsigned int i = 0; i < VectorLen; i++)
	{
		h_A[i] = i * 10;
		h_B[i] = i * 0.01;
		h_C[i] = 0;
	}
}
void FreeHostMem()
{
	printf("Free Host Memory.\n");

	free(h_A);
	free(h_B);
	free(h_C);
}

void InitDeviceMem()
{
	printf("Init Device Memory.\n");

	printf("Malloc Device Memory.\n");
	HIP_ASSERT(hipMalloc((void**)&d_A, VectorLen * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)&d_B, VectorLen * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)&d_C, VectorLen * sizeof(float)));

	printf("Copy Host Memory To Device Memory.\n");
	HIP_ASSERT(hipMemcpy(d_A, h_A, VectorLen * sizeof(float), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(d_B, h_B, VectorLen * sizeof(float), hipMemcpyHostToDevice));
}
void FreeDeviceMem()
{
	printf("Free Device Memory.\n");

	hipFree(d_A);
	hipFree(d_B);
	hipFree(d_C);
}

// ==========================================================================================
void RunGpuCalculation()
{
	printf("Run Gpu Kernel.\n");

	hipLaunchKernelGGL(VectorAdd, dim3(GROUP_NUM), dim3(GROUP_SIZE), 
						0, 0, 
						d_A,d_B,d_C,VectorLen);
}
void RunCpuCalculation()
{
	printf("Do Cpu Calculation.\n");

	for (unsigned int i = 0; i < VectorLen; i++)
	{
		h_C[i] = h_A[i] + h_B[i];
	}
}

// ==========================================================================================
void VerifyResult()
{
	float * h_rslt = (float*)malloc(VectorLen * sizeof(float));
	HIP_ASSERT(hipMemcpy(h_rslt, d_C, VectorLen * sizeof(float), hipMemcpyDeviceToHost));

	uint32_t col_num = 8;
	uint32_t raw_num = 8;
	for(uint32_t i = 0; i < VectorLen; i++)
	{
		if (i % col_num == 0)
		{
			printf("[%03d~%03d]: ", i, i + col_num - 1);
		}
		printf("%.2f, ", h_rslt[i]);
		if ((i + 1) % col_num == 0)
		{
			printf("\n");
		}
		if((i / col_num) >= raw_num)
		{
			printf("...\n");
			break;
		}
	}
	
	for (uint32_t i = 0; i < VectorLen; i++)
	{
		if (fabs(h_rslt[i] - h_C[i]) > FLT_MIN)
		{
			printf("    - First Error:\n");
			printf("    - Host  : [%d] = %.2f.\n", i, h_C[i]);
			printf("    - Device: [%d] = %.2f.\n", i, h_rslt[i]);
			break;
		}

		if (i == VectorLen - 1)
		{
			printf("    - Verify Success.\n");
		}
	}

	free(h_rslt);
}
void TestEfficiency()
{
	printf("Test Gpu Kernel Efficiency.\n");

	printf("Warmup.\n");
	hipLaunchKernelGGL(VectorAdd, dim3(GROUP_NUM), dim3(GROUP_SIZE), 0, 0, d_A,d_B,d_C,VectorLen);

	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);
	float elapsed_ms = 0;

	printf(("Run GpuKernel for " + to_string(ITERATION_TIMES) + " times.\n").c_str());
	hipStreamSynchronize(0);
	hipEventRecord(start, NULL);

	for (unsigned int i = 0; i < ITERATION_TIMES; i++)
	{
		hipLaunchKernelGGL(VectorAdd, dim3(GROUP_NUM), dim3(GROUP_SIZE), 0, 0, d_A,d_B,d_C,VectorLen);
	}

	hipEventRecord(stop, NULL);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&elapsed_ms, start, stop);
	printf("    - Kernel Elapsed Time = %.3f(ms).\n", elapsed_ms / ITERATION_TIMES);
}

// ==========================================================================================
void RunTest()
{
	printf("\n---------------------------------------\n");

	RunGpuCalculation();
	RunCpuCalculation();
	VerifyResult();
	TestEfficiency();

	printf("\n---------------------------------------\n");
}
int main(int argc, char *argv[])
{
	printf("\nHello ROCM.\n\n");

	VectorLen = VECTOR_LEN;
	InitHostMem();
	InitDeviceMem();

	RunTest();

	FreeDeviceMem();
	FreeHostMem();

	printf("\nByeBye ROCM.\n\n");

	return 0;
}
