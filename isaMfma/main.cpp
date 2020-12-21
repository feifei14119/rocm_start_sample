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
#define M					(32)
#define N					(32)
#define K					(128)
#define ITERATION_TIMES		(1000)

// ==========================================================================================
/*
column major NT:

math:

				_____N_____
				|	      | K
				|	 B	  |
				|_________|

	____K____	_____N_____
	|       | 	|         |
	|		|	|	      |
  M |	A	| 	|	 C	  | M
	|		|	|		  |
	|_______|	|_________|

memory: 
A:	______M_______
	|		  	 |
	|		  	 | K
	|____________|
B:	_____N_____
	|	      |
	|	      | K
	|_________|
C:	______M_______
	|			 | 
	|			 | N
	|____________|
*/
float *h_A, *h_B, *h_C;				// cpu memory handle
float *d_A, *d_B, *d_C;				// gpu memory handle

void InitHostMem()
{
	PrintStep1("Init Host Memory");

	h_A = (float*)malloc(M*K * sizeof(float));
	h_B = (float*)malloc(N*K * sizeof(float));
	h_C = (float*)malloc(M*N * sizeof(float));

	for(uint32_t k = 0; k < K; k++)
	{
		for(uint32_t m = 0; m < M; m++)
		{
			h_A[M*k + m] = 1.0f * m;
		}
	}
	for(uint32_t k = 0; k < K; k++)
	{
		for(uint32_t n = 0; n < N; n++)
		{
			h_B[N*k + n] = -1.0f * 1;
		}
	}
	for(uint32_t n = 0; n < N; n++)
	{
		for(uint32_t m = 0; m < M; m++)
		{
			h_C[M*n + m] = 0;
		}
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
	HIP_ASSERT(hipMalloc((void**)&d_A, M*K * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)&d_B, N*K * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)&d_C, M*N * sizeof(float)));

	PrintStep2("Copy Host Memory To Device Memory");
	HIP_ASSERT(hipMemcpy(d_A, h_A, M*K * sizeof(float), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(d_B, h_B, N*K * sizeof(float), hipMemcpyHostToDevice));
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
	AddArg(M);
	AddArg(N);
	AddArg(K);
}
void SetKernelWorkload()
{
	PrintStep2("Setup Kernel Workload");

	SetGroupSize(WAVE_SIZE * 1);
	SetGlobalSize(M);
}

void RunGpuCalculation()
{
	PrintStep1("Run Gpu Kernel");

	SetKernelArgs(); PrintKernelArgs();
	SetKernelWorkload(); PrintWorkload();
	LaunchKernel();
	FreeKernelArgs();

	//printf("\nDevice Vector C\n"); PrintDeviceData(d_C, M*N);
}
void RunCpuCalculation()
{
	PrintStep1("Do Cpu Calculation");

	for(uint32_t n = 0; n < N; n++)
	{
		for(uint32_t m = 0; m < M; m++)
		{
			float c = 0;
			for(uint32_t k = 0; k < K; k++)
			{
				c += h_A[M*k + m] * h_B[N*k + n];
			}
			h_C[M*n + m] = c;
		}
	}

	//printf("\nHost Vector C\n");	PrintHostData(h_C, M*N);
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
	CreateAsmKernel("isaMfma", "isaMfma_v3.s");
#else
	CreateAsmKernel("isaMfma", "isaMfma_v2.s");
#endif
#else
	CreateHipKernel("isaMfma");
#endif

	RunGpuCalculation();
	RunCpuCalculation();
	CompareData(h_C, d_C, M*N);
	//TestEfficiency();

	printf("\n---------------------------------------\n");
}
int main(int argc, char *argv[])
{
	printf("\nHello ROCM.\n\n");
	InitHipRuntime();

	InitHostMem();
	InitDeviceMem();

	RunTest();

	FreeDeviceMem();
	FreeHostMem();

	ReleaseRuntime();
	printf("\nByeBye ROCM.\n\n");

	return 0;
}
