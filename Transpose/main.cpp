#include "../common/utils.h"

using namespace std;

#define DATA_TYPE		double
#define TILE 			(64)
#define ELE_PER_THR 	(4)
#define GROUP_SIZE 		(TILE*TILE / ELE_PER_THR)
#define ITERATION_TIMES (1000)
const uint32_t WIDTH  = 100 * 1;
const uint32_t HEIGHT = 100 * 51;
const uint32_t LEN = WIDTH * HEIGHT;

// ==========================================================================================
DATA_TYPE *h_in, *h_out;				// cpu memory handle
DATA_TYPE *d_in, *d_out;				// gpu memory handle

void InitHostMem()
{
	PrintStep1("Init Host Memory");

	h_in  = (DATA_TYPE*)malloc(LEN * sizeof(DATA_TYPE));
	h_out = (DATA_TYPE*)malloc(LEN * sizeof(DATA_TYPE));

	for (uint32_t i = 0; i < HEIGHT; i++)
	{
		for (uint32_t j = 0; j < WIDTH; j++)
		{
			uint32_t idx = i*WIDTH + j;
			h_in[idx] = i*100.0f + j*0.01f;
		}
	}
}
void FreeHostMem()
{
	PrintStep1("Free Host Memory");

	free(h_in);
	free(h_out);
}

void InitDeviceMem()
{
	PrintStep1("Init Device Memory");

	PrintStep2("Malloc Device Memory");
	HIP_ASSERT(hipMalloc((void**)&d_in,  LEN * sizeof(DATA_TYPE)));
	HIP_ASSERT(hipMalloc((void**)&d_out, LEN * sizeof(DATA_TYPE)));

	PrintStep2("Copy Host Memory To Device Memory");
	HIP_ASSERT(hipMemcpy(d_in, h_in, LEN * sizeof(DATA_TYPE), hipMemcpyHostToDevice));
}
void FreeDeviceMem()
{
	PrintStep1("Free Device Memory");

	hipFree(d_in);
	hipFree(d_out);
}

// ==========================================================================================
void SetKernelArgs()
{
	PrintStep2("Setup Kernel Arguments");

	uint32_t width  = WIDTH;
	uint32_t height = HEIGHT;

	AddArg(d_in);
	AddArg(d_out);
	AddArg(width);
	AddArg(height);
}
void SetKernelWorkload()
{
	PrintStep2("Setup Kernel Workload");

	SetGroupSize(GROUP_SIZE);
	SetGroupNum((WIDTH + TILE - 1) / TILE, (HEIGHT + TILE - 1) / TILE, 1);
}

void RunGpuCalculation()
{
	PrintStep1("Run Gpu Kernel");

	SetKernelArgs(); PrintKernelArgs();
	SetKernelWorkload(); PrintWorkload();
	LaunchKernel(); LaunchKernel();
	FreeKernelArgs();
}
void RunCpuCalculation()
{
	for (uint32_t i = 0; i < HEIGHT; i++)
	{
		for (uint32_t j = 0; j < WIDTH; j++)
		{
			uint32_t idx_in = i*WIDTH + j;
			uint32_t idx_out = j*HEIGHT + i;
			h_out[idx_out] = h_in[idx_in];
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
	for (uint32_t i = 0; i < ITERATION_TIMES; i++)
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

	CreateHipKernel("Transpose","Transpose.cpp");
	RunGpuCalculation();

	RunCpuCalculation();
	CompareData(h_out, d_out, LEN);

	TestEfficiency();

	return;
	printf("device index(dword):\n");
	uint32_t print_col = 8;
	uint32_t print_row = 8;
	for (uint32_t i = 0; i < WIDTH; i++)
	{
		for (uint32_t j = 0; j < HEIGHT; j++)
		{
			if (j == 0)
			{
				printf("[%03d][%03d~%03d]: ", i, j, j + print_col - 1);
			}

			uint32_t idx = i*HEIGHT + j;
			printf("%.2f, ", h_out[idx]);

			if ((j + 1) == print_col)
			{
				printf("...\n");
				break;
			}
		}

		if ((i + 1) == print_row)
		{
			printf("...\n");
			break;
		}
	}

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
