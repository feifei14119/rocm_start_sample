#include "../common/utils.h"

using namespace std;

//#define ASM_KERNEL
//#define HIP_KERNEL

#define	VECTOR_LEN			(128)
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
		h_B[i] = i * 0.1f;
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
double ClockIntervalMs;
void PrintDeviceClockCapbility()
{
	ClockIntervalMs = 1.0 / HipDeviceProp.clockInstructionRate;

	printf("\nDevice Atomic Capbility:\n");
	Logout("    - Device Timer clock() Frequency: " + FormatFreq(HipDeviceProp.clockInstructionRate * 1000));
	printf("    - Device Timer clock() Interval = %.3f(us)\n", ClockIntervalMs * 1000);
}
void RunTest()
{
	PrintDeviceClockCapbility();

#ifdef ASM_KERNEL
	printf("assembly kernel not support for this sample.\n");
#endif

	CreateHipKernel("hipClock", "hipClock.cpp");
	RunGpuCalculation();

	PrintStep2("Copy Device Result To Host");
	HIP_ASSERT(hipMemcpy(h_C, d_C, VectorLen * sizeof(float), hipMemcpyDeviceToHost));
	printf("    - load a  elapsed time = %.3f(us).\n", h_C[0] * ClockIntervalMs * 1000);
	printf("    - load a  elapsed time = %.3f(us).\n", h_C[1] * ClockIntervalMs * 1000);
	printf("    - a + b   elapsed time = %.3f(us).\n", h_C[2] * ClockIntervalMs * 1000);
	printf("    - write c elapsed time = %.3f(us).\n", h_C[3] * ClockIntervalMs * 1000);
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
