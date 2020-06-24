#include <hip/hip_fp16.h>

#include "../common/utils.h"

using namespace std;

#define ASM_KERNEL
//#define HIP_KERNEL
#define OBJ_V2

#define	VECTOR_LEN			(128)

// ==========================================================================================
uint32_t VectorLen = VECTOR_LEN;
// cpu:   h_A(fp32)   +   h_B(fp32) = h_C(fp32)  <-- compare --> h_fp16tofp32DevC(fp32)
//         |               |                                             ^
//         V			   V                                             |
// gpu: d_fp16A(fp16) + d_fp16B(fp16)              =                  d_fp16C(fp16)
float *h_A, *h_B, *h_C;
half *h_fp16A, *h_fp16B;
half *d_fp16A, *d_fp16B, *d_fp16C;
float *h_fp16tofp32DevC;

void InitHostMem()
{
	PrintStep1("Init Host Memory");

	h_A = (float*)malloc(VectorLen * sizeof(float));
	h_B = (float*)malloc(VectorLen * sizeof(float));
	h_C = (float*)malloc(VectorLen * sizeof(float));
	h_fp16A = (half*)malloc(VectorLen * sizeof(half));
	h_fp16B = (half*)malloc(VectorLen * sizeof(half));
	h_fp16tofp32DevC = (float*)malloc(VectorLen * sizeof(float));

	for (uint32_t i = 0; i < VectorLen; i++)
	{
		h_A[i] = i * 1.0f;
		h_B[i] = 3.141f;
		h_fp16A[i] = __float2half(h_A[i]);
		h_fp16B[i] = __float2half(h_B[i]);
		h_C[i] = 0;
		h_fp16tofp32DevC[i] = 0;
	}
}
void FreeHostMem()
{
	PrintStep1("Free Host Memory");

	free(h_A);
	free(h_B);
	free(h_C);
	free(h_fp16A);
	free(h_fp16B);
	free(h_fp16tofp32DevC);
}

void InitDeviceMem()
{
	PrintStep1("Init Device Memory");

	PrintStep2("Malloc Device Memory");
	HIP_ASSERT(hipMalloc((void**)&d_fp16A, VectorLen * sizeof(half)));
	HIP_ASSERT(hipMalloc((void**)&d_fp16B, VectorLen * sizeof(half)));
	HIP_ASSERT(hipMalloc((void**)&d_fp16C, VectorLen * sizeof(half)));

	PrintStep2("Copy Host Memory To Device Memory");
	HIP_ASSERT(hipMemcpy(d_fp16A, h_fp16A, VectorLen * sizeof(half), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(d_fp16B, h_fp16B, VectorLen * sizeof(half), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemset(d_fp16C, 0, VectorLen * sizeof(half)));
}
void FreeDeviceMem()
{
	PrintStep1("Free Device Memory");

	hipFree(d_fp16A);
	hipFree(d_fp16B);
	hipFree(d_fp16C);
}

// ==========================================================================================
void SetKernelArgs()
{
	PrintStep2("Setup Kernel Arguments");

	AddArg(d_fp16A);
	AddArg(d_fp16B);
	AddArg(d_fp16C);
	AddArg(VectorLen);
}
void SetKernelWorkload()
{
	PrintStep2("Setup Kernel Workload");

	SetGroupSize(WAVE_SIZE);	
	SetGroupNum((VectorLen/2 + GroupSize.x - 1) / GroupSize.x);// every thread process 2 elements
}
void RunGpuCalculation()
{
	PrintStep1("Run Gpu Kernel");

	SetKernelArgs(); //PrintKernelArgs();
	SetKernelWorkload(); //PrintWorkload();
	LaunchKernel();
	FreeKernelArgs();
}
void RunCpuCalculation()
{
	PrintStep1("Do Cpu Calculation");

	for (uint32_t i = 0; i < VectorLen; i++)
	{
		h_C[i] = h_A[i] * h_B[i];
	}
}

// ==========================================================================================
void TestDataType()
{
	half fp16_a, fp16_b;
	float fp32_a, fp32_b;
	half2 fp162;
	__half2_raw fp16raw; // to access half2 value

	printf("\n---------------------------------------\n");
	printf("data size of half half2 and float:\n");
	printf("size of half is %d Byte.\n", sizeof(half));
	printf("size of half2 is %d Byte.\n", sizeof(half2));
	printf("size of float is %d Byte.\n", sizeof(float));

	printf("\n---------------------------------------\n");
	printf("convert float and half:\n");
	fp32_a = 3.1415926535898f;
	printf("fp32 befor convert = %.12f(0x%08X).\n", fp32_a, *(int*)&fp32_a);
	fp16_a = __float2half(fp32_a);
	printf("convert to fp16 = 0x%04X.\n", *(short*)&fp16_a);
	fp32_a = __half2float(fp16_a);
	printf("fp32 after convert = %.12f(0x%08X).\n", fp32_a, *(int*)&fp32_a);

	printf("\n---------------------------------------\n");
	printf("convert float and half2:\n");
	fp32_a = 3.1415926535898f;
	fp32_b = 2.718281828459f;
	printf("fp32_a befor convert = 0x%08X(%.12f).\n", *(int*)&fp32_a, fp32_a);
	printf("fp32_b befor convert = 0x%08X(%.12f).\n", *(int*)&fp32_b, fp32_b);
	fp16_a = __float2half(fp32_a);
	fp16_b = __float2half(fp32_b);
	printf("fp16_a = 0x%04X.\n", *(short*)&fp16_a);
	printf("fp16_b = 0x%04X.\n", *(short*)&fp16_b);
	fp162 = half2(fp32_a, fp32_b);
	printf("fp16_raw = 0x%08X.\n", *(int*)&fp162);
	fp16raw = __half2_raw(fp162);
	fp32_a = __half2float(*(half*)&fp16raw.x);
	fp32_b = __half2float(*(half*)&fp16raw.y);
	printf("fp32_a after convert = %.12f(0x%08X).\n", fp32_a, *(int*)&fp32_a);
	printf("fp32_b after convert = %.12f(0x%08X).\n", fp32_b, *(int*)&fp32_b);
}
void CompareResult()
{
	PrintStep1("Verify GPU Result");

	half * h_f16DevC = (half*)malloc(VectorLen * sizeof(half));

	PrintStep2("Copy Device Half Result To Host");
	HIP_ASSERT(hipMemcpy(h_f16DevC, d_fp16C, VectorLen * sizeof(half), hipMemcpyDeviceToHost));

	PrintStep2("Convert Half to Float on Host");
	for (uint32_t i = 0; i < VectorLen; i++)
	{
		h_fp16tofp32DevC[i] = __half2float(h_f16DevC[i]);
	}

	PrintStep2("Compare Device Result With Cpu Result");
	double err_tmp, err_max, err_sum = 0;
	uint32_t err_max_idx;
	for (uint32_t i = 0; i < VectorLen; i++)
	{
		err_tmp = fabs(h_C[i] - h_fp16tofp32DevC[i]);
		err_sum += err_tmp;
		if (err_tmp > err_max)
		{
			err_max = err_tmp;
			err_max_idx = i;
		}
		printf("idx[%03d]: hst = %.3f, dev = %.3f, err = %.3f\n", i, h_C[i], h_fp16tofp32DevC[i], err_tmp);
	}
	printf("    - Max Error = %.3f @ %d.\n", err_max, err_max_idx);
	printf("    - Sum Error = %.3f.\n", err_sum);

	free(h_f16DevC);
}
void RunTest()
{
	printf("\n=======================================\n");
	printf("half and float data type test:\n");
	TestDataType();

	printf("\n---------------------------------------\n");
#ifdef ASM_KERNEL
#ifdef OBJ_V3
	CreateAsmKernel("isaPackedFp16", "isaPackedFp16_v3.s");
#else
	CreateAsmKernel("isaPackedFp16", "isaPackedFp16_v2.s");
#endif
#else
	printf("hip kernel not support for this sample.\n");
	printf("hip doesn't support fp16 kenrel opt.\n");
	printf("hip doesn't support fp16 cpu opt.\n");
#endif

	RunGpuCalculation();
	RunCpuCalculation();

	CompareResult();
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
