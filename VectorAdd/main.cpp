#include <float.h>
#include <math.h>
#include <string>
#include <hip/hip_hcc.h>

using namespace std;

//#define ASM_KERNEL			
//#define HIP_KERNEL

#define VECTOR_LEN			(1024*15)
#define ITERATION_TIMES		(1000)

#define HIP_ASSERT(x)		(assert((x)==hipSuccess))
typedef struct { unsigned int x, y, z; }Dim3;

// ==========================================================================================
int Step = 0, SubStep = 0;
void PrintStep1(string s)
{
	Step++;
	printf("%d. %s.\n", Step, s.c_str());	
	SubStep = 0;
}
void PrintStep2(string s)
{
	SubStep++;
	printf("  %d.%d %s.\n", Step, SubStep, s.c_str());
}
void PrintHostData(float * pData)
{
	unsigned int col_num = 8;
	for (unsigned int i = 0; i < VECTOR_LEN; i++)
	{
		if (i % col_num == 0)
		{
			printf("[%03d~%03d]: ", i, i + col_num - 1);
		}
		printf("%.2f, ", *(pData + i));
		if ((i+1)%col_num == 0)
		{
			printf("\n");
		}
	}
}
void ExecCommand(string cmd)
{
#ifdef _WIN32
	system(cmd.c_str());
#else
	FILE * pfp = popen(cmd.c_str(), "r");
	auto status = pclose(pfp);
	WEXITSTATUS(status);
#endif
}

// ==========================================================================================
float *h_A, *h_B, *h_C;				// cpu memory handle
float *d_A, *d_B, *d_C;				// gpu memory handle

void InitHostMem()
{
	PrintStep1("Init Host Memory");

	h_A = (float*)malloc(VECTOR_LEN * sizeof(float));
	h_B = (float*)malloc(VECTOR_LEN * sizeof(float));
	h_C = (float*)malloc(VECTOR_LEN * sizeof(float));

	for (unsigned int i = 0; i < VECTOR_LEN; i++)
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
	HIP_ASSERT(hipMalloc((void**)&d_A, VECTOR_LEN * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)&d_B, VECTOR_LEN * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)&d_C, VECTOR_LEN * sizeof(float)));

	PrintStep2("Copy Host Memory To Device Memory");
	HIP_ASSERT(hipMemcpy(d_A, h_A, VECTOR_LEN * sizeof(float), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(d_B, h_B, VECTOR_LEN * sizeof(float), hipMemcpyHostToDevice));
}
void FreeDeviceMem()
{
	PrintStep1("Free Device Memory");

	hipFree(d_A);
	hipFree(d_B);
	hipFree(d_C);
}

// ==========================================================================================
int HipDeviceCnt;					// device number on the hip platform
int HipDeviceId;					// used device index
hipDevice_t HipDevice;				// device handle
hipDeviceProp_t HipDeviceProp;		// device property
hipCtx_t HipContext;				// context handle
hipStream_t HipStream;				// stream handle
hipEvent_t HipStartEvt, HipStopEvt;	// event handle

void InitHipPlatform()
{
	PrintStep2("Init Hip Platform");

	HIP_ASSERT(hipInit(0));

	HIP_ASSERT(hipGetDeviceCount(&HipDeviceCnt));
	printf("    - Device Number: %d.\n", HipDeviceCnt);
}
void InitHipDevice()
{
	PrintStep2("Init Hip Device");

	HipDeviceId = 0;
	printf("    - Use Hip Device %d.\n", HipDeviceId);

	PrintStep2("Get Hip Device Info");
	HIP_ASSERT(hipGetDeviceProperties(&HipDeviceProp, HipDeviceId));

	printf("    - Device Name: %s.\n", HipDeviceProp.name);
	printf("    - GCN Arch: gfx%d.\n", HipDeviceProp.gcnArch);
	printf("    - Multi Processor Number: %d.\n", HipDeviceProp.multiProcessorCount);
	printf("    - Core Clock: %.3f(MHz).\n", HipDeviceProp.clockRate / 1000.0);
	printf("    - Memory Clock: %.3f(MHz).\n", HipDeviceProp.memoryClockRate / 1000.0);
	printf("    - Total Global Memory: %.3f(GB).\n", HipDeviceProp.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
	printf("    - Shared Memory per Block: %.3f(KB).\n", HipDeviceProp.sharedMemPerBlock / 1024.0);

	HIP_ASSERT(hipDeviceGet(&HipDevice, 0));
}
void InitHipContext()
{
	PrintStep2("Init Hip Context");

	HIP_ASSERT(hipCtxCreate(&HipContext, 0, HipDevice));
}
void InitHipStream()
{
	PrintStep2("Init Hip Stream");

	HIP_ASSERT(hipStreamCreate(&HipStream));
}
void InitHipEvent()
{
	PrintStep2("Init Hip Event");

	HIP_ASSERT(hipEventCreate(&HipStartEvt));
	HIP_ASSERT(hipEventCreate(&HipStopEvt));
}

void InitHipRuntime()
{
	PrintStep1("Init Hip Runtime");

	InitHipPlatform();
	InitHipDevice();
	InitHipStream();
	InitHipEvent();
}
void ReleaseRuntime()
{
	PrintStep1("Release Hip Runtime");

	hipEventDestroy(HipStartEvt);
	hipEventDestroy(HipStopEvt);

	hipStreamDestroy(HipStream);
}

// ==========================================================================================
string Compiler;
string BuildOption;
string CompileCmd;
string KernelName = "VectorAdd";
string KernelDir = "../";
string KernelSrcFile;
string KernelOutFile;
string KernelBinFile;
hipModule_t HipModule;				// module handle(kernel object)
hipFunction_t HipFunction;			// function handle(program object)

void CompileKernelFromHipFile()
{
	PrintStep2("Compile Hip Kernel File");

	Compiler = "/opt/rocm/bin/hipcc ";

	switch (HipDeviceProp.gcnArch)
	{
	case 803:BuildOption = "--genco --targets gfx803 "; break;
	case 900:BuildOption = "--genco --targets gfx900 "; break;
	case 906:BuildOption = "--genco --targets gfx906 "; break;
	case 908:BuildOption = "--genco --targets gfx908 "; break;
	default:printf("NOT Supportted Hardware.\n");
	}

	KernelSrcFile = KernelDir + KernelName + ".cpp";
	KernelBinFile = KernelDir + KernelName + ".bin";
	printf("    - kernel name = %s\n", KernelName.c_str());
	printf("    - kernel file = %s\n", KernelSrcFile.c_str());
	printf("    - bin file = %s\n", KernelBinFile.c_str());

	CompileCmd = Compiler + " " + BuildOption + "-o " + KernelBinFile + " " + KernelSrcFile;
	printf("    - Compile Command = %s\n", CompileCmd.c_str());
	ExecCommand(CompileCmd);
}
void CompileKernelFromAsmFile()
{
	PrintStep2("Compile Assembly Kernel File");
	
	Compiler = "/opt/rocm/bin/hcc ";

	switch (HipDeviceProp.gcnArch)
	{
	case 803:BuildOption = "-x assembler -target amdgcn-amd-amdhsa -mcpu=gfx803 -mno-code-object-v3 -c "; break;
	case 900:BuildOption = "-x assembler -target amdgcn-amd-amdhsa -mcpu=gfx900 -mno-code-object-v3 -c "; break;
	case 906:BuildOption = "-x assembler -target amdgcn-amd-amdhsa -mcpu=gfx906 -mno-code-object-v3 -c "; break;
	case 908:BuildOption = "-x assembler -target amdgcn-amd-amdhsa -mcpu=gfx908 -mno-code-object-v3 -c "; break;
	default:printf("NOT Supportted Hardware.\n");
	}

	KernelSrcFile = KernelDir + KernelName + ".s";
	KernelOutFile = KernelDir + KernelName + ".o";
	KernelBinFile = KernelDir + KernelName + ".bin";
	printf("    - kernel name = %s\n", KernelName.c_str());
	printf("    - kernel file = %s\n", KernelSrcFile.c_str());
	printf("    - out file = %s\n", KernelOutFile.c_str());
	printf("    - bin file = %s\n", KernelBinFile.c_str());

	CompileCmd = Compiler + " " + BuildOption + "-o " + KernelOutFile + " " + KernelSrcFile;
	printf("    - Compile Command = %s\n", CompileCmd.c_str());
	ExecCommand(CompileCmd);

	CompileCmd = Compiler + "-target amdgcn-amd-amdhsa " + KernelOutFile + " -o " + KernelBinFile;
	printf("    - Compile Command = %s\n", CompileCmd.c_str());
	ExecCommand(CompileCmd);
}
void LoadHipModule()
{
	PrintStep2("Load Hip Module");

	HIP_ASSERT(hipModuleLoad(&HipModule, KernelBinFile.c_str()));
}
void GetHipFunction()
{
	PrintStep2("Get Hip Function");

	HIP_ASSERT(hipModuleGetFunction(&HipFunction, HipModule, KernelName.c_str()));
}

void CreateKernel()
{
	PrintStep1("Create Kernel");
#ifdef HIP_KERNEL
	CompileKernelFromHipFile();
#endif
#ifdef ASM_KERNEL
	CompileKernelFromAsmFile();
#endif
	LoadHipModule();
	GetHipFunction();
}

// ==========================================================================================
#define WAVE_SIZE			(64)
#define SIMD_PER_CU			(4)
unsigned char * KernelArgsBuff;
size_t KernelArgsSize;
Dim3 GroupSize;
Dim3 GroupNum;
Dim3 GlobalSize;

void SetKernelArgs()
{
	PrintStep2("Setup Kernel Arguments");

	unsigned int vec_len = VECTOR_LEN;

	KernelArgsSize = 0;
	KernelArgsSize += sizeof(d_A);
	KernelArgsSize += sizeof(d_B);
	KernelArgsSize += sizeof(d_C);
	KernelArgsSize += sizeof(vec_len);

	KernelArgsBuff = (unsigned char*)malloc(KernelArgsSize);
	unsigned char * pBuff = KernelArgsBuff;
	memcpy(pBuff, &d_A, sizeof(d_A)); pBuff += sizeof(d_A);
	memcpy(pBuff, &d_B, sizeof(d_B)); pBuff += sizeof(d_B);
	memcpy(pBuff, &d_C, sizeof(d_C)); pBuff += sizeof(d_C);
	memcpy(pBuff, &vec_len, sizeof(vec_len));
}
void FreeKernelArgs()
{
	PrintStep2("Free Kernel Arguments");

	free(KernelArgsBuff);
}
void SetKernelWorkload()
{
	PrintStep2("Setup Kernel Workload");

	GroupSize.x = WAVE_SIZE * SIMD_PER_CU;
	GroupSize.y = 1; 
	GroupSize.z = 1;

	GroupNum.x = (VECTOR_LEN + GroupSize.x - 1) / GroupSize.x;
	GroupNum.y = 1;
	GroupNum.z = 1;

	GlobalSize.x = GroupSize.x * GroupNum.x;
	GlobalSize.y = GroupSize.y * GroupNum.y;
	GlobalSize.z = GroupSize.z * GroupNum.z;
}
void PrintWorkload()
{
	printf("    - Group Size = [%d, %d, %d].\n", GroupSize.x, GroupSize.y, GroupSize.z);
	printf("    - Group Number = [%d, %d, %d].\n", GroupNum.x, GroupNum.y, GroupNum.z);
	printf("    - Global Size = [%d, %d, %d].\n", GlobalSize.x, GlobalSize.y, GlobalSize.z);
}
void LaunchKernel()
{
	PrintStep2("Launch Kernel");

	void * config[] =
	{
		HIP_LAUNCH_PARAM_BUFFER_POINTER, 	KernelArgsBuff,
		HIP_LAUNCH_PARAM_BUFFER_SIZE,		&KernelArgsSize,
		HIP_LAUNCH_PARAM_END
	};

	HIP_ASSERT(
		hipExtModuleLaunchKernel(
		HipFunction,
		GlobalSize.x, GlobalSize.y, GlobalSize.z,
		GroupSize.x, GroupSize.y, GroupSize.z,
		0,
		0,
		NULL,
		(void**)&config,
		HipStartEvt,
		HipStopEvt));

	PrintStep2("Wait Kernel Complite");
	do
	{
	} while (hipEventQuery(HipStopEvt) != hipSuccess);
}

void RunGpuCalculation()
{
	PrintStep1("Run Gpu Kernel");

	SetKernelArgs();
	SetKernelWorkload();
	PrintWorkload();
	LaunchKernel();
	FreeKernelArgs();
}
void RunCpuCalculation()
{
	PrintStep1("Do Cpu Calculation");

	for (unsigned int i = 0; i < VECTOR_LEN; i++)
	{
		h_C[i] = h_A[i] + h_B[i];
	}

	//printf("\nHost Vector C\n");	PrintHostData(h_C);
}
void VerifyResult()
{
	PrintStep1("Verify GPU Result");

	float * dev_rslt = (float*)malloc(VECTOR_LEN * sizeof(float));

	PrintStep2("Copy Device Result To Host");
	HIP_ASSERT(hipMemcpy(dev_rslt, d_C, VECTOR_LEN * sizeof(float), hipMemcpyDeviceToHost));

	PrintStep2("Compare Device Result With Cpu Result");
	for (unsigned int i = 0; i < VECTOR_LEN; i++)
	{
		if (fabs(h_C[i] - dev_rslt[i]) > FLT_MIN)
		{
			printf("    - First Error:\n");
			printf("    - Host  : [%d] = %.2f.\n", i, h_C[i]);
			printf("    - Device: [%d] = %.2f.\n", i, dev_rslt[i]);
			break;
		}

		if (i == VECTOR_LEN - 1)
		{
			printf("    - Verify Success.\n");
		}
	}
}

// ==========================================================================================
float KernelExecMs;					// kernel elapsed time in ms
void LaunchKernelElapsed()
{
	void * config[] =
	{
		HIP_LAUNCH_PARAM_BUFFER_POINTER, 	KernelArgsBuff,
		HIP_LAUNCH_PARAM_BUFFER_SIZE,		&KernelArgsSize,
		HIP_LAUNCH_PARAM_END
	}; 

	hipExtModuleLaunchKernel(
		HipFunction,
		GlobalSize.x, GlobalSize.y, GlobalSize.z,
		GroupSize.x, GroupSize.y, GroupSize.z,
		0,
		0,
		NULL,
		(void**)&config,
		HipStartEvt,
		HipStopEvt);

	hipDeviceSynchronize();
	HIP_ASSERT(hipEventElapsedTime(&KernelExecMs, HipStartEvt, HipStopEvt));
}
void TestEfficiency()
{
	PrintStep1("Test Gpu Kernel Efficiency");

	SetKernelArgs();
	SetKernelWorkload();

	PrintStep2("Warmup");
	LaunchKernelElapsed();

	PrintStep2("Run GpuKernel for " + to_string(ITERATION_TIMES) + " times");
	double elapsed_ms = 0;
	for (unsigned int i = 0; i < ITERATION_TIMES; i++)
	{
		LaunchKernelElapsed();
		elapsed_ms += KernelExecMs;
	}
	printf("    - Kernel Elapsed Time = %.3f(ms).\n", elapsed_ms / ITERATION_TIMES);

	FreeKernelArgs();
}

// ==========================================================================================
int main(int argc, char *argv[])
{
	printf("\nHello ROCM.\n\n");
	InitHipRuntime();

	InitHostMem();
	InitDeviceMem();
	CreateKernel();

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
