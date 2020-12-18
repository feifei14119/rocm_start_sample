#include "utils.h"

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
void Logout(string s)
{
	printf("%s.\n", s.c_str());
}
void PrintHostData(float * pData, uint32_t len)
{
	unsigned int col_num = 8;
	for (unsigned int i = 0; i < len; i++)
	{
		if (i % col_num == 0)
		{
			printf("[%03d~%03d]: ", i, i + col_num - 1);
		}
		printf("%.2f, ", *(pData + i));
		if ((i + 1) % col_num == 0)
		{
			printf("\n");
		}
	}

	if (len % 8 != 0)
		printf("\n");
}
void PrintHostData(float2 * pData, uint32_t len)
{
	unsigned int col_num = 8;
	for (unsigned int i = 0; i < len; i++)
	{
		if (i % col_num == 0)
		{
			printf("[%03d~%03d]: ", i, i + col_num - 1);
		}
		printf("<%.1f, %.1f> ", (*(pData + i)).x, (*(pData + i)).y);
		if ((i + 1) % col_num == 0)
		{
			printf("\n");
		}
	}

	if (len % 8 != 0)
		printf("\n");
}
void PrintDeviceData(float * pData, uint32_t len)
{
	float * h_data = (float*)malloc(len * sizeof(float));
	HIP_ASSERT(hipMemcpy(h_data, pData, len * sizeof(float), hipMemcpyDeviceToHost));

	unsigned int col_num = 8;
	for (unsigned int i = 0; i < len; i++)
	{
		if (i % col_num == 0)
		{
			printf("[%03d~%03d]: ", i, i + col_num - 1);
		}
		printf("%.2f, ", h_data[i]);
		if ((i + 1) % col_num == 0)
		{
			printf("\n");
		}
	}

	if (len % 8 != 0)
		printf("\n");

	free(h_data);
}
void PrintDeviceData(float2 * pData, uint32_t len)
{
	float2 * h_data = (float2*)malloc(len * sizeof(float2));
	HIP_ASSERT(hipMemcpy(h_data, pData, len * sizeof(float2), hipMemcpyDeviceToHost));

	unsigned int col_num = 8;
	for (unsigned int i = 0; i < len; i++)
	{
		if (i % col_num == 0)
		{
			printf("[%03d~%03d]: ", i, i + col_num - 1);
		}
		printf("<%.1f, %.1f> ", (h_data[i]).x, (h_data[i]).y);
		if ((i + 1) % col_num == 0)
		{
			printf("\n");
		}
	}

	if (len % 8 != 0)
		printf("\n");

	free(h_data);
}
void PrintHostData(int * pData, uint32_t len)
{
	unsigned int col_num = 8;
	for (unsigned int i = 0; i < len; i++)
	{
		if (i % col_num == 0)
		{
			printf("[%03d~%03d]: ", i, i + col_num - 1);
		}
		printf("%03d, ", *(pData + i));
		if ((i + 1) % col_num == 0)
		{
			printf("\n");
		}
	}

	if(len % 8 != 0)
		printf("\n");
}
void PrintDeviceData(int * pData, uint32_t len)
{
	int * h_data = (int*)malloc(len * sizeof(int));
	HIP_ASSERT(hipMemcpy(h_data, pData, len * sizeof(int), hipMemcpyDeviceToHost));

	unsigned int col_num = 8;
	for (unsigned int i = 0; i < len; i++)
	{
		if (i % col_num == 0)
		{
			printf("[%03d~%03d]: ", i, i + col_num - 1);
		}
		printf("%03d, ", h_data[i]);
		if ((i + 1) % col_num == 0)
		{
			printf("\n");
		}
	}

	if (len % 8 != 0)
		printf("\n");

	free(h_data);
}
void CompareData(float * h_data, float * d_data, uint32_t len)
{
	PrintStep1("Verify GPU Result");

	float * dev_rslt = (float*)malloc(len * sizeof(float));

	PrintStep2("Copy Device Result To Host");
	HIP_ASSERT(hipMemcpy(dev_rslt, d_data, len * sizeof(float), hipMemcpyDeviceToHost));

	PrintStep2("Compare Device Result With Cpu Result");
	for (unsigned int i = 0; i < len; i++)
	{
		if (fabs(h_data[i] - dev_rslt[i]) > FLT_MIN)
		{
			printf("    - First Error:\n");
			printf("    - Host  : [%d] = %.2f.\n", i, h_data[i]);
			printf("    - Device: [%d] = %.2f.\n", i, dev_rslt[i]);
			break;
		}

		if (i == len - 1)
		{
			printf("    - Verify Success.\n");
		}
	}

	free(dev_rslt);
}
void CompareData(double * h_data,  double* d_data, uint32_t len)
{
	PrintStep1("Verify GPU Result");

	double * dev_rslt = (double*)malloc(len * sizeof(double));

	PrintStep2("Copy Device Result To Host");
	HIP_ASSERT(hipMemcpy(dev_rslt, d_data, len * sizeof(double), hipMemcpyDeviceToHost));

	PrintStep2("Compare Device Result With Cpu Result");
	for (unsigned int i = 0; i < len; i++)
	{
		if (fabs(h_data[i] - dev_rslt[i]) > FLT_MIN)
		{
			printf("    - First Error:\n");
			printf("    - Host  : [%d] = %.2f.\n", i, h_data[i]);
			printf("    - Device: [%d] = %.2f.\n", i, dev_rslt[i]);
			break;
		}

		if (i == len - 1)
		{
			printf("    - Verify Success.\n");
		}
	}

	free(dev_rslt);
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

string FormatSize(size_t sz)
{
	float fsz;
	char cbuff[1024];
	if (sz >= 1024 * 1024 * 1024)
	{
		fsz = sz / 1024.0f / 1024.0f / 1024.0f;
		sprintf(cbuff, "%.2f GB", fsz);
		return string(cbuff);
	}
	if (sz >= 1024 * 1024)
	{
		fsz = sz / 1024.0f / 1024.0f;
		sprintf(cbuff, "%.2f MB", fsz);
		return string(cbuff);
	}
	if (sz >= 1024)
	{
		fsz = sz / 1024.0f;
		sprintf(cbuff, "%.2f KB", fsz);
		return string(cbuff);
	}
	else
	{
		fsz = sz * 1.0f;
		sprintf(cbuff, "%.2f Byte", fsz);
		return string(cbuff);
	}
}
string FormatFreq(int clk)
{
	float fclk;
	char cbuff[1024];
	if (clk >= 1000 * 1000 * 1000)
	{
		fclk = clk / 1000.0f / 1000.0f / 1000.0f;
		sprintf(cbuff, "%.2f GHz", fclk);
		return string(cbuff);
	}
	if (clk >= 1000 * 1000)
	{
		fclk = clk / 1000.0f / 1000.0f;
		sprintf(cbuff, "%.2f MHz", fclk);
		return string(cbuff);
	}
	if (clk >= 1000)
	{
		fclk = clk / 1000.0f;
		sprintf(cbuff, "%.2f KHz", fclk);
		return string(cbuff);
	}
	else
	{
		fclk = clk * 1.0f;
		sprintf(cbuff, "%.2f Hz", fclk);
		return string(cbuff);
	}
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
string KernelName;
string KernelDir = "../";
string BuildDir = "./";
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
	case 803:BuildOption = "--genco -mcpu=gfx803 "; break;
	case 900:BuildOption = "--genco -mcpu=gfx900 "; break;
	case 906:BuildOption = "--genco -mcpu=gfx906 "; break;
	case 908:BuildOption = "--genco -mcpu=gfx908 "; break;
	default:printf("NOT Supportted Hardware.\n");
	}

	CompileCmd = Compiler + " " + BuildOption + "-o " + KernelBinFile + " " + KernelSrcFile;
	printf("    - Compile Command = %s\n", CompileCmd.c_str());
	ExecCommand(CompileCmd);
}
void CompileKernelFromAsmFile()
{
	PrintStep2("Compile Assembly Kernel File");

#ifdef CMP_LLVM
	Compiler = "/opt/rocm/llvm/bin/clang++ ";	// only support from rocm 3.5+
#else
	Compiler = "/opt/rocm/bin/hcc ";			// for object v2 and rocm 3.5-
#endif

	switch (HipDeviceProp.gcnArch)
	{
	case 803:BuildOption = "-x assembler -target amdgcn-amd-amdhsa -mcpu=gfx803 -save-temps "; break;
	case 900:BuildOption = "-x assembler -target amdgcn-amd-amdhsa -mcpu=gfx900 -save-temps "; break;
	case 906:BuildOption = "-x assembler -target amdgcn-amd-amdhsa -mcpu=gfx906 -save-temps "; break;
	case 908:BuildOption = "-x assembler -target amdgcn-amd-amdhsa -mcpu=gfx908 -save-temps "; break;
	default:printf("NOT Supportted Hardware.\n");
	}

#ifndef OBJ_V3
	BuildOption = BuildOption + "-mno-code-object-v3 "; // for object v2
#endif

	CompileCmd = Compiler + BuildOption + KernelSrcFile + " -o " + KernelBinFile;
	printf("    - Compile Command = %s\n", CompileCmd.c_str());
	ExecCommand(CompileCmd);
}
void LoadHipModule()
{
	PrintStep2("Load Hip Module");

	HIP_ASSERT(hipModuleLoad(&HipModule, KernelBinFile.c_str()));
	printf("kernel bin file = %s\n",KernelBinFile.c_str());
}
void GetHipFunction()
{
	PrintStep2("Get Hip Function");

	int err = hipModuleGetFunction(&HipFunction, HipModule, KernelName.c_str());
	printf("kernel name = %s\n",KernelName.c_str());
	printf("err = %d\n",err);
	HIP_ASSERT(err);
	HIP_ASSERT(hipModuleGetFunction(&HipFunction, HipModule, KernelName.c_str()));
}

void CreateHipKernel(string kernelName, string kernelFile)
{
	PrintStep1("Create Kernel");

	KernelName = kernelName;
	if (kernelFile == "")
	{
		KernelSrcFile = KernelDir + KernelName + ".cpp";
	}
	else
	{
		KernelSrcFile = KernelDir + kernelFile;
	}
	KernelBinFile = BuildDir + KernelName + ".bin";
	printf("    - kernel name = %s\n", KernelName.c_str());
	printf("    - kernel file = %s\n", KernelSrcFile.c_str());
	printf("    - bin file = %s\n", KernelBinFile.c_str());

	CompileKernelFromHipFile();
	LoadHipModule();
	GetHipFunction();
}
void CreateAsmKernel(string kernelName, string kernelFile)
{
	PrintStep1("Create Kernel");

	KernelName = kernelName;
	if (kernelFile == "")
	{
		KernelSrcFile = KernelDir + KernelName + ".s";
	}
	else
	{
		KernelSrcFile = KernelDir + kernelFile;
	}
	KernelOutFile = BuildDir + KernelName + ".o";
	KernelBinFile = BuildDir + KernelName + ".bin";
	printf("    - kernel name = %s\n", KernelName.c_str());
	printf("    - kernel file = %s\n", KernelSrcFile.c_str());
	printf("    - out file = %s\n", KernelOutFile.c_str());
	printf("    - bin file = %s\n", KernelBinFile.c_str());

	CompileKernelFromAsmFile();
	LoadHipModule();
	GetHipFunction();
}

// ==========================================================================================
#define KERNEL_ARGS_BUFF_MAX_LEN (1024)
unsigned char KernelArgsBuff[KERNEL_ARGS_BUFF_MAX_LEN];
size_t KernelArgsSize = 0;
uint32_t KernelArgsNum = 0;
void PrintKernelArgs()
{
	printf("    - Kerenl Args Number = %d.\n", KernelArgsNum);
	printf("    - Kerenl Args Size = %d.\n", KernelArgsSize);
}
void FreeKernelArgs()
{
	PrintStep2("Free Kernel Arguments");

	//free(KernelArgsBuff);
	KernelArgsSize = 0;
	KernelArgsNum = 0;
}

dim3 GroupSize;
dim3 GroupNum;
dim3 GlobalSize;
void SetGroupSize(uint32_t x, uint32_t y, uint32_t z)
{
	GroupSize.x = x;
	GroupSize.y = y;
	GroupSize.z = z;
}
void SetGroupNum(uint32_t x, uint32_t y, uint32_t z)
{
	GroupNum.x = x;
	GroupNum.y = y;
	GroupNum.z = z;

	GlobalSize.x = GroupSize.x * GroupNum.x;
	GlobalSize.y = GroupSize.y * GroupNum.y;
	GlobalSize.z = GroupSize.z * GroupNum.z;

	printf("group  size = [%d, %d, %d]\n", GroupSize.x, GroupSize.y, GroupSize.z);
	printf("group  num  = [%d, %d, %d]\n", GroupNum.x, GroupNum.y, GroupNum.z);
	printf("global size = [%d, %d, %d]\n", GlobalSize.x, GlobalSize.y, GlobalSize.z);
}
void SetGlobalSize(uint32_t x, uint32_t y, uint32_t z)
{
	GlobalSize.x = x;
	GlobalSize.y = y;
	GlobalSize.z = z;

	GroupNum.x = (GlobalSize.x + GroupSize.x - 1) / GroupSize.x;
	GroupNum.y = (GlobalSize.y + GroupSize.y - 1) / GroupSize.y;
	GroupNum.z = (GlobalSize.z + GroupSize.z - 1) / GroupSize.z;

	SetGroupNum(GroupNum.x, GroupNum.y, GroupNum.z); return;

	printf("group  size = [%d, %d, %d]\n", GroupSize.x, GroupSize.y, GroupSize.z);
	printf("group  num  = [%d, %d, %d]\n", GroupNum.x, GroupNum.y, GroupNum.z);
	printf("global size = [%d, %d, %d]\n", GlobalSize.x, GlobalSize.y, GlobalSize.z);
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
float LaunchKernelGetElapsedMs()
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
	float elapsedMs;
	HIP_ASSERT(hipEventElapsedTime(&elapsedMs, HipStartEvt, HipStopEvt));

	return elapsedMs;
}
