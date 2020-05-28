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
	KernelBinFile = KernelDir + KernelName + ".bin";
	printf("    - kernel name = %s\n", KernelName.c_str());
	printf("    - kernel file = %s\n", KernelSrcFile.c_str());
	printf("    - bin file = %s\n", KernelBinFile.c_str());

	CompileKernelFromHipFile();
	LoadHipModule();
	GetHipFunction();
}
void CreateAsmKernel(string kernelName)
{
	PrintStep1("Create Kernel");

	KernelName = kernelName;
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
}
void SetGlobalSize(uint32_t x, uint32_t y, uint32_t z)
{
	GlobalSize.x = x;
	GlobalSize.y = y;
	GlobalSize.z = z;

	GroupNum.x = GlobalSize.x / GroupSize.x;
	GroupNum.y = GlobalSize.y / GroupSize.y;
	GroupNum.z = GlobalSize.z / GroupSize.z;
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
