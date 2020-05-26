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
void Logout(string s)
{
	printf("%s.\n", s.c_str());
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
		fclk = clk *1.0f;
		sprintf(cbuff, "%.2f Hz", fclk);
		return string(cbuff);
	}
}

// ==========================================================================================
void InitHipDevice(int devId)
{
	printf("  ---------------------------\n");
	PrintStep2("Get Hip Device " + to_string(devId) + " Info");

	hipDeviceProp_t deviceProp;	// device property
	HIP_ASSERT(hipGetDeviceProperties(&deviceProp, devId));

	printf("    - Device Name: %s.\n", deviceProp.name);
	printf("    - GCN Arch: %d.\n", deviceProp.gcnArch);
	printf("    - %s.\n", deviceProp.integrated ? "APU" : "dGPU");
	printf("    - Is Multi-Gpu Board: %s.\n", deviceProp.isMultiGpuBoard ? "TRUE" : "FALSE");
	printf("\n");
	Logout("    - Core Clock: " + FormatFreq(deviceProp.clockRate * 1000));
	printf("    - Multi Processor(CU) Number: %d.\n", deviceProp.multiProcessorCount);
	Logout("    - Device Timer clock() Frequency: " + FormatFreq(deviceProp.clockInstructionRate * 1000));
	printf("    - Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
	printf("    - Compute Mode: %d.\n", deviceProp.computeMode);
	printf("    - Support Concurrent Kernels: %s.\n", deviceProp.concurrentKernels ? "TRUE" : "FALSE");
	printf("    - Support Cooperative Launch: %s.\n", deviceProp.cooperativeLaunch ? "TRUE" : "FALSE");
	printf("    - Support Cooperative Launch on Mult-devices: %s.\n", deviceProp.cooperativeMultiDeviceLaunch ? "TRUE" : "FALSE");
	printf("    - Support Runtime Limit for Kernels Executed: %s.\n", deviceProp.kernelExecTimeoutEnabled ? "TRUE" : "FALSE");
	printf("    - Support ECC: %s.\n", deviceProp.ECCEnabled ? "TRUE" : "FALSE");
	printf("    - Support TCC(only for Tesla): %s.\n", deviceProp.tccDriver ? "TRUE" : "FALSE");
	printf("    - Architectural Feature Flags: %X.\n", deviceProp.arch);
	printf("        - 32-bit Atomics.\n");
	printf("        - Support Global Memory Int32 Atomic: %s.\n", deviceProp.arch.hasGlobalInt32Atomics ? "TRUE" : "FALSE");
	printf("        - Support Global Memory float Atomic: %s.\n", deviceProp.arch.hasGlobalFloatAtomicExch ? "TRUE" : "FALSE");
	printf("        - Support Shared Memory Int32 Atomic: %s.\n", deviceProp.arch.hasSharedInt32Atomics ? "TRUE" : "FALSE");
	printf("        - Support Shared Memory float Atomic: %s.\n", deviceProp.arch.hasSharedFloatAtomicExch ? "TRUE" : "FALSE");
	printf("        - Support Global and Shared Memory float Atomic: %s.\n", deviceProp.arch.hasFloatAtomicAdd ? "TRUE" : "FALSE");
	printf("        - 64-bit Atomics.\n");
	printf("        - Support Global Memory Int64 Atomic: %s.\n", deviceProp.arch.hasGlobalInt64Atomics ? "TRUE" : "FALSE");
	printf("        - Support Shared Memory Int64 Atomic: %s.\n", deviceProp.arch.hasSharedInt64Atomics ? "TRUE" : "FALSE");
	printf("        - Doubles.\n");
	printf("        - Support Double Precision: %s.\n", deviceProp.arch.hasDoubles ? "TRUE" : "FALSE");
	printf("        - Warp cross-lane operations.\n");
	printf("        - Warp vote instructions (__any, __all): %s.\n", deviceProp.arch.hasWarpVote ? "TRUE" : "FALSE");
	printf("        - Warp ballot instructions (__ballot): %s.\n", deviceProp.arch.hasWarpBallot ? "TRUE" : "FALSE");
	printf("        - Warp shuffle operations. (__shfl_*): %s.\n", deviceProp.arch.hasWarpShuffle ? "TRUE" : "FALSE");
	printf("        - Funnel two words into one with shift&mask caps: %s.\n", deviceProp.arch.hasFunnelShift ? "TRUE" : "FALSE");
	printf("        - Sync.\n");
	printf("        - __threadfence_system: %s.\n", deviceProp.arch.hasThreadFenceSystem ? "TRUE" : "FALSE");
	printf("        - __syncthreads_count, syncthreads_and, syncthreads_or: %s.\n", deviceProp.arch.hasSyncThreadsExt ? "TRUE" : "FALSE");
	printf("        - Misc.\n");
	printf("        - Support Surface Functions: %s.\n", deviceProp.arch.hasSurfaceFuncs ? "TRUE" : "FALSE");
	printf("        - Grid and group dims are 3D (rather than 2D): %s.\n", deviceProp.arch.has3dGrid ? "TRUE" : "FALSE");
	printf("        - Support Dynamic Parallelism: %s.\n", deviceProp.arch.hasDynamicParallelism ? "TRUE" : "FALSE");
	printf("\n");
	Logout("    - Max Global Memory Clock: " + FormatFreq(deviceProp.memoryClockRate * 1000));
	printf("    - Global Memory Bus Width: %d(bit).\n", deviceProp.memoryBusWidth);
	Logout("    - Total Global Memory: " + FormatSize(deviceProp.totalGlobalMem));
	Logout("    - Shared Memory per Block: " + FormatSize(deviceProp.sharedMemPerBlock));
	Logout("    ? Max Shared Memory Per CU: " + FormatSize(deviceProp.maxSharedMemoryPerMultiProcessor));
	Logout("    - Total Constant Memory: " + FormatSize(deviceProp.totalConstMem));
	printf("    ? Registers per Block: %d.\n", deviceProp.regsPerBlock);
	Logout("    ? L2 Cache Size: " + FormatSize(deviceProp.l2CacheSize));
	printf("    - Can Hip Map Host Memory: %s.\n", deviceProp.canMapHostMemory ? "TRUE" : "FALSE");
	Logout("    - Max Pitch Allowed by MemCopy: " + FormatSize(deviceProp.memPitch));
	Logout("    - Alignment Requirement for Textures: " + FormatSize(deviceProp.textureAlignment));
	printf("\n");
	printf("    - Warp size: %d.\n", deviceProp.warpSize);
	printf("    - Max Threads per Block: %d.\n", deviceProp.maxThreadsPerBlock);
	printf("    - Max Threads per CU: %d.\n", deviceProp.maxThreadsPerMultiProcessor);
	printf("    - Max Threads dim: (%d, %d, %d).\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("    - Max Grid Size: (%d, %d, %d).\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("    - Max Number of Elements of 1D Images: %d.\n", deviceProp.maxTexture1D);
	printf("    - Max Width&Height of Elements of 2D Images: (%d,%d).\n", deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
	printf("    - Max Width&Height&Depth of Elements of 3D Images: (%d,%d).\n", deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
	printf("\n");
	printf("    - PCI DomainID Rate: %d.\n", deviceProp.pciDomainID);
	printf("    - PCI BusID: %d.\n", deviceProp.pciBusID);
	printf("    - PCI DeviceID: %d.\n", deviceProp.pciDeviceID);
	printf("\n");
	printf("    - HDP_MEM_COHERENCY_FLUSH_CNTL Address: 0x%08X.\n", deviceProp.hdpMemFlushCntl);
	printf("    - HDP_REG_COHERENCY_FLUSH_CNTL Address: 0x%08X.\n", deviceProp.hdpRegFlushCntl);
}
void InitHipPlatform()
{
	PrintStep1("Get Hip Platform Info");

	HIP_ASSERT(hipInit(0));

	int deviceCnt;
	HIP_ASSERT(hipGetDeviceCount(&deviceCnt));
	printf("    - Device Count: %d.\n", deviceCnt);

	PrintStep1("Get Hip Devices Info");
	for (int devId = 0; devId < deviceCnt; devId++)
	{
		InitHipDevice(devId);
	}
}
void InitHipRuntime()
{
	PrintStep1("Get Hip Runtime Info"); 
	
	int runtimeVersion;
	HIP_ASSERT(hipRuntimeGetVersion(&runtimeVersion));
	printf("    - Runtime Version: %d.\n", runtimeVersion);

	InitHipPlatform();
}
void ReleaseRuntime()
{
	PrintStep1("Release Hip Runtime");
}

// ==========================================================================================
int main(int argc, char *argv[])
{
	printf("\nHello ROCM.\n\n");
	InitHipRuntime();
	ReleaseRuntime();
	printf("\nByeBye ROCM.\n\n");

	return 0;
}
