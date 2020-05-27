#include <float.h>
#include <math.h>
#include <string>
#include <hip/hip_hcc.h>

using namespace std;

#define ITERATION_TIMES		(100)

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

void InitHipRuntime()
{
	PrintStep1("Init Hip Runtime");

	InitHipPlatform();
	InitHipDevice();
}
void ReleaseRuntime()
{
}

// ==========================================================================================
float *h_A, *h_B;				// cpu memory handle
float *d_A, *d_B;				// gpu memory handle
size_t TestMemByteSize[] = { 512,
	1024,1024 * 2,1024 * 4,1024 * 8,1024 * 16,
	1024 * 32, 1024 * 64,1024 * 128,1024 * 256,1024 * 512,
	1024 * 1024,1024 * 1024 * 2,1024 * 1024 * 4,1024 * 1024 * 8,1024 * 1024 * 16,
	1024 * 1024 * 32,1024 * 1024 * 64,1024 * 1024 * 128,1024 * 1024 * 256,1024 * 1024 * 512,
	1024 * 1024 * 1024 };
float CopyElapsedMs;
float BandwidthMBperSec;
void TestPinnedH2DBandwidth(size_t sz)
{
	HIP_ASSERT(hipHostMalloc((void**)&h_A, sz));
	HIP_ASSERT(hipMalloc((void**)&d_A, sz));

	hipEvent_t start, stop;
	HIP_ASSERT(hipEventCreate(&start));
	HIP_ASSERT(hipEventCreate(&stop));

	hipEventRecord(start, 0);
	for (int i = 0; i < ITERATION_TIMES; i++)
	{
		HIP_ASSERT(hipMemcpyAsync(d_A, h_A, sz, hipMemcpyHostToDevice, NULL));
	}
	hipEventRecord(stop, 0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&CopyElapsedMs, start, stop);

	CopyElapsedMs /= ITERATION_TIMES;
	BandwidthMBperSec = sz / (CopyElapsedMs / 1000.0) / 1024 / 1024;

	printf("    - %s:    %.2f(ms),     %.3f(MB/s)\n", FormatSize(sz).c_str(), CopyElapsedMs, BandwidthMBperSec);
	
	hipHostFree((void*)h_A);
	hipFree(d_A);
	hipEventDestroy(start);
	hipEventDestroy(stop);
}
void TestPinnedD2HBandwidth(size_t sz)
{
	HIP_ASSERT(hipHostMalloc((void**)&h_A, sz));
	HIP_ASSERT(hipMalloc((void**)&d_A, sz));

	hipEvent_t start, stop;
	HIP_ASSERT(hipEventCreate(&start));
	HIP_ASSERT(hipEventCreate(&stop));

	hipEventRecord(start, 0);
	for (int i = 0; i < ITERATION_TIMES; i++)
	{
		HIP_ASSERT(hipMemcpyAsync(h_A, d_A, sz, hipMemcpyDeviceToHost, NULL));
	}
	hipEventRecord(stop, 0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&CopyElapsedMs, start, stop);

	CopyElapsedMs /= ITERATION_TIMES;
	BandwidthMBperSec = sz / (CopyElapsedMs / 1000.0) / 1024 / 1024;

	printf("    - %s:    %.2f(ms),     %.3f(MB/s)\n", FormatSize(sz).c_str(), CopyElapsedMs, BandwidthMBperSec);

	hipHostFree((void*)h_A);
	hipFree(d_A);
	hipEventDestroy(start);
	hipEventDestroy(stop);
}
void TestPageableH2DBandwidth(size_t sz)
{
	h_A = (float*)malloc(sz);
	HIP_ASSERT(hipMalloc((void**)&d_A, sz));

	hipEvent_t start, stop;
	HIP_ASSERT(hipEventCreate(&start));
	HIP_ASSERT(hipEventCreate(&stop));
	
	hipEventRecord(start, 0);
	for (int i = 0; i < ITERATION_TIMES; i++)
	{
		HIP_ASSERT(hipMemcpy(d_A, h_A, sz, hipMemcpyHostToDevice));
	}
	hipEventRecord(stop, 0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&CopyElapsedMs, start, stop);
	
	CopyElapsedMs /= ITERATION_TIMES;
	BandwidthMBperSec = sz / (CopyElapsedMs / 1000.0) / 1024 / 1024;

	printf("    - %s:    %.2f(ms),     %.3f(MB/s)\n", FormatSize(sz).c_str(), CopyElapsedMs, BandwidthMBperSec);

	free(h_A);
	hipFree(d_A);
	hipEventDestroy(start);
	hipEventDestroy(stop);
}
void TestPageableD2HBandwidth(size_t sz)
{
	h_A = (float*)malloc(sz);
	HIP_ASSERT(hipMalloc((void**)&d_A, sz));

	hipEvent_t start, stop;
	HIP_ASSERT(hipEventCreate(&start));
	HIP_ASSERT(hipEventCreate(&stop));

	hipEventRecord(start, 0);
	for (int i = 0; i < ITERATION_TIMES; i++)
	{
		HIP_ASSERT(hipMemcpy(h_A, d_A, sz, hipMemcpyDeviceToHost));
	}
	hipEventRecord(stop, 0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&CopyElapsedMs, start, stop);

	CopyElapsedMs /= ITERATION_TIMES;
	BandwidthMBperSec = sz / (CopyElapsedMs / 1000.0) / 1024 / 1024;

	printf("    - %s:    %.2f(ms),     %.3f(MB/s)\n", FormatSize(sz).c_str(), CopyElapsedMs, BandwidthMBperSec);

	free(h_A);
	hipFree(d_A);
	hipEventDestroy(start);
	hipEventDestroy(stop);
}
void TestD2DBandwidth(size_t sz)
{
	HIP_ASSERT(hipMalloc((void**)&d_A, sz));
	HIP_ASSERT(hipMalloc((void**)&d_B, sz));

	hipEvent_t start, stop;
	HIP_ASSERT(hipEventCreate(&start));
	HIP_ASSERT(hipEventCreate(&stop));

	hipEventRecord(start, 0);
	for (int i = 0; i < ITERATION_TIMES; i++)
	{
		HIP_ASSERT(hipMemcpy(d_B, d_A, sz, hipMemcpyDeviceToDevice));
	}
	hipEventRecord(stop, 0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&CopyElapsedMs, start, stop);

	CopyElapsedMs /= ITERATION_TIMES;
	BandwidthMBperSec = sz / (CopyElapsedMs / 1000.0) / 1024 / 1024;

	printf("    - %s:    %.2f(ms),     %.3f(MB/s)\n", FormatSize(sz).c_str(), CopyElapsedMs, BandwidthMBperSec);

	hipFree(d_A);
	hipFree(d_B);
	hipEventDestroy(start);
	hipEventDestroy(stop);
}
void TestBandwidth()
{
	uint32_t sz_num = sizeof(TestMemByteSize) / sizeof(size_t);

	// -------------------------------------------------------------
	size_t rcdSzPageH2D;
	float rcdTimePageH2D, rcdBandwidthPageH2D;
	rcdTimePageH2D = rcdBandwidthPageH2D = 0.0f;
	printf("\n-------------------------------------------------\n");
	printf("Test Pageable Memory Copy Bandwidth Host -> Device:\n");
	for (uint32_t i = 0; i < sz_num; i++)
	{
		size_t sz = TestMemByteSize[i];
		TestPageableH2DBandwidth(sz);

		if (BandwidthMBperSec > rcdBandwidthPageH2D)
		{
			rcdTimePageH2D = CopyElapsedMs;
			rcdBandwidthPageH2D = BandwidthMBperSec;
			rcdSzPageH2D = sz;
		}
	}

	// -------------------------------------------------------------
	size_t rcdSzPageD2H;
	float rcdTimePageD2H, rcdBandwidthPageD2H;
	rcdTimePageD2H = rcdBandwidthPageD2H = 0.0f;
	rcdTimePageD2H = rcdBandwidthPageD2H = 0.0f;
	printf("\n-------------------------------------------------\n");
	printf("Test Pageable Memory Copy Bandwidth Device -> Host:\n");
	for (uint32_t i = 0; i < sz_num; i++)
	{
		size_t sz = TestMemByteSize[i];
		TestPageableD2HBandwidth(sz);

		if (BandwidthMBperSec > rcdBandwidthPageD2H)
		{
			rcdTimePageD2H = CopyElapsedMs;
			rcdBandwidthPageD2H = BandwidthMBperSec;
			rcdSzPageD2H = sz;
		}
	}

	// -------------------------------------------------------------
	size_t rcdSzPinH2D;
	float rcdTimePinH2D, rcdBandwidthPinH2D;
	rcdTimePinH2D = rcdBandwidthPinH2D = 0.0f;
	printf("\n-------------------------------------------------\n");
	printf("Test Pinned Memory Copy Bandwidth Host -> Device:\n");
	for (uint32_t i = 0; i < sz_num; i++)
	{
		size_t sz = TestMemByteSize[i];
		TestPinnedH2DBandwidth(sz);

		if (BandwidthMBperSec > rcdBandwidthPinH2D)
		{
			rcdTimePinH2D = CopyElapsedMs;
			rcdBandwidthPinH2D = BandwidthMBperSec;
			rcdSzPinH2D = sz;
		}
	}

	// -------------------------------------------------------------
	size_t rcdSzPinD2H;
	float rcdTimePinD2H, rcdBandwidthPinD2H;
	rcdTimePinD2H = rcdBandwidthPinD2H = 0.0f;
	printf("\n-------------------------------------------------\n");
	printf("Test Pinned Memory Copy Bandwidth Device -> Host:\n");
	for (uint32_t i = 0; i < sz_num; i++)
	{
		size_t sz = TestMemByteSize[i];
		TestPinnedD2HBandwidth(sz);

		if (BandwidthMBperSec > rcdBandwidthPinD2H)
		{
			rcdTimePinD2H = CopyElapsedMs;
			rcdBandwidthPinD2H = BandwidthMBperSec;
			rcdSzPinD2H = sz;
		}
	}

	// -------------------------------------------------------------
	size_t rcdSzD2D;
	float rcdTimeD2D, rcdBandwidthD2D;
	rcdTimeD2D = rcdBandwidthD2D = 0.0f;
	printf("\n-------------------------------------------------\n");
	printf("Test Memory Copy Bandwidth Device -> Device:\n");
	for (uint32_t i = 0; i < sz_num; i++)
	{
		size_t sz = TestMemByteSize[i];
		TestD2DBandwidth(sz);

		if (BandwidthMBperSec > rcdBandwidthD2D)
		{
			rcdTimeD2D = CopyElapsedMs;
			rcdBandwidthD2D = BandwidthMBperSec;
			rcdSzD2D = sz;
		}
	}

	// -------------------------------------------------------------
	printf("\n=================================================\n");
	printf("PAGEABLE H2D: %s:    %.2f(ms),     %.3f(MB/s)\n", FormatSize(rcdSzPageH2D).c_str(), rcdTimePageH2D, rcdBandwidthPageH2D);
	printf("PAGEABLE D2H: %s:    %.2f(ms),     %.3f(MB/s)\n", FormatSize(rcdSzPageD2H).c_str(), rcdTimePageD2H, rcdBandwidthPageD2H);
	printf("PINNED   H2D: %s:    %.2f(ms),     %.3f(MB/s)\n", FormatSize(rcdSzPinH2D).c_str(), rcdTimePinH2D, rcdBandwidthPinH2D);
	printf("PINNED   D2H: %s:    %.2f(ms),     %.3f(MB/s)\n", FormatSize(rcdSzPinD2H).c_str(), rcdTimePinD2H, rcdBandwidthPinD2H);
	printf("         D2D: %s:    %.2f(ms),     %.3f(MB/s)\n", FormatSize(rcdSzD2D).c_str(), rcdTimeD2D, rcdBandwidthD2D);
}

// ==========================================================================================
int main(int argc, char *argv[])
{
	printf("\nHello ROCM.\n\n");
	InitHipRuntime();

	TestBandwidth();

	ReleaseRuntime();
	printf("\nByeBye ROCM.\n\n");

	return 0;
}
