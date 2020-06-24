#include <float.h>
#include <math.h>
#include <string>

#include <hip/hip_ext.h>

//#include <hip/hcc_detail/hip_fp16.h>
//#include <hip/hip_complex.h>
//#include <hip/math_functions.h>

using namespace std;

#define WAVE_SIZE			(64)
#define SIMD_PER_CU			(4)

#define HIP_ASSERT(x)		(assert((x)==hipSuccess))

extern void PrintStep1(string s);
extern void PrintStep2(string s);
extern void Logout(string s);
extern void PrintHostData(int * pData, uint32_t len);
extern void PrintHostData(float * pData, uint32_t len);
extern void PrintDeviceData(int * pData, uint32_t len);
extern void PrintDeviceData(float * pData, uint32_t len);
extern void CompareData(float * h_data, float * d_data, uint32_t len);
extern void ExecCommand(string cmd);

extern string FormatSize(size_t sz);
extern string FormatFreq(int clk);

extern hipDeviceProp_t HipDeviceProp;
extern void InitHipRuntime();
extern void ReleaseRuntime();

extern void CreateHipKernel(string kernelName, string kernelFile = "");
extern void CreateAsmKernel(string kernelName, string kernelFile = "");

extern unsigned char KernelArgsBuff[];
extern size_t KernelArgsSize;
extern uint32_t KernelArgsNum;
template<typename T> void AddArg(T d_addr)
{
	unsigned char * pBuff = KernelArgsBuff + KernelArgsSize;
	memcpy(pBuff, &d_addr, sizeof(d_addr));

	KernelArgsSize += sizeof(d_addr);
	KernelArgsNum++;
}
extern void PrintKernelArgs();
extern void FreeKernelArgs();

extern dim3 GroupSize;
extern dim3 GroupNum;
extern dim3 GlobalSize;
extern void SetGroupSize(uint32_t x, uint32_t y = 1, uint32_t z = 1);
extern void SetGroupNum(uint32_t x, uint32_t y = 1, uint32_t z = 1);
extern void SetGlobalSize(uint32_t x, uint32_t y = 1, uint32_t z = 1);
extern void PrintWorkload();

extern void LaunchKernel();
extern float LaunchKernelGetElapsedMs();
