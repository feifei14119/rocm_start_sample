#include <hip/hip_runtime.h>

// __shfl Support int32, float
extern "C" __global__ void Shfl(const float * a, float * b, float * c, const unsigned int len)
{
	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	
	int old_val;
	int shf_val;

	old_val = a[x];
	// thread[n] = thread[21]
	shf_val = __shfl(old_val, 21);
	b[x] = shf_val;
}

// __shfl_up Support int32, float
extern "C" __global__ void ShflUp(const float * a, float * b, float * c, const unsigned int len)
{
	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	
	float old_val;
	float shf_val;

	old_val = a[x];
	// thread[0] = thread[0]
	// thread[1] = thread[1]
	// thread[n] = thread[n-2]
	shf_val = __shfl_up(old_val, 2);
	b[x] = shf_val;
}

// __shfl_down Support int32, float
extern "C" __global__ void ShflDown(const float * a, float * b, float * c, const unsigned int len)
{
	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

	float old_val;
	float shf_val;

	old_val = a[x];
	// thread[62] = thread[62]
	// thread[63] = thread[63]
	// thread[n] = thread[n+2]
	shf_val = __shfl_down(old_val, 2);
	b[x] = shf_val;
}

// __shfl_xor Support int32, float
extern "C" __global__ void ShflXor(const float * a, float * b, float * c, const unsigned int len)
{
	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

	float old_val;
	float shf_val;

	old_val = a[x];
	// 1 wave = 4 lane
	// 1 lane = 16 thread
	// mask for one lane
	// thread[n] = thread[n xor 0x03]
	shf_val = __shfl_xor(old_val, 0x03);
	b[x] = shf_val;
}
