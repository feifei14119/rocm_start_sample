#include <hip/hip_runtime.h>

extern "C" __global__ void
VectorAdd(
	const float* __restrict__ a, 
	const float* __restrict__ b, 
	float* __restrict__ c, 
	const unsigned int len) 
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (x >= len)
		return;

	c[x] = a[x] + b[x];
}
