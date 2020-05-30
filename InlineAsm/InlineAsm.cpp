#include <hip/hip_runtime.h>

extern "C" __global__ void InlineAsm(const float * a, const float * b, float * c, const unsigned int len)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	
	float aa, bb, cc;

	aa = a[x];
	bb = b[x];

	asm volatile("v_add_f32 %0, %1 %2 \n" : "=v"(cc) : "v"(aa) , "v"(bb));

	c[x] = cc;
}

