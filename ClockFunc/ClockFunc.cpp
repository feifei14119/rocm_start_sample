#include <hip/hip_runtime.h>

extern "C" __global__ void ClockFunc(const float * a, const float * b, float * c, const unsigned int len)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	
	float aa, bb, cc;
	long long clk1, clk2;
	float dlt1, dlt2, dlt3, dlt4;

	clk1 = clock();
	aa = a[x];
	clk2 = clock();
	dlt1 = (clk2 - clk1) * 1.0f;

	clk1 = clock();
	bb = b[x];
	clk2 = clock();
	dlt2 = (clk2 - clk1) * 1.0f;

	clk1 = clock();
	cc = aa + bb;
	clk2 = clock();
	dlt3 = (clk2 - clk1) * 1.0f;

	clk1 = clock();
	c[x] = cc;
	clk2 = clock();
	dlt4 = (clk2 - clk1) * 1.0f;

	if (threadIdx.x == 0)
	{
		c[0] = dlt1;
		c[1] = dlt2;
		c[2] = dlt3;
		c[3] = dlt4;
	}
}

