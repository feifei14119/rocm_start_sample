#include <hip/hip_runtime.h>

extern "C" __global__ void hipVote(const int * a, int * b, int * c, const unsigned int len)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	
	int val;
	int any_val;
	int all_val;
	int ballot_val;

	val = a[x];

	// if any thread != 0, any_val = 1
	any_val = __any(val);
	// if all thread != 0, any_val = 1
	all_val = __all(val);

	ballot_val = __ballot(val);

	if (threadIdx.x == 0)
	{
		b[0] = any_val;
		b[1] = all_val;
		b[2] = ballot_val;
	}
}

