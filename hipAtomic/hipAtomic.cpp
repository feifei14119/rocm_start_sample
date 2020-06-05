#include <hip/hip_runtime.h>

// Atomic Add/Sub Support int32, uint32, uint64, float
extern "C" __global__ void AtomicAdd(const int * a, int * b, int * c, const unsigned int len)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (x >= len)
		return;

	int add_val;
	int old_val;

	add_val = a[x];
	old_val = atomicAdd(&b[0], add_val);
	c[x] = old_val;
}

// Atomic Add/Sub Support int32, uint32, uint64
extern "C" __global__ void AtomicMax(const int * a, int * b, int * c, const unsigned int len)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (x >= len)
		return;

	int cmp_val;
	int old_val;

	cmp_val = a[x];
	old_val = atomicMax(&b[0], cmp_val);
	c[x] = old_val;
}

// Atomic Inc/Dec Support uint32
extern "C" __global__ void AtomicInc(const unsigned int * a, unsigned int * b, unsigned int * c, const unsigned int len)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (x >= len)
		return;

	unsigned int threshold;
	unsigned int old_val;

	threshold = 20;
	// b[0] = old_val > threshold ? 0 : b[0]+1 
	old_val = atomicInc(&b[0], threshold);
	c[x] = old_val;
}

// Atomic And/Or/Xor Support uint32
extern "C" __global__ void AtomicOr(const unsigned int * a, unsigned int * b, unsigned int * c, const unsigned int len)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (x >= len)
		return;

	unsigned int or_val;
	unsigned int old_val;

	or_val = a[x];
	old_val = atomicOr(&b[0], or_val);
	c[x] = old_val;
}

// Atomic Exch Support int32, uint32, uint64, float
extern "C" __global__ void AtomicExch(const unsigned int * a, unsigned int * b, unsigned int * c, const unsigned int len)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (x >= len)
		return;

	unsigned int exch_val;
	unsigned int old_val;

	exch_val = a[x];
	old_val = atomicExch(&b[0], exch_val);
	c[x] = old_val;
}

// Atomic CAS Support int32, uint32, uint64
extern "C" __global__ void AtomicCAS(const unsigned int * a, unsigned int * b, unsigned int * c, const unsigned int len)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (x >= len)
		return;

	unsigned int cmp_val;
	unsigned int new_val;
	unsigned int old_val;

	cmp_val = 100;
	new_val = a[x];
	old_val = atomicCAS(&b[0], cmp_val, new_val);
	c[x] = old_val;
}
