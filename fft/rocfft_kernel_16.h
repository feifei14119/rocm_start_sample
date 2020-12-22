#pragma once
#include "rocfft_butterfly_template.h"

template <typename T >
inline __device__ void Transpose(unsigned int me, T *R0, T *R1, T *R2, T *R3)
{
	// can't use __shlf_up function
	// cus __shlf_up doesn't support cyclic shift
	asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[3,0,1,2] \n" : "=v"((*R1).x) : "v"((*R1).x));
	asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] \n" : "=v"((*R2).x) : "v"((*R2).x));
	asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[1,2,3,0] \n" : "=v"((*R3).x) : "v"((*R3).x));
	asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[3,0,1,2] \n" : "=v"((*R1).y) : "v"((*R1).y));
	asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] \n" : "=v"((*R2).y) : "v"((*R2).y));
	asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[1,2,3,0] \n" : "=v"((*R3).y) : "v"((*R3).y));

	// can't use inline assembly v_swap_b32 instruction
	// cus compiler will reallocate vgpr
	T R; 
	if((me == 1) || (me == 3))
	{
		R = (*R0);
		(*R0) = (*R1);
		(*R1) = R;

		R = (*R2);
		(*R2) = (*R3);
		(*R3) = R;
	}
	if((me == 0) || (me == 3))
	{
		R = (*R1);
		(*R1) = (*R3);
		(*R3) = R;
	}
	if((me == 2) || (me == 3))
	{
		R = (*R0);
		(*R0) = (*R2);
		(*R2) = R;
	}
	__syncthreads(); // don't know why

	asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[1,2,3,0] \n" : "=v"((*R1).x) : "v"((*R1).x));
	asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] \n" : "=v"((*R2).x) : "v"((*R2).x));
	asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[3,0,1,2] \n" : "=v"((*R3).x) : "v"((*R3).x));
	asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[1,2,3,0] \n" : "=v"((*R1).y) : "v"((*R1).y));
	asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] \n" : "=v"((*R2).y) : "v"((*R2).y));
	asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[3,0,1,2] \n" : "=v"((*R3).y) : "v"((*R3).y));
}

////////////////////////////////////////Passes kernels
template <typename T >
inline __device__ void
FwdPass0_len16(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, float2 * dbg)
{
	if(rw)
	{
		(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 )*stride_in];
		(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 4 )*stride_in];
		(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 8 )*stride_in];
		(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 12 )*stride_in];
	}

	FwdRad4B1(R0, R1, R2, R3);
	Transpose(me, R0, R1, R2, R3);

	/*if(rw)
	{
		bufOutRe[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 0 ) ] = (*R0).x;
		bufOutRe[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 1 ) ] = (*R1).x;
		bufOutRe[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 2 ) ] = (*R2).x;
		bufOutRe[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 3 ) ] = (*R3).x;

		__syncthreads();

		(*R0).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 0 ) ];
		(*R1).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 4 ) ];
		(*R2).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 8 ) ];
		(*R3).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 12 ) ];

		__syncthreads();

		bufOutIm[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 0 ) ] = (*R0).y;
		bufOutIm[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 1 ) ] = (*R1).y;
		bufOutIm[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 2 ) ] = (*R2).y;
		bufOutIm[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 3 ) ] = (*R3).y;


		__syncthreads();

		(*R0).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 0 ) ];
		(*R1).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 4 ) ];
		(*R2).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 8 ) ];
		(*R3).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 12 ) ];

		__syncthreads();
	}*/
}

template <typename T >
inline __device__ void
FwdPass1_len16(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, T *bufOut, T *R0, T *R1, T *R2, T *R3, float2 * dbg)
{

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 0];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R1).x; ry = (*R1).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 1];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R2).x; ry = (*R2).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 2];
		real_type_t<T> TR, TI;
		real_type_t<T>wx, wy, rx, ry;
		wx = W.x; wy = W.y;
		rx = (*R3).x; ry = (*R3).y;
		TR = wx * rx - wy * ry;
		TI = wy * rx + wx * ry;
		(*R3).x = TR;
		(*R3).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);

	if(rw)
	{
	bufOut[outOffset + ( 1*me + 0 + 0 )*stride_out] = (*R0);
	bufOut[outOffset + ( 1*me + 0 + 4 )*stride_out] = (*R1);
	bufOut[outOffset + ( 1*me + 0 + 8 )*stride_out] = (*R2);
	bufOut[outOffset + ( 1*me + 0 + 12 )*stride_out] = (*R3);
	}

}

////////////////////////////////////////Encapsulated passes kernels
template <typename T >
inline __device__ void 
fwd_len16_device(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, real_type_t<T> *lds, float2 * dbg)
{
	T R0, R1, R2, R3;
	FwdPass0_len16<T >(twiddles, stride_in, stride_out, rw, b, me, 0, ldsOffset,  lwbIn, lds, lds, &R0, &R1, &R2, &R3, dbg);
	FwdPass1_len16<T >(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, 0, lds, lds,  lwbOut, &R0, &R1, &R2, &R3, dbg);
}

////////////////////////////////////////Global kernels
//Kernel configuration: number of threads per thread block: 64, maximum transforms: 16, Passes: 2
template <typename T, StrideBin sb>
__global__ void 
fft_fwd_op_len16( const T * __restrict__ twiddles, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, T * __restrict__ gbIn, T * __restrict__ gbOut)
{

	__shared__ real_type_t<T> lds[256];
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	T *lwbIn;
	T *lwbOut;

	unsigned int upper_count = batch_count;
	for(int i=1; i<dim; i++){
		upper_count *= lengths[i];
	}
	// do signed math to guard against underflow
	unsigned int rw = (static_cast<int>(me) < (static_cast<int>(upper_count)  - static_cast<int>(batch)*16)*4) ? 1 : 0;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = (batch*16 + (me/4));
	if(dim == 1){
		iOffset += counter_mod*stride_in[1];
		oOffset += counter_mod*stride_out[1];
	}
	else if(dim == 2){
		int counter_1 = counter_mod / lengths[1];
		int counter_mod_1 = counter_mod % lengths[1];
		iOffset += counter_1*stride_in[2] + counter_mod_1*stride_in[1];
		oOffset += counter_1*stride_out[2] + counter_mod_1*stride_out[1];
	}
	else if(dim == 3){
		int counter_2 = counter_mod / (lengths[1] * lengths[2]);
		int counter_mod_2 = counter_mod % (lengths[1] * lengths[2]);
		int counter_1 = counter_mod_2 / lengths[1];
		int counter_mod_1 = counter_mod_2 % lengths[1];
		iOffset += counter_2*stride_in[3] + counter_1*stride_in[2] + counter_mod_1*stride_in[1];
		oOffset += counter_2*stride_out[3] + counter_1*stride_out[2] + counter_mod_1*stride_out[1];
	}
	else{
		for(int i = dim; i>1; i--){
			int currentLength = 1;
			for(int j=1; j<i; j++){
				currentLength *= lengths[j];
			}

			iOffset += (counter_mod / currentLength)*stride_in[i];
			oOffset += (counter_mod / currentLength)*stride_out[i];
			counter_mod = counter_mod % currentLength;
		}
		iOffset+= counter_mod * stride_in[1];
		oOffset+= counter_mod * stride_out[1];
	}
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
	// rw, b, me% control read/write; then ldsOffset, lwb, lds
	fwd_len16_device<T, sb>(twiddles, stride_in[0], stride_out[0],  rw, b, me%4, (me/4)*16, lwbIn, lwbOut, lds);
}


//Kernel configuration: number of threads per thread block: 64, maximum transforms: 16, Passes: 2
__global__ void 
fft_fwd_op_len16( const float2 * twiddles,
				const uint32_t length,
				const uint32_t batch_count,
				float2 * gbIn, float2 * gbOut,
				float2 * dbg)
{
	__shared__ float lds[256];
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	float2 *lwbIn;
	float2 *lwbOut;

	unsigned int upper_count = batch_count;
	
	// do signed math to guard against underflow
	unsigned int rw = (me < (upper_count - batch*16)*4) ? 1 : 0;
	unsigned int b = 0;

	size_t counter_mod = (batch*16 + (me/4));
	//if(dim == 1)
	{
		iOffset += counter_mod*length;
		oOffset += counter_mod*length;
	}
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
	// rw, b, me% control read/write; then ldsOffset, lwb, lds
	size_t stride_in = 1;
	size_t stride_out = 1;
	fwd_len16_device<float2>(twiddles, stride_in, stride_out, rw, b, me%4, (me/4)*16, lwbIn, lwbOut, lds, dbg);
}

