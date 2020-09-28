#include <hip/hip_runtime.h>

#define DATA_TYPE   double
#define TILE        (64)
#define ELE_PER_THR (4)
#define WAVE_SIZE   (64)
#define GROUP_SIZE  (TILE*TILE / ELE_PER_THR)
#define WAVE_NUM    (GROUP_SIZE / WAVE_SIZE)

extern "C"  __global__ void Transpose(DATA_TYPE * input, DATA_TYPE * output, uint32_t width, uint32_t height)
{   
    __shared__ DATA_TYPE shared[TILE][TILE];

    uint32_t tid_x = hipThreadIdx_x % TILE;
    uint32_t tid_y = hipThreadIdx_x / TILE;

    uint32_t iOffset = hipBlockIdx_y * TILE * width  + hipBlockIdx_x * TILE;
    uint32_t oOffset = hipBlockIdx_x * TILE * height + hipBlockIdx_y * TILE;
    input  += iOffset;
    output += oOffset;

    uint32_t lmt_width  = min(width  - hipBlockIdx_x * TILE, TILE);
    uint32_t lmt_height = min(height - hipBlockIdx_y * TILE, TILE);
    
    for(uint32_t i = 0; i < lmt_height; i += WAVE_NUM)
    {
        if(tid_x < lmt_width && (tid_y + i) < lmt_height)
        {
            shared[tid_x][tid_y + i] = input[tid_x + (tid_y + i) * width];
        }
    }

    uint32_t tmp;
    tmp = width; width = height; height = tmp;
    tmp = lmt_width; lmt_width = lmt_height; lmt_height = tmp;
    __syncthreads();

    for(uint32_t i = 0; i < lmt_height; i += WAVE_NUM)
    {
        if(tid_x < lmt_width && (tid_y + i) < lmt_height)
        {
            output[tid_x + (tid_y + i) * width] = shared[tid_y + i][tid_x];
        }
    }
}

extern "C"  __global__ void Transpose111(DATA_TYPE * input, DATA_TYPE * output, uint32_t width, uint32_t height)
{   
    __shared__ DATA_TYPE shared[TILE][TILE];

    uint32_t tid_x = hipThreadIdx_x % TILE;
    uint32_t tid_y = hipThreadIdx_x / TILE;

    uint32_t iOffset = hipBlockIdx_y * TILE * width  + hipBlockIdx_x * TILE;
    uint32_t oOffset = hipBlockIdx_x * TILE * height + hipBlockIdx_y * TILE;

    uint32_t lmt_width  = min(width  - hipBlockIdx_x * TILE, TILE);
    uint32_t lmt_height = min(height - hipBlockIdx_y * TILE, TILE);
    
    uint32_t loop_num = lmt_height / WAVE_NUM;
    input += (iOffset + tid_x);
    if(tid_x < lmt_width)
    {
#pragma unroll
        for(uint32_t loop_cnt = 0; loop_cnt < loop_num; loop_cnt++)
        {
            shared[tid_x][tid_y + loop_cnt * WAVE_NUM] = input[(tid_y + loop_cnt * WAVE_NUM) * width];
        }

        if((tid_y + loop_num * WAVE_NUM) < lmt_height)
        {
            shared[tid_x][tid_y + loop_num* WAVE_NUM] = input[(tid_y + loop_num* WAVE_NUM) * width];
        }
    }

    uint32_t tmp;
    tmp = width; width = height; height = tmp;
    tmp = lmt_width; lmt_width = lmt_height; lmt_height = tmp;
    __syncthreads();
    
    loop_num = lmt_height / WAVE_NUM;
    output += (oOffset + tid_x);
    if(tid_x < lmt_width)
    {
#pragma unroll
        for(uint32_t loop_cnt = 0; loop_cnt < loop_num; loop_cnt++)
        {
            output[(tid_y + loop_cnt * WAVE_NUM) * width] = shared[tid_y + loop_cnt * WAVE_NUM][tid_x];
        }

        if((tid_y + loop_num * WAVE_NUM) < lmt_height)
        {
            output[(tid_y + loop_num * WAVE_NUM) * width] = shared[tid_y + loop_num * WAVE_NUM][tid_x];
        }
    }
}
