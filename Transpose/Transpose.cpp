#include <hip/hip_runtime.h>

#define DATA_TYPE double
#define TILE 64
#define SUB_TILE_CNT 16

extern "C"  __global__ void Transpose000(DATA_TYPE * input, DATA_TYPE * output, uint32_t width, uint32_t height)
{   
    __shared__ DATA_TYPE shared[TILE][TILE];

    uint32_t tid = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
    uint32_t tx1 = tid % TILE;
    uint32_t ty1 = tid / TILE;

    uint32_t iOffset = hipBlockIdx_y * TILE * width  + hipBlockIdx_x * TILE;
    uint32_t oOffset = hipBlockIdx_x * TILE * height + hipBlockIdx_y * TILE;

    uint32_t lmt_width  = min(width  - hipBlockIdx_x * TILE, TILE);
    uint32_t lmt_height = min(height - hipBlockIdx_y * TILE, TILE);

    uint32_t dim_x = TILE; 
    uint32_t dim_y = SUB_TILE_CNT;
    
    for(size_t i = 0; i < lmt_height; i += SUB_TILE_CNT)
    {
        if(tx1 < lmt_width && (ty1 + i) < lmt_height)
        {
            shared[tx1][ty1 + i] = input[iOffset + tx1 + (ty1 + i) * width];
        }
    }

    __syncthreads();

    for(uint32_t i = 0; i < lmt_width; i += SUB_TILE_CNT)
    {
        if(tx1 < lmt_height && (ty1 + i) < lmt_width)
        {
            output[oOffset + tx1 + (i + ty1) * height] = shared[ty1 + i][tx1];
        }
    }
}

extern "C"  __global__ void Transpose(DATA_TYPE * input, DATA_TYPE * output, uint32_t width, uint32_t height)
{   
    __shared__ DATA_TYPE shared[TILE][TILE];

    uint32_t tid = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
    uint32_t tx1 = tid % TILE;
    uint32_t ty1 = tid / TILE;

    uint32_t iOffset = hipBlockIdx_y * TILE * width  + hipBlockIdx_x * TILE;
    uint32_t oOffset = hipBlockIdx_x * TILE * height + hipBlockIdx_y * TILE;

    uint32_t lmt_width  = min(width  - hipBlockIdx_x * TILE, TILE);
    uint32_t lmt_height = min(height - hipBlockIdx_y * TILE, TILE);

    uint32_t dim_x = TILE; 
    uint32_t dim_y = SUB_TILE_CNT;
    uint32_t loop_num;
    
    loop_num = lmt_height / dim_y;
    input += (iOffset + tx1);
    if(tx1 < lmt_width)
    {
#pragma unroll
        for(uint32_t loop_cnt = 0; loop_cnt < loop_num; loop_cnt++)
        {
            shared[tx1][ty1 + loop_cnt * dim_y] = input[(ty1 + loop_cnt * dim_y) * width];
        }

        if((ty1 + loop_num * dim_y) < lmt_height)
        {
            shared[tx1][ty1 + loop_num* dim_y] = input[(ty1 + loop_num* dim_y) * width];
        }
    }

    uint32_t tmp;
    tmp = width; width = height; height = tmp;
    tmp = lmt_width; lmt_width = lmt_height; lmt_height = tmp;
    __syncthreads();
    
    loop_num = lmt_height / dim_y;
    output += (oOffset + tx1);
    if(tx1 < lmt_width)
    {
#pragma unroll
        for(uint32_t loop_cnt = 0; loop_cnt < loop_num; loop_cnt++)
        {
            output[(ty1 + loop_cnt * dim_y) * width] = shared[ty1 + loop_cnt * dim_y][tx1];
        }

        if((ty1 + loop_num * dim_y) < lmt_height)
        {
            output[(ty1 + loop_num * dim_y) * width] = shared[ty1 + loop_num * dim_y][tx1];
        }
    }
}
