
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
using namespace std;


template<class DType>
__global__ void block_sum(const DType *input,
        DType *per_block_results,
                        const size_t n)
{
    extern __shared__ DType sdata[];
 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
 
    // 一个线程负责把一个元素从全局内存载入到共享内存
    DType x = 0;
    if(i < n){
        x = input[i];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();//等待所有线程把自己负责的元素载入到共享内存
 
    // 块内进行合并操作，每次合并变为一半.注意threadIdx.x是块内的偏移，上面算出的i是全局的偏移。
    for(int offset = blockDim.x / 2;
            offset > 0;
            offset >>= 1)
    {
        if(threadIdx.x < offset)//控制只有某些线程才进行操作。
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }
 
    //每个块的线程0负责存放块内求和的结果
    if(threadIdx.x == 0)
    {
        per_block_results[blockIdx.x] = sdata[0];
    }
}

int main()
{
	const int num_elements=1024;   //设置数组长度
    //分配内存
	float h_input[num_elements];
	for (int i = 0; i < num_elements; i++){
		h_input[i]=1.0f;
	}
	
    float *d_input = 0;

    cudaMalloc((void**)&d_input, sizeof(float) * num_elements);
    cudaMemcpy(d_input, &h_input, sizeof(float) * num_elements, cudaMemcpyHostToDevice);
 
    const size_t block_size = 512;//线程块的大小。目前有些gpu的线程块最大为512，有些为1024.
    const size_t num_blocks = (num_elements/block_size) + ((num_elements%block_size) ? 1 : 0);
 
    float *d_partial_sums_and_total = 0;//一个线程块一个和，另外加一个元素，存放所有线程块的和。
    cudaMalloc((void**)&d_partial_sums_and_total, sizeof(float) * (num_blocks + 1));
 
    //把每个线程块的和求出来
    block_sum<<<num_blocks,block_size,block_size *sizeof(float)>>>(d_input, d_partial_sums_and_total, num_elements);
 
    
     //再次用一个线程块把上一步的结果求和。
    //注意这里有个限制，上一步线程块的数量，必须不大于一个线程块线程的最大数量，因为这一步得把上一步的结果放在一个线程块操作。
    //即num_blocks不能大于线程块的最大线程数量。
    block_sum<<<1,num_blocks,num_blocks * sizeof(float)>>>(d_partial_sums_and_total, d_partial_sums_and_total + num_blocks, num_blocks);
 
    
	float device_result = 0;
	cudaMemcpy(&device_result, d_partial_sums_and_total + num_blocks, sizeof(float), cudaMemcpyDeviceToHost);
 
    std::cout << "Device sum: " << device_result << std::endl;
 
    // 释放显存容量
    cudaFree(d_input);
    cudaFree(d_partial_sums_and_total);
	
	return 0;
}
