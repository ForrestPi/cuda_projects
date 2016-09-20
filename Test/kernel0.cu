
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
 
    // һ���̸߳����һ��Ԫ�ش�ȫ���ڴ����뵽�����ڴ�
    DType x = 0;
    if(i < n){
        x = input[i];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();//�ȴ������̰߳��Լ������Ԫ�����뵽�����ڴ�
 
    // ���ڽ��кϲ�������ÿ�κϲ���Ϊһ��.ע��threadIdx.x�ǿ��ڵ�ƫ�ƣ����������i��ȫ�ֵ�ƫ�ơ�
    for(int offset = blockDim.x / 2;
            offset > 0;
            offset >>= 1)
    {
        if(threadIdx.x < offset)//����ֻ��ĳЩ�̲߳Ž��в�����
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }
 
    //ÿ������߳�0�����ſ�����͵Ľ��
    if(threadIdx.x == 0)
    {
        per_block_results[blockIdx.x] = sdata[0];
    }
}

int main()
{
	const int num_elements=1024;   //�������鳤��
    //�����ڴ�
	float h_input[num_elements];
	for (int i = 0; i < num_elements; i++){
		h_input[i]=1.0f;
	}
	
    float *d_input = 0;

    cudaMalloc((void**)&d_input, sizeof(float) * num_elements);
    cudaMemcpy(d_input, &h_input, sizeof(float) * num_elements, cudaMemcpyHostToDevice);
 
    const size_t block_size = 512;//�߳̿�Ĵ�С��Ŀǰ��Щgpu���߳̿����Ϊ512����ЩΪ1024.
    const size_t num_blocks = (num_elements/block_size) + ((num_elements%block_size) ? 1 : 0);
 
    float *d_partial_sums_and_total = 0;//һ���߳̿�һ���ͣ������һ��Ԫ�أ���������߳̿�ĺ͡�
    cudaMalloc((void**)&d_partial_sums_and_total, sizeof(float) * (num_blocks + 1));
 
    //��ÿ���߳̿�ĺ������
    block_sum<<<num_blocks,block_size,block_size *sizeof(float)>>>(d_input, d_partial_sums_and_total, num_elements);
 
    
     //�ٴ���һ���߳̿����һ���Ľ����͡�
    //ע�������и����ƣ���һ���߳̿�����������벻����һ���߳̿��̵߳������������Ϊ��һ���ð���һ���Ľ������һ���߳̿������
    //��num_blocks���ܴ����߳̿������߳�������
    block_sum<<<1,num_blocks,num_blocks * sizeof(float)>>>(d_partial_sums_and_total, d_partial_sums_and_total + num_blocks, num_blocks);
 
    
	float device_result = 0;
	cudaMemcpy(&device_result, d_partial_sums_and_total + num_blocks, sizeof(float), cudaMemcpyDeviceToHost);
 
    std::cout << "Device sum: " << device_result << std::endl;
 
    // �ͷ��Դ�����
    cudaFree(d_input);
    cudaFree(d_partial_sums_and_total);
	
	return 0;
}
