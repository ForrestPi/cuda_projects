
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cuda_runtime.h>  
#include <cctype>  
#include <cassert>  
#include <cstdio>  
#include <ctime>  
#include <cstdlib>


#define DATA_SIZE 1048576  
#define BLOCK_NUM 32  
#define THREAD_NUM 256  
#ifndef nullptr  
#define nullptr 0  
#endif  
   
using namespace std;  
   
////////////////////////在设备上运行的内核函数/////////////////////////////  
__global__ static void Kernel_SquareSum( int* pIn, size_t* pDataSize,  
                               int*pOut, clock_t* pTime )  
{  
   // 声明一个动态分配的共享存储器  
   extern __shared__ int sharedData[];  
   
   const size_t computeSize =*pDataSize / THREAD_NUM;  
   const size_t tID = size_t(threadIdx.x );// 线程  
   const size_t bID = size_t(blockIdx.x );// 块  
   
   int offset = 1;    // 记录每轮增倍的步距  
   int mask = 1;      // 选择合适的线程  
   
   // 开始计时  
   if ( tID == 0 ) pTime[bID] =clock( );// 选择任意一个线程进行计时  
   
   // 执行计算  
   for ( size_t i = bID * THREAD_NUM+ tID;  
      i < DATA_SIZE;  
      i += BLOCK_NUM * THREAD_NUM )  
   {  
      sharedData[tID] += pIn[i] * pIn[i];  
   }  
   
   // 同步一个块中的其它线程  
   __syncthreads( );  
   
   while ( offset < THREAD_NUM )  
   {  
      if ( ( tID & mask ) == 0 )  
      {  
         sharedData[tID] += sharedData[tID + offset];  
      }  
      offset += offset;     // 左移一位  
      mask = offset + mask; // 掩码多一位二进制位  
   
      __syncthreads( );  
   }  
   
   if ( tID == 0 )// 如果线程ID为，那么计算结果，并记录时钟  
   {  
      pOut[bID] = sharedData[0];  
      pTime[bID + BLOCK_NUM] = clock( );  
   }  
}  
   
bool CUDA_SquareSum( int* pOut,clock_t* pTime,  
                int* pIn, size_t dataSize )  
{  
   assert( pIn != nullptr );  
   assert( pOut != nullptr );  
   
   int* pDevIn = nullptr;  
   int* pDevOut = nullptr;  
   size_t* pDevDataSize = nullptr;  
   clock_t* pDevTime = nullptr;  
   
   // 1、设置设备  
   cudaError_t cudaStatus = cudaSetDevice( 0 );// 只要机器安装了英伟达显卡，那么会调用成功  
   if ( cudaStatus != cudaSuccess )  
   {  
      fprintf( stderr, "调用cudaSetDevice()函数失败！" );  
      return false;  
   }  
   
   switch ( true)  
   {  
   default:  
      // 2、分配显存空间  
      cudaStatus = cudaMalloc( (void**)&pDevIn,dataSize * sizeof( int) );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "调用cudaMalloc()函数初始化显卡中数组时失败！" );  
         break;  
      }  
   
      cudaStatus = cudaMalloc( (void**)&pDevOut,BLOCK_NUM * sizeof( int) );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "调用cudaMalloc()函数初始化显卡中返回值时失败！" );  
         break;  
      }  
   
      cudaStatus = cudaMalloc( (void**)&pDevDataSize,sizeof( size_t ) );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "调用cudaMalloc()函数初始化显卡中数据大小时失败！" );  
         break;  
      }  
   
      cudaStatus = cudaMalloc( (void**)&pDevTime,BLOCK_NUM * 2 * sizeof( clock_t ) );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "调用cudaMalloc()函数初始化显卡中耗费用时变量失败！" );  
         break;  
      }  
   
      // 3、将宿主程序数据复制到显存中  
      cudaStatus = cudaMemcpy( pDevIn, pIn, dataSize * sizeof( int ),cudaMemcpyHostToDevice );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "调用cudaMemcpy()函数初始化宿主程序数据数组到显卡时失败！" );  
         break;  
      }  
   
      cudaStatus = cudaMemcpy( pDevDataSize, &dataSize, sizeof( size_t ), cudaMemcpyHostToDevice );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "调用cudaMemcpy()函数初始化宿主程序数据大小到显卡时失败！" );  
         break;  
      }  
   
      // 4、执行程序，宿主程序等待显卡执行完毕  
      Kernel_SquareSum<<<BLOCK_NUM, THREAD_NUM, THREAD_NUM *sizeof( int)>>>  
         ( pDevIn, pDevDataSize, pDevOut, pDevTime );  
   
      // 5、查询内核初始化的时候是否出错  
      cudaStatus = cudaGetLastError( );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "显卡执行程序时失败！" );  
         break;  
      }  
   
      // 6、与内核同步等待执行完毕  
      cudaStatus = cudaDeviceSynchronize( );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "在与内核同步的过程中发生问题！" );  
         break;  
      }  
   
      // 7、获取数据  
      cudaStatus = cudaMemcpy( pOut, pDevOut, BLOCK_NUM * sizeof( int ),cudaMemcpyDeviceToHost );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "在将结果数据从显卡复制到宿主程序中失败！" );  
         break;  
      }  
   
      cudaStatus = cudaMemcpy( pTime, pDevTime, BLOCK_NUM * 2 * sizeof( clock_t ), cudaMemcpyDeviceToHost );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "在将耗费用时数据从显卡复制到宿主程序中失败！" );  
         break;  
      }  
   
      cudaFree( pDevIn );  
      cudaFree( pDevOut );  
      cudaFree( pDevDataSize );  
      cudaFree( pDevTime );  
      return true;  
   }  
   
   cudaFree( pDevIn );  
   cudaFree( pDevOut );  
   cudaFree( pDevDataSize );  
   cudaFree( pDevTime );  
   return false;  
}  
   
void GenerateData( int* pData,size_t dataSize )// 产生数据  
{  
   assert( pData != nullptr );  
   for ( size_t i = 0; i <dataSize; i++ )  
   {  
      srand( i + 3 );  
      pData[i] = rand( ) % 100;  
   }  
}  
   
int main( int argc, char** argv )// 函数的主入口  
{  
   int* pData = nullptr;  
   int* pResult = nullptr;  
   clock_t* pTime = nullptr;  
   
   // 使用CUDA内存分配器分配host端  
   cudaError_t cudaStatus = cudaMallocHost( &pData, DATA_SIZE * sizeof( int ) );  
   if ( cudaStatus != cudaSuccess )  
   {  
      fprintf( stderr, "在主机中分配资源失败！" );  
      return 1;  
   }  
   
   cudaStatus = cudaMallocHost( &pResult, BLOCK_NUM * sizeof( int ) );  
   if ( cudaStatus != cudaSuccess )  
   {  
      fprintf( stderr, "在主机中分配资源失败！" );  
      return 1;  
   }  
   
   cudaStatus = cudaMallocHost( &pTime, BLOCK_NUM * 2 * sizeof( clock_t ) );  
   if ( cudaStatus != cudaSuccess )  
   {  
      fprintf( stderr, "在主机中分配资源失败！" );  
      return 1;  
   }  
   
   GenerateData( pData, DATA_SIZE );// 通过随机数产生数据  
   CUDA_SquareSum( pResult, pTime, pData, DATA_SIZE );// 执行平方和  
   
   // 在CPU中将结果组合起来  
   int totalResult=0;  
   for ( int i = 0; i < BLOCK_NUM; ++i )  
   {  
      totalResult += pResult[i];  
   }  
   
   // 计算执行的时间  
   clock_t startTime = pTime[0];  
   clock_t endTime = pTime[BLOCK_NUM];  
   for ( int i = 0; i < BLOCK_NUM; ++i )  
   {  
      if ( startTime > pTime[i] )startTime = pTime[i];  
      if ( endTime < pTime[i +BLOCK_NUM] ) endTime = pTime[i + BLOCK_NUM];  
   }  
   clock_t elapsed = endTime - startTime;  
   
   
   // 判断是否溢出  
   char* pOverFlow = nullptr;  
   if ( totalResult < 0 )pOverFlow = "（溢出）";  
   else pOverFlow = "";  
   
   // 显示基准测试  
   printf( "用CUDA计算平方和的结果是：%d%s\n耗费用时：%d\n",  
      totalResult, pOverFlow, elapsed );  
   
   cudaDeviceProp prop;  
   if ( cudaGetDeviceProperties(&prop, 0 ) == cudaSuccess )  
   {  
      float actualTime = float( elapsed ) / float(prop.clockRate );  
      printf( "实际执行时间为：%.2fms\n", actualTime );  
      printf( "带宽为：%.2fMB/s\n",  
         float( DATA_SIZE * sizeof( int )>> 20 ) * 1000.0f / actualTime );  
      printf( "GPU设备型号：%s\n", prop.name );  
   }  
   
   cudaFreeHost( pData );  
   cudaFreeHost( pResult );  
   cudaFreeHost( pTime );  
   
   return 0;  
}  