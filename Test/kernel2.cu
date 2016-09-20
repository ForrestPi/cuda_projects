
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
   
////////////////////////���豸�����е��ں˺���/////////////////////////////  
__global__ static void Kernel_SquareSum( int* pIn, size_t* pDataSize,  
                               int*pOut, clock_t* pTime )  
{  
   // ����һ����̬����Ĺ���洢��  
   extern __shared__ int sharedData[];  
   
   const size_t computeSize =*pDataSize / THREAD_NUM;  
   const size_t tID = size_t(threadIdx.x );// �߳�  
   const size_t bID = size_t(blockIdx.x );// ��  
   
   int offset = 1;    // ��¼ÿ�������Ĳ���  
   int mask = 1;      // ѡ����ʵ��߳�  
   
   // ��ʼ��ʱ  
   if ( tID == 0 ) pTime[bID] =clock( );// ѡ������һ���߳̽��м�ʱ  
   
   // ִ�м���  
   for ( size_t i = bID * THREAD_NUM+ tID;  
      i < DATA_SIZE;  
      i += BLOCK_NUM * THREAD_NUM )  
   {  
      sharedData[tID] += pIn[i] * pIn[i];  
   }  
   
   // ͬ��һ�����е������߳�  
   __syncthreads( );  
   
   while ( offset < THREAD_NUM )  
   {  
      if ( ( tID & mask ) == 0 )  
      {  
         sharedData[tID] += sharedData[tID + offset];  
      }  
      offset += offset;     // ����һλ  
      mask = offset + mask; // �����һλ������λ  
   
      __syncthreads( );  
   }  
   
   if ( tID == 0 )// ����߳�IDΪ����ô������������¼ʱ��  
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
   
   // 1�������豸  
   cudaError_t cudaStatus = cudaSetDevice( 0 );// ֻҪ������װ��Ӣΰ���Կ�����ô����óɹ�  
   if ( cudaStatus != cudaSuccess )  
   {  
      fprintf( stderr, "����cudaSetDevice()����ʧ�ܣ�" );  
      return false;  
   }  
   
   switch ( true)  
   {  
   default:  
      // 2�������Դ�ռ�  
      cudaStatus = cudaMalloc( (void**)&pDevIn,dataSize * sizeof( int) );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "����cudaMalloc()������ʼ���Կ�������ʱʧ�ܣ�" );  
         break;  
      }  
   
      cudaStatus = cudaMalloc( (void**)&pDevOut,BLOCK_NUM * sizeof( int) );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "����cudaMalloc()������ʼ���Կ��з���ֵʱʧ�ܣ�" );  
         break;  
      }  
   
      cudaStatus = cudaMalloc( (void**)&pDevDataSize,sizeof( size_t ) );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "����cudaMalloc()������ʼ���Կ������ݴ�Сʱʧ�ܣ�" );  
         break;  
      }  
   
      cudaStatus = cudaMalloc( (void**)&pDevTime,BLOCK_NUM * 2 * sizeof( clock_t ) );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "����cudaMalloc()������ʼ���Կ��кķ���ʱ����ʧ�ܣ�" );  
         break;  
      }  
   
      // 3���������������ݸ��Ƶ��Դ���  
      cudaStatus = cudaMemcpy( pDevIn, pIn, dataSize * sizeof( int ),cudaMemcpyHostToDevice );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "����cudaMemcpy()������ʼ�����������������鵽�Կ�ʱʧ�ܣ�" );  
         break;  
      }  
   
      cudaStatus = cudaMemcpy( pDevDataSize, &dataSize, sizeof( size_t ), cudaMemcpyHostToDevice );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "����cudaMemcpy()������ʼ�������������ݴ�С���Կ�ʱʧ�ܣ�" );  
         break;  
      }  
   
      // 4��ִ�г�����������ȴ��Կ�ִ�����  
      Kernel_SquareSum<<<BLOCK_NUM, THREAD_NUM, THREAD_NUM *sizeof( int)>>>  
         ( pDevIn, pDevDataSize, pDevOut, pDevTime );  
   
      // 5����ѯ�ں˳�ʼ����ʱ���Ƿ����  
      cudaStatus = cudaGetLastError( );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "�Կ�ִ�г���ʱʧ�ܣ�" );  
         break;  
      }  
   
      // 6�����ں�ͬ���ȴ�ִ�����  
      cudaStatus = cudaDeviceSynchronize( );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "�����ں�ͬ���Ĺ����з������⣡" );  
         break;  
      }  
   
      // 7����ȡ����  
      cudaStatus = cudaMemcpy( pOut, pDevOut, BLOCK_NUM * sizeof( int ),cudaMemcpyDeviceToHost );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "�ڽ�������ݴ��Կ����Ƶ�����������ʧ�ܣ�" );  
         break;  
      }  
   
      cudaStatus = cudaMemcpy( pTime, pDevTime, BLOCK_NUM * 2 * sizeof( clock_t ), cudaMemcpyDeviceToHost );  
      if ( cudaStatus != cudaSuccess)  
      {  
         fprintf( stderr, "�ڽ��ķ���ʱ���ݴ��Կ����Ƶ�����������ʧ�ܣ�" );  
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
   
void GenerateData( int* pData,size_t dataSize )// ��������  
{  
   assert( pData != nullptr );  
   for ( size_t i = 0; i <dataSize; i++ )  
   {  
      srand( i + 3 );  
      pData[i] = rand( ) % 100;  
   }  
}  
   
int main( int argc, char** argv )// �����������  
{  
   int* pData = nullptr;  
   int* pResult = nullptr;  
   clock_t* pTime = nullptr;  
   
   // ʹ��CUDA�ڴ����������host��  
   cudaError_t cudaStatus = cudaMallocHost( &pData, DATA_SIZE * sizeof( int ) );  
   if ( cudaStatus != cudaSuccess )  
   {  
      fprintf( stderr, "�������з�����Դʧ�ܣ�" );  
      return 1;  
   }  
   
   cudaStatus = cudaMallocHost( &pResult, BLOCK_NUM * sizeof( int ) );  
   if ( cudaStatus != cudaSuccess )  
   {  
      fprintf( stderr, "�������з�����Դʧ�ܣ�" );  
      return 1;  
   }  
   
   cudaStatus = cudaMallocHost( &pTime, BLOCK_NUM * 2 * sizeof( clock_t ) );  
   if ( cudaStatus != cudaSuccess )  
   {  
      fprintf( stderr, "�������з�����Դʧ�ܣ�" );  
      return 1;  
   }  
   
   GenerateData( pData, DATA_SIZE );// ͨ���������������  
   CUDA_SquareSum( pResult, pTime, pData, DATA_SIZE );// ִ��ƽ����  
   
   // ��CPU�н�����������  
   int totalResult=0;  
   for ( int i = 0; i < BLOCK_NUM; ++i )  
   {  
      totalResult += pResult[i];  
   }  
   
   // ����ִ�е�ʱ��  
   clock_t startTime = pTime[0];  
   clock_t endTime = pTime[BLOCK_NUM];  
   for ( int i = 0; i < BLOCK_NUM; ++i )  
   {  
      if ( startTime > pTime[i] )startTime = pTime[i];  
      if ( endTime < pTime[i +BLOCK_NUM] ) endTime = pTime[i + BLOCK_NUM];  
   }  
   clock_t elapsed = endTime - startTime;  
   
   
   // �ж��Ƿ����  
   char* pOverFlow = nullptr;  
   if ( totalResult < 0 )pOverFlow = "�������";  
   else pOverFlow = "";  
   
   // ��ʾ��׼����  
   printf( "��CUDA����ƽ���͵Ľ���ǣ�%d%s\n�ķ���ʱ��%d\n",  
      totalResult, pOverFlow, elapsed );  
   
   cudaDeviceProp prop;  
   if ( cudaGetDeviceProperties(&prop, 0 ) == cudaSuccess )  
   {  
      float actualTime = float( elapsed ) / float(prop.clockRate );  
      printf( "ʵ��ִ��ʱ��Ϊ��%.2fms\n", actualTime );  
      printf( "����Ϊ��%.2fMB/s\n",  
         float( DATA_SIZE * sizeof( int )>> 20 ) * 1000.0f / actualTime );  
      printf( "GPU�豸�ͺţ�%s\n", prop.name );  
   }  
   
   cudaFreeHost( pData );  
   cudaFreeHost( pResult );  
   cudaFreeHost( pTime );  
   
   return 0;  
}  