#include <stdio.h>
#include <chrono>
#include <iostream>
#include <cmath>

#define TPB 256

/*
array size 5000: CPU computing time:1.21e-05 GPU computing time:0.167205 The difference is 0.000000
array size 50000: CPU computing time:0.0001292 GPU computing time:0.158399 The difference is 0.000000
array size 500000: CPU computing time:0.0013506 GPU computing time:0.167526 The difference is 0.000000
array size 5000000: CPU computing time:0.0176518 GPU computing time:0.173352 The difference is 0.000000
array size 50000000: CPU computing time:0.199978 GPU computing time:0.274253 The difference is 0.000000
array size 500000000: CPU computing time:2.09953 GPU computing time:1.18089 The difference is 0.000000
*/

__global__ void saxpyKernel(float *x, float *y, const float a, const int n)
{
  const int id = blockIdx.x*blockDim.x + threadIdx.x;
  if(id>=n)
  {
    return;
  }
  y[id] = x[id] * y[id] + a;
}

void cpuSaxpy(float *x, float *y, const float a, const int n)
{
  for(size_t i=0; i<n;++i)
  {
    y[i] = x[i] * y[i] + a;
  }
}
float difference(float *a, float *b, const int n)
{
  float result = 0.0;
  for(size_t i=0; i<n;++i)
  {
    result += std::abs(a[i] - b[i]);
  }
  return result;
}
int main()
{
  int array_size;
  std::cin >> array_size;
  const float a = 1.0;
  // switch to heap memory to allow for larger array size
  float *x1 = new float[array_size];
  float *y1 = new float[array_size];
  for(size_t i = 0; i<array_size;++i)
  {
    x1[i] = 3.3+1e-10;
    y1[i] = 3.4+1e-10;
  }
  auto start_time_cpu = std::chrono::high_resolution_clock::now();
  cpuSaxpy(x1,y1,a, array_size);
  auto end_time_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_cpu = end_time_cpu-start_time_cpu;
  printf("CPU SAXPY completed!\n");
  std::cout<<"CPU computing time:"<<time_cpu.count()<<std::endl;

  float *x2= new float[array_size];
  float *y2= new float[array_size];
  for(size_t i = 0; i<array_size;++i)
  {
    x2[i] = 3.3+1e-10;
    y2[i] = 3.4+1e-10;
  }
  auto start_time_gpu = std::chrono::high_resolution_clock::now();
  float *dx2 = nullptr;
  float *dy2 = nullptr;
  auto byteSize = array_size * sizeof(float);
  cudaMalloc(&dx2, byteSize);
  cudaMalloc(&dy2, byteSize);
  cudaMemcpy(dx2, x2, byteSize, cudaMemcpyHostToDevice);
  cudaMemcpy(dy2, y2, byteSize, cudaMemcpyHostToDevice);
  saxpyKernel<<<(array_size+TPB-1)/TPB, TPB>>>(dx2,dy2,a, array_size);
  cudaDeviceSynchronize();
  cudaMemcpy(y2, dy2, byteSize, cudaMemcpyDeviceToHost);
  cudaFree(dx2);
  cudaFree(dy2);
  auto end_time_gpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_gpu = end_time_gpu-start_time_gpu;
  printf("GPU SAXPY completed!\n");
  std::cout<<"GPU computing time:"<<time_gpu.count()<<std::endl;
  
  float diff = difference(y1, y2,array_size);
  printf("Comparison completed! The difference is %f\n", diff);
  delete x1,x2,y1,y2;
  return 0;
}