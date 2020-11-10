#include <stdio.h>
#include <chrono>
#include <iostream>

#define ARRAY_SIZE 1000
#define TPB 256

__global__ void saxpyKernel(float *x, float *y, const float a)
{
  const int id = blockIdx.x*blockDim.x + threadIdx.x;
  y[id] = x[id] * y[id] + a;
}

void cpuSaxpy(float *x, float *y, const float a)
{
  for(size_t i=0; i<ARRAY_SIZE;++i)
  {
    y[i] = x[i] * y[i] + a;
  }
}
float difference(float *a, float *b)
{
  float result = 0.0;
  for(size_t i=0; i<ARRAY_SIZE;++i)
  {
    result += abs(a[i] - b[i]);
  }
  return result;
}
int main()
{
  
  const float a = 1.0;
  auto start_time_cpu = std::chrono::high_resolution_clock::now();
  float x1[ARRAY_SIZE] = {0.0};
  float y1[ARRAY_SIZE] = {0.0};
  for(size_t i = 0; i<ARRAY_SIZE;++i)
  {
    x1[i] = 3.3+1e-10;
    y1[i] = 3.4+1e-10;
  }
  cpuSaxpy(x1,y1,a);
  auto end_time_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_cpu = end_time_cpu-start_time_cpu;
  printf("CPU SAXPY completed!\n");
  std::cout<<"CPU computing time:"<<time_cpu.count()<<std::endl;

  auto start_time_gpu = std::chrono::high_resolution_clock::now();
  float x2[ARRAY_SIZE] = {0.0};
  float y2[ARRAY_SIZE] = {0.0};
  for(size_t i = 0; i<ARRAY_SIZE;++i)
  {
    x2[i] = 3.3+1e-10;
    y2[i] = 3.4+1e-10;
  }
  float *dx2 = nullptr;
  float *dy2 = nullptr;
  auto byteSize = ARRAY_SIZE * sizeof(float);
  cudaMalloc(&dx2, byteSize);
  cudaMalloc(&dy2, byteSize);
  cudaMemcpy(dx2, x2, byteSize, cudaMemcpyHostToDevice);
  cudaMemcpy(dy2, y2, byteSize, cudaMemcpyHostToDevice);
  saxpyKernel<<<(ARRAY_SIZE+TPB-1)/TPB, TPB>>>(dx2,dy2,a);
  cudaDeviceSynchronize();
  cudaMemcpy(y2, dy2, byteSize, cudaMemcpyDeviceToHost);
  cudaFree(dx2);
  cudaFree(dy2);
  auto end_time_gpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_gpu = end_time_gpu-start_time_gpu;
  printf("GPU SAXPY completed!\n");
  std::cout<<"GPU computing time:"<<time_gpu.count()<<std::endl;
  
  float diff = difference(y1, y2);
  printf("Comparison completed! The difference is %f\n", diff);
  return 0;
}