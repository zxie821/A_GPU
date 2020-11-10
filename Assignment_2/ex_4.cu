#include <stdio.h>
#include <chrono>
#include <iostream>

#include <curand_kernel.h>
#include <curand.h>

__device__ bool checkCircle(float x, float y)
{
  if(sqrt((x*x) + (y*y)) <= 1.0)
  {
    return true;
  }
  return false;
}

__global__ void piKernel(int *d_res, int iterations, int totalIterations, curandState *states)
{
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id >= totalIterations)
    return;
  int localCount = 0;
  const int seed = id;
  curand_init(seed, id, 0, &states[id]);
  for(size_t it=0; it<iterations; ++it)
  {
    float x = curand_uniform(&states[id]);
    float y = curand_uniform(&states[id]);
    if(checkCircle(x,y))
    {
      ++localCount;
    }
  }
  atomicAdd(d_res, localCount);
}
int main()
{
  int blockSize, iterationPerCThread, totalIterations;
  std::cin >>blockSize >>iterationPerCThread >> totalIterations;

  auto start_gpu = std::chrono::high_resolution_clock::now();
  int counter = 0;
  int *dCounter=0;
  cudaMalloc(&dCounter, sizeof(int));
  cudaMemset(dCounter, 0, sizeof(int));

  curandState *dev_random;
  int numThread = totalIterations / iterationPerCThread;
  int numBlock = (numThread+blockSize-1)/blockSize;
  cudaMalloc((void**)&dev_random, numBlock*blockSize*sizeof(curandState));

  piKernel<<<numBlock, blockSize>>>(dCounter, iterationPerCThread,
    totalIterations, dev_random);
  cudaDeviceSynchronize();

  cudaMemcpy(&counter, dCounter, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dCounter);

  double pi = ((double)counter / (double)totalIterations) * 4.0;  
  float pi_f = ((float)counter / (float)totalIterations) * 4.0;  
  auto end_gpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_gpu = end_gpu-start_gpu;

  std::cout<<"GPU simulation time:"<<time_gpu.count()<<std::endl;
  std::cout<<"Pi result is:"<<pi<<std::endl;
  std::cout<<"float Pi result is:"<<pi_f<<std::endl;
  std::cout<<"difference between float and double Pi is:"<<pi_f-pi<<std::endl;
  return 0;
}