#include <stdio.h>
#include <chrono>
#include <iostream>

__host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
  return {a.x+b.x, a.y+b.y, a.z+b.z};
}
__host__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
  return {a.x-b.x, a.y-b.y, a.z-b.z};
}

__host__ __device__ void update(float3 &p, float3 &v, const int it)
{
  p = p + v;
  v = v + make_float3(0.1f*it, 0.1f*it, 0.1f*it);
}

__global__ void simKernel(float3 *particles, float3 *velocities, int iterations)
{
  const int id = blockIdx.x*blockDim.x + threadIdx.x;
  for(size_t it = 0; it<iterations;++it)
  {
    update(particles[id], velocities[id], it);
  }
}


int main()
{
  int numParticles, numIterations, blockSize;
  std::cin >> numParticles >> numIterations >>blockSize;
  int byteSize = numParticles * sizeof(float3);

  //GPU SIMULATION:
  auto start_gpu = std::chrono::high_resolution_clock::now();

  float3 *gpu_particles;
  float3 *gpu_velocities;
  cudaMallocHost(&gpu_particles, byteSize);
  cudaMallocHost(&gpu_velocities, byteSize);

  for(size_t i=0; i<numParticles; ++i)
  {
    gpu_particles[i] = make_float3(.1f,.1f,.1f);
    gpu_velocities[i] = make_float3(.01f,.01f,.01f);
  }

  float3 *dgpu_particles;
  float3 *dgpu_velocities;
  cudaMalloc(&dgpu_particles, byteSize);
  cudaMalloc(&dgpu_velocities, byteSize);
  for(int i=0; i<numIterations; ++i)
  {
    cudaMemcpy(dgpu_particles, gpu_particles, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dgpu_velocities, gpu_velocities, byteSize, cudaMemcpyHostToDevice);

    simKernel<<<(numParticles+blockSize-1)/blockSize, blockSize>>>(dgpu_particles, 
      dgpu_velocities, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_particles, dgpu_particles, byteSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_velocities, dgpu_velocities, byteSize, cudaMemcpyDeviceToHost);
  }

  cudaFree(dgpu_particles);
  cudaFree(dgpu_velocities);
  cudaFreeHost(gpu_particles);
  cudaFreeHost(gpu_velocities);

  auto end_gpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_gpu = end_gpu-start_gpu;
  std::cout<<"GPU simulation time:"<<time_gpu.count()<<std::endl;

  return 0;
}