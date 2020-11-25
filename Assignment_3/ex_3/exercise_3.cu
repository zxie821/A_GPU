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

__global__ void simKernel(float3 *particles, float3 *velocities, int iterations, int offset)
{
  const int id = offset + blockIdx.x*blockDim.x + threadIdx.x;
  for(size_t it = 0; it<iterations;++it)
  {
    update(particles[id], velocities[id], it);
  }
}


int main()
{
  int numParticles, numIterations, blockSize, nStream;
  std::cin >> numParticles >> numIterations >>blockSize >>nStream;
  int byteSize = numParticles * sizeof(float3);
  const int streamSize = numParticles/nStream;
  const int streamBytes = streamSize * sizeof(float);

  // cudaEvent_t startEvent, stopEvent, dummyEvent;
  // cudaEventCreate(&startEvent);
  // cudaEventCreate(&stopEvent);
  // cudaEventCreate(&dummyEvent);
  cudaStream_t *stream = new cudaStream_t[nStream];
  for(size_t i=0; i<nStream; ++i)
  {
    cudaStreamCreate(&stream[i]);
  }

  float3 *gpu_particles;
  float3 *gpu_velocities;
  cudaMallocHost(&gpu_particles, byteSize);
  cudaMallocHost(&gpu_velocities, byteSize);
  for(size_t i=0; i<numParticles; ++i)
  {
    gpu_particles[i] = make_float3(.1f,.1f,.1f);
    gpu_velocities[i] = make_float3(.01f,.01f,.01f);
  }

  //GPU SIMULATION:
  auto start_gpu = std::chrono::high_resolution_clock::now();

  float3 *dgpu_particles = 0;
  float3 *dgpu_velocities = 0;
  cudaMalloc(&dgpu_particles, byteSize);
  cudaMalloc(&dgpu_velocities, byteSize);
  for(int i=0; i<numIterations; ++i)
  {
    for(int j=0;j<nStream;++j)
    {
      int offset = j*streamSize;
      cudaMemcpyAsync(dgpu_particles+offset, gpu_particles+offset, 
        streamBytes, cudaMemcpyHostToDevice, stream[j]);
      cudaMemcpyAsync(dgpu_velocities+offset, gpu_velocities+offset, 
        streamBytes, cudaMemcpyHostToDevice, stream[j]);
  
      simKernel<<<(streamSize+blockSize-1)/blockSize, blockSize,0,stream[j]>>>(dgpu_particles, 
        dgpu_velocities, 1, offset);
      //cudaDeviceSynchronize();
      cudaMemcpyAsync(gpu_particles+offset, dgpu_particles+offset, 
        streamBytes, cudaMemcpyDeviceToHost, stream[j]);
      cudaMemcpyAsync(gpu_velocities+offset, dgpu_velocities+offset, 
        streamBytes, cudaMemcpyDeviceToHost, stream[j]);
    }
  }

  cudaFree(dgpu_particles);
  cudaFree(dgpu_velocities);
  cudaFreeHost(gpu_particles);
  cudaFreeHost(gpu_velocities);

  auto end_gpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_gpu = end_gpu-start_gpu;
  std::cout<<"GPU simulation time:"<<time_gpu.count()<<std::endl;
  for(size_t i=0; i<nStream; ++i)
  {
    cudaStreamDestroy(stream[i]);
  }
  return 0;
}