#include <stdio.h>
#include <chrono>
#include <iostream>

#define ARRAY_SIZE 1000
#define TPB 256
__host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
  return {a.x+b.x, a.y+b.y, a.z+b.z};
}
__host__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
  return {a.x-b.x, a.y-b.y, a.z-b.z};
}
__global__ void saxpyKernel(float *x, float *y, const float a)
{
  const int id = blockIdx.x*blockDim.x + threadIdx.x;
  y[id] = x[id] * y[id] + a;
}
__host__ __device__ void update(float3 &p, float3 &v, const int it)
{
  p = p + v;
  v = v + make_float3(0.1f*it, 0.1f*it, 0.1f*it);
}
void cpuSimulation(float3 *particles, float3 *velocities, int iterations, int particleSize)
{
    for(size_t i=0; i<iterations; ++i)
    {
        for(size_t ip=0; ip<particleSize; ++ip)
        {
            update(particles[ip], velocities[ip], i);
        }
    }
}
__global__ void simKernel(float3 *particles, float3 *velocities, int iterations)
{
  const int id = blockIdx.x*blockDim.x + threadIdx.x;
  for(size_t it = 0; it<iterations;++it)
  {
    update(particles[id], velocities[id], it);
  }
}
float difference(float3 *a, float3 *b, int size)
{
  float result = 0.0;
  for(size_t i=0; i<size;++i)
  {
    auto diff = a[i] - b[i];
    result += abs(diff.x) + abs(diff.y) + abs(diff.z);
  }
  return result;
}

int main()
{
  int numParticles, numIterations, blockSize;
  std::cin >> numParticles >> numIterations >>blockSize;
  int byteSize = numParticles * sizeof(float3);
  float3 *particles = new float3[numParticles];
  float3 *velocities = new float3[numParticles];
  for(size_t i=0; i<numParticles; ++i)
  {
    particles[i] = make_float3(.1f,.1f,.1f);
    velocities[i] = make_float3(.01f,.01f,.01f);
  }
  // CPU SIMULATION:
  auto start_cpu = std::chrono::high_resolution_clock::now();

  float3 *cpu_particles = new float3[numParticles];
  float3 *cpu_velocities = new float3[numParticles];
  memcpy(cpu_particles, particles, byteSize);
  memcpy(cpu_velocities, velocities, byteSize);

  cpuSimulation(cpu_particles, cpu_velocities, numIterations, numParticles);
  auto end_cpu = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> time_cpu = end_cpu-start_cpu;
  std::cout<<"CPU simulation time:"<<time_cpu.count()<<std::endl;


  //GPU SIMULATION:
  auto start_gpu = std::chrono::high_resolution_clock::now();

  float3 *gpu_particles = new float3[numParticles];
  float3 *gpu_velocities = new float3[numParticles];
  memcpy(gpu_particles, particles, byteSize);
  memcpy(gpu_velocities, velocities, byteSize);

  float3 *dgpu_particles = 0;
  float3 *dgpu_velocities = 0;
  cudaMalloc(&dgpu_particles, byteSize);
  cudaMalloc(&dgpu_velocities, byteSize);
  cudaMemcpy(dgpu_particles, gpu_particles, byteSize, cudaMemcpyHostToDevice);
  cudaMemcpy(dgpu_velocities, gpu_velocities, byteSize, cudaMemcpyHostToDevice);

  simKernel<<<(numParticles+blockSize-1)/blockSize, blockSize>>>(dgpu_particles, 
    dgpu_velocities, numIterations);
  cudaDeviceSynchronize();
  cudaMemcpy(gpu_particles, dgpu_particles, byteSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(gpu_velocities, dgpu_velocities, byteSize, cudaMemcpyDeviceToHost);
  cudaFree(dgpu_particles);
  cudaFree(dgpu_velocities);

  auto end_gpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_gpu = end_gpu-start_gpu;
  std::cout<<"GPU simulation time:"<<time_gpu.count()<<std::endl;

  std::cout<<"Result comparing:\n";
  std::cout<<"Position difference:"<<difference(gpu_particles, cpu_particles, numParticles)<<'\n';
  std::cout<<"Velocity difference:"<<difference(gpu_velocities, cpu_velocities, numParticles)<<std::endl;
  delete[] gpu_particles;
  delete[] gpu_velocities;
  delete[] cpu_particles;
  delete[] cpu_velocities;
  delete[] particles;
  delete[] velocities;
  return 0;
}