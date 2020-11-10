#include <stdio.h>


__global__ void helloKernel()
{
  printf("Hello World! My threadId is %d\n", threadIdx.x);
}

int main()
{
  helloKernel<<<1, 256>>>();
  return 0;
}