#include <stdio.h>

__global__ void kernel() {
    printf("Hello from kernel\n");
}

int main() {
    kernel << <1, 1 >> > ();
    cudaDeviceSynchronize();
    return 0;
}