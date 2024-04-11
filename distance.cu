#include <stdio.h>
#include <cuda.h>
#define N 16
#define D 2

// Memory Allocated in Device
__device__ float dgrid[N * D][N * D];

// Kernel Function
__global__ void findDistance(int x, int y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float n = ((i - x) * (i - x)) + ((j - y) * (j - y));
    dgrid[i][j] = sqrt(n);
}

// Main Function
void main()
{
    int i, j;

    // Memory Allocated in Host
    float hgrid[N * D][N * D];

    // 2D Grid (4 * 4 Blocks)
    dim3 dGrid(D, D);

    // 2D Block (16 * 16)
    dim3 dBlock(N, N);

    printf("Enter the x coordinate of node : ");
    scanf_s("%d", &i);
    printf("Enter the y coordinate of node : ");
    scanf_s("%d", &j);

    // Calling the kernel function with 1 - 2D_Grid, 1 - 2D_Block, 16x16 - Threads
    findDistance<<<dGrid, dBlock>>>(i, j);

    // Copy the matrix from device to host to print to console
    cudaMemcpyFromSymbol(&hgrid, dgrid, sizeof(dgrid));

    printf("Values in hgrid!\n\n");
    for (i = 0; i < N * D; i++)
    {
        for (j = 0; j < N * D; j++)
            printf("\t%.0lf", hgrid[i][j]);
        printf("\n\n");
    }
}
