Matrix Multiplication

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>

using namespace std;


__global__ void gpu_matrix_transpose(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols) 
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}


__global__ void matMulKernel(int* a, int* b, int* c, int N1, int N2, int N3) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N1 && col < N3) {
		int tempSum = 0;
		for (int i = 0; i < N2; i++) {
			tempSum += a[row * N2 + i] * b[i * N3 + col];
		}
		c[row * N3 + col] = tempSum;
	}
}

int main() {
	int N1 = 10, N2 = 16, N3 = 3;

	int *a;
	int *b;
	int *c;

	cudaMallocManaged(&a, N1 * N2 * sizeof(float));
	cudaMallocManaged(&b, N2 * N3 * sizeof(float));
	cudaMallocManaged(&c, N1 * N3 * sizeof(float));

	for (int i = 0; i < N1; i++) {
		for (int j = 0; j < N2; j++) {
			a[i * N2 + j] = rand() % 10;
		}
	}

	for (int i = 0; i < N2; i++) {
		for (int j = 0; j < N3; j++) {
			b[i * N3 + j] = rand() % 10;
		}
	}

	int gridRows = (N1 + 16 - 1) / 16;
	int gridCols = (N3 + 16 - 1) / 16;
	dim3 dimBlock(16, 16);
	dim3 dimGrid(gridCols, gridRows);

	cout << "Calling CUDA\n";
	matMulKernel << <dimGrid, dimBlock >> > (a, b, c, N1, N2, N3);
	
	cudaDeviceSynchronize();
	
	for (int i = 0; i < N1; i++) {
		for (int j = 0; j < N3; j++) {
			cout << c[i * N3 + j] << "\t";
		}
		cout << "\n";
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return 0;
}