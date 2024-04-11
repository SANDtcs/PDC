#include <stdio.h>
#include <cuda.h>

__global__ void vector_add(int* a, int* b, int* out, int n){
    for(int i=0;i<n;i++)
        out[i] = a[i]+b[i];
}

void printArray(int* a, int n){
    for(int i=0;i<n;i++)
        printf("%d ", a[i]);
    printf("\n");
}

int main(){
    int n=5;
    int* a, *b, *out;
    int* d_a, *d_b, *d_out;

    //Host Memory
    a = (int*) malloc(sizeof(int) * n);
    b = (int*) malloc(sizeof(int) * n);
    out = (int*) malloc(sizeof(int) * n);

    for(int i=0;i<n;i++){
        a[i] = i+1;
        b[i] = i*10+1;
    }

    //Device Memory
    cudaMalloc((void**)&d_a, sizeof(int)*n);
    cudaMalloc((void**)&d_b, sizeof(int)*n);
    cudaMalloc((void**)&d_out, sizeof(int)*n);

    cudaMemcpy(d_a, a, sizeof(int)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int)*n, cudaMemcpyHostToDevice);

    vector_add<<<2, 5>>> (d_a, d_b, d_out, n);

    cudaMemcpy(out, d_out, sizeof(int)*n, cudaMemcpyDeviceToHost);

    printArray(a, n);
    printArray(b, n);
    printArray(out, n);

    return 0;
}