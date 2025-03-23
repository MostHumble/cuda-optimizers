#include <cuda_runtime.h>
#include <iostream>

__global__ void vanilla_sgd(float* weights, float* gradients, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop to handle large datasets
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        // Update weights using the vanilla SGD rule
        weights[i] -= learning_rate * gradients[i];
    }
}

void vanilla_sgd(float* d_weights, float* d_gradients, float learning_rate, int size, int block_size) {
    // Launch kernel
    int grid_size = (size + block_size - 1) / block_size;
    vanilla_sgd<<<grid_size, block_size>>>(d_weights, d_gradients, learning_rate, size);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

int main() {
    // Simulated data
    const int size = 1024 * 1024;
    const int block_size = 256;

    float h_weights[size], h_gradients[size];
    for (int i = 0; i < size; ++i) {
        h_weights[i] = static_cast<float>(i);
        h_gradients[i] = static_cast<float>(-i);
    }

    float* d_weights, *d_gradients;
    if (cudaMalloc((void**)&d_weights, size * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**)&d_gradients, size * sizeof(float)) != cudaSuccess) {
        std::cerr << "Memory allocation failed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_weights, h_weights, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradients, h_gradients, size * sizeof(float), cudaMemcpyHostToDevice);

    float learning_rate = 0.01f;

    vanilla_sgd(d_weights, d_gradients, learning_rate, size, block_size);

    cudaMemcpy(h_weights, d_weights, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_gradients);

    for (int i = 0; i < 10; ++i) {
        std::cout << "Weight[" << i << "] = " << h_weights[i] << std::endl;
    }

    return 0;
}
