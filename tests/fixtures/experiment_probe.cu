// Small real CUDA workload for the queue acceptance test; no model required.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define PROBE_VALUE 11

__global__ void candidate(int* result) { *result = PROBE_VALUE; }

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    int* device = nullptr;
    int result = 0;
    if (cudaMalloc(&device, 1024 * 1024) != cudaSuccess) return 3;
    candidate<<<1, 1>>>(device);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    if (cudaMemcpy(&result, device, sizeof(result), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    FILE* output = std::fopen(argv[1], "w");
    if (!output) return 6;
    std::fprintf(output, "{\"candidate_value\":%d,\"cuda_passed\":%s}\n",
                 result, result == PROBE_VALUE ? "true" : "false");
    std::fclose(output);
    std::this_thread::sleep_for(std::chrono::seconds(std::atoi(argv[2])));
    return cudaFree(device) == cudaSuccess && result == PROBE_VALUE ? 0 : 7;
}
