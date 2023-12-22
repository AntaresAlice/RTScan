#include "rt.h"

// return bitmap_a = bitmap_a & bitmap_b
__global__ 
void bit_andKernel(BITS* bitmap_a, BITS* bitmap_b, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        bitmap_a[i] &= bitmap_b[i];
}

// copy bitmap_b to bitmap_a
// bitmap_a = bitmap_b
__global__ 
void bitmap_copyKernel(BITS* bitmap_a, BITS* bitmap_b, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        bitmap_a[i] = bitmap_b[i];
}

// copy bitmap_b's negation to bitmap_a
// bitmap_a = ~bitmap_b
__global__ 
void bitmap_copyNegationKernel(BITS* bitmap_a, BITS* bitmap_b, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        bitmap_a[i] = ~(bitmap_b[i]);
}

__global__ 
void bitmap_copySIMDKernel(BITS *result, BITS* bitmap_a, BITS* bitmap_b, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        result[i] = bitmap_b[i] & (~(bitmap_a[i]));
}

cudaError_t GPUbitAndWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int n)
{
    cudaError_t cudaStatus;

    int bitnum = bits_num_needed(n);
    dim3 blockSize(1024);
    dim3 gridSize((bitnum + blockSize.x - 1) / blockSize.x);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    timer.commonGetStartTime(2);
    bit_andKernel <<<gridSize, blockSize>>> (dev_bitmap_a, dev_bitmap_b, bitnum);
    // Check for any errors launching the kernel

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    timer.commonGetEndTime(2);


Error:
    return cudaStatus;
}

cudaError_t GPUbitCopyWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int bitnum)
{
    cudaError_t cudaStatus;

    // int bitnum = bits_num_needed(n);
    dim3 blockSize(1024);
    dim3 gridSize((bitnum + blockSize.x - 1) / blockSize.x);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    timer.commonGetStartTime(3);
    bitmap_copyKernel <<<gridSize, blockSize>>> (dev_bitmap_a, dev_bitmap_b, bitnum);
    // Check for any errors launching the kernel

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    timer.commonGetEndTime(3);


Error:
    return cudaStatus;
}

cudaError_t GPUbitCopyNegationWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int bitnum)
{
    cudaError_t cudaStatus;

    // int bitnum = bits_num_needed(n);
    dim3 blockSize(1024);
    dim3 gridSize((bitnum + blockSize.x - 1) / blockSize.x);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    timer.commonGetStartTime(3);
    bitmap_copyNegationKernel <<<gridSize, blockSize>>> (dev_bitmap_a, dev_bitmap_b, bitnum);
    // Check for any errors launching the kernel

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    timer.commonGetEndTime(3);

Error:
    return cudaStatus;
}

cudaError_t GPUbitCopySIMDWithCuda(BITS* result, BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int bitnum)
{
    cudaError_t cudaStatus;

    dim3 blockSize(1024);
    dim3 gridSize((bitnum + blockSize.x - 1) / blockSize.x);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    timer.commonGetStartTime(3);
    bitmap_copySIMDKernel <<<gridSize, blockSize>>> (result, dev_bitmap_a, dev_bitmap_b, bitnum);
    // Check for any errors launching the kernel

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    timer.commonGetEndTime(3);

Error:
    return cudaStatus;
}
