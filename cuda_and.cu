#include "bindex.h"

void showGPUInfo() {
    // int dev = 0;
    
    // cudaDeviceProp devProp;
    // cudaGetDeviceProperties(&devProp, dev);
    // cout << "GPU" << devProp.name << endl;
    // cout << "SM的数量：" << devProp.multiProcessorCount << endl;
    // cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << endl;
    // cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << endl;
    // cout << "每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << endl;
    // cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << endl;

    // int num;
    // cudaDeviceGetAttribute(&num, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    // cout << "每个SM的最大线程数：" << num << endl;
}


//__global__ void refineKernel(BITS* bitmap, int offset)
//{
//    int i = threadIdx.x;
//    bitmap[(i + offset)>> BITSSHIFT] ^= (1U << (BITSWIDTH - 1 - (i + offset) % BITSWIDTH));
//}
__global__ 
void refineKernel(BITS* bitmap, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        bitmap[(i) >> BITSSHIFT] ^= (1U << (BITSWIDTH - 1 - (i) % BITSWIDTH));
}

// return bitmap_a = bitmap_a & bitmap_b
__global__ 
void bit_andKernel(BITS* bitmap_a, BITS* bitmap_b, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        bitmap_a[i] &= bitmap_b[i];
}

// return bitmap_a = bitmap_a & bitmap_b
__global__ 
void bit_orKernel(BITS* bitmap_a, BITS* bitmap_b, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        bitmap_a[i] |= bitmap_b[i];
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

// memset bitmap_a to value
// bitmap_a[i] = value
__global__ 
void bitmap_memsetKernel(BITS* bitmap_a, BITS value, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        bitmap_a[i] = value;
}

class Block {
public:
    POSTYPE *pos;
    CODE *val;
    int length;
    BITS *bitmap;
};


cudaError_t refineWithCuda(CODE* bitmap, unsigned int size)
{
    CODE* dev_bitmap = 0;
    cudaError_t cudaStatus;
    dim3 blockSize(1024);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_bitmap, size * sizeof(BITS));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_bitmap, bitmap, size * sizeof(BITS), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    // Launch a kernel on the GPU with one thread for each element.
    // int threadNum = 1024;
    /*for (int i = 0; i < size; i += threadNum) {
        refineKernel <<<1, threadNum >>> (dev_bitmap, i);
    }*/
    refineKernel <<<gridSize, blockSize>>> (dev_bitmap, size);

    // refineKernel <<<1, (size % threadNum) >>> (dev_bitmap, size - (size % threadNum));

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

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(bitmap, dev_bitmap, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_bitmap);

    return cudaStatus;
}

cudaError_t CPUbitAndWithCuda(BITS* bitmap_a, BITS* bitmap_b, unsigned int size)
{
    BITS* dev_bitmap_a = 0;
    BITS* dev_bitmap_b = 0;
    cudaError_t cudaStatus;

    
    int bitnum = bits_num_needed(size);
    dim3 blockSize(1024);
    dim3 gridSize((bitnum + blockSize.x - 1) / blockSize.x);

    cout << blockSize.x << " " << blockSize.y << " " << blockSize.z << endl;
    cout << gridSize.x << " " << gridSize.y << " " << gridSize.z << endl;


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_bitmap_a, bitnum * sizeof(BITS));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_bitmap_b, bitnum * sizeof(BITS));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    timer.commonGetStartTime(3);
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_bitmap_a, bitmap_a, bitnum * sizeof(BITS), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_bitmap_b, bitmap_b, bitnum * sizeof(BITS), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    // int threadNum = 1024;
    /*for (int i = 0; i < size; i += threadNum) {
        refineKernel <<<1, threadNum >>> (dev_bitmap, i);
    }*/
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

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(bitmap_a, dev_bitmap_a, bitnum * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    timer.commonGetEndTime(3);

Error:
    cudaFree(dev_bitmap_a);
    cudaFree(dev_bitmap_b);

    return cudaStatus;
}

cudaError_t GPUbitAndWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int n)
{
    cudaError_t cudaStatus;

    int bitnum = bits_num_needed(n);
    dim3 blockSize(1024);
    dim3 gridSize((bitnum + blockSize.x - 1) / blockSize.x);

    // Choose which GPU to run on, change this on a multi-GPU system.
    // TODO:maybe this should be used only once?
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
    // cudaFree(dev_bitmap_a);
    // cudaFree(dev_bitmap_b);

    return cudaStatus;
}

cudaError_t GPUbitOrWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int n)
{
    cudaError_t cudaStatus;

    int bitnum = bits_num_needed(n);
    dim3 blockSize(1024);
    dim3 gridSize((bitnum + blockSize.x - 1) / blockSize.x);

    // Choose which GPU to run on, change this on a multi-GPU system.
    // TODO:maybe this should be used only once?
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    timer.commonGetStartTime(3);
    bit_orKernel <<<gridSize, blockSize>>> (dev_bitmap_a, dev_bitmap_b, bitnum);
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
    // cudaFree(dev_bitmap_a);
    // cudaFree(dev_bitmap_b);

    return cudaStatus;
}

cudaError_t GPUbitCopyWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int bitnum)
{
    cudaError_t cudaStatus;

    // int bitnum = bits_num_needed(n);
    dim3 blockSize(1024);
    dim3 gridSize((bitnum + blockSize.x - 1) / blockSize.x);

    // Choose which GPU to run on, change this on a multi-GPU system.
    // TODO:maybe this should be used only once?
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
    // cudaFree(dev_bitmap_a);
    // cudaFree(dev_bitmap_b);

    return cudaStatus;
}

cudaError_t GPUbitCopyNegationWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int bitnum)
{
    cudaError_t cudaStatus;

    // int bitnum = bits_num_needed(n);
    dim3 blockSize(1024);
    dim3 gridSize((bitnum + blockSize.x - 1) / blockSize.x);

    // Choose which GPU to run on, change this on a multi-GPU system.
    // TODO:maybe this should be used only once?
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
    // cudaFree(dev_bitmap_a);
    // cudaFree(dev_bitmap_b);

    return cudaStatus;
}

cudaError_t GPUbitCopySIMDWithCuda(BITS* result, BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int bitnum)
{
    cudaError_t cudaStatus;

    dim3 blockSize(1024);
    dim3 gridSize((bitnum + blockSize.x - 1) / blockSize.x);

    // Choose which GPU to run on, change this on a multi-GPU system.
    // TODO:maybe this should be used only once?
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
    // cudaFree(dev_bitmap_a);
    // cudaFree(dev_bitmap_b);

    return cudaStatus;
}

// memset bitmap_a to value
// bitmap_a[i] = value
// notice that value should be as wide as bitmap_a[i] (32 bit for example)
cudaError_t GPUbitMemsetWithCuda(BITS* dev_bitmap_a, BITS value, unsigned int n)
{
    cudaError_t cudaStatus;

    int bitnum = bits_num_needed(n);
    dim3 blockSize(1024);
    dim3 gridSize((bitnum + blockSize.x - 1) / blockSize.x);

    // Choose which GPU to run on, change this on a multi-GPU system.
    // TODO:maybe this should be used only once?
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    timer.commonGetStartTime(2);
    bitmap_memsetKernel <<<gridSize, blockSize>>> (dev_bitmap_a, value, bitnum);
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
    // cudaFree(dev_bitmap_a);
    // cudaFree(dev_bitmap_b);

    return cudaStatus;
}

void bitAndWithCPU(BITS* bitmap_a, BITS* bitmap_b, unsigned int size)
{
    int bitnum = bits_num_needed(size);
    for (int i = 0; i < bitnum; i++) {
        bitmap_a[i] &= bitmap_b[i];
    }
}

BITS *randomInitializeBitmap(int size)
{
    int bitnum = bits_num_needed(size);
    BITS *bitmap = (BITS *)malloc(bitnum * sizeof(BITS));
    for (int i = 0; i < bitnum; i++) {
        bitmap[i] = rand() % (BITS)(1 << (BITSWIDTH - 1));
    }
    return bitmap;
}

BITS *makeCopyOfBitmap(BITS *bitmap_src, int size)
{
    int bitnum = bits_num_needed(size);
    BITS *bitmap = (BITS *)malloc(bitnum * sizeof(BITS));
    for (int i = 0; i < bitnum; i++) {
        bitmap[i] = bitmap_src[i];
    }
    return bitmap;
}

bool compareBitmap(BITS *bitmap_a, BITS *bitmap_b, int size)
{
    int bitnum = bits_num_needed(size);
    for(int i = 0; i < bitnum; i++) {
        if(bitmap_a[i] != bitmap_b[i]) {
            cout << "[ERROR] error in bitnum " << i << endl;
            cout << "[ERROR] a is " << bitmap_a[i] << " and b is " << bitmap_b[i] << endl;
            return false;
        }
    }
    return true;
}

// int main()
// {

//     showGPUInfo();

//     int N = 1e8;

//     cout << "[INFO] Init bitmap by random." << endl;
//     BITS *bitmap_a = randomInitializeBitmap(N);
//     BITS *bitmap_b = randomInitializeBitmap(N);
//     BITS *bitmap_a_for_cpu = makeCopyOfBitmap(bitmap_a, N);
//     BITS *bitmap_b_for_cpu = makeCopyOfBitmap(bitmap_b, N);
//     cout << "[INFO] check bit for a is " << bitmap_a[0] << endl;
//     cout << "[INFO] check bit for b is " << bitmap_b[0] << endl;
//     if (compareBitmap(bitmap_a_for_cpu, bitmap_a, N)) {
//         cout << "[INFO] all init checks done." << endl;
//     }
//     cout << "[INFO] bitmap inited." << endl;

//     cout << "[INFO] bit and with CPU." << endl;
//     timer.commonGetStartTime(0);
//     // refine_positions(m_block.bitmap, m_block.length);
//     bitAndWithCPU(bitmap_a_for_cpu, bitmap_b_for_cpu, N);
//     timer.commonGetEndTime(0);
//     cout << "[INFO] bit and with CPU done." << endl;

//     // Add vectors in parallel.

//     cout << "[INFO] bit and with GPU." << endl;
//     timer.commonGetStartTime(1);
//     cudaError_t cudaStatus = bitAndWithCuda(bitmap_a, bitmap_b, N);
//     timer.commonGetEndTime(1);
//     cout << "[INFO] bit and with GPU done." << endl;

//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "addWithCuda failed!");
//         return 1;
//     }

//     // check if the GPU answer is correct here
//     if (compareBitmap(bitmap_a_for_cpu, bitmap_a, N)) {
//         cout << "[INFO] all checks done." << endl;
//     } else {
//         cout << "[ERROR] CPU and GPU answer are not consistent." << endl;
//     }

//     timer.showTime();
//     timer.clear();
//     // cudaDeviceReset must be called before exiting in order for profiling and
//     // tracing tools such as Nsight and Visual Profiler to show complete traces.
//     cudaStatus = cudaDeviceReset();
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaDeviceReset failed!");
//         return 1;
//     }
//     getchar();
//     return 0;
// }
