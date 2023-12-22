#include "bindex.h"

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

// refine, check block and set bitmap
__global__
void refineBlock(BITS *bitmap, POSTYPE *pos, CODE **raw_data, CODE *compares, int selected_id, int column_num, int n, bool inverse)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int j;
    for (int i = index; i < n; i += stride) {
        if (raw_data[selected_id][pos[i]] <= compares[selected_id * 2] || 
            raw_data[selected_id][pos[i]] >= compares[selected_id * 2 + 1]) {
            return;
        }
        for (j = 0; j < column_num; j++) {
            if (j == selected_id) continue;
            if (raw_data[j][pos[i]] <= compares[j * 2] || raw_data[j][pos[i]] >= compares[j * 2 + 1]) {
                break;
            }
        }
        if (j < column_num) continue; // Conditions not met
        if (inverse) {
            atomicAnd(&bitmap[(pos[i]) >> BITSSHIFT], ~(1U << (BITSWIDTH - 1 - (pos[i]) % BITSWIDTH)));
        } else {
            atomicOr(&bitmap[(pos[i]) >> BITSSHIFT], (1U << (BITSWIDTH - 1 - (pos[i]) % BITSWIDTH)));
        }
    }
}

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

cudaError_t GPURefineAreaWithCuda(BinDex **bindexs, BITS *dev_result_bitmap, CODE *predicate, int selected_id, int column_num, bool inverse)
{
    cudaError_t cudaStatus;

    CODE **compares = (CODE **)malloc(column_num * sizeof(CODE *));
    for (int i = 0; i < column_num; i++) {
        compares[i] = &(predicate[i * 2]);
    }

#if DEBUG_INFO == 1 
    for (int i = 0; i < column_num; i++) {
        printf("[INFO] compares[%d][0] = %u, compares[%d][1] = %u\n", i, compares[i][0], i, compares[i][1]);
    }
#endif

    CODE *dev_compares;
    cudaStatus = cudaMalloc((void**)&(dev_compares), column_num * 2 * sizeof(CODE));
    cudaStatus = cudaMemcpy(dev_compares, predicate, column_num * 2 * sizeof(CODE), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed when init dev compares!");
        exit(-1);
    }

    CODE **dev_raw_data;
    cudaStatus = cudaMalloc((void***)&(dev_raw_data), column_num * sizeof(CODE *));
    for (int i = 0; i < column_num; i++) {
        cudaStatus = cudaMemcpy(&(dev_raw_data[i]), &(bindexs[i]->rawDataInGPU), sizeof(CODE *), cudaMemcpyHostToDevice);
    }
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed when init dev compares!");
        exit(-1);
    }

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit(-1);
    }

    CODE compare = compares[selected_id][1];
    int areaIdx = in_which_area(bindexs[selected_id], compare);
    if (bindexs[selected_id]->areaStartValues[areaIdx] == compare) {
        compare = compares[selected_id][0];
        areaIdx = in_which_area(bindexs[selected_id], compare);
    }
    int blockIdx = in_which_block(bindexs[selected_id]->areas[areaIdx], compare);
    // printf("areaIdx: %d, blockIdx: %d, compare: %u\n", areaIdx, blockIdx, compare);
    // printf("bindexs[%d]->areas[%d]->blocks[%d]->val[0] = %u\n", selected_id, areaIdx, blockIdx, bindexs[selected_id]->areas[areaIdx]->blocks[blockIdx]->val[0]);

#if ONLY_REFINE == 1
    for (int j = bindexs[selected_id]->areas[areaIdx]->blockNum - 1; j >= blockIdx; j--) {
        dim3 blockSize(1024);
        dim3 gridSize((bindexs[selected_id]->areas[areaIdx]->blocks[j]->length + blockSize.x - 1) / blockSize.x);
        refineBlock<<<gridSize, blockSize>>>(dev_result_bitmap,
                                             bindexs[selected_id]->areasInGPU[areaIdx]->blocks[j]->pos,
                                             dev_raw_data,
                                             dev_compares,
                                             selected_id,
                                             column_num,
                                             bindexs[selected_id]->areas[areaIdx]->blocks[j]->length,
                                             inverse);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "launch refine block failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
            goto Error;
        }
    }
#else
    if (!inverse) {
        for (int j = 0; j <= blockIdx; j++) { // only refine the bound area
            // refine blocks[j]
            dim3 blockSize(1024);
            dim3 gridSize((bindexs[selected_id]->areas[areaIdx]->blocks[j]->length + blockSize.x - 1) / blockSize.x); 
            refineBlock <<<gridSize, blockSize>>> (dev_result_bitmap, 
                                                    bindexs[selected_id]->areasInGPU[areaIdx]->blocks[j]->pos,
                                                    dev_raw_data,
                                                    dev_compares,
                                                    selected_id,
                                                    column_num,
                                                    bindexs[selected_id]->areas[areaIdx]->blocks[j]->length,
                                                    inverse
                                                    );
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "refine block failed: %s\n", cudaGetErrorString(cudaStatus));
                exit(-1);
                goto Error;
            }
        }
    } 
    else {
        for (int j = bindexs[selected_id]->areas[areaIdx]->blockNum - 1; j >= blockIdx; j--) {
            dim3 blockSize(1024);
            dim3 gridSize((bindexs[selected_id]->areas[areaIdx]->blocks[j]->length + blockSize.x - 1) / blockSize.x);
            // refine blocks[j]
            // printf("[PreINFO] compare: %d %d\n",compares[selected_id][0],compares[selected_id][1]);
            // int val1 = bindexs[selected_id]->areas[areaIdx]->blocks[j]->val[0];
            // int val2 = bindexs[selected_id]->areas[areaIdx]->blocks[j]->val[bindexs[selected_id]->areas[areaIdx]->blocks[j]->length - 1];
            // printf("[PreINFO] block %d : [%d,%d]\n",j, val1, val2);
            // one refine scan three columns
            refineBlock  <<<gridSize, blockSize>>> (dev_result_bitmap, 
                                                    bindexs[selected_id]->areasInGPU[areaIdx]->blocks[j]->pos,
                                                    dev_raw_data,
                                                    dev_compares,
                                                    selected_id,
                                                    column_num,
                                                    bindexs[selected_id]->areas[areaIdx]->blocks[j]->length,
                                                    inverse
                                                    );
            // refineKernel <<<gridSize, blockSize>>> (dev_result_bitmap, bindexs[selected_id]->areas[areaIdx]->blocks[j]->length);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "launch refine block failed: %s\n", cudaGetErrorString(cudaStatus));
                exit(-1);
                goto Error;
            }
        }
    } 
#endif
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching refineKernel!\n", cudaStatus);
        goto Error;
    }

    if (DEBUG_INFO) printf("[INFO] refine done.\n");

Error:
    return cudaStatus;
}

cudaError_t GPURefineEqAreaWithCuda(BinDex **bindexs, BITS *dev_result_bitmap, CODE *predicate, int selected_id, int column_num, bool inverse)
{
    cudaError_t cudaStatus;

    CODE **compares = (CODE **)malloc(column_num * sizeof(CODE *));
    for (int i = 0; i < column_num; i++) {
        compares[i] = &(predicate[i * 2]);
    }

    if (DEBUG_INFO) {
        for (int i = 0; i < column_num; i++) {
            printf("[INFO] compares[%d][0] = %u, compares[%d][1] = %u\n", i, compares[i][0], i, compares[i][1]);
        }
    }

    CODE *dev_compares;
    cudaStatus = cudaMalloc((void**)&(dev_compares), column_num * 2 * sizeof(CODE));
    cudaStatus = cudaMemcpy(dev_compares, predicate, column_num * 2 * sizeof(CODE), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed when init dev compares!");
        exit(-1);
    }

    CODE **dev_raw_data;
    cudaStatus = cudaMalloc((void***)&(dev_raw_data), column_num * sizeof(CODE *));
    for (int i = 0; i < column_num; i++) {
        cudaStatus = cudaMemcpy(&(dev_raw_data[i]), &(bindexs[i]->rawDataInGPU), sizeof(CODE *), cudaMemcpyHostToDevice);
    }
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed when init dev compares!");
        exit(-1);
    }

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit(-1);
    }

    CODE refineAreaLeft;
    CODE refineAreaRight;
    if (compares[selected_id][0] < compares[selected_id][1]) {
        refineAreaLeft = compares[selected_id][0];
        refineAreaRight = compares[selected_id][1];
    }
    else {
        refineAreaLeft = compares[selected_id][1];
        refineAreaRight = compares[selected_id][0];
    }

    printf("[selected_id]: %d\n", selected_id);
    int areaIdxLeft = in_which_area(bindexs[selected_id], refineAreaLeft);
    int areaIdxRight = in_which_area(bindexs[selected_id], refineAreaRight);
    int blockIdxLeft, blockIdxRight;
    if (areaIdxLeft >= 0) {
        blockIdxLeft = in_which_block(bindexs[selected_id]->areas[areaIdxLeft], refineAreaLeft);
    } 
    blockIdxRight = in_which_block(bindexs[selected_id]->areas[areaIdxRight], refineAreaRight);

    if (areaIdxLeft == areaIdxRight) {
        // refine block from blockIdxLeft to blockIdxRight in area[areaIdxLeft]
        for (int j = blockIdxLeft; j <= blockIdxRight; j++) {
            dim3 blockSize(1024);
            dim3 gridSize((bindexs[selected_id]->areas[areaIdxLeft]->blocks[j]->length + blockSize.x - 1) / blockSize.x); 
            refineBlock  <<<gridSize, blockSize>>> (dev_result_bitmap, 
                                                    bindexs[selected_id]->areasInGPU[areaIdxLeft]->blocks[j]->pos,
                                                    dev_raw_data,
                                                    dev_compares,
                                                    selected_id,
                                                    column_num,
                                                    bindexs[selected_id]->areas[areaIdxLeft]->blocks[j]->length,
                                                    inverse
                                                    );
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "refine block failed: %s\n", cudaGetErrorString(cudaStatus));
                exit(-1);
                goto Error;
            }
        }
        return cudaStatus;
    }
    else {
        // refine block from blockIdxLeft to blockNum -1 in area[areaIdxLeft]
        int blockCount = 0;
        if (areaIdxLeft >= 0) {
            if (DEBUG_INFO) {
                blockCount += bindexs[selected_id]->areas[areaIdxLeft]->blockNum - blockIdxLeft;
            }
            for (int j = bindexs[selected_id]->areas[areaIdxLeft]->blockNum - 1; j >= blockIdxLeft; j--) {
                dim3 blockSize(1024);
                dim3 gridSize((bindexs[selected_id]->areas[areaIdxLeft]->blocks[j]->length + blockSize.x - 1) / blockSize.x);
                refineBlock  <<<gridSize, blockSize>>> (dev_result_bitmap, 
                                                        bindexs[selected_id]->areasInGPU[areaIdxLeft]->blocks[j]->pos,
                                                        dev_raw_data,
                                                        dev_compares,
                                                        selected_id,
                                                        column_num,
                                                        bindexs[selected_id]->areas[areaIdxLeft]->blocks[j]->length,
                                                        inverse
                                                        );
                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "launch refine block failed: %s\n", cudaGetErrorString(cudaStatus));
                    exit(-1);
                    goto Error;
                }
            }
        }
        for (int i = areaIdxLeft + 1; i <= areaIdxRight - 1; i++) {
            // refine block from 0 to blockNum -1 in area[areaIdxLeft]
            if (DEBUG_INFO) {
                blockCount += bindexs[selected_id]->areas[i]->blockNum;
            }  
            for (int j = bindexs[selected_id]->areas[i]->blockNum - 1; j >= 0; j--) {
            dim3 blockSize(1024);
            dim3 gridSize((bindexs[selected_id]->areas[i]->blocks[j]->length + blockSize.x - 1) / blockSize.x);
            refineBlock  <<<gridSize, blockSize>>> (dev_result_bitmap, 
                                                    bindexs[selected_id]->areasInGPU[i]->blocks[j]->pos,
                                                    dev_raw_data,
                                                    dev_compares,
                                                    selected_id,
                                                    column_num,
                                                    bindexs[selected_id]->areas[i]->blocks[j]->length,
                                                    inverse
                                                    );
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "launch refine block failed: %s\n", cudaGetErrorString(cudaStatus));
                exit(-1);
                goto Error;
            }
        }
        }
        // refine block from 0 to blockIdxRight in area[areaIdxRight]
        if (DEBUG_INFO) {
            blockCount += blockIdxRight;
        }
        for (int j = 0; j <= blockIdxRight; j++) {
            dim3 blockSize(1024);
            dim3 gridSize((bindexs[selected_id]->areas[areaIdxRight]->blocks[j]->length + blockSize.x - 1) / blockSize.x); 
            refineBlock  <<<gridSize, blockSize>>> (dev_result_bitmap, 
                                                    bindexs[selected_id]->areasInGPU[areaIdxRight]->blocks[j]->pos,
                                                    dev_raw_data,
                                                    dev_compares,
                                                    selected_id,
                                                    column_num,
                                                    bindexs[selected_id]->areas[areaIdxRight]->blocks[j]->length,
                                                    inverse
                                                    );
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "refine block failed: %s\n", cudaGetErrorString(cudaStatus));
                exit(-1);
                goto Error;
            }
        }
        printf("[CUDA Refine] %d blocks refined.\n", blockCount);
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching refineKernel!\n", cudaStatus);
        goto Error;
    }

    if (DEBUG_INFO) printf("[INFO] refine done.\n");

Error:
    return cudaStatus;
}
