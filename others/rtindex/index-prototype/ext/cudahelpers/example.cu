#include <vector>
#include "cuda_helpers.cuh"

#define N ((1L)<<(28))

GLOBALQUALIFIER
void reverse_kernel(int * array, size_t n) {

    size_t thid = blockDim.x*blockIdx.x+threadIdx.x;

    if (thid < n/2) {
        const int lower = array[thid];
        const int upper = array[n-thid-1];
        array[thid] = upper;
        array[n-thid-1] = lower;
    }
}

int main () {
    init_cuda_context();                                                  CUERR

    TIMERSTART(allover)

    debug_printf("this message will only be shown if NDEBUG is undefined");

    std::vector<int> host(N);
    for (size_t i = 0; i < N; i++)
        host[i] = i;

    int * device = NULL;
    cudaMalloc(&device, sizeof(int)*N);                                   CUERR
    cudaMemcpy(device, host.data(), sizeof(int)*N, H2D);                  CUERR

    TIMERSTART(kernel)
    reverse_kernel<<<SDIV(N, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(device, N);   CUERR
    TIMERSTOP(kernel)

    TIMERSTART(lambda_kernel)
    lambda_kernel<<<SDIV(N, MAXBLOCKSIZE), MAXBLOCKSIZE>>>([=] DEVICEQUALIFIER {
        size_t thid = blockDim.x*blockIdx.x+threadIdx.x;

        if (thid < N/2) {
            const int lower = device[thid];
            const int upper = device[N-thid-1];
            device[thid] = upper;
            device[N-thid-1] = lower;
        }
    });                                                                   CUERR
    TIMERSTOP(lambda_kernel)

    THROUGHPUTSTART(copy)
    cudaMemcpy(host.data(), device, sizeof(int)*N, D2H);                  CUERR
    THROUGHPUTSTOP(copy, sizeof(int), N)

    TIMERSTOP(allover)

    const std::vector<std::uint32_t> store{1, 2, 3, 4, 42, 6};
    const std::string fname{"test_file.dump"};
    dump_binary(store, fname);
    const auto load = load_binary<std::uint32_t>(fname);
    for(const auto& x : load)
    {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    std::cout << "available GPU memory: " <<                                   \
    B2GB(available_gpu_memory()) << " GB" << std::endl <<                      \
    "causing memory error by allocating 2^60 bytes" << std::endl;              \
    cudaMalloc(&device, (1L<<60));                                        CUERR
    cudaDeviceSynchronize();

}
