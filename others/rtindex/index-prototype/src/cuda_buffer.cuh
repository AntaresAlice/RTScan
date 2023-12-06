#pragma once

#include "optix_helpers.cuh"
#include "cuda_helpers.cuh"

#include <cassert>
#include <vector>

struct cuda_buffer {

    inline ~cuda_buffer() {
        free();
    }

    inline CUdeviceptr cu_ptr() const {
        return (CUdeviceptr)raw_ptr;
    }

    template <typename T>
    inline T* ptr() const {
        return (T*)raw_ptr;
    }

    void resize(size_t size) {
        if (raw_ptr) free();
        alloc(size);
    }

    void alloc(size_t size) {
        assert(raw_ptr == nullptr);
        size_in_bytes = size;
        cudaMalloc((void**)&raw_ptr, size_in_bytes); CUERR
    }

    void free() {
        cudaFree(raw_ptr); CUERR
        this->raw_ptr = nullptr;
        size_in_bytes = 0;
    }

    template<typename T>
    void alloc_and_upload(const std::vector<T> &vt) {
        alloc(vt.size() * sizeof(T));
        upload((const T*)vt.data(), vt.size());
    }

    template<typename T>
    void upload(const T* t, size_t count) {
        assert(raw_ptr != nullptr);
        assert(size_in_bytes == count * sizeof(T));
        cudaMemcpy(raw_ptr, (void *)t, count * sizeof(T), cudaMemcpyHostToDevice); CUERR
    }

    template<typename T>
    void download(T* t, size_t count) {
        assert(raw_ptr != nullptr);
        assert(size_in_bytes == count * sizeof(T));
        cudaMemcpy((void *)t, raw_ptr, count * sizeof(T), cudaMemcpyDeviceToHost); CUERR
    }

    template<typename T>
    void download_and_output(size_t count) {
        size_t actual_size = std::min(count, size_in_bytes / sizeof(T));
        std::vector<T> temp(actual_size);
        download((const T*)temp.data(), actual_size);
        for (const auto& entry : temp) {
            std::cout << entry << " ";
        }
        std::cout << std::endl;
    }

    size_t size_in_bytes{0};
    void* raw_ptr{nullptr};
};
