#ifndef BINSEARCH_INDEX_H
#define BINSEARCH_INDEX_H

#include "definitions.h"
#include "cuda_buffer.cuh"
#include "cuda_helpers.cuh"
#include "utilities.h"


template <typename element_type>
DEVICEQUALIFIER INLINEQUALIFIER
size_t device_binary_search(element_type key, element_type* buf, size_t size) {
    size_t match_index = 0;
    for (size_t skip = size_t(1u) << 30u; skip != 0; skip >>= 1u) {
        if (match_index + skip >= size)
            continue;

        if (buf[match_index + skip] <= key)
            match_index += skip;
    }
    return match_index;
}


template <typename element_type>
DEVICEQUALIFIER INLINEQUALIFIER
size_t reverse_device_binary_search(element_type key, element_type* buf, size_t size) {
    size_t match_index = size - 1;
    for (size_t skip = size_t(1u) << 30u; skip != 0; skip >>= 1u) {
        if (match_index < skip)
            continue;

        if (buf[match_index - skip] >= key)
            match_index -= skip;
    }
    return match_index;
}


template <typename key_type_>
class binsearch {
public:
    using key_type = key_type_;

private:
    cuda_buffer sorted_keys_buffer, sorted_offsets_buffer;
    size_t stored_size = 0;

public:
    static std::string short_description() {
        return "sorted_array";
    }

    size_t gpu_resident_bytes() {
        return sorted_keys_buffer.size_in_bytes + sorted_offsets_buffer.size_in_bytes;
    }

    void build(const key_type* keys, size_t size, double* build_time_ms, size_t* build_bytes) {
        cuda_buffer temp_buffer, offsets_buffer;

        sorted_keys_buffer.alloc(sizeof(key_type) * size);
        offsets_buffer.alloc(sizeof(rti_idx) * size);
        sorted_offsets_buffer.alloc(sizeof(rti_idx) * size);
        init_offsets(offsets_buffer.ptr<rti_idx>(), size, build_time_ms);

        cudaDeviceSynchronize(); CUERR

        size_t temp_storage_bytes = find_pair_sort_buffer_size<key_type, rti_idx>(size);
        temp_buffer.alloc(temp_storage_bytes);
        timed_pair_sort(
            temp_buffer.raw_ptr, temp_storage_bytes,
            keys, sorted_keys_buffer.ptr<key_type>(), offsets_buffer.ptr<rti_idx>(), sorted_offsets_buffer.ptr<rti_idx>(), size, build_time_ms);

        if (build_bytes) *build_bytes += sorted_keys_buffer.size_in_bytes + sorted_offsets_buffer.size_in_bytes + temp_buffer.size_in_bytes + offsets_buffer.size_in_bytes;

        stored_size = size;

        cudaDeviceSynchronize(); CUERR
    }

    template <typename value_type>
    void query(const value_type* value_column, const key_type* keys, value_type* result, size_t size, cudaStream_t stream) {
        auto sorted_keys = sorted_keys_buffer.ptr<key_type>();
        auto sorted_offsets = sorted_offsets_buffer.ptr<rti_idx>();
        auto stored_size_ = stored_size;

        lambda_kernel<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
            [=] DEVICEQUALIFIER {
                const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
                if (tid >= size) return;

                key_type key = keys[tid];

                size_t match_index = reverse_device_binary_search(key, sorted_keys, stored_size_);
                if (sorted_keys[match_index] == key) {
                    result[tid] = value_column[sorted_offsets[match_index]];
                } else {
                    result[tid] = not_found<value_type>;
                }
            });
    }

    template <typename value_type>
    void range_query_sum(const value_type* value_column, const key_type* lower, const key_type* upper, value_type* result, size_t size, cudaStream_t stream) {
        auto sorted_keys = sorted_keys_buffer.ptr<key_type>();
        auto sorted_offsets = sorted_offsets_buffer.ptr<rti_idx>();
        auto stored_size_ = stored_size;

        lambda_kernel<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
            [=] DEVICEQUALIFIER {
                const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
                if (tid >= size) return;

                key_type lower_bound = lower[tid];
                key_type upper_bound = upper[tid];
                size_t lower_index = reverse_device_binary_search(lower_bound, sorted_keys, stored_size_);

                value_type agg = 0;
                for (size_t it = lower_index; it < stored_size_; ++it) {
                    if (sorted_keys[it] < lower_bound || sorted_keys[it] > upper_bound)
                        break;
                    agg += value_column[sorted_offsets[it]];
                }
                result[tid] = agg;
            });
    }

    void destroy() {
        sorted_keys_buffer.free();
        sorted_offsets_buffer.free();
        stored_size = 0;
    }
};

#endif
