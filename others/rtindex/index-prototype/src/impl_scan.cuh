#ifndef SCAN_BASELINE_H
#define SCAN_BASELINE_H

#include <cooperative_groups.h>

#include "definitions.h"
#include "cuda_buffer.cuh"
#include "cuda_helpers.cuh"
#include "utilities.h"


namespace cg = cooperative_groups;


template <typename key_type, typename value_type>
GLOBALQUALIFIER
void naive_point_query_scan(const key_type* stored_keys, size_t stored_size, const value_type* value_column, const key_type* keys, value_type* result, size_t size) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    key_type key = keys[tid];
    for (size_t i = 0; i < stored_size; ++i) {
        if (key == stored_keys[i]) {
            result[tid] = value_column[i];
            return;
        }
    }
    result[tid] = not_found<value_type>;
}


template <typename key_type, typename value_type>
GLOBALQUALIFIER
void naive_range_query_scan(const key_type* stored_keys, size_t stored_size, const value_type* value_column, const key_type* lower, const key_type* upper, value_type* result, size_t size) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    key_type l = lower[tid];
    key_type u = upper[tid];

    value_type agg = 0;
    for (size_t i = 0; i < stored_size; ++i) {
        if (l <= stored_keys[i] && stored_keys[i] <= u) {
            agg += value_column[i];
        }
    }
    result[tid] = agg;
}


template <typename key_type, typename value_type, size_t cg_size>
GLOBALQUALIFIER
void coop_point_query_scan(const key_type* stored_keys, size_t stored_size, const value_type* value_column, const key_type* keys, value_type* result, size_t size) {
    
    auto tile = cg::tiled_partition<cg_size>(cg::this_thread_block());
    size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    size_t group_id = thread_id / cg_size;

    if (group_id >= size) return;

    key_type key = keys[group_id];

    for (size_t ofst = 0; ofst < stored_size; ofst += cg_size) {
        size_t pos = ofst + tile.thread_rank();
        bool found = pos < stored_size && key == stored_keys[pos];

        if (found) {
            result[group_id] = value_column[pos];
        }

        if (tile.ballot(found) != 0) {
            return;
        }
    }
    if (tile.thread_rank() == 0) {
        result[group_id] = not_found<value_type>;
    }
}


template <typename key_type, typename value_type, size_t cg_size>
GLOBALQUALIFIER
void coop_range_query_scan(const key_type* stored_keys, size_t stored_size, const value_type* value_column, const key_type* lower, const key_type* upper, value_type* result, size_t size) {

    auto tile = cg::tiled_partition<cg_size>(cg::this_thread_block());
    size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    size_t group_id = thread_id / cg_size;

    if (group_id >= size) return;

    key_type l = lower[group_id];
    key_type u = upper[group_id];

    value_type agg = 0;
    for (size_t ofst = 0; ofst < stored_size; ofst += cg_size) {
        size_t pos = ofst + tile.thread_rank();
        bool found = pos < stored_size && l <= stored_keys[pos] && stored_keys[pos] <= u;

        if (found) {
            agg += value_column[pos];
        }
    }
    for (size_t i = cg_size >> 1u; i > 0; i >>= 1u) {
        agg += tile.shfl_down(agg, i);
    }
    if (tile.thread_rank() == 0) {
        result[group_id] = agg;
    }
}


template <typename key_type_>
class scan {
public:
    using key_type = key_type_;

private:
    const key_type* stored_keys = nullptr;
    size_t stored_size = 0;
        
    constexpr static size_t cg_size_log = 4;

public:
    static std::string short_description() {
        return "scan";
    }

    size_t gpu_resident_bytes() {
        return 0;
    }

    void build(const key_type* keys, size_t size, double* build_time_ms, size_t* build_bytes) {
        stored_keys = keys;
        stored_size = size;
    }

    template <typename value_type>
    void query(const value_type* value_column, const key_type* keys, value_type* result, size_t size, cudaStream_t stream) {
        naive_point_query_scan<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(stored_keys, stored_size, value_column, keys, result, size);
        //coop_point_query_scan<<<SDIV(size << cg_size_log, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(stored_keys, stored_size, value_column, keys, result, size);
    }

    template <typename value_type>
    void range_query_sum(const value_type* value_column, const key_type* lower, const key_type* upper, value_type* result, size_t size, cudaStream_t stream) {
        naive_range_query_scan<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(stored_keys, stored_size, value_column, lower, upper, result, size);
        //coop_range_query_scan<<<SDIV(size << cg_size_log, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(stored_keys, stored_size, value_column, lower, upper, result, size);
    }

    void destroy() {
        stored_keys = nullptr;
        stored_size = 0;
    }
};

#endif
