#ifndef UTILITIES_H
#define UTILITIES_H

#include <algorithm>
#include <chrono>
#include <random>

#include <cub/cub.cuh>

#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>

#include "cuda_buffer.cuh"


class cuda_timer final {
    cudaEvent_t timerstart, timerstop;
    cudaStream_t stream;

public:
    cuda_timer(cudaStream_t stream) : stream(stream) {
        cudaEventCreate(&timerstart);
        cudaEventCreate(&timerstop);
    }
    ~cuda_timer() {
        cudaEventDestroy(timerstart);
        cudaEventDestroy(timerstop);
    }
    void start() {
        cudaEventRecord(timerstart, stream);
    }
    void stop() {
        cudaEventRecord(timerstop, stream);
    }
    float time_ms() {
        float timerdelta;
        cudaEventSynchronize(timerstop);
        cudaEventElapsedTime(&timerdelta, timerstart, timerstop);
        return timerdelta;
    }
};


void rti_assert(bool predicate, const std::string& desc = {}) {
    if (!predicate) throw std::runtime_error(desc);
}


class zipf_index_distribution final {
    std::vector<double> cdf;
    double normalization;
    std::uniform_real_distribution<double> dis;

public:
    zipf_index_distribution(size_t size, double exp) : dis(0.0, 1.0) {
        double sum = 0;
        for (size_t i = 0; i < size; ++i) {
            sum += 1.0 / std::pow(i + 1, exp);
        }
        normalization = 1.0 / sum;

        cdf.resize(size);
        size_t cumsum = 0;
        for (size_t i = 0; i < size; ++i) {
            cdf[i] = cumsum = cumsum + normalization / std::pow(i + 1, exp);
        }
    }

    template <typename gen_type>
    size_t operator()(gen_type gen) {
        double draw = dis(gen);

        size_t offset = 0;
        for (size_t skip = size_t(1) << 63u; skip > 0; skip >>= 1) {
            if (offset + skip >= cdf.size())
                continue;
            if (draw >= cdf[offset + skip])
                continue;
            offset += skip;
        }
        return offset;
    }
};


template <typename key_type, typename iterator_type>
void draw_without_replacement(
    iterator_type output_iterator,
    size_t size,
    key_type min_key,
    key_type max_key
) {
    std::hash<key_type> h;
    std::mt19937 gen(h(min_key) ^ h(max_key));
    std::uniform_int_distribution<key_type> key_dist(min_key, max_key);

    // reservior sampling would be more appropriate here

    if (size_t(max_key - min_key + 1) > size * 2) {
        std::unordered_set<key_type> output_set;
        // large amount of keys: draw randomly, discard duplicates
        while (output_set.size() < size) {
            key_type key = key_dist(gen);
            output_set.insert(key);
        }
        std::copy(output_set.begin(), output_set.end(), output_iterator);
    } else {
        // small amount of keys: shuffle and slice
        std::vector<key_type> all_keys(max_key - min_key + 1);
        std::iota(all_keys.begin(), all_keys.end(), min_key);
        std::shuffle(all_keys.begin(), all_keys.end(), gen);
        std::copy(all_keys.begin(), all_keys.begin() + size, output_iterator);
    }
}


template <typename key_type, typename iterator_type>
void draw_skewed_without_replacement(
    iterator_type output_iterator,
    size_t center_key_count,
    size_t uniform_key_count,
    key_type min_key,
    key_type max_key
) {
    std::hash<key_type> h;
    std::mt19937 gen(h(min_key) ^ h(max_key));
    std::uniform_int_distribution<key_type> key_dist(min_key, max_key);

    size_t total_key_count = center_key_count + uniform_key_count;

    size_t key_range = max_key - min_key;
    size_t center_key = min_key + (key_range >> 1u);
    size_t center_key_offset = center_key - (center_key_count >> 1u);

    std::unordered_set<key_type> output_set;
    // generate center keys
    for (size_t i = 0; i < center_key_count; ++i) {
        auto key = key_type(center_key_offset + i);
        output_set.insert(key);
    }
    // draw remaining keys randomly
    while (output_set.size() < total_key_count) {
        auto key = key_dist(gen);
        output_set.insert(key);
    }
    std::copy(output_set.begin(), output_set.end(), output_iterator);
}


template <typename elem_type>
void sort_vector(std::vector<elem_type>& vec) {
    thrust::sort(thrust::host, vec.begin(), vec.end());
    //std::sort(vec.begin(), vec.end());
}


template <typename key_type, typename compare_type>
std::vector<std::size_t> sort_permutation(const std::vector<key_type>& vec, compare_type compare, size_t num_batches, size_t batch_size) {
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    for (size_t batch = 0; batch < num_batches; ++batch) {
        thrust::sort(thrust::host, p.begin() + batch_size * batch, p.begin() + batch_size * (batch + 1), [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
        //std::sort(p.begin() + batch_size * batch, p.begin() + batch_size * (batch + 1), [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    }
    return p;
}


template <typename key_type>
void apply_permutation(std::vector<key_type>& vec, const std::vector<std::size_t>& permutation) {
    std::vector<key_type> sorted_vec(vec.size());
    thrust::transform(thrust::host, permutation.begin(), permutation.end(), sorted_vec.begin(), [&](std::size_t i){ return vec[i]; });
    //std::transform(permutation.begin(), permutation.end(), sorted_vec.begin(), [&](std::size_t i){ return vec[i]; });
    std::swap(vec, sorted_vec);
}


template <typename value_type>
void show_vector(const std::vector<value_type>& local_buffer, size_t max_output = std::numeric_limits<size_t>::max()) {
    for (size_t i = 0; i < std::min(max_output, local_buffer.size()); ++i) {
        std::cout << local_buffer[i] << "  ";
    }
    std::cout << std::endl;
}


template <typename value_type>
void show_buffer(cuda_buffer& buffer, size_t output_count) {
    std::vector<value_type> local_buffer(output_count);
    buffer.download(local_buffer.data(), local_buffer.size());
    show_vector(local_buffer);
}


template <typename key_type, typename value_type>
void check_result(const std::vector<key_type>& lower, const std::vector<key_type>& upper, const std::vector<value_type>& expected, cuda_buffer& result_buffer) {
    std::vector<value_type> test_output(expected.size());
    result_buffer.download(test_output.data(), test_output.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != test_output[i]) {
            std::cerr << "data mismatch at index " << i << " for range " << lower[i] << "-" << upper[i] << ": expected " << expected[i] << ", but received " << test_output[i] << std::endl;
            throw std::logic_error("stop");
        }
    }
}


template <typename key_type>
size_t find_sort_buffer_size(size_t input_size) {
    size_t temp_bytes_required = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes_required, (key_type*)nullptr, (key_type*)nullptr, input_size, 0, sizeof(key_type) * 8);
    return temp_bytes_required;
}


template <typename key_type, typename value_type>
size_t find_pair_sort_buffer_size(size_t input_size) {
    size_t temp_bytes_required = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes_required, (key_type*)nullptr, (key_type*)nullptr, (value_type*)nullptr, (value_type*)nullptr, input_size, 0, sizeof(key_type) * 8);
    return temp_bytes_required;
}


template <typename key_type>
void untimed_sort(void* temp, size_t temp_bytes, const key_type* input, key_type* output, size_t input_size) {
    cub::DeviceRadixSort::SortKeys(temp, temp_bytes, input, output, input_size, 0, sizeof(key_type) * 8);
}


template <typename key_type>
void timed_sort(void* temp, size_t temp_bytes, const key_type* input, key_type* output, size_t input_size, double* time_ms) {
    cuda_timer timer(0);

    if (time_ms) {
        timer.start();
    }

    untimed_sort(temp, temp_bytes, input, output, input_size);

    if (time_ms) {
        timer.stop();
        *time_ms += timer.time_ms();
    }
}


template <typename key_type, typename value_type>
void untimed_pair_sort(void* temp, size_t temp_bytes, const key_type* ki, key_type* ko, const value_type* vi, value_type* vo, size_t input_size) {
    cub::DeviceRadixSort::SortPairs(temp, temp_bytes, ki, ko, vi, vo, input_size, 0, sizeof(key_type) * 8);
}


template <typename key_type, typename value_type>
void timed_pair_sort(void* temp, size_t temp_bytes, const key_type* ki, key_type* ko, const value_type* vi, value_type* vo, size_t input_size, double* time_ms) {
    cuda_timer timer(0);

    if (time_ms) {
        timer.start();
    }

    untimed_pair_sort(temp, temp_bytes, ki, ko, vi, vo, input_size);

    if (time_ms) {
        timer.stop();
        *time_ms += timer.time_ms();
    }
}


void init_offsets(rti_idx* buffer, size_t size, double* time_ms) {
    cuda_timer timer(0);

    if (time_ms) {
        timer.start();
    }

    lambda_kernel<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(
        [=] DEVICEQUALIFIER {
            const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid >= size) return;
            buffer[tid] = static_cast<rti_idx>(tid);
        });

    if (time_ms) {
        timer.stop();
        *time_ms += timer.time_ms();
    }
}


#endif
