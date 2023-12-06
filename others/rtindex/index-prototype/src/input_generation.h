#ifndef INPUT_GENERATION_H
#define INPUT_GENERATION_H

#include "definitions.h"
#include "utilities.h"

#include <random>
#include <vector>


template <typename key_type, typename value_type>
void generate_point_query_input(
    size_t build_size,
    size_t probe_size,
    bool sort_insert,
    bool sort_probe,
    double build_key_uniformity,
    double probe_zipf_parameter,
    double hit_rate,
    size_t num_batches,
    size_t batch_size,
    const std::vector<key_type>& unique_build_key_pool,
    bool generate_expected_result,
    std::vector<key_type>& build_keys,
    std::vector<value_type>& build_values,
    std::vector<key_type>& probe_keys,
    std::vector<value_type>& expected_result
) {
    if (build_size > unique_build_key_pool.size()) throw std::logic_error("not enough keys");
    if (build_size + min_usable_key<key_type> > max_usable_key<key_type>) throw std::logic_error("not enough keys");

    std::uniform_int_distribution<key_type> key_dist(min_usable_key<key_type>, max_usable_key<key_type>);
    std::bernoulli_distribution coin(hit_rate);
    std::uniform_int_distribution<size_t> index_dist(0, build_size - 1);
    std::optional<zipf_index_distribution> skewed_index_dist;
    if (probe_zipf_parameter > 0) {
        // when skewed, use a zipf distribution to select the index
        skewed_index_dist.emplace(build_size, probe_zipf_parameter);
    }

    build_keys.resize(build_size);
    build_values.resize(build_size);
    probe_keys.resize(probe_size);
    expected_result.resize(probe_size);

    // build_key_uniformity = 1 => keys are uniform
    // build_key_uniformity = 0 => keys are densely concentrated in the center of their allowed value range
    if (build_key_uniformity != 1) {
        size_t uniform_key_count = size_t(build_size * build_key_uniformity);
        size_t center_key_count = build_size - uniform_key_count;
        draw_skewed_without_replacement(build_keys.begin(), center_key_count, uniform_key_count, min_usable_key<key_type>, max_usable_key<key_type>);
    } else {
        // use pre-generated uniform keys
        std::copy(unique_build_key_pool.begin(), unique_build_key_pool.begin() + build_size, build_keys.begin());
    }
    std::unordered_set<key_type> build_keys_set(build_keys.begin(), build_keys.end());

    if (sort_insert) {
        sort_vector(build_keys);
    }
    for (size_t i = 0; i < build_size; ++i) {
        build_values[i] = value_for_key<key_type, value_type>(build_keys[i]);
    }

    constexpr size_t num_threads = 32;
    #pragma omp parallel for
    for (size_t thread = 0; thread < num_threads; ++thread) {
        size_t start_index = probe_size * thread / num_threads;
        size_t end_index = probe_size * (thread + 1) / num_threads;
        std::mt19937 local_gen(thread);

        for (size_t i = start_index; i < end_index; ++i) {
            if (coin(local_gen)) {
                // pick a build key
                size_t index = probe_zipf_parameter > 0 ? skewed_index_dist.value()(local_gen) : index_dist(local_gen);
                probe_keys[i] = build_keys[index];
            } else {
                // pick any key NOT in the build set
                key_type key = key_dist(local_gen);
                while (build_keys_set.find(key) != build_keys_set.end()) {
                    key = key_dist(local_gen);
                }
                probe_keys[i] = key;
            }
        }
    }

    if (generate_expected_result) {
        std::vector<uint64_t> probe_keys_copy(probe_keys.begin(), probe_keys.end());
        if (sort_probe) {
            auto perm = sort_permutation(probe_keys_copy, std::less<key_type>(), num_batches, batch_size);
            apply_permutation(probe_keys_copy, perm);
        }

        #pragma omp parallel for
        for (size_t i = 0; i < probe_size; ++i) {
            expected_result[i] = build_keys_set.find(probe_keys_copy[i]) != build_keys_set.end() ? value_for_key<key_type, value_type>(probe_keys_copy[i]) : not_found<value_type>;
        }
    }
}


template <typename key_type, typename value_type>
void generate_range_query_input(
    size_t build_size,
    size_t probe_size,
    bool sort_insert,
    bool sort_probe,
    key_type range_query_size,
    key_type key_range,
    size_t num_batches,
    size_t batch_size,
    bool generate_expected_result,
    std::vector<key_type>& build_keys,
    std::vector<value_type>& build_values,
    std::vector<key_type>& lower_keys,
    std::vector<key_type>& upper_keys,
    std::vector<value_type>& expected_result
) {
    if (build_size > max_usable_key<key_type> - min_usable_key<key_type> + 1) throw std::logic_error("not enough keys");
    key_type min_generated_key = min_usable_key<key_type>;
    key_type max_generated_key = min_generated_key + key_range - 1;
    if (range_query_size > max_generated_key - min_generated_key + 1) throw std::logic_error("range query too large");
    std::uniform_int_distribution<key_type> range_start_dist(min_generated_key, max_generated_key - range_query_size);

    build_keys.resize(build_size);
    build_values.resize(build_size);
    lower_keys.resize(probe_size);
    upper_keys.resize(probe_size);
    expected_result.resize(probe_size);

    //std::cerr << "start generating keys" << std::endl;
    draw_without_replacement(build_keys.begin(), build_keys.size(), min_generated_key, max_generated_key);
    //std::cerr << "end generating keys" << std::endl;

    if (sort_insert) {
        sort_vector(build_keys);
    }
    for (size_t i = 0; i < build_size; ++i) {
        build_values[i] = value_for_key<key_type, value_type>(build_keys[i]);
    }

    //std::cerr << "start generating queries" << std::endl;
    constexpr size_t num_threads = 32;
    #pragma omp parallel for
    for (size_t thread = 0; thread < num_threads; ++thread) {
        size_t start_index = probe_size * thread / num_threads;
        size_t end_index = probe_size * (thread + 1) / num_threads;
        std::mt19937 local_gen(thread);

        for (size_t i = start_index; i < end_index; ++i) {
            lower_keys[i] = range_start_dist(local_gen);
            upper_keys[i] = lower_keys[i] + range_query_size - 1;
            if (upper_keys[i] < lower_keys[i])
                throw std::logic_error("upper key is smaller than lower key for some reason");
        }
    }
    //std::cerr << "end generating queries" << std::endl;

    if (generate_expected_result) {
        //std::cerr << "start generating expected results" << std::endl;
        std::vector<uint64_t> lower_copy(lower_keys.begin(), lower_keys.end());
        std::vector<uint64_t> upper_copy(upper_keys.begin(), upper_keys.end());
        if (sort_probe) {
            auto perm = sort_permutation(lower_copy, std::less<key_type>(), num_batches, batch_size);
            apply_permutation(lower_copy, perm);
            apply_permutation(upper_copy, perm);
        }

        std::vector<key_type> sorted_build_keys(build_keys.begin(), build_keys.end());
        sort_vector(sorted_build_keys);

        #pragma omp parallel for
        for (size_t i = 0; i < probe_size; ++i) {
            auto lower = std::lower_bound(sorted_build_keys.begin(), sorted_build_keys.end(), lower_copy[i]);
            auto upper = std::upper_bound(sorted_build_keys.begin(), sorted_build_keys.end(), upper_copy[i]);
            value_type agg = 0;
            for (auto it = lower; it != upper; ++it) {
                agg += value_for_key<key_type, value_type>(*it);
            }
            expected_result[i] = agg;
        }
        //std::cerr << "end generating expected results" << std::endl;
    }
}

#endif
