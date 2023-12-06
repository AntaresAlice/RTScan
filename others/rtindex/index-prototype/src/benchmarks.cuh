#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include "input_generation.h"
#include "definitions.h"
#include "result_collector.h"
#include "utilities.h"


template <typename key_type, typename value_type>
size_t estimate_pq_memory_consumption(size_t build_size, size_t probe_size) {
    size_t input_output_memory_consumption = (build_size + probe_size) * (sizeof(key_type) + sizeof(value_type));
    size_t estimated_index_memory_consumption = build_size * std::max(9 * sizeof(float), 2 * (sizeof(key_type) + sizeof(rti_idx)));
    size_t estimated_auxiliary_memory_consumption = std::max(estimated_index_memory_consumption, probe_size * (sizeof(key_type) + sizeof(rti_idx)) * 13 / 10);
    return input_output_memory_consumption + estimated_index_memory_consumption + estimated_auxiliary_memory_consumption;
}


template <typename key_type, typename value_type>
size_t estimate_rq_memory_consumption(size_t build_size, size_t probe_size) {
    size_t base = estimate_pq_memory_consumption<key_type, value_type>(build_size, probe_size);
    size_t additional_input_memory_consumption = probe_size * sizeof(key_type);
    size_t additional_auxiliary_memory_consumption = additional_input_memory_consumption * 13 / 10;
    return base + additional_input_memory_consumption + additional_auxiliary_memory_consumption;
}


struct pq_param {
    double hit_rate;
    size_t log_num_batches;
    double build_key_uniformity;
    double probe_zipf_coefficient;
};


template <typename index_type, typename value_type>
void benchmark_point_query(rc::result_collector& rc, size_t runs, bool check_all_results) {

    using key_type = typename index_type::key_type;
    constexpr size_t max_build_size = size_t(1) << 26u;

    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);

    std::vector<std::pair<size_t, size_t>> log_build_probe_size_options {
        //{20, 27},
        //{21, 27},
        //{22, 27},
        {23, 27},
        {24, 27},
        {25, 27},

        {26, 12},
        //{26, 21},
        //{26, 22},
        //{26, 23},
        {26, 24},
        {26, 25},
        {26, 26},
        {26, 27},
    };
    std::vector<pq_param> other_parameter_options {
        // default
        {1.00,  0, 1.0, 0.0},
        // miss
        {0.50,  0, 1.0, 0.0},
        {0.10,  0, 1.0, 0.0},
        {0.01,  0, 1.0, 0.0},
        {0.00,  0, 1.0, 0.0},
        // batch
        {1.00,  4, 1.0, 0.0},
        {1.00,  8, 1.0, 0.0},
        {1.00, 12, 1.0, 0.0},
        // build skew
        {1.00,  0, 0.5, 0.0},
        {1.00,  0, 0.1, 0.0},
        {1.00,  0, 0.0, 0.0},
        // probe skew
        {1.00,  0, 1.0, 0.1},
        {1.00,  0, 1.0, 0.2},
        {1.00,  0, 1.0, 0.5},
        {1.00,  0, 1.0, 1.0},
    };
    
    //std::vector<std::pair<size_t, size_t>> log_build_probe_size_options { {26, 27} };
    //std::vector<pq_param> other_parameter_options { {1.00, 0, 1.0, 0} };

    // pre-generate unique keys
    std::vector<key_type> unique_build_key_pool(max_build_size);
    draw_without_replacement(unique_build_key_pool.begin(), unique_build_key_pool.size(), min_usable_key<key_type>, max_usable_key<key_type>);
    std::cerr << "key pool generated" << std::endl;

    for (auto log_build_probe_size : log_build_probe_size_options) {
    for (bool sort_insert : {false, true}) {
    for (bool sort_probe : {false, true}) {
    for (auto other_parameters : other_parameter_options) {

        size_t log_build_size, log_probe_size;
        std::tie(log_build_size, log_probe_size) = log_build_probe_size;
        
        double hit_rate = other_parameters.hit_rate;
        size_t log_num_batches = other_parameters.log_num_batches;
        double build_key_uniformity = other_parameters.build_key_uniformity;
        double probe_zipf_coefficient = other_parameters.probe_zipf_coefficient;

        std::cerr << "PQ " << index_type::short_description() << " (" << runs << "): " << sizeof(key_type) * 8 << "b " << log_build_size << "/" << log_probe_size << " " << log_num_batches << " " << sort_insert << " " << sort_probe << " " << hit_rate << " " << build_key_uniformity << " " << probe_zipf_coefficient << std::endl;

        // cannot have more batches than elements
        if (log_num_batches > log_probe_size)
            continue;

        size_t build_size = size_t{1} << log_build_size;
        size_t probe_size = size_t{1} << log_probe_size;
        size_t num_batches = size_t{1} << log_num_batches;
        size_t batch_size = size_t{1} << (log_probe_size - log_num_batches);

        if (free_memory < estimate_pq_memory_consumption<key_type, value_type>(build_size, probe_size))
            continue;

        rti_assert(batch_size * num_batches == probe_size);
        rti_assert(0 <= hit_rate && hit_rate <= 1);

        std::vector<key_type> build_keys, probe_keys;
        std::vector<value_type> build_values, expected_result;
        generate_point_query_input(
            build_size, probe_size, sort_insert, sort_probe, build_key_uniformity, probe_zipf_coefficient, hit_rate, num_batches, batch_size, unique_build_key_pool, check_all_results,
            build_keys, build_values, probe_keys, expected_result);

        cuda_buffer build_keys_buffer, build_values_buffer, probe_keys_buffer, result_buffer;
        build_keys_buffer.alloc_and_upload(build_keys);
        build_values_buffer.alloc_and_upload(build_values);
        probe_keys_buffer.alloc_and_upload(probe_keys);
        result_buffer.alloc(probe_size * sizeof(value_type));
        cudaMemset(result_buffer.raw_ptr, 0, result_buffer.size_in_bytes);

        double build_time_ms = 0, sort_time_ms = 0, probe_time_ms = 0;
        size_t build_bytes = 0, gpu_resident_bytes = 0;

        std::cerr << " setup complete" << std::endl;

        for (size_t run = 0; run < runs + 1; ++run) {
            // ignore first run due to weird optix behavior
            bool ignore = run == 0;

            index_type index;

            //std::cerr << " start build" << std::endl;
            index.build(build_keys_buffer.ptr<key_type>(), build_size, ignore ? nullptr : &build_time_ms, ignore ? nullptr : &build_bytes);

            gpu_resident_bytes += ignore ? 0 : index.gpu_resident_bytes();

            // alloc sort buffers
            cuda_buffer sort_temp_buffer, sorted_probe_keys_buffer;
            size_t sort_temp_bytes = 0;
            if (sort_probe) {
                sort_temp_bytes = find_sort_buffer_size<key_type>(batch_size);
                sort_temp_buffer.alloc(sort_temp_bytes);
                sorted_probe_keys_buffer.alloc(batch_size * sizeof(key_type));
            }

            for (size_t batch = 0; batch < num_batches; ++batch) {

                size_t offset = batch * batch_size;
                
                key_type* probe_keys_batch_d = probe_keys_buffer.ptr<key_type>() + offset;
                value_type* result_batch_d = result_buffer.ptr<value_type>() + offset;
    
                if (sort_probe) {
                    key_type* sorted_probe_keys_d = sorted_probe_keys_buffer.ptr<key_type>();
                    timed_sort(sort_temp_buffer.raw_ptr, sort_temp_bytes, probe_keys_batch_d, sorted_probe_keys_d, batch_size, ignore ? nullptr : &sort_time_ms);
                    probe_keys_batch_d = sorted_probe_keys_d;
                }

                //std::cerr << "  start q" << std::endl;
                cuda_timer timer(0);
                timer.start();
                index.query(build_values_buffer.ptr<value_type>(), probe_keys_batch_d, result_batch_d, batch_size, 0);
                timer.stop();
                cudaDeviceSynchronize(); CUERR
                probe_time_ms += ignore ? 0 : timer.time_ms();
            }

            if (check_all_results) check_result(probe_keys, probe_keys, expected_result, result_buffer);
        }

        std::cerr << " -> " << (probe_time_ms / runs) << " ms" << std::endl;

        rc.add("i_type", index_type::short_description());
        rc.add("i_runs", runs);
        rc.add("i_key_size", sizeof(key_type) * 8);
        rc.add("i_value_size", sizeof(value_type) * 8);
        rc.add("i_log_build_size", log_build_size);
        rc.add("i_log_probe_size", log_probe_size);
        rc.add("i_log_num_batches", log_num_batches);
        rc.add("i_sort_insert", sort_insert);
        rc.add("i_sort_probe", sort_probe);
        rc.add("i_hit_rate", hit_rate);
        rc.add("i_build_key_uniformity", build_key_uniformity);
        rc.add("i_probe_zipf_coefficient", probe_zipf_coefficient);

        rc.add("checked_result", check_all_results);
        rc.add("build_time_ms", build_time_ms / runs);
        rc.add("sort_time_ms", sort_time_ms / runs);
        rc.add("probe_time_ms", probe_time_ms / runs);
        rc.add("build_bytes", build_bytes / runs);
        rc.add("gpu_resident_bytes", gpu_resident_bytes / runs);
        rc.commit_line();
    }}}}
}


template <typename index_type, typename value_type>
void benchmark_range_query(rc::result_collector& rc, size_t runs, size_t max_log_build_size, size_t max_log_probe_size, bool check_all_results) {

    using key_type = typename index_type::key_type;

    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    
    std::vector<size_t> log_build_size_options {25};
    std::vector<size_t> log_key_range_factor_options {0, 2, 4, 6, 8, 10};
    std::vector<size_t> log_num_batches_options {0};
    std::vector<std::pair<size_t, size_t>> log_probe_size_range_query_size_option {
        {12, 10},

        {27, 0},
        {27, 2},
        {27, 4},
        {27, 6},
        {27, 8},
        {27, 10},
    };
    std::vector<bool> sort_insert_options {false};
    std::vector<bool> sort_probe_options {false, true};

    for (auto log_build_size : log_build_size_options) {
    for (auto log_probe_size_range_query_size : log_probe_size_range_query_size_option) {
    for (auto log_num_batches : log_num_batches_options) {
    for (bool sort_insert : sort_insert_options) {
    for (bool sort_probe : sort_probe_options) {
    for (auto log_key_range_factor : log_key_range_factor_options) {

        size_t log_probe_size, log_range_query_size;
        std::tie(log_probe_size, log_range_query_size) = log_probe_size_range_query_size;

        std::cerr << "RQ " << index_type::short_description() << " (" << runs << "): " << sizeof(key_type) * 8 << "b " << log_build_size << "/" << log_probe_size << "/" << log_range_query_size << " " << log_num_batches << " " << sort_insert << " " << sort_probe << " " << log_key_range_factor << std::endl;

        // cannot have more batches than elements
        if (log_num_batches > log_probe_size)
            continue;
        if (log_build_size > max_log_build_size)
            continue;
        if (log_probe_size > max_log_probe_size)
            continue;

        size_t build_size = size_t{1} << log_build_size;
        size_t probe_size = size_t{1} << log_probe_size;
        key_type range_query_size = key_type{1} << log_range_query_size;
        size_t num_batches = size_t{1} << log_num_batches;
        size_t batch_size = size_t{1} << (log_probe_size - log_num_batches);
        size_t key_range_factor = size_t{1} << log_key_range_factor;
        size_t key_range = build_size * key_range_factor;

        // make sure all keys can be represented
        if (std::log2(key_range) >= 8 * sizeof(key_type))
            continue;
        // make sure there are enough keys to allow a range query of the specified size (with added margin)
        if (range_query_size * 2 > key_range)
            continue;
        if (free_memory < estimate_rq_memory_consumption<key_type, value_type>(build_size, probe_size))
            continue;

        std::vector<key_type> build_keys, lower_keys, upper_keys;
        std::vector<value_type> build_values, expected_result;
        generate_range_query_input(
            build_size, probe_size, sort_insert, sort_probe, range_query_size, (key_type) key_range, num_batches, batch_size, check_all_results,
            build_keys, build_values, lower_keys, upper_keys, expected_result);

        cuda_buffer build_keys_buffer, build_values_buffer, lower_keys_buffer, upper_keys_buffer, result_buffer;
        build_keys_buffer.alloc_and_upload(build_keys);
        build_values_buffer.alloc_and_upload(build_values);
        lower_keys_buffer.alloc_and_upload(lower_keys);
        upper_keys_buffer.alloc_and_upload(upper_keys);
        result_buffer.alloc(probe_size * sizeof(value_type));
        cudaMemset(result_buffer.raw_ptr, 0, result_buffer.size_in_bytes);

        double build_time_ms = 0, sort_time_ms = 0, probe_time_ms = 0;
        size_t build_bytes = 0, gpu_resident_bytes = 0;

        std::cerr << " setup complete" << std::endl;

        for (size_t run = 0; run < runs; ++run) {
            index_type index;

            std::cerr << " start build" << std::endl;
            index.build(build_keys_buffer.ptr<key_type>(), build_size, &build_time_ms, &build_bytes);
            
            gpu_resident_bytes += index.gpu_resident_bytes();

            // alloc sort buffers
            cuda_buffer sort_temp_buffer, sorted_lower_keys_buffer, sorted_upper_keys_buffer;
            size_t sort_temp_bytes = 0;
            if (sort_probe) {
                sort_temp_bytes = find_pair_sort_buffer_size<key_type, key_type>(batch_size);
                sort_temp_buffer.alloc(sort_temp_bytes);
                sorted_lower_keys_buffer.alloc(batch_size * sizeof(key_type));
                sorted_upper_keys_buffer.alloc(batch_size * sizeof(key_type));
            }

            for (size_t batch = 0; batch < num_batches; ++batch) {

                size_t offset = batch * batch_size;
                
                key_type* probe_lower_batch_d = lower_keys_buffer.ptr<key_type>() + offset;
                key_type* probe_upper_batch_d = upper_keys_buffer.ptr<key_type>() + offset;
                value_type* result_batch_d = result_buffer.ptr<value_type>() + offset;
    
                if (sort_probe) {
                    key_type* sorted_probe_lower_d = sorted_lower_keys_buffer.ptr<key_type>();
                    key_type* sorted_probe_upper_d = sorted_upper_keys_buffer.ptr<key_type>();
                    timed_pair_sort(sort_temp_buffer.raw_ptr, sort_temp_bytes, probe_lower_batch_d, sorted_probe_lower_d, probe_upper_batch_d, sorted_probe_upper_d, batch_size, &sort_time_ms);
                    probe_lower_batch_d = sorted_probe_lower_d;
                    probe_upper_batch_d = sorted_probe_upper_d;
                }

                std::cerr << "  start q" << std::endl;
                cuda_timer timer(0);
                timer.start();
                index.range_query_sum(build_values_buffer.ptr<value_type>(), probe_lower_batch_d, probe_upper_batch_d, result_batch_d, batch_size, 0);
                timer.stop();
                cudaDeviceSynchronize(); CUERR
                probe_time_ms += timer.time_ms();
            }

            cudaDeviceSynchronize(); CUERR

            if (check_all_results) check_result(lower_keys, upper_keys, expected_result, result_buffer);
        }

        std::cerr << " -> " << (probe_time_ms / runs) << " ms" << std::endl;

        rc.add("i_type", index_type::short_description());
        rc.add("i_runs", runs);
        rc.add("i_key_size", sizeof(key_type) * 8);
        rc.add("i_value_size", sizeof(value_type) * 8);
        rc.add("i_log_build_size", log_build_size);
        rc.add("i_log_probe_size", log_probe_size);
        rc.add("i_log_range_query_size", log_range_query_size);
        rc.add("i_log_num_batches", log_num_batches);
        rc.add("i_sort_insert", sort_insert);
        rc.add("i_sort_probe", sort_probe);
        rc.add("i_log_key_range_factor", log_key_range_factor);
        rc.add("i_key_range", key_range);

        rc.add("checked_result", check_all_results);
        rc.add("build_time_ms", build_time_ms / runs);
        rc.add("sort_time_ms", sort_time_ms / runs);
        rc.add("probe_time_ms", probe_time_ms / runs);
        rc.add("build_bytes", build_bytes / runs);
        rc.add("gpu_resident_bytes", gpu_resident_bytes / runs);
        rc.commit_line();
    }}}}}}
}


template <typename index_type, typename value_type>
void benchmark_range_query(rc::result_collector& rc, size_t runs, bool check_all_results) {
    benchmark_range_query<index_type, value_type>(rc, runs, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), check_all_results);
}

#endif
