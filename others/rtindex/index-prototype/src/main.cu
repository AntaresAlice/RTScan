#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <numeric>
#include <random>
#include <unordered_set>

#include "omp.h"

#include "benchmarks.cuh"
#include "cuda_helpers.cuh"
#include "optix_wrapper.h"
#include "optix_pipeline.h"
#include "optix_helpers.cuh"
#include "launch_parameters.cuh"
#include "definitions.h"
#include "result_collector.h"
#include "impl_binsearch.cuh"
#include "impl_hashtable.cuh"
#include "impl_rtx_index.cuh"
#include "impl_scan.cuh"
#include "impl_tree.cuh"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>


optix_wrapper optix;
optix_pipeline pipeline(&optix, false);


template <typename key_type, typename value_type>
void test_point_query(size_t runs, bool check_all_results, std::ostream& csv_stream = std::cout) {
    rc::result_collector rc("0", ", ");
    benchmark_point_query<rtx_index<key_type>, value_type>(rc, runs, check_all_results);
    benchmark_point_query<binsearch<key_type>, value_type>(rc, runs, check_all_results);
    benchmark_point_query<hashtable<key_type>, value_type>(rc, runs, check_all_results);
    if constexpr (sizeof(key_type) == 4) {
        benchmark_point_query<tree<key_type>, value_type>(rc, runs, check_all_results);
    }
    rc.write_csv(csv_stream);
}


template <typename key_type, typename value_type>
void test_range_query(size_t runs, bool check_all_results, std::ostream& csv_stream = std::cout) {
    rc::result_collector rc("0", ", ");
    benchmark_range_query<scan<key_type>, value_type>(rc, runs, 25, 15, check_all_results);
    benchmark_range_query<rtx_index<key_type>, value_type>(rc, runs, check_all_results);
    benchmark_range_query<binsearch<key_type>, value_type>(rc, runs, check_all_results);
    if constexpr (sizeof(key_type) == 4) {
        benchmark_range_query<tree<key_type>, value_type>(rc, runs, check_all_results);
    }
    rc.write_csv(csv_stream);
}


template <typename key_type, typename value_type>
void test_concurrency(size_t runs, std::ostream& csv_stream = std::cout) {
    rc::result_collector rc("0", ", ");
    benchmark_concurrency<rtx_index<key_type>, value_type>(rc, runs);
    benchmark_concurrency<binsearch<key_type>, value_type>(rc, runs);
    benchmark_concurrency<hashtable<key_type>, value_type>(rc, runs);
    if constexpr (sizeof(key_type) == 4) {
        benchmark_concurrency<tree<key_type>, value_type>(rc, runs);
    }
    rc.write_csv(csv_stream);
}


int main() {
    constexpr size_t runs = 5;
    constexpr bool check_all_results = true;

    {
        std::ofstream result_file("point_query_k32_v32.csv");
        test_point_query<rti_k32, rti_v32>(runs, check_all_results, result_file);
    }
    {
        std::ofstream result_file("range_query_k32_v32.csv");
        test_range_query<rti_k32, rti_v32>(runs, check_all_results, result_file);
    }
    {
        std::ofstream result_file("point_query_k64_v32.csv");
        test_point_query<rti_k64, rti_v32>(runs, check_all_results, result_file);
    }
    {
        std::ofstream result_file("range_query_k64_v32.csv");
        test_range_query<rti_k64, rti_v32>(runs, check_all_results, result_file);
    }

    return 0;
}
