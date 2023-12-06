#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <set>
#include <unordered_set>
#include <sys/time.h>
#include <cub/cub.cuh>

#include "test_configuration.h"

#include "cuda_helpers.cuh"
#include "optix_wrapper.h"
#include "optix_pipeline.h"
#include "optix_helpers.cuh"
#include "launch_parameters.cuh"


#define OVAR(x) (#x) << "=" << (x)
#define IVAR(x) ("i_" #x) << "=" << (x)

char DATA_PATH[256] = "\0";
char SCAN_FILE[256] = "\0";
size_t NUM_QUERIES = 1;

void convert_keys_to_primitives(
    const key_type* keys_device_pointer,
    size_t key_count,
    cuda_buffer& primitive_buffer,
    double& convert_time_ms
) {
    cudaEvent_t convert_start, convert_stop;
    cudaEventCreate(&convert_start);
    cudaEventCreate(&convert_stop);

#if PRIMITIVE == 0
    primitive_buffer.alloc(3 * key_count * sizeof(float3));
    auto buffer_pointer = primitive_buffer.ptr<float3>();

    cudaEventRecord(convert_start, 0);
    lambda_kernel<<<SDIV(key_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>([=] DEVICEQUALIFIER {
        const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= key_count) return;

        float x, y, z;
        key_to_coordinates(keys_device_pointer[tid], x, y, z);

        float just_below_x = minus_eps(x);
        float just_above_x = plus_eps(x);
        float just_below_y = minus_eps(y);
        float just_above_y = plus_eps(y);
        float just_below_z = minus_eps(z);
        float just_above_z = plus_eps(z);

        // triangle (-eps, eps, -eps) -- (-eps, -eps, -eps) -- (eps, 0, eps) includes the point (0, 0, 0)
        // offset this triangle in xyz direction
        buffer_pointer[3 * tid + 0] = make_float3(just_below_x, just_above_y, just_below_z);
        buffer_pointer[3 * tid + 1] = make_float3(just_below_x, just_below_y, just_below_z);
        buffer_pointer[3 * tid + 2] = make_float3(just_above_x,            y, just_above_z);
    });
    cudaEventRecord(convert_stop, 0);

#elif PRIMITIVE == 1
    primitive_buffer.alloc(key_count * sizeof(float3));
    auto buffer_pointer = primitive_buffer.ptr<float3>();

    cudaEventRecord(convert_start, 0);
    lambda_kernel<<<SDIV(key_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>([=] DEVICEQUALIFIER {
        const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= key_count) return;

        key_type key = keys_device_pointer[tid];
        float x, y, z;
        key_to_coordinates(keys_device_pointer[tid], x, y, z);

        buffer_pointer[tid] = make_float3(x, y, z);
    });
    cudaEventRecord(convert_stop, 0);

#elif PRIMITIVE == 2
    primitive_buffer.alloc(2 * key_count * sizeof(float3));
    auto buffer_pointer = primitive_buffer.ptr<float3>();

    cudaEventRecord(convert_start, 0);
    lambda_kernel<<<SDIV(key_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>([=] DEVICEQUALIFIER {
        const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= key_count) return;

        key_type key = keys_device_pointer[tid];
        float x, y, z;
        key_to_coordinates(keys_device_pointer[tid], x, y, z);

        float just_above_x = plus_eps(x);
        float just_above_y = plus_eps(y);
        float just_above_z = plus_eps(z);

        buffer_pointer[2 * tid + 0] = make_float3(x, y, z);
        buffer_pointer[2 * tid + 1] = make_float3(just_above_x, just_above_y, just_above_z);
    });
    cudaEventRecord(convert_stop, 0);

#else
#error unknown primitive type
#endif

    cudaEventSynchronize(convert_stop);
    float delta;
    cudaEventElapsedTime(&delta, convert_start, convert_stop);
    convert_time_ms = delta;
    cudaDeviceSynchronize(); CUERR
}

void setup_structure_input(OptixBuildInput& bi, void** buffer, void** secondary_buffer, size_t key_count) {

#if FORCE_SINGLE_ANYHIT == 1
    static uint32_t build_input_flags[1] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
#else
    static uint32_t build_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
#endif

#if PRIMITIVE == 0
    bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.numVertices         = 3 * (unsigned) key_count;
    bi.triangleArray.vertexBuffers       = (CUdeviceptr*) buffer;
    bi.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    bi.triangleArray.vertexStrideInBytes = sizeof(float3);
    bi.triangleArray.numIndexTriplets    = 0;
    bi.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_NONE;
    bi.triangleArray.preTransform        = 0;
    bi.triangleArray.flags               = build_input_flags;
    bi.triangleArray.numSbtRecords       = 1;

#elif PRIMITIVE == 1
    bi.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    bi.sphereArray.numVertices         = (unsigned) key_count;
    bi.sphereArray.vertexBuffers       = (CUdeviceptr*) buffer;
    bi.sphereArray.radiusBuffers       = (CUdeviceptr*) secondary_buffer;
    bi.sphereArray.vertexStrideInBytes = 0;
    bi.sphereArray.radiusStrideInBytes = 0;
    bi.sphereArray.singleRadius        = true;
    bi.sphereArray.flags               = build_input_flags;
    bi.sphereArray.numSbtRecords       = 1;

#elif PRIMITIVE == 2
    bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    bi.customPrimitiveArray.numPrimitives = (unsigned) key_count;
    bi.customPrimitiveArray.aabbBuffers   = (CUdeviceptr*) buffer;
    bi.customPrimitiveArray.strideInBytes = 0;
    bi.customPrimitiveArray.flags         = build_input_flags;
    bi.customPrimitiveArray.numSbtRecords = 1;

#else
#error unknown primitive type
#endif
}

void setup_build_input(OptixAccelBuildOptions& bi, bool update = false) {
    bi.buildFlags = OPTIX_BUILD_FLAG_NONE;
#if COMPACTION != 0
    bi.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
#endif
#if PERFORM_UPDATES != 0
    bi.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
#endif
    bi.motionOptions.numKeys = 1;
    bi.operation = update ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;
}

OptixTraversableHandle build_traversable(
    const optix_wrapper& optix,
    const key_type* keys_device_pointer,
    size_t key_count,
    cuda_buffer& as_buffer,
    size_t& uncompacted_size,
    size_t& final_size,
    double& convert_time_ms,
    double& build_time_ms,
    double& compact_time_ms,
    bool allow_update = false
) {
    uncompacted_size = 0;
    final_size = 0;
    convert_time_ms = 0;
    build_time_ms = 0;
    compact_time_ms = 0;

    cuda_buffer primitive_buffer;
    cuda_buffer secondary_primitive_buffer;
    convert_keys_to_primitives(keys_device_pointer, key_count, primitive_buffer, convert_time_ms);
#if PRIMITIVE == 1
    // we need an additional radius buffer for the spheres
    // in theory, we could append to the other buffer, but there might be alignment issues
    std::vector<float> default_radius{0.25};
    secondary_primitive_buffer.alloc_and_upload(default_radius);
#endif

    OptixTraversableHandle structure_handle{0};

    OptixBuildInput structure_input = {};
    setup_structure_input(structure_input, &primitive_buffer.raw_ptr, &secondary_primitive_buffer.raw_ptr, key_count);

    OptixAccelBuildOptions structure_options = {};
    setup_build_input(structure_options);

    OptixAccelBufferSizes structure_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix.optix_context,
            &structure_options,
            &structure_input,
            1,  // num_build_inputs
            &structure_buffer_sizes
    ))

    uncompacted_size = structure_buffer_sizes.outputSizeInBytes;

#if COMPACTION == 1
    // ==================================================================
    // prepare compaction
    // ==================================================================
    cuda_buffer compacted_size_buffer;
    compacted_size_buffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emit_desc;
    emit_desc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = compacted_size_buffer.cu_ptr();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    cuda_buffer uncompacted_structure_buffer;
    uncompacted_structure_buffer.alloc(structure_buffer_sizes.outputSizeInBytes);
#else
    final_size = uncompacted_size;
    as_buffer.alloc(structure_buffer_sizes.outputSizeInBytes);
#endif

    cuda_buffer temp_buffer;
    temp_buffer.alloc(structure_buffer_sizes.tempSizeInBytes);

    cudaEvent_t build_start, build_stop;
    cudaEventCreate(&build_start);
    cudaEventCreate(&build_stop);
    cudaEventRecord(build_start, optix.stream);

    OPTIX_CHECK(optixAccelBuild(
            optix.optix_context,
            optix.stream,
            &structure_options,
            &structure_input,
            1,
            temp_buffer.cu_ptr(),
            temp_buffer.size_in_bytes,
#if COMPACTION == 1
            uncompacted_structure_buffer.cu_ptr(),
            uncompacted_structure_buffer.size_in_bytes,
#else
            as_buffer.cu_ptr(),
            as_buffer.size_in_bytes,
#endif
            &structure_handle,
#if COMPACTION == 1
            &emit_desc, 1
#else
            nullptr, 0
#endif
    ))

    cudaEventRecord(build_stop, optix.stream);
    cudaEventSynchronize(build_stop);
    float build_delta;
    cudaEventElapsedTime(&build_delta, build_start, build_stop);
    build_time_ms = build_delta;
    cudaDeviceSynchronize(); CUERR

    primitive_buffer.free();
    secondary_primitive_buffer.free();

#if COMPACTION == 1
    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compacted_size;
    compacted_size_buffer.download(&compacted_size, 1);
    final_size = compacted_size;

    as_buffer.alloc(compacted_size);

    cudaEvent_t compact_start, compact_stop;
    cudaEventCreate(&compact_start);
    cudaEventCreate(&compact_stop);
    cudaEventRecord(compact_start, optix.stream);
    OPTIX_CHECK(optixAccelCompact(
            optix.optix_context,
            optix.stream,
            structure_handle,
            as_buffer.cu_ptr(),
            as_buffer.size_in_bytes,
            &structure_handle));

    cudaEventRecord(compact_stop, optix.stream);
    cudaEventSynchronize(compact_stop);
    float compact_delta;
    cudaEventElapsedTime(&compact_delta, compact_start, compact_stop);
    compact_time_ms = compact_delta;
    cudaDeviceSynchronize(); CUERR

    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    uncompacted_structure_buffer.free();
    compacted_size_buffer.free();
#endif
    temp_buffer.free();

    return structure_handle;
}

void update_traversable(
    const optix_wrapper& optix,
    const key_type* keys_device_pointer,
    size_t key_count,
    cuda_buffer& as_buffer,
    OptixTraversableHandle structure_handle,
    size_t& update_temp_buffer_size,
    double& update_convert_time_ms,
    double& update_time_ms
) {
    update_temp_buffer_size = 0;
    update_time_ms = 0;

    cuda_buffer primitive_buffer;
    cuda_buffer secondary_primitive_buffer;
    convert_keys_to_primitives(keys_device_pointer, key_count, primitive_buffer, update_convert_time_ms);
#if PRIMITIVE == 1
    // we need an additional radius buffer for the spheres
    // in theory, we could append to the other buffer, but there might be alignment issues
    std::vector<float> default_radius{0.25};
    secondary_primitive_buffer.alloc_and_upload(default_radius);
#endif

    OptixBuildInput structure_input = {};
    setup_structure_input(structure_input, &primitive_buffer.raw_ptr, &secondary_primitive_buffer.raw_ptr, key_count);

    OptixAccelBuildOptions structure_options = {};
    setup_build_input(structure_options, true);


    OptixAccelBufferSizes structure_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix.optix_context,
            &structure_options,
            &structure_input,
            1,  // num_build_inputs
            &structure_buffer_sizes
    ))

    update_temp_buffer_size = structure_buffer_sizes.tempUpdateSizeInBytes;
    cuda_buffer temp_buffer;
    temp_buffer.alloc(update_temp_buffer_size);

    cudaEvent_t build_start, build_stop;
    cudaEventCreate(&build_start);
    cudaEventCreate(&build_stop);
    cudaEventRecord(build_start, optix.stream);

    OPTIX_CHECK(optixAccelBuild(
            optix.optix_context,
            optix.stream,
            &structure_options,
            &structure_input,
            1,
            temp_buffer.cu_ptr(),
            temp_buffer.size_in_bytes,
            as_buffer.cu_ptr(),
            as_buffer.size_in_bytes,
            &structure_handle,
            nullptr, 0
    ))

    cudaEventRecord(build_stop, optix.stream);
    cudaEventSynchronize(build_stop);
    float build_delta;
    cudaEventElapsedTime(&build_delta, build_start, build_stop);
    update_time_ms = build_delta;
    cudaDeviceSynchronize(); CUERR
    temp_buffer.free();
}


// https://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of
template <typename key_type, typename compare_type = std::less<key_type>>
std::vector<std::size_t> sort_permutation(const std::vector<key_type>& vec, compare_type compare = {}) {
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    return p;
}
template <typename key_type>
void apply_permutation(std::vector<key_type>& vec, const std::vector<std::size_t>& permutation) {
    std::vector<key_type> sorted_vec(vec.size());
    std::transform(permutation.begin(), permutation.end(), sorted_vec.begin(), [&](std::size_t i){ return vec[i]; });
    std::swap(vec, sorted_vec);
}

void getDataFromFile(char* DATA_PATH, key_type* data, size_t N) {
    FILE* fp;
    if (!(fp = fopen(DATA_PATH, "rb"))) {
        printf("init_data_from_file: fopen(%s) faild\n", DATA_PATH);
        exit(-1);
    }
    printf("initing data from %s, num: %lu\n", DATA_PATH, N);

    if (fread(data, sizeof(uint32_t), N, fp) == 0) {
        printf("init_data_from_file: fread faild.\n");
        exit(-1);
    }
    printf("[CHECK] first num: %u  last num: %u\n", data[0], data[N - 1]);
    
    fclose(fp);
}

void getQueryFromFile(char* DATA_PATH, key_type* queries, size_t N) {
    std::ifstream fin;
    fin.open(SCAN_FILE);
    if (!fin.is_open()) {
        std::cerr << "Fail to open " << SCAN_FILE << "!" << std::endl;
        exit(-1);
    }
    
    std::string input;
    for (size_t i = 0; i < N; i++) {
        getline(fin, input);
        queries[i] = std::stoul(input);
        input.clear();
    }
    fin.close();
}

void getQueryFromFile(char* DATA_PATH, key_type* probe_lower, key_type* probe_upper, size_t N) {
    std::ifstream fin;
    fin.open(SCAN_FILE);
    if (!fin.is_open()) {
        std::cerr << "Fail to open " << SCAN_FILE << "!" << std::endl;
        exit(-1);
    }
    
    std::string input;
    for (size_t i = 0; i < N; i++) {
        getline(fin, input);
        sscanf(input.c_str(), "%u,%u", &probe_lower[i], &probe_upper[i]);
        input.clear();
    }
    fin.close();
}

bool generate_input(
    size_t num_build_keys,
    size_t num_probe_keys,
    size_t key_stride,
    size_t key_offset,
    bool reserve_keys_for_misses,
    size_t miss_percent,
    size_t out_of_range_percent,
    std::vector<key_type>& build_keys,
    std::vector<value_type>& build_values,
    std::vector<key_type>& probe_keys,
    std::vector<value_type>& expected_values
) {
    build_keys.resize(num_build_keys);
    build_values.resize(num_build_keys);
    // read data from file
    getDataFromFile(DATA_PATH, build_keys.data(), num_build_keys);
    build_values.assign(build_keys.begin(), build_keys.end());
    
    std::vector<key_type> sorted_keys(build_keys);
    std::sort(sorted_keys.begin(), sorted_keys.end());

    // generate queries
    probe_keys.resize(num_probe_keys);
    if (!strlen(SCAN_FILE)) {
        for (size_t probe_id = 0; probe_id < num_probe_keys; probe_id++) {
            probe_keys[probe_id] = (probe_id + 1) * sorted_keys[num_build_keys - 1] / num_probe_keys;
        }
    } else {
        getQueryFromFile(SCAN_FILE, probe_keys.data(), num_probe_keys);
    }
    // for(int i = 0; i < num_probe_keys; i++) {
    //     std::cout << probe_keys[i] << std::endl;
    // }

    // set expected_values
    expected_values.resize(num_probe_keys);
    for (size_t i = 0; i < num_probe_keys; i++) {
        if (std::binary_search(sorted_keys.begin(), sorted_keys.end(), probe_keys[i])) {
            expected_values[i] = probe_keys[i];
        } else {
            expected_values[i] = NOT_FOUND;
        }
    }

    return true;

    // static std::mt19937 gen(std::random_device{}());

    // if (miss_percent + out_of_range_percent > 0 && !reserve_keys_for_misses) {
    //     // cannot generate misses without reserving keys
    //     return false;
    // }

    // size_t num_miss_keys_to_probe = num_probe_keys * miss_percent / 100;
    // size_t num_out_of_range_keys_to_probe = num_probe_keys * out_of_range_percent / 100;
    // size_t num_reserved_keys = reserve_keys_for_misses ? 62u : 0u;

    // generate dense keys with the specified offset and stride
    // std::vector<key_type> generated_keys(num_build_keys);
    // for (size_t i = 0; i < generated_keys.size(); ++i) {
    //     size_t new_key = i * key_stride + key_offset;
    //     if (new_key > max_key) return false;
    //     generated_keys[i] = key_type(new_key);
    // }
    
    // shuffle the keys
    // if (reserve_keys_for_misses) {
    //     // reserve the first and the last key for out-of-range misses by swapping them to the end
    //     std::swap(generated_keys[0], generated_keys[num_build_keys - 2]); //* reserve min and max key
    //     std::shuffle(generated_keys.begin(), generated_keys.end() - 2, gen);
    // } else {
    //     std::shuffle(generated_keys.begin(), generated_keys.end(), gen);
    // }

    // copy the first part to use for building the index
    // build_keys.resize(num_build_keys - num_reserved_keys - 2);
    // build_values.resize(build_keys.size());
    // for (size_t i = 0; i < build_keys.size(); ++i) {
    //     build_keys[i] = generated_keys[i];
    //     build_values[i] = generated_keys[i] << 1u;
    // }
    
    // reserve the remaining keys to simulate misses
    // std::vector<key_type> reserved_keys(num_reserved_keys);
    // for (size_t i = 0; i < reserved_keys.size(); ++i) {
    //     reserved_keys[i] = generated_keys[i + build_keys.size()];
    // }
    // these are the out-of-range misses
    // key_type smallest_value = generated_keys[num_build_keys - 2];
    // key_type largest_value = generated_keys[num_build_keys - 1];

    // probe_keys.resize(num_probe_keys);
    // expected_values.resize(num_probe_keys);
    // fill first part with missed keys
    // for (size_t i = 0; i < num_miss_keys_to_probe; ++i) {
    //     std::uniform_int_distribution<size_t> dist(0, reserved_keys.size() - 1);
    //     size_t random_index = dist(gen);
    //     probe_keys[i] = reserved_keys[random_index];
    //     expected_values[i] = NOT_FOUND;
    // }
    // fill second part with out-of-range keys
    // for (size_t i = 0; i < num_out_of_range_keys_to_probe; ++i) {
    //     size_t offset = num_miss_keys_to_probe;
    //     probe_keys[offset + i] = i & 1 ? smallest_value : largest_value;
    //     expected_values[offset + i] = NOT_FOUND;
    // }
    // fill last part with existing keys
    // for (size_t i = 0; i < num_probe_keys - num_miss_keys_to_probe - num_out_of_range_keys_to_probe; ++i) {
    //     std::uniform_int_distribution<size_t> dist(0, build_keys.size() - 1);
    //     size_t random_index = dist(gen);
    //     size_t offset = num_miss_keys_to_probe + num_out_of_range_keys_to_probe;
    //     probe_keys[offset + i] = build_keys[random_index];
    //     expected_values[offset + i] = build_values[random_index];
    // }
    // shuffle the entire probe set
    // for (size_t i = 0; i < num_probe_keys; ++i) {
    //     std::uniform_int_distribution<size_t> dis(0, i);
    //     size_t j = dis(gen);
    //     std::swap(probe_keys[i], probe_keys[j]);
    //     std::swap(expected_values[i], expected_values[j]);
    // }

    // return true;
}

bool generate_range_query_input(
    size_t num_build_keys,
    size_t num_probe_ranges,
    size_t key_stride,
    size_t key_offset,
    size_t range_query_hit_count,
    std::vector<key_type>& build_keys,
    std::vector<value_type>& build_values,
    std::vector<key_type>& probe_lower,
    std::vector<key_type>& probe_upper,
    std::vector<value_type>& expected_values
) {
    build_keys.resize(num_build_keys);
    build_values.resize(num_build_keys);
    // read data from file
    getDataFromFile(DATA_PATH, build_keys.data(), num_build_keys);
    build_values.assign(build_keys.begin(), build_keys.end());
    
    std::vector<key_type> sorted_keys(build_keys);
    std::sort(sorted_keys.begin(), sorted_keys.end());

    // generate queries
    probe_lower.resize(num_probe_ranges);
    probe_upper.resize(num_probe_ranges);
    if (!strlen(SCAN_FILE)) {
        key_type span = UINT32_MAX / (num_probe_ranges + 1);
        std::cout << "range query span: " << span << std::endl;
        for (size_t i = 0; i < num_probe_ranges; i++) {
            probe_lower[i] = 0;
            probe_upper[i] = (i + 1) * span;
        }
    } else {
        getQueryFromFile(SCAN_FILE, probe_lower.data(), probe_upper.data(), num_probe_ranges);
    }

    // set expected_values
    expected_values.resize(num_probe_ranges); // use accumulation to vertify
    for (size_t i = 0; i < num_probe_ranges; i++) {
        auto lower = std::lower_bound(sorted_keys.begin(), sorted_keys.end(), probe_lower[i]);
        auto upper = std::upper_bound(sorted_keys.begin(), sorted_keys.end(), probe_upper[i]);
        // std::cout << "num keys: " << upper - lower + 1 << std::endl;

        value_type expected = 0;
        for (auto it = lower; it != upper; ++it) {
            expected += *it;
        }
        expected_values[i] = expected;
    }

    return true;

    // static std::mt19937 gen(std::random_device{}());

    // build_keys.resize(num_build_keys);
    // build_values.resize(build_keys.size());
    // // generate dense keys
    // for (size_t i = 0; i < num_build_keys; ++i) {
    //     size_t new_key = i * key_stride + key_offset;
    //     if (new_key > max_key) return false;
    //     build_keys[i] = key_type(new_key);
    // }
    // // shuffle keys
    // std::shuffle(build_keys.begin(), build_keys.end(), gen);
    // // generate values
    // std::map<key_type, value_type> simulated_tree;
    // for (size_t i = 0; i < build_keys.size(); ++i) {
    //     build_values[i] = value_type(build_keys[i]) << 1u;
    //     simulated_tree.emplace(build_keys[i], build_values[i]);
    // }

    // // not enough keys to meet range size requirement
    // // if (num_build_keys < range_query_hit_count) return false;
    // // size_t largest_possible_range_start = num_build_keys - range_query_hit_count;

    // probe_lower.resize(num_probe_ranges);
    // probe_upper.resize(num_probe_ranges);
    // expected_values.resize(num_probe_ranges);
    // // draw ranges uniformly
    // // for (size_t i = 0; i < num_probe_ranges; ++i) {
    // //     std::uniform_int_distribution<size_t> dist(0, largest_possible_range_start - 1);

    // //     probe_lower[i] = key_offset + dist(gen);
    // //     probe_upper[i] = probe_lower[i] + key_stride * range_query_hit_count - 1;
    // // }
    // probe_lower[0] = 0;
    // probe_upper[0] = 16777216;

    // #pragma omp parallel for
    // for (size_t i = 0; i < num_probe_ranges; ++i) {
    //     auto lower = simulated_tree.lower_bound(probe_lower[i]);
    //     auto upper = simulated_tree.upper_bound(probe_upper[i]);

    //     // pre-compute checksum
    //     value_type expected = 0;
    //     for (auto it = lower; it != upper; ++it) {
    //         expected += it->second;
    //     }
    //     expected_values[i] = expected;
    // }

    // // shuffle probes
    // // for (size_t i = 0; i < num_probe_ranges; ++i) {
    // //     std::uniform_int_distribution<size_t> dis(0, i);
    // //     size_t j = dis(gen);
    // //     std::swap(probe_lower[i], probe_lower[j]);
    // //     std::swap(probe_upper[i], probe_upper[j]);
    // //     std::swap(expected_values[i], expected_values[j]);
    // // }

    // return true;
}

void generate_updates(
    size_t num_updates,
    std::vector<key_type>& build_keys,
    std::vector<value_type>& build_values
) {
    static std::mt19937 gen(std::random_device{}());

    std::vector<size_t> indices(build_keys.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    // perform fisher-yates shuffle on a random subset of the keys
    for (size_t i = 0; i < num_updates - 1; ++i) {
        // prevent fixed points by excluding i=j
        std::uniform_int_distribution<size_t> dist(i + 1, num_updates - 1);
        size_t j = dist(gen);
        std::swap(build_keys[indices[i]], build_keys[indices[j]]);
        std::swap(build_values[indices[i]], build_values[indices[j]]);
    }
}

#define STR(x) #x
#define STRING(s) STR(s)

void start_timer(struct timeval* t) {
    gettimeofday(t, NULL);
}

void stop_timer(struct timeval* t, double* elapsed_time) {
    struct timeval end;
    gettimeofday(&end, NULL);
    *elapsed_time += (end.tv_sec - t->tv_sec) * 1000.0 + (end.tv_usec - t->tv_usec) / 1000.0;
}

void benchmark() {
    std::cout << std::setprecision(20);

    constexpr bool debug = false;
    optix_wrapper optix(debug);
    optix_pipeline pipeline(&optix);

    constexpr size_t key_offset = 0;
    constexpr size_t key_stride_log = KEY_STRIDE_LOG;
    constexpr size_t key_stride = size_t{1} << key_stride_log;
    constexpr size_t miss_percentage = MISS_PERCENTAGE;
    constexpr size_t out_of_range_percentage = OUT_OF_RANGE_PERCENTAGE;
    constexpr size_t range_query_hit_count_log = RANGE_QUERY_HIT_COUNT_LOG;
    constexpr size_t num_updates_log = NUM_UPDATES_LOG;
    constexpr size_t num_build_keys_log = NUM_BUILD_KEYS_LOG;
    constexpr size_t num_probe_keys_log = NUM_PROBE_KEYS_LOG;
    constexpr bool reserve_keys_for_misses = LEAVE_GAPS_FOR_MISSES;
    constexpr bool start_ray_at_zero = START_RAY_AT_ZERO;
    constexpr bool perform_updates = PERFORM_UPDATES;
    constexpr bool large_keys = LARGE_KEYS;

    do {
        std::cerr << "starting input generation" << std::endl;

        // size_t num_build_keys = (size_t{1} << num_build_keys_log) - 1u;
        // size_t num_probe_keys = size_t{1} << num_probe_keys_log;
        // size_t num_build_keys = (size_t) (1 << 23) - 1u;
        size_t num_build_keys = (size_t) (1e8);
        size_t num_probe_keys = NUM_QUERIES;
#if PERFORM_UPDATES != 0
        size_t num_updates = size_t{1} << num_updates_log;
#else
        size_t num_updates = 0;
#endif
        // ==================================================================
        // generate input and expected output
        // ==================================================================

        std::vector<key_type> build_keys;
        std::vector<value_type> build_values;
        std::vector<key_type> probe_lower;
        std::vector<key_type> probe_upper;
        std::vector<value_type> expected_values;
        std::vector<value_type> intersection_num;
#if RANGE_QUERY_HIT_COUNT_LOG == 0
        bool possible = generate_input(
            num_build_keys,
            num_probe_keys,
            key_stride,
            key_offset,
            reserve_keys_for_misses,
            miss_percentage,
            out_of_range_percentage,
            build_keys,
            build_values,
            probe_lower,
            expected_values
            );
#else
        bool possible = generate_range_query_input(
            num_build_keys,
            num_probe_keys,
            key_stride,
            key_offset,
            size_t{1} << range_query_hit_count_log,
            build_keys,
            build_values,
            probe_lower,
            probe_upper,
            expected_values
            );
#endif
        if (!possible) continue;

        std::cerr << "generated input" << std::endl;

        {
#if INSERT_SORTED == 1
            auto permutation = sort_permutation(build_keys, std::less<key_type>());
#elif INSERT_SORTED == -1
            auto permutation = sort_permutation(build_keys, std::greater<key_type>());
#endif
#if INSERT_SORTED == 1 || INSERT_SORTED == -1
            apply_permutation(build_keys, permutation);
            apply_permutation(build_values, permutation);
#endif
        }
        {
#if PROBE_SORTED == 1 || PROBE_SORTED == 2
            auto permutation = sort_permutation(probe_lower, std::less<key_type>());
#elif PROBE_SORTED == -1
            auto permutation = sort_permutation(probe_lower, std::greater<key_type>());
#endif
#if PROBE_SORTED == 1 || PROBE_SORTED == -1
            apply_permutation(probe_lower, permutation);
#if RANGE_QUERY_HIT_COUNT_LOG != 0
            apply_permutation(probe_upper, permutation);
#endif
            apply_permutation(expected_values, permutation);
#endif
#if PROBE_SORTED == 2
            apply_permutation(expected_values, permutation);
#endif
        }

        std::cerr << "ordered input" << std::endl;

        cuda_buffer build_keys_buffer_d, build_values_buffer_d;
        cuda_buffer probe_lower_buffer_d, probe_upper_buffer_d, result_buffer_d;
        cuda_buffer data_structure_d;
        cuda_buffer launch_params_d;
        cuda_buffer intersection_num_d;

        // ==================================================================
        // set launch parameters
        // ==================================================================

        timeval start;
        double elapsed_time = 0.0;
        timeval build_start;
        double all_build_time = 0.0;

        size_t avail_init_gpu_mem, total_gpu_mem;
        size_t avail_curr_gpu_mem, used_gpu_mem;
        cudaMemGetInfo( &avail_init_gpu_mem, &total_gpu_mem );

        start_timer(&build_start);
        build_keys_buffer_d.alloc_and_upload(build_keys);
        build_values_buffer_d.alloc_and_upload(build_values);
        start_timer(&start);
        size_t base_data_gpu_mem;
        cudaMemGetInfo( &base_data_gpu_mem, &total_gpu_mem );
        base_data_gpu_mem = avail_init_gpu_mem - base_data_gpu_mem;
        base_data_gpu_mem = 1.0 * base_data_gpu_mem / (1 << 20);

        probe_lower_buffer_d.alloc_and_upload(probe_lower);
#if RANGE_QUERY_HIT_COUNT_LOG != 0
        probe_upper_buffer_d.alloc_and_upload(probe_upper);
#endif
        stop_timer(&start, &elapsed_time);
        result_buffer_d.alloc(num_probe_keys * sizeof(value_type));
        intersection_num.push_back(0);
        intersection_num_d.alloc_and_upload(intersection_num);

        cudaDeviceSynchronize(); CUERR

        launch_parameters launch_params;

        size_t uncompacted_size = 0;
        size_t final_size = 0;
        size_t update_temp_buffer_size = 0;
        double convert_time_ms = 0;
        double build_time_ms = 0;
        double compact_time_ms = 0;
        double update_time_ms = 0;
        double update_convert_time_ms = 0;

        launch_params.traversable = build_traversable(
            optix, build_keys_buffer_d.ptr<key_type>(), build_keys.size(), data_structure_d,
            uncompacted_size, final_size, convert_time_ms, build_time_ms, compact_time_ms, perform_updates);

        std::cerr << "built structure" << std::endl;

        start_timer(&start);
        launch_params.build_keys = build_keys_buffer_d.ptr<key_type>();
        launch_params.build_values = build_values_buffer_d.ptr<value_type>();
        launch_params.query_lower = probe_lower_buffer_d.ptr<key_type>();
#if RANGE_QUERY_HIT_COUNT_LOG != 0
        launch_params.query_upper = probe_upper_buffer_d.ptr<key_type>();
#else
        launch_params.query_upper = nullptr;
#endif
        launch_params.result = result_buffer_d.ptr<value_type>();
        launch_params.intersection_num = intersection_num_d.ptr<value_type>();
        launch_params_d.alloc(sizeof(launch_params));
        launch_params_d.upload(&launch_params, 1);
        stop_timer(&start, &elapsed_time);

        cudaDeviceSynchronize(); CUERR
        
        stop_timer(&build_start, &all_build_time);
        std::cerr << "uploaded launch parameters" << std::endl;

        // ==================================================================
        // update structure
        // ==================================================================

        if (num_updates > 0) {
            generate_updates(num_updates, build_keys, build_values);
            build_keys_buffer_d.upload(build_keys.data(), build_keys.size());
            build_values_buffer_d.upload(build_values.data(), build_values.size());
            std::cerr << "generated 2^" << num_updates_log << " updates" << std::endl;
            update_traversable(
                optix, build_keys_buffer_d.ptr<key_type>(), build_keys.size(), data_structure_d,
                launch_params.traversable, update_temp_buffer_size, update_convert_time_ms, update_time_ms);
            std::cerr << "updated data structure" << std::endl;
        }

        // ==================================================================
        // sort probes
        // ==================================================================

        double sort_time_ms = 0;
#if RANGE_QUERY_HIT_COUNT_LOG == 0 && PROBE_SORTED == 2
        {
            cuda_buffer temp_d, dest_d;
            size_t temp_storage_bytes = 0;
            cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes,
                probe_lower_buffer_d.ptr<key_type>(), dest_d.ptr<key_type>(),
                probe_lower.size(), 0, sizeof(key_type)*8, optix.stream);
            temp_d.alloc(temp_storage_bytes);
            dest_d.alloc(sizeof(key_type) * probe_lower.size());

            cudaEvent_t sort_start, sort_stop;
            float sort_delta;
            cudaEventCreate(&sort_start);
            cudaEventCreate(&sort_stop);
            cudaEventRecord(sort_start, optix.stream);
            cub::DeviceRadixSort::SortKeys(temp_d.raw_ptr, temp_storage_bytes,
                probe_lower_buffer_d.ptr<key_type>(), dest_d.ptr<key_type>(),
                probe_lower.size(), 0, sizeof(key_type)*8, optix.stream);
            cudaMemcpyAsync(probe_lower_buffer_d.raw_ptr, dest_d.raw_ptr, sizeof(key_type) * probe_lower.size(), D2D, optix.stream);
            cudaEventRecord(sort_stop, optix.stream);
            cudaEventSynchronize(sort_stop);
            cudaEventElapsedTime(&sort_delta, sort_start, sort_stop);

            sort_time_ms = sort_delta;
            std::cerr << "sort: " << sort_time_ms << "ms" << std::endl;
        }
#endif
#if RANGE_QUERY_HIT_COUNT_LOG != 0 && PROBE_SORTED == 2
        {
            cuda_buffer temp_d, lower_dest_d, upper_dest_d;
            size_t temp_storage_bytes = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                probe_lower_buffer_d.ptr<key_type>(), lower_dest_d.ptr<key_type>(),
                probe_upper_buffer_d.ptr<key_type>(), upper_dest_d.ptr<key_type>(),
                probe_lower.size(), 0, sizeof(key_type)*8, optix.stream);
            temp_d.alloc(temp_storage_bytes);
            lower_dest_d.alloc(sizeof(key_type) * probe_lower.size());
            upper_dest_d.alloc(sizeof(key_type) * probe_lower.size());

            cudaEvent_t sort_start, sort_stop;
            float sort_delta;
            cudaEventCreate(&sort_start);
            cudaEventCreate(&sort_stop);
            cudaEventRecord(sort_start, optix.stream);
            cub::DeviceRadixSort::SortPairs(temp_d.raw_ptr, temp_storage_bytes,
                probe_lower_buffer_d.ptr<key_type>(), lower_dest_d.ptr<key_type>(),
                probe_upper_buffer_d.ptr<key_type>(), upper_dest_d.ptr<key_type>(),
                probe_lower.size(), 0, sizeof(key_type)*8, optix.stream);
            cudaMemcpyAsync(probe_lower_buffer_d.raw_ptr, lower_dest_d.raw_ptr, sizeof(key_type) * probe_lower.size(), D2D, optix.stream);
            cudaMemcpyAsync(probe_upper_buffer_d.raw_ptr, upper_dest_d.raw_ptr, sizeof(key_type) * probe_lower.size(), D2D, optix.stream);
            cudaEventRecord(sort_stop, optix.stream);
            cudaEventSynchronize(sort_stop);
            cudaEventElapsedTime(&sort_delta, sort_start, sort_stop);

            sort_time_ms = sort_delta;
            std::cerr << "sort with values: " << sort_time_ms << "ms" << std::endl;
        }
#endif

        // ==================================================================
        // launch
        // ==================================================================

        OPTIX_CHECK(optixLaunch(
                pipeline.pipeline,
                optix.stream,
                launch_params_d.cu_ptr(),
                launch_params_d.size_in_bytes,
                &pipeline.sbt,
                probe_lower.size(),
                1,
                1
        ))

        cudaDeviceSynchronize(); CUERR

        std::cerr << "warmup succesful" << std::endl;

        // ==================================================================
        // output
        // ==================================================================

        std::vector<value_type> output(expected_values.size());
        result_buffer_d.download(output.data(), expected_values.size());
        for (size_t i = 0; i < expected_values.size(); ++i) {
            if (output[i] != expected_values[i]) {
                std::cerr << i << ": " << expected_values[i] << " != " << output[i] << std::endl;
                throw std::exception();
            }
            //std::cout << i << " " << output[i] << std::endl;
        }
        std::cerr << "no errors detected" << std::endl;
        intersection_num_d.download(intersection_num.data(), intersection_num.size());
        std::cerr << "intersection num: " << intersection_num[0] << std::endl;

        // ==================================================================
        // timing
        // ==================================================================

        size_t runs = 1;
        double accumulated_runtime_ms = 0;

        for (size_t i = 0; i < runs; ++i) {
            cudaEvent_t timerstart, timerstop;
            float timerdelta;
            cudaEventCreate(&timerstart);
            cudaEventCreate(&timerstop);
            cudaEventRecord(timerstart, optix.stream);

            OPTIX_CHECK(optixLaunch(
                    pipeline.pipeline,
                    optix.stream,
                    launch_params_d.cu_ptr(),
                    launch_params_d.size_in_bytes,
                    &pipeline.sbt,
                    probe_lower.size(),
                    1,
                    1
            ))

            cudaEventRecord(timerstop, optix.stream);
            cudaEventSynchronize(timerstop);
            cudaEventElapsedTime(&timerdelta, timerstart, timerstop);
            accumulated_runtime_ms += timerdelta;
        }

        cudaDeviceSynchronize(); CUERR

        cudaMemGetInfo( &avail_curr_gpu_mem, &total_gpu_mem );
        used_gpu_mem = avail_init_gpu_mem - avail_curr_gpu_mem;
        used_gpu_mem = 1.0 * used_gpu_mem / (1 << 20);

        bool perpendicular_rays = PERPENDICULAR_RAYS;
        bool force_single_anyhit = FORCE_SINGLE_ANYHIT;
        bool compaction = COMPACTION;
        int64_t exponent_bias = EXPONENT_BIAS;
        double total_probe_time_ms = accumulated_runtime_ms / runs + sort_time_ms + elapsed_time;

#if PRIMITIVE == 0
        std::cout << "i_primitive=triangle,";
#elif PRIMITIVE == 1
        std::cout << "i_primitive=sphere,";
#elif PRIMITIVE == 2
        std::cout << "i_primitive=aabb,";
#endif
#if INT_TO_FLOAT_CONVERSION_MODE == 3
        std::cout << "i_key_mode=3d,";
#elif INT_TO_FLOAT_CONVERSION_MODE == 2
        std::cout << "i_key_mode=ext,";
#elif INT_TO_FLOAT_CONVERSION_MODE == 1
        std::cout << "i_key_mode=unsafe,";
#else
        std::cout << "i_key_mode=safe,";
#endif
#if INSERT_SORTED == 1
        std::cout << "i_build_mode=b_asc,";
#elif INSERT_SORTED == -1
        std::cout << "i_build_mode=b_dsc,";
#else
        std::cout << "i_build_mode=b_sfl,";
#endif
#if PROBE_SORTED == 2
        std::cout << "i_probe_mode=p_cubsort,";
#elif PROBE_SORTED == 1
        std::cout << "i_probe_mode=p_asc,";
#elif PROBE_SORTED == -1
        std::cout << "i_probe_mode=p_dsc,";
#else
        std::cout << "i_probe_mode=p_sfl,";
#endif
        std::cout << IVAR(num_build_keys_log) << ",";
        std::cout << IVAR(key_offset) << ",";
        std::cout << IVAR(key_stride_log) << ",";
        std::cout << IVAR(reserve_keys_for_misses) << ",";
        std::cout << IVAR(perform_updates) << ",";
        std::cout << IVAR(num_updates_log) << ",";
        std::cout << IVAR(num_probe_keys_log) << ",";
        std::cout << IVAR(miss_percentage) << ",";
        std::cout << IVAR(out_of_range_percentage) << ",";
        std::cout << IVAR(range_query_hit_count_log) << ",";
        std::cout << IVAR(exponent_bias) << ",";
        std::cout << IVAR(force_single_anyhit) << ",";
        std::cout << IVAR(perpendicular_rays) << ",";
        std::cout << IVAR(start_ray_at_zero) << ",";
        std::cout << IVAR(compaction) << ",";
        std::cout << IVAR(large_keys) << ",";
        std::cout << IVAR(debug) << ",";

        std::cout << OVAR(uncompacted_size) << ",";
        std::cout << OVAR(final_size) << ",";
        std::cout << OVAR(update_temp_buffer_size) << ",";
        std::cout << OVAR(convert_time_ms) << ",";
        std::cout << OVAR(build_time_ms) << ",";
        std::cout << OVAR(compact_time_ms) << ",";
        std::cout << OVAR(update_time_ms) << ",";
        std::cout << OVAR(update_convert_time_ms) << ",";
        std::cout << OVAR(sort_time_ms) << ",";
        std::cout << OVAR(total_probe_time_ms) << ",";
        std::cout << OVAR(all_build_time) << ",";
        std::cout << OVAR(base_data_gpu_mem) << ",";
        std::cout << OVAR(used_gpu_mem) << std::endl;

    } while (0);
}

int main(int argc, char* argv[]) {
    // get data and queries?
    char opt;
    while ((opt = getopt(argc, argv, "hf:s:q:")) != -1) {
        switch (opt) {
            case 'h':
                printf(
                    "Usage: %s \n"
                    "[-f <input-file>]\n",
                    argv[0]);
                exit(0);
            case 'f':
                strcpy(DATA_PATH, optarg);
                break;
            case 's':
                strcpy(SCAN_FILE, optarg);
                break;
            case 'q':
                NUM_QUERIES = std::stoi(optarg);
                break;
            default:
                printf("Error: unknown option %c\n", (char)opt);
                exit(-1);
        }
    }

    benchmark();
}
