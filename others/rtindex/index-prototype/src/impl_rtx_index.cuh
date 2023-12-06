#ifndef RTX_INDEX_H
#define RTX_INDEX_H

#include <chrono>

#include "definitions.h"
#include "cuda_buffer.cuh"
#include "cuda_helpers.cuh"
#include "utilities.h"
#include "optix_wrapper.h"
#include "optix_pipeline.h"
#include "launch_parameters.cuh"


// these are initialized in main.cu
extern optix_wrapper optix;
extern optix_pipeline pipeline;


template <typename key_type>
void convert_keys_to_primitives(
    const key_type* keys_device_pointer,
    size_t key_count,
    cuda_buffer& primitive_buffer,
    double* build_time_ms
) {
    primitive_buffer.alloc(3 * key_count * sizeof(float3));
    auto buffer_pointer = primitive_buffer.ptr<float3>();

    cuda_timer timer(0);
    timer.start();
    lambda_kernel<<<SDIV(key_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>([=] DEVICEQUALIFIER {
        const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= key_count) return;

        rti_k64 key = keys_device_pointer[tid];
        float x = uint32_as_float(key & 0x3fffffu);
        float y = uint32_as_float((key >> 22u) & 0x3fffffu);
        float z = uint32_as_float(key >> 44u);

        float just_below_x = minus_eps(x);
        float just_above_x = plus_eps(x);
        float just_below_y = minus_eps(y);
        float just_above_y = plus_eps(y);
        float just_below_z = minus_eps(z);
        float just_above_z = plus_eps(z);

        buffer_pointer[3 * tid + 0] = make_float3(just_below_x, just_above_y, just_below_z);
        buffer_pointer[3 * tid + 1] = make_float3(just_below_x, just_below_y, just_below_z);
        buffer_pointer[3 * tid + 2] = make_float3(just_above_x,            y, just_above_z);
    });
    timer.stop();
    if (build_time_ms) *build_time_ms += timer.time_ms();
}


template <typename key_type>
OptixTraversableHandle build_traversable(
    const optix_wrapper& optix,
    const key_type* keys_device_pointer,
    size_t key_count,
    cuda_buffer& as_buffer,
    double* build_time_ms,
    size_t* build_bytes
) {
    cuda_buffer primitive_buffer, compacted_size_buffer, temp_buffer, uncompacted_structure_buffer;
    convert_keys_to_primitives(keys_device_pointer, key_count, primitive_buffer, build_time_ms);

    OptixTraversableHandle structure_handle{0};

    uint32_t build_input_flags[1] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
    CUdeviceptr vertices = primitive_buffer.cu_ptr();

    OptixBuildInput structure_input = {};
    structure_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    structure_input.triangleArray.numVertices         = 3 * (unsigned) key_count;
    structure_input.triangleArray.vertexBuffers       = &vertices;
    structure_input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    structure_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    structure_input.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_NONE;
    structure_input.triangleArray.flags               = build_input_flags;
    structure_input.triangleArray.numSbtRecords       = 1;

    OptixAccelBuildOptions structure_options = {};
    structure_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    structure_options.motionOptions.numKeys = 1;
    structure_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes structure_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix.optix_context,
            &structure_options,
            &structure_input,
            1,  // num_build_inputs
            &structure_buffer_sizes
    ))

    compacted_size_buffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emit_desc;
    emit_desc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = compacted_size_buffer.cu_ptr();

    uncompacted_structure_buffer.alloc(structure_buffer_sizes.outputSizeInBytes);
    temp_buffer.alloc(structure_buffer_sizes.tempSizeInBytes);
    
    cudaDeviceSynchronize();

    {
        cuda_timer timer(0);
        timer.start();
        OPTIX_CHECK(optixAccelBuild(
                optix.optix_context,
                0,
                &structure_options,
                &structure_input,
                1,
                temp_buffer.cu_ptr(),
                temp_buffer.size_in_bytes,
                uncompacted_structure_buffer.cu_ptr(),
                uncompacted_structure_buffer.size_in_bytes,
                &structure_handle,
                &emit_desc, 1
        ))
        timer.stop();
        if (build_time_ms) *build_time_ms += timer.time_ms();
    }

    uint64_t compacted_size;
    compacted_size_buffer.download(&compacted_size, 1);

    size_t temp_bytes1 = as_buffer.size_in_bytes + primitive_buffer.size_in_bytes + compacted_size_buffer.size_in_bytes + temp_buffer.size_in_bytes + uncompacted_structure_buffer.size_in_bytes;

    primitive_buffer.free();
    compacted_size_buffer.free();

    as_buffer.alloc(compacted_size);

    {
        cuda_timer timer(0);
        timer.start();
        OPTIX_CHECK(optixAccelCompact(
                optix.optix_context,
                0,
                structure_handle,
                as_buffer.cu_ptr(),
                as_buffer.size_in_bytes,
                &structure_handle));
        timer.stop();
        if (build_time_ms) *build_time_ms += timer.time_ms();
    }

    size_t temp_bytes2 = as_buffer.size_in_bytes + primitive_buffer.size_in_bytes + compacted_size_buffer.size_in_bytes + temp_buffer.size_in_bytes + uncompacted_structure_buffer.size_in_bytes;
    if (build_bytes) *build_bytes += std::max(temp_bytes1, temp_bytes2);

    return structure_handle;
}


GLOBALQUALIFIER
void setup_build_data(
    query_params* launch_params,
    OptixTraversableHandle traversable
) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1) return;

    launch_params->traversable = traversable;
}

template <typename key_type, typename value_type>
GLOBALQUALIFIER
void setup_probe_data(
    query_params* launch_params,
    const value_type* stored_values,
    bool long_keys,
    bool has_range_queries,
    const key_type* query_lower,
    const key_type* query_upper,
    value_type* result
) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1) return;

    launch_params->stored_values = stored_values;
    launch_params->long_keys = long_keys;
    launch_params->has_range_queries = has_range_queries;
    launch_params->query_lower = query_lower;
    launch_params->query_upper = query_upper;
    launch_params->result = result;
}


template <typename key_type_>
class rtx_index {
public:
    using key_type = key_type_;

private:
    cuda_buffer launch_params_buffer;
    cuda_buffer as_buffer;

public:
    static std::string short_description() {
        return "rtx_index";
    }

    size_t gpu_resident_bytes() {
        return as_buffer.size_in_bytes + launch_params_buffer.size_in_bytes;
    }

    void build(const key_type* keys, size_t size, double* build_time_ms, size_t* build_bytes) {

        launch_params_buffer.alloc(sizeof(query_params));
        if (build_bytes) *build_bytes += sizeof(query_params);

        auto traversable = build_traversable(optix, keys, size, as_buffer, build_time_ms, build_bytes); CUERR
        setup_build_data<<<1, 1>>>(launch_params_buffer.ptr<query_params>(), traversable); CUERR

        cudaDeviceSynchronize(); CUERR
    }

    template <typename value_type>
    void query(const value_type* value_column, const key_type* keys, value_type* result, size_t size, cudaStream_t stream) {

        setup_probe_data<<<1, 1, 0, stream>>>(
            launch_params_buffer.ptr<query_params>(),
            value_column,
            sizeof(key_type) == 8,
            false,
            keys,
            keys,
            result
        ); CUERR

        OPTIX_CHECK(optixLaunch(
                pipeline.pipeline,
                stream,
                launch_params_buffer.cu_ptr(),
                launch_params_buffer.size_in_bytes,
                &pipeline.sbt,
                size,
                1,
                1
        ))
    }

    template <typename value_type>
    void range_query_sum(const value_type* value_column, const key_type* lower, const key_type* upper, value_type* result, size_t size, cudaStream_t stream) {

        setup_probe_data<<<1, 1, 0, stream>>>(
            launch_params_buffer.ptr<query_params>(),
            value_column,
            sizeof(key_type) == 8,
            true,
            lower,
            upper,
            result
        ); CUERR

        OPTIX_CHECK(optixLaunch(
                pipeline.pipeline,
                stream,
                launch_params_buffer.cu_ptr(),
                launch_params_buffer.size_in_bytes,
                &pipeline.sbt,
                size,
                1,
                1
        ))
    }

    void destroy() {
        as_buffer.free();
        launch_params_buffer.free();
    }
};

#endif
