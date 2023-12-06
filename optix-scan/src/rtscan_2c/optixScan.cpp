//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "state.h"
#include "timer.h"

#include <array>
#include <bitset>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <map>

#include <sutil/Camera.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#ifndef DATA_N
#define DATA_N  1e8
#endif

#define THREAD_NUM (sysconf(_SC_NPROCESSORS_ONLN) - 2)   // (20 - 2) = 18
#define col_num 3

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

typedef uint32_t CODE;
typedef uint32_t BITS;

//
//  variable
//
CODE                    *data[3];
int                      data_num = DATA_N; 
double3                  *vertices;
double3                  *triangle_vertices = nullptr;
double                   selec     = 0.5f;
double                   upper_val = UINT32_MAX * selec;                
Predicate               predicate = { .x1 = 0.0f, .x2 = upper_val, 
                                      .y1 = 0.0f, .y2 = upper_val, 
                                      .z1 = 0.0f, .z2 = upper_val };
Timer                   timer_;
ScanState               state;

extern "C" void kGenAABB(void* raw_points, double radius, unsigned int numPrims, OptixAabb *d_aabb, int column_num);

void printUsageAndExit(const char* argv0) {
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for data input\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set ray dimensions; defaults to 400x400\n";
    std::cerr << "         --depth <int>               Set ray depth; defaults to 1\n";
    std::cerr << "         --selec <double>             Set scan selectivity; defaults to 0.5f\n";
    std::cerr << "         --n <int>                   Set data num; defaults to 1e8\n";
    std::cerr << "         --primitive <int>           Set primitive type, 0 for cube, 1 for triangle with anyhit; defaults to 0\n";
    std::cerr << "         --nc                        No Comparison\n";
    exit(1);
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

void uint32_to_double3(double3* vertices) {
    for (int i = 0; i < data_num; i++) {
        vertices[i] = {
            static_cast<double>(data[0][i]),
            static_cast<double>(data[1][i]),
            static_cast<double>(data[2][i]),
        };
    }
    // printf("[CHECK FLOAT] col 0  first num: %f  last num: %f\n", vertices[0].x, vertices[data_num - 1].x);
    // printf("[CHECK FLOAT] col 1  first num: %f  last num: %f\n", vertices[0].y, vertices[data_num - 1].y);
    // printf("[CHECK FLOAT] col 2  first num: %f  last num: %f\n", vertices[0].z, vertices[data_num - 1].z);
}

void uint32_to_double3(double3* vertices, CODE **data, int data_num) {
    for (int i = 0; i < data_num; i++) {
        vertices[i] = {
            static_cast<double>(data[0][i]),
            static_cast<double>(data[1][i]),
            static_cast<double>(data[2][i]),
        };
    }
}

void uint32_to_double2(double2* vertices, CODE **data, int data_num) {
    for (int i = 0; i < data_num; i++) {
        vertices[i] = {
            static_cast<double>(data[0][i]),
            static_cast<double>(data[1][i])
        };
    }
}

// void uint32_to_double3(double3* vertices, double **data, int data_num) {
//     for (int i = 0; i < data_num; i++) {
//         vertices[i] = {
//             static_cast<double>(data[0][i]),
//             static_cast<double>(data[1][i]),
//             static_cast<double>(data[2][i]),
//         };
//     }
// }

void vertices_to_triangles(ScanState &state, double3 *vertices, double3 *triangle_vertices) {
    for (int i = 0; i < data_num; i++) {
        triangle_vertices[3 * i] = vertices[i];
        triangle_vertices[3 * i + 1] = {
            vertices[i].x, vertices[i].y + state.params.aabb_width * 2, vertices[i].z
        };
        triangle_vertices[3 * i + 2] = {
            vertices[i].x, vertices[i].y, vertices[i].z + state.params.aabb_width * 2
        };
    }

    // for (int i = 0; i < 30; i++) {
    //     printf("triangle_vertices[%d] = {%f, %f, %f}\n", i, triangle_vertices[i].x, triangle_vertices[i].y, triangle_vertices[i].z);
    // }
}

void read_data(std::string& outfile) {
    FILE* fp;
    if (!(fp = fopen(outfile.c_str(), "rb"))) {
        printf("init_data_from_file: fopen(%s) faild\n", outfile.c_str());
        exit(-1);
    }
    printf("initing data from %s\n", outfile.c_str());

    for (int col_id = 0; col_id < col_num; col_id++) {
        data[col_id] = (CODE*)malloc(data_num * sizeof(CODE));
        CODE* file_data = data[col_id];
        if (fread(file_data, sizeof(CODE), data_num, fp) == 0) {
            printf("init_data_from_file: fread faild.\n");
            exit(-1);
        }
        // CODE first_num = file_data[0];
        // CODE last_num = file_data[data_num - 1];
        // printf("[CHECK] col %d  first num: %u  last num: %u\n", col_id, first_num, last_num);
    }

    vertices = (double3*) malloc(data_num * sizeof(double3));
    uint32_to_double3(vertices);
    
    for (int col_id = 0; col_id < col_num; col_id++) {
        free(data[col_id]);
    }
}

inline void set_result_cpu(unsigned int *result, int pos, bool inverse) {
    assert(pos >= 0 && pos < 32);
    if (inverse) {
        *result &= ~(1 << (31 - pos));
    } else {
        *result |= (1 << (31 - pos));
    }
}

void scan_with_cpu(const double3 *vertices, int data_num,
                   unsigned int *result, Predicate &predicate, bool inverse) {
    fprintf(stdout, "[OptiX] go into scan_with_cpu\n");
    predicate.print();
    for (int i = 0; i < data_num; i++) {
        if (vertices[i].x > predicate.x1 && vertices[i].x < predicate.x2 &&
            vertices[i].y > predicate.y1 && vertices[i].y < predicate.y2 &&
            vertices[i].z > predicate.z1 && vertices[i].z < predicate.z2) {
            set_result_cpu(result + i / 32, i % 32, inverse);
        }
    }
}

void scan_with_cpu(const double2 *vertices, int data_num,
                   unsigned int *result, Predicate &predicate, bool inverse) {
    fprintf(stdout, "[OptiX] go into scan_with_cpu\n");
    predicate.print();
    for (int i = 0; i < data_num; i++) {
        if (vertices[i].x > predicate.x1 && vertices[i].x < predicate.x2 &&
            vertices[i].y > predicate.y1 && vertices[i].y < predicate.y2) {
            set_result_cpu(result + i / 32, i % 32, inverse);
        }
    }
}

int find_different_pos(unsigned int result_cpu, unsigned int result_gpu) {
    int pos = 0;
    for (int i = 0; i < 32; i++) {
        unsigned int val = 1 << (31 - i);
        if ((result_cpu & val) != (result_gpu & val)) {
            pos = i;
            break;
        }
    }
    return pos;
}

void compare_result(unsigned int* result_cpu, unsigned int* result_gpu, int data_num, Predicate &p) {
    int len = (data_num - 1) / 32 + 1;
    for (int i = 0; i < len; i++) {
        if (result_cpu[i] != result_gpu[i]) {
            int pos = find_different_pos(result_cpu[i], result_gpu[i]);
            int val_pos = i * 32 + pos;
            printf(
                "compare result: result_cpu[%d] = 0x%08x, result_gpu[%d] = 0x%08x, pos = %d\n"
                "data[%d] = {x: %.0f, y: %.0f, z: %.0f}\n"
                "predicate = {x1: %.0f, x2: %.0f, y1: %.0f, y2: %.0f, z1: %.0f, z2: %.0f}\n",
                i, result_cpu[i],
                i, result_gpu[i],
                pos,
                val_pos, state.vertices3[val_pos].x, state.vertices3[val_pos].y, state.vertices3[val_pos].z,
                p.x1, p.x2, p.y1, p.y2, p.z1, p.z2);
            assert(0);
        }
    }
    printf("\033[1;32m##### result_cpu == result_gpu #####\033[0m\n");
}

void parse_args(ScanState &state, int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            printUsageAndExit(argv[0]);
        } else if (arg == "--file" || arg == "-f") {
            if (i < argc - 1) {
                state.infile = argv[++i];
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg.substr(0, 6) == "--dim=") {
            const std::string dims_arg = arg.substr(6);
            sutil::parseDimensions(dims_arg.c_str(), state.width, state.height);
        } else if (arg == "--depth") {
            if (i < argc - 1) {
                state.depth = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            } 
        } else if (arg == "--selec") {
            if (i < argc - 1) {
                selec = stof(argv[++i]);
                upper_val = UINT32_MAX * selec;
                predicate = {.x1 = 0.0f, .x2 = upper_val, 
                             .y1 = 0.0f, .y2 = upper_val, 
                             .z1 = 0.0f, .z2 = upper_val};
            } else {
                printUsageAndExit(argv[0]);
            } 
        } else if (arg == "--n") {
            if (i < argc - 1) {
                data_num = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--primitive") {
            if (i < argc - 1) {
                state.primitive_type = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if(arg == "--nc") {
            state.no_comparison = true;
        } else {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit(argv[0]);
        }
    }
}

void checkoptixScanOnline()
{
    std::cout << "[OptiX] optix Scan online now." << std::endl;
}

void initialize_optix(ScanState &state) {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK(optixInit());

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    CUcontext cuCtx = 0; // zero means take the current context
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &state.context));
}

void make_gas_triangle(ScanState &state) {
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION; // compaction is needed
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    // 重新设置各个顶点
    const size_t triangle_vertices_size = sizeof(double3) * data_num * 3;
    triangle_vertices = (double3*) malloc(triangle_vertices_size);
    vertices_to_triangles(state, vertices, triangle_vertices);
    CUdeviceptr d_vertices = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices), triangle_vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_vertices),
        triangle_vertices,
        triangle_vertices_size,
        cudaMemcpyHostToDevice));
    state.params.points3 = reinterpret_cast<double3 *>(d_vertices);

    // Our build input is a simple list of non-indexed triangle vertices
    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices   = 3 * data_num;
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(double3);
    triangle_input.triangleArray.flags         = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &triangle_input,
        1, // Number of build inputs
        &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_temp_buffer_gas),
        gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output and size of compacted GAS.
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0, // CUDA stream
        &accel_options,
        &triangle_input,
        1, // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        &emitProperty, // emitted property list
        1              // num emitted properties
        ));

    // We can now free the scratch space buffer used during build and the vertex
    // inputs, since they are not needed by our trivial shading method
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_vertices)));  // 放到了最后来释放 params.points 的空间

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpyAsync(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost, 0));
    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        // compacted size is smaller, so store the compacted GAS in new device memory and free the original GAS memory/
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer), compacted_gas_size));
        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle));
        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    } else {
        // original size is smaller, so point d_gas_output_buffer directly to the original device GAS memory.
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
    fprintf(stdout, "Final GAS size: %f MB\n", (double)compacted_gas_size / (1024 * 1024));
}

void make_gas(ScanState &state) {
    // Use default options for simplicity.  In a real use case we would want to
    // enable compaction, etc
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    CUdeviceptr d_vertices = 0;
    OptixAabb  *d_aabb;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_aabb), state.length * sizeof(OptixAabb)));
    
    size_t vertices_size;
    if (state.column_num == 3) {
        vertices_size = sizeof(double3) * state.length;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices_size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(d_vertices),
            state.vertices3,
            vertices_size,
            cudaMemcpyHostToDevice));
        state.params.points3 = reinterpret_cast<double3 *>(d_vertices);
    } else if (state.column_num == 2) {
        vertices_size = sizeof(double2) * state.length;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices_size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(d_vertices),
            state.vertices2,
            vertices_size,
            cudaMemcpyHostToDevice));
        state.params.points2 = reinterpret_cast<double2 *>(d_vertices);
    }
    
    kGenAABB(reinterpret_cast<double3*>(d_vertices), state.params.aabb_width / 2, state.length, d_aabb, state.column_num);
    CUdeviceptr d_aabb_ptr = reinterpret_cast<CUdeviceptr>(d_aabb);

    // Our build input is a simple list of non-indexed triangle vertices
    const uint32_t vertex_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput vertex_input = {};
    vertex_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    vertex_input.customPrimitiveArray.aabbBuffers = &d_aabb_ptr;
    vertex_input.customPrimitiveArray.flags = vertex_input_flags;
    vertex_input.customPrimitiveArray.numSbtRecords = 1; // 看后续是否设置 sbt
    vertex_input.customPrimitiveArray.numPrimitives = state.length;
    // it's important to pass 0 to sbtIndexOffsetBuffer
    vertex_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    vertex_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    vertex_input.customPrimitiveArray.primitiveIndexOffset = 0;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &vertex_input,
        1, // Number of build inputs
        &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_temp_buffer_gas),
        gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output and size of compacted GAS.
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0, // CUDA stream
        &accel_options,
        &vertex_input,
        1, // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        &emitProperty, // emitted property list
        1              // num emitted properties
        ));

    // We can now free the scratch space buffer used during build and the vertex
    // inputs, since they are not needed by our trivial shading method
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_vertices)));  // 放到了最后来释放 params.points 的空间

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpyAsync(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost, 0));
    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        // compacted size is smaller, so store the compacted GAS in new device memory and free the original GAS memory/
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    } else {
        // original size is smaller, so point d_gas_output_buffer directly to the original device GAS memory.
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
    fprintf(stdout, "Final GAS size: %f MB\n", (double)compacted_gas_size / (1024 * 1024));
}

void make_module(ScanState &state) {
    char log[2048];

    OptixModuleCompileOptions module_compile_options = {};
#if !defined(NDEBUG)
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 3;
    state.pipeline_compile_options.numAttributeValues = 0;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    // By default (usesPrimitiveTypeFlags == 0) it supports custom and triangle primitives
    state.pipeline_compile_options.usesPrimitiveTypeFlags = 
        state.primitive_type == 0 ? OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM : OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    size_t inputSize = 0;
    const char *input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixScan.cu", inputSize);
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input,
        inputSize,
        log,
        &sizeof_log,
        &state.module));
}

void make_program_groups(ScanState &state) {
    char log[2048];

    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = state.module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &raygen_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.raygen_prog_group));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = state.module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.miss_prog_group));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    if (state.primitive_type == 0) {            // cube
        hitgroup_prog_group_desc.hitgroup.moduleIS = state.module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cube";
    } else if (state.primitive_type == 1) {     // triangle
        hitgroup_prog_group_desc.hitgroup.moduleAH = state.module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__triangle"; 
    } else {                                    // triangle without anyhit
    }                                   
      
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &hitgroup_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.hitgroup_prog_group));
}

void make_pipeline(ScanState &state) {
    char log[2048];
    const uint32_t max_trace_depth = 1;
    std::vector<OptixProgramGroup> program_groups{state.raygen_prog_group, state.miss_prog_group, state.hitgroup_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        program_groups.size(),
        log,
        &sizeof_log,
        &state.pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto &prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0, // maxCCDepth
                                           0, // maxDCDEpth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          1 // maxTraversableDepth
                                          ));
}

void make_sbt(ScanState &state) {
    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
    MissSbtRecord ms_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(miss_record),
        &ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    hg_sbt.data = predicate; // copy data
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(hitgroup_record),
        &hg_sbt,
        hitgroup_record_size,
        cudaMemcpyHostToDevice));

    state.sbt.raygenRecord = raygen_record;
    state.sbt.missRecordBase = miss_record;
    state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    state.sbt.hitgroupRecordCount = 1;
}

void display_results(ScanState &state, BITS *result_cpu, Predicate &p, unsigned init_hit_num) {
    unsigned int *intersection_test_num;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&intersection_test_num), sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(intersection_test_num),
        state.params.intersection_test_num,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost));
    fprintf(stdout, "[OptiX] intersection_test_num: %u\n", *intersection_test_num);
    unsigned int *hit_num;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&hit_num), sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(hit_num),
        state.params.hit_num,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost));
    fprintf(stdout, "[OptiX] hit_num: %u\n", *hit_num);
    
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&state.h_result), state.result_byte_num));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(state.h_result),
        state.params.result,
        state.result_byte_num,
        cudaMemcpyDeviceToHost));

    // Obtain the number of hit points from the result bit vector calculated by #CPU#
    unsigned int point_num_cpu = 0;
    for (int i = 0; i < state.length; i++) {
        int pos_in_unsigned_val = 1 << (31 - i);
        if (result_cpu[i / 32] & pos_in_unsigned_val) {
            point_num_cpu++;
        }
    }
    fprintf(stdout, "[OptiX] result_cpu_point_num: %u\n", point_num_cpu);
    
    // Obtain the number of hit points from the result bit vector calculated by #GPU#
    unsigned int point_num = 0;
    for (int i = 0; i < state.length; i++) {
        int pos_in_unsigned_val = 1 << (31 - i);
        if (state.h_result[i / 32] & pos_in_unsigned_val) {
            point_num++;
        }
    }
    fprintf(stdout, "[OptiX] result_gpu_point_num: %u\n", point_num);
    fprintf(stdout, "[OptiX] actually refine num: %u\n", point_num > init_hit_num ? point_num - init_hit_num : init_hit_num - point_num);

    compare_result(result_cpu, state.h_result, state.length, p);
    CUDA_CHECK(cudaFreeHost(state.h_result));
    CUDA_CHECK(cudaFreeHost(intersection_test_num));
    CUDA_CHECK(cudaFreeHost(hit_num));
    state.h_result = nullptr;
}

void display_ray_hits(ScanState &state) {
    int ray_num = state.launch_width * state.launch_height * state.depth;
    unsigned int *ray_hits = (unsigned int *) malloc(sizeof(unsigned int) * ray_num);
    CUDA_CHECK(cudaMemcpy(ray_hits, state.params.ray_primitive_hits, sizeof(unsigned int) * ray_num, cudaMemcpyDeviceToHost));
    std::map<unsigned int, int> hitNum_rayNum;
    int sum = 0;
    for (int i = 0; i < ray_num; i++) {
        sum += ray_hits[i];
        if (hitNum_rayNum.count(ray_hits[i])) {
            hitNum_rayNum[ray_hits[i]]++;
        } else {
            hitNum_rayNum[ray_hits[i]] = 1;
        }       
    }

    int min, max, median = -1;
    double avg;
    int tmp_sum = 0;
    min = hitNum_rayNum.begin()->first;
    max = (--hitNum_rayNum.end())->first;
    avg = 1.0 * sum / ray_num;
    printf("光线穿过 cube 数: 对应光线数\n");
    for (auto &item: hitNum_rayNum) {
        fprintf(stdout, "%d: %d\n", item.first, item.second);
        tmp_sum += item.second;
        if (median == -1 && tmp_sum >= ray_num / 2) {
            median = item.first;
        }
    }
    printf("min: %d, max: %d, average: %lf, median: %d\n", min, max, avg, median);
    free(ray_hits);

    // first hit, next hit
    int first_hit, next_hit;
    if (min == 0) {
        first_hit = ray_num - hitNum_rayNum.begin()->second;
    } else {
        first_hit = ray_num;
    }
    next_hit = sum - first_hit;
    printf("first_hit: %d, next_hit: %d\n", first_hit, next_hit);

    int real_next_hit = 0;
    for (auto it = ++(++hitNum_rayNum.begin()); it != hitNum_rayNum.end(); it++) {
        real_next_hit += it->second;
    }
    printf("real_next_hit: %d\n", real_next_hit);
}
void cleanup(ScanState &state) {
    // free host memory
    free(vertices);
    if (state.primitive_type) {
        free(triangle_vertices);
    }

    // free device memory
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_gas_output_buffer)));

    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.hitgroup_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixModuleDestroy(state.module));

    OPTIX_CHECK(optixDeviceContextDestroy(state.context));
}

void log_common_info(ScanState &state) {
    printf("max ray num:                %dx%dx%d\n", state.width, state.height, state.depth);
    printf("data num:                   %d\n", state.length);
    printf("aabb_width:                 %f\n", state.params.aabb_width);
    printf("column_num:                 %d\n", state.column_num);
}

void initializeOptixRTScan_2c(CODE **raw_data, int length, int density_width, int density_height, int column_num) {
    fprintf(stdout, "[OptiX]initializeOptix begin...\n");
    int epi                 = 32;
    state                   = {};
    state.length            = length;
    state.width             = density_width;
    state.height            = density_height;
    state.column_num        = column_num;
    state.params.aabb_width = (state.length - 1) / state.width + 1.0f + epi;
    state.result_byte_num   = ((state.length - 1) / 32 + 1) * 4;

    state.vertices2 = (double2 *) malloc(length * sizeof(double2));
    uint32_to_double2(state.vertices2, raw_data, length);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.params.intersection_test_num), sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.params.hit_num), sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.params.predicate), 6 * sizeof(double)));
    
    log_common_info(state);

    timer_.commonGetStartTime(0); // build time
    initialize_optix(state);
    make_gas(state);
    make_module(state);
    make_program_groups(state);
    make_pipeline(state);
    make_sbt(state);
    timer_.commonGetEndTime(0);
    timer_.showTime(0, "initializeOptix");
    fprintf(stdout, "[OptiX]initializeOptix end\n");
}

// called by bindex
void refineWithOptixRTScan_2c(BITS *dev_result_bitmap, double *predicate, unsigned *range, 
                              int column_num, int ray_segment_num, bool inverse) { 
#if DEBUG_INFO == 1
    timer_.commonGetStartTime(1);
#endif
    
#if DEBUG_ISHIT_CMP_RAY == 1
    Predicate p = {
        .x1 = predicate[0], .x2 = predicate[1], 
        .y1 = predicate[2], .y2 = predicate[3]
    };
    unsigned point_num_cpu;
    BITS* result_cpu;
    int uint_num = (state.length - 1) / 32 + 1;
    result_cpu = (BITS *) malloc(sizeof(BITS) * uint_num);
    CUDA_CHECK(cudaMemcpy(result_cpu, dev_result_bitmap, sizeof(BITS) * uint_num, cudaMemcpyDeviceToHost));
    point_num_cpu = 0;
    for (int i = 0; i < state.length; i++) {
        int pos_in_unsigned_val = 1 << (31 - i);
        if (result_cpu[i / 32] & pos_in_unsigned_val) {
            point_num_cpu++;
        }
    }
    fprintf(stdout, "[OptiX] initial point num: %u\n", point_num_cpu);
    
    if (state.column_num == 3) {
        p.z1 = predicate[4];
        p.z2 = predicate[5];
        scan_with_cpu(state.vertices3, state.length, result_cpu, p, inverse);
    } else {
        scan_with_cpu(state.vertices2, state.length, result_cpu, p, inverse);
    }

    CUDA_CHECK(cudaMemset(state.params.intersection_test_num, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(state.params.hit_num, 0, sizeof(unsigned int)));
#endif

    state.params.result     = dev_result_bitmap;
    state.params.width      = state.width;
    state.params.height     = state.height;
    state.depth             = ray_segment_num;
    state.params.handle     = state.gas_handle;
    state.params.inverse    = inverse;
    state.params.column_num = state.column_num;
    state.params.ray_stride = 2.0 + state.params.aabb_width;
    // state.launch_width      = static_cast<int>((predicate[1] - predicate[0]) / UINT32_MAX * state.width) + 1;
    // state.launch_height     = static_cast<int>((predicate[3] - predicate[2]) / UINT32_MAX * state.height) + 1;
    state.launch_width      = static_cast<int>((predicate[1] - predicate[0]) / state.params.aabb_width) + 1;
    state.launch_height     = static_cast<int>((predicate[3] - predicate[2]) / state.params.aabb_width) + 1;
#if DEBUG_ISHIT_CMP_RAY == 1
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.params.ray_primitive_hits), sizeof(unsigned int) * state.launch_width * state.launch_height * state.depth));
#endif
    CUDA_CHECK(cudaMemcpy(state.params.predicate, predicate, 6 * sizeof(double), cudaMemcpyHostToDevice));

    //************
    //* Memcpy Params
    //************
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_params), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(state.d_params),
        &state.params,
        sizeof(Params),
        cudaMemcpyHostToDevice));

#if DEBUG_INFO == 1
    fprintf(stdout, "[OptiX] range: ");
    for (int i = 0; i < column_num * 2; i++) {
        fprintf(stdout, "%u ", range[i]);
    }
    printf("[OptiX] ray_segment_num: %d\n", ray_segment_num);
    printf("[OptiX] inverse: %s\n", inverse ? "true" : "false");
    printf("[OptiX] ray_stride: %lf\n", state.params.ray_stride);
    printf("[OptiX] launch_width = %d, launch_height = %d, depth = %d, total ray num = %d\n", state.launch_width, state.launch_height, state.depth, state.launch_width * state.launch_height * state.depth);
#endif

#if DEBUG_INFO == 1
    timer_.commonGetStartTime(2);
#endif
    OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.launch_width, state.launch_height, state.depth));
    CUDA_SYNC_CHECK();
#if DEBUG_INFO == 1
    timer_.commonGetEndTime(2);
    timer_.commonGetEndTime(1);
#endif

#if DEBUG_INFO == 1
    timer_.showTime(1, "refineWithOptix");
    timer_.showTime(2, "optixLaunch");
    timer_.clear();
    fprintf(stdout, "[OptiX] refineWithOptix done\n");
#endif

#if DEBUG_ISHIT_CMP_RAY == 1
    display_results(state, result_cpu, p, point_num_cpu);
#if DEBUG_RAY == 1
    display_ray_hits(state);
#endif
#endif
    
    // cleanup
    CUDA_CHECK(cudaFree((void *)state.d_params));
#if DEBUG_RAY == 1
    CUDA_CHECK(cudaFree(state.params.ray_primitive_hits));
#endif
}