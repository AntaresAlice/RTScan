#ifndef STATE_H
#define STATE_H

#include <float.h>
#include <vector_types.h>
#include <optix_types.h>
#include <unordered_set>
#include "optixScan.h"

struct ScanState
{
    Params                          params;
    CUdeviceptr                     d_params;
    OptixDeviceContext              context                   = nullptr;
    OptixTraversableHandle          gas_handle;
    CUdeviceptr                     d_gas_output_buffer;

    OptixModule                     module                    = nullptr;

    OptixProgramGroup               raygen_prog_group         = nullptr;
    OptixProgramGroup               miss_prog_group           = nullptr;
    OptixProgramGroup               hitgroup_prog_group       = nullptr;

    OptixPipeline                   pipeline                  = nullptr;
    OptixPipelineCompileOptions     pipeline_compile_options  = {};

    double3                         *h_points;
    unsigned int                    num_points;

    OptixShaderBindingTable         sbt                       = {}; 

    uint32_t                        bound                     = UINT32_MAX;
    unsigned*                       range;
    double3*                        vertices;
    int                             length;
    int                             width;
    int                             height;
    int                             depth                     = 1;
    int                             launch_width;
    int                             launch_height;

    unsigned int*                   h_result;
    unsigned int                    result_byte_num;
    int                             epsilon; // solve precise problem
};


#endif