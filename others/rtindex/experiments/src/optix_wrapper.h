#pragma once

#include "cuda_buffer.cuh"


struct optix_wrapper {

    optix_wrapper(bool debug = false);
    ~optix_wrapper();

protected:
    void init_optix();

    void create_context();

    void create_module();

    void create_sphere_module();

public:
    CUcontext          cuda_context;
    cudaStream_t       stream;

    OptixDeviceContext optix_context;
    
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipelineLinkOptions    pipeline_link_options = {};

    OptixModule                 module;
    OptixModule                 sphere_module;
    OptixModuleCompileOptions   module_compile_options = {};

    bool debug;
};
