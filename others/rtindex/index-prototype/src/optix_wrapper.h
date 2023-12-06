#pragma once

#include "cuda_buffer.cuh"


struct optix_wrapper {

    optix_wrapper();
    ~optix_wrapper();

protected:
    void init_optix();

    void create_context();

    void create_module();

public:
    CUcontext cuda_context;

    OptixDeviceContext optix_context;
    
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipelineLinkOptions    pipeline_link_options = {};

    OptixModule                 module;
    OptixModuleCompileOptions   module_compile_options = {};
};



