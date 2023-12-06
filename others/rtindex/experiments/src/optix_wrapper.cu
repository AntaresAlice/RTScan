#include "optix_wrapper.h"
#include "test_configuration.h"
#include "cuda_helpers.cuh"

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>


extern "C" char embedded_ptx_code[];


optix_wrapper::optix_wrapper(bool debug) : debug{debug} {
    init_optix();
    create_context();
    create_module();
#if PRIMITIVE == 1
    create_sphere_module();
#endif
}

optix_wrapper::~optix_wrapper() {
    OPTIX_CHECK(optixModuleDestroy(module));
#if PRIMITIVE == 1
    OPTIX_CHECK(optixModuleDestroy(sphere_module));
#endif
    OPTIX_CHECK(optixDeviceContextDestroy(optix_context));
    cudaStreamDestroy(stream); CUERR
}


/*! helper function that initializes optix and checks for errors */
void optix_wrapper::init_optix() {
    cudaFree(0);
    int num;
    cudaGetDeviceCount(&num);
    if (num == 0)
        throw std::runtime_error("no CUDA capable devices found!");
    OPTIX_CHECK(optixInit());
}


static void context_log_cb(unsigned int level,
                           const char *tag,
                           const char *message,
                           void *) {
    // ENABLE IF NEEDED
    // fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}


static void print_log(const char *message) {
    // ENABLE IF NEEDED
    // std::cout << "log=" << message << std::endl;
}


void optix_wrapper::create_context() {
    cudaSetDevice(0); CUERR
    cudaStreamCreate(&stream); CUERR
    cuCtxGetCurrent(&cuda_context); CUERR
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context, 0, &optix_context));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optix_context, context_log_cb, nullptr, 4));
}


void optix_wrapper::create_module() {

    // figure out payload semantics and register usage impact
    // https://raytracing-docs.nvidia.com/optix7/guide/index.html#payload

    module_compile_options.maxRegisterCount  = 0;
    module_compile_options.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipeline_compile_options = {};
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.usesMotionBlur        = false;
    pipeline_compile_options.numPayloadValues      = 2;
    pipeline_compile_options.numAttributeValues    = 0;
    pipeline_compile_options.exceptionFlags        = debug ? OPTIX_EXCEPTION_FLAG_DEBUG : OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
#if PRIMITIVE == 1
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
#endif

    pipeline_link_options.maxTraceDepth = 2;

    const std::string ptx = embedded_ptx_code;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(
            optix_context,
            &module_compile_options,
            &pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            log,&sizeof_log,
            &module
    ));
    if (sizeof_log > 1) print_log(log);
}


void optix_wrapper::create_sphere_module() {
    OptixBuiltinISOptions builtin_is_options = {};
    builtin_is_options.usesMotionBlur      = false;
    builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OPTIX_CHECK(optixBuiltinISModuleGet(optix_context, &module_compile_options, &pipeline_compile_options, &builtin_is_options, &sphere_module));
}

