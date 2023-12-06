#include "optix_pipeline.h"
#include "cuda_helpers.cuh"
#include "test_configuration.h"


static void print_log(const char *message) {
    std::cout << "log=" << message << std::endl;
}


struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) raygen_sbt_record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) miss_sbt_record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) hitgroup_sbt_record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};


optix_pipeline::optix_pipeline(optix_wrapper* optix, bool verbose) : optix{optix}, verbose{verbose} {
    if (verbose) {std::cout << "#rtx: creating raygen programs ..." << std::endl;}
    create_raygen_programs();

    if (verbose) {std::cout << "#rtx: creating miss programs ..." << std::endl;}
    create_miss_programs();

    if (verbose) {std::cout << "#rtx: creating hitgroup programs ..." << std::endl;}
    create_hitgroup_programs();

    if (verbose) {std::cout << "#rtx: setting up optix pipeline ..." << std::endl;}
    assemble_pipeline();

    if (verbose) {std::cout << "#rtx: building SBT ..." << std::endl;}
    build_sbt();

    if (verbose) {std::cout << "#rtx: pipeline all set up ..." << std::endl;}
}

optix_pipeline::~optix_pipeline() {
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    for (auto pg : raygen_program_groups)
        OPTIX_CHECK(optixProgramGroupDestroy(pg));
    for (auto pg : miss_program_groups)
        OPTIX_CHECK(optixProgramGroupDestroy(pg));
    for (auto pg : hitgroup_program_groups)
        OPTIX_CHECK(optixProgramGroupDestroy(pg));
}


void optix_pipeline::create_raygen_programs() {
    raygen_program_groups.resize(1);

    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc pg_desc       = {};
    pg_desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pg_desc.raygen.module               = optix->module;
    pg_desc.raygen.entryFunctionName    = "__raygen__test";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
            optix->optix_context,
            &pg_desc,
            1,
            &pg_options,
            log,&sizeof_log,
            &raygen_program_groups[0]
    ));
    if (verbose && sizeof_log > 1) print_log(log);
}


void optix_pipeline::create_miss_programs() {
    miss_program_groups.resize(1);

    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc pg_desc       = {};
    pg_desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pg_desc.miss.module                 = optix->module;
    pg_desc.miss.entryFunctionName      = "__miss__test";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
            optix->optix_context,
            &pg_desc,
            1,
            &pg_options,
            log,&sizeof_log,
            &miss_program_groups[0]
    ));
    if (verbose && sizeof_log > 1) print_log(log);
}


void optix_pipeline::create_hitgroup_programs() {
    hitgroup_program_groups.resize(1);

    OptixProgramGroupOptions pg_options  = {};
    OptixProgramGroupDesc pg_desc        = {};
    pg_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pg_desc.hitgroup.moduleCH            = optix->module;
    pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__test";
    pg_desc.hitgroup.moduleAH            = optix->module;
    pg_desc.hitgroup.entryFunctionNameAH = "__anyhit__test";
#if PRIMITIVE == 1
    pg_desc.hitgroup.moduleIS            = optix->sphere_module;
    pg_desc.hitgroup.entryFunctionNameIS = nullptr;
#elif PRIMITIVE == 2
    pg_desc.hitgroup.moduleIS            = optix->module;
    pg_desc.hitgroup.entryFunctionNameIS = "__intersection__test";
#endif

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
            optix->optix_context,
            &pg_desc,
            1,
            &pg_options,
            log,&sizeof_log,
            &hitgroup_program_groups[0]
    ));
    if (verbose && sizeof_log > 1) print_log(log);
}


void optix_pipeline::assemble_pipeline() {
    std::vector<OptixProgramGroup> program_groups;
    for (auto pg : raygen_program_groups)
        program_groups.push_back(pg);
    for (auto pg : miss_program_groups)
        program_groups.push_back(pg);
    for (auto pg : hitgroup_program_groups)
        program_groups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
            optix->optix_context,
            &optix->pipeline_compile_options,
            &optix->pipeline_link_options,
            program_groups.data(),
            (int)program_groups.size(),
            log,&sizeof_log,
            &pipeline
    ));
    if (verbose && sizeof_log > 1) print_log(log);

    // see https://raytracing-docs.nvidia.com/optix7/guide/index.html#program_pipeline_creation#pipeline-stack-size
    /*
    OPTIX_CHECK(optixPipelineSetStackSize
            (// [in] The pipeline to configure the stack size for
                    pipeline,
                    // [in] The direct stack size requirement for direct callables invoked from IS or AH.
                    2*1024,
                    // [in] The direct stack size requirement for direct callables invoked from RG, MS, or CH.
                    2*1024,
                    // [in] The continuation stack requirement.
                    2*1024,
                    // [in] The maximum depth of a traversable graph passed to trace.
                    1));
    if (verbose && sizeof_log > 1) print_log(log);
    */
}


void optix_pipeline::build_sbt() {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<raygen_sbt_record> raygen_records;
    for (int i = 0; i < raygen_program_groups.size(); i++) {
        raygen_sbt_record rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_program_groups[i], &rec));
        rec.data = nullptr;
        raygen_records.push_back(rec);
    }
    raygen_records_buffer.alloc_and_upload(raygen_records);
    sbt.raygenRecord = raygen_records_buffer.cu_ptr();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<miss_sbt_record> miss_records;
    for (int i = 0; i < miss_program_groups.size(); i++) {
        miss_sbt_record rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_program_groups[i], &rec));
        rec.data = nullptr;
        miss_records.push_back(rec);
    }
    miss_records_buffer.alloc_and_upload(miss_records);
    sbt.missRecordBase          = miss_records_buffer.cu_ptr();
    sbt.missRecordStrideInBytes = sizeof(miss_sbt_record);
    sbt.missRecordCount         = (int)miss_records.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------

    // we don't actually have any objects in this example, but let's
    // create a dummy one so the SBT doesn't have any null pointers
    // (which the sanity checks in compilation would complain about)
    std::vector<hitgroup_sbt_record> hitgroup_records;
    for (int i = 0; i < hitgroup_program_groups.size(); i++) {
        hitgroup_sbt_record rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_program_groups[i], &rec));
        rec.data = nullptr;
        hitgroup_records.push_back(rec);
    }
    hitgroup_records_buffer.alloc_and_upload(hitgroup_records);
    sbt.hitgroupRecordBase          = hitgroup_records_buffer.cu_ptr();
    sbt.hitgroupRecordStrideInBytes = sizeof(hitgroup_sbt_record);
    sbt.hitgroupRecordCount         = (int)hitgroup_records.size();
}
