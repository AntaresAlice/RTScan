cmake_minimum_required(VERSION 3.20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(run_experiment C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
                       
mark_as_advanced(CUDA_SDK_ROOT_DIR)

include(cmake/configure_optix.cmake)

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code device_code.cu)

add_executable(run_experiment
        ${embedded_ptx_code}
        launch_parameters.cuh
        main.cu
        cuda_helpers.cuh
        cuda_buffer.cuh
        optix_pipeline.cu
        optix_pipeline.h
        optix_wrapper.cu
        optix_wrapper.h
        launch_parameters.cuh
        optix_helpers.cuh
        test_configuration.h
        test_configuration_override.h
        default_test_configuration.h
        )

set_property(TARGET run_experiment PROPERTY CUDA_ARCHITECTURES
        75     # TURING
        80 86  # AMPERE
        89     # ADA
        )

target_compile_options(run_experiment PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror ${OpenMP_CXX_FLAGS}>
        )
        
target_compile_options(run_experiment PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#        -lineinfo -g --profile
        --extended-lambda --use_fast_math --expt-relaxed-constexpr
        -O3
        -Xcompiler=-fopenmp
#        -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wpedantic
        >)

target_link_libraries(run_experiment
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        OpenMP::OpenMP_CXX
        )
