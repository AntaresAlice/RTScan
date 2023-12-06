set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/aabb.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

# message(STATUS ${DEBUG_INFO})
# add_definitions(-DDEBUG_INFO=${DEBUG_INFO})
message(STATUS ${PRIMITIVE_TYPE})
add_definitions(-DPRIMITIVE_TYPE=${PRIMITIVE_TYPE})
add_definitions(-DDEBUG_ISHIT_CMP_RAY=${DEBUG_ISHIT_CMP_RAY})
add_definitions(-DDEBUG_INFO=${DEBUG_INFO})

OPTIX_add_sample_executable( rtscan_interval_spacing target_name
  optixScan.cpp
  optixScan.cu
  aabb.cu
  optixScan.h
  state.h
  timer.h
  # OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )