#include <optix.h>

#include "optixScan.h"

#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void computeRay( uint3 idx, float3& origin, float3& direction ) {
    if (params.direction == 0) { // x
        origin = {
            float((params.predicate[0] + 0.5 * params.ray_space) + idx.z * params.ray_stride),
            float(params.predicate[2] + (idx.x + 0.5) * params.ray_interval),
            float(params.predicate[4] + (idx.y + 0.5) * params.ray_interval)
        };
        direction = {1.0f, 0.0f, 0.0f};
    } else if (params.direction == 1) { // y
        origin = {
            float(params.predicate[0] + (idx.x + 0.5) * params.ray_interval),
            float((params.predicate[2] + 0.5 * params.ray_space) + idx.z * params.ray_stride),
            float(params.predicate[4] + (idx.y + 0.5) * params.ray_interval),
        };
        direction = {0.0f, 1.0f, 0.0f};
    } else { // z
        origin = {
            float(params.predicate[0] + (idx.x + 0.5) * params.ray_interval),
            float(params.predicate[2] + (idx.y + 0.5) * params.ray_interval),
            float((params.predicate[4] + 0.5 * params.ray_space) + idx.z * params.ray_stride)
        };
        direction = {0.0f, 0.0f, 1.0f};
    }
}

static __forceinline__ __device__ void computeRay_ray_interval_1( uint3 idx, float3& origin, float3& direction ) {
    if (params.direction == 0) { // x
        origin = {
            float((params.predicate[0] + params.ray_space) + idx.z * params.ray_stride),
            float(params.predicate[2] + (idx.x + 1) * params.ray_interval),
            float(params.predicate[4] + (idx.y + 1) * params.ray_interval)
        };
        direction = {1.0f, 0.0f, 0.0f};
    } else if (params.direction == 1) { // y
        origin = {
            float(params.predicate[0] + (idx.x + 1) * params.ray_interval),
            float((params.predicate[2] + params.ray_space) + idx.z * params.ray_stride),
            float(params.predicate[4] + (idx.y + 1) * params.ray_interval),
        };
        direction = {0.0f, 1.0f, 0.0f};
    } else { // z
        origin = {
            float(params.predicate[0] + (idx.x + 1) * params.ray_interval),
            float(params.predicate[2] + (idx.y + 1) * params.ray_interval),
            float((params.predicate[4] + params.ray_space) + idx.z * params.ray_stride)
        };
        direction = {0.0f, 0.0f, 1.0f};
    }
}

static __forceinline__ __device__ void set_result(unsigned int *result, int idx) {
    int size = sizeof(unsigned int) << 3;
    int pos = idx / size;
    int pos_in_size = idx & (size - 1);
    // printf("set_result, idx: %d, pos_in_size: %d, 1 << (size - 1 - pos_in_size): %u\n", idx, pos_in_size, 1 << (size - 1 - pos_in_size));
    if (params.inverse) {
        atomicAnd( result + pos, ~(1 << (size - 1 - pos_in_size)) );
        // result[pos] &= ~(1 << (size - 1 - pos_in_size));
    } else {
        atomicOr( result + pos, 1 << (size - 1 - pos_in_size) );
        // result[pos] |= (1 << (size - 1 - pos_in_size));
    }
}

extern "C" __global__ void __raygen__rg() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();    
    const uint3 dim = optixGetLaunchDimensions(); 

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen 
    float3 ray_origin, ray_direction;
#if SMALL_DATA_RANGE == 1
    computeRay_ray_interval_1( idx, ray_origin, ray_direction );
#else
    computeRay( idx, ray_origin, ray_direction );
#endif

    // Trace the ray against our scene hierarchy
    unsigned int intersection_test_num = 0;
    unsigned int hit_num = 0;
    double ray_length = params.ray_length;
    if (idx.z == dim.z - 1) {
        ray_length = params.ray_last_length;
    }
    optixTrace(
            params.handle, 
            ray_origin,
            ray_direction,
            0.0f,                           // Min intersection distance
            (float) ray_length,      // Max intersection distance
            0.0f,                           // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ),     // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            intersection_test_num,
            hit_num
            );
#if DEBUG_ISHIT_CMP_RAY == 1
    atomicAdd(params.intersection_test_num, intersection_test_num);
    atomicAdd(params.hit_num, hit_num);
    params.ray_primitive_hits[dim.x * dim.y * idx.z + dim.x * idx.y + idx.x] = hit_num;
#endif
}

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __intersection__cube() {
    unsigned int primIdx = optixGetPrimitiveIndex();
    double3 point = params.points[primIdx];

#if DEBUG_ISHIT_CMP_RAY == 1
    optixSetPayload_0(optixGetPayload_0() + 1); // number of intersection test
#endif
    if (point.x > params.predicate[0] && point.x < params.predicate[1] &&
        point.y > params.predicate[2] && point.y < params.predicate[3] && 
        point.z > params.predicate[4] && point.z < params.predicate[5] ) {
        set_result(params.result, primIdx);
#if DEBUG_ISHIT_CMP_RAY == 1
        optixSetPayload_1(optixGetPayload_1() + 1);
#endif
    }
}

// anyhit maybe called repeatedly.
extern "C" __global__ void __anyhit__triangle() {
    unsigned int primIdx = optixGetPrimitiveIndex();
    double3 point = params.points[(primIdx << 1) + primIdx];
    
    unsigned int intersection_test_num = optixGetPayload_0();
    optixSetPayload_0(intersection_test_num + 1); // number of intersection test
    if (point.x > params.predicate[0] && point.x < params.predicate[0] &&
        point.y > params.predicate[2] && point.y < params.predicate[2] && 
        point.z > params.predicate[4] && point.z < params.predicate[4] ) {
        set_result(params.result, primIdx);
        optixSetPayload_1(optixGetPayload_1() + 1);
    }
    optixIgnoreIntersection();
}

