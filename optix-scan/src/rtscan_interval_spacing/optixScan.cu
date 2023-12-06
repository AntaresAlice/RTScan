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

#include "optixScan.h"

#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void computeRayForTriangle( uint3 idx, float3& origin, float3& direction ) {
    if (params.direction == 0) { // x
        origin = {
            float(params.predicate[0] + idx.z * params.ray_stride),
            float(params.predicate[2] + idx.x * params.ray_interval),
            float(params.predicate[4] + idx.y * params.ray_interval)
        };
        direction = {1.0f, 0.0f, 0.0f};
    } else if (params.direction == 1) { // y
        origin = {
            float(params.predicate[0] + idx.x * params.ray_interval),
            float(params.predicate[2] + idx.z * params.ray_stride),
            float(params.predicate[4] + idx.y * params.ray_interval),
        };
        direction = {0.0f, 1.0f, 0.0f};
    } else { // z
        origin = {
            float(params.predicate[0] + idx.x * params.ray_interval),
            float(params.predicate[2] + idx.y * params.ray_interval),
            float(params.predicate[4] + idx.z * params.ray_stride)
        };
        direction = {0.0f, 0.0f, 1.0f};
    }
}

static __forceinline__ __device__ void computeRayForAABB( uint3 idx, float3& origin, float3& direction ) {
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
#if PRIMITIVE_TYPE == 0
    computeRayForTriangle(idx, ray_origin, ray_direction);
#elif PRIMITIVE_TYPE == 1
    computeRayForAABB(idx, ray_origin, ray_direction);
#else
    computeRayForAABB(idx, ray_origin, ray_direction);
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
        optixSetPayload_1(optixGetPayload_1() + 1); // number of hit
#endif        
    }
}

extern "C" __global__ void __anyhit__get_prim_id() {
    unsigned int primIdx = optixGetPrimitiveIndex();
    double3 point = params.points[primIdx];
    // printf("point = %.0lf %.0lf %.0lf\n", point.x, point.y, point.z);
    // printf("params.predicate = %.0lf %.0lf %.0lf %.0lf %.0lf %.0lf\n", params.predicate[0], params.predicate[1], params.predicate[2], params.predicate[3], params.predicate[4], params.predicate[5]);
    
    unsigned int intersection_test_num = optixGetPayload_0();
    optixSetPayload_0(intersection_test_num + 1); // number of intersection test
    if (point.x > params.predicate[0] && point.x < params.predicate[1] &&
        point.y > params.predicate[2] && point.y < params.predicate[3] && 
        point.z > params.predicate[4] && point.z < params.predicate[5] ) {
        set_result(params.result, primIdx);
        optixSetPayload_1(optixGetPayload_1() + 1);
    }
    optixIgnoreIntersection();
}