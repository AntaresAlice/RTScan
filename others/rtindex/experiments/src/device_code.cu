#include <cuda_runtime.h>

#include "test_configuration.h"

#include "launch_parameters.cuh"
#include "optix_helpers.cuh"


extern "C" __constant__ launch_parameters params;


extern "C" __global__ void __closesthit__test() {
    // do nothing
}

extern "C" __global__ void __miss__test() {
    // do nothing
}

extern "C" __global__ void __intersection__test() {
    const unsigned int ix = optixGetLaunchIndex().x;
    const unsigned int primitive_id = optixGetPrimitiveIndex();
 
    key_type hit = params.build_keys[primitive_id];
    key_type lower_bound = params.query_lower[ix];

#if RANGE_QUERY_HIT_COUNT_LOG != 0
    key_type upper_bound = params.query_upper[ix];
    if (lower_bound <= hit && hit <= upper_bound) {
        value_type value = params.build_values[primitive_id];    
        set_payload_32(get_payload_32<value_type>() + value);
        // do not report hit, since we are done processing this hit!
    }
#else
    if (lower_bound == hit) {
        value_type value = params.build_values[primitive_id];
        set_payload_32(value);
        // do not report hit, since we are done processing this hit!
    }
#endif
}

extern "C" __global__ void __anyhit__test() {
    const unsigned int primitive_id = optixGetPrimitiveIndex();

    value_type value = params.build_values[primitive_id];
    optixSetPayload_1(optixGetPayload_1() + 1);

#if RANGE_QUERY_HIT_COUNT_LOG != 0
    set_payload_32(get_payload_32<value_type>() + value);
    // reject the hit, this prevents tmax from being reduced
    optixIgnoreIntersection();
#else
    set_payload_32(value);
#endif
}


#if INT_TO_FLOAT_CONVERSION_MODE == 3

extern "C" __global__ void __raygen__test() {
    const unsigned int ix = optixGetLaunchIndex().x;

    key_type key = params.query_lower[ix];

    // point query vs range query distinction
    // this can also be decided at run-time
#if RANGE_QUERY_HIT_COUNT_LOG != 0
    key_type upper_bound = params.query_upper[ix];

    // decompose the key into x and yz
    key_type smallest_yz = key >> 22u;
    key_type largest_yz = upper_bound >> 22u;
    float smallest_x = uint32_as_float(key & 0x3fffffu);
    float largest_x = uint32_as_float(upper_bound & 0x3fffffu);
    float smallest_possible_x = uint32_as_float(0);
    float largest_possible_x = uint32_as_float(0x3fffffu);

    value_type i0 = 0;
    value_type is = 0;
    // cast one ray per yz offset
    for (uint64_t yz = smallest_yz; yz <= largest_yz; ++yz) {
        float offset_y = uint32_as_float(yz & 0x3fffffu);
        float offset_z = uint32_as_float(yz >> 22u); // 0

#if START_RAY_AT_ZERO != 0
        float3 origin = make_float3(0, offset_y, offset_z);
        float3 direction = make_float3(1, 0, 0);
        float tmin = minus_eps(yz == smallest_yz ? smallest_x : smallest_possible_x);
        float tmax = plus_eps(yz == largest_yz ? largest_x : largest_possible_x);
#else
        float start_x = minus_eps(yz == smallest_yz ? smallest_x : smallest_possible_x);
        float end_x = plus_eps(yz == largest_yz ? largest_x : largest_possible_x);
        float3 origin = make_float3(start_x, offset_y, offset_z);
        float3 direction = make_float3(1, 0, 0);
        float tmin = 0;
        float tmax = end_x - start_x;
#endif

        optixTrace(
                params.traversable,
                origin,
                direction,
                tmin,
                tmax,
                0.0f,
                OptixVisibilityMask(255),
                // we can use TERMINATE_ON_FIRST_HIT for a range
                // query since all hits will be rejected anyway
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                0,
                0,
                0,
                i0,
                is);
    }
    params.result[ix] = i0;
    params.intersection_num[ix] = is;

#else
    float offset_x, offset_y, offset_z;
    key_to_coordinates(key, offset_x, offset_y, offset_z);

    float start_x = minus_eps(offset_x);
    float end_x = plus_eps(offset_x);
    float start_z = minus_eps(offset_z);
    float end_z = plus_eps(offset_z);

#if PERPENDICULAR_RAYS != 0
#if START_RAY_AT_ZERO != 0
    float3 origin = make_float3(offset_x, offset_y, 0);
    float3 direction = make_float3(0, 0, 1);
    float tmin = start_z;
    float tmax = end_z;
#else
    float3 origin = make_float3(offset_x, offset_y, start_z);
    float3 direction = make_float3(0, 0, 1);
    float tmin = 0;
    float tmax = end_z - start_z;
#endif
#else
#if START_RAY_AT_ZERO != 0
    float3 origin = make_float3(0, offset_y, offset_z);
    float3 direction = make_float3(1, 0, 0);
    float tmin = start_x;
    float tmax = end_x;
#else
    float3 origin = make_float3(start_x, offset_y, offset_z);
    float3 direction = make_float3(1, 0, 0);
    float tmin = 0;
    float tmax = end_x - start_x;
#endif
#endif
    
    value_type i0 = NOT_FOUND;
    optixTrace(
            params.traversable,
            origin,
            direction,
            tmin,
            tmax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            0,
            0,
            0,
            i0);
    params.result[ix] = i0;
#endif
}

#else

extern "C" __global__ void __raygen__test() {
    const unsigned int ix = optixGetLaunchIndex().x;

    float key = uint32_as_float(params.query_lower[ix]);
    float zero = uint32_as_float(0);

    // point query vs range query distinction
    // this can also be decided at run-time
#if RANGE_QUERY_HIT_COUNT_LOG != 0
    float upper_bound = uint32_as_float(params.query_upper[ix]);

#if START_RAY_AT_ZERO != 0
    float3 origin = make_float3(0, zero, zero);
    float3 direction = make_float3(1, 0, 0);
    float tmin = minus_eps(key);
    float tmax = plus_eps(upper_bound);
#else
    float start_x = minus_eps(key);
    float end_x = plus_eps(upper_bound);
    float3 origin = make_float3(start_x, zero, zero);
    float3 direction = make_float3(1, 0, 0);
    float tmin = 0;
    float tmax = end_x - start_x;
#endif

    value_type i0 = 0;
    optixTrace(
            params.traversable,
            origin,
            direction,
            tmin,
            tmax,
            0.0f,
            OptixVisibilityMask(255),
            // we can use TERMINATE_ON_FIRST_HIT for a range
            // query since all hits will be rejected anyway
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            0,
            0,
            0,
            i0);
    params.result[ix] = i0;
#else

#if PERPENDICULAR_RAYS != 0
    float3 origin = make_float3(key, zero, 0);
    float3 direction = make_float3(0, 0, 1);
    float tmin = 0;
    float tmax = plus_eps(zero);
#else
#if START_RAY_AT_ZERO != 0
    float3 origin = make_float3(0, zero, zero);
    float3 direction = make_float3(1, 0, 0);
    float tmin = minus_eps(key);
    float tmax = plus_eps(key);
#else
    float start_x = minus_eps(key);
    float end_x = plus_eps(key);
    float3 origin = make_float3(start_x, zero, zero);
    float3 direction = make_float3(1, 0, 0);
    float tmin = 0;
    float tmax = end_x - start_x;
#endif
#endif

    value_type i0 = NOT_FOUND;
    optixTrace(
            params.traversable,
            origin,
            direction,
            tmin,
            tmax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            0,
            0,
            0,
            i0);
    params.result[ix] = i0;
#endif
}

#endif
