#include "launch_parameters.cuh"
#include "optix_helpers.cuh"


extern "C" __constant__ query_params params;

extern "C" GLOBALQUALIFIER void __closesthit__query() {
    // do nothing
}

extern "C" GLOBALQUALIFIER void __miss__query() {
    // do nothing
}

extern "C" GLOBALQUALIFIER void __anyhit__query() {
    const unsigned int primitive_id = optixGetPrimitiveIndex();

    rti_v32 value = params.stored_values[primitive_id];

    bool is_range_query = get_secondary_payload_32<bool>();
    if (is_range_query) {
        set_payload_32(get_payload_32<rti_v32>() + value);
        // reject the hit, this prevents tmax from being reduced
        optixIgnoreIntersection();
    } else {
        set_payload_32(value);
    }
}


extern "C" GLOBALQUALIFIER void __raygen__query() {
    const unsigned int ix = optixGetLaunchIndex().x;

    rti_k64 key = params.long_keys ? ((rti_k64*)params.query_lower)[ix] : ((rti_k32*)params.query_lower)[ix];
    rti_k64 lower_bound, upper_bound;

    bool is_range_query = params.has_range_queries;
    if (is_range_query) {
        lower_bound = key;
        upper_bound = params.long_keys ? ((rti_k64*)params.query_upper)[ix] : ((rti_k32*)params.query_upper)[ix];
    }

    uint32_t i0 = is_range_query ? 0 : not_found<rti_v32>;
    uint32_t i1 = is_range_query;

    // if lower_bound == upper_bound, we can cast a perpendicular ray!
    if (is_range_query && lower_bound != upper_bound) {
        // decompose the key into x and yz
        rti_k64 smallest_yz = lower_bound >> 22u;
        rti_k64 largest_yz = upper_bound >> 22u;
        float smallest_x = uint32_as_float(lower_bound & 0x3fffffu);
        float largest_x = uint32_as_float(upper_bound & 0x3fffffu);
        float smallest_possible_x = uint32_as_float(0);
        float largest_possible_x = uint32_as_float(0x3fffffu);

        // cast one ray per yz offset
        for (uint64_t yz = smallest_yz; yz <= largest_yz; ++yz) {
            float y = uint32_as_float(yz & 0x3fffffu);
            float z = uint32_as_float(yz >> 22u);

            float from = minus_eps(yz == smallest_yz ? smallest_x : smallest_possible_x);
            float to = plus_eps(yz == largest_yz ? largest_x : largest_possible_x);

            float3 origin = make_float3(from, y, z);
            float3 direction = make_float3(1, 0, 0);
            float tmin = 0;
            float tmax = to - from;

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
                    i1);
        }
        params.result[ix] = i0;
    } else {
        float x = uint32_as_float(key & 0x3fffffu);
        float y = uint32_as_float((key >> 22u) & 0x3fffffu);
        float z = uint32_as_float(key >> 44u);
    
        // perpendicular ray
        float3 origin = make_float3(x, y, minus_eps(z));
        float3 direction = make_float3(0, 0, 1);
        float tmin = 0;
        float tmax = 1;
        
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
                i0,
                i1);
        params.result[ix] = i0;
    }
}
