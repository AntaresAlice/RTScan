#pragma once

#include "optix_helpers.cuh"
#include "definitions.h"


template<typename key_type, typename value_type>
struct typed_query_params {
    OptixTraversableHandle traversable;

    bool long_keys;
    bool has_range_queries;

    const key_type* query_lower;
    const key_type* query_upper;

    const value_type* stored_values;

    value_type* result;
};

using query_params = typed_query_params<void, rti_v32>;
