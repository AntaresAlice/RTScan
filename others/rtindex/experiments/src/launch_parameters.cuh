#pragma once

#include "optix_helpers.cuh"

template <typename key, typename value>
struct typed_launch_parameters {
    OptixTraversableHandle traversable;

    key* build_keys;
    value* build_values;
    key* query_lower;
    key* query_upper;
    value* result;

    value* intersection_num;
};

using untyped_launch_parameters = typed_launch_parameters<void, void>;
using launch_parameters = typed_launch_parameters<key_type, value_type>;
