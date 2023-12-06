#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include "definitions.h"
#include "cuda_helpers.cuh"


#define OPTIX_CHECK( call )                                             \
  {                                                                     \
    OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
        exit( 2 );                                                      \
      }                                                                 \
  }


HOSTQUALIFIER DEVICEQUALIFIER INLINEQUALIFIER
float uint32_as_float(uint32_t i) {
    return static_cast<float>(i + 1);
}

DEVICEQUALIFIER INLINEQUALIFIER
float plus_eps(float f) {
    return f + 0.5;
}

DEVICEQUALIFIER INLINEQUALIFIER
float minus_eps(float f) {
    return f - 0.5;
}

template <typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
packed_type unpack(uint32_t i0, uint32_t i1) {
    static_assert(sizeof(packed_type) <= 8);
    uint64_t uptr = static_cast<uint64_t>(i0) << 32u | i1;
    packed_type ptr = *reinterpret_cast<packed_type*>(&uptr);
    return ptr;
}

template <typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
void pack(packed_type ptr, uint32_t& i0, uint32_t& i1) {
    static_assert(sizeof(packed_type) <= 8);
    const uint64_t uptr = *reinterpret_cast<uint64_t*>(&ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffffull;
}

template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
packed_type get_payload_64() {
    static_assert(sizeof(packed_type) <= 8);
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return unpack<packed_type>(u0, u1);
}

template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
void set_payload_64(packed_type i) {
    static_assert(sizeof(packed_type) <= 8);
    uint32_t i0, i1;
    pack(i, i0, i1);
    optixSetPayload_0(i0);
    optixSetPayload_1(i1);
}

template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
packed_type get_payload_32() {
    static_assert(sizeof(packed_type) <= 4);
    return (packed_type) optixGetPayload_0();
}

template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
void set_payload_32(packed_type i) {
    static_assert(sizeof(packed_type) <= 4);
    optixSetPayload_0((uint32_t) i);
}

template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
packed_type get_secondary_payload_32() {
    static_assert(sizeof(packed_type) <= 4);
    return (packed_type) optixGetPayload_1();
}

template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
void set_secondary_payload_32(packed_type i) {
    static_assert(sizeof(packed_type) <= 4);
    optixSetPayload_1((uint32_t) i);
}
