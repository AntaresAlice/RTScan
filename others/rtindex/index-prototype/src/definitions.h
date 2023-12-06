#ifndef TEST_CODE_RTINDEX_TYPES_H
#define TEST_CODE_RTINDEX_TYPES_H

#include <limits>

using rti_k32 = uint32_t;
using rti_v32 = uint32_t;

using rti_k64 = uint64_t;
using rti_v64 = uint64_t;

using rti_idx = uint32_t;


template <typename key_type, typename value_type>
value_type value_for_key(key_type key) {
    return value_type(size_t(key) << 1u) + 1;
}

template <typename value_type>
constexpr value_type not_found = std::numeric_limits<value_type>::max();

// never generate negative keys (conflict with b+ tree)
// never generate key 0 (conflict with b+ tree)
// never generate MAX_KEY (conflict with b+ tree and hash table)
// never generate MAX_KEY - 1 (conflict with hash table)
template <typename key_type>
constexpr key_type min_usable_key = 1;
template <typename key_type>
constexpr key_type max_usable_key = std::numeric_limits<key_type>::max() - 2;

#endif
