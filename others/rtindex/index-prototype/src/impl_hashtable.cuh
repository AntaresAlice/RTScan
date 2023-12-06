#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <optional>

#include "../warpcore/warpcore.cuh"
#include "definitions.h"


namespace cg = cooperative_groups;


template<class wc_table>
GLOBALQUALIFIER
void warpcore_insert(wc_table hash_table, const typename wc_table::key_type* input_keys, const size_t num_in) {
    const size_t tid = global_thread_id();
    const size_t gid = tid / wc_table::cg_size();
    const auto group = cg::tiled_partition<wc_table::cg_size()>(cg::this_thread_block());

    if (gid >= num_in) return;
    hash_table.insert(input_keys[gid], static_cast<rti_idx>(gid), group);
}


template<class wc_table, typename value_type>
GLOBALQUALIFIER
void warpcore_probe(wc_table hash_table, const typename wc_table::key_type* keys, const value_type* stored_values, value_type* result, const size_t num_in) {
    const size_t tid = global_thread_id();
    const size_t gid = tid / wc_table::cg_size();
    const auto group = cg::tiled_partition<wc_table::cg_size()>(cg::this_thread_block());
    if (gid >= num_in) return;

    typename wc_table::key_type key = keys[gid];
    rti_idx local_offset;
    const auto status = hash_table.retrieve(key, local_offset, group);
    if (group.thread_rank() == 0) {
        if (status.has_key_not_found() || status.has_probing_length_exceeded()) {
            result[gid] = not_found<value_type>;
        } else {
            result[gid] = stored_values[local_offset];
        }
    }
}


template <typename key_type_>
class hashtable {
public:
    using key_type = key_type_;

private:
    using wc_table = warpcore::SingleValueHashTable<key_type, rti_idx, key_type(-2), key_type(-1)>;
    constexpr static double load_factor = 0.8;
    
    std::optional<wc_table> wrapped_table;

public:
    static std::string short_description() {
        return "warpcore";
    }

    size_t gpu_resident_bytes() {
        return wrapped_table.value().bytes_total();
    }

    void build(const key_type* keys, size_t size, double* build_time_ms, size_t* build_bytes) {
        const size_t capacity = static_cast<size_t>(size / load_factor);
        wrapped_table.emplace(capacity);

        cuda_timer timer(0);
        timer.start();

        warpcore_insert<<<SDIV(size * wc_table::cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE>>>(
            wrapped_table.value(),
            keys,
            size
        );

        timer.stop();
        if (build_time_ms) *build_time_ms += timer.time_ms();
        if (build_bytes) *build_bytes += gpu_resident_bytes();

        cudaDeviceSynchronize(); CUERR
    }

    template <typename value_type>
    void query(const value_type* value_column, const key_type* keys, value_type* result, size_t size, cudaStream_t stream) {

        warpcore_probe<<<SDIV(size * wc_table::cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
            wrapped_table.value(),
            keys,
            value_column,
            result,
            size
        );
    }

    // unsupported
    // void range_query_sum()

    void destroy() {
        wrapped_table.reset();
    }
};


#endif
