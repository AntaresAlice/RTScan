#ifndef WARPCORE_JOIN_CUH
#define WARPCORE_JOIN_CUH

#include "multi_value_hash_table.cuh"
#include "bucket_list_hash_table.cuh"

namespace warpcore
{

namespace join
{

template<class Key, class ValueR, class ValueS>
HOSTQUALIFIER INLINEQUALIFIER
Status natural_join(
    const Key * const keys_r_in,
    const ValueR * const values_r_in,
    const index_t num_r_in,
    const Key * const keys_s_in,
    const ValueS * const values_s_in,
    const index_t num_s_in,
    thrust::tuple<Key, ValueR, ValueS> * const result_out,
    index_t& num_out,
    const cudaStream_t stream = 0,
    const index_t probing_length = defaults::probing_length()) noexcept
{
    MultiValueHashTable<Key, ValueR> hash_table(float(num_r_in) / 0.6);
    //BucketListHashTable<Key, ValueR> hash_table(float(num_r_in) / 0.9, float(num_r_in) / 0.2);

    hash_table.insert(keys_r_in, values_r_in, num_r_in, stream, probing_length);

    hash_table.natural_join(
        keys_s_in,
        values_s_in,
        num_s_in,
        result_out,
        num_out,
        stream,
        probing_length);
    return hash_table.peek_status(stream);
}

} // namespace join

} // namespace warpcore

#endif /* WARPCORE_JOIN_CUH */