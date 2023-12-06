#ifndef TREE_H
#define TREE_H

#include "gpu_btree.h"

namespace cg = cooperative_groups;

using GpuBTree::gpu_blink_tree;


template <typename key_type, typename value_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_find_kernel(
    const key_type* keys,
    const value_type* stored_values,
    value_type* results,
    const size_type keys_count,
    btree tree,
    bool concurrent = false
) {
  auto thread_id  = threadIdx.x + blockIdx.x * blockDim.x;

  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<btree::branching_factor>(block);

  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto key     = btree::invalid_key;
  auto value   = btree::invalid_value;
  bool to_find = false;
  if (thread_id < keys_count) {
    key     = keys[thread_id];
    to_find = true;
  }

  using allocator_type = device_allocator_context<typename btree::allocator_type>;
  allocator_type allocator{tree.allocator_, tile};

  auto work_queue = tile.ballot(to_find);
  while (work_queue) {
    auto cur_rank = __ffs(work_queue) - 1;
    auto cur_key  = tile.shfl(key, cur_rank);
    value_type cur_result;
    cur_result = tree.cooperative_find(cur_key, tile, allocator, concurrent);
    if (cur_rank == tile.thread_rank()) {
      value   = cur_result;
      to_find = false;
    }
    work_queue = tile.ballot(to_find);
  }

  if (thread_id < keys_count) {
    results[thread_id] = value != std::numeric_limits<uint32_t>::max() ? stored_values[value] : not_found<value_type>;
  }
}

template <typename key_type, typename value_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_range_query_kernel(
    const key_type* lower_bounds,
    const key_type* upper_bounds,
    const value_type* stored_values,
    value_type* results,
    const size_type keys_count,
    btree tree,
    bool concurrent = false
) {
    auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    auto block                = cg::this_thread_block();
    auto tile                 = cg::tiled_partition<btree::branching_factor>(block);
    auto tile_id              = thread_id / btree::branching_factor;
    auto first_tile_thread_id = tile_id * btree::branching_factor;

    if ((thread_id - tile.thread_rank()) >= keys_count) { return; }
  
    auto lower_bound = btree::invalid_key;
    auto upper_bound = btree::invalid_key;
  
    bool to_find    = false;
    if (thread_id < keys_count) {
      lower_bound = lower_bounds[thread_id];
      upper_bound = upper_bounds[thread_id];
      to_find     = true;
    }
  
    using allocator_type = device_allocator_context<typename btree::allocator_type>;
    allocator_type allocator{tree.allocator_, tile};
  
    value_type result = 0;
    auto work_queue = tile.ballot(to_find);
    while (work_queue) {
      value_type cur_result = 0;
      auto cur_rank        = __ffs(work_queue) - 1;
      auto cur_lower_bound = tile.shfl(lower_bound, cur_rank);
      auto cur_upper_bound = tile.shfl(upper_bound, cur_rank);
#ifdef SINGLE_ELEMENT_RANGE_IS_POINT_QUERY
      if (cur_lower_bound == cur_upper_bound) {
        auto r = tree.cooperative_find(cur_lower_bound, tile, allocator, concurrent);
        cur_result = r != std::numeric_limits<uint32_t>::max() ? r : 0;
      } else {
#endif
        cur_result = tree.modified_cooperative_range_query(
          cur_lower_bound,
          cur_upper_bound,
          tile,
          allocator,
          stored_values,
          concurrent);
#ifdef SINGLE_ELEMENT_RANGE_IS_POINT_QUERY
      }
#endif

      if (cur_rank == tile.thread_rank()) {
        result = cur_result;
        to_find = false;
      }
      work_queue = tile.ballot(to_find);
    }
  
    if (thread_id < keys_count) {
        results[thread_id] = result;
    }
}


template <typename key_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_insert_kernel(
    const key_type* keys,
    const size_type keys_count,
    btree tree
) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block     = cg::this_thread_block();
  auto tile      = cg::tiled_partition<btree::branching_factor>(block);

  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto key       = btree::invalid_key;
  auto value     = btree::invalid_value;
  bool to_insert = false;
  if (thread_id < keys_count) {
    key       = keys[thread_id];
    value     = thread_id;
    to_insert = true;
  }
  using allocator_type = typename btree::device_allocator_context_type;
  allocator_type allocator{tree.allocator_, tile};

  size_type num_inserted = 1;
  auto work_queue        = tile.ballot(to_insert);
  while (work_queue) {
    auto cur_rank  = __ffs(work_queue) - 1;
    auto cur_key   = tile.shfl(key, cur_rank);
    auto cur_value = tile.shfl(value, cur_rank);

    tree.cooperative_insert(cur_key, cur_value, tile, allocator);

    if (tile.thread_rank() == cur_rank) { to_insert = false; }
    num_inserted++;
    work_queue = tile.ballot(to_insert);
  }
}


void investigate_tree_deadlock() {
    using key_type = uint32_t;
    using value_type = uint32_t;

    size_t build_size = size_t{1} << 25;
    key_type min_usable_key = 1;
    key_type max_usable_key = std::numeric_limits<key_type>::max() - 2;

    std::mt19937_64 gen(42);
    std::uniform_int_distribution<key_type> key_dist(min_usable_key, max_usable_key);
    std::vector<key_type> build_keys(build_size);
    std::unordered_set<key_type> build_keys_set;
    while (build_keys_set.size() < build_size) {
        key_type key = key_dist(gen);
        build_keys_set.insert(key);
    }
    std::copy(build_keys_set.begin(), build_keys_set.end(), build_keys.begin());
    std::sort(build_keys.begin(), build_keys.end());

    key_type* keys_on_gpu;
    cudaMalloc(&keys_on_gpu, build_size * sizeof(key_type));
    cudaMemcpy(keys_on_gpu, build_keys.data(), build_size * sizeof(key_type), cudaMemcpyHostToDevice);

    for (size_t i = 0; i < 10000; ++i) {
        std::cout << "round " << i << " starting" << std::endl;

        gpu_blink_tree<key_type, value_type, 16> tree;
        modified_insert_kernel<<<(build_size + 511) / 512, 512>>>(keys_on_gpu, build_size, tree);

        std::cout << "tree uses " << tree.compute_memory_usage() << " GB" << std::endl;
        std::cout << "round " << i << " done" << std::endl;
    }

    cudaFree(keys_on_gpu);
}


void test_modified_btree() {
    std::vector<uint32_t> keys {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<uint32_t> offsets(keys.size());
    std::iota(offsets.begin(), offsets.end(), 0);
    std::vector<uint32_t> values {10, 20, 30, 40, 50, 60, 70, 80, 90};
    std::vector<uint32_t> probes       {1, 4, 5, 2, 1, 3, 7, 10};
    std::vector<uint32_t> probes_upper {5, 5, 5, 2, 1, 3, 8, 20};
    std::vector<uint32_t> results(probes.size());
    cuda_buffer kb, ob, vb, pb, pub, rb;
    kb.alloc_and_upload(keys);
    ob.alloc_and_upload(offsets);
    vb.alloc_and_upload(values);
    pb.alloc_and_upload(probes);
    pub.alloc_and_upload(probes_upper);
    rb.alloc_and_upload(results);

    using btree = gpu_blink_tree<uint32_t, uint32_t>;

    btree tree;
    tree.bulk_load(kb.ptr<uint32_t>(), ob.ptr<uint32_t>(), keys.size(), true);
    cudaDeviceSynchronize(); CUERR
    //modified_insert_kernel<<<SDIV(keys.size(), 512), 512>>>(kb.ptr<uint32_t>(), keys.size(), tree);
    //modified_find_kernel<<<SDIV(probes.size(), 512), 512>>>(pb.ptr<uint32_t>(), vb.ptr<uint32_t>(), rb.ptr<uint32_t>(), probes.size(), tree);
    modified_range_query_kernel<<<SDIV(probes.size(), 512), 512>>>(pb.ptr<uint32_t>(), pub.ptr<uint32_t>(), vb.ptr<uint32_t>(), rb.ptr<uint32_t>(), probes.size(), tree);
    cudaDeviceSynchronize(); CUERR

    rb.download(results.data(), results.size());
    for (auto x: results) {
        std::cout << x << std::endl;
    }
}

template <typename key_type_>
class tree {
public:
    using key_type = key_type_;

private:

    std::optional<gpu_blink_tree<key_type, rti_v32, 16>> wrapped_tree;

public:
    static std::string short_description() {
        return "b_link_tree";
    }

    size_t gpu_resident_bytes() {
        return wrapped_tree.value().compute_memory_usage_bytes();
    }

    void build(const key_type* keys, size_t size, double* build_time_ms, size_t* build_bytes) {

        cuda_buffer sorted_keys_buffer, sorted_offsets_buffer;

        {
            cuda_buffer temp_buffer, offsets_buffer;
            sorted_keys_buffer.alloc(sizeof(key_type) * size);
            offsets_buffer.alloc(sizeof(rti_idx) * size);
            sorted_offsets_buffer.alloc(sizeof(rti_idx) * size);
            init_offsets(offsets_buffer.ptr<rti_idx>(), size, build_time_ms);

            cudaDeviceSynchronize(); CUERR

            size_t temp_storage_bytes = find_pair_sort_buffer_size<key_type, rti_idx>(size);
            temp_buffer.alloc(temp_storage_bytes);
            timed_pair_sort(
                temp_buffer.raw_ptr, temp_storage_bytes,
                keys, sorted_keys_buffer.ptr<key_type>(), offsets_buffer.ptr<rti_idx>(), sorted_offsets_buffer.ptr<rti_idx>(), size, build_time_ms);

            if (build_bytes) *build_bytes += sorted_keys_buffer.size_in_bytes + sorted_offsets_buffer.size_in_bytes + temp_buffer.size_in_bytes + offsets_buffer.size_in_bytes;

            cudaDeviceSynchronize(); CUERR
        }

        wrapped_tree.emplace();

        cuda_timer timer(0);
        timer.start();

        wrapped_tree.value().bulk_load(sorted_keys_buffer.ptr<key_type>(), sorted_offsets_buffer.ptr<rti_idx>(), size, true, 0);
        //modified_insert_kernel<<<SDIV(size, 512), 512>>>(keys, size, wrapped_tree.value());

        timer.stop();
        if (build_time_ms) *build_time_ms += timer.time_ms();
        if (build_bytes) *build_bytes += gpu_resident_bytes();

        cudaDeviceSynchronize(); CUERR
    }

    template <typename value_type>
    void query(const value_type* value_column, const key_type* keys, value_type* result, size_t size, cudaStream_t stream) {
        modified_find_kernel<<<SDIV(size, 512), 512, 0, stream>>>(keys, value_column, result, size, wrapped_tree.value());
    }

    template <typename value_type>
    void range_query_sum(const value_type* value_column, const key_type* lower, const key_type* upper, value_type* result, size_t size, cudaStream_t stream) {
        modified_range_query_kernel<<<SDIV(size, 512), 512, 0, stream>>>(lower, upper, value_column, result, size, wrapped_tree.value());
    }

    void destroy() {
        wrapped_tree.reset();
    }
};

#endif
