
min_num_ops=1'000'000
max_num_ops=45'000'000
num_ops_step=1'000'000

num_experiments=20

exec_dir="./bin"
exec_name="concurrent_insert_range_bench"
device=0
output_dir="../results"
mkdir -p output_dir
additional_args="--validate=false --num-experiments=1"

# update_ratios=(0.05 0.50 0.90)
# range_length=(8 32)
# initial_sizes=(1'000'000 40'000'000)
update_ratios=(0.0)
range_length=(100'000'000)
initial_sizes=(100'000'000)

# for range in "${range_length[@]}"
# do
#     for isize in "${initial_sizes[@]}"
#     do
#         for uratio in "${update_ratios[@]}"
#         do
#             # for num_ops in $(seq $min_num_ops $num_ops_step $max_num_ops)
#             echo ${exec_dir}/${exec_name} --range-length=${range} --initial-size=${isize} --update-ratio=${uratio} --num-ops=1 --selectivity=${selec_list[i]} --input-file=${input_file_list[i]} --output-dir=${output_dir} ${additional_args}
#             ${exec_dir}/${exec_name} --range-length=${range} --initial-size=${isize} --update-ratio=${uratio} --num-ops=1 --selectivity=${selec_list[i]} --input-file=${input_file_list[i]} --output-dir=${output_dir} ${additional_args} >> ../results/bench_concurrent_insert_range.log
#             # for selec in "${selec_list[@]}"
#             # do
#             #     for input_file in "${input_file_list[@]}"
#             #     do
#             #         echo ${exec_dir}/${exec_name} --range-length=${range} --initial-size=${isize} --update-ratio=${uratio} --num-ops=1 --selectivity=${selec} --input-file=${input_file} --output-dir=${output_dir} ${additional_args}
#             #         ${exec_dir}/${exec_name} --range-length=${range} --initial-size=${isize} --update-ratio=${uratio} --num-ops=1 --selectivity=${selec} --input-file=${input_file} --output-dir=${output_dir} ${additional_args} >> ../results/bench_concurrent_insert_range.log
#             #     done
#             # done
#         done
#     done
# done

# input_file=/home/wzm/bindex-raytracing/data/zipf1.5_data_1e8_3.dat
input_file=/home/wzm/bindex-raytracing/data/uniform_data_1e8_3.dat
echo ${exec_dir}/${exec_name} --range-length=${range_length} --initial-size=${initial_sizes} --update-ratio=${update_ratios} --num-ops=1 --selectivity=1 --input-file=${input_file} --output-dir=${output_dir} ${additional_args}
${exec_dir}/${exec_name} --range-length=${range_length} --initial-size=${initial_sizes} --update-ratio=${update_ratios} --num-ops=1 --selectivity=1 --input-file=${input_file} --output-dir=${output_dir} ${additional_args} > ../results/bench_concurrent_insert_range_index_overhead.log