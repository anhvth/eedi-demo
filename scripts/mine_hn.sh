root_dir=$(pwd)


cd 3rd/FlagEmbedding/scripts
input_file=${root_dir}/$2
output_file=${root_dir}/$3
echo "input_file: $input_file"
echo "output_file: $output_file"



python hn_mine.py \
--embedder_name_or_path $1 \
--input_file $input_file \
--output_file $output_file \
--range_for_sampling 2-200 \
--negative_number 15

# --use_gpu_for_searching 