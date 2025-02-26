# Mistral
set -e

MODE=$1
OUTPUT_DIR=outputs/debug
BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct-AWQ
BASE_MODEL_32BIT=Qwen/Qwen2.5-0.5B
BASE_MODEL_TRAIN=$BASE_MODEL
PEFT=""
# PEFT="--from_peft outputs/llm_hn_qw32b/r2_from_32b_qnq.lb48"
#=====================================
# Define PEFT variable with appropriate value

export CUDA_VISIBLE_DEVICES=1

GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

TARGET_MODULE="q_proj k_proj v_proj o_proj gate_proj down_proj up_proj"
FLAG_DIR=/mnt/data/anhvth5/projects/FlagEmbedding
EPOCH=1
TORCHRUN=/home/ubuntu/.conda/envs/flag-embedding/bin/torchrun

                    # ./data/select_better_reasoning.jsonl \
if [ "$MODE" == "train" ]; then
    $TORCHRUN \
        --nproc_per_node $GPU_COUNT --master_port 29501 \
        -m FlagEmbedding.finetune.embedder.decoder_only.icl \
        --model_name_or_path $BASE_MODEL_TRAIN \
        --use_lora True \
        --additional_special_tokens '<instruct>' '<query>' '<response>' '<doc_embed>' \
        --lora_rank 16 \
        --lora_alpha 32 \
        --target_modules $TARGET_MODULE \
        --train_data /mnt/data/anhvth5/projects/models/kaggle-models/2211-lora-14b-transformers-default-v1/23k_synth_4k3_real_hn_data.k25.jsonl \
        --cache_path .cache/data \
        --train_group_size 2 \
        --query_max_len 256 \
        --passage_max_len 40 \
        --pad_to_multiple_of 8 \
        --query_instruction_for_retrieval "Given a math multiple-choice problem with a student's wrong answer, retrieve the math misconceptions" \
        --query_instruction_format '<instruct>{}\n<query>{}' \
        --passage_instruction_format '{}{}<doc_embed>' \
        --passage_instruction_for_retrieval ''\
        --knowledge_distillation True \
        --same_dataset_within_batch True \
        --small_threshold 0 \
        --drop_threshold 0 \
        --example_query_max_len 256 \
        --example_passage_max_len 40 \
        --retrieval_use_examples True \
        --icl_suffix_str '\n<response>' \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir \
        --learning_rate 1e-4 \
        --fp16 \
        --num_train_epochs $EPOCH \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --dataloader_drop_last True \
        --warmup_ratio 0.01 \
        --logging_steps 1 \
        --save_steps 500 \
        --negatives_cross_device \
        --temperature 0.02 \
        --sentence_pooling_method last_token \
        --normalize_embeddings True \
        --kd_loss_type kl_div \
        --optim adamw_8bit \
        $PEFT

        # --gradient_checkpointing True \
        # --deepspeed $FLAG_DIR/examples/finetune/ds_stage1.json \
    /home/ubuntu/.conda/envs/awq/bin/python scripts/to_awq.py $BASE_MODEL_32BIT $OUTPUT_DIR $OUTPUT_DIR/awq/
elif [ "$MODE" == "awq" ]; then
    if [ -z "$2" ]; then
        echo "Error: Output directory for AWQ mode not specified."
        exit 1
    fi
    LORA_PATH=$2
    /home/ubuntu/.conda/envs/awq/bin/python scripts/to_awq.py $BASE_MODEL_32BIT $LORA_PATH $LORA_PATH/awq/
else
    echo "Invalid mode. Please use 'train' or 'awq'."
    exit 1
fi
