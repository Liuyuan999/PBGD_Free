#!/bin/bash
set -x

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "MASTER_PORT=$MASTER_PORT"

# List of algorithms and their corresponding save paths
declare -A algorithms=(
  [ALRIGHT]="/data/liuyuan/bipost/checkpoint/llama3-3b-dpo-sft_ALRIGHT"
  [VaFF]="/data/liuyuan/bipost/checkpoint/llama3-3b-dpo-sft_VaFF"
  [F2SA]="/data/liuyuan/bipost/checkpoint/llama3-3b-dpo-sft_F2SA"
  [BOME]="/data/liuyuan/bipost/checkpoint/llama3-3b-dpo-sft_BOME"
)


# Loop over each algorithm
for algo in "${!algorithms[@]}"; do
  save_path="${algorithms[$algo]}"
  
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
  echo "Visible to this job: $CUDA_VISIBLE_DEVICES"
  for stage in 1 2; do
  training_command="CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --include localhost:0,1,2,3  --master_port=$MASTER_PORT --module bipost.cli.semantic_analysis \
   --model_stage $stage \
   --pretrain meta-llama/Llama-3.2-3B \
   --lora_alpha 16 \
   --lora_rank 16 \
   --target_module q_proj v_proj \
   --lambd 0.5 \
   --train_batch_size 72 \
   --micro_train_batch_size 1 \
   --learning_rate 1e-5 \
   --obj_1 DPO \
   --dataset_1 Dahoas/rm-hh-rlhf \
   --pref_prompt_key prompt \
   --pref_chosen_key chosen \
   --pref_rejected_key rejected \
   --max_samples_1 4800 \
   --max_eval_samples_1 1200 \
   --ref_model meta-llama/Llama-3.2-3B-Instruct \
   --obj_2 SFT \
   --dataset_2 Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --max_samples_2 4800 \
   --max_eval_samples_2 1200 \
   --train True \
   --max_training_time 18000 \
   --max_SFT_stage_time 3600 \
   --algorithm $algo \
   --output_layer_name lm_head \
   --gam 10 \
   --inner_stepsize 3 \
   --save_path $save_path \
   --max_epochs 60 \
   --zero_stage 2 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --bf16 \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing"

  if [[ ${1} != "slurm" ]]; then
      echo "Running command:"
      echo "$training_command"
      eval "$training_command"
  fi
done
done


#!/bin/bash
set -x

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "MASTER_PORT=$MASTER_PORT"

# List of algorithms and their corresponding save paths
declare -A algorithms=(
  [ALRIGHT]="/data/liuyuan/bipost/checkpoint/test_model_ALRIGHT"
  [VaFF]=[VaFF]="/data/liuyuan/bipost/checkpoint/test_model_VaFF"
  [F2SA]="/data/liuyuan/bipost/checkpoint/test_model_F2SA"
  [BOME]="/data/liuyuan/bipost/checkpoint/test_model_BOME"
)


# Loop over each algorithm
for algo in "${!algorithms[@]}"; do
  save_path="${algorithms[$algo]}"
  
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
  echo "Visible to this job: $CUDA_VISIBLE_DEVICES"
  for stage in 1 2; do
  training_command="CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --include localhost:2  --master_port=$MASTER_PORT --module bipost.cli.semantic_analysis \
   --model_stage $stage \
   --pretrain EleutherAI/pythia-1b \
   --lora_alpha 16 \
   --lora_rank 16 \
   --target_modules query_key_value \
   --lambd 0.5 \
   --train_batch_size 16 \
   --micro_train_batch_size 1 \
   --learning_rate 1e-5 \
   --obj_1 DPO \
   --dataset_1 Dahoas/rm-hh-rlhf \
   --pref_prompt_key prompt \
   --pref_chosen_key chosen \
   --pref_rejected_key rejected \
   --max_samples_1 1800 \
   --ref_model EleutherAI/pythia-1b \
   --obj_2 SFT \
   --dataset_2 Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --max_samples_2 1800 \
   --train True \
   --max_training_time 7200 \
   --max_SFT_stage_time 1200 \
   --algorithm $algo \
   --output_layer_name embed_out \
   --gam 15 \
   --inner_stepsize 5 \
   --save_path $save_path \
   --max_epochs 10 \
   --zero_stage 2 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --bf16 \
   --load_checkpoint \
   --gradient_checkpointing"

  if [[ ${1} != "slurm" ]]; then
      echo "Running command:"
      echo "$training_command"
      eval "$training_command"
  fi
done
done
