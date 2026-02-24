# #!/bin/bash
# set -x

# # List of algorithms and their corresponding save paths
# declare -A algorithms=(
#   [ALRIGHT]="/data/liuyuan/bipost/checkpoint/llama3-8b-dpo-sft_ALRIGHT"
#   [VaFF]="/data/liuyuan/bipost/checkpoint/llama3-8b-dpo-sft_VaFF"
#   [PBGD]="/data/liuyuan/bipost/checkpoint/llama3-8b-dpo-sft_PBGD"
# )

# # Loop over each algorithm
# for algo in "${!algorithms[@]}"; do
#   save_path="${algorithms[$algo]}"

#   read -r -d '' training_commands <<EOF
# bipost.cli.train_bilevel \
#    --pretrain meta-llama/Llama-3.2-3B \
#    --lora_alpha 16 \
#    --lora_rank 16 \
#    --target_module q_proj v_proj \
#    --lambd 0.5 \

#    --train_batch_size 32 \
#    --micro_train_batch_size 1 \
#    --learning_rate 1e-5 \
   
#    --obj_1 DPO \
#    --dataset_1 Dahoas/rm-hh-rlhf \
#    --pref_prompt_key prompt \
#    --pref_chosen_key chosen \
#    --pref_rejected_key rejected \
#    --max_samples_1 9600 \

#    --ref_model meta-llama/Llama-3.2-3B-Instruct \
   
#    --obj_2 SFT \
#    --dataset_2 Open-Orca/OpenOrca \
#    --input_key question \
#    --output_key response \
#    --max_samples_2 9600 \

#    --train True \
#    --max_training_time 7200 \

#    --algorithm $algo \
#    --output_layer_name lm_head \
#    --gam 10 \
#    --inner_stepsize 5 \

#    --save_path $save_path \
#    --max_epochs 5 \
#    --zero_stage 2 \
#    --save_steps -1 \
#    --logging_steps 1 \
#    --eval_steps -1 \
#    --bf16 \
#    --flash_attn \
#    --load_checkpoint \
#    --gradient_checkpointing 
# EOF

#   if [[ ${1} != "slurm" ]]; then
#       deepspeed --num_gpus 2 --module $training_commands
#   fi
# done

# #!/bin/bash
# set -x

# # List of algorithms and their corresponding save paths
# declare -A algorithms=(
#   [VaFF]="/data/liuyuan/bipost/checkpoint/test_model_VaFF"
#   # [VaFF2]="/data/liuyuan/bipost/checkpoint/test_model_VaFF2"
#   # [F2SA]="/data/liuyuan/bipost/checkpoint/test_model_F2SA"
#   # [BOME]="/data/liuyuan/bipost/checkpoint/test_model_BOME"
#   [ALRIGHT]="/data/liuyuan/bipost/checkpoint/test_model_ALRIGHT"
# )

# # Loop over each algorithm
# for algo in "${!algorithms[@]}"; do
#   save_path="${algorithms[$algo]}"

#   read -r -d '' training_commands <<EOF
# bipost.cli.train_bilevel \
#    --pretrain meta-llama/Llama-3.2-3B \
#    --lora_alpha 16 \
#    --lora_rank 16 \
#    --target_module q_proj v_proj \
#    --lambd 0.5 \

#    --train_batch_size 32 \
#    --micro_train_batch_size 1 \
#    --learning_rate 1e-5 \
   
#    --obj_1 DPO \
#    --dataset_1 Dahoas/rm-hh-rlhf \
#    --pref_prompt_key prompt \
#    --pref_chosen_key chosen \
#    --pref_rejected_key rejected \
#    --max_samples_1 4800 \

#    --ref_model EleutherAI/pythia-1b \
   
#    --obj_2 SFT \
#    --dataset_2 Open-Orca/OpenOrca \
#    --input_key question \
#    --output_key response \
#    --max_samples_2 4800 \

#    --train True \
#    --max_training_time 120 \
#    --max_SFT_stage_time 120 \

#    --algorithm $algo \
#    --output_layer_name lm_head \
#    --gam 15 \
#    --inner_stepsize 5 \

#    --save_path $save_path \
#    --max_epochs 60 \
#    --zero_stage 2 \
#    --save_steps -1 \
#    --logging_steps 1 \
#    --eval_steps -1 \
#    --bf16 \
#    --flash_attn \
#    --load_checkpoint \
#    --gradient_checkpointing 
# EOF

#   if [[ ${1} != "slurm" ]]; then
#       deepspeed --num_gpus 2 --module $training_commands
#   fi
# done

#!/bin/bash
set -x

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "MASTER_PORT=$MASTER_PORT"

# List of algorithms and their corresponding save paths
declare -A algorithms=(
  # [ALRIGHT]="/data/liuyuan/bipost/checkpoint/llama3-3b-dpo-sft_ALRIGHT"
  [VaFF]="/data/liuyuan/bipost/checkpoint/llama3-3b-dpo-sft_VaFF"
  # [SEQ]="/data/liuyuan/bipost/checkpoint/llama3-3b-dpo-sft_SEQ"
  # [F2SA]="/data/liuyuan/bipost/checkpoint/llama3-3b-dpo-sft_F2SA_record"
  # [BOME]="/data/liuyuan/bipost/checkpoint/llama3-3b-dpo-sft_BOME"
)

# Loop over each algorithm
for algo in "${!algorithms[@]}"; do
  save_path="${algorithms[$algo]}"

  export CUDA_VISIBLE_DEVICES="0"
  echo "Visible to this job: $CUDA_VISIBLE_DEVICES"

  # training_command="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} deepspeed --master_port=$MASTER_PORT --module bipost.cli.train_bilevel \
  training_command="CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --include localhost:0,1,3  --master_port=$MASTER_PORT --module bipost.cli.train_bilevel \
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
   --evaluate True \
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
