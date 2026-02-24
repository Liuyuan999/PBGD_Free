export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "MASTER_PORT=$MASTER_PORT"

# List of algorithms and their corresponding save paths
declare -A algorithms=(
  [VaFF]="/checkpoint/test_model_VaFF"
)

# Loop over each algorithm
for algo in "${!algorithms[@]}"; do
  save_path="${algorithms[$algo]}"

  export CUDA_VISIBLE_DEVICES="0"
  echo "Visible to this job: $CUDA_VISIBLE_DEVICES"

  training_command="CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --include localhost:3  --master_port=$MASTER_PORT --module bipost.cli.semantic_analysis \
   --pretrain EleutherAI/pythia-1b \
   --lora_alpha 16 \
   --lora_rank 16 \
   --target_modules query_key_value \
   --lambd 0.5 \
   --train_batch_size 1 \
   --micro_train_batch_size 1 \
   --learning_rate 1e-5 \
   --obj_1 DPO \
   --dataset_1 Dahoas/rm-hh-rlhf \
   --pref_prompt_key prompt \
   --pref_chosen_key chosen \
   --pref_rejected_key rejected \
   --max_samples_1 50 \
   --ref_model EleutherAI/pythia-1b \
   --obj_2 SFT \
   --dataset_2 Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --max_samples_2 50 \
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
