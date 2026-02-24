set -x

read -r -d '' training_commands <<EOF
bipost.cli.evaluate_bilevel \
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
   
   --algorithm obj1 \
   --output_layer_name embed_out \
   --gam 10\
   --inner_stepsize 5 \


   --save_path /data/liuyuan/bipost/checkpoint/test_model_biobjective \
   --max_epochs 3 \
   --zero_stage 2 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --bf16 \
   --load_checkpoint \
   --gradient_checkpointing 

EOF
    # --use_wandb [WANDB_TOKENS]

if [[ ${1} != "slurm" ]]; then
    deepspeed --num_gpus 2 --module $training_commands
fi
