#!/bin/bash

# # rewardbench evaluation
# CUDA_VISIBLE_DEVICES=7 rewardbench \
#     --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
#     --not_quantized \
#     --batch_size 8 \
#     --max_length 2048 \
#     --torch_dtype bfloat16 \
#     --attn_implementation flash_attention_2 \
#     --force_truncation

# # debug
# CUDA_VISIBLE_DEVICES=4,5,6,7 python train_rm.py \
#     --model_name_or_path facebook/opt-350m \
#     --dataset_name Dahoas/full-hh-rlhf \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --num_train_epochs 1 \
#     --gradient_checkpointing False \
#     --lr_scheduler_type cosine \
#     --learning_rate 5e-5 \
#     --weight_decay 0.1 \
#     --logging_steps 100 \
#     --eval_strategy steps \
#     --eval_steps 1000 \
#     --max_length 2048

# # opt-350m-reward-hh-rlhf
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python train_rm.py \
#     --model_name_or_path facebook/opt-350m \
#     --dataset_name Dahoas/full-hh-rlhf \
#     --output_dir lblaoke_local/opt-350m-reward-hh-rlhf \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --num_train_epochs 1 \
#     --gradient_checkpointing False \
#     --lr_scheduler_type cosine \
#     --learning_rate 5e-5 \
#     --weight_decay 0.1 \
#     --logging_steps 100 \
#     --eval_strategy steps \
#     --eval_steps 1000 \
#     --max_length 2048 \
#     &> opt-350m-reward-hh-rlhf.log &

# # qwen2-0.5b-reward-ultrafeedback
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python train_rm.py \
#     --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
#     --dataset_name trl-lib/ultrafeedback_binarized \
#     --output_dir lblaoke_local/qwen2-0.5b-reward-ultrafeedback \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --num_train_epochs 1 \
#     --gradient_checkpointing True \
#     --lr_scheduler_type linear \
#     --learning_rate 1e-5 \
#     --weight_decay 0.0 \
#     --logging_steps 100 \
#     --eval_strategy steps \
#     --eval_steps 1000 \
#     --max_length 2048 \
#     &> qwen2-0.5b-reward-ultrafeedback.log &

#######################################################################

# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#     ppo.py \
#     --dataset_name preference_datasets/Llama-2-7b-sft_generated_10k/human \
#     --model_name_or_path AmberYifan/llama2-7b-sft-ultrachat-safeRLHF \
#     --reward_model_path lblaoke/llama2-7b-rm-human \
#     --output_dir lblaoke/llama2-7b-ppo-human\
#     --num_ppo_epochs 4 \
#     --local_rollout_forward_batch_size 4 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 1.4e-5 \
#     --logging_steps 20 \
#     --response_length 32 \
#     --use_peft \
#     &> llama2-7b-ppo-human.log &


# nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#     ppo.py \
#     --dataset_name preference_datasets/Mistral-7B-v0.1-sft_generated_10k/human \
#     --model_name_or_path AmberYifan/mistral-v0.1-7b-sft-ultrachat-safeRLHF \
#     --reward_model_path lblaoke/mistral-v0.1-7b-rm-human \
#     --output_dir lblaoke/mistral-v0.1-7b-ppo-human \
#     --num_ppo_epochs 4 \
#     --local_rollout_forward_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 1.4e-5 \
#     --logging_steps 20 \
#     --response_length 24 \
#     --use_peft \
#     &> mistral-v0.1-7b-ppo-human.log &

# nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#     ppo.py \
#     --dataset_name preference_datasets/Llama-2-7b-sft_generated_10k/self \
#     --model_name_or_path AmberYifan/llama2-7b-sft-ultrachat-safeRLHF \
#     --reward_model_path lblaoke/llama2-7b-rm-self \
#     --output_dir lblaoke/llama2-7b-ppo-self\
#     --num_ppo_epochs 4 \
#     --local_rollout_forward_batch_size 4 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 1.4e-5 \
#     --logging_steps 20 \
#     --response_length 32 \
#     --use_peft \
#     &> llama2-7b-ppo-self.log &

# nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#     ppo.py \
#     --dataset_name preference_datasets/Mistral-7B-v0.1-sft_generated_10k/self \
#     --model_name_or_path AmberYifan/mistral-v0.1-7b-sft-ultrachat-safeRLHF \
#     --reward_model_path lblaoke/mistral-v0.1-7b-rm-self \
#     --output_dir lblaoke/mistral-v0.1-7b-ppo-self \
#     --num_ppo_epochs 4 \
#     --local_rollout_forward_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 1.4e-5 \
#     --logging_steps 20 \
#     --response_length 24 \
#     --use_peft \
#     &> mistral-v0.1-7b-ppo-self.log &

# nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#     ppo.py \
#     --dataset_name preference_datasets/Mistral-7B-v0.3-sft-generated_10k/self \
#     --model_name_or_path AmberYifan/mistral-7B-v0.3-sft-ultrachat-safeRLHF \
#     --reward_model_path lblaoke/mistral-v0.3-7b-rm-self \
#     --output_dir lblaoke/mistral-v0.3-7b-ppo-self \
#     --num_ppo_epochs 4 \
#     --local_rollout_forward_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 1.4e-5 \
#     --logging_steps 20 \
#     --response_length 24 \
#     --use_peft \
#     &> mistral-v0.3-7b-ppo-self.log &



# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#     ppo.py \
#     --dataset_name preference_datasets/Llama-3.1-8B-sft-generated_10k/human \
#     --model_name_or_path AmberYifan/Llama-3.1-8B-sft-ultrachat-safeRLHF \
#     --reward_model_path lblaoke/llama-3.1-8b-rm-human \
#     --output_dir lblaoke/llama-3.1-8b-ppo-human\
#     --num_ppo_epochs 4 \
#     --local_rollout_forward_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --learning_rate 3e-7 \
#     --lr_scheduler_type cosine \
#     --logging_steps 20 \
#     --response_length 32 \
#     --use_peft \
#     &> llama-3.1-8b-ppo-human.log &

# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#     ppo.py \
#     --dataset_name preference_datasets/Llama-3.1-8B-sft-generated_10k/self \
#     --model_name_or_path AmberYifan/Llama-3.1-8B-sft-ultrachat-safeRLHF \
#     --reward_model_path lblaoke/llama-3.1-8b-rm-self \
#     --output_dir lblaoke/llama-3.1-8b-ppo-self\
#     --num_ppo_epochs 4 \
#     --local_rollout_forward_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --learning_rate 3e-7 \
#     --lr_scheduler_type cosine \
#     --logging_steps 20 \
#     --response_length 32 \
#     --use_peft \
#     &> llama-3.1-8b-ppo-self.log &

# nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#     ppo.py \
#     --dataset_name preference_datasets/Llama-3.1-8B-sft-generated_10k/self-human \
#     --model_name_or_path AmberYifan/Llama-3.1-8B-sft-ultrachat-safeRLHF \
#     --reward_model_path lblaoke/llama-3.1-8b-rm-self-human \
#     --output_dir lblaoke/llama-3.1-8b-ppo-self-human\
#     --num_ppo_epochs 4 \
#     --local_rollout_forward_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --learning_rate 3e-7 \
#     --lr_scheduler_type cosine \
#     --logging_steps 20 \
#     --response_length 32 \
#     --use_peft \
#     &> llama-3.1-8b-ppo-self-human.log &


# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#     ppo.py \
#     --dataset_name preference_datasets/Qwen2.5-7B-sft_generated_10k/human \
#     --model_name_or_path AmberYifan/Qwen2.5-7B-sft-ultrachat-safeRLHF \
#     --reward_model_path lblaoke/qwen2.5-7b-rm-human \
#     --output_dir lblaoke/qwen2.5-7b-ppo-human \
#     --num_ppo_epochs 4 \
#     --local_rollout_forward_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --learning_rate 7e-6 \
#     --weight_decay 0.1 \
#     --lr_scheduler_type linear \
#     --logging_steps 20 \
#     --response_length 32 \
#     --use_peft \
#     &> qwen2.5-7b-ppo-human.log &

# CUDA_VISIBLE_DEVICES=5,6,7,3 nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#     ppo.py \
#     --dataset_name preference_datasets/Qwen2.5-7B-sft_generated_10k/self \
#     --model_name_or_path AmberYifan/Qwen2.5-7B-sft-ultrachat-safeRLHF \
#     --reward_model_path lblaoke/qwen2.5-7b-rm-self \
#     --output_dir lblaoke/qwen2.5-7b-ppo-self \
#     --num_ppo_epochs 4 \
#     --local_rollout_forward_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --learning_rate 7e-6 \
#     --weight_decay 0.1 \
#     --lr_scheduler_type linear \
#     --logging_steps 20 \
#     --response_length 32 \
#     --use_peft \
#     &> qwen2.5-7b-ppo-self.log &

# CUDA_VISIBLE_DEVICES=5,6,7,3 nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#     ppo.py \
#     --dataset_name preference_datasets/Qwen2.5-7B-sft_generated_10k/self-human \
#     --model_name_or_path AmberYifan/Qwen2.5-7B-sft-ultrachat-safeRLHF \
#     --reward_model_path lblaoke/mistral-v0.3-7b-rm-self-human \
#     --output_dir lblaoke/mistral-v0.3-7b-rm-self-human \
#     --num_ppo_epochs 4 \
#     --local_rollout_forward_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --learning_rate 7e-6 \
#     --weight_decay 0.1 \
#     --lr_scheduler_type linear \
#     --logging_steps 20 \
#     --response_length 32 \
#     --use_peft \
#     &> qwen2.5-7b-ppo-self-human.log &

# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#     ppo.py \
#     --dataset_name preference_datasets/Mistral-7B-v0.3-sft-generated_10k/human \
#     --model_name_or_path AmberYifan/mistral-7B-v0.3-sft-ultrachat-safeRLHF \
#     --reward_model_path lblaoke/mistral-v0.3-7b-rm-human \
#     --output_dir lblaoke/mistral-v0.3-7b-ppo-human \
#     --num_ppo_epochs 4 \
#     --local_rollout_forward_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 1.4e-5 \
#     --logging_steps 20 \
#     --response_length 24 \
#     --use_peft \
#     &> mistral-v0.3-7b-ppo-human.log &

# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#     ppo.py \
#     --dataset_name preference_datasets/Mistral-7B-v0.3-sft-generated_10k/self \
#     --model_name_or_path AmberYifan/mistral-7B-v0.3-sft-ultrachat-safeRLHF \
#     --reward_model_path lblaoke/mistral-v0.3-7b-rm-self \
#     --output_dir lblaoke/mistral-v0.3-7b-ppo-self \
#     --num_ppo_epochs 4 \
#     --local_rollout_forward_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 1.4e-5 \
#     --logging_steps 20 \
#     --response_length 24 \
#     --use_peft \
#     &> mistral-v0.3-7b-ppo-self.log &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
    ppo.py \
    --dataset_name preference_datasets/Mistral-7B-v0.3-sft-generated_10k/self-human \
    --model_name_or_path AmberYifan/mistral-7B-v0.3-sft-ultrachat-safeRLHF \
    --reward_model_path lblaoke/mistral-v0.3-7b-rm-self-human \
    --output_dir lblaoke/mistral-v0.3-7b-ppo-self-human \
    --num_ppo_epochs 4 \
    --local_rollout_forward_batch_size 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.4e-5 \
    --logging_steps 20 \
    --response_length 24 \
    --use_peft \
    &> mistral-v0.3-7b-ppo-self-human.log &

# huggingface-cli upload-large-folder --repo-type=model lblaoke/xxx ./xxx
