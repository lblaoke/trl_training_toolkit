#!/bin/bash

accelerate launch --config_file configs/deepspeed_zero2.yaml \
    main_exp/reward_modeling.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_length 512 \
    --logging_steps 5 \
    --save_steps 549 \
    --eval_strategy no \
    --report_to wandb \
    --output_dir results/qwen2.5-1.5b-rm-lr1e-5 \
    --run_name qwen2.5-1.5b-rm-lr1e-5

sleep 10

accelerate launch --config_file configs/deepspeed_zero2.yaml \
    main_exp/reward_modeling.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_length 512 \
    --logging_steps 5 \
    --save_steps 549 \
    --eval_strategy no \
    --report_to wandb \
    --output_dir results/qwen2.5-1.5b-rm-lr1e-4 \
    --run_name qwen2.5-1.5b-rm-lr1e-4

sleep 10

accelerate launch --config_file configs/deepspeed_zero2.yaml \
    main_exp/reward_modeling.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --learning_rate 1e-3 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_length 512 \
    --logging_steps 5 \
    --save_steps 549 \
    --eval_strategy no \
    --report_to wandb \
    --output_dir results/qwen2.5-1.5b-rm-lr1e-3 \
    --run_name qwen2.5-1.5b-rm-lr1e-3
