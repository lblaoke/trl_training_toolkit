# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
llama-2-7b hyper-parameters: https://huggingface.co/vincentmin/llama-2-7b-reward-oasst1/blob/main/README.md

CUDA_VISIBLE_DEVICES=3,2,1,0 nohup python reward_modeling.py \
    --model_name_or_path AmberYifan/llama2-7b-sft-ultrachat-safeRLHF \
    --dataset_name preference_datasets/Llama-2-7b-sft_generated_10k/human \
    --output_dir lblaoke/llama2-7b-rm-human \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --gradient_checkpointing False \
    --optim adamw_torch_fused \
    --lr_scheduler_type linear \
    --learning_rate 2e-5 \
    --logging_steps 100 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --max_length 1024 \
    &> llama2-7b-rm-human.log &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python reward_modeling.py \
    --model_name_or_path AmberYifan/llama2-7b-sft-ultrachat-safeRLHF \
    --dataset_name preference_datasets/Llama-2-7b-sft_generated_10k/self \
    --output_dir lblaoke/llama2-7b-rm-self \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --gradient_checkpointing False \
    --optim adamw_torch_fused \
    --lr_scheduler_type linear \
    --learning_rate 2e-5 \
    --logging_steps 100 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --max_length 1024 \
    &> llama2-7b-rm-self.log &

CUDA_VISIBLE_DEVICES=7,6,5,4 nohup python reward_modeling.py \
    --model_name_or_path AmberYifan/llama2-7b-sft-ultrachat-safeRLHF \
    --dataset_name preference_datasets/Llama-2-7b-sft_generated_10k/self-human \
    --output_dir lblaoke/llama2-7b-rm-self-human \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --gradient_checkpointing False \
    --optim adamw_torch_fused \
    --lr_scheduler_type linear \
    --learning_rate 2e-5 \
    --logging_steps 100 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --max_length 1024 \
    &> llama2-7b-rm-self-human.log &


mistral-v0.1-7b hyper-parameters: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/bradley-terry-rm/mistral_7B_rm.py#L42

CUDA_VISIBLE_DEVICES=7,4,6,5 nohup python reward_modeling.py \
    --model_name_or_path AmberYifan/mistral-v0.1-7b-sft-ultrachat-safeRLHF \
    --dataset_name preference_datasets/Mistral-7B-v0.1-sft_generated_10k/human \
    --output_dir lblaoke/mistral-v0.1-7b-rm-human \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --num_train_epochs 1 \
    --gradient_checkpointing False \
    --optim adamw_torch_fused \
    --lr_scheduler_type cosine \
    --learning_rate 5e-6 \
    --weight_decay 0.001 \
    --warmup_ratio 0.03 \
    --eval_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --max_length 1024 \
    &> mistral-v0.1-7b-rm-human.log &

CUDA_VISIBLE_DEVICES=2,1,3,0 nohup python reward_modeling.py \
    --model_name_or_path AmberYifan/mistral-v0.1-7b-sft-ultrachat-safeRLHF \
    --dataset_name preference_datasets/Mistral-7B-v0.1-sft_generated_10k/self \
    --output_dir lblaoke/mistral-v0.1-7b-rm-self \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --num_train_epochs 1 \
    --gradient_checkpointing False \
    --optim adamw_torch_fused \
    --lr_scheduler_type cosine \
    --learning_rate 5e-6 \
    --weight_decay 0.001 \
    --warmup_ratio 0.03 \
    --eval_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --max_length 1024 \
    &> mistral-v0.1-7b-rm-self.log &

CUDA_VISIBLE_DEVICES=4,7,5,6 nohup python reward_modeling.py \
    --model_name_or_path AmberYifan/mistral-v0.1-7b-sft-ultrachat-safeRLHF \
    --dataset_name preference_datasets/Mistral-7B-v0.1-sft_generated_10k/self-human \
    --output_dir lblaoke/mistral-v0.1-7b-rm-self-human \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --num_train_epochs 1 \
    --gradient_checkpointing False \
    --optim adamw_torch_fused \
    --lr_scheduler_type cosine \
    --learning_rate 5e-6 \
    --weight_decay 0.001 \
    --warmup_ratio 0.03 \
    --eval_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --max_length 1024 \
    &> mistral-v0.1-7b-rm-self-human.log &


mistral-v0.3-7b hyper-parameters: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/bradley-terry-rm/mistral_7B_rm.py#L42

CUDA_VISIBLE_DEVICES=2,1,3,0 nohup python reward_modeling.py \
    --model_name_or_path AmberYifan/Mistral-7B-v0.3-sft-ultrachat-safeRLHF \
    --dataset_name preference_datasets/Mistral-7B-v0.3-sft-generated_10k/human \
    --output_dir lblaoke/mistral-v0.3-7b-rm-human \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --num_train_epochs 1 \
    --gradient_checkpointing False \
    --optim adamw_torch_fused \
    --lr_scheduler_type cosine \
    --learning_rate 5e-6 \
    --weight_decay 0.001 \
    --warmup_ratio 0.03 \
    --eval_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --max_length 1024 \
    &> mistral-v0.3-7b-rm-human.log &

CUDA_VISIBLE_DEVICES=4,7,5,6 nohup python reward_modeling.py \
    --model_name_or_path AmberYifan/Mistral-7B-v0.3-sft-ultrachat-safeRLHF \
    --dataset_name preference_datasets/Mistral-7B-v0.3-sft-generated_10k/self \
    --output_dir lblaoke/mistral-v0.3-7b-rm-self \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --num_train_epochs 1 \
    --gradient_checkpointing False \
    --optim adamw_torch_fused \
    --lr_scheduler_type cosine \
    --learning_rate 5e-6 \
    --weight_decay 0.001 \
    --warmup_ratio 0.03 \
    --eval_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --max_length 1024 \
    &> mistral-v0.3-7b-rm-self.log &

CUDA_VISIBLE_DEVICES=3,2,1,0 nohup python reward_modeling.py \
    --model_name_or_path AmberYifan/Mistral-7B-v0.3-sft-ultrachat-safeRLHF \
    --dataset_name preference_datasets/Mistral-7B-v0.3-sft-generated_10k/self-human \
    --output_dir lblaoke/mistral-v0.3-7b-rm-self-human \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --num_train_epochs 1 \
    --gradient_checkpointing False \
    --optim adamw_torch_fused \
    --lr_scheduler_type cosine \
    --learning_rate 5e-6 \
    --weight_decay 0.001 \
    --warmup_ratio 0.03 \
    --eval_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --max_length 1024 \
    &> mistral-v0.3-7b-rm-self-human.log &


Model Upload: huggingface-cli upload-large-folder --repo-type=model lblaoke/xxx ./xxx
"""

import warnings

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.report_to = []

    ################
    # Model & Tokenizer
    ################
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=1,
        trust_remote_code=True,
        **model_kwargs
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    ##############
    # Load dataset
    ##############
    data_files = {"train": "train.jsonl", "test": "test.jsonl"}
    dataset = load_dataset(script_args.dataset_name, data_files=data_files)

    dataset["train"] = dataset["train"].rename_column("real", "chosen")
    dataset["train"] = dataset["train"].rename_column("generated", "rejected")
    dataset["test"] = dataset["test"].rename_column("real", "chosen")
    dataset["test"] = dataset["test"].rename_column("generated", "rejected")

    ##########
    # Training
    ##########
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
    )
    trainer.train()

    # ############################
    # # Save model and push to Hub
    # ############################

    if training_args.eval_strategy != 'no':
        metrics = trainer.evaluate()
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    # Save and push to hub
    if training_args.output_dir is not None:
        trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(training_args.output_dir)
