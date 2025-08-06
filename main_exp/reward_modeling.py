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

from accelerate import PartialState
import wandb

import my_trainer

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    if PartialState().is_local_main_process:
        wandb.init(project="rm_generalization", name=training_args.run_name)

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=1, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
            UserWarning,
        )

    ##############
    # Load dataset
    ##############
    with PartialState().local_main_process_first():
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ##########
    # Training
    ##########
    trainer = my_trainer.MyRewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

"""
llama-2-7b hyper-parameters: https://huggingface.co/vincentmin/llama-2-7b-reward-oasst1/blob/main/README.md
mistral-v0.1-7b hyper-parameters: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/bradley-terry-rm/mistral_7B_rm.py#L42
mistral-v0.3-7b hyper-parameters: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/bradley-terry-rm/mistral_7B_rm.py#L42
llama-3.1-8b hyper-parameters: https://huggingface.co/allenai/llama-3.1-tulu-2-8b-uf-mean-rm
qwen2.5-7b hyper-parameters: deepseek

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
    --logging_steps 1 \
    --save_steps 100 \
    --eval_strategy no \
    --report_to wandb \
    --output_dir results/qwen2.5-1.5b-rm \
    --run_name debug

Model Upload: huggingface-cli upload-large-folder --repo-type=model lblaoke/xxx ./xxx
"""
