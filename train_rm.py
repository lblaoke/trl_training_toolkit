import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from datasets import load_dataset

from trl import (
    ModelConfig,
    RewardConfig,
    ScriptArguments,
    setup_chat_format,
)

from rm_trainer import RMTrainer

if __name__ == '__main__':
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
        attn_implementation='flash_attention_2',
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=1,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs
    )
    # Align padding tokens between tokenizer and model
    reward_model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        reward_model, tokenizer = setup_chat_format(reward_model, tokenizer)

    ##############
    # Load dataset
    ##############
    dataset = load_dataset(script_args.dataset_name)

    ##########
    # Training
    ##########
    trainer = RMTrainer(
        reward_model=reward_model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################

    if training_args.eval_strategy != 'no':
        metrics = trainer.evaluate()
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    # Save and push to hub
    if training_args.output_dir is not None:
        trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
