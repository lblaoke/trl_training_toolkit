# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_processes

llama2-7b-ppo: https://github.com/jasonvanf/llama-trl

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
    ppo.py \
    --dataset_name preference_datasets/Llama-2-7b-sft_generated_10k/human \
    --model_name_or_path AmberYifan/llama2-7b-sft-ultrachat-safeRLHF \
    --reward_model_path lblaoke/llama2-7b-rm-human \
    --output_dir lblaoke/llama2-7b-ppo-human\
    --num_ppo_epochs 4 \
    --local_rollout_forward_batch_size 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.4e-5 \
    --logging_steps 20 \
    --response_length 32 \
    --use_peft \
    &> llama2-7b-ppo-human.log &

nohup accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
    ppo.py \
    --dataset_name preference_datasets/Llama-2-7b-sft_generated_10k/self \
    --model_name_or_path AmberYifan/llama2-7b-sft-ultrachat-safeRLHF \
    --reward_model_path lblaoke/llama2-7b-rm-self \
    --output_dir lblaoke/llama2-7b-ppo-self\
    --num_ppo_epochs 4 \
    --local_rollout_forward_batch_size 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.4e-5 \
    --logging_steps 20 \
    --response_length 32 \
    --use_peft \
    &> llama2-7b-ppo-self.log &
'''

import shutil

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from accelerate import PartialState, Accelerator
from peft import LoraConfig

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
)

from trl import (
    ModelConfig,
    PPOConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from ppo_trainer import MyPPOTrainer

if __name__ == '__main__':
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    training_args.report_to = []

    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    shutil.rmtree(training_args.output_dir + '_tmp', ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model_kwargs = dict(
        revision=model_args.model_revision,
        torch_dtype=torch.bfloat16,
        device_map=get_kbit_device_map() if bnb_config is not None else 'auto',
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side='left', trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    reward_model = AutoModelForSequenceClassification.from_pretrained(training_args.reward_model_path, trust_remote_code=True, num_labels=1, **model_kwargs)
    policy = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **model_kwargs)

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **model_kwargs)
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    data_files = {'train': 'train.jsonl', 'test': 'test.jsonl'}
    dataset = load_dataset(script_args.dataset_name, data_files=data_files)

    def prepare_dataset(dataset, tokenizer):
        '''pre-tokenize the dataset before training; only collate during training'''

        def tokenize(element):
            outputs = tokenizer(
                element['real']['role'=='user']['content'],
                padding=False,
            )
            return {'input_ids': outputs['input_ids']}

        return dataset.map(
            tokenize,
            batched=False,
            remove_columns=dataset.column_names,
            num_proc=32,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(dataset['train'], tokenizer)
        train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) <= 512, num_proc=32)

        eval_dataset = prepare_dataset(dataset['test'], tokenizer)
        eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) <= 512, num_proc=32)

    ################
    # Training
    ################
    trainer = MyPPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    if training_args.output_dir is not None:
        trainer.save_model(model_args.model_name_or_path, training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(training_args.output_dir)

    # trainer.generate_completions()
