import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser, BitsAndBytesConfig
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy, count_parameters
from dataset_helpers import cola, sst, mrpc, qqp, rte, stsb
import os
from functools import partial
import json
import torch
import torch.nn as nn
import evaluate
from lora import LoRALayer, LinearWithLoRA, LinearWithDoRA, ProgressiveLoRANet, LoRAFALayer, VeRA, FiLMA
import wandb
import numpy as np
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from run import get_task_kwargs, get_train_head
# wandb.init(mode="disabled")
NUM_PREPROCESSING_WORKERS = 2

sweep_config = {
    'method': 'grid',
    'name': 'task_lora_sweep',
}

parameters_dict = {
    'task': {
        # 'values': ['cola', 'mrpc', 'qqp', 'rte', 'sst', 'stsb']'
        'values': ['mrpc', 'qqp', 'rte', 'sst', 'stsb']
    },
    'lora': {
        'values': ['qlora']
    }
}

def get_learning_rate(lora):
    match lora:
        case 'vera':
            return 0.001
        case 'qlora':
            return 0.001
        case _:
            return 0.00002


sweep_config["parameters"] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="lora")

def get_train(args):
    def train(config=None):
        with wandb.init(config=config) as wandb_run:
            config = wandb.config

            wandb_run.name = f"Sweep_{config.task}_{config.lora if config.lora is not None else 'FT'}"
            # TODO: get right labels per task
            # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
            task_kwargs = get_task_kwargs(config.task)
            # TODO: get right finetuning head for each task
            model_classes = {'qa': AutoModelForQuestionAnswering,
                            'mnli': AutoModelForSequenceClassification,
                            'cola': AutoModelForSequenceClassification,
                            'sst2': AutoModelForSequenceClassification,
                            'mrpc': AutoModelForSequenceClassification,
                            'qqp':AutoModelForSequenceClassification,
                            'rte':AutoModelForSequenceClassification,
                            'stsb':AutoModelForSequenceClassification}
                    
            model_class = model_classes[config.task]
            # Initialize the model and tokenizer from the specified pretrained model/checkpoint
            bnb_config = None
            if config.lora == 'qlora':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    load_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            model = model_class.from_pretrained(args.model, quantization_config=bnb_config, **task_kwargs)

            if config.lora == 'qlora':
                model = prepare_model_for_kbit_training(model)


            #print(model)
            debug = True #TODO: add as argp param
            if config.lora != None and config.lora != 'qlora':
                # freeze parameters
                for param in model.parameters():
                    param.requires_grad = False

                # Experimental setup in LoRA original paper: notice, only query and values are lora'd
                lora_r = 4
                lora_alpha = 4
                lora_dropout = 0.05
                lora_query = True
                lora_key = False
                lora_value = True
                lora_projection = False
                lora_mlp = False
                train_head = True

                if config.lora == 'lora':
                    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, lora=LoRALayer)
                elif config.lora == 'dora':
                    assign_lora = partial(LinearWithDoRA, rank=lora_r, alpha=lora_alpha)
                elif config.lora == 'lorafa':
                    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, lora=LoRAFALayer)
                elif config.lora == 'vera':
                    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, lora=VeRA)
                elif config.lora == 'qlora':
                    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, lora=LoRALayer)
                elif config.lora == 'progressive':
                    # TODO: add progressive net lora
                    raise NotImplementedError
                else:
                    raise ValueError("no lora type found!")
                
                # WARNING: this will break for other models, it is hardcoded.
                for layer in model.roberta.encoder.layer:
                    if lora_query:
                        layer.attention.self.query = assign_lora(layer.attention.self.query)
                    if lora_key:
                        layer.attention.self.key = assign_lora(layer.attention.self.key)
                    if lora_value:
                        layer.attention.self.value = assign_lora(layer.attention.self.value)
                    if lora_projection:
                        layer.attention.output.dense = assign_lora(layer.attention.output.dense)
                    if lora_mlp:
                        layer.intermediate.dense = assign_lora(layer.intermediate.dense)
                        layer.output.dense = assign_lora(layer.output.dense)
                if train_head:
                    for param in get_train_head(model, config.task).parameters():
                        param.requires_grad = True
            
            if config.lora == 'qlora':
                lora_config = LoraConfig(
                    r=4,
                    lora_alpha=4,
                    target_modules=['query', 'value'],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )

                model = get_peft_model(model, lora_config)

            if debug:
                print(model)
                for name, param in model.named_parameters():
                    print(f"{name}: {param.requires_grad}")
                print("Total number of trainable parameters:", count_parameters(model))

            training_args = TrainingArguments(
                output_dir='task_lora_sweep',
                report_to="wandb",
                num_train_epochs=40,
                learning_rate=get_learning_rate(config.lora),
                per_device_train_batch_size=32,
                per_device_eval_batch_size=64,
                save_strategy="epoch",
                evaluation_strategy="epoch",
                logging_strategy="epoch",
                load_best_model_at_end=True,
                remove_unused_columns=False,
                save_total_limit=10,
            )

            dataset = datasets.load_dataset('nyu-mll/glue', config.task)
            
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

            if config.task == 'mnli':
                eval_split = 'validation_matched'
                prepare_train_dataset = prepare_eval_dataset = lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
            elif config.task == 'cola':
                prepare_train_dataset = prepare_eval_dataset = lambda exs: cola.prepare_dataset(exs, tokenizer, args.max_length)
                eval_split = 'validation'
            elif config.task == 'sst2':
                prepare_train_dataset = prepare_eval_dataset = lambda exs: sst.prepare_dataset(exs, tokenizer, args.max_length)
                eval_split = 'validation'
            elif config.task == 'mrpc':
                prepare_train_dataset = prepare_eval_dataset = lambda exs: mrpc.prepare_dataset(exs, tokenizer, args.max_length)
                eval_split = 'validation'
            elif config.task == 'qqp':
                prepare_train_dataset = prepare_eval_dataset = lambda exs: qqp.prepare_dataset(exs, tokenizer, args.max_length)
                eval_split = 'validation'
            elif config.task == 'rte':
                prepare_train_dataset = prepare_eval_dataset = lambda exs: rte.prepare_dataset(exs, tokenizer, args.max_length)
                eval_split = 'validation'
            elif config.task == 'stsb':
                prepare_train_dataset = prepare_eval_dataset = lambda exs: stsb.prepare_dataset(exs, tokenizer, args.max_length)
                eval_split = 'validation'
            else:
                raise ValueError('Unrecognized task name: {}'.format(config.task))

            train_dataset = None
            eval_dataset = None
            train_dataset_featurized = None
            eval_dataset_featurized = None
            train_dataset = dataset['train']
            if args.max_train_samples:
                train_dataset = train_dataset.select(range(args.max_train_samples))
            train_dataset_featurized = train_dataset.map(
                prepare_train_dataset,
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=train_dataset.column_names
            )

            eval_dataset = dataset[eval_split]
            if args.max_eval_samples:
                eval_dataset = eval_dataset.select(range(args.max_eval_samples))
            eval_dataset_featurized = eval_dataset.map(
                prepare_eval_dataset,
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=eval_dataset.column_names
            )

            eval_kwargs = {}

            # If you want to use custom metrics, you should define your own "compute_metrics" function.
            # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
            #TODO: custom metrics?
            compute_metrics = None
            if config.task == 'mnli':
                compute_metrics = compute_accuracy
            elif config.task == 'cola':
                metric = evaluate.load('glue', 'cola')
                def compute_metrics(eval_preds):
                    predictions = np.argmax(eval_preds.predictions, axis=1)
                    references = eval_preds.label_ids
                    return metric.compute(predictions=predictions, references=references)
            elif config.task == 'sst2':
                compute_metrics = compute_accuracy
            elif config.task == 'mrpc':
                metric = evaluate.load('glue', 'mrpc') 
                def compute_metrics(eval_preds):
                    predictions = np.argmax(eval_preds.predictions, axis=1)
                    references = eval_preds.label_ids
                    return metric.compute(predictions=predictions, references=references)
            elif config.task == 'qqp':
                compute_metrics = compute_accuracy # could also be F1
            elif config.task == 'rte':
                compute_metrics = compute_accuracy
            elif config.task == 'stsb':
                metric = evaluate.load('glue', 'stsb') 
                def compute_metrics(eval_preds):
                    predictions = eval_preds.predictions.flatten()
                    references = eval_preds.label_ids
                    return metric.compute(predictions=predictions, references=references)
                
            #compute_metrics = lambda eval_preds: metric.compute(predictions=np.argmax(eval_preds.predictions, axis=1), references=eval_preds.label_ids)
            # This function wraps the compute_metrics function, storing the model's predictions
            # so that they can be dumped along with the computed metrics
            eval_predictions = None
            def compute_metrics_and_store_predictions(eval_preds):
                nonlocal eval_predictions
                eval_predictions = eval_preds
                outs = compute_metrics(eval_preds)
                print(outs)
                return outs

            full_trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset_featurized, eval_dataset=eval_dataset_featurized, tokenizer=tokenizer, compute_metrics=compute_metrics_and_store_predictions)
            full_trainer.train()



    return train

def main():

    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    # python run.py --dataset nyu-mll/glue --task qa --lora lora --output_dir output
    #python run.py --dataset nyu-mll/glue --task cola --lora lora --output_dir output --do_train --save_steps 500 --save_total_limit 1 --load_best_model_at_end --logging_steps 500 --per_device_train_batch_size 32 --evaluation_strategy steps --per_device_eval_batch_size 32 --num_train_epochs 80 --learning_rate .0004 --warmup_ratio .06 --max_length 128
    argp.add_argument('--model', type=str,
                      default='FacebookAI/roberta-base',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    
    training_args, args = argp.parse_args_into_dataclasses()


    train = get_train(args)
    wandb.agent(sweep_id, train)


if __name__ == "__main__":
    main()
