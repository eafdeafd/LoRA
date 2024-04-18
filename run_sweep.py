import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy, count_parameters
from dataset_helpers import cola, sst, mrpc
import os
from functools import partial
import json
import torch
import torch.nn as nn
import evaluate
from lora import LoRALayer, LinearWithLoRA, LinearWithDoRA, ProgressiveLoRANet
import wandb
import numpy as np
# wandb.init(mode="disabled")
NUM_PREPROCESSING_WORKERS = 2

sweep_config = {
    'method': 'random'
}

parameters_dict = {
    'epochs': {
        'values': [80]
    },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 0.0000005,
        'max': 0.00008
    },
    'lora_r': {
        'values': [4, 8, 16, 32]
    },
    'lora_alpha': {
        'values': [4, 8, 16, 32]
    }
}

sweep_config["parameters"] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="lora")

def get_task_kwargs(task):
    match task:
        case 'qa':
            return {}
        case 'mnli':
            return {'num_labels': 3}
        case 'cola':
            return {'num_labels': 2}
        case 'sst2':
            return {'num_labels': 2}
        case 'mrpc':
            return {'num_labels': 2}
        case _:
            raise ValueError(f"Invalid Task: {task}")

def get_train_head(model, task):
    match task:
        case 'qa':
            return model.qa_ouputs
        case 'mnli':
            return model.classifier
        case 'cola':
            return model.classifier
        case 'sst2':
            return model.classifier
        case 'mrpc':
            return model.classifier
        case _:
            raise ValueError(f"Invalid Task: {task}")

def get_train(trainer, args):
    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config

            # TODO: get right labels per task
            # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
            task_kwargs = get_task_kwargs(args.task)
            # TODO: get right finetuning head for each task
            model_classes = {'qa': AutoModelForQuestionAnswering,
                            'mnli': AutoModelForSequenceClassification,
                            'cola': AutoModelForSequenceClassification,
                            'sst2': AutoModelForSequenceClassification,
                            'mrpc': AutoModelForSequenceClassification}
            
            model_class = model_classes[args.task]
            # Initialize the model and tokenizer from the specified pretrained model/checkpoint
            model = model_class.from_pretrained(args.model, **task_kwargs)
            #print(model)
            debug = True #TODO: add as argp param
            if args.lora != None:
                # freeze parameters
                for param in model.parameters():
                    param.requires_grad = False

                # Experimental setup in LoRA original paper: notice, only query and values are lora'd
                lora_r = config.lora_r
                lora_alpha = config.lora_alpha
                lora_dropout = 0.05
                lora_query = True
                lora_key = False
                lora_value = True
                lora_projection = False
                lora_mlp = False
                train_head = True

                if args.lora == 'lora':
                    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
                elif args.lora == 'dora':
                    assign_lora = partial(LinearWithDoRA, rank=lora_r, alpha=lora_alpha)
                elif args.lora == 'progressive':
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
                    for param in get_train_head(model, args.task).parameters():
                        param.requires_grad = True
            if debug:
                print(model)
                for name, param in model.named_parameters():
                    print(f"{name}: {param.requires_grad}")
                print("Total number of trainable parameters:", count_parameters(model))

            training_args = TrainingArguments(
                output_dir="sweeps",
                report_to="wandb",
                num_train_epochs=config.epochs,
                learning_rate=config.learning_rate,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=64,
                save_strategy="epoch",
                evaluation_strategy="epoch",
                logging_strategy="epoch",
                load_best_model_at_end=True,
                remove_unused_columns=False,
                save_total_limit=10,
            )

            full_trainer = trainer(model=model, args=training_args)
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
    argp.add_argument('--task', type=str, choices=['mnli', 'qa', 'cola', 'sst2', 'mrpc'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    argp.add_argument('--lora', type=str, default=None)
    
    training_args, args = argp.parse_args_into_dataclasses()

    #TODO: load GLUE properly, iterate over all tasks and evaluate over all tasks
    # dataset = datasets.load_dataset('nyu-mll/glue', 'mnli')
    dataset = datasets.load_dataset('nyu-mll/glue', args.task)
    
            
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    #TODO: make this work for all GLUE tasks
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'mnli':
        eval_split = 'validation_matched'
        prepare_train_dataset = prepare_eval_dataset = lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        # prepare_eval_dataset = prepare_dataset_nli
    elif args.task == 'cola':
        prepare_train_dataset = prepare_eval_dataset = lambda exs: cola.prepare_dataset(exs, tokenizer, args.max_length)
        eval_split = 'validation'
    elif args.task == 'sst2':
        prepare_train_dataset = prepare_eval_dataset = lambda exs: sst.prepare_dataset(exs, tokenizer, args.max_length)
        eval_split = 'validation'
    elif args.task == 'mrpc':
        prepare_train_dataset = prepare_eval_dataset = lambda exs: mrpc.prepare_dataset(exs, tokenizer, args.max_length)
        eval_split = 'validation'
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    dataset_id = None
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    #TODO: use same trainer as Lora Paper does 
    trainer_class = Trainer
    eval_kwargs = {}

    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    #TODO: custom metrics?
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = datasets.load_metric('squad')
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'mnli':
        compute_metrics = compute_accuracy
    elif args.task == 'cola':
        metric = evaluate.load('glue', 'cola') 
        def compute_metrics(eval_preds):
            predictions = np.argmax(eval_preds.predictions, axis=1)
            references = eval_preds.label_ids
            return metric.compute(predictions=predictions, references=references)
    elif args.task == 'sst2':
        compute_metrics = compute_accuracy
    elif args.task == 'mrpc':
        metric = evaluate.load('glue', 'mrpc') 
        def compute_metrics(eval_preds):
            predictions = np.argmax(eval_preds.predictions, axis=1)
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

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = partial(trainer_class, train_dataset=train_dataset_featurized, eval_dataset=eval_dataset_featurized, tokenizer=tokenizer, compute_metrics=compute_metrics_and_store_predictions)
    train = get_train(trainer, args)
    wandb.agent(sweep_id, train, count=8)

    # trainer = trainer_class(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset_featurized,
    #     eval_dataset=eval_dataset_featurized,
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics_and_store_predictions,
    # )

    # Train and/or evaluate
    # if training_args.do_train:
    #     trainer.train()
    #     trainer.save_model()
        

    # if training_args.do_eval:
    #     results = trainer.evaluate(**eval_kwargs)

    #     print('Evaluation results:')
    #     print(results)

    #     os.makedirs(training_args.output_dir, exist_ok=True)

    #     with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
    #         json.dump(results, f)

    #     with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
    #         if args.task == 'qa':
    #             predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
    #             for example in eval_dataset:
    #                 example_with_prediction = dict(example)
    #                 example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
    #                 f.write(json.dumps(example_with_prediction))
    #                 f.write('\n')
    #         else:
    #             for i, example in enumerate(eval_dataset):
    #                 example_with_prediction = dict(example)
    #                 example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
    #                 example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
    #                 f.write(json.dumps(example_with_prediction))
    #                 f.write('\n')


if __name__ == "__main__":
    main()
