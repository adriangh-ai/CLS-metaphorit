"""
Training and evaluation loop for the project of Automatic Metaphor Detection

"""
import evaluate

import pandas as pd
import torch
import random

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments
from transformers import logging

from data_preprocess import data_preprocess
from datasets import DatasetDict
from tqdm import tqdm
from itertools import product
from typing import Union


def load_model(checkpoint_path: str) -> Union[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Loads a pre-trained model and tokenizer.

    Args:
        checkpoint_path (str): The path to the pre-trained checkpoint.

    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: A tuple with the loaded model and tokenizer objects.
    """
    logging.set_verbosity_error()
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    logging.set_verbosity_info()
    # Return the loaded model and tokenizer as a tuple
    return model, tokenizer


def compute_metrics(model_output: tuple) -> dict:
    """
    Computes the metrics of the model.

    Args:
        model_output (tuple): Tuple with the predictions and the references.

    Returns:
        dict: Dictionary with the metrics.
    """
    # Load the metrics
    metric_f1 = evaluate.load('f1')
    metric_accuracy = evaluate.load('accuracy')
    metric_precision = evaluate.load('precision')
    metric_recall = evaluate.load('recall')

    # Unpack the model output into predictions and references
    predictions, references = model_output
    # Compute the metrics and return a dictionary with the concatenation of the results
    results = {
        **metric_f1.compute(predictions=predictions, references=references, average='macro'),
        **metric_accuracy.compute(predictions=predictions, references=references),
        **metric_precision.compute(predictions=predictions, references=references, average='macro'),
        **metric_recall.compute(predictions=predictions, references=references, average='macro'),
    }
    return results


def preprocess_logits_for_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Preprocesses logits for the metrics calculation. This step is needed to limit the total size of the
    Tensors and reduce the memory profile.

    Args:
        predictions (torch.Tensor): Model predictions.
        labels (torch.Tensor): Labels.

    Returns:
        torch.Tensor: Preprocessed logits.
    """
    # Find the maximum index of the predictions along the last axis (axis=-1)
    predicted_labels = torch.argmax(predictions, axis=-1)
    return predicted_labels


def total_steps_for_warmup(dataset, num_epochs: int,
                             num_batch_size: int) -> int:
    """Calculate the total number of steps for the warmup process

    Parameters
    ----------
    dataset: DatasetDict
        Dictionary with the splitted corpora.
    num_epochs : int
        Number of epochs.
    num_batch_size : int
        Batch size.

    Returns
    -------
    int
        Total number of warmup steps.
    """
    num_of_warmup_steps = int(((len(dataset)*num_epochs)//num_batch_size)*0.1)
    return num_of_warmup_steps


def hp_search(model_name: str, train: DatasetDict, dev: DatasetDict, search_space: dict) -> dict:
    """
    Performs a hyperparameter search on a model using a given training and development dataset.

    Args:
        model_name (str): The name of the model to train.
        train (DatasetDict): The training dataset.
        dev (DatasetDict): The development dataset used for validation during training.
        search_space (dict): The hyperparameters to search over, in the form {'lr': [], 'epochs': [], 'batchsize': []}.

    Returns:
        dict: The best hyperparameters found, in the form {'lr': float, 'epochs': int, 'batchsize': int}.
    """
    max_score = float('-inf')
    best_hyperparameter = {'lr':0, 'epochs':0, 'batchsize':0}
    # Grid search
    for lr, epoch, batchsize in product(*search_space.values()):
        print(f'HP Search:\nLearning rate: {lr}\nEpoch: {epoch}\nBatchsize: {batchsize}')
        trial_hp = {'lr':lr, 'epochs':epoch, 'batchsize':batchsize}
        trainer = train_model(model_name, train, dev, trial_hp )
        results = trainer.evaluate()['eval_f1']
        if results > max_score:
            max_score = results
            best_hyperparameter = {'lr' : lr, 'epochs' : epoch, 'batchsize' : batchsize}
            print(f'Best score: {max_score}')
            print(f'New best hyperparameters: {best_hyperparameter}')
    return best_hyperparameter


def train_model(model_name: str, train: DatasetDict, dev: DatasetDict, hp: dict) -> Trainer:
    """
    Trains a model on a given training and development dataset with specified hyperparameters.

    Args:
        model_name (str): The name of the model to train.
        train (DatasetDict): The training dataset.
        dev (DatasetDict): The development dataset used for validation during training.
        hp (dict): The hyperparameters to use, in the form {'lr': float, 'epochs': int, 'batchsize': int}.

    Returns:
        Trainer: The trained model.
    """
    model, tokenizer = load_model(model_name)
    
    trainer_params = {
        'lr_scheduler_type' : 'linear',
        'optim': 'adamw_torch',
        'output_dir' : 'model/' + model_name.split('/')[-1],
        'overwrite_output_dir': True,
        'evaluation_strategy' : 'epoch',
        'save_strategy' : 'no',
        'save_total_limit': 1,
        'save_total_limit' : 0,
        'learning_rate' : hp['lr'],
        'num_train_epochs' : hp['epochs'],
        'per_device_train_batch_size': hp['batchsize'],
        'per_device_eval_batch_size': hp['batchsize'],
        'warmup_steps': total_steps_for_warmup(train ,hp['epochs'],  hp['batchsize']),
        'push_to_hub' : False,
        'fp16': True,
        'load_best_model_at_end':False,
        'metric_for_best_model' : 'f1',
        'seed': random.randint(1,99999),
    }
    
    training_args = TrainingArguments(**trainer_params)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=DataCollatorWithPadding(tokenizer))
    
    # Start training
    trainer.train(resume_from_checkpoint=False)
    return trainer


def train_evaluate_model(model_name: str, train: DatasetDict, dev: DatasetDict, test: DatasetDict, hp: dict) -> dict:
    """
    Trains and evaluates a model on a given training, development, and test dataset with specified hyperparameters.

    Args:
        model_name (str): The name of the model to train.
        train (DatasetDict): The training dataset.
        dev (DatasetDict): The development dataset used for validation during training.
        test (DatasetDict): The test dataset used for final evaluation.
        hp (dict): The hyperparameters to use, in the form {'lr': float, 'epochs': int, 'batchsize': int}.

    Returns:
        dict: The results of the evaluation.
    """
    trainer = trainer = train_model(model_name, train, dev, hp)
    trainer.eval_dataset = test
    result = trainer.evaluate()
    return result


def main():
    # Hyper-parameter search space
    hp_search_space = {
        'lr': [2e-5, 3e-5, 5e-5],
        'epochs': [2,3,4],
        'batchsize': [8,16,32]
    }
    model_name = 'bert-base-multilingual-cased'

    # Data loading and pre-processing
    dataframe = pd.read_csv('dataset/binary_undersampling_filtered_ds_remove_discrepancies.csv')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = data_preprocess.corpus_to_hf_dataset(dataframe, tokenizer)

    hyperparameters = hp_search(model_name, dataset['train'], dataset['validation'], hp_search_space)
    
    # Traning
    print(f'Starting training with: {hyperparameters}')
    results = [train_evaluate_model(model_name, dataset['train'], 
                                    dataset['validation'], 
                                    dataset['test'], 
                                    hyperparameters)
                                    for _ in range(10)]
    
    # Result formatting and file save
    print(results)
    results_df = pd.DataFrame(results)
    print(results_df.mean().T)
    print(results_df.std().T)
    print(hyperparameters)

if __name__=='__main__':
    main()