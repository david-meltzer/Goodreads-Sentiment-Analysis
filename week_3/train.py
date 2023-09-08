# Standard Library Imports
import os
import sys

import argparse
import evaluate
import wandb
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_metric, load_from_disk

from params import default_cfg

# Set environment variable for parallelism
os.environ['TOKENIZERS_PARALLELISM'] = "false"

def parse_args():
    "Overriding default arguments for model"
    argparser = argparse.ArgumentParser(
        description="Process base parameters and hyperparameters"
    )
    argparser.add_argument(
        "--MODEL_NAME",
        type=str,
        default=default_cfg.MODEL_NAME,
        help="Model architecture to use"
    )
    argparser.add_argument(
        "--NUM_EPOCHS",
        type=int,
        default=default_cfg.NUM_EPOCHS,
        help="number of training epochs"
    )
    argparser.add_argument(
        "--TRAIN_BATCH_SIZE",
        type=int,
        default=default_cfg.TRAIN_BATCH_SIZE,
        help="Train batch size"
    )
    argparser.add_argument(
        "--VALID_BATCH_SIZE",
        type=int,
        default=default_cfg.VALID_BATCH_SIZE,
        help="Validation batch size"
    )
    argparser.add_argument(
        "--WARMUP_STEPS",
        type=int,
        default=default_cfg.WARMUP_STEPS,
        help="number of warmup steps"
    )
    argparser.add_argument(
        "--LEARNING_RATE",
        type=float,
        default=default_cfg.LEARNING_RATE,
        help="learning rate"
    )
    argparser.add_argument(
        "--FP16",
        type=int,
        default=int(default_cfg.FP16),
        help="Set to true to use half precision"
    )

    argparser.add_argument(
        "--GRADIENT_ACCUMULATION_STEPS",
        type=int,
        default=default_cfg.GRADIENT_ACCUMULATION_STEPS,
        help="Set to true to use half precision"
    )

    return argparser.parse_args()

def load_data(run, cfg):
    """
    Load training and validation datasets from Wandb Artifacts.

    Args:
        run (wandb.Run): Wandb run object.
        cfg (Config): Configuration object containing file paths and settings.

    Returns:
        train_dataset (datasets.Dataset): Training dataset.
        valid_dataset (datasets.Dataset): Validation dataset.
    """
    # Load the latest training artifact from Wandb
    train_artifact = run.use_artifact(f"{cfg.TRAIN_DATA_ARTIFACT}:latest")
    
    # Download the training data to the specified folder
    train_artifact.download(root=cfg.TRAIN_DATA_FOLDER)
    
    # Load the training dataset from disk
    train_dataset = load_from_disk(cfg.TRAIN_DATA_FOLDER)

    # Load the latest validation artifact from Wandb
    valid_dataset = run.use_artifact(f"{cfg.VALID_DATA_ARTIFACT}:latest")
    
    # Download the validation data to the specified folder
    valid_dataset.download(root=cfg.VALID_DATA_FOLDER)
    
    # Load the validation dataset from disk
    valid_dataset = load_from_disk(cfg.VALID_DATA_FOLDER)

    # Identify columns to be dropped from the datasets
    drop_cols = [col for col in list(train_dataset.features) if col not in ['input_ids', 'attention_mask', 'rating']]

    # Remove unnecessary columns from both datasets
    train_dataset = train_dataset.remove_columns(drop_cols)
    valid_dataset = valid_dataset.remove_columns(drop_cols)

    # Rename 'rating' column to 'labels' for consistency
    train_dataset = train_dataset.rename_column('rating', 'labels')
    valid_dataset = valid_dataset.rename_column('rating', 'labels')

    # Set the format of the datasets to 'torch' for compatibility with PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    valid_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_dataset, valid_dataset

def compute_metrics(eval_pred):
    """
    Compute various classification metrics based on model predictions.

    Args:
        eval_pred (tuple): Tuple containing logits and labels.

    Returns:
        dict: Dictionary containing computed metrics (accuracy, f1-score, recall, precision).
    """
    # Load evaluation metrics from the 'evaluate' module
    acc_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')
    recall_metric = evaluate.load('recall')
    precision_metric = evaluate.load('precision')
    
    # Unpack logits and labels from eval_pred
    logits, labels = eval_pred
    
    # Compute predictions based on logits
    predictions = np.argmax(logits, axis=-1)
    
    # Compute accuracy
    acc = acc_metric.compute(predictions=predictions, references=labels)
    
    # Compute f1-score
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='micro')
    
    # Compute recall
    recall = recall_metric.compute(predictions=predictions, references=labels, average='micro')
    
    # Compute precision
    precision = precision_metric.compute(predictions=predictions, references=labels, average='micro')

    return {
        "accuracy": acc['accuracy'],
        'f1': f1["f1"],
        'recall': recall['recall'],
        'precision': precision['precision']
    }

def train(cfg):
    """
    Train a sequence classification model.

    Args:
        cfg (Config): Configuration object containing model training settings.

    Returns:
        None
    """
    # Disable Wandb services (useful in some environments)
    os.environ['WANDB_DISABLE_SERVICE'] = 'True'  

    # Initialize a new run with Wandb
    with wandb.init(
        project=cfg.PROJECT_NAME, job_type=cfg.MODEL_TRAINING_JOB_TYPE,
        config=dict(cfg)
    ) as run:
        cfg = wandb.config

        # Set up training arguments for the Trainer
        training_args = TrainingArguments(
            output_dir=cfg.MODEL_DATA_FOLDER,
            num_train_epochs=cfg.NUM_EPOCHS,
            per_device_train_batch_size=cfg.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=cfg.VALID_BATCH_SIZE,
            warmup_steps=cfg.WARMUP_STEPS,
            gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
            fp16=bool(cfg.FP16),
            learning_rate=float(cfg.LEARNING_RATE),
            logging_dir=f"{cfg.MODEL_DATA_FOLDER}/logs",
            logging_steps=500,
            eval_steps=500,
            evaluation_strategy='steps',
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            report_to='wandb'
        )

        # Load and prepare training and validation datasets
        train_dataset, valid_dataset = load_data(run, cfg)

        # Initialize tokenizer and data collator
        num_classes = cfg.NUM_CLASSES
        tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Initialize the model for sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.MODEL_NAME, num_labels=num_classes)

        # Initialize the Trainer with the defined settings
        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            compute_metrics=compute_metrics
        )

        # Start the training process
        trainer.train()

        # Save the trained model to the specified directory
        trainer.save_model(cfg.MODEL_DATA_FOLDER)

        # Push both the model and tokenizer to the Hugging Face Model Hub
        model.push_to_hub(cfg.HUB_MODEL_ID)
        tokenizer.push_to_hub(cfg.HUB_MODEL_ID)

        # Get the current user from the Hugging Face API
        hf_api = HfApi()
        user = hf_api.whoami()

        # Create a new Wandb Artifact for the trained model
        trained_model_art = wandb.Artifact(cfg.MODEL_DATA_FOLDER, type=cfg.MODEL_TYPE)
        hub_id = cfg.HUB_MODEL_ID
        trained_model_art.metadata = {"hub_id": hub_id}

        # Log the trained model artifact to Wandb
        run.log_artifact(trained_model_art)


if __name__ == "__main__":
    
    # Update the default configuration with parsed command-line arguments
    default_cfg.update(vars(parse_args()))
    
    # Call the 'train' function with the updated configuration
    train(default_cfg)