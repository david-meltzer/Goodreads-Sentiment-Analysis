import os
import sys
import argparse

import numpy as np
import pandas as pd
import wandb
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from huggingface_hub import HfApi
from transformers.integrations import WandbCallback
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_metric, load_from_disk
import evaluate

from params import default_cfg


def parse_args():
    "Overriding default arguments for model"
    argparser = argparse.ArgumentParser(
        description="Process base parameters and hyperparameteres"
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
    Loads and prepares the training and validation datasets.

    Parameters:
        run (wandb.Run): WandB run object for accessing artifacts.
        cfg (ConfigDict): Configuration object containing file paths.

    Returns:
        train_dataset (Dataset): Processed training dataset.
        valid_dataset (Dataset): Processed validation dataset.
    """

    # Download and load the training dataset
    train_artifact = run.use_artifact(f"{cfg.TRAIN_DATA_ARTIFACT}:latest")
    train_artifact.download(root=cfg.TRAIN_DATA_FOLDER)
    train_dataset = load_from_disk(cfg.TRAIN_DATA_FOLDER)

    # Download and load the validation dataset
    valid_dataset = run.use_artifact(f"{cfg.VALID_DATA_ARTIFACT}:latest")
    valid_dataset.download(root=cfg.VALID_DATA_FOLDER)
    valid_dataset = load_from_disk(cfg.VALID_DATA_FOLDER)

    # Remove unnecessary columns
    drop_cols = [col for col in list(train_dataset.features) if col not in ['input_ids', 'attention_mask', 'rating']]
    train_dataset = train_dataset.remove_columns(drop_cols)
    valid_dataset = valid_dataset.remove_columns(drop_cols)

    # Rename the 'rating' column to 'labels'
    train_dataset = train_dataset.rename_column('rating', 'labels')
    valid_dataset = valid_dataset.rename_column('rating', 'labels')

    # Set the format for torch compatibility
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    valid_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_dataset, valid_dataset

def compute_metrics(eval_pred):
    """
    Compute accuracy metric for evaluation predictions.

    Parameters:
        eval_pred (tuple): Tuple containing logits and labels.

    Returns:
        dict: Dictionary containing the computed accuracy.
    """
    # Load accuracy metric from evaluation module
    acc_metric = evaluate.load('accuracy')

    # Unpack logits and labels from eval_pred tuple
    logits, labels = eval_pred

    # Generate predictions by finding the index with maximum logit value
    predictions = np.argmax(logits, axis=-1)

    # Compute accuracy using the loaded metric
    acc = acc_metric.compute(predictions=predictions, references=labels)

    return {
        "accuracy": acc['accuracy']
    }

def train(cfg):
    """
    Train a model using the provided configuration.

    Parameters:
        cfg (ConfigDict): Configuration for the training job.

    Returns:
        None
    """
    # Disable the Weights and Biases service
    os.environ['WANDB_DISABLE_SERVICE'] = 'True'

    # Initialize Weights and Biases run
    with wandb.init(
        project=cfg.PROJECT_NAME, job_type=cfg.MODEL_TRAINING_JOB_TYPE,
        config=dict(cfg)
    ) as run:
        # Retrieve the configuration from Weights and Biases
        cfg = wandb.config

        # Define training arguments
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
            logging_steps=250,
            eval_steps=250,
            evaluation_strategy='steps',
            save_steps=250,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            report_to='wandb'
            #,
            #push_to_hub=cfg.PUSH_TO_HUB,
            #hub_strategy=cfg.HUB_STRATEGY,
            #hub_model_id=cfg.HUB_MODEL_ID
        )

        # Load training and validation datasets
        train_dataset, valid_dataset = load_data(run, cfg)

        # Load tokenizer, data collator, and model
        num_classes = cfg.NUM_CLASSES
        tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL_NAME, num_labels=num_classes)

        # Initialize trainer
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

        # Train the model
        trainer.train()

        # Save the trained model and push to Hugging Face Model Hub
        trainer.save_model(cfg.MODEL_DATA_FOLDER)
        model.push_to_hub(cfg.HUB_MODEL_ID)
        tokenizer.push_to_hub(cfg.HUB_MODEL_ID)

        # Access the Hugging Face API
        hf_api = HfApi()
        user = hf_api.whoami()

        # Log the trained model as an artifact
        trained_model_art = wandb.Artifact(cfg.MODEL_DATA_FOLDER, type=cfg.MODEL_TYPE)
        hub_id = cfg.HUB_MODEL_ID
        trained_model_art.metadata = {"hub_id": hub_id}
        run.log_artifact(trained_model_art)

if __name__ == "__main__":
    
    # Update default configuration variables. 
    default_cfg.update(vars(parse_args()))
    train(default_cfg)
