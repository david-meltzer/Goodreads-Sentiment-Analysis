import argparse, wandb, transformers, torch, os, sys
import numpy as np
import pandas as pd
import evaluate
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
from ml_collections import config_dict

os.environ['TOKENIZERS_PARALLELISM']="false"

default_cfg = config_dict.ConfigDict()

# WANDB BASE PARAMETERS
default_cfg.PROJECT_NAME = "mlops-course-assgn2"

#WANDB JOB TYPES
default_cfg.RAW_DATA_JOB_TYPE='fetch_raw_data'
default_cfg.DATA_PROCESSING_JOB_TYPE='process-data'
default_cfg.SPLIT_DATA_JOB_TYPE='split-data'
default_cfg.MODEL_TRAINING_JOB_TYPE='model-training'
default_cfg.MODEL_INFERENCE_JOB_TYPE='model-inference'

# WANDB ARTIFACT TYPES
default_cfg.DATASET_TYPE='dataset'
default_cfg.MODEL_TYPE='model'
default_cfg.MODEL_TRAINING_JOB_TYPE='model_training'

#WANDB ARTIFACT NAMES
default_cfg.RAW_DATA_ARTIFACT='goodreads_raw_data'
default_cfg.PROCESSED_DATA_ARTIFACT='processed_data'
default_cfg.TRAIN_DATA_ARTIFACT='goodreads_train_data'
default_cfg.VALID_DATA_ARTIFACT='goodreads_valid_data'
default_cfg.TEST_DATA_ARTIFACT='goodreads_test_data'

#DATA FOLDERS
default_cfg.DATA_FOLDER='data'
default_cfg.PROCESSED_DATA_FOLDER='data/processed_data'
default_cfg.FIGURE_FOLDER='figures'
default_cfg.TRAIN_DATA_FOLDER='data/train_data'
default_cfg.TEST_DATA_FOLDER='data/test_data'
default_cfg.VALID_DATA_FOLDER='data/valid_data'
default_cfg.MODEL_DATA_FOLDER = 'distilbert-goodreads-model'

# COLUMNS TO KEEP
default_cfg.FEATURE_COLUMN='review_text'
default_cfg.LABEL_COLUMN='rating'

# TRANSFORMERS PARAMETERS
default_cfg.MODEL_NAME = "prajjwal1/bert-tiny"
default_cfg.NUM_EPOCHS = 6
default_cfg.TRAIN_BATCH_SIZE = 32
default_cfg.VALID_BATCH_SIZE = 32
default_cfg.TEST_BATCH_SIZE = 32
default_cfg.WARMUP_STEPS = 1500
default_cfg.LEARNING_RATE = 5e-5
default_cfg.FP16 = True
default_cfg.NUM_CLASSES=6

# HUB PARAMETERS
default_cfg.PUSH_TO_HUB = True
default_cfg.HUB_MODEL_ID = "dhmeltzer/tinybert-goodreads-wandb"
default_cfg.HUB_STRATEGY = "every_save"


def parse_args():
    "Overriding default arguments for model"
    argparser = argparse.ArgumentParser(
        description="Process base parameters and hyperparameteres"
    )
    argparser.add_argument(
        --"model_name",
        type=str,
        default=default_cfg.model_name,
        help="Model architecture to use"
    )
    argparser.add_argument(
        "--num_epochs",
        type=int,
        default=default_cfg.num_epochs,
        help="number of training epochs"
    )
    argparser.add_argument(
        "--train_batch_size",
        type=int,
        default=default_cfg.train_batch_size,
        help="Train batch size"
    )
    argparser.add_argument(
        "--eval_batch_size",
        type=int,
        default=default_cfg.eval_batch_size,
        help="Validation batch size"
    )
    argparser.add_argument(
        "--warmup_steps",
        type=int,
        default=default_cfg.warmup_steps,
        help="number of warmup steps"
    )
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=default_cfg.learning_rate,
        help="learning rate"
    )
    argparser.add_argument(
        "--fp16",
        type=str,
        default=default_cfg.pg16,
        help="Set to true to use half precision"
    )

def compute_metrics(eval_pred):
    acc_metric=evaluate.load('accuracy')
    f1_metric=evaluate.load('f1')
    recall_metric=evaluate.load('recall')
    precision_metric=evaluate.load('precision')
    logits,labels=eval_pred
    predictions=np.argmax(logits,axis=-1)
    acc= acc_metric.compute(predictions=predictions,references=labels)
    f1= f1_metric.compute(predictions=predictions,references=labels)
    recall= recall_metric.compute(predictions=predictions,references=labels)
    precision= precision_metric.compute(predictions=predictions,references=labels)

    return {
        "accuracy":acc['accuracy'],
        'f1':f1["f1"],
        'recall':recall['recall'],
        'precision':precision['precision']
    }

def train(cfg):
    with wandb.init(
        project=cfg.PROJECT_NAME
    )