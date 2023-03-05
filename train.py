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
from huggingface_hub import HfApi

from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_metric, load_from_disk
from params import default_cfg
#eval_batch
os.environ['TOKENIZERS_PARALLELISM']="false"

def parse_args():
    "Overriding default arguments for model"
    argparser = argparse.ArgumentParser(
        description="Process base parameters and hyperparameteres"
    )
    argparser.add_argument(
        "--model_name",
        type=str,
        default=default_cfg.MODEL_NAME,
        help="Model architecture to use"
    )
    argparser.add_argument(
        "--num_epochs",
        type=int,
        default=default_cfg.NUM_EPOCHS,
        help="number of training epochs"
    )
    argparser.add_argument(
        "--train_batch_size",
        type=int,
        default=default_cfg.TRAIN_BATCH_SIZE,
        help="Train batch size"
    )
    argparser.add_argument(
        "--eval_batch_size",
        type=int,
        default=default_cfg.VALID_DATA_ARTIFACT,
        help="Validation batch size"
    )
    argparser.add_argument(
        "--warmup_steps",
        type=int,
        default=default_cfg.WARMUP_STEPS,
        help="number of warmup steps"
    )
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=default_cfg.LEARNING_RATE,
        help="learning rate"
    )
    argparser.add_argument(
        "--fp16",
        type=str,
        default=default_cfg.FP16,
        help="Set to true to use half precision"
    )

def load_data(run,cfg):
    train_artifact = run.use_artifact(f"{cfg.TRAIN_DATA_ARTIFACT}:latest")
    train_artifact.download(root=cfg.TRAIN_DATA_FOLDER)
    train_dataset=load_from_disk(cfg.TRAIN_DATA_FOLDER)

    valid_dataset = run.use_artifact(f"{cfg.VALID_DATA_ARTIFACT}:latest")
    valid_dataset.download(root=cfg.VALID_DATA_FOLDER)
    valid_dataset = load_from_disk(cfg.VALID_DATA_FOLDER)

    drop_cols=[col for col in list(train_dataset.features) if col not in ['input_ids','attention_mask','rating']]

    train_dataset=train_dataset.remove_columns(drop_cols)
    valid_dataset=valid_dataset.remove_columns(drop_cols)

    train_dataset=train_dataset.rename_column('rating','labels')
    valid_dataset=valid_dataset.rename_column('rating','labels')

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    valid_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_dataset, valid_dataset

def compute_metrics(eval_pred):
    acc_metric=evaluate.load('accuracy')
    f1_metric=evaluate.load('f1')
    recall_metric=evaluate.load('recall')
    precision_metric=evaluate.load('precision')
    logits,labels=eval_pred
    predictions=np.argmax(logits,axis=-1)
    acc= acc_metric.compute(predictions=predictions,references=labels)
    f1= f1_metric.compute(predictions=predictions,references=labels,average='micro')
    recall= recall_metric.compute(predictions=predictions,references=labels,average='micro')
    precision= precision_metric.compute(predictions=predictions,references=labels,average='micro')

    return {
        "accuracy":acc['accuracy'],
        'f1':f1["f1"],
        'recall':recall['recall'],
        'precision':precision['precision']
    }

def train(cfg):
    os.environ['WANDB_DISABLE_SERVICE']='True'  

    with wandb.init(
        project=cfg.PROJECT_NAME, job_type=cfg.MODEL_TRAINING_JOB_TYPE,
        config=dict(cfg)
    ) as run:
        cfg=wandb.config
        

        training_args=TrainingArguments(
            output_dir=cfg.MODEL_DATA_FOLDER,
            num_train_epochs=cfg.NUM_EPOCHS,
            per_device_train_batch_size=cfg.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=cfg.VALID_BATCH_SIZE,
            warmup_steps=cfg.WARMUP_STEPS,
            fp16=cfg.FP16,
            learning_rate=float(cfg.LEARNING_RATE),
            logging_dir=f"{cfg.MODEL_DATA_FOLDER}/logs",
            logging_steps=2000,
            evaluation_strategy='steps',
            save_steps=2000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            report_to='wandb'
            #,
            #push_to_hub=cfg.PUSH_TO_HUB,
            #hub_strategy=cfg.HUB_STRATEGY,
            #hub_model_id=cfg.HUB_MODEL_ID
            )
        train_dataset, valid_dataset=load_data(run,cfg)
    
    
        num_classes = cfg.NUM_CLASSES
        tokenizer=AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL_NAME,
                                            num_labels=num_classes)

        trainer=Trainer(
            model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
            compute_metrics=compute_metrics
            )
        
        trainer.train()
        
        trainer.save_model("./tinybert-goodreads-model")
        model.push_to_hub('dhmeltzer/tinybert-goodreads-wandb')
        tokenizer.push_to_hub('dhmeltzer/tinybert-goodreads-wandb')

        hf_api=HfApi()
        user=hf_api.whoami()

        trained_model_art=wandb.Artifact('tinybert-goodreads-wandb',type=cfg.MODEL_TYPE)
        hub_id=f"{user['name']}/tinybert-goodreads-wandb"
        trained_model_art.metadata={"hub_id":hub_id}
        run.log_artifact(trained_model_art)

if __name__ == "__main__":
    
    #print(parse_args())
    #default_cfg.update(vars(parse_args()))
    train(default_cfg)
