import wandb
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
from params import default_cfg
from datasets import load_from_disk, load_dataset, Dataset, load_metric


import imblearn
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import GroupShuffleSplit


def log_raw_data(cfg):
    """
    Downloads Goodreads dataset from Kaggle and logs it as an artifact.

    Parameters
    ----------
        cfg (ConfigDict): ConfigDict object containing configuration for experiment.
    
    Returns:
    --------
        None

    """
    with wandb.init(
        project=cfg.PROJECT_NAME,
        job_type=cfg.RAW_DATA_JOB_TYPE,
        config=dict(cfg)
    ) as run:
        cfg=wandb.config
        os.makedirs('./data',exist_ok=True)
        os.makedirs('./data/raw_data',exist_ok=True)
        os.system('kaggle competitions download -c goodreads-books-reviews-290312')
        os.system('unzip -d ./data/raw_data goodreads-books-reviews-290312.zip')

        train_path='./data/raw_data/goodreads_train.csv'

        raw_data_art=wandb.Artifact(cfg.RAW_DATA_ARTIFACT,type=cfg.DATASET_TYPE)
        raw_data_art.add_file(train_path)
        run.log_artifact(raw_data_art)

def downsample_and_log(cfg):
    """
    Downsamples the classses and tokenizes the data.
    
    Parameters
    ----------
        cfg (ConfigDict): ConfigDict object containing configuration for experiment.
    
    Returns:
    --------
        None

    """
    with wandb.init(
        project=cfg.PROJECT_NAME,
        entity=None,
        job_type=cfg.PROCESSED_DATA_ARTIFACT,
        config=dict(cfg)
    ) as run:
        
        cfg=wandb.config
        raw_data_at = run.use_artifact(f'{cfg.RAW_DATA_ARTIFACT}:latest')
        path=raw_data_at.download()
        df=pd.read_csv(path+'/goodreads_train.csv')

        df['review_text']=df.loc[:,'review_text'].map(lambda x:x.lower())
        df['review_text']=df.loc[:,'review_text'].map(lambda x:' '.join(x.split()).strip())

        df.drop_duplicates(subset=['review_text'], inplace=True, keep='first')

        undersample = RandomUnderSampler(random_state = 42)

        df, y_bal = undersample.fit_resample(df.drop(columns=['rating']), df['rating'])

        df['rating']=y_bal
        del y_bal

        random_perm= np.random.permutation(len(df))
        df = df.iloc[random_perm]
        df.reset_index(inplace=True)

        df.drop(columns='index',inplace=True)

        df['full_length']=df['review_text'].map(lambda x:len(x))
        df['mean_word_length']=df['review_text'].map(lambda x:np.mean(list(map(len,x.split()))))

        path_to_processed= f'./data/{cfg.PROCESSED_DATA_FOLDER}/processed.csv'
        
        os.makedirs(f'./data/{cfg.PROCESSED_DATA_FOLDER}',exist_ok=True)
        df.to_csv(path_to_processed)

        processed_data_art=wandb.Artifact(cfg.PROCESSED_DATA_ARTIFACT,type=cfg.DATASET_TYPE)
        processed_data_art.add_file(path_to_processed)
        run.log_artifact(processed_data_art)

def split_and_log(cfg):
    """
    Splits the data into train/valid/test splits.
    
    Parameters
    ----------
        cfg (ConfigDict): ConfigDict object containing configuration for experiment.
    
    Returns:
    --------
        None

    """

    with wandb.init(
        project=cfg.PROJECT_NAME,
        job_type=cfg.SPLIT_DATA_JOB_TYPE,
        config=dict(cfg)
    ) as run:
        cfg=wandb.config

        processed_data_at=run.use_artifact(f'{cfg.PROCESSED_DATA_ARTIFACT}:latest')
        _ = processed_data_at.download()
        df=pd.read_csv(f'./data/{cfg.PROCESSED_DATA_FOLDER}/processed.csv')

        gs_test = GroupShuffleSplit(n_splits=2, train_size=.8, random_state=42)
        gs_valid = GroupShuffleSplit(n_splits=2, train_size=.75, random_state=43)

        train_idx,test_idx=next(iter(gs_test.split(df,groups=df.book_id)))

        train_df=df.loc[train_idx].reset_index(drop=True)
        test_df=df.loc[test_idx].reset_index(drop=True)

        train_idx,valid_idx=next(iter(gs_valid.split(train_df,groups=train_df.book_id)))

        valid_df=train_df.loc[valid_idx]
        train_df=train_df.loc[train_idx]

        train_df.reset_index(drop=True,inplace=True)
        valid_df.reset_index(drop=True,inplace=True)
        test_df.reset_index(drop=True,inplace=True)

        tokenizer=AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        
        def tokenize_batch(batch):
            tokenized_batch = tokenizer(batch['review_text'],padding='max_length',
                                        max_length=512,
                                        truncation=True)
            return tokenized_batch
        
        train_dataset=Dataset.from_pandas(train_df)
        valid_dataset=Dataset.from_pandas(valid_df)
        test_dataset=Dataset.from_pandas(test_df)

        train_dataset=train_dataset.map(tokenize_batch,batched=True)
        valid_dataset=valid_dataset.map(tokenize_batch,batched=True)
        test_dataset=test_dataset.map(tokenize_batch,batched=True)

        train_dataset.save_to_disk(cfg.TRAIN_DATA_FOLDER)

        valid_dataset.save_to_disk(cfg.VALID_DATA_FOLDER)
        test_dataset.save_to_disk(cfg.TEST_DATA_FOLDER)

        train_data_art=wandb.Artifact(cfg.TRAIN_DATA_ARTIFACT,type=cfg.DATASET_TYPE)
        valid_data_art=wandb.Artifact(cfg.VALID_DATA_ARTIFACT,type=cfg.DATASET_TYPE)
        test_data_art=wandb.Artifact(cfg.TEST_DATA_ARTIFACT,type=cfg.DATASET_TYPE)

        train_data_art.add_dir(cfg.TRAIN_DATA_FOLDER)
        valid_data_art.add_dir(cfg.VALID_DATA_FOLDER)
        test_data_art.add_dir(cfg.TEST_DATA_FOLDER)

        run.log_artifact(train_data_art)
        run.log_artifact(valid_data_art)
        run.log_artifact(test_data_art)

def run_data_pipeline(cfg):
    """
    Runs the data processing pipeline.
    """
    processed_file=f'./data/{cfg.PROCESSED_DATA_FOLDER}/processed.csv'

    raw_data_path='./data/raw_data/goodreads_train.csv'
    if not os.path.isfile(raw_data_path):
        log_raw_data(cfg)
    
    if not os.path.isfile(processed_file):
        downsample_and_log(cfg)

    split_and_log(cfg)


if __name__ == "__main__":
    run_data_pipeline(default_cfg)