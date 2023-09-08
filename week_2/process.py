import os
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset, Dataset, load_metric

from params import default_cfg
from imblearn.under_sampling import RandomUnderSampler

def log_raw_data(cfg):
    """
    Downloads the Goodreads dataset from Kaggle and logs it as an artifact in Wandb.

    Parameters:
        cfg (ConfigDict): Configuration object containing settings for the experiment.

    Returns:
        None
    """
    # Initialize a new run with Wandb
    with wandb.init(
        project=cfg.PROJECT_NAME,
        job_type=cfg.RAW_DATA_JOB_TYPE,
        config=dict(cfg)
    ) as run:
        # Update the config with Wandb configuration
        cfg = wandb.config
        
        # Create necessary directories if they don't exist
        os.makedirs('./data', exist_ok=True)
        os.makedirs('./data/raw_data', exist_ok=True)
        
        # Download the dataset from Kaggle
        os.system('kaggle competitions download -c goodreads-books-reviews-290312')
        
        # Unzip the downloaded dataset
        os.system('unzip -d ./data/raw_data goodreads-books-reviews-290312.zip')

        # Define the path to the training data
        train_path = './data/raw_data/goodreads_train.csv'

        # Create a new Wandb Artifact for the raw data
        raw_data_art = wandb.Artifact(cfg.RAW_DATA_ARTIFACT, type=cfg.DATASET_TYPE)
        
        # Add the training data file to the artifact
        raw_data_art.add_file(train_path)
        
        # Log the raw data artifact to Wandb
        run.log_artifact(raw_data_art)

def downsample_and_log(cfg):
    """
    Downsamples the classes, preprocesses the text, and logs the processed data as an artifact.

    Parameters:
        cfg (ConfigDict): Configuration object containing settings for the experiment.

    Returns:
        None
    """
    # Initialize a new run with Wandb
    with wandb.init(
        project=cfg.PROJECT_NAME,
        entity=None,
        job_type=cfg.PROCESSED_DATA_ARTIFACT,
        config=dict(cfg)
    ) as run:
        # Update the config with Wandb configuration
        cfg = wandb.config
        
        # Retrieve the latest raw data artifact
        raw_data_at = run.use_artifact(f'{cfg.RAW_DATA_ARTIFACT}:latest')
        
        # Download the raw data artifact
        path = raw_data_at.download()
        
        # Read the raw data into a DataFrame
        df = pd.read_csv(path + '/goodreads_train.csv')

        # Preprocess the review_text column
        df['review_text'] = df.loc[:, 'review_text'].map(lambda x: x.lower())
        df['review_text'] = df.loc[:, 'review_text'].map(lambda x: ' '.join(x.split()).strip())

        # Remove duplicate rows based on review_text
        df.drop_duplicates(subset=['review_text'], inplace=True, keep='first')

        # Perform class downsampling using RandomUnderSampler
        undersample = RandomUnderSampler(random_state=42)
        df, y_bal = undersample.fit_resample(df.drop(columns=['rating']), df['rating'])

        # Assign the balanced ratings back to the DataFrame
        df['rating'] = y_bal
        del y_bal

        # Randomly permute the rows in the DataFrame
        random_perm = np.random.permutation(len(df))
        df = df.iloc[random_perm]
        df.reset_index(inplace=True)

        # Drop the 'index' column
        df.drop(columns='index', inplace=True)

        # Compute additional features and statistics on the text
        df['full_length'] = df['review_text'].map(lambda x: len(x))
        df['mean_word_length'] = df['review_text'].map(lambda x: np.mean(list(map(len, x.split()))))

        # Define the path to the processed data CSV file
        path_to_processed = f'./data/{cfg.PROCESSED_DATA_FOLDER}/processed.csv'
        
        # Create the directory if it doesn't exist
        os.makedirs(f'./data/{cfg.PROCESSED_DATA_FOLDER}', exist_ok=True)
        
        # Save the preprocessed DataFrame to a CSV file
        df.to_csv(path_to_processed)

        # Create a new Wandb Artifact for the processed data
        processed_data_art = wandb.Artifact(cfg.PROCESSED_DATA_ARTIFACT, type=cfg.DATASET_TYPE)
        
        # Add the processed data CSV file to the artifact
        processed_data_art.add_file(path_to_processed)
        
        # Log the processed data artifact to Wandb
        run.log_artifact(processed_data_art)

def split_and_log(cfg):
    """
    Splits the data into train/valid/test splits and logs them as artifacts in Wandb.

    Parameters:
        cfg (ConfigDict): Configuration object containing settings for the experiment.

    Returns:
        None
    """
    # Initialize a new run with Wandb
    with wandb.init(
        project=cfg.PROJECT_NAME,
        job_type=cfg.SPLIT_DATA_JOB_TYPE,
        config=dict(cfg)
    ) as run:
        # Update the config with Wandb configuration
        cfg = wandb.config

        # Initialize StratifiedGroupKFold for creating test and validation splits
        sgkf_test = StratifiedGroupKFold(n_splits=5)
        sgkf_valid = StratifiedGroupKFold(n_splits=4)

        # Retrieve the latest processed data artifact
        processed_data_at = run.use_artifact(f'{cfg.PROCESSED_DATA_ARTIFACT}:latest')
        
        # Download the processed data artifact
        _ = processed_data_at.download()
        
        # Read the preprocessed data into a DataFrame
        df = pd.read_csv(f'./data/{cfg.PROCESSED_DATA_FOLDER}/processed.csv')

        # Prepare data for test split
        groups_test = df['book_id'].to_numpy()
        y_test = df['user_id'].to_numpy()

        # Generate indices for test split
        train_idxs, test_idxs = next(iter(sgkf_test.split(np.arange(len(groups_test)), y_test, groups_test)))
        
        # Create test DataFrame
        test_df = df.iloc[test_idxs]
        train_df = df.iloc[train_idxs]

        # Prepare data for validation split
        groups_valid = train_df['book_id'].to_numpy()
        y_valid = train_df['user_id'].to_numpy()

        # Generate indices for validation split
        train_idxs, valid_idxs = next(iter(sgkf_valid.split(np.arange(len(groups_valid)), y_valid, groups_valid)))

        # Create validation DataFrame
        valid_df = train_df.iloc[valid_idxs]
        train_df = train_df.iloc[train_idxs]

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        
        # Tokenization function for batching
        def tokenize_batch(batch):
            tokenized_batch = tokenizer(batch['review_text'], padding='max_length',
                                        max_length=512,
                                        truncation=True)
            return tokenized_batch
        
        # Convert DataFrames to Datasets and apply tokenization
        train_dataset = Dataset.from_pandas(train_df)
        valid_dataset = Dataset.from_pandas(valid_df)
        test_dataset = Dataset.from_pandas(test_df)

        train_dataset = train_dataset.map(tokenize_batch, batched=True)
        valid_dataset = valid_dataset.map(tokenize_batch, batched=True)
        test_dataset = test_dataset.map(tokenize_batch, batched=True)

        # Save the datasets to disk
        train_dataset.save_to_disk(cfg.TRAIN_DATA_FOLDER)
        valid_dataset.save_to_disk(cfg.VALID_DATA_FOLDER)
        test_dataset.save_to_disk(cfg.TEST_DATA_FOLDER)

        # Create Wandb Artifacts for train, valid, and test data
        train_data_art = wandb.Artifact(cfg.TRAIN_DATA_ARTIFACT, type=cfg.DATASET_TYPE)
        valid_data_art = wandb.Artifact(cfg.VALID_DATA_ARTIFACT, type=cfg.DATASET_TYPE)
        test_data_art = wandb.Artifact(cfg.TEST_DATA_ARTIFACT, type=cfg.DATASET_TYPE)

        # Add directories to the artifacts
        train_data_art.add_dir(cfg.TRAIN_DATA_FOLDER)
        valid_data_art.add_dir(cfg.VALID_DATA_FOLDER)
        test_data_art.add_dir(cfg.TEST_DATA_FOLDER)

        # Log the artifacts to Wandb
        run.log_artifact(train_data_art)
        run.log_artifact(valid_data_art)
        run.log_artifact(test_data_art)

        
def run_data_pipeline(cfg):
    """
    Runs the data processing pipeline.

    Parameters:
        cfg (ConfigDict): Configuration object containing settings for the experiment.

    Returns:
        None
    """
    # Step 1: Log Raw Data
    log_raw_data(cfg)

    # Step 2: Downsample and Log Processed Data
    downsample_and_log(cfg)

    # Step 3: Split and Log Data
    split_and_log(cfg)


if __name__ == "__main__":
    run_data_pipeline(default_cfg)