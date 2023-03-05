
from ml_collections import config_dict
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
default_cfg.PROCESSED_DATA_FOLDER='processed_data'
default_cfg.FIGURE_FOLDER='figures'
default_cfg.TRAIN_DATA_FOLDER='data/train_data'
default_cfg.TEST_DATA_FOLDER='data/test_data'
default_cfg.VALID_DATA_FOLDER='data/valid_data'
default_cfg.MODEL_DATA_FOLDER = 'tiny_bert_goodreads'

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
