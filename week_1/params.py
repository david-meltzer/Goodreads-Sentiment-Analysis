# WANDB BASE PARAMETERS
WANDB_PROJECT = "mlops-course-assgn1"

# WANDB ARTIFACT TYPES
DATASET_TYPE='dataset'
MODEL_TYPE='model'

#WANDB JOB TYPES
RAW_DATA_JOB_TYPE='fetch_raw_data'
DATA_PROCESSING_JOB_TYPE='process-data'
SPLIT_DATA_JOB_TYPE='split-data'
MODEL_TRAINING_JOB_TYPE='model-training'
MODEL_INFERENCE_JOB_TYPE='model-inference'

#WANDB ARTIFACT NAMES
RAW_DATA_ARTIFACT='goodreads_raw_data'
PROCESSED_DATA_ARTIFACT='processed_data'
TRAIN_DATA_ARTIFACT='goodreads_train_data'
VALID_DATA_ARTIFACT='goodreads_valid_data'
TEST_DATA_ARTIFACT='goodreads_test_data'

#DATA FOLDERS
DATA_FOLDER='data'
PROCESSED_DATA_FOLDER='data/processed_data'
FIGURE_FOLDER='figures'
TRAIN_DATA_FOLDER='data/train_data'
TEST_DATA_FOLDER='data/test_data'
VALID_DATA_FOLDER='data/valid_data'
MODEL_DATA_FOLDER = 'distilbert-goodreads-model'

# COLUMNS TO KEEP
FEATURE_COLUMN='review_text'
LABEL_COLUMN='rating'

# TRANSFORMERS PARAMETERS
MODEL_NAME = "distilbert-base-uncased"
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
WARMUP_STEPS = 500
LEARNING_RATE = 5e-5
FP16 = True
NUM_CLASSES=6

# HUB PARAMETERS
PUSH_TO_HUB = True
HUB_MODEL_ID = "distilbert-goodreads-wandb"
HUB_STRATEGY = "every_save"