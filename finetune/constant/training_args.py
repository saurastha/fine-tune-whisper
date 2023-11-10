"""
Hyperparameters and Constants

This file contains hyperparameters and constants used for training and data preprocessing.
"""

# Hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE: int = 16
GRADIENT_ACCUMULATION_STEPS: int = 1
LEARNING_RATE: float = 1e-5
WARMUP_STEPS: int = 5
GRADIENT_CHECKPOINTING: bool = True
FP16: bool = False
SAVE_TOTAL_LIMIT: int = 1
PER_DEVICE_EVAL_BATCH_SIZE: int = 16
PREDICT_WITH_GENERATE: bool = True
GENERATION_MAX_LENGTH: int = 225
LOGGING_STEPS: int = 10
LOAD_BEST_MODEL_AT_END: bool = True
METRIC_FOR_BEST_MODEL: str = 'wer'
GREATER_IS_BETTER: bool = False
OPTIM: str = "adamw_bnb_8bit"
RESUME_FROM_CHECKPOINT: bool = False
USE_CPU: bool = False

# Used when training strategy is 'epoch'
NUM_TRAIN_EPOCHS: int = 20

# Used when training strategy is 'steps'
EVAL_STEPS: int = 10
SAVE_STEPS: int = 10
MAX_STEPS: int = 100

# Constants for preprocessing data
MAX_AUDIO_LENGTH: float = 30.0
MAX_LABEL_LENGTH: int = 448  # Default value for whisper models

