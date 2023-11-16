"""
Data Preprocessing Script

This script contains functions for data preprocessing, including loading data,
processing audio, and filtering based on length constraints.
"""

from pathlib import Path
from typing import Tuple
from datasets import Audio, load_dataset, load_from_disk, Dataset
from finetune.engine.prep_custom_data import prepare_custom_data
from finetune.utils.functions import split_data
from finetune.constant.training_args import *
from finetune.utils.functions import update_vocabulary


def load_hf_data(hf_dataset_id: str, hf_dataset_config: str = None):
    """
        Load data from the Hugging Face Datasets library.

        Args:
            hf_dataset_id (str): Hugging Face dataset identifier.
            hf_dataset_config (str): Hugging Face dataset configuration (if available).

        Returns:
            Dataset: The loaded dataset.
        """
    if hf_dataset_config is not None:
        data = load_dataset(hf_dataset_id, name=hf_dataset_config)
    else:
        data = load_dataset(hf_dataset_id)

    if 'validation' not in data:
        train_val = split_data(data['train'], num_splits=2)
        return train_val

    return data


def preprocess(data_source: str, processor, hf_dataset_config: str = None, is_custom_audio_data: bool = False,
               prepare_custom_audio_data: bool = False, custom_audio_data_save_path: Path = None) -> Tuple:
    """
        Preprocess data for training or evaluation.

        Args:
            data_source (str): Path to the data or Hugging Face dataset identifier.
            processor: Data processor for audio and transcriptions.
            hf_dataset_config (dict): Hugging Face dataset configuration (if available).
            is_custom_audio_data (bool): True if the data is custom audio and transcript data.
            prepare_custom_audio_data (bool): True if custom data should be prepared.
            custom_audio_data_save_path (str): Path to save the custom data.

        Returns:
            Dataset: The preprocessed dataset.
        """
    print('######### Data Preprocessing Started #########')

    def prepare_dataset(example):
        audio = example['audio']

        example = updated_processor(
            audio=audio['array'],
            sampling_rate=audio['sampling_rate'],
            text=example['transcription'],
        )

        # compute input length of audio sample in seconds
        example["input_length"] = len(audio['array']) / audio['sampling_rate']

        example['label_length'] = len(example['labels'])

        return example

    if not is_custom_audio_data:
        data = load_hf_data(hf_dataset_id=data_source, hf_dataset_config=hf_dataset_config)
    else:
        if prepare_custom_audio_data:
            data = prepare_custom_data(data_path=Path(data_source), save_path=custom_audio_data_save_path)
        else:
            data = load_from_disk(data_source)

    if 'sentence' in data['train'].column_names:
        data = data.rename_column('sentence', 'transcription')
    if 'text' in data['train'].column_names:
        data = data.rename_column('text', 'transcription')
    if 'transcript' in data['train'].column_names:
        data = data.rename_column('transcript', 'transcription')

    data = data.select_columns(['audio', 'transcription'])

    sampling_rate = processor.feature_extractor.sampling_rate

    # resampling the audio to the sampling rate expected by the model
    data = data.cast_column("audio", Audio(sampling_rate=sampling_rate))

    updated_processor = update_vocabulary(data, processor, custom_audio_data_save_path.parent)

    data['train'] = data['train'].map(
        prepare_dataset, remove_columns=data.column_names['train'], num_proc=1
    )

    data['validation'] = data['validation'].map(
        prepare_dataset, remove_columns=data.column_names['validation'], num_proc=1
    )

    # filter data that is less than 30s
    data['train'] = data['train'].filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
    )

    data['validation'] = data['validation'].filter(
        is_audio_in_length_range,
        input_columns=['input_length'],
    )

    data['train'] = data['train'].filter(
        is_label_length_in_range,
        input_columns=['label_length'],
    )

    data['validation'] = data['validation'].filter(
        is_label_length_in_range,
        input_columns=['label_length'],
    )

    print('######### Data Preprocessing Finished #########')

    return data, updated_processor


def is_audio_in_length_range(length: float) -> bool:
    """
        Check if audio length is within the specified range.

        Args:
            length (float): Audio length in seconds.

        Returns:
            bool: True if the audio length is within the range, otherwise False.
        """
    return length < MAX_AUDIO_LENGTH


def is_label_length_in_range(length: int) -> bool:
    """
        Check if label length is within the specified range.

        Args:
            length (int): Label length.

        Returns:
            bool: True if the label length is within the range, otherwise False.
        """
    return length < MAX_LABEL_LENGTH
