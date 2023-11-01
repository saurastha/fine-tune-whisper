from datasets import Audio, load_dataset, load_from_disk
from utils import split_data
from typing import List


def load_hf_data(hf_data_id, hf_data_config=None):
    if hf_data_config is not None:
        data = load_dataset(hf_data_id, name=hf_data_config)
    else:
        data = load_dataset(hf_data_id)

    if 'validation' not in data:
        train_val = split_data(data['train'], num_splits=2)
        return train_val

    return data


def preprocess(data_path, processor, hf_data_config=None, is_custom_data=False):
    def prepare_dataset(example):
        audio = example['audio']

        example = processor(
            audio=audio['array'],
            sampling_rate=audio['sampling_rate'],
            text=example['transcription'],
        )

        # compute input length of audio sample in seconds
        example["input_length"] = len(audio['array']) / audio['sampling_rate']

        example['label_length'] = len(example['labels'])

        return example

    if not is_custom_data:
        data = load_hf_data(data_path, hf_data_config)
    else:
        data = load_from_disk(data_path)

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

    return data


def is_audio_in_length_range(length):
    return length < 30


def is_label_length_in_range(length):
    return length < 448