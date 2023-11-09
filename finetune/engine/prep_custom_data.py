"""
Custom Data Preparation Script

This script prepares custom audio and transcript data for further processing. It organizes audio files and their
corresponding transcriptions into a dataset, performs data type conversions, and optionally splits the data into
train, test, and validation sets.

Usage:
    python custom_data_prep.py --data_dir <data_directory> --save_dir <save_directory>

Args:
    --data_dir (str): Root directory where the data is located.
    --save_dir (str): Directory where the processed data will be saved after processing.

Note:
    The script expects the 'data_dir' to contain 'audio' and 'transcript' directories,
    as per the following directory structure:
    data_dir/
        ├── audio/
        │   ├── collection_1/
        │   │   ├── (audio files related to collection_1)
        │   ├── collection_2/
        │   │   ├── (audio files related to collection_2)
        │   ├── collection_3/
        │   │   ├── (audio files related to collection_3)
        ├── transcripts/
        │   ├── collection_1/
        │   │   ├── transcript.csv (transcriptions for collection_1 audio)
        │   ├── collection_2/
        │   │   ├── transcript.csv (transcriptions for collection_2 audio)
        │   ├── collection_3/
        │   │   ├── transcript.csv (transcriptions for collection_3 audio)

"""

import os
import pandas as pd
from typing import Union
from datasets import Dataset, Audio, DatasetDict, concatenate_datasets, Value
from tqdm import tqdm
import argparse
from pathlib import Path
from finetune.utils.functions import split_data, create_directories


def get_args() -> argparse.Namespace:
    """
        Parse command-line arguments and return them as a namespace.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        default=None,
                        help='Specify the root directory where your data is located.')

    parser.add_argument('--save_dir',
                        default='data/',
                        help='Choose the directory where the processed custom data will be saved.')

    args = parser.parse_args()
    return args


def prepare_custom_data(data_path: Path, save_path: Union[Path, None], eval: bool = False):
    """
        Prepare custom audio and transcript data for further processing.

        Args:
            data_path (Path): Root directory where the data is located.
            save_path (Path): Directory where the processed data will be saved.
            eval (bool): If True, return the final dataset without splitting.

        Returns:
            Dataset: The prepared dataset if eval is True, else splits the data and saves it.

        Raises:
            Exception: If the 'audio' and 'transcript' directories are not found.

        """
    print('######## Preparing data ########')
    data = DatasetDict({})

    directories = os.listdir(data_path)

    # Check if the directories are in specified format
    if 'audio' not in directories or 'transcript' not in directories:
        raise Exception('Directory not found.\n'
                        'Check whether the data path contains audio and transcript directory.')
    else:
        audio_dir = data_path / 'audio'
        transcript_dir = data_path / 'transcript'

        # Extracting the directories as per the custom data directory structure
        temp_fol = [item.parts[-1] for item in transcript_dir.iterdir() if item.is_dir()]
        folders = [item for item in audio_dir.iterdir() if item.parts[-1] in temp_fol]

        for fol in tqdm(folders):
            audios = list(map(str, fol.glob('*.wav'))) + list(map(str, fol.glob('*.mp3')))

            # Get transcription
            transcript_path = transcript_dir / fol.parts[-1] / 'transcript.csv'
            transcript = pd.read_csv(transcript_path)

            audio_transcript_map = dict()

            for audio_id, text in transcript[['id', 'transcript']].values:
                check_id = str(fol / audio_id.strip())
                if check_id in audios:
                    audio_transcript_map[check_id] = text

            audio_dataset = Dataset.from_dict({'audio': [key for key, item in audio_transcript_map.items()],
                                               'transcription': [item for key, item in audio_transcript_map.items()]})

            audio_dataset = audio_dataset.cast_column('audio', Audio(sampling_rate=16_000))
            audio_dataset = audio_dataset.cast_column('transcription', Value('string'))

            data[fol.parts[-1]] = audio_dataset

        final_data = concatenate_datasets([data[dd] for dd in data])

        if eval:
            return final_data

        train_test_val = split_data(final_data, num_splits=3)

        create_directories([save_path])

        train_test_val.save_to_disk(save_path)

        print('######## Preparing Finished. ########')
        print(f'Data saved to {save_path}')

        return train_test_val


if __name__ == '__main__':
    args = get_args()
    _ = prepare_custom_data(data_path=args.data_dir, save_path=args.save_dir)
