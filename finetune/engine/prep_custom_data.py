import os
import pandas as pd
from datasets import Dataset, Audio, DatasetDict, concatenate_datasets, Value
from tqdm import tqdm
import argparse
from pathlib import Path
from finetune.utils.functions import split_data, create_directories


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        default=None,
                        help='Root directory where the data lies.')

    parser.add_argument('--save_dir',
                        default='data/',
                        help='Directory where the custom data is saved after processing.')

    args = parser.parse_args()
    return args


def prepare_custom_data(data_path, save_path, eval=False):
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

        # getting the common directories inside the audio and transcripts folder match
        temp_fol = [item.parts[-1] for item in transcript_dir.iterdir()]
        folders = [item for item in audio_dir.iterdir() if item.parts[-1] in temp_fol]

        for fol in tqdm(folders):
            audios = list(map(str, fol.glob('*.wav'))) + list(map(str, fol.glob('*.mp3')))

            # Get transcription
            transcript_path = transcript_dir / fol.parts[-1] / 'transcript.csv'
            transcript = pd.read_csv(transcript_path)

            audio_transcript_map = dict()

            for audio_id, text in transcript[['id', 'transcription']].values:
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
