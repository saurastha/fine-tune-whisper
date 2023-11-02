import os
import argparse

import torch
from transformers import pipeline


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath',
                        nargs='+',
                        help='list of filepath of the audio to be transcribed')

    parser.add_argument('--model',
                        help='openai whisper model (tiny, small, medium..) or local checkpoint')

    parser.add_argument('--language',
                        help='language that the audio contains')

    parser.add_argument('--task',
                        default='transcribe',
                        help='task that is to be performed')

    args = parser.parse_args()
    return args


def transcribe_speech(filepath, pipe, task, language):
    output = pipe(
        filepath,
        generate_kwargs={
            "task": task,
            "language": language,
        },  # update with the language you've fine-tuned on
        chunk_length_s=30,
        batch_size=8,
    )
    return output["text"]


if __name__ == '__main__':
    args = get_args()
    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline("automatic-speech-recognition", model=args.model, device=device)

    results = {}

    for file in args.filepath:
        print(f'Transcribing {file}.....')
        text = transcribe_speech(filepath=file,
                                 pipe=pipe,
                                 task=args.task,
                                 language=args.language)

        print(f'{os.path.basename(file)} : {text}\n')

