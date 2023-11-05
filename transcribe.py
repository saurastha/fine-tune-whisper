import os
import argparse

import torch
from transformers import pipeline


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath',
                        nargs='+',
                        help='List of filepaths of the audio to be transcribed')

    parser.add_argument('--model',
                        help='Specify the Whisper model (e.g., tiny, small, medium) or a local checkpoint.')

    parser.add_argument('--language',
                        help='Specify the language that the audio contains.')

    parser.add_argument('--task',
                        default='transcribe',
                        help='Specify the task to be performed (e.g., transcribe, translate).')

    args = parser.parse_args()
    return args


def transcribe_speech(filepath, pipe, task, language):
    """
        Transcribe audio files using the specified pipeline.

        Args:
            filepath (str): Filepath of the audio to be transcribed.
            pipe (pipeline): Hugging Face pipeline for automatic speech recognition.
            task (str): The task to be performed.
            language (str): The language contained in the audio.

        Returns:
            str: Transcribed text.
        """
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
    # device = 0 if torch.cuda.is_available() else -1
    device = -1

    pipe = pipeline("automatic-speech-recognition", model=args.model, device=device)

    results = {}

    for file in args.filepath:
        print(f'Transcribing {file}.....')
        text = transcribe_speech(filepath=file,
                                 pipe=pipe,
                                 task=args.task,
                                 language=args.language)

        print(f'{os.path.basename(file)} : {text}\n')

