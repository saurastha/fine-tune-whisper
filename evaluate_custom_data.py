import argparse
from pathlib import Path
from datasets import load_from_disk
from finetune.engine.prep_custom_data import prepare_custom_data
from finetune.engine.eval_engine import evaluate_model


def main():
    """
    Perform evaluation of a speech recognition model using specified parameters and data.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        help='Specify the Whisper model (e.g., tiny, small, medium) for evaluation.')

    parser.add_argument('--language',
                        help='Specify the target language for evaluation.')

    parser.add_argument('--task',
                        default='transcribe',
                        help='Specify the task to evaluate the data on (e.g., transcribe, translate).')

    parser.add_argument('--custom_audio_data_path',
                        help='Specify the path to custom data for evaluation.')

    parser.add_argument('--prepare_custom_audio_data',
                        default=True,
                        help='Set to False if the data is already in Hugging Face dataset format and saved on disk. '
                             'Default: True')

    args = parser.parse_args()

    if args.prepare_custom_audio_data:
        data = prepare_custom_data(data_path=Path(args.custom_audio_data_path), save_path=None, eval=True)
    else:
        data = load_from_disk(args.custom_data_path)

    evaluate_model(model=args.model, language=args.language, task=args.task, data=data)


if __name__ == '__main__':
    main()
