import argparse
from pathlib import Path
from pydantic import ValidationError
from finetune.entity.speech import SpeechSegment
from finetune.engine.trainer import train


def main():
    """
        Fine-tunes a speech recognition model with the specified configuration.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        help='Specify the Whisper model (e.g., tiny, small, medium) to be fine-tuned.')

    parser.add_argument('--language',
                        help='Specify the target language for fine-tuning.')

    parser.add_argument('--hf_dataset_id',
                        default=None,
                        help='Specify the Hugging Face dataset ID if using a pre-defined dataset available in Hugging '
                             'Face hub.')

    parser.add_argument('--hf_dataset_config',
                        default=None,
                        help='Specify the configuration (e.g., ne_np) for specific datasets like Nepali in '
                             'Google/fleurs.')

    parser.add_argument('--is_custom_audio_data',
                        default=False,
                        help='Set this flag to True if fine-tuning is to be performed on custom data.')

    parser.add_argument('--custom_audio_data_path',
                        default=None,
                        help='Specify the path to the custom data if using custom data for fine-tuning.')

    parser.add_argument('--prepare_custom_audio_data',
                        default=False,
                        help='Set this flag to True if the data is in raw format (i.e. the custom data is not '
                             'converted to Hugging Face data). Default: False.')

    parser.add_argument('--save_preprocessed_data',
                        default=True,
                        help='Specify whether to save the preprocessed data. Defaults to True.')

    parser.add_argument('--output_dir',
                        help='Specify the directory where the model checkpoints, preprocessed data and training '
                             'results will be saved.')

    parser.add_argument('--training_strategy',
                        help='Specify the training strategy (steps or epoch) for fine-tuning.')

    arguments = parser.parse_args()

    try:
        args = SpeechSegment(
            model=arguments.model,
            language=arguments.language,
            hf_dataset_id=arguments.hf_dataset_id,
            hf_dataset_config=arguments.hf_dataset_config,
            is_custom_audio_data=arguments.is_custom_audio_data,
            custom_audio_data_path=arguments.custom_audio_data_path,
            prepare_custom_audio_data=arguments.prepare_custom_audio_data,
            save_preprocessed_data=arguments.save_preprocessed_data,
            output_dir=Path(arguments.output_dir),
            training_strategy=arguments.training_strategy)

    except ValidationError as e:
        raise e

    train(args)


if __name__ == '__main__':
    main()
