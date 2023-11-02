import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model',
                    help='openai whisper model (tiny, small, medium..)')

parser.add_argument('--language',
                    help='language that is to be fine-tuned')

parser.add_argument('--hf_data',
                    default=None,
                    help='huggingface dataset id')

parser.add_argument('--hf_data_config',
                    default=None,
                    help='like ne_np for nepali dataset in google/fleurs')

parser.add_argument('--is_custom_data',
                    default=False,
                    help='set to True if fine tune is to be done on custom data')

parser.add_argument('--custom_data_path',
                    default=None,
                    help='Path to custom data')

parser.add_argument('--save_preprocessed_data',
                    default=True,
                    help='to save or not save the data after preprocessing. defaults to True')

parser.add_argument('--output_dir',
                    help='directory where the outputs like model checkpoint and training results are saved')

parser.add_argument('--training_strategy',
                    help='steps or epoch')

parser.add_argument('--resume_from_ckpt',
                    default=False,
                    help='resume the training from the last saved checkpoint or not default=False')

args = parser.parse_args()