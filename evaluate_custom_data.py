import argparse
from datasets import load_from_disk
from finetune.engine.prep_custom_data import prepare_custom_data
from finetune.engine.eval_engine import evaluate_model

parser = argparse.ArgumentParser()

parser.add_argument('--model',
                    help='openai whisper model (tiny, small, medium..)')

parser.add_argument('--language',
                    help='language that is to be fine-tuned')

parser.add_argument('--task',
                    default='transcribe',
                    help='task to evaluate the data on (like transcribe, translate)')

parser.add_argument('--custom_data_path',
                    default=None,
                    help='dataset name in of huggingface dataset')

parser.add_argument('--is_data_raw',
                    default=True,
                    help='set to False if the data is already converted to huggingface dataset and is saved on disk')

parser.add_argument('--split',
                    default='test',
                    help='split of data that is to be evaluated. '
                         'split is only taken into account when the data is already processed as huggingface dataset')

args = parser.parse_args()

if args.is_data_raw:
    data = prepare_custom_data(data_path=args.custom_data_path, save_path=None, eval=True)
else:
    data = load_from_disk(args.custom_data_path)


evaluate_model(model=args.model, language=args.language, task=args.task, data=data)
