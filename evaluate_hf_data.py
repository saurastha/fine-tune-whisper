import argparse
from datasets import load_dataset
from finetune.engine.eval_engine import evaluate_model

parser = argparse.ArgumentParser()

parser.add_argument('--model',
                    help='openai whisper model (tiny, small, medium..)')

parser.add_argument('--language',
                    help='language that is to be fine-tuned')

parser.add_argument('--task',
                    default='transcribe',
                    help='task to evaluate the data on (like transcribe, translate)')

parser.add_argument('--hf_data',
                    default=None,
                    help='dataset name in of huggingface dataset')

parser.add_argument('--hf_data_config',
                    default=None,
                    help='like ne_np for nepali dataset in google/fleurs')

parser.add_argument('--split',
                    default='test',
                    help='split of data that is to be evaluated')

args = parser.parse_args()

data = load_dataset(args.hf_data, args.hf_data_config, split=args.split)

evaluate_model(model=args.model, language=args.language, task=args.task, data=data)
