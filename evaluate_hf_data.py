import os
import re
import glob
import argparse
import pandas as pd
import torch
import evaluate
from datasets import load_dataset, Audio
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

parser = argparse.ArgumentParser()

parser.add_argument('--model',
                    help='openai whisper model (tiny, small, medium..)')

parser.add_argument('--language',
                    help='language that is to be fine-tuned')

parser.add_argument('--hf_data',
                    default=None,
                    help='dataset name in of huggingface dataset')

parser.add_argument('--split',
                    default='test',
                    help='split of data that is to be evaluated')

parser.add_argument('--hf_data_config',
                    default=None,
                    help='like ne_np for nepali dataset in google/fleurs')

parser.add_argument('--task',
                    default='transcribe',
                    help='task to evaluate the data on (like transcribe, translate)')

# parser.add_argument('--output_dir',
#                     help='directory where the evaluation result is saved')

args = parser.parse_args()


def iter_data(dataset):
    for i, item in enumerate(dataset):
        yield item['audio']


BATCH_SIZE = 8

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("automatic-speech-recognition", model=args.model, chunk_length_s=30, device=device)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=args.language, task="transcribe")

wer_metric = evaluate.load("wer")
normalizer = BasicTextNormalizer()

data = load_dataset(args.hf_data, args.hf_data_config, split=args.split)
data = data.cast_column("audio", Audio(sampling_rate=16000))

if 'sentence' in data[args.split].column_names:
    data = data.rename_column('sentence', 'transcription')
if 'text' in data[args.split].column_names:
    data = data.rename_column('sentence', 'transcription')
if 'transcript' in data[args.split].column_names:
    data = data.rename_column('sentence', 'transcription')
if 'normalized_text' in data[args.split].column_names:
    data = data.rename_column('sentence', 'transcription')

references = data['transcription']
norm_references = []
predictions = []
norm_predictions = []

print('####### Evaluation Started #######')

for i, out in enumerate(pipe(iter_data(data), batch_size=BATCH_SIZE)):
    predictions.append(out["text"])
    norm_references.append(normalizer(references[i]))
    norm_predictions.append(normalizer(out['text']))

wer = wer_metric.compute(references=references, predictions=predictions) * 100
norm_wer = wer_metric.compute(references=norm_references, predictions=norm_predictions) * 100

result = (f'Dataset:{args.hf_data} Config:{args.hf_data_config} '
          f'Split:{args.split} Results:WER: {wer} WER Normalised: {norm_wer}')

with open('evaluation_result.txt', 'w') as f:
    f.write(result)

print('####### Evaluation Complete #######')
print('####### Evaluation Result #######\n')
print(result)
