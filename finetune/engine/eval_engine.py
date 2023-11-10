import torch
import evaluate
from datasets import Audio, Dataset
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def iter_data(dataset: Dataset):
    """
        Generator function that iterates over a dataset and yields audio items.

        Args:
            dataset (Dataset): The dataset to iterate over.

        Yields:
            dict: A dictionary containing audio data.
        """
    for i, item in enumerate(dataset):
        yield item['audio']


BATCH_SIZE = 8
device = 0 if torch.cuda.is_available() else -1

# device = -1


def evaluate_model(model: str, language: str, task: str, data: Dataset):
    """
        Evaluate a speech-to-text model on a dataset.

        Args:
            model (str): The speech-to-text model to evaluate. Could be hugging face model or local checkpoint.
            language (str): The language for the evaluation.
            task (str): The task for the evaluation.
            data (Dataset): The dataset to evaluate the model on.

        This function evaluates the model on the provided dataset and computes Word Error Rate (WER) and
        Normalized WER (WER Normalized).
        The results are written to 'evaluation_result.txt'.
        """

    pipe = pipeline("automatic-speech-recognition", model=model, chunk_length_s=30, device=device)
    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=language, task=task)

    wer_metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    data = data.cast_column("audio", Audio(sampling_rate=16000))

    if 'sentence' in data.column_names:
        data = data.rename_column('sentence', 'transcription')
    if 'text' in data.column_names:
        data = data.rename_column('sentence', 'transcription')
    if 'transcript' in data.column_names:
        data = data.rename_column('sentence', 'transcription')
    if 'normalized_text' in data.column_names:
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

    result = f'Results:WER: {wer} WER Normalised: {norm_wer}'

    with open('evaluation_result.txt', 'w') as f:
        f.write(result)

    print('####### Evaluation Complete #######')
    print('####### Evaluation Result #######\n')
    print(result)
