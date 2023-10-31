import os
import torch
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import argparse
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from functools import partial
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor

parser = argparse.ArgumentParser()

parser.add_argument("--model",
                    help="openai whisper model (tiny, small, medium..)")

parser.add_argument("--language",
                    help="language that is to be fine-tuned")

parser.add_argument("--hf_data",
                    help="huggingface dataset id")

parser.add_argument("--hf_data_config",
                    default=None,
                    help="like ne_np for nepali dataset in google/fleurs")

parser.add_argument("--output_dir",
                    help="directory where the outputs like model checkpoint and training results are saved")

parser.add_argument("--training_strategy",
                    help="steps or epochs")

parser.add_argument("--resume_from_ckpt",
                    default=False,
                    help="resume the training from the last saved checkpoint or not default=False")

args = parser.parse_args()

model = WhisperForConditionalGeneration.from_pretrained(args.model)
processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task='transcribe')
metric = evaluate.load('wer')
normalizer = BasicTextNormalizer()

model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate, language=args.language, task="transcribe", use_cache=True
)


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    A data collator for speech-to-sequence tasks with padding support.

    Args:
        processor (Any): The Hugging Face Whisper processor used for feature extraction and tokenization.

    Methods:
        __call__(features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            Processes a list of input features, including audio inputs and tokenized labels.
            Returns a dictionary with padded input features and tokenized labels.
    """
    processor: Any

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyway
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

if args.train_strategy == 'epoch':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=,
        gradient_accumulation_steps=1,
        learning_rate=,
        warmup_steps=,
        gradient_checkpointing=,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=,
        save_total_limit=,
        per_device_eval_batch_size=,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_bnb_8bit",
        resume_from_checkpoint=args.resume_from_ckpt,
    )

elif args.train_strategy == 'steps':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=,
        gradient_accumulation_steps=,
        learning_rate=,
        warmup_steps=,
        gradient_checkpointing=,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=,
        save_strategy="steps",
        save_steps=,
        max_steps=,
        save_total_limit=,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_bnb_8bit",
        resume_from_checkpoint=args.resume_from_ckpt,
    )

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

results = []
for obj in trainer.state.log_history:
    results.append(obj)

results_json = json.dumps(results)

with open(f'{args.output_dir}/training_logs.json', 'w') as f:
    f.write(results_json)

processor.save_pretrained(os.path.join(args.output_dir, 'whisper-processor'))
class ModelTrainer:
    """
        A class for training a speech-to-sequence model using Hugging Face's Seq2SeqTrainer.

        Args:
            config (ModelTrainerConfig): An instance of ModelTrainerConfig containing configuration settings.

        Attributes:
            config (ModelTrainerConfig): Configuration settings for model training.
            processor (WhisperProcessor): A Hugging Face Whisper processor for audio and text processing.
            metric (Metric): A metric for evaluating the model's performance.

        Methods:
            train():
                Trains a speech-to-sequence model using Hugging Face's Seq2SeqTrainer.

            compute_metrics(pred):
                Computes evaluation metrics for the model's predictions.
    """

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

        if not config.load_model_from_checkpoint:
            self.processor = WhisperProcessor.from_pretrained(
                self.config.huggingface_model, language=self.config.language, task='transcribe'
            )
        else:
            try:
                self.processor = WhisperProcessor.from_pretrained(
                    self.config.local_model_ckpt, language=self.config.language, task='transcribe'
                )
            except FileNotFoundError as e:
                logger.exception('Checkpoint does not exits. Provide correct directory.')
                raise e

        self.metric = evaluate.load("wer")

        self.normalizer = BasicTextNormalizer()

    def train(self):
        data = load_from_disk(self.config.data_path)

        if self.config.load_model_from_checkpoint:
            try:
                model = WhisperForConditionalGeneration.from_pretrained(self.config.local_model_ckpt)
                logger.info(f'Checkpoint loaded: {model.config._name_or_path}')
            except FileNotFoundError as e:
                logger.exception('Checkpoint does not exits. Provide correct directory.')
                raise e
        else:
            model = WhisperForConditionalGeneration.from_pretrained(self.config.huggingface_model)

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        # disable cache during training since it's incompatible with gradient checkpointing
        model.config.use_cache = False

        # set language and task for generation and re-enable cache
        model.generate = partial(
            model.generate, language=self.config.language, task="transcribe", use_cache=True
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.root_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            evaluation_strategy=self.config.evaluation_strategy,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            predict_with_generate=self.config.predict_with_generate,
            generation_max_length=self.config.generation_max_length,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            logging_dir=f'{self.config.root_dir}/runs',
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            push_to_hub=self.config.push_to_hub,
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=data["train"],
            eval_dataset=data["validation"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor,
        )

        if self.config.load_model_from_checkpoint:
            trainer.train(self.config.local_model_ckpt)
        else:
            trainer.train()

        results = []

        for obj in trainer.state.log_history:
            results.append(obj)

        results_json = json.dumps(results)

        with open(f'{self.config.root_dir}/training_logs.json', 'w') as f:
            f.write(results_json)

        model.save_pretrained(os.path.join(self.config.root_dir, 'whisper-model'))

        self.processor.save_pretrained(os.path.join(self.config.root_dir, 'whisper-processor'))


