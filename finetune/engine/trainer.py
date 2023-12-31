import os
import json
import torch
from functools import partial
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
from datasets import load_from_disk
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import (Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor, WhisperTokenizer,
                          WhisperFeatureExtractor, WhisperForConditionalGeneration)

from finetune.utils.functions import create_directories, get_size
from finetune.engine.data_setup import preprocess
from finetune.constant.training_args import *


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


def train(args):
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

    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    # tokenizer = WhisperTokenizer.from_pretrained(args.model, language=args.language, task="transcribe")
    # feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)

    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task='transcribe')
    metric = evaluate.load('wer')
    normalizer = BasicTextNormalizer()

    model.config.use_cache = False

    # Set language and task for generation and re-enable cache
    model.generate = partial(
        model.generate, language=args.language, task="transcribe", use_cache=True
    )

    create_directories([args.output_dir])

    preprocessed_data_path = args.output_dir / 'preprocessed_data'

    # Load preprocessed data if it exists
    if os.path.exists(preprocessed_data_path):
        processed_data_size = get_size(preprocessed_data_path)
        if processed_data_size != 0:
            print(f'\nProcessed data already exists of size: {processed_data_size}')

        data = load_from_disk(str(preprocessed_data_path))
        updated_processor = WhisperProcessor.from_pretrained(f'{args.output_dir}/whisper-processor')

    else:
        # Load Hugging Face dataset
        if not args.is_custom_audio_data:
            data, updated_processor = preprocess(data_source=args.hf_dataset_id,
                                                 hf_dataset_config=args.hf_dataset_config,
                                                 processor=processor)
        else:
            # Load custom data
            custom_data_save_path = args.output_dir / 'custom_data'
            data, updated_processor = preprocess(data_source=args.custom_audio_data_path,
                                                 processor=processor,
                                                 is_custom_audio_data=True,
                                                 prepare_custom_audio_data=args.prepare_custom_audio_data,
                                                 custom_audio_data_save_path=custom_data_save_path)

        if args.save_preprocessed_data:
            create_directories([preprocessed_data_path])
            data.save_to_disk(preprocessed_data_path)

    model.resize_token_embeddings(len(updated_processor.tokenizer))

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=updated_processor)

    # Defining training arguments
    if args.training_strategy == 'epoch':
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            fp16=FP16,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=NUM_TRAIN_EPOCHS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            predict_with_generate=PREDICT_WITH_GENERATE,
            generation_max_length=GENERATION_MAX_LENGTH,
            logging_steps=LOGGING_STEPS,
            load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
            metric_for_best_model=METRIC_FOR_BEST_MODEL,
            greater_is_better=GREATER_IS_BETTER,
            resume_from_checkpoint=RESUME_FROM_CHECKPOINT,
        )

    elif args.training_strategy == 'steps':
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            fp16=FP16,
            evaluation_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            max_steps=MAX_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            predict_with_generate=PREDICT_WITH_GENERATE,
            generation_max_length=GENERATION_MAX_LENGTH,
            logging_steps=LOGGING_STEPS,
            load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
            metric_for_best_model=METRIC_FOR_BEST_MODEL,
            greater_is_better=GREATER_IS_BETTER,
            resume_from_checkpoint=RESUME_FROM_CHECKPOINT
        )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=updated_processor
    )

    print('Training in progress......')
    trainer.train()
    print('######### Training Completed #########')

    results = []
    for obj in trainer.state.log_history:
        results.append(obj)

    results_json = json.dumps(results)

    with open(f'{args.output_dir}/training_logs.json', 'w') as f:
        f.write(results_json)

    print(f'Outputs saved to {args.output_dir}')
