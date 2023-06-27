import os
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datasets import load_dataset, Audio
from transformers.audio_utils import mel_filter_bank
import torch
from evaluate import load
import sys
from tqdm import tqdm
import torchaudio.transforms as T
from time import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def training(ds_factor=1, dataset_fraction=0.01, store_path="models/run_01", epochs=3):
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to("cuda")
    # print(model)

    print("Loading datasets...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.model.encoder.conv1.parameters():
        param.requires_grad = True
    for param in model.model.encoder.conv2.parameters():
        param.requires_grad = True

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    librispeech_train_clean = load_dataset("librispeech_asr", "clean", split="train.100")
    total_samples = len(librispeech_train_clean)
    librispeech_train_clean = load_dataset("librispeech_asr", "clean", split="train.100", streaming=True)
    librispeech_train_clean = librispeech_train_clean.take(int(dataset_fraction * total_samples))

    print("Done [1/2]")
    librispeech_val_clean = load_dataset("librispeech_asr", "clean", split="validation")
    total_val_samples = len(librispeech_val_clean)
    librispeech_val_clean = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
    librispeech_val_clean = librispeech_val_clean.take(int(dataset_fraction * total_val_samples))
    print("Done [2/2]")
    # librispeech_train_clean = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    # librispeech_val_clean = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    print("Downsampling datasets...")
    librispeech_val_clean = librispeech_val_clean.cast_column("audio", Audio(sampling_rate=16000//ds_factor))
    print("Done [1/2]")
    librispeech_train_clean = librispeech_train_clean.cast_column("audio", Audio(sampling_rate=16000//ds_factor))
    print("Done [2/2]")
    # print(librispeech_train_clean["audio"][0])
    # print(librispeech_val_clean["audio"][0])


    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    librispeech_train_clean = librispeech_train_clean.map(prepare_dataset)
    librispeech_val_clean = librispeech_val_clean.map(prepare_dataset)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = load("wer")
    normalizer = BasicTextNormalizer()  # 'official' text normalizer from OpenAI

    def compute_metrics(pred, do_normalize_eval=True):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        if do_normalize_eval:
            pred_str = [normalizer(pred) for pred in pred_str]
            label_str = [normalizer(label) for label in label_str]
            # filtering step to only evaluate the samples that correspond to non-zero references:
            pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
            label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-base-finetuned",  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=40,
        # num_train_epochs=epochs,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=25,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=True,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=librispeech_train_clean,
        eval_dataset=librispeech_val_clean,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    processor.save_pretrained(training_args.output_dir)
    trainer.train()
    trainer.save_model(store_path)


def evaluate(ds_factor=1, dataset_fraction=0.01, load_path="openai/whisper-base"):
    print(ds_factor)
    # librispeech_test_clean = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test")
    total_samples = len(librispeech_test_clean)
    librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    librispeech_test_clean = librispeech_test_clean.take(int(dataset_fraction * total_samples))

    librispeech_test_clean = librispeech_test_clean.cast_column("audio", Audio(sampling_rate=16000//ds_factor))
    # print(librispeech_test_clean["audio"][0])
    # """ This prints:
    # {'path': '/home/lennux/.cache/huggingface/datasets/downloads/extracted/80f773ffef1f9e7c284a356bb99db2f740655f871257acd846751dde70987539/dev_clean/1272/128104/1272-128104-0000.flac', 'array': array([0.00238037, 0.0020752 , 0.00198364, ..., 0.00042725, 0.00057983,
    #    0.0010376 ]), 'sampling_rate': 16000}

    # """
    # print(len(librispeech_test_clean["audio"][0]["array"]))
    # print(type(librispeech_test_clean["audio"][0]["array"][0]))

    model = WhisperForConditionalGeneration.from_pretrained(load_path).to("cuda")


    def map_to_pred(batch):
        audio = batch["audio"]
        # input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        my_temp_audio = batch["audio"]["array"].copy()
        input_features = processor(my_temp_audio, sampling_rate=16000//ds_factor, return_tensors="pt").input_features
        batch["reference"] = processor.tokenizer._normalize(batch['text'])

        with torch.no_grad():
            predicted_ids = model.generate(input_features.to("cuda"))[0]
        transcription = processor.decode(predicted_ids)
        batch["prediction"] = processor.tokenizer._normalize(transcription)
        return batch

    result = librispeech_test_clean.map(map_to_pred)
    # result = {}
    # for batch in librispeech_test_clean:
    #     result = map_to_pred(batch=batch)
    #     print(result)
    # print(len(result))
    # print(type(result))
    # print(result["reference"])

    wer = load("wer")
    print(">>>>>>>> EVALUATION <<<<<<<<")
    print(100 * wer.compute(references=result["reference"], predictions=result["prediction"]))


if __name__ == "__main__":
    ds_factor = 2
    ds_feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    # ds_feature_extractor.n_fft = ds_feature_extractor.n_fft
    ds_feature_extractor.sampling_rate = ds_feature_extractor.sampling_rate // ds_factor
    # ds_feature_extractor.hop_length = ds_feature_extractor.hop_length // 1
    ds_feature_extractor.mel_filters = mel_filter_bank(
        num_frequency_bins=1 + ds_feature_extractor.n_fft // 2,
        num_mel_filters=ds_feature_extractor.feature_size,
        min_frequency=0.0,
        max_frequency=8000.0 // ds_factor,
        sampling_rate=ds_feature_extractor.sampling_rate,
        norm="slaney",
        mel_scale="slaney",
    )

    processor = WhisperProcessor(
        feature_extractor=ds_feature_extractor,
        tokenizer=WhisperTokenizer.from_pretrained("openai/whisper-base"),
    )
    # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to("cuda")

    # evaluate(ds_factor, dataset_fraction=0.03)
    # training(ds_factor, dataset_fraction=0.03, store_path="models/run_04", epochs=3)
    evaluate(ds_factor, dataset_fraction=0.03, load_path="models/run_04")
    # https://github.com/krylm/whisper-event-tuning/blob/master/run_speech_recognition_seq2seq_streaming_mikr.py
    # https://github.com/vasistalodagala/whisper-finetune/blob/master/train/fine-tune_on_hf_dataset.py
