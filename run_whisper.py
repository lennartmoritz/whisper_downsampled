from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from datasets import load_dataset, Audio, Split
from transformers.audio_utils import mel_filter_bank
import torch
from evaluate import load
import sys
from tqdm import tqdm
import numpy as np

from transformers import Seq2SeqTrainer

def evaluate():
    # librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test")
    ds_factor = 1
    librispeech_test_clean = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    librispeech_test_clean = librispeech_test_clean.cast_column("audio", Audio(sampling_rate=16000//ds_factor))
    print(librispeech_test_clean["audio"][0])
    """ This prints:
    {'path': '/home/lennux/.cache/huggingface/datasets/downloads/extracted/80f773ffef1f9e7c284a356bb99db2f740655f871257acd846751dde70987539/dev_clean/1272/128104/1272-128104-0000.flac', 'array': array([0.00238037, 0.0020752 , 0.00198364, ..., 0.00042725, 0.00057983,
       0.0010376 ]), 'sampling_rate': 16000}

    """
    print(len(librispeech_test_clean["audio"][0]["array"]))
    print(type(librispeech_test_clean["audio"][0]["array"][0]))

    ds_feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    # ds_factor = 2
    ds_feature_extractor.n_fft = ds_feature_extractor.n_fft
    ds_feature_extractor.sampling_rate = ds_feature_extractor.sampling_rate // ds_factor
    ds_feature_extractor.hop_length = ds_feature_extractor.hop_length // 1
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
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to("cuda")
    print(model)

    def map_to_pred(batch):
        print(len(batch))
        print(len(batch["audio"]))
        audio = batch["audio"]
        # input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        my_temp_audio = batch["audio"]["array"].copy()
        print(ds_factor)
        input_features = processor(my_temp_audio, sampling_rate=16000/ds_factor, return_tensors="pt").input_features
        batch["reference"] = processor.tokenizer._normalize(batch['text'])

        with torch.no_grad():
            predicted_ids = model.generate(input_features.to("cuda"))[0]
        transcription = processor.decode(predicted_ids)
        batch["prediction"] = processor.tokenizer._normalize(transcription)
        return batch

    result = librispeech_test_clean.map(map_to_pred)

    wer = load("wer")
    print(100 * wer.compute(references=result["reference"], predictions=result["prediction"]))

def train_on_data():
    # librispeech_train_clean = load_dataset("librispeech_asr", "clean", split="train.100")
    librispeech_train_clean = load_dataset("librispeech_asr", "clean", split=Split.VALIDATION)
    librispeech_train_clean = librispeech_train_clean.cast_column("audio", Audio(sampling_rate=16000//ds_factor))
    ds_factor = 1
    ds_feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    ds_feature_extractor.n_fft = ds_feature_extractor.n_fft
    ds_feature_extractor.sampling_rate = ds_feature_extractor.sampling_rate // ds_factor
    ds_feature_extractor.hop_length = ds_feature_extractor.hop_length // 1
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
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to("cuda")
    
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test")
    pass

if __name__ == "__main__":
    evaluate()