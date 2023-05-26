from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from datasets import load_dataset
from transformers.audio_utils import mel_filter_bank
import torch
from evaluate import load
import soundfile as sf


def transcribe():
    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model.config.forced_decoder_ids = None

    # load dummy dataset and read audio files
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]["audio"]
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
    print(transcription)

    print("----------------------------------------------")

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(transcription)

def evaluate():
    librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test")
    librispeech_test_clean = librispeech_test_clean.map(resample_audio)

    # processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    ds_feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    # ds_feature_extractor = WhisperFeatureExtractor() # TODO: delete line
    ds_factor = 2
    ds_feature_extractor.n_fft = ds_feature_extractor.n_fft // ds_factor
    ds_feature_extractor.sampling_rate = ds_feature_extractor.sampling_rate // ds_factor
    ds_feature_extractor.hop_length = ds_feature_extractor.hop_length // ds_factor
    ds_feature_extractor.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + ds_feature_extractor.n_fft // ds_factor,
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

    def map_to_pred(batch):
        audio = batch["audio"]
        input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        batch["reference"] = processor.tokenizer._normalize(batch['text'])

        with torch.no_grad():
            predicted_ids = model.generate(input_features.to("cuda"))[0]
        transcription = processor.decode(predicted_ids)
        batch["prediction"] = processor.tokenizer._normalize(transcription)
        return batch

    result = librispeech_test_clean.map(map_to_pred)

    wer = load("wer")
    print(100 * wer.compute(references=result["reference"], predictions=result["prediction"]))


# Custom function to downsample audio to 8 kHz
def resample_audio(example):
    audio = example["audio"]
    sample_rate = audio["sampling_rate"]
    downsampled_audio = sf.resample(audio, sample_rate, 8000)
    example["file"] = downsampled_audio
    example["sampling_rate"] = 8000
    return example

if __name__ == "__main__":
    evaluate()