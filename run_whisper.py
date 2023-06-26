from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from datasets import load_dataset, Audio
from transformers.audio_utils import mel_filter_bank
import torch
from evaluate import load
import sys
from tqdm import tqdm
import numpy as np

def evaluate():
    # librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test")
    librispeech_test_clean = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    librispeech_test_clean = librispeech_test_clean.cast_column("audio", Audio(sampling_rate=8000))
    print(librispeech_test_clean["audio"][0])
    """ This prints:
    {'path': '/home/lennux/.cache/huggingface/datasets/downloads/extracted/80f773ffef1f9e7c284a356bb99db2f740655f871257acd846751dde70987539/dev_clean/1272/128104/1272-128104-0000.flac', 'array': array([0.00238037, 0.0020752 , 0.00198364, ..., 0.00042725, 0.00057983,
       0.0010376 ]), 'sampling_rate': 16000}

    """
    print(len(librispeech_test_clean["audio"][0]["array"]))
    print(type(librispeech_test_clean["audio"][0]["array"][0]))
    # librispeech_test_clean = librispeech_test_clean.map(resample_audio)
    # librispeech_test_clean = resample_audio(librispeech_test_clean)
    # sys.exit()

    # processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    ds_feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    # ds_feature_extractor = WhisperFeatureExtractor() # TODO: delete line
    ds_factor = 2
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
        # print(f"BEFORE: {len(my_temp_audio)}")
        # my_temp_audio = resample(my_temp_audio)
        # print(f"AFTER: {len(my_temp_audio)}")
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        input_features = processor(my_temp_audio, sampling_rate=8000, return_tensors="pt").input_features
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
def resample_audio(example, ds_factor = 2):
    for index in tqdm(range(30, len(example["audio"]))):
        example["audio"][index]["sampling_rate"] = example["audio"][index]["sampling_rate"] // ds_factor
        # example["audio"][index]["array"] = example["audio"][index]["array"][::2]
        if len(example["audio"][index]["array"]) % 2:
            print(len(example["audio"][index]["array"])) # this is 39119
            example["audio"][index]["array"] = example["audio"][index]["array"][:-1].copy()
            print(len(example["audio"][index]["array"][:-1])) # this is 39118

            # example["audio"][index]["array"] = example["audio"][index]["array"][np.arange(example["audio"][index]["array"].size - 1)]
            temp = example["audio"][index]["array"][np.arange(example["audio"][index]["array"].size - 1)]
            del(example["audio"][index]["array"])
            example["audio"][index]["array"] = None
            print(example["audio"][index]["array"])
            example["audio"][index]["array"] = temp
            
            print("+-+-+-+-+-+-++-+-+-+-+-++-++-+-+-+-++--+-++-+-+-+-+-YEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            print(len(example["audio"][index]["array"])) # this is 39119
        example["audio"][index]["array"] = example["audio"][index]["array"].reshape(-1, 2)[:,1].flatten()
    return example

def resample(x):
     if len(x) % 2:
             x = x[:-1]
     x = x.reshape(-1,2)[:,:1].flatten()
     return x

def test_resample():
    array_list = [np.random.normal(loc=0, scale=0.04, size=np.random.randint(32000, 448000))
                  for _ in range(2620)]
    print(array_list[0][:6])
    print(type(array_list[0][1]))
    with tqdm(total=len(array_list), desc="Resampling") as pbar:
        for i in range(len(array_list)):
            array_list[i] = resample(array_list[i])
            pbar.update(1)
    print(array_list[0][:6])

if __name__ == "__main__":
    evaluate()
    # test_resample()