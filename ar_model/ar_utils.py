from .pretrained_models import mini_llm_tokenizer
from ..configs import config
from ..music_data.audio_utils import read_audio
from .canary import *
import os
import torchaudio
import pydub
import re
from typing import Literal
import random
import librosa
import numpy as np
from torch import nn
import gc
import torch


class config:
    device = torch.device("cuda")


class model_configs:
    encodec_id = "facebook/encodec_32khz"
    llama_id = "meta-llama/Llama-3.2-1B"
    canary_id = "tensorkelechi/canary_mini"


class data_configs:
    sample_rate = 32000
    split = 1000
    max_duration = 10
    dtype = torch.float16
    batch_size = 4
    dataset_id = "benjamin-paine/freesound-laion-640k"
    mini_dataset_id = "lewtun/music_genres"
    processed_repo_id = "tensorkelechi/freesound_mini"


sample_rate = audio_processor.sample_rate  # or data_configs.sample_rate

music_prefix = "🎶"
start_of_music = "<somu>"
end_of_music = "<eomu>"
music_codebook_size = 2048
music_codebook_num = 4
music_vocab_size = 8192

music_tokens = {
    "prefix": music_prefix,
    "sos": start_of_music,
    "eos": end_of_music,
}


def modality_tokens_to_string(tokens):
    """
    Convert audio/music tokens to a single string with prefix and postfix.
    """
    prefix = music_tokens["prefix"]
    start = music_tokens["sos"]
    end = music_tokens["eos"]

    tokens_str = []
    # music tokens are 2-dim array
    # Convert each token to its corresponding string representation
    for idx in range(len(tokens[0])):
        for layer_idx in range(len(tokens)):
            tokens_str.append(
                f"<{prefix}{tokens[layer_idx][idx] + music_codebook_size * layer_idx}>"
            )

    tokens_string = "".join(tokens_str)
    tokens_string = f"{start}{tokens_string}{end}"

    return tokens_string


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    return model


def clear_mem():
    torch.cuda.empty_cache()
    gc.collect()


def trimpad_audio(audio):
    samples = int(data_configs.sample_rate * data_configs.max_duration)
    audio = audio.numpy()

    if len(audio) > samples:
        audio = audio[:samples]

    else:
        pad_width = samples - len(audio)
        audio = np.pad(audio.numpy(), (0, pad_width), mode="reflect")

    return torch.as_tensor(audio)


def set_channels(audio, target=Literal["mono", "stereo"]):
    channels = len(audio.shape)

    if target == "mono":
        if channels > 1:
            audio = np.mean(audio, axis=1)

    elif target == "stereo":
        audio = np.tile(audio, (2, 1)) if channels < 2 else None

    return audio


def seed_everything(seed=333):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def mp3_to_wav(file: str):
    outpath = os.path.basename(file).split(".")[0]
    outpath = f"{outpath}.wav"
    sound = pydub.AudioSegment.from_mp3(file)
    sound.export(outpath)

    return outpath


def read_audio(audio_file):
    if not audio_file.endswith(".wav"):
        audio_file = mp3_to_wav(audio_file)
    waveform, _ = torchaudio.load(audio_file)
    waveform = trimpad_audio(waveform)

    return waveform


"""
Code for audio/music tokenization and model processing
"""


# configure tokenizer, add tokens
# vocab_ = mini_llm_tokenizer.get_vocab().keys()
# vocab_size = len(vocab_)

# resized model embeddings in canary.py


def prepare_tokenizer(tokenizer, tokens: list):
    special_tokens = [f"<{music_prefix}{x}>" for x in range(music_vocab_size)]
    tokenizer.add_tokens(special_tokens)
    tokenizer.add_tokens(tokens)

    return tokenizer


# encoding/compressing music/audio waveform to tokens
def encode_music(audio, encodec_model=encodec_model, audio_processor=audio_processor):
    audio_array = read_audio(audio)  # read audio_file to waveform

    audio_proc = audio_processor(
        raw_audio=audio_array, sample_rate=sample_rate
    )  # preprocess audio waveform for encoding

    masks = audio_proc["input_masks"]  # get processor masks for decoding

    with torch.no_grad():
        audio_tokens = encodec_model.encode(
            # tokenize/encode with pretrained neural codec
            audio_proc["input_values"],
            audio_proc["input_masks"],
        )

    return audio_tokens.audio_codes, masks


# dealing with LLM string output
def tokens_to_string(tokens, modality="music"):
    """
    Convert visual tokens to a single string with prefix and postfix.
    """
    prefix = music_tokens["prefix"]
    start = music_tokens["sos"]
    end = music_tokens["eos"]

    # music tokens are 2-dim array
    # Convert each token to its corresponding string representation
    tokens_str = []

    for idx in range(len(tokens[0])):
        #         print('layer 1')

        for layer_idx in range(len(tokens)):
            #             print('layer2')
            tokens_str.append(
                f"<{prefix}{tokens[layer_idx][idx] + music_codebook_size * layer_idx}>"
            )

    return start + "".join(tokens_str) + end


def extractor2(text, tag1=start_of_music, tag2=end_of_music):
    start = None
    try:
        # print(text)
        start = text.index(tag1) + len(tag1)
        end = text.index(tag2, start)
        extracted_text = text[start:end].strip()
        if not extracted_text:
            try:
                extracted_text = text[start:]
            except:
                extracted_text = text
        return extracted_text
    except ValueError:
        try:
            extracted_text = text[start:]
        except Exception as e:
            print(e)
            extracted_text = text
        return extracted_text


# for audio decoding
def content2rvq_codes(content, codebook_size=2048, codebook_num=4):
    codes = [int(code) for code in re.findall(r"\d+", content)]
    print(len(codes))  # 6004
    codes = np.array([code % codebook_size for code in codes])
    print(codes.shape)  # (6004,)
    n = codes.shape[0] // codebook_num
    print(n)  # (1501)
    # Transpose the last two dimensions to match the desired output
    # if can't divide evenly, drop the last few codes
    codes = codes[: n * codebook_num]
    print(codes.shape)
    codes = codes.reshape(n, codebook_num).T
    print(codes.shape)  # (4, 1501)
    codes = np.expand_dims(codes, 0)
    codes = np.expand_dims(codes, 0)
    print(codes.shape)  # (1, 1, 4, 1501)
    codes = torch.tensor(codes).long().to(config.device)
    print(codes.shape)
    return codes


def decode_music(content):
    # codes = content2rvq_codes(content, music_codebook_size, music_codebook_num)
    music = encodec_model.decode(content, [None])
    music = music[0].squeeze(0).detach().cpu()
    return music
