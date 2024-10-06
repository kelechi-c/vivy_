import soundfile as sf
import os
import torchaudio
import pydub
import re
from typing import Literal
import random
import numpy as np
from torch import nn
import gc
import torch
import wandb


class config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpath = "samples"


class model_configs:
    encodec_id = "facebook/encodec_32khz"
    llama_id = "meta-llama/Llama-3.2-1B"
    canary_id = "tensorkelechi/kaminari_v1"


class data_configs:
    sample_rate = 32000
    split = 4000
    max_duration = 5
    dtype = torch.float16
    batch_size = 4
    dataset_id = "benjamin-paine/freesound-laion-640k"
    mini_dataset_id = "lewtun/music_genres"
    # processed_repo_id = "tensorkelechi/freesound_mini"


class train_configs:
    precision = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_steps = 4
    epochs = 1
    lr = 1e-4
    sft_file = "kaminari.safetensors"
    model_file = "kaminari.pth"
    outpath = "kaminari"


os.mkdir(config.outpath)

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


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    return model


def clear_mem():
    torch.cuda.empty_cache()
    gc.collect()


def trimpad_audio(audio):
    samples = int(data_configs.sample_rate * data_configs.max_duration)

    if len(audio) > samples:
        audio = audio[:samples]

    else:
        pad_width = samples - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="reflect")

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


def prepare_tokenizer(tokenizer, tokens: list):
    special_tokens = [f"<{music_prefix}{x}>" for x in range(music_vocab_size)]
    tokenizer.add_tokens(special_tokens)
    tokenizer.add_tokens(tokens)

    return tokenizer


# encoding/compressing music/audio waveform to tokens
def encode_music(audio, encodec_model, audio_processor):
    audio_array = trimpad_audio(audio)

    audio_proc = audio_processor(
        raw_audio=audio_array,
        sampling_rate=data_configs.sample_rate,
        return_tensors="pt",
    )  # preprocess audio waveform for encoding

    masks = audio_proc["padding_mask"]  # get processor masks for decoding

    with torch.no_grad():
        audio_tokens = encodec_model.encode(
            # tokenize/encode with pretrained neural codec
            audio_proc["input_values"],
            audio_proc["padding_mask"],
        )
    audio_codes = audio_tokens.audio_codes

    return audio_codes[0][0], masks


# dealing with LLM string output
def tokens2string(tokens):
    """
    Convert visual tokens to a single string with prefix and postfix.
    """
    prefix = music_prefix
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

    tokens_string = "".join(tokens_str)
    tokens_string = f" - {start}{tokens_string}{end}"
    return tokens_string


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
    music = encodec_model.decode(content.cpu(), [None])
    #     print(f'decoded = {music}')
    music = music[0].squeeze(0).detach().cpu()
    print(f"decoded audio = {music.shape}")
    return music


def _postprocess(input):
    extract = extractor2(input)
    reconstruct_codes = content2rvq_codes(extract)
    print(f"recoded {reconstruct_codes.shape}")
    waveform = decode_music(reconstruct_codes)

    waveform = waveform[0].squeeze(0).detach().cpu()

    return waveform


@torch.no_grad()
def bird_call(
    prompt, model, tokenizer=tokenizer
):  # prompt might be just a class/single word/description for v1
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

    input_ids = inputs["input_ids"].to(config.device)
    attn_mask = inputs["attention_mask"].to(config.device)

    gen_tokens = model(input_ids=input_ids, attention_mask=attn_mask)[0]

    gen_tokens = model.lm_head(gen_tokens)
    gen_tokens = gen_tokens.argmax(dim=-1)[0]

    tokens = tokenizer.decode(gen_tokens.cpu(), skip_special_tokens=True)
    print(tokens)
    output = _postprocess(tokens)
    print(f"postprocessed: {output}")

    return output


def count_params(model: nn.Module):
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return p_count


def clearmem():
    torch.cuda.empty_cache()
    gc.collect()


def logger(model) -> None:
    wandb.login()
    wandb.init(project="kaminari_v1", name="audiogen-sandbox-5")
    wandb.watch(model)


logger(model)


@torch.no_grad
def epoch_sample(model: LlamaModel = model, prompt="classical"):
    sample_tokens = bird_call(prompt, model, tokenizer)
    sample_numpy = sample_tokens.cpu().numpy().astype(np.float32)
    print(sample_numpy.shape)
    print(sample_numpy.dtype)

    now = datetime.datetime.now()
    filename = now.strftime("%m%d_%H%M%S") + ".wav"
    file_name = os.path.join(config.outpath, filename)

    sf.write(file_name, sample_numpy, data_configs.sample_rate)
    #     torchaudio.save(file_name, sample_tokens, data_configs.sample_rate)#, channels_first=True)
    print("saved: ", file_name)

    return os.path.join(config.outpath, filename)
