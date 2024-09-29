"""
Dataset loading and preprocessing
"""

from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset, Audio
from .ar_utils import encode_music, tokens_to_string
from configs import data_configs, config


# for class-based music data, lewtun/music_genres
musiclass_data = load_dataset(
    data_configs.mini_dataset_id,
    split="train",
    streaming=True,
    trust_remote_code=True,
).cast_column("audio", Audio(sampling_rate=32000))

musiclass_data = musiclass_data.take(1000)


class MusicalClassData(IterableDataset):
    def __init__(self, dataset=musiclass_data):
        self.dataset = dataset

    def __len__(self):
        return data_configs.split

    def __iter__(self):
        for sample in self.dataset:
            audio_tokens = encode_music(
                sample["audio"]["array"],
                encodec_model=encodec_model,
                audio_processor=audio_processor,
            )
            audio_string = tokens_to_string(audio_tokens)

            label = sample["genre"]
            data_string = label + audio_string

            input_tokens = tokenizer(data_string)
            token_ids = input_tokens["input_ids"]
            attn_mask = input_tokens["attention_mask"]

            yield {"input_ids": token_ids, "attention_mask": attn_mask}


mini_train_loader = DataLoader(
    dataset=MusicalClassData(), batch_size=data_configs.batch_size
)


# dataset for LAION audio+caption data


laion_data = load_dataset(
    "benjamin-paine/freesound-laion-640k",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

laion_data = laion_data.cast_column("audio", Audio(sampling_rate=32000)).take(
    data_configs.split
)

x_audio = next(iter(laion_data))
# x_audio
