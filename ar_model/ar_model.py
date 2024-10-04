from transformers import (
    AutoTokenizer,
    EncodecModel,
    AutoProcessor,
    LlamaModel,
    LlamaConfig,
)
from .ar_utils import model_configs

# Audio encoder, FaceBook Encodec-32khz
encodec_model = EncodecModel.from_pretrained(model_configs.encodec_id)

audio_processor = AutoProcessor.from_pretrained(
    model_configs.encodec_id
)  # preprocessor for neural audio codec

# freeze or prevent gradient update
encodec_model.eval()

# LLM tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_configs.llama_id,
)

# Llama model architecture, for initial experiments
llama_config = LlamaConfig(
    num_attention_heads=16,
    num_hidden_layers=8,
    num_key_value_heads=4,
    hidden_size=1024,
    intermediate_size=4096,
)

tiny_llama = LlamaModel(config=llama_config)
