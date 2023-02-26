from transformers import CLIPTextModel, CLIPTokenizer

model_id = "stabilityai/stable-diffusion-2-1-base"

tokenizer = CLIPTokenizer.from_pretrained(
    model_id,
    subfolder="tokenizer",
    use_auth_token=True,
)
text_encoder = CLIPTextModel.from_pretrained(
    model_id, subfolder="text_encoder", use_auth_token=True
)

vocab_size = tokenizer.vocab_size