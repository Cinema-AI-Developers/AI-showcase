import torch
from huggingface_hub import hf_hub_url
import requests
from tokenizer import text_encoder, tokenizer
from Config import embed_repos, embed_paths

model_id = "stabilityai/stable-diffusion-2-1-base"
def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    print(trained_token)


for embed_repo, embed_path in zip(embed_repos, embed_paths):
    token_url=hf_hub_url(repo_id=embed_repo,
                         filename="learned_embeds.bin")
    r = requests.get(token_url)
    open(embed_path, 'wb').write(r.content)

for embed in embed_paths:
    load_learned_embed_in_clip(embed, text_encoder, tokenizer)