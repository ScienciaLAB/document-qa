import os
from typing import Annotated, List
from fastapi import Request, HTTPException, Form

import modal
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        "fastapi[standard]",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

MODELS_DIR = "/llamas"
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
MODEL_REVISION = "84344a23ee1820ac951bc365f1e91d094a911763"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("intfloat-multilingual-e5-large-instruct-embeddings")


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    print("Loading model...")
    device = get_device()
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct').to(device)
    print("Model loaded successfully.")

    return tokenizer, model, device


N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@app.function(
    image=image,
    gpu=f"L40S:{N_GPU}",
    # gpu=f"A10G:{N_GPU}",
    # how long should we stay up with no requests?
    scaledown_window=3 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("document-qa-embedding-key")]
)
@modal.concurrent(
    max_inputs=5
)  # how many requests can one replica handle? tune carefully!
@modal.fastapi_endpoint(method="POST")
def embed(request: Request, text: Annotated[str, Form()]):
    api_key = request.headers.get("x-api-key")
    expected_key = os.environ["API_KEY"]

    if api_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


    texts = [t for t in text.split("\n") if t.strip()]
    if not texts:
        return []
        
    tokenizer, model, device = load_model()
    model.eval()

    print(f"Start embedding {len(texts)} texts")
    try:
        with torch.no_grad():
            # Move inputs to the same device as model
            batch_dict = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            
            # Forward pass
            outputs = model(**batch_dict)
            
            # Process embeddings
            embeddings = average_pool(
                outputs.last_hidden_state, 
                batch_dict['attention_mask']
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Move to CPU and convert to list for serialization
            embeddings = embeddings.cpu().numpy().tolist()
            
        print("Finished embedding texts.")
        return embeddings
        
    except RuntimeError as e:
        print(f"Error during embedding: {str(e)}")
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory error. Try reducing batch size or using a smaller model.")
        raise
