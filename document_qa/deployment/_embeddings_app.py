"""Shared building blocks for the Modal embedding endpoints.

``modal_embeddings_en.py`` and ``modal_embeddings_multilang.py`` each define a tiny
``EmbeddingModel`` class at module scope (Modal requires globally-defined classes
with stacked ``@app.cls`` / ``@modal.concurrent`` decorators) that delegates to the
helpers here. All the heavy lifting — the container image, model loading, pooling,
and the embedding request handler — lives in this module so it is written once.

The endpoint contract (consumed by ``document_qa.custom_embeddings.ModalEmbeddings``):

- **Method**: ``POST``
- **Auth**: ``x-api-key`` header, compared against the ``API_KEY`` secret.
- **Body**: form field ``text`` containing newline-separated strings.
- **Response**: JSON list of L2-normalised embedding vectors, one per input line.
"""

import os

import modal
import torch
import torch.nn.functional as F
from fastapi import HTTPException, Request
from torch import Tensor

MINUTES = 60  # seconds
N_GPU = 1

# Shared container image for every embedding model.
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
    # Modal 1.0 no longer auto-mounts imported local modules; the wrapper scripts
    # import this module by name, so it must be added explicitly. Kept last so it
    # doesn't invalidate the (expensive) pip layer above on every code edit.
    .add_local_python_source("_embeddings_app")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


def cls_kwargs() -> dict:
    """Common ``@app.cls`` configuration shared by every embedding endpoint."""
    return dict(
        image=image,
        gpu=f"L40S:{N_GPU}",
        # how long should we stay up with no requests?
        scaledown_window=3 * MINUTES,
        volumes={
            "/root/.cache/huggingface": hf_cache_vol,
            "/root/.cache/vllm": vllm_cache_vol,
        },
        secrets=[modal.Secret.from_name("document-qa-embedding-key")],
    )


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool token embeddings, ignoring padding positions."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def load_embedding_model(model_name: str, model_revision: str):
    """Load a tokenizer + model onto the best available device, once per container.

    Returns:
        tuple: ``(tokenizer, model, device)`` with ``model`` already in eval mode.
    """
    # transformers is only available inside the Modal image, so import lazily.
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    model = AutoModel.from_pretrained(model_name, revision=model_revision).to(device)
    model.eval()
    print("Model loaded successfully.")
    return tokenizer, model, device


def run_embed(tokenizer, model, device, request: Request, text: str):
    """Authenticate, embed newline-separated ``text``, and return normalised vectors."""
    api_key = request.headers.get("x-api-key")
    if api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    texts = [t for t in text.split("\n") if t.strip()]
    if not texts:
        return []

    print(f"Start embedding {len(texts)} texts")
    try:
        with torch.no_grad():
            batch_dict = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            )
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

            outputs = model(**batch_dict)
            embeddings = average_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.cpu().numpy().tolist()

        print("Finished embedding texts.")
        return embeddings

    except RuntimeError as e:
        print(f"Error during embedding: {str(e)}")
        if "CUDA out of memory" in str(e):
            print("CUDA OOM. Try reducing batch size or using a smaller model.")
        raise
