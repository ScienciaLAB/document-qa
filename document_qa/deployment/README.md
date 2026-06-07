# Modal deployment scripts

This folder contains the [Modal](https://modal.com) apps that serve the LLM and
embedding endpoints used by document-qa. Each script is an independent Modal app:
deploy the ones you need, then point the matching `.env` variables at the URLs
Modal prints.

| Script | Modal app | Serves | Maps to `.env` |
|--------|-----------|--------|----------------|
| `modal_inference_phi.py` | `phi-4-mini-instruct-qa-vllm` | `microsoft/Phi-4-mini-instruct` (vLLM, OpenAI-compatible) | `PHI_URL` |
| `modal_inference_qwen.py` | `qwen-0.6b-qa-vllm` | `Qwen/Qwen3-0.6B` (vLLM, reasoning) | `QWEN_URL` |
| `modal_embeddings_multilang.py` | `intfloat-multilingual-e5-large-instruct-embeddings` | `intfloat/multilingual-e5-large-instruct` | `EMBEDS_URL` |
| `modal_embeddings_en.py` | `intfloat-e5-large-v2-embeddings` | `intfloat/e5-large-v2` (English-only) | `EMBEDS_URL` |

> Both embedding scripts define a tiny global `EmbeddingModel` class that delegates
> to the shared helpers in `_embeddings_app.py` (`cls_kwargs`, `load_embedding_model`,
> `run_embed`). The shared module holds the container image and the embedding logic;
> the model is loaded **once per container** via `@modal.enter()`. To add another
> embedding model, copy one wrapper and change `MODEL_NAME` / `MODEL_REVISION` / the
> app name.

## Prerequisites

```bash
pip install modal
modal token new      # one-time browser auth
```

## Secrets

The scripts read an `API_KEY` from a Modal [Secret](https://modal.com/docs/guide/secrets).
Create the two secrets once (the value is the bearer token clients must send):

```bash
# Used by the inference scripts (phi, qwen)
modal secret create document-qa-api-key API_KEY=<your-llm-token>

# Used by the embedding scripts
modal secret create document-qa-embedding-key API_KEY=<your-embedding-token>
```

| Secret | Used by | Provides |
|--------|---------|----------|
| `document-qa-api-key` | `modal_inference_phi.py`, `modal_inference_qwen.py` | `API_KEY` for the vLLM `--api-key` flag |
| `document-qa-embedding-key` | `modal_embeddings_*.py` | `API_KEY` checked against the `x-api-key` header |

## Deploy

```bash
modal deploy document_qa/deployment/modal_inference_phi.py
modal deploy document_qa/deployment/modal_inference_qwen.py
modal deploy document_qa/deployment/modal_embeddings_multilang.py
# modal deploy document_qa/deployment/modal_embeddings_en.py   # optional English-only
```

Each deploy prints a public `https://<...>.modal.run` URL. Copy it into `.env`:

```env
PHI_URL=https://<account>--phi-4-mini-instruct-qa-vllm-serve.modal.run/v1
QWEN_URL=https://<account>--qwen-0-6b-qa-vllm-serve.modal.run/v1
EMBEDS_URL=https://<account>--embeddings-multilang.modal.run   # English-only: --embeddings-en
API_KEY=<your-llm-token>            # matches document-qa-api-key
EMBEDS_API_KEY=<your-embedding-token>  # matches document-qa-embedding-key
```

> **Inference endpoints** are OpenAI-compatible vLLM servers, so their URLs end in
> `/v1`. **Embedding endpoints** are a custom form endpoint (see below), so their
> URL has no `/v1` suffix.

## Endpoint contracts

### Inference (vLLM)

Standard OpenAI Chat Completions API at `<PHI_URL|QWEN_URL>`, authenticated with the
`Authorization: Bearer <API_KEY>` header. Used by `langchain_openai.ChatOpenAI` in
`streamlit_app.py`.

### Embeddings

A custom `POST` endpoint consumed by
[`ModalEmbeddings`](../custom_embeddings.py):

- **Auth**: `x-api-key: <EMBEDS_API_KEY>` header.
- **Body**: form field `text` with newline-separated strings.
- **Response**: JSON list of L2-normalised vectors, one per input line.

Smoke test:

```bash
curl -X POST "$EMBEDS_URL" \
  -H "x-api-key: $EMBEDS_API_KEY" \
  -F $'text=first sentence\nsecond sentence'
```

## Tuning

These knobs live near the top of each script (or in `_embeddings_app.py`):

| Setting | Where | Notes |
|---------|-------|-------|
| `gpu` | `@app.function` / `@app.cls` | `A10G` is cheaper; `L40S` is faster. Embeddings default to `L40S`, inference to `A10G`. |
| `scaledown_window` | decorator | Idle time before a replica is stopped (cost vs. cold starts). |
| `max_inputs` | `@modal.concurrent` | Concurrent requests per replica — tune to GPU memory. |
| `LABEL` | `modal_embeddings_*.py` | Pins the public URL (`--<label>.modal.run`). Without it Modal truncates the long auto-name and appends a random hash. |
| `FAST_BOOT` | `modal_inference_phi.py` | `--enforce-eager` for faster cold starts vs. peak throughput. |
