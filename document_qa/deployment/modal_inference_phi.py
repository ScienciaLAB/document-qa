import os

import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

MODELS_DIR = "/llamas"
MODEL_NAME = "microsoft/Phi-4-mini-instruct"
MODEL_REVISION = "c0fb9e74abda11b496b7907a9c6c9009a7a0488f"

FAST_BOOT = True

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


app = modal.App("phi-4-mini-instruct-qa-vllm")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    # gpu=f"L40S:{N_GPU}",
    gpu=f"A10G:{N_GPU}",
    # how long should we stay up with no requests?
    scaledown_window=5 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("document-qa-api-key")]
)
@modal.concurrent(
    max_inputs=5
)  # how many requests can one replica handle? tune carefully!
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--max-model-len",
        "32768",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        os.environ["API_KEY"],
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    subprocess.Popen(" ".join(cmd), shell=True)