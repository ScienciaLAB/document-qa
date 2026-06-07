"""Modal deployment: multilingual embeddings (``intfloat/multilingual-e5-large-instruct``).

Deploy with::

    modal deploy document_qa/deployment/modal_embeddings_multilang.py

See ``document_qa/deployment/README.md`` for secrets, tuning, and how the
resulting URL maps to ``EMBEDS_URL`` in ``.env``. The shared logic lives in
``_embeddings_app.py``.
"""

from typing import Annotated

import modal
from fastapi import Form, Request

from _embeddings_app import cls_kwargs, load_embedding_model, run_embed

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
MODEL_REVISION = "84344a23ee1820ac951bc365f1e91d094a911763"

# Pin the public URL label. Without this, Modal derives the label from
# "<app>-<function>", which is too long here and gets truncated with a random
# hash suffix (e.g. ...-embed-c5fe6f.modal.run). The label gives a stable URL:
# https://<workspace>--embeddings-multilang.modal.run
LABEL = "embeddings-multilang"

app = modal.App("intfloat-multilingual-e5-large-instruct-embeddings")


@app.cls(**cls_kwargs())
@modal.concurrent(max_inputs=5)  # requests per replica; tune carefully!
class EmbeddingModel:
    @modal.enter()
    def load_model(self):
        self.tokenizer, self.model, self.device = load_embedding_model(MODEL_NAME, MODEL_REVISION)

    @modal.fastapi_endpoint(method="POST", label=LABEL)
    def embed(self, request: Request, text: Annotated[str, Form()]):
        return run_embed(self.tokenizer, self.model, self.device, request, text)
