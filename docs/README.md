# 📝 document-qa-engine documentation

>  **License**: Apache 2.0 · **PyPI**: `pip install document-qa-engine`

A Python library and Streamlit application for **Question/Answering on scientific PDF documents** using Retrieval-Augmented Generation (RAG). It uses [GROBID](https://github.com/kermitt2/grobid) for structured text extraction, [ChromaDB](https://www.trychroma.com/) for vector storage, and any OpenAI-compatible LLM for answering.


## Overview

Most PDF Q/A tools feed raw extracted text to an LLM, which is noisy and loses document structure. **document-qa-engine** takes a different approach:

1. **Structured extraction** Sends the PDF to a GROBID server, which returns TEI-XML with separate sections (title, abstract, body paragraphs, figures, back matter) and precise bounding-box coordinates for every paragraph.
2. **Smart chunking** Paragraphs can be kept as-is or merged into larger chunks using token-aware merging, while preserving coordinate metadata.
3. **Vector embeddings** Each chunk is embedded (via a remote API or local model) and stored in an in-memory ChromaDB collection.
4. **Retrieval + LLM answering** User questions are embedded, the most similar chunks are retrieved, and an LLM generates an answer from that context.
5. **PDF highlighting**  The Streamlit frontend highlights the exact PDF regions the LLM used, with a color gradient (orange = most relevant, blue = least relevant).
6. **NER post-processing** *(optional)* LLM responses are scanned for physical quantities (via grobid-quantities) and materials mentions (via grobid-superconductors), then annotated inline.


## Installation

### Option 1: PyPI (library only)

```bash
pip install document-qa-engine
```

### Option 2: From source (full app)

```bash
git clone https://github.com/lfoppiano/document-qa.git
cd document-qa
pip install -r requirements.txt
```

### Option 3: Docker

```bash
# Latest stable release
docker run -p 8501:8501 lfoppiano/document-insights-qa:latest

# Latest development build
docker run -p 8501:8501 lfoppiano/document-insights-qa:latest-develop
```

### Prerequisites

You need access to:

| Service | Required? | Purpose |
|---------|-----------|---------|
| **GROBID server** | ✅ Yes | Parses PDFs into structured text |
| **Embedding API** | ✅ Yes | Converts text to vectors |
| **LLM API** (OpenAI-compatible) | ✅ Yes | Answers questions |
| **grobid-quantities** | ❌ Optional | NER for measurements |
| **grobid-superconductors** | ❌ Optional | NER for materials |



## Configuration

All configuration is through environment variables. Create a `.env` file in the project root:

```env
# ── LLM Endpoints ────────────────────────────────────────
# Each key in API_MODELS maps a model name to its base URL.
PHI_URL=http://localhost:1234/v1          # Phi-4-mini-instruct endpoint
QWEN_URL=http://localhost:1234/v1         # Qwen3-0.6B endpoint
API_KEY=your-llm-api-key                  # Auth key for LLM APIs

# ── Embedding Endpoint ───────────────────────────────────
EMBEDS_URL=http://127.0.0.1:1234/v1      # Embedding service URL
EMBEDS_API_KEY=your-embedding-api-key     # Auth key for embedding API

# ── Defaults ─────────────────────────────────────────────
DEFAULT_MODEL=microsoft/Phi-4-mini-instruct
DEFAULT_EMBEDDING=intfloat/multilingual-e5-large-instruct-modal

# ── GROBID Services ──────────────────────────────────────
GROBID_URL=https://your-grobid-url
GROBID_QUANTITIES_URL=https://your-grobid-quantities-url/
GROBID_MATERIALS_URL=https://your-grobid-superconductors-url/
```

### Variable Reference

| Variable | Description |
|----------|-------------|
| `PHI_URL` | Base URL for the Phi-4-mini-instruct vLLM server (OpenAI-compatible) |
| `QWEN_URL` | Base URL for the Qwen3-0.6B vLLM server (OpenAI-compatible) |
| `API_KEY` | Bearer token for authenticating with the LLM endpoints |
| `EMBEDS_URL` | Base URL for the embedding service (must expose `/embeddings` endpoint) |
| `EMBEDS_API_KEY` | Bearer token for authenticating with the embedding service |
| `DEFAULT_MODEL` | Model name pre-selected in the UI dropdown |
| `DEFAULT_EMBEDDING` | Embedding name pre-selected in the UI dropdown |
| `GROBID_URL` | Full URL to a running GROBID server |
| `GROBID_QUANTITIES_URL` | URL to a grobid-quantities server (for measurement NER) |
| `GROBID_MATERIALS_URL` | URL to a grobid-superconductors server (for materials NER) |

---

## Quick Start — Streamlit App

```bash
# 1. Set up environment
cp .env.example .env   # Edit with your endpoints

# 2. Run the app
streamlit run streamlit_app.py
```

Then open `http://localhost:8501`, upload a PDF, and ask questions.

---

## Quick Start — As a Python Library

```python
from langchain_openai import ChatOpenAI
from document_qa.custom_embeddings import ModalEmbeddings
from document_qa.document_qa_engine import DocumentQAEngine, DataStorage

# 1. Set up the LLM
llm = ChatOpenAI(
    model="microsoft/Phi-4-mini-instruct",
    temperature=0.0,
    base_url="http://localhost:1234/v1",
    api_key="your-api-key"
)

# 2. Set up embeddings
embeddings = ModalEmbeddings(
    url="http://localhost:1234/v1",
    model_name="intfloat/multilingual-e5-large-instruct",
    api_key="your-embedding-key"
)

# 3. Create the storage and engine
storage = DataStorage(embeddings)
engine = DocumentQAEngine(
    llm=llm,
    data_storage=storage,
    grobid_url="https://lfoppiano-grobid.hf.space/"
)

# 4. Load a PDF (creates in-memory embeddings)
doc_id = engine.create_memory_embeddings(
    pdf_path="path/to/paper.pdf",
    chunk_size=500       # tokens per chunk (-1 = keep paragraphs)
)

# 5. Ask a question
_, answer, coordinates = engine.query_document(
    query="What is the main contribution of this paper?",
    doc_id=doc_id,
    context_size=10      # number of chunks to use as context
)
print(answer)

# 6. Or just retrieve relevant passages (no LLM)
passages, coordinates = engine.query_storage(
    query="What materials were studied?",
    doc_id=doc_id,
    context_size=5
)
for p in passages:
    print(p)
```


## Streamlit App Features

### Query Modes

| Mode | What It Does | When to Use |
|------|-------------|-------------|
| **LLM Q/A** | Retrieves context → sends to LLM → returns a natural language answer | Default — for asking questions |
| **Embeddings** | Returns the raw text passages most similar to your question | Debugging — to see what context the LLM would receive |
| **Question Coefficient** | Computes `min_similarity - mean_similarity` as a quality estimate | Experimental — to predict answer reliability |

### Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Chunk size | `-1` (paragraphs) | Token count per text chunk. `-1` keeps GROBID paragraphs intact. |
| Context size | `10` (paragraphs) / `4` (chunks) | Number of chunks sent to the LLM as context |
| Scroll to context | Off | Auto-scroll the PDF viewer to the most relevant passage |
| NER processing | Off | Run grobid-quantities + grobid-superconductors on LLM responses |

### PDF Annotations

After each query, the PDF viewer highlights the passages used as context:
- **Orange** (warm) = most relevant passage
- **Blue** (cold) = least relevant passage
- **Dotted border** = the single most relevant passage



## Troubleshooting

### SQLite version error

```
streamlit: Your system has an unsupported version of sqlite3.
Chroma requires sqlite3 >= 3.35.0.
```

**Linux fix**: See [this StackOverflow answer](https://stackoverflow.com/questions/76958817/streamlit-your-system-has-an-unsupported-version-of-sqlite3-chroma-requires-sq).
**More info**: [Chroma troubleshooting docs](https://docs.trychroma.com/troubleshooting#sqlite).

### "The information is not provided in the given context"

The LLM couldn't find the answer in the retrieved passages. Try:
1. **Increase context size** — use the sidebar slider to retrieve more passages
2. **Decrease chunk size** — smaller chunks may match more precisely
3. **Use Embeddings mode** — switch to "Embeddings" query mode to see what passages are being retrieved and verify they contain the answer

### MissingSchema error on embeddings

```
requests.exceptions.MissingSchema: Invalid URL
```

Ensure `EMBEDS_URL` in your `.env` starts with `https://` or `http://`. Example:
```env
EMBEDS_URL=https://your-modal-endpoint.modal.run/v1
```

### GROBID connection errors

Make sure your GROBID server is running and accessible:
```bash
curl https://grobid.hf.space/api/isalive
```

If using a local GROBID instance:
```bash
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0
# Then set GROBID_URL=http://localhost:8070
```

### Embedding API returning empty results

- Verify the API is running: `curl {EMBEDS_URL}/embeddings`
- Check that `EMBEDS_API_KEY` matches the server's expected key
- Ensure the URL does **not** have a trailing `/embeddings` (the client appends it automatically)

---

