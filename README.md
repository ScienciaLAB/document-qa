---
title: Scientific Document Insights Q/A
emoji: 📝
colorFrom: yellow
colorTo: pink
sdk: streamlit
sdk_version: 1.37.1
app_file: streamlit_app.py
pinned: false
license: apache-2.0
app_port: 8501
---

# DocumentIQA: Scientific Document Insights Q/A

**Work in progress** :construction_worker: 

<img src="https://github.com/lfoppiano/document-qa/assets/15426/f0a04a86-96b3-406e-8303-904b93f00015" width=300 align="right" />

https://lfoppiano-document-qa.hf.space/

**NOTE**: The LLM API is kindly provided by [Modal.com](https://www.modal.com) which offers 30$/month for computing. When these are done, the app will stop answering. 😅

## Introduction

Question/Answering on scientific documents using LLMs. The tool can be customized to use different types of LLM APIs.
The streamlit application demonstrates the implementation of a RAG (Retrieval Augmented Generation) on scientific documents.
**Different from most of the projects**, we focus on scientific articles and extract text from a structured document. 
We target only the full text using [Grobid](https://github.com/kermitt2/grobid) which provides cleaner results than the raw PDF2Text converter (which is comparable with most of the other solutions).

Additionally, this frontend provides the visualisation of named entities on LLM responses to extract <span stype="color:yellow">physical quantities, measurements</span> (with [grobid-quantities](https://github.com/kermitt2/grobid-quantities)) and <span stype="color:blue">materials</span> mentions (with [grobid-superconductors](https://github.com/lfoppiano/grobid-superconductors)).

(The image on the right was generated with https://huggingface.co/spaces/stabilityai/stable-diffusion)

[<img src="https://img.youtube.com/vi/M4UaYs5WKGs/hqdefault.jpg" height="300" align="right" 
/>](https://www.youtube.com/embed/M4UaYs5WKGs)

## Getting started

- Upload a scientific article as a PDF document. You will see a spinner or loading indicator while the processing is in progress. 
- Once the spinner disappears, you can proceed to ask your questions

 ![screenshot2.png](docs%2Fimages%2Fscreenshot2.png)

## Documentation

 **For full technical documentation** of the `document-qa-engine` library **[`docs/README.md`](docs/README.md)**.

### Embedding selection
In the latest version, there is the possibility to select both embedding functions and LLMs. There are some limitations, OpenAI embeddings cannot be used with open source models, and vice-versa. 

### Context size
Allow to change the number of blocks from the original document that are considered for responding. 
The default size of each block is 250 tokens (which can be changed before uploading the first document). 
With default settings, each question uses around 1000 tokens.

**NOTE**: if the chat answers something like "the information is not provided in the given context", **changing the context size will likely help**. 

### Chunks size
When uploaded, each document is split into blocks of a determined size (250 tokens by default). 
This setting allows users to modify the size of such blocks. 
Smaller blocks will result in a smaller context, yielding more precise sections of the document. 
Larger blocks will result in a larger context less constrained around the question.

### Query mode
Indicates whether sending a question to the LLM (Language Model) or the vector storage. 
 - **LLM** (default) enables question/answering related to the document content.
 - **Embeddings**: the response will consist of the raw text from the document related to the question (based on the embeddings). This mode helps to test why sometimes the answers are not satisfying or incomplete.
 - **Question coefficient** (experimental): provide a coefficient that indicates how the question has been far or closed to the retrieved context

### NER (Named Entities Recognition)
This feature is specifically crafted for people working with scientific documents in materials science. 
It enables to run NER on the response from the LLM, to identify materials mentions and properties (quantities, measurements).
This feature leverages both [grobid-quantities](https://github.com/kermitt2/grobid-quanities) and [grobid-superconductors](https://github.com/lfoppiano/grobid-superconductors) external services. 

### Troubleshooting
Error: `streamlit: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0`.
Here is the [solution on Linux](https://stackoverflow.com/questions/76958817/streamlit-your-system-has-an-unsupported-version-of-sqlite3-chroma-requires-sq).
For more information, see the [details](https://docs.trychroma.com/troubleshooting#sqlite) on the Chroma website.

## Disclaimer on Data, Security, and Privacy ⚠️

Please read carefully:

- Avoid uploading sensitive data. We temporarily store text from the uploaded PDF documents only for processing your request, and we disclaim any responsibility for subsequent use or handling of the submitted data by third-party LLMs.
- Mistral and Zephyr are FREE to use and do not require any API, but as we leverage the free API entrypoint, there is no guarantee that all requests will go through. Use at your own risk.
- We do not assume responsibility for how the data is utilized by the LLM end-points API.

## Development notes

To release a new version: 

- `bump-my-version bump patch` 
- `git push --tags`

To use docker: 

- docker run `lfoppiano/document-insights-qa:{latest_version}`

- docker run `lfoppiano/document-insights-qa:latest-develop` for the latest development version 

To install the library with Pypi: 

- `pip install document-qa-engine` 


## Acknowledgement 

The project was initiated at the [National Institute for Materials Science](https://www.nims.go.jp) (NIMS) in Japan. 
Currently, the development is possible thanks to [ScienciLAB](https://www.sciencialab.com).
This project was contributed by [Guillaume Lambard](https://github.com/GLambard) and the [Lambard-ML-Team](https://github.com/Lambard-ML-Team), [Pedro Ortiz Suarez](https://github.com/pjox), and [Tomoya Mato](https://github.com/t29mato).
Thanks also to [Patrice Lopez](https://www.science-miner.com), the author of [Grobid](https://github.com/kermitt2/grobid).




