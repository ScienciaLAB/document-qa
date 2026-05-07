"""Core Q/A engine for scientific PDF documents.

This module provides the main classes for building a Retrieval-Augmented
Generation (RAG) pipeline over scientific PDFs.
"""

import copy
import os
from pathlib import Path
from typing import Union, Any, List

import tiktoken
from langchain.chains import create_extraction_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.question_answering import stuff_prompt, refine_prompts, map_reduce_prompt, \
    map_rerank_prompt
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.vectorstores import VectorStore
from tqdm import tqdm

from document_qa.grobid_processors import GrobidProcessor
from document_qa.langchain import ChromaAdvancedRetrieval


class TextMerger:
    """Token-aware text merger that preserves PDF coordinate metadata.

    Unlike LangChain's ``RecursiveTextSplitter``, this merger keeps the
    bounding-box coordinates extracted by GROBID so that downstream
    consumers (e.g. the PDF viewer) can highlight the exact regions.

    Args:
        model_name: A tiktoken model name (e.g. ``"gpt-4"``).  When given,
            the tokenizer for that model is used.  
        encoding_name: A tiktoken encoding name (default ``"gpt2"``).
            Ignored when *model_name* is provided.
    """

    def __init__(self, model_name=None, encoding_name="gpt2"):
        if model_name is not None:
            self.enc = tiktoken.encoding_for_model(model_name)
        else:
            self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text, allowed_special=set(), disallowed_special="all"):
        """Tokenize *text* and return a list of token IDs.

        Thin wrapper around ``tiktoken.Encoding.encode`` that exposes the
        same special-token controls.

        Args:
            text: The string to tokenize.
            allowed_special: Set of special tokens allowed in *text*.
            disallowed_special: Special-token handling policy.

        Returns:
            list[int]: Token IDs produced by the configured tokenizer.
        """
        return self.enc.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )

    def merge_passages(self, passages, chunk_size, tolerance=0.2):
        """Merge consecutive passages into chunks of approximately *chunk_size* tokens.

        Args:
            passages: List of dicts, each with ``"text"`` (str) and
                ``"coordinates"`` (str) keys — as returned by
                method:`GrobidProcessor.process_structure`.
            chunk_size: Target number of tokens per merged chunk.
            tolerance: Fraction of *chunk_size* allowed as overflow
                (default ``0.2``).

        Returns:
            list[dict]: Merged passages.  Each dict has:

            - ``"text"`` — concatenated paragraph texts.
            - ``"coordinates"`` — semicolon-joined coordinate strings.
            - ``"type"`` — always ``"aggregated chunks"``.
            - ``"section"`` / ``"subSection"`` — always ``"mixed"``.
        """
        new_passages = []
        new_coordinates = []
        current_texts = []
        current_coordinates = []
        for idx, passage in enumerate(passages):
            text = passage['text']
            coordinates = passage['coordinates']
            current_texts.append(text)
            current_coordinates.append(coordinates)

            accumulated_text = " ".join(current_texts)

            encoded_accumulated_text = self.encode(accumulated_text)

            if len(encoded_accumulated_text) > chunk_size + chunk_size * tolerance:
                if len(current_texts) > 1:
                    new_passages.append(current_texts[:-1])
                    new_coordinates.append(current_coordinates[:-1])
                    current_texts = [current_texts[-1]]
                    current_coordinates = [current_coordinates[-1]]
                else:
                    new_passages.append(current_texts)
                    new_coordinates.append(current_coordinates)
                    current_texts = []
                    current_coordinates = []

            elif chunk_size <= len(encoded_accumulated_text) < chunk_size + chunk_size * tolerance:
                new_passages.append(current_texts)
                new_coordinates.append(current_coordinates)
                current_texts = []
                current_coordinates = []

        if len(current_texts) > 0:
            new_passages.append(current_texts)
            new_coordinates.append(current_coordinates)

        new_passages_struct = []
        for i, passages in enumerate(new_passages):
            text = " ".join(passages)
            coordinates = ";".join(new_coordinates[i])

            new_passages_struct.append(
                {
                    "text": text,
                    "coordinates": coordinates,
                    "type": "aggregated chunks",
                    "section": "mixed",
                    "subSection": "mixed"
                }
            )

        return new_passages_struct


class BaseRetrieval:
    """Abstract base for retrieval backends.
    """

    def __init__(
            self,
            persist_directory: Path,
            embedding_function
    ):
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory


class NER_Retrival(VectorStore):
    """
    This class implement a retrieval based on NER models.
    This is an alternative retrieval to embeddings that relies on extracted entities.
    """
    pass


engines = {
    'chroma': ChromaAdvancedRetrieval,
    'ner': NER_Retrival
}


class DataStorage:
    """Manages per-document vector-store collections.

    Each uploaded PDF gets its own ChromaDB collection,
    keyed by a document ID (typically an MD5 hash).  Collections can live
    in memory or be persisted to disk.

    Args:
        embedding_function: A LangChain-compatible ``Embeddings`` instance
        root_path: Optional directory for persisted embeddings. 
        engine: The vector-store class to use.

    """

    embeddings_dict = {}
    embeddings_map_from_md5 = {}
    embeddings_map_to_md5 = {}

    def __init__(
            self,
            embedding_function,
            root_path: Path = None,
            engine=ChromaAdvancedRetrieval,
    ) -> None:
        self.root_path = root_path
        self.engine = engine
        self.embedding_function = embedding_function

        if root_path is not None:
            self.embeddings_root_path = root_path
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            else:
                self.load_embeddings(self.embeddings_root_path)

    def load_embeddings(self, embeddings_root_path: Union[str, Path]) -> None:
        """
        Load the vector storage assuming they are all persisted and stored in a single directory.
        The root path of the embeddings containing one data store for each document in each subdirectory
        """

        embeddings_directories = [f for f in os.scandir(embeddings_root_path) if f.is_dir()]

        if len(embeddings_directories) == 0:
            print("No available embeddings")
            return

        for embedding_document_dir in embeddings_directories:
            self.embeddings_dict[embedding_document_dir.name] = self.engine(
                persist_directory=embedding_document_dir.path,
                embedding_function=self.embedding_function
            )

            filename_list = list(Path(embedding_document_dir).glob('*.storage_filename'))
            if filename_list:
                filenam = filename_list[0].name.replace(".storage_filename", "")
                self.embeddings_map_from_md5[embedding_document_dir.name] = filenam
                self.embeddings_map_to_md5[filenam] = embedding_document_dir.name

        print("Embedding loaded: ", len(self.embeddings_dict.keys()))

    def get_loaded_embeddings_ids(self):
        """Return the document IDs (MD5 hashes) of all loaded collections."""
        return list(self.embeddings_dict.keys())

    def get_md5_from_filename(self, filename):
        """Look up the MD5 document ID for a given original *filename*."""
        return self.embeddings_map_to_md5[filename]

    def get_filename_from_md5(self, md5):
        """Look up the original filename for a given *md5* document ID."""
        return self.embeddings_map_from_md5[md5]

    def embed_document(self, doc_id, texts, metadatas):
        """Create (or replace) an in-memory vector collection for a document.

        Args:
            doc_id: Unique identifier for the document.
            texts: List of text chunks to embed.
            metadatas: List of metadata dicts (one per chunk).
        """
        if doc_id not in self.embeddings_dict.keys():
            self.embeddings_dict[doc_id] = self.engine.from_texts(
                texts,
                embedding=self.embedding_function,
                metadatas=metadatas,
                collection_name=doc_id)
        else:
            # Workaround Chroma (?) breaking change
            self.embeddings_dict[doc_id].delete_collection()
            self.embeddings_dict[doc_id] = self.engine.from_texts(
                texts,
                embedding=self.embedding_function,
                metadatas=metadatas,
                collection_name=doc_id)

        self.embeddings_root_path = None


class DocumentQAEngine:
    """End-to-end RAG engine for scientific PDF documents.

    Orchestrates the full pipeline:

    1. **PDF parsing** via a GROBID server (structured text + coordinates).
    2. **Chunking** — paragraphs kept as-is or merged with :class:`TextMerger`.
    3. **Embedding and storage**  chunks are embedded and stored.
    4. **Retrieval + LLM** — relevant chunks are retrieved and fed to an LLM
       to produce an answer.

    Args:
        llm: A LangChain chat model (e.g. ``ChatOpenAI``).
        data_storage: A `DataStorage` instance for managing embeddings.
        grobid_url: URL of the GROBID server. 
        memory: Optional ``ConversationBufferMemory`` for multi-turn context.

    """

    llm = None
    qa_chain_type = None

    default_prompts = {
        'stuff': stuff_prompt,
        'refine': refine_prompts,
        "map_reduce": map_reduce_prompt,
        "map_rerank": map_rerank_prompt
    }

    def __init__(self,
                 llm,
                 data_storage: DataStorage,
                 grobid_url=None,
                 memory=None
                 ):

        self.llm = llm
        self.memory = memory
        self.chain = create_stuff_documents_chain(llm, self.default_prompts['stuff'].PROMPT)
        self.text_merger = TextMerger()
        self.data_storage = data_storage

        if grobid_url:
            self.grobid_processor = GrobidProcessor(grobid_url)

    def query_document(
            self,
            query: str,
            doc_id,
            output_parser=None,
            context_size=4,
            extraction_schema=None,
            verbose=False
    ) -> tuple[Any, str]:
        """Ask a question and get an LLM-generated answer.

        Retrieves the most relevant chunks from the vector store, feeds
        them as context to the LLM, and returns the response.

        Args:
            query: The natural-language question.
            doc_id: Document identifier returned by create_memory_embeddings`.
            output_parser: Optional LangChain output parser.  If provided the
                raw LLM response is re-processed into structured output.
            context_size: Number of chunks to retrieve as context (default 4).
            extraction_schema: Optional extraction schema.
            verbose: Print debug information.

        Returns:
            tuple: ``(parsed_output | None, raw_text_response, coordinates)``

            - *parsed_output* — structured data if a parser/schema was given,
              otherwise ``None``.
            - *raw_text_response* — the LLM's raw text answer.
            - *coordinates* — list of lists of coordinate strings for each
              retrieved chunk (for PDF highlighting).
        """
        # self.load_embeddings(self.embeddings_root_path)

        if verbose:
            print(query)

        response, coordinates = self._run_query(doc_id, query, context_size=context_size)
        response = response['output_text'] if 'output_text' in response else response

        if verbose:
            print(doc_id, "->", response)

        if output_parser:
            try:
                return self._parse_json(response, output_parser), response
            except Exception as oe:
                print("Failing to parse the response", oe)
                return None, response, coordinates
        elif extraction_schema:
            try:
                chain = create_extraction_chain(extraction_schema, self.llm)
                parsed = chain.run(response)
                return parsed, response, coordinates
            except Exception as oe:
                print("Failing to parse the response", oe)
                return None, response, coordinates
        else:
            return None, response, coordinates

    def query_storage(self, query: str, doc_id, context_size=4) -> tuple[List[Document], list]:
        """Retrieve relevant text passages without calling the LLM.

        Useful for debugging which chunks would be used as context, or for
        building custom pipelines on top of the retrieval step.

        Args:
            query: The natural-language question.
            doc_id: Document identifier.
            context_size: Number of chunks to retrieve (default 4).

        Returns:
            tuple: ``(texts, coordinates)``

            - *texts* — list of passage strings.
            - *coordinates* — list of lists of coordinate strings.
        """
        documents, coordinates = self._get_context(doc_id, query, context_size)

        context_as_text = [doc.page_content for doc in documents]
        return context_as_text, coordinates

    def query_storage_and_embeddings(self, query: str, doc_id, context_size=4) -> List[Document]:
        """Retrieve passages with their similarity scores and raw embeddings.

        Each returned ``Document`` has extra metadata keys:

        - ``__similarity`` — cosine distance to the query.
        - ``__embeddings`` — the chunk's embedding vector.

        Args:
            query: The natural-language question.
            doc_id: Document identifier.
            context_size: Number of chunks to retrieve (default 4).

        Returns:
            list[Document]: Retrieved documents enriched with similarity and
            embedding metadata.
        """
        db = self.data_storage.embeddings_dict[doc_id]
        retriever = db.as_retriever(
            search_kwargs={"k": context_size},
            search_type="similarity_with_embeddings"
        )
        relevant_documents = retriever.invoke(query)

        return relevant_documents

    def analyse_query(self, query, doc_id, context_size=4):
        """Compute a relevance coefficient for *query* against *doc_id*.

        The coefficient is ``min_similarity - mean_similarity`` over the
        top-k retrieved chunks.  A value close to zero suggests the
        question matches multiple passages equally well.

        Args:
            query: The natural-language question.
            doc_id: Document identifier.
            context_size: Number of chunks to consider (default 4).

        Returns:
            tuple: ``(summary_string, coordinates)``
        """
        db = self.data_storage.embeddings_dict[doc_id]
        # retriever = db.as_retriever(
        #     search_kwargs={"k": context_size, 'score_threshold': 0.0},
        #     search_type="similarity_score_threshold"
        # )
        retriever = db.as_retriever(search_kwargs={"k": context_size}, search_type="similarity_with_embeddings")
        relevant_documents = retriever.invoke(query)
        relevant_document_coordinates = [doc.metadata['coordinates'].split(";") if 'coordinates' in doc.metadata else []
                                         for doc in
                                         relevant_documents]
        all_documents = db.get(include=['documents', 'metadatas', 'embeddings'])
        # all_documents_embeddings = all_documents["embeddings"]
        # query_embedding = db._embedding_function.embed_query(query)

        # distance_evaluator = load_evaluator("pairwise_embedding_distance",
        #                               embeddings=db._embedding_function,
        #                               distance_metric=EmbeddingDistance.EUCLIDEAN)

        # distance_evaluator.evaluate_string_pairs(query=query_embedding, documents="")

        similarities = [doc.metadata['__similarity'] for doc in relevant_documents]
        min_similarity = min(similarities)
        mean_similarity = sum(similarities) / len(similarities)
        coefficient = min_similarity - mean_similarity

        return f"Coefficient: {coefficient}, (Min similarity {min_similarity}, Mean similarity: {mean_similarity})", relevant_document_coordinates

    def _parse_json(self, response, output_parser):
        system_message = "You are an useful assistant expert in materials science, physics, and chemistry " \
                         "that can process text and transform it to JSON."
        human_message = """Transform the text between three double quotes in JSON.\n\n\n\n
        {format_instructions}\n\nText: \"\"\"{text}\"\"\""""

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_message)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_message)

        prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        results = self.llm(
            prompt_template.format_prompt(
                text=response,
                format_instructions=output_parser.get_format_instructions()
            ).to_messages()
        )
        parsed_output = output_parser.parse(results.content)

        return parsed_output

    def _run_query(self, doc_id, query, context_size=4) -> tuple[List[Document], list]:
        relevant_documents, relevant_document_coordinates = self._get_context(doc_id, query, context_size)
        response = self.chain.invoke({"context": relevant_documents, "question": query})
        return response, relevant_document_coordinates

    def _get_context(self, doc_id, query, context_size=4) -> tuple[List[Document], list]:
        db = self.data_storage.embeddings_dict[doc_id]
        retriever = db.as_retriever(search_kwargs={"k": context_size})
        relevant_documents = retriever.invoke(query)
        relevant_document_coordinates = [
            doc.metadata['coordinates'].split(";") if 'coordinates' in doc.metadata else []
            for doc in
            relevant_documents
        ]
        if self.memory and len(self.memory.buffer_as_messages) > 0:
            relevant_documents.append(
                Document(
                    page_content="""Following, the previous question and answers. Use these information only when in the question there are unspecified references:\n{}\n\n""".format(
                        self.memory.buffer_as_str))
            )
        return relevant_documents, relevant_document_coordinates

    def get_full_context_by_document(self, doc_id):
        """
        Return the full context from the document
        """
        db = self.data_storage.embeddings_dict[doc_id]
        docs = db.get()
        return docs['documents']

    def _get_context_multiquery(self, doc_id, query, context_size=4):
        db = self.data_storage.embeddings_dict[doc_id].as_retriever(search_kwargs={"k": context_size})
        multi_query_retriever = MultiQueryRetriever.from_llm(retriever=db, llm=self.llm)
        relevant_documents = multi_query_retriever.invoke(query)
        return relevant_documents

    def get_text_from_document(self, pdf_file_path, chunk_size=-1, perc_overlap=0.1, verbose=False):
        """Extract and chunk text from a PDF via GROBID.

        Sends the PDF to the configured GROBID server, parses the returned
        TEI-XML into passages with coordinate metadata, and optionally
        merges passages into larger token-based chunks.

        Args:
            pdf_file_path: Path to the PDF file on disk.
            chunk_size: Target tokens per chunk.  ``-1`` (default) keeps
                GROBID paragraphs as-is; a positive value merges them.
            perc_overlap: Reserved for future overlap support.
            verbose: Print debug information.

        Returns:
            tuple: ``(texts, metadatas, ids)``

            - *texts* — list of passage strings.
            - *metadatas* — list of metadata dicts (coordinates, section, …).
            - *ids* — list of integer chunk IDs.

        Raises:
            AttributeError: If ``grobid_url`` was not provided at init time.
        """
        if verbose:
            print("File", pdf_file_path)
        filename = Path(pdf_file_path).stem
        coordinates = True  # if chunk_size == -1 else False
        structure = self.grobid_processor.process_structure(pdf_file_path, coordinates=coordinates)

        biblio = structure['biblio']
        biblio['filename'] = filename.replace(" ", "_")

        if verbose:
            print("Generating embeddings for:", hash, ", filename: ", filename)

        texts = []
        metadatas = []
        ids = []

        if chunk_size > 0:
            new_passages = self.text_merger.merge_passages(structure['passages'], chunk_size=chunk_size)
        else:
            new_passages = structure['passages']

        for passage in new_passages:
            biblio_copy = copy.copy(biblio)
            if len(str.strip(passage['text'])) > 0:
                texts.append(passage['text'])

                biblio_copy['type'] = passage['type']
                biblio_copy['section'] = passage['section']
                biblio_copy['subSection'] = passage['subSection']
                biblio_copy['coordinates'] = passage['coordinates']
                metadatas.append(biblio_copy)

                # ids.append(passage['passage_id'])

            ids = [id for id, t in enumerate(new_passages)]

        return texts, metadatas, ids

    def create_memory_embeddings(
            self,
            pdf_path,
            doc_id=None,
            chunk_size=500,
            perc_overlap=0.1
    ):
        """Parse a PDF and create an in-memory vector collection.

        This is the main entry-point for ingesting a new document.  It
        calls GROBID, chunks the text, embeds it, and stores everything in `data_storage`.

        Args:
            pdf_path: Path to the PDF file.
            doc_id: Optional explicit document ID.  When ``None``, the
                MD5 hash extracted by GROBID is used.
            chunk_size: Token count per chunk (default 500).  Use ``-1``
                to keep GROBID paragraphs intact.
            perc_overlap: Reserved for future overlap support.

        Returns:
            str: The document ID.
        """
        texts, metadata, ids = self.get_text_from_document(
            pdf_path,
            chunk_size=chunk_size,
            perc_overlap=perc_overlap)
        if doc_id:
            hash = doc_id
        else:
            hash = metadata[0]['hash'] if len(metadata) > 0 and 'hash' in metadata[0] else ""

        self.data_storage.embed_document(hash, texts, metadata)

        return hash

    def create_embeddings(
            self,
            pdfs_dir_path: Path,
            chunk_size=500,
            perc_overlap=0.1,
            include_biblio=False
    ):
        """Batch-process a directory of PDFs and persist their embeddings.

        Walks *pdfs_dir_path*, processes each ``.pdf`` file through GROBID,
        creates embeddings, and persists the resulting ChromaDB collection
        to a subdirectory named after the file's MD5.

        Args:
            pdfs_dir_path: Directory containing PDF files.
            chunk_size: Token count per chunk (default 500).
            perc_overlap: Reserved for future overlap support.
            include_biblio: Reserved flag (currently unused).
        """
        input_files = []
        for root, dirs, files in os.walk(pdfs_dir_path, followlinks=False):
            for file_ in files:
                if not (file_.lower().endswith(".pdf")):
                    continue
                input_files.append(os.path.join(root, file_))

        for input_file in tqdm(input_files, total=len(input_files), unit='document',
                               desc="Grobid + embeddings processing"):

            md5 = self.calculate_md5(input_file)
            data_path = os.path.join(self.data_storage.embeddings_root_path, md5)

            if os.path.exists(data_path):
                print(data_path, "exists. Skipping it ")
                continue
            # include = ["biblio"] if include_biblio else []
            texts, metadata, ids = self.get_text_from_document(
                input_file,
                chunk_size=chunk_size,
                perc_overlap=perc_overlap)
            filename = metadata[0]['filename']

            vector_db_document = Chroma.from_texts(texts,
                                                   metadatas=metadata,
                                                   embedding=self.embedding_function,
                                                   persist_directory=data_path)
            vector_db_document.persist()

            with open(os.path.join(data_path, filename + ".storage_filename"), 'w') as fo:
                fo.write("")

    @staticmethod
    def calculate_md5(input_file: Union[Path, str]):
        """Return the uppercase hex MD5 digest of *input_file*."""

        import hashlib
        md5_hash = hashlib.md5()
        with open(input_file, 'rb') as fi:
            md5_hash.update(fi.read())
        return md5_hash.hexdigest().upper()
