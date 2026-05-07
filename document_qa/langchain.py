"""LangChain vector store extensions for document-qa.

Extends ChromaDB with support for returning similarity scores **and**
raw embedding vectors alongside retrieved documents.  This enables
the Streamlit frontend to compute relevance gradients and the
``question_coefficient`` analysis mode.

"""

from typing import Any, Optional, List, Dict, Tuple, ClassVar, Collection

from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma, DEFAULT_K
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.utils import xor_args
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever


class AdvancedVectorStoreRetriever(VectorStoreRetriever):
    """Retriever that can enrich documents with similarity scores and embeddings.

    Extends LangChain's ``VectorStoreRetriever`` with a
    ``"similarity_with_embeddings"`` search type.  When used, each
    returned document's ``metadata`` dict gains ``__similarity`` (float)
    and ``__embeddings`` (list[float]) keys.
    """

    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
        "similarity_with_embeddings"
    )

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Fetch relevant documents for the configured search type.

        Supports all standard search types plus
        ``"similarity_with_embeddings"`` which attaches score and
        embedding vector metadata to each document.

        Args:
            query: The search query string.
            run_manager: LangChain callback manager.

        Returns:
            list[Document]: Retrieved documents, optionally enriched
            with similarity scores and embeddings.
        """

        if self.search_type == "similarity_with_embeddings":
            docs_scores_and_embeddings = (
                self.vectorstore.advanced_similarity_search(
                    query, **self.search_kwargs
                )
            )

            for doc, score, embeddings in docs_scores_and_embeddings:
                if '__embeddings' not in doc.metadata.keys():
                    doc.metadata['__embeddings'] = embeddings
                if '__similarity' not in doc.metadata.keys():
                    doc.metadata['__similarity'] = score

            docs = [doc for doc, _, _ in docs_scores_and_embeddings]
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            for doc, similarity in docs_and_similarities:
                if '__similarity' not in doc.metadata.keys():
                    doc.metadata['__similarity'] = similarity

            docs = [doc for doc, _ in docs_and_similarities]
        else:
            docs = super()._get_relevant_documents(query, run_manager=run_manager)

        return docs


class AdvancedVectorStore(VectorStore):
    """
    Extension of LangChain's VectorStore that returns a custom retriever
    supporting advanced search features.
    """

    def as_retriever(self, **kwargs: Any) -> AdvancedVectorStoreRetriever:
        """Create a retriever supporting ``similarity_with_embeddings``.

        Accepts the same keyword arguments as the base ``as_retriever``.
        """
        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return AdvancedVectorStoreRetriever(vectorstore=self, **kwargs, tags=tags)


class ChromaAdvancedRetrieval(Chroma, AdvancedVectorStore):
    """Chroma vector store with support for embeddings + similarity scores.

    Extends the standard LangChain ``Chroma`` store with
    `advanced_similarity_search` which returns ``(Document, score,
    embedding)`` triples.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @xor_args(("query_texts", "query_embeddings"))
    def __query_collection(
            self,
            query_texts: Optional[List[str]] = None,
            query_embeddings: Optional[List[List[float]]] = None,
            n_results: int = 4,
            where: Optional[Dict[str, str]] = None,
            where_document: Optional[Dict[str, str]] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Query the chroma collection."""
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )
        return self._collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            **kwargs,
        )

    def advanced_similarity_search(
            self,
            query: str,
            k: int = DEFAULT_K,
            filter: Optional[Dict[str, str]] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float, List[float]]]:
        """Return documents, similarity scores, and embeddings for *query*.

        Args:
            query: The search query.
            k: Number of results to return.
            filter: Optional Chroma metadata filter.

        Returns:
            list[tuple[Document, float, list[float]]]: Triples of
            (document, distance, embedding_vector).
        """
        docs_scores_and_embeddings = self.similarity_search_with_scores_and_embeddings(query, k, filter=filter)
        return docs_scores_and_embeddings

    def similarity_search_with_scores_and_embeddings(
            self,
            query: str,
            k: int = DEFAULT_K,
            filter: Optional[Dict[str, str]] = None,
            where_document: Optional[Dict[str, str]] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float, List[float]]]:
        """Low-level search returning docs with scores and embeddings.

        Queries the Chroma collection requesting ``distances`` and
        ``embeddings`` in addition to the usual documents and metadata.

        Args:
            query: The search query.
            k: Number of results.
            filter: Optional metadata filter.
            where_document: Optional document-content filter.

        Returns:
            list[tuple[Document, float, list[float]]]: Triples of
            (document, distance, embedding_vector).
        """

        if self._embedding_function is None:
            results = self.__query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
                include=['metadatas', 'documents', 'embeddings', 'distances']
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=where_document,
                include=['metadatas', 'documents', 'embeddings', 'distances']
            )

        return _results_to_docs_scores_and_embeddings(results)


def _results_to_docs_scores_and_embeddings(results: Any) -> List[Tuple[Document, float, List[float]]]:
    """Unpack raw Chroma query results into ``(Document, score, embedding)`` tuples.

    Args:
        results: Dict returned by ``Collection.query()`` with
            ``include=['documents', 'metadatas', 'distances', 'embeddings']``.

    Returns:
        list[tuple[Document, float, list[float]]]: One tuple per result.
    """
    return [
        (Document(page_content=result[0], metadata=result[1] or {}), result[2], result[3])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["embeddings"][0],
        )
    ]
