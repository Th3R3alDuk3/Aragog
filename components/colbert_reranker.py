from logging import getLogger
from typing import Any

from haystack import Document, component

logger = getLogger(__name__)


@component
class ColBERTReranker:
    """
    ColBERT late-interaction second-pass reranker — Haystack component.

    Args:
        model_name: HuggingFace model ID (default: ``colbert-ir/colbertv2.0``).
        top_k:      Number of documents to return after reranking.
    """

    def __init__(self, 
        model_name: str = "colbert-ir/colbertv2.0", 
        top_k: int = 5, 
        device: str = "cpu"
    ) -> None:

        self.top_k = top_k
        self._model_name = model_name
        self._device = device
        self._model: Any = None   # lazy-loaded on first use

    def _load_model(self) -> None:
        """Load the ColBERT model lazily on first use.

        Raises:
            RuntimeError: If the ``pylate`` package is not installed.
        """
        if self._model is None:
            try:
                from pylate import models
                self._model = models.ColBERT(model_name_or_path=self._model_name, device=self._device)
                logger.info("ColBERT: loaded model '%s'", self._model_name)
            except ImportError as exc:
                raise RuntimeError(
                    "pylate is required for ColBERT reranking. "
                    "Install it with: uv add pylate"
                ) from exc

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document]) -> dict[str, list[Document]]:
        """Re-score documents against the query using ColBERT late interaction.

        Falls back to the cross-encoder input order on any error so the endpoint
        never fails due to a ColBERT issue.

        Args:
            query:     The user query string.
            documents: Candidate documents from the upstream cross-encoder reranker.

        Returns:
            Dict with ``documents`` key — at most ``top_k`` docs sorted by ColBERT
            MaxSim score, or the input list (truncated to ``top_k``) if reranking fails.
        """
        if not documents:
            return {"documents": documents}

        try:
            self._load_model()
            return {"documents": self._rerank_internal(query, documents)}
        except Exception as exc:
            logger.warning(
                "ColBERT reranking failed (%s), keeping cross-encoder order", exc
            )
            return {"documents": documents[: self.top_k]}

    def _rerank_internal(self, query: str, documents: list) -> list:
        """Encode query and documents with ColBERT and return top-k by MaxSim score.

        Args:
            query:     The user query string.
            documents: Candidate documents to rerank.

        Returns:
            Top-k documents sorted by descending ColBERT score.
        """
        from pylate import rank

        doc_texts = [doc.content or "" for doc in documents]

        query_embeddings = self._model.encode(
            [query],
            is_query        = True,
            show_progress_bar = False,
        )
        doc_embeddings = self._model.encode(
            doc_texts,
            is_query        = False,
            show_progress_bar = False,
        )

        # rank.rerank returns list[list[dict]] — one list per query
        # Each dict has keys "id" (0-based doc index) and "score"
        ranked = rank.rerank(
            documents_ids       = list(range(len(documents))),
            queries_embeddings  = query_embeddings,
            documents_embeddings= doc_embeddings,
        )
        ranked_indices = [entry["id"] for entry in ranked[0]]
        return [documents[i] for i in ranked_indices][: self.top_k]
