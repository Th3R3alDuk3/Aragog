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

        Acts as a fast pre-filter (RRF output → top_k) before the cross-encoder
        reranker.  Falls back to the input order on any error so the endpoint
        never fails due to a ColBERT issue.

        Args:
            query:     The user query string.
            documents: Candidate documents from the upstream RRF joiner.

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
                "ColBERT reranking failed (%s), keeping upstream order", exc
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

        # rank.rerank expects nested lists — one inner list per query.
        # Each result dict has keys "id" (0-based doc index) and "score".
        ranked = rank.rerank(
            documents_ids        = [list(range(len(documents)))],
            queries_embeddings   = query_embeddings,
            documents_embeddings = [doc_embeddings],
        )
        n_tokens = max(len(query_embeddings[0]), 1)
        result = []
        for entry in ranked[0][: self.top_k]:
            doc = documents[entry["id"]]
            doc.score = min(1.0, max(0.0, entry["score"] / n_tokens))
            result.append(doc)
        return result
