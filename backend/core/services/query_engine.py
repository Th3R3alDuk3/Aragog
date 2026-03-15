import logging

from haystack.dataclasses import GeneratedAnswer

from core.config import Settings
from core.models.query import QueryInput, QueryResult
from core.services.retrieval_engine import NoDocumentsFoundError, RetrievalEngine

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Raised when answer generation fails or returns no answer."""


class QueryEngine:
    def __init__(
        self,
        settings: Settings,
        generation_pipeline,
        retrieval_engine: RetrievalEngine,
    ) -> None:
        self.settings = settings
        self.generation_pipeline = generation_pipeline
        self.retrieval_engine = retrieval_engine

    async def query(self, request: QueryInput) -> QueryResult:
        ctx = await self.retrieval_engine.prepare(request)
        if not ctx.merged_docs:
            raise NoDocumentsFoundError("No relevant documents found for your query.")

        logger.info("Generate → %d doc(s) as context...", len(ctx.merged_docs))

        try:
            answers = await run_generation(
                self.generation_pipeline,
                ctx.merged_docs,
                ctx.sub_questions,
                request.query,
            )
        except Exception as error:
            raise GenerationError("Answer generation failed.") from error

        if not answers:
            raise GenerationError("LLM returned no answer.")

        answer = answers[0]
        sources = self.retrieval_engine.format_source_docs(ctx.source_docs[: request.top_k])

        logger.info(
            "Done  → answer=%d chars | sources=%d%s",
            len(answer.data or ""),
            len(sources),
            " | low_confidence=true" if ctx.low_confidence else "",
        )

        return QueryResult(
            answer=answer.data or "",
            sources=sources,
            query=request.query,
            sub_questions=ctx.sub_questions if ctx.is_compound else [],
            is_compound=ctx.is_compound,
            low_confidence=ctx.low_confidence,
            extracted_filters=self.retrieval_engine.analysis_to_filter_dict(ctx.analysis),
        )


async def run_generation(
    pipeline,
    documents: list,
    questions: list[str],
    query: str,
) -> list[GeneratedAnswer]:
    result = await pipeline.run_async(
        {
            "prompt_builder": {"documents": documents, "questions": questions},
            "answer_builder": {"query": query, "documents": documents},
        }
    )
    return result.get("answer_builder", {}).get("answers", [])
