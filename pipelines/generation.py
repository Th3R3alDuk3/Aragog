"""
Generation pipeline — Haystack 2.x PromptBuilder + OpenAIGenerator

Assembles the LLM generation stage separately from retrieval so the router
can run retrieval once per sub-question and call the LLM exactly once with
the merged context from all sub-questions.

Pipeline flow
─────────────
  documents + questions (runtime inputs)
          │
    PromptBuilder  (RAG_PROMPT template)
          │
    OpenAIGenerator
          │
    AnswerBuilder
"""

from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.core.pipeline.async_pipeline import AsyncPipeline

from config import Settings
from pipelines._factories import build_generator

# ---------------------------------------------------------------------------
# RAG prompt — multi-question aware
# ---------------------------------------------------------------------------

RAG_PROMPT = """\
You are a precise, helpful assistant. Answer the question(s) below using ONLY
the provided context documents.  If the context does not contain sufficient
information to answer a question, state that clearly instead of guessing.
Do not invent facts.

{% if questions | length > 1 %}
The user asked multiple questions. Address each one separately and clearly.

Questions:
{% for q in questions %}
  {{ loop.index }}. {{ q }}
{% endfor %}
{% else %}
Question: {{ questions[0] }}
{% endif %}

Context documents:
{% for doc in documents %}
────────────────────────────────────────
Source  : {{ doc.meta.get("source", "unknown") }}
Section : {{ doc.meta.get("section_title", "") }}
{% if doc.meta.get("summary") %}Summary : {{ doc.meta.get("summary") }}
{% endif %}
{{ doc.content }}
{% endfor %}
────────────────────────────────────────

Answer:"""


def build_generation_pipeline(settings: Settings) -> AsyncPipeline:
    """Build the LLM generation pipeline.

    The pipeline accepts ``documents`` and ``questions`` as runtime inputs and
    produces a generated answer via the configured OpenAI-compatible LLM.

    Args:
        settings: Application settings used to configure the LLM generator.

    Returns:
        A Haystack ``AsyncPipeline`` with components ``prompt_builder``, ``llm``,
        and ``answer_builder`` wired in sequence.
    """
    prompt_builder = PromptBuilder(
        template=RAG_PROMPT,
        required_variables=["questions", "documents"],
    )
    generator      = build_generator(settings)
    answer_builder = AnswerBuilder()

    generation = AsyncPipeline()
    generation.add_component("prompt_builder", prompt_builder)
    generation.add_component("llm",            generator)
    generation.add_component("answer_builder", answer_builder)
    generation.connect("prompt_builder.prompt", "llm.prompt")
    generation.connect("llm.replies",           "answer_builder.replies")
    generation.connect("llm.meta",              "answer_builder.meta")

    return generation
