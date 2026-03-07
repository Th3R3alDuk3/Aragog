from pydantic import BaseModel, ConfigDict, Field

# ── Stage 2: MetadataEnricher — document identity ──────────────────────────


class DocMeta(BaseModel):
    """Document-level identity and provenance fields."""

    source: str = Field(default="", description="Original filename.")
    doc_id: str = Field(
        default="", description="SHA-256(source + content) — stable re-index key."
    )
    title: str = Field(default="", description="First H1 heading or filename stem.")
    word_count: int = Field(
        default=0, description="Approximate word count of the full document."
    )
    indexed_at: str = Field(
        default="", description="ISO-8601 UTC timestamp of indexing."
    )
    indexed_at_ts: int = Field(
        default=0, description="Unix epoch seconds — use for range filters."
    )
    language: str = Field(default="unknown", description="ISO 639-1 language code.")


# ── Stage 2: MetadataEnricher — embedding provenance ───────────────────────


class EmbeddingMeta(BaseModel):
    """Records which embedding model produced the vectors stored in Qdrant."""

    embedding_model: str = Field(
        default="", description="HuggingFace model id used for dense embeddings."
    )
    embedding_provider: str = Field(
        default="", description="Embedding backend (always 'huggingface')."
    )
    embedding_dimension: int = Field(default=0, description="Dense vector dimension.")


# ── Stage 2→6: ephemeral (stripped before writing to Qdrant) ───────────────


class EphemeralMeta(BaseModel):
    """Temporary fields that flow through the pipeline but are never stored."""

    doc_beginning: str = Field(
        default="",
        description=(
            "First N characters of the document — passed to ContentAnalyzer "
            "for LLM context, then excluded via model_dump(exclude={'doc_beginning'})."
        ),
    )


# ── Stage 4: HierarchicalDocumentSplitter (Haystack-internal) ──────────────
# Note: HierarchicalDocumentSplitter does NOT inject header/parent_headers.
# Heading context is extracted from chunk content by ChunkEnricher instead.


class HaystackMeta(BaseModel):
    """Placeholder for any future Haystack-injected fields at stage 4."""


# ── Stage 5: ChunkEnricher ─────────────────────────────────────────────────


class ChunkStructureMeta(BaseModel):
    """Chunk position and structural context within the document."""

    chunk_index: int = Field(
        default=0, description="0-based position of this chunk within its document."
    )
    chunk_total: int = Field(
        default=0, description="Total number of chunks in the source document."
    )
    section_title: str = Field(
        default="", description="Heading of the section this chunk belongs to."
    )
    section_path: str = Field(
        default="",
        description="Breadcrumb from root to section, e.g. 'Intro › Overview'.",
    )
    chunk_type: str = Field(
        default="text",
        description="Dominant content type: text | code | table | list | figure_caption.",
    )


# ── Stage 6: ContentAnalyzer ───────────────────────────────────────────────


class Entities(BaseModel):
    """
    Named-entity categories following the OntoNotes 5 taxonomy.

    Used only as the structured LLM output type in ``ChunkAnalysis``
    (content_analyzer.py).  Stored in Qdrant as flat ``ent_*`` fields on
    ``SemanticMeta`` so they can be used in ``meta_fields_to_embed``.
    """

    persons: list[str] = Field(default_factory=list, description="Full person names.")
    organizations: list[str] = Field(
        default_factory=list, description="Companies, agencies, institutions."
    )
    locations: list[str] = Field(
        default_factory=list, description="Countries, cities, regions."
    )
    dates: list[str] = Field(
        default_factory=list, description="All temporal expressions."
    )
    products: list[str] = Field(
        default_factory=list, description="Product names, software, brand names."
    )
    laws_and_standards: list[str] = Field(
        default_factory=list,
        description="Laws, regulations, norms (e.g. GDPR, ISO 9001, §17).",
    )
    events: list[str] = Field(
        default_factory=list,
        description="Named events, projects, incidents, conferences.",
    )
    quantities: list[str] = Field(
        default_factory=list,
        description="Monetary values, percentages, measurements with units.",
    )


class SemanticMeta(BaseModel):
    """AI-generated contextual and semantic metadata produced by ContentAnalyzer."""

    original_content: str = Field(
        default="",
        description=(
            "Raw chunk text before context_prefix is prepended "
            "(used for citation display)."
        ),
    )
    context_prefix: str = Field(
        default="",
        description="1-2 sentence LLM-generated context situating the chunk in the document.",
    )
    summary: str = Field(
        default="", description="2-3 sentence abstractive summary of the chunk."
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="5-10 key terms or phrases extracted by the LLM.",
    )
    classification: str = Field(
        default="general",
        description="Document category from the configured taxonomy.",
    )

    # ── Named entities — flat for meta_fields_to_embed + Qdrant indexing ──
    ent_persons: list[str] = Field(default_factory=list, description="Full person names.")
    ent_organizations: list[str] = Field(
        default_factory=list, description="Companies, agencies, institutions."
    )
    ent_locations: list[str] = Field(
        default_factory=list, description="Countries, cities, regions."
    )
    ent_dates: list[str] = Field(
        default_factory=list, description="All temporal expressions."
    )
    ent_products: list[str] = Field(
        default_factory=list, description="Product names, software, brand names."
    )
    ent_laws: list[str] = Field(
        default_factory=list,
        description="Laws, regulations, norms (e.g. GDPR, ISO 9001, §17).",
    )
    ent_events: list[str] = Field(
        default_factory=list,
        description="Named events, projects, incidents, conferences.",
    )
    ent_quantities: list[str] = Field(
        default_factory=list,
        description="Monetary values, percentages, measurements with units.",
    )


# ── Combined schema ────────────────────────────────────────────────────────


class ChunkMetadata(
    DocMeta,
    EmbeddingMeta,
    EphemeralMeta,
    HaystackMeta,
    ChunkStructureMeta,
    SemanticMeta,
):
    """
    Complete metadata schema for a stored document chunk.

    Combines all stage-specific sub-models in pipeline order.
    ``extra="allow"`` preserves any additional Haystack-internal fields (e.g. split_id).
    """

    model_config = ConfigDict(extra="allow")
