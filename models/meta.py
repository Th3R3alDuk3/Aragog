from pydantic import BaseModel, Field


class Meta(BaseModel):
    context: str = Field(
        default="",
        description=(
            "2-3 full sentences that situate this chunk within the overall "
            "document AND state its key point: use the document title and the "
            "heading path at the start of the chunk to name what the chunk is "
            "about, then summarize what the chunk actually says (e.g. 'This "
            "section of the Product X manual covers the warranty: 24 months "
            "from delivery, wearing parts excluded.'). Do not simply repeat the "
            "heading or title — write complete sentences describing the "
            "chunk's content. Written in the chunk's language to improve "
            "retrieval."
        ),
    )
    keywords: list[str] = Field(
        default=[],
        description=(
            "3-8 single words or short noun phrases (max 3 words each) "
            "capturing the most salient domain terms and named concepts as "
            "they appear in the text (e.g. ['Garantiezeit', 'Lieferdatum']), "
            "excluding names already captured as entities. Never full "
            "sentences. Empty list if none stand out."
        ),
    )
    hypothetical_questions: list[str] = Field(
        default=[],
        description=(
            "2-3 distinct questions a user might ask that this chunk "
            "directly answers, phrased naturally as a real query (e.g. "
            "['Wie lange gilt die Garantie auf Produkt X?']). Written in the "
            "chunk's language to improve retrieval. Empty list if the chunk "
            "answers no clear question."
        ),
    )
    dates: list[str] = Field(
        default=[],
        description=(
            "All distinct dates the chunk refers to, each normalized to ISO format "
            "YYYY-MM-DD (e.g. ['2023-04-01', '2024-12-31']). Empty list if the chunk "
            "states no clear date."
        ),
    )
    ent_persons: list[str] = Field(
        default=[],
        description=(
            "Full names of people explicitly mentioned in this chunk, as written "
            "(e.g. ['Angela Merkel']). Empty list if none."
        ),
    )
    ent_organizations: list[str] = Field(
        default=[],
        description=(
            "Organizations mentioned in this chunk — companies, agencies, institutions "
            "(e.g. ['Siemens AG', 'European Commission']). Empty list if none."
        ),
    )
    ent_products: list[str] = Field(
        default=[],
        description=(
            "Products, software or brands mentioned in this chunk "
            "(e.g. ['Windows 11', 'iPhone']). Empty list if none."
        ),
    )
    ent_locations: list[str] = Field(
        default=[],
        description=(
            "Geographic locations mentioned in this chunk — countries, cities, regions "
            "(e.g. ['Berlin', 'Bayern']). Empty list if none."
        ),
    )
