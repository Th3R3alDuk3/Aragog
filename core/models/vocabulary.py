from typing import Literal

LanguageCode = Literal[
    "ar",  # Arabic
    "bn",  # Bengali
    "cs",  # Czech
    "da",  # Danish
    "de",  # German
    "el",  # Greek
    "en",  # English
    "es",  # Spanish
    "fa",  # Persian
    "fi",  # Finnish
    "fil",  # Filipino / Tagalog
    "fr",  # French
    "gu",  # Gujarati
    "he",  # Hebrew
    "hi",  # Hindi
    "hu",  # Hungarian
    "id",  # Indonesian
    "it",  # Italian
    "ja",  # Japanese
    "ko",  # Korean
    "mr",  # Marathi
    "ms",  # Malay
    "nl",  # Dutch
    "no",  # Norwegian
    "pa",  # Punjabi
    "pl",  # Polish
    "pt",  # Portuguese
    "ro",  # Romanian
    "ru",  # Russian
    "sv",  # Swedish
    "sw",  # Swahili
    "ta",  # Tamil
    "te",  # Telugu
    "th",  # Thai
    "tr",  # Turkish
    "uk",  # Ukrainian
    "unknown",
    "ur",  # Urdu
    "vi",  # Vietnamese
    "zh",  # Chinese
]

DocumentType = Literal[
    "report",  # annual, quarterly, financial, status reports
    "manual",  # user guides, handbooks, operating instructions
    "contract",  # agreements, terms, service contracts, NDAs
    "correspondence",  # letters, emails, memos, notifications
    "policy",  # policies, procedures, guidelines, governance docs
    "invoice",  # invoices, bills, receipts, purchase orders
    "article",  # research papers, journal articles, studies, whitepapers
    "specification",  # technical specs, requirements, RFCs, API docs
    "presentation",  # slide decks, pitch decks
    "spreadsheet",  # data tables, budgets, financial models
    "form",  # application forms, questionnaires, surveys
    "minutes",  # meeting minutes, agendas, action items
    "proposal",  # project proposals, bids, offers, RFP responses
    "regulation",  # laws, regulations, compliance documents, standards
    "press_release",  # announcements, news releases, public statements
    "case_study",  # customer stories, use cases, project retrospectives
    "certificate",  # certificates, licenses, permits, accreditations
    "patent",  # patent applications, IP documents
    "datasheet",  # product datasheets, technical data sheets
    "release_notes",  # changelogs, version notes, patch notes
    "plan",  # project plans, roadmaps, schedules, strategies
    "audit",  # audit reports, inspection reports, assessments
    "training",  # training materials, course content, e-learning
    "general",  # fallback for unclassified documents
]

AudienceType = Literal[
    "general",  # no specific audience
    "technical",  # engineers, developers, architects
    "management",  # executives, managers, decision makers
    "legal",  # lawyers, compliance officers
    "financial",  # accountants, analysts, investors
    "scientific",  # researchers, academics
    "operational",  # field staff, operators, support
    "public",  # external, customer-facing
]

ChunkType = Literal[
    "text",
    "code",
    "table",
    "list",
    "figure_caption",
    "hier_summary_section",
    "hier_summary_doc",
]

ChunkClassification = Literal[
    "overview",
    "background",
    "definition",
    "requirement",
    "procedure",
    "example",
    "warning",
    "decision",
    "result",
    "reference",
    "financial",
    "legal",
    "technical",
    "operational",
    "other",
]

DEFAULT_CHUNK_CLASSIFICATION_TAXONOMY = ",".join(
    (
        "overview",
        "background",
        "definition",
        "requirement",
        "procedure",
        "example",
        "warning",
        "decision",
        "result",
        "reference",
        "financial",
        "legal",
        "technical",
        "operational",
        "other",
    )
)
