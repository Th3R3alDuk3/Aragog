from haystack import Document

from components.metadata_enricher import MetadataEnricher, _extract_title


def test_extract_title_prefers_h1_and_uses_filename_fallback() -> None:
    assert _extract_title("# Main Title\n\nBody", "report.pdf") == "Main Title"
    assert _extract_title("## Sub heading only", "my-file_name.pdf") == "My File Name"


def test_metadata_enricher_adds_document_level_fields() -> None:
    enricher = MetadataEnricher(
        embedding_model="test-model",
        embedding_provider="huggingface",
        embedding_dimension=1024,
        doc_beginning_chars=10,
    )
    doc = Document(
        content="# Titel\nDas ist ein Testdokument.",
        meta={"source": "bericht.pdf", "file_path": "/tmp/bericht.pdf"},
    )

    out = enricher.run([doc])["documents"][0]
    meta = out.meta

    assert len(meta["doc_id"]) == 64
    assert meta["title"] == "Titel"
    assert isinstance(meta["word_count"], int)
    assert isinstance(meta["indexed_at_ts"], int)
    assert isinstance(meta["language"], str)
    assert meta["doc_beginning"] == out.content[:10]
    assert meta["embedding_model"] == "test-model"
    assert meta["embedding_provider"] == "huggingface"
    assert meta["embedding_dimension"] == 1024


def test_metadata_enricher_merges_extra_meta() -> None:
    enricher = MetadataEnricher(
        embedding_model="m", embedding_provider="hf", embedding_dimension=512,
    )
    doc = Document(content="text", meta={"source": "f.pdf"})

    out = enricher.run([doc], extra_meta={"minio_url": "http://minio/docs/f.pdf", "minio_key": "abc-f.pdf"})["documents"][0]

    assert out.meta["minio_url"] == "http://minio/docs/f.pdf"
    assert out.meta["minio_key"] == "abc-f.pdf"
