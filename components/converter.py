from typing import Any

from haystack import Document, component


@component
class ExportConverter:

    def __init__(self,
        title_field: str,
    ) -> None:
        self._title_field = title_field

    @component.output_types(documents=list[Document])
    def run(self,
        exports: list[dict[str, Any]],
    ) -> dict[str, list[Document]]:

        documents: list[Document] = []

        for export in exports:

            document_meta = {
                field: value
                for field, value in export["document"].items()
                if field != "content"
            }

            title = document_meta.get(self._title_field) or export["path"]

            # top-level fields plus `field.key` for objects one level down
            fields = [
                *document_meta.items(),
                *(
                    (f"{field}.{key}", value)
                    for field, entries in document_meta.items()
                    for entry in (entries if isinstance(entries, list) else [entries])
                    if isinstance(entry, dict)
                    for key, value in entry.items()
                ),
            ]

            documents.append(Document(
                # one version of the document; its chunk ids derive from it
                id=export["hash"],
                content=export["document"]["content"]["text"],
                meta={
                    "title": str(title),
                    # the document across versions, and the link the agent cites
                    "url": export["path"],
                    "hash": export["hash"],
                    "document_meta": document_meta,
                    # all in one payload index; long texts are no filter values
                    "document_facets": [
                        f"{field}={value}"
                        for field, values in fields
                        for value in (values if isinstance(values, list) else [values])
                        if isinstance(value, str | int) and len(str(value)) <= 100
                    ],
                },
            ))

        return {"documents": documents}
