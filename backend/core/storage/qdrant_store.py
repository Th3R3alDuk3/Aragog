from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from pydantic import BaseModel, ConfigDict


class QdrantStores(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    children: QdrantDocumentStore
    parents: QdrantDocumentStore
