"""Vector search endpoint and index management for causal inference chunks."""

from __future__ import annotations

from databricks.vector_search.client import VectorSearchClient
from loguru import logger

from causal_inference_curator.config import Config


class VectorSearchManager:
    """Manages a Databricks Vector Search endpoint and delta-sync index.

    Handles endpoint creation, index creation, syncing, and search queries
    for the causal_inference_chunks table.
    """

    def __init__(
        self,
        config: Config,
        endpoint_name: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.config = config
        self.endpoint_name = endpoint_name or config.project.vector_search_endpoint
        self.embedding_model = embedding_model or config.project.embedding_endpoint
        self.catalog = config.project.catalog
        self.schema = config.project.schema
        self.index_name = f"{self.catalog}.{self.schema}.causal_inference_chunks_index"
        self.chunks_table = f"{self.catalog}.{self.schema}.causal_inference_chunks"
        self.client = VectorSearchClient()

    def create_endpoint_if_not_exists(self) -> None:
        """Create the vector search endpoint if it doesn't exist; wait until ONLINE."""
        endpoints_response = self.client.list_endpoints()
        endpoints = (
            endpoints_response.get("endpoints", [])
            if isinstance(endpoints_response, dict)
            else []
        )
        existing_names = [
            (ep.get("name") if isinstance(ep, dict) else getattr(ep, "name", None))
            for ep in endpoints
        ]

        if self.endpoint_name in existing_names:
            logger.info(f"Endpoint '{self.endpoint_name}' already exists")
            return

        logger.info(f"Creating endpoint '{self.endpoint_name}'...")
        self.client.create_endpoint_and_wait(
            name=self.endpoint_name,
            endpoint_type="STANDARD",
        )
        logger.info(f"✅ Endpoint '{self.endpoint_name}' is ONLINE")

    def create_or_get_index(self) -> object:
        """Create the delta sync index, or return it if it already exists."""
        self.create_endpoint_if_not_exists()

        # Try to get the index first (avoids redundant creation attempts)
        try:
            index = self.client.get_index(index_name=self.index_name)
            logger.info(f"✅ Index '{self.index_name}' already exists")
            return index
        except Exception:
            logger.info(f"Index '{self.index_name}' not found — creating...")

        # Create the index
        try:
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                index_name=self.index_name,
                source_table_name=self.chunks_table,
                pipeline_type="TRIGGERED",
                primary_key="id",
                embedding_source_column="text",
                embedding_model_endpoint_name=self.embedding_model,
            )
            logger.info(f"✅ Index '{self.index_name}' created")
            return index
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise
            # Transient: get_index failed earlier but index exists now
            logger.info(f"✅ Index '{self.index_name}' exists (concurrent create)")
            return self.client.get_index(index_name=self.index_name)

    def sync_index(self) -> None:
        """Trigger a sync of the vector search index with the source Delta table."""
        index = self.create_or_get_index()
        logger.info(f"Syncing index '{self.index_name}'...")
        index.sync()
        logger.info("✅ Index sync triggered")
