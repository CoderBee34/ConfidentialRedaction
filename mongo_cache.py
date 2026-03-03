"""
MongoDB caching layer for Azure Document Intelligence results.

Uses Motor (async MongoDB driver) to store and retrieve Azure DI
``AnalyzeResult`` objects keyed by ``doc_id``.  This avoids redundant
API calls when the same document is processed more than once.
"""

import logging
import urllib.parse
from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from azure.ai.documentintelligence.models import AnalyzeResult

logger = logging.getLogger(__name__)


class AzureResultCache:
    """Async MongoDB cache for Azure Document Intelligence results."""

    def __init__(
        self,
        username: str,
        password: str,
        server_ip: str,
        port: str,
        db_name: str,
        collection_name: str,
        *,
        connection_timeout_ms: int = 5_000,
        server_selection_timeout_ms: int = 5_000,
    ):
        encoded_user = urllib.parse.quote_plus(username)
        encoded_pass = urllib.parse.quote_plus(password)
        self._uri = (
            f"mongodb://{encoded_user}:{encoded_pass}@{server_ip}:{port}/"
        )
        self._db_name = db_name
        self._collection_name = collection_name
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._connection_timeout_ms = connection_timeout_ms
        self._server_selection_timeout_ms = server_selection_timeout_ms

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the Motor client and ensure the collection index exists."""
        self._client = AsyncIOMotorClient(
            self._uri,
            connectTimeoutMS=self._connection_timeout_ms,
            serverSelectionTimeoutMS=self._server_selection_timeout_ms,
        )
        self._db = self._client[self._db_name]
        collection = self._db[self._collection_name]

        # Unique index on doc_id for fast look-ups and upsert safety
        await collection.create_index("doc_id", unique=True)
        logger.info("MongoDB cache connected (db=%s, collection=%s)", self._db_name, self._collection_name)

    async def close(self) -> None:
        """Gracefully close the Motor client."""
        if self._client:
            self._client.close()
            logger.info("MongoDB cache connection closed.")

    # ── Public API ────────────────────────────────────────────────────────

    async def get(self, doc_id: str) -> Optional[AnalyzeResult]:
        """
        Retrieve a cached ``AnalyzeResult`` for *doc_id*.

        Returns ``None`` on cache miss or deserialisation failure.
        """
        collection = self._db[self._collection_name]
        try:
            document = await collection.find_one(
                {"doc_id": doc_id},
                {"_id": 0, "azure_result": 1},
            )
            if document and "azure_result" in document:
                result = AnalyzeResult(document["azure_result"])
                logger.info("Cache HIT for doc_id=%s", doc_id)
                return result
        except Exception:
            logger.warning("Cache lookup failed for doc_id=%s", doc_id, exc_info=True)

        logger.info("Cache MISS for doc_id=%s", doc_id)
        return None

    async def put(self, doc_id: str, result: AnalyzeResult) -> None:
        """
        Store an ``AnalyzeResult`` in the cache, keyed by *doc_id*.

        Uses an upsert so repeated calls for the same ``doc_id`` simply
        overwrite the previous entry.
        """
        collection = self._db[self._collection_name]
        try:
            serialised = result.as_dict()
            await collection.update_one(
                {"doc_id": doc_id},
                {
                    "$set": {
                        "azure_result": serialised,
                        "updated_at": datetime.now(timezone.utc),
                    },
                    "$setOnInsert": {
                        "created_at": datetime.now(timezone.utc),
                    },
                },
                upsert=True,
            )
            logger.info("Cached Azure result for doc_id=%s", doc_id)
        except Exception:
            # Caching failure should never break the main pipeline
            logger.warning("Failed to cache result for doc_id=%s", doc_id, exc_info=True)
