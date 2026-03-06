"""
MongoDB caching layer for Azure Document Intelligence results.

Uses Motor (async MongoDB driver) to store and retrieve Azure DI
``AnalyzeResult`` objects keyed by ``doc_id``.  This avoids redundant
API calls when the same document is processed more than once.
"""

import asyncio
import logging
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError
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
        overwrite the previous entry.  Clears any processing claim.
        """
        collection = self._db[self._collection_name]
        try:
            serialised = result.as_dict()
            await collection.update_one(
                {"doc_id": doc_id},
                {
                    "$set": {
                        "azure_result": serialised,
                        "status": "ready",
                        "updated_at": datetime.now(timezone.utc),
                    },
                    "$unset": {
                        "claimed_at": "",
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

    async def get_or_claim(
        self, doc_id: str, claim_ttl_seconds: int = 300,
    ) -> Tuple[str, Optional[AnalyzeResult]]:
        """
        Atomically check cache and claim processing rights.

        Prevents duplicate Azure API calls across multiple workers.

        Returns
        -------
        ("hit", AnalyzeResult)
            Cached result found.
        ("claimed", None)
            This worker claimed the right to call Azure.
        ("wait", None)
            Another worker is already processing this document.
        """
        collection = self._db[self._collection_name]
        now = datetime.now(timezone.utc)

        # 1. Check for an existing document
        doc = await collection.find_one({"doc_id": doc_id})

        if doc:
            # Full result already cached
            if "azure_result" in doc:
                logger.info("Cache HIT for doc_id=%s", doc_id)
                return "hit", AnalyzeResult(doc["azure_result"])

            # Document exists but is still being processed
            claimed_at = doc.get("claimed_at")
            if claimed_at:
                # MongoDB may return naive datetimes; treat them as UTC
                if claimed_at.tzinfo is None:
                    claimed_at = claimed_at.replace(tzinfo=timezone.utc)
                age = (now - claimed_at).total_seconds()
                if age < claim_ttl_seconds:
                    logger.info(
                        "doc_id=%s claimed by another worker %ds ago",
                        doc_id, int(age),
                    )
                    return "wait", None

                # Claim has expired — try to re-claim atomically
                updated = await collection.find_one_and_update(
                    {"doc_id": doc_id, "claimed_at": claimed_at},
                    {"$set": {"claimed_at": now}},
                )
                if updated:
                    logger.info("Re-claimed expired doc_id=%s", doc_id)
                    return "claimed", None
                # Lost the race; another worker re-claimed first
                return "wait", None

        # 2. No document yet — insert a processing placeholder
        try:
            await collection.insert_one({
                "doc_id": doc_id,
                "status": "processing",
                "claimed_at": now,
                "created_at": now,
            })
            logger.info("Claimed new doc_id=%s for processing", doc_id)
            return "claimed", None
        except DuplicateKeyError:
            # Another worker inserted between our find_one and insert
            doc = await collection.find_one({"doc_id": doc_id})
            if doc and "azure_result" in doc:
                logger.info("Cache HIT (race) for doc_id=%s", doc_id)
                return "hit", AnalyzeResult(doc["azure_result"])
            return "wait", None

    async def wait_for_result(
        self,
        doc_id: str,
        timeout: float = 120,
        initial_interval: float = 1.0,
        max_interval: float = 5.0,
    ) -> Optional[AnalyzeResult]:
        """
        Poll the cache until a result appears or *timeout* seconds elapse.

        Uses exponential back-off between polls to reduce DB load.
        """
        start = time.monotonic()
        interval = initial_interval
        while time.monotonic() - start < timeout:
            result = await self.get(doc_id)
            if result is not None:
                return result
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                break
            await asyncio.sleep(min(interval, remaining))
            interval = min(interval * 1.5, max_interval)

        logger.warning(
            "Timed out waiting for doc_id=%s after %.0fs", doc_id, timeout,
        )
        return None

    async def release_claim(self, doc_id: str) -> None:
        """Release a processing claim so another worker can retry."""
        collection = self._db[self._collection_name]
        try:
            await collection.delete_one(
                {"doc_id": doc_id, "status": "processing"}
            )
            logger.info("Released claim for doc_id=%s", doc_id)
        except Exception:
            logger.warning(
                "Failed to release claim for doc_id=%s", doc_id, exc_info=True,
            )
