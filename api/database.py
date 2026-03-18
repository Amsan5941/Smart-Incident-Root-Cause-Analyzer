"""
Database layer using MongoDB (Motor async driver).
Collections:
  - incident_analyses   : every analysis request/response (audit trail)
  - training_incidents  : historical incident data for similarity search
"""

import json
import os
from datetime import datetime
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB   = os.environ.get("MONGO_DB", "incident_analyzer")

# ─── Client singleton ────────────────────────────────────────────────────────

_client: Optional[AsyncIOMotorClient] = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URL)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    return get_client()[MONGO_DB]


# FastAPI dependency — yields the db for use in route handlers
async def get_db() -> AsyncIOMotorDatabase:
    yield get_database()


# ─── Init (create indexes) ───────────────────────────────────────────────────

async def init_db():
    db = get_database()

    # incident_analyses indexes
    await db.incident_analyses.create_index("request_id", unique=True)
    await db.incident_analyses.create_index("analyzed_at")
    await db.incident_analyses.create_index("service")
    await db.incident_analyses.create_index("category")

    # training_incidents indexes
    await db.training_incidents.create_index("incident_id", unique=True)
    await db.training_incidents.create_index("category")
    await db.training_incidents.create_index("service")


# ─── Training data loader ─────────────────────────────────────────────────────

async def load_training_data(jsonl_path: str) -> int:
    db = get_database()
    count = 0

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            incident_id = data.get("incident_id", f"INC-{count:05d}")

            # Skip if already exists
            existing = await db.training_incidents.find_one({"incident_id": incident_id})
            if existing:
                continue

            ts_str = data.get("timestamp")
            ts = None
            if ts_str:
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    pass

            doc = {
                "incident_id": incident_id,
                "timestamp": ts,
                "service": data.get("service"),
                "severity": data.get("severity"),
                "category": data.get("category"),
                "logs": data.get("logs"),
                "metrics": data.get("metrics"),
                "error_trace": data.get("error_trace"),
                "root_cause": data.get("root_cause"),
                "resolution_steps": data.get("resolution_steps", []),
            }
            await db.training_incidents.insert_one(doc)
            count += 1

    return count


# ─── Similarity search ────────────────────────────────────────────────────────

async def find_similar_incidents(
    db: AsyncIOMotorDatabase,
    category: str,
    service: str,
    limit: int = 3,
) -> list[dict]:
    results = []

    # Same category + same service first
    async for doc in db.training_incidents.find(
        {"category": category, "service": service},
        {"_id": 0, "incident_id": 1, "service": 1, "root_cause": 1, "timestamp": 1, "category": 1},
        limit=limit,
    ):
        results.append(doc)

    # Fill remainder with same category only
    if len(results) < limit:
        seen = {r["incident_id"] for r in results}
        async for doc in db.training_incidents.find(
            {"category": category, "incident_id": {"$nin": list(seen)}},
            {"_id": 0, "incident_id": 1, "service": 1, "root_cause": 1, "timestamp": 1, "category": 1},
            limit=limit - len(results),
        ):
            results.append(doc)

    # Serialize datetimes
    for r in results:
        if isinstance(r.get("timestamp"), datetime):
            r["timestamp"] = r["timestamp"].isoformat()

    return results


# ─── Analysis CRUD ────────────────────────────────────────────────────────────

async def save_analysis(db: AsyncIOMotorDatabase, doc: dict) -> dict:
    await db.incident_analyses.insert_one(doc)
    doc.pop("_id", None)
    return doc


async def get_analysis(db: AsyncIOMotorDatabase, request_id: str) -> Optional[dict]:
    doc = await db.incident_analyses.find_one(
        {"request_id": request_id}, {"_id": 0}
    )
    if doc and isinstance(doc.get("analyzed_at"), datetime):
        doc["analyzed_at"] = doc["analyzed_at"].isoformat()
    return doc


async def list_analyses(
    db: AsyncIOMotorDatabase,
    service: Optional[str] = None,
    category: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[int, list[dict]]:
    filt = {}
    if service:
        filt["service"] = service
    if category:
        filt["category"] = category

    total = await db.incident_analyses.count_documents(filt)
    docs = []
    async for doc in db.incident_analyses.find(
        filt,
        {"_id": 0, "request_id": 1, "analyzed_at": 1, "service": 1,
         "severity": 1, "root_cause": 1, "confidence": 1,
         "category": 1, "inference_time_ms": 1},
        sort=[("analyzed_at", -1)],
        skip=(page - 1) * page_size,
        limit=page_size,
    ):
        if isinstance(doc.get("analyzed_at"), datetime):
            doc["analyzed_at"] = doc["analyzed_at"].isoformat()
        docs.append(doc)

    return total, docs


async def update_feedback(
    db: AsyncIOMotorDatabase,
    request_id: str,
    score: int,
    correct: bool,
    comment: Optional[str],
) -> bool:
    result = await db.incident_analyses.update_one(
        {"request_id": request_id},
        {"$set": {
            "feedback_score": score,
            "feedback_correct": 1 if correct else 0,
            "feedback_comment": comment,
        }},
    )
    return result.matched_count > 0


async def get_stats(db: AsyncIOMotorDatabase) -> dict:
    pipeline = [
        {"$group": {
            "_id": None,
            "total": {"$sum": 1},
            "avg_confidence": {"$avg": "$confidence"},
            "avg_inference_ms": {"$avg": "$inference_time_ms"},
        }}
    ]
    agg = await db.incident_analyses.aggregate(pipeline).to_list(1)
    base = agg[0] if agg else {"total": 0, "avg_confidence": 0, "avg_inference_ms": 0}

    # Category distribution
    cat_pipeline = [
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    cat_dist = {
        doc["_id"]: doc["count"]
        async for doc in db.incident_analyses.aggregate(cat_pipeline)
        if doc["_id"]
    }

    # Feedback accuracy
    correct = await db.incident_analyses.count_documents({"feedback_correct": 1})
    total_fb = await db.incident_analyses.count_documents(
        {"feedback_correct": {"$exists": True}}
    )

    return {
        "total_analyses": base.get("total", 0),
        "avg_confidence": round(float(base.get("avg_confidence") or 0), 3),
        "avg_inference_ms": round(float(base.get("avg_inference_ms") or 0)),
        "feedback_accuracy": round(correct / total_fb, 3) if total_fb else None,
        "feedback_count": total_fb,
        "category_distribution": cat_dist,
    }
