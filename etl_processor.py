"""
FinQueryFlow - Pipeline Health Check
Pings all infrastructure components and returns unified status
"""

from fastapi import APIRouter
from models.schemas import PipelineHealth, ComponentStatus
from datetime import datetime
import chromadb
from chromadb.config import Settings
import psycopg2
import redis as redis_lib
from kafka import KafkaAdminClient
from kafka.errors import KafkaError
import os

router = APIRouter()

KAFKA_BROKER = os.getenv("KAFKA_BROKER",   "localhost:9092")
CHROMA_HOST  = os.getenv("CHROMA_HOST",    "localhost")
CHROMA_PORT  = int(os.getenv("CHROMA_PORT","8001"))
PG_DSN       = os.getenv("DATABASE_URL",   "postgresql://finuser:finpass@localhost:5432/finqueryflow")
REDIS_URL    = os.getenv("REDIS_URL",      "redis://localhost:6379")


def check_kafka() -> ComponentStatus:
    try:
        admin = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER, request_timeout_ms=3000)
        admin.close()
        return ComponentStatus.healthy
    except KafkaError:
        return ComponentStatus.down
    except Exception:
        return ComponentStatus.degraded


def check_chroma() -> ComponentStatus:
    try:
        client = chromadb.HttpClient(
            host=CHROMA_HOST, port=CHROMA_PORT,
            settings=Settings(anonymized_telemetry=False),
        )
        client.heartbeat()
        return ComponentStatus.healthy
    except Exception:
        return ComponentStatus.down


def check_postgres() -> ComponentStatus:
    try:
        conn = psycopg2.connect(PG_DSN, connect_timeout=3)
        conn.close()
        return ComponentStatus.healthy
    except Exception:
        return ComponentStatus.down


def check_redis() -> ComponentStatus:
    try:
        r = redis_lib.from_url(REDIS_URL, socket_timeout=2)
        r.ping()
        return ComponentStatus.healthy
    except Exception:
        return ComponentStatus.down


@router.get("/health/pipeline", response_model=PipelineHealth)
async def pipeline_health():
    """
    Returns health status of every pipeline component.
    Called by the dashboard every 30s to update status indicators.
    """
    kafka    = check_kafka()
    chroma   = check_chroma()
    postgres = check_postgres()
    redis    = check_redis()

    # Derived statuses
    ingestion = ComponentStatus.healthy if kafka == ComponentStatus.healthy else ComponentStatus.degraded
    etl       = ComponentStatus.healthy if (kafka == ComponentStatus.healthy and chroma == ComponentStatus.healthy and postgres == ComponentStatus.healthy) else ComponentStatus.degraded
    rag       = ComponentStatus.healthy if (chroma == ComponentStatus.healthy and redis == ComponentStatus.healthy) else ComponentStatus.degraded

    return PipelineHealth(
        kafka      = kafka,
        chromadb   = chroma,
        postgres   = postgres,
        redis      = redis,
        ingestion  = ingestion,
        etl        = etl,
        rag        = rag,
        checked_at = datetime.utcnow(),
    )
