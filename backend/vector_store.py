"""
Supabase pgvector adapter.

Reads/writes chunks to a Postgres table via the Supabase REST API. Retrieval
is performed by an RPC (`match_chunks`) defined in `migrations/0001_init.sql`.
"""

from __future__ import annotations

import logging
import os
import urllib.parse
import uuid
from typing import Any

import psycopg2
from psycopg2.extras import Json, RealDictCursor, execute_values
from psycopg2.pool import ThreadedConnectionPool

from database import get_supabase

logger = logging.getLogger("chatpdf.vector_store")

CHUNKS_TABLE = "chunks"
MATCH_RPC = "match_chunks"

_pg_pool: ThreadedConnectionPool | None = None


def _get_pg_pool() -> ThreadedConnectionPool | None:
    """Initialize or reuse a ThreadedConnectionPool for SUPABASE_DB_URL."""
    global _pg_pool
    if _pg_pool is None:
        raw_url = os.getenv("SUPABASE_DB_URL")
        if not raw_url:
            return None
        raw_url = raw_url.strip().strip('"').strip("'")
        # Handle unencoded '@' in passwords (e.g. postgres://user:p@ss@host...)
        if raw_url.count("@") > 1 and "://" in raw_url:
            prefix, remainder = raw_url.split("://", 1)
            userinfo, hostinfo = remainder.rsplit("@", 1)
            if ":" in userinfo:
                user, password = userinfo.split(":", 1)
                encoded_password = urllib.parse.quote_plus(password)
                raw_url = f"{prefix}://{user}:{encoded_password}@{hostinfo}"
        try:
            _pg_pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=raw_url)
            logger.info("Initialized psycopg2 connection pool for Supabase pgvector.")
        except Exception as e:
            logger.exception(f"Failed to initialize PostgreSQL pool for SUPABASE_DB_URL: {e}")
            _pg_pool = None
    return _pg_pool


def _vector_to_pg(vector: list[float]) -> str:
    """pgvector accepts text in '[v1,v2,...]' form."""
    return "[" + ",".join(f"{v:.7f}" for v in vector) + "]"


def insert_chunks(rows: list[dict[str, Any]]) -> None:
    """
    Bulk-insert chunk rows directly into public.chunks using psycopg2 and SUPABASE_DB_URL.
    Inserts exact mapped columns:
      id (uuid), user_id, pdf_id, chunk_index, page, page_width, page_height, bbox, text, embedding
    """
    if not rows:
        return

    pool = _get_pg_pool()
    if pool is not None:
        conn = None
        try:
            conn = pool.getconn()
            with conn.cursor() as cur:
                records = [
                    (
                        str(r.get("id") or uuid.uuid4()),
                        str(r["user_id"]),
                        str(r["pdf_id"]),
                        int(r["chunk_index"]),
                        int(r["page"]),
                        float(r["page_width"]),
                        float(r["page_height"]),
                        Json(r["bbox"]),
                        str(r["text"]),
                        _vector_to_pg(r["embedding"]) if isinstance(r["embedding"], list) else str(r["embedding"]),
                    )
                    for r in rows
                ]

                sql = """
                INSERT INTO public.chunks (
                    id, user_id, pdf_id, chunk_index, page, page_width, page_height, bbox, text, embedding
                ) VALUES %s;
                """
                execute_values(
                    cur,
                    sql,
                    records,
                    template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
                    page_size=200,
                )
                conn.commit()
                logger.info(f"Successfully inserted {len(records)} chunks into Supabase pgvector.")
                return
        except Exception as e:
            logger.exception(f"[insert_chunks] Direct Postgres insert failed, falling back to REST API: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                pool.putconn(conn)

    # Fallback to Supabase REST client if DB connection pool is unavailable
    payload = []
    for r in rows:
        payload.append(
            {
                "id": str(r.get("id") or uuid.uuid4()),
                "user_id": r["user_id"],
                "pdf_id": r["pdf_id"],
                "chunk_index": r["chunk_index"],
                "page": r["page"],
                "page_width": r["page_width"],
                "page_height": r["page_height"],
                "bbox": r["bbox"],
                "text": r["text"],
                "embedding": _vector_to_pg(r["embedding"]) if isinstance(r["embedding"], list) else str(r["embedding"]),
            }
        )

    supabase = get_supabase()
    # PostgREST imposes a row-count cap per request; chunk into batches of 200.
    BATCH = 200
    for i in range(0, len(payload), BATCH):
        batch = payload[i : i + BATCH]
        resp = supabase.table(CHUNKS_TABLE).insert(batch).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Supabase insert failed: {resp.error}")


def match_chunks(
    q_vec: list[float] | None = None,
    user_id: str = "",
    pdf_id: str = "",
    top_k: int = 5,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Top-k nearest neighbours via direct Postgres (SUPABASE_DB_URL) or Supabase client RPC.
    Orders by vector distance using the cosine distance (<=>) operator.
    """
    vector = q_vec if q_vec is not None else query_embedding
    if vector is None:
        raise ValueError("q_vec or query_embedding must be provided.")

    vector_str = _vector_to_pg(vector)
    pool = _get_pg_pool()

    if pool is not None:
        conn = None
        try:
            conn = pool.getconn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Ensure ivfflat index probe depth is sufficient for small PDF partitions
                cur.execute("SET LOCAL ivfflat.probes = 10;")
                sql = """
                SELECT
                    id,
                    chunk_index,
                    page,
                    page_width,
                    page_height,
                    bbox,
                    text,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM public.chunks
                WHERE user_id = %s
                  AND pdf_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """
                cur.execute(sql, (vector_str, user_id, pdf_id, vector_str, top_k))
                rows = cur.fetchall()
                results = []
                for r in rows:
                    row_dict = dict(r)
                    row_dict["id"] = str(row_dict["id"])
                    results.append(row_dict)
                return results
        except Exception as e:
            logger.exception(f"[match_chunks] Direct Postgres query failed, falling back to REST: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                pool.putconn(conn)

    # Fallback to Supabase REST client RPC if direct DB connection is unavailable
    supabase = get_supabase()
    resp = supabase.rpc(
        MATCH_RPC,
        {
            "query_embedding": vector_str,
            "p_user_id": user_id,
            "p_pdf_id": pdf_id,
            "match_count": top_k,
        },
    ).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Supabase RPC failed: {resp.error}")
    return resp.data or []


def delete_pdf_chunks(user_id: str, pdf_id: str) -> int:
    """Remove all chunks for a PDF — called on PDF deletion."""
    pool = _get_pg_pool()
    if pool is not None:
        conn = None
        try:
            conn = pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM public.chunks WHERE user_id = %s AND pdf_id = %s;",
                    (user_id, pdf_id),
                )
                deleted = cur.rowcount
                conn.commit()
                return deleted
        except Exception as e:
            logger.exception(f"[delete_pdf_chunks] Direct Postgres delete failed: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                pool.putconn(conn)

    supabase = get_supabase()
    resp = (
        supabase.table(CHUNKS_TABLE)
        .delete()
        .eq("user_id", user_id)
        .eq("pdf_id", pdf_id)
        .execute()
    )
    if getattr(resp, "error", None):
        raise RuntimeError(f"Supabase delete failed: {resp.error}")
    return len(resp.data or [])
