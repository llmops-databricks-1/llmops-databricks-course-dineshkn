"""Session memory management using Lakebase (Databricks PostgreSQL)."""

import json
import os
import urllib.parse
from typing import Any
from uuid import uuid4

import psycopg
from databricks.sdk import WorkspaceClient
from loguru import logger
from psycopg_pool import ConnectionPool


class LakebaseMemory:
    """Handles session message persistence using Lakebase (PostgreSQL).

    Supports two auth modes:
    - User (dev): Uses default WorkspaceClient auth (~/.databrickscfg)
    - SPN (prod): Set DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET env vars
    """

    def __init__(self, host: str, instance_name: str) -> None:
        self.host = host
        self.instance_name = instance_name
        self._pool: ConnectionPool | None = None
        self.client_id = os.getenv("DATABRICKS_CLIENT_ID", None)

    def _get_connection_string(self) -> str:
        """Build PostgreSQL connection string for Lakebase."""
        w = WorkspaceClient()

        if self.client_id:
            username = self.client_id
        else:
            user = w.current_user.me()
            username = urllib.parse.quote_plus(user.user_name)

        pg_credential = w.database.generate_database_credential(
            request_id=str(uuid4()), instance_names=[self.instance_name]
        )

        return (
            f"postgresql://{username}:{pg_credential.token}@{self.host}:5432/"
            "databricks_postgres?sslmode=require"
        )

    def _get_pool(self) -> ConnectionPool:
        """Get or create connection pool."""
        if self._pool is None:
            conn_string = self._get_connection_string()
            self._pool = ConnectionPool(conninfo=conn_string, min_size=1, max_size=5)
        return self._pool

    def _reset_pool(self) -> None:
        """Reset pool to force fresh credentials on next use."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _ensure_messages_table(self, conn: psycopg.Connection) -> None:
        """Create messages table and index if they don't exist."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
            ON session_messages(session_id)
        """)

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Load previous messages for a session, ordered by creation time."""
        try:
            with self._get_pool().connection() as conn:
                self._ensure_messages_table(conn)
                result = conn.execute(
                    """
                    SELECT message_data FROM session_messages
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                    """,
                    (session_id,),
                ).fetchall()
                return [row[0] for row in result]
        except psycopg.OperationalError:
            self._reset_pool()
            raise
        except Exception as e:
            logger.warning(f"Failed to load session messages: {e}")
            return []

    def save_messages(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> None:
        """Append new messages to a session (does not replace existing)."""
        try:
            with self._get_pool().connection() as conn:
                self._ensure_messages_table(conn)
                for msg in messages:
                    conn.execute(
                        "INSERT INTO session_messages (session_id, message_data) "
                        "VALUES (%s, %s)",
                        (session_id, json.dumps(msg)),
                    )
        except psycopg.OperationalError:
            self._reset_pool()
            raise
        except Exception as e:
            logger.warning(f"Failed to save session messages: {e}")
