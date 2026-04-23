"""SQLite persistence for analysis history."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import List, Optional

from analyzer.models import AnalysisRecord, AnalysisResult, ChatRecord, GatewayQARecord, QARecord, WriteRecord

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS analysis_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    input_text     TEXT    NOT NULL,
    language       TEXT    NOT NULL DEFAULT 'zh',
    result_json    TEXT    NOT NULL,
    provider_name  TEXT    NOT NULL DEFAULT '',
    model_name     TEXT    NOT NULL DEFAULT '',
    created_at     TEXT    NOT NULL
);
"""

_CREATE_QA_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS qa_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    book_name      TEXT    NOT NULL,
    question       TEXT    NOT NULL,
    answer         TEXT    NOT NULL,
    language       TEXT    NOT NULL DEFAULT 'zh',
    provider_name  TEXT    NOT NULL DEFAULT '',
    model_name     TEXT    NOT NULL DEFAULT '',
    created_at     TEXT    NOT NULL
);
"""

_CREATE_SETTINGS_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_CREATE_CHAT_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS chat_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    question       TEXT    NOT NULL,
    answer         TEXT    NOT NULL,
    language       TEXT    NOT NULL DEFAULT 'zh',
    provider_name  TEXT    NOT NULL DEFAULT '',
    model_name     TEXT    NOT NULL DEFAULT '',
    created_at     TEXT    NOT NULL
);
"""

_CREATE_GATEWAY_QA_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS gateway_qa_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    question       TEXT    NOT NULL,
    answer         TEXT    NOT NULL,
    language       TEXT    NOT NULL DEFAULT 'zh',
    provider_name  TEXT    NOT NULL DEFAULT '',
    model_name     TEXT    NOT NULL DEFAULT '',
    created_at     TEXT    NOT NULL
);
"""

_CREATE_WRITE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS write_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    input_text     TEXT    NOT NULL,
    result         TEXT    NOT NULL,
    action_type    TEXT    NOT NULL DEFAULT '',
    language       TEXT    NOT NULL DEFAULT 'zh',
    provider_name  TEXT    NOT NULL DEFAULT '',
    model_name     TEXT    NOT NULL DEFAULT '',
    created_at     TEXT    NOT NULL
);
"""


class Database:
    """Thin wrapper around SQLite for analysis history CRUD."""

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_QA_TABLE_SQL)
            conn.execute(_CREATE_SETTINGS_TABLE_SQL)
            conn.execute(_CREATE_CHAT_TABLE_SQL)
            conn.execute(_CREATE_GATEWAY_QA_TABLE_SQL)
            conn.execute(_CREATE_WRITE_TABLE_SQL)
            # Migrate: add provider_name/model_name columns if missing
            for table in ("analysis_history", "qa_history", "chat_history", "gateway_qa_history", "write_history"):
                for col in ("provider_name", "model_name"):
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
                    except sqlite3.OperationalError:
                        pass  # column already exists

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save_record(
        self,
        input_text: str,
        language: str,
        result: AnalysisResult,
        provider_name: str = "",
        model_name: str = "",
    ) -> int:
        """Insert a new analysis record. Returns the new row id."""
        now = datetime.now().isoformat(timespec="seconds")
        result_json = result.model_dump_json(ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO analysis_history (input_text, language, result_json, provider_name, model_name, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (input_text, language, result_json, provider_name, model_name, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_all_records(self, limit: int = 50) -> List[AnalysisRecord]:
        """Return the most recent records (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM analysis_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_record_by_id(self, record_id: int) -> Optional[AnalysisRecord]:
        """Fetch a single record by primary key."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_history WHERE id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def delete_record(self, record_id: int) -> bool:
        """Delete a record. Returns True if a row was actually deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM analysis_history WHERE id = ?",
                (record_id,),
            )
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> AnalysisRecord:
        result_data = json.loads(row["result_json"])
        return AnalysisRecord(
            id=row["id"],
            input_text=row["input_text"],
            language=row["language"],
            result=AnalysisResult.model_validate(result_data),
            provider_name=row["provider_name"] if "provider_name" in row.keys() else "",
            model_name=row["model_name"] if "model_name" in row.keys() else "",
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ------------------------------------------------------------------
    # Q&A CRUD
    # ------------------------------------------------------------------

    def save_qa_record(
        self,
        book_name: str,
        question: str,
        answer: str,
        language: str,
        provider_name: str = "",
        model_name: str = "",
    ) -> int:
        """Insert a new Q&A record. Returns the new row id."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO qa_history (book_name, question, answer, language, provider_name, model_name, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (book_name, question, answer, language, provider_name, model_name, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_all_qa_records(self, limit: int = 50) -> List[QARecord]:
        """Return the most recent Q&A records (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM qa_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_qa_record(r) for r in rows]

    def get_qa_record_by_id(self, record_id: int) -> Optional[QARecord]:
        """Fetch a single Q&A record by primary key."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM qa_history WHERE id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_qa_record(row)

    def delete_qa_record(self, record_id: int) -> bool:
        """Delete a Q&A record. Returns True if a row was actually deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM qa_history WHERE id = ?",
                (record_id,),
            )
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_qa_record(row: sqlite3.Row) -> QARecord:
        return QARecord(
            id=row["id"],
            book_name=row["book_name"],
            question=row["question"],
            answer=row["answer"],
            language=row["language"],
            provider_name=row["provider_name"] if "provider_name" in row.keys() else "",
            model_name=row["model_name"] if "model_name" in row.keys() else "",
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ------------------------------------------------------------------
    # Free Chat CRUD
    # ------------------------------------------------------------------

    def save_chat_record(
        self,
        question: str,
        answer: str,
        language: str,
        provider_name: str = "",
        model_name: str = "",
    ) -> int:
        """Insert a new free-chat record. Returns the new row id."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO chat_history (question, answer, language, provider_name, model_name, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (question, answer, language, provider_name, model_name, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_all_chat_records(self, limit: int = 50) -> List[ChatRecord]:
        """Return the most recent chat records (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_chat_record(r) for r in rows]

    def get_chat_record_by_id(self, record_id: int) -> Optional[ChatRecord]:
        """Fetch a single chat record by primary key."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chat_history WHERE id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_chat_record(row)

    def delete_chat_record(self, record_id: int) -> bool:
        """Delete a chat record. Returns True if a row was actually deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM chat_history WHERE id = ?",
                (record_id,),
            )
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_chat_record(row: sqlite3.Row) -> ChatRecord:
        return ChatRecord(
            id=row["id"],
            question=row["question"],
            answer=row["answer"],
            language=row["language"],
            provider_name=row["provider_name"] if "provider_name" in row.keys() else "",
            model_name=row["model_name"] if "model_name" in row.keys() else "",
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ------------------------------------------------------------------
    # Gateway Q&A CRUD
    # ------------------------------------------------------------------

    def save_gateway_qa_record(
        self,
        question: str,
        answer: str,
        language: str,
        provider_name: str = "",
        model_name: str = "",
    ) -> int:
        """Insert a new gateway Q&A record. Returns the new row id."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO gateway_qa_history (question, answer, language, provider_name, model_name, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (question, answer, language, provider_name, model_name, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_all_gateway_qa_records(self, limit: int = 50) -> List[GatewayQARecord]:
        """Return the most recent gateway Q&A records (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM gateway_qa_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_gateway_qa_record(r) for r in rows]

    def get_gateway_qa_record_by_id(self, record_id: int) -> Optional[GatewayQARecord]:
        """Fetch a single gateway Q&A record by primary key."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM gateway_qa_history WHERE id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_gateway_qa_record(row)

    def delete_gateway_qa_record(self, record_id: int) -> bool:
        """Delete a gateway Q&A record."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM gateway_qa_history WHERE id = ?",
                (record_id,),
            )
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_gateway_qa_record(row: sqlite3.Row) -> GatewayQARecord:
        return GatewayQARecord(
            id=row["id"],
            question=row["question"],
            answer=row["answer"],
            language=row["language"],
            provider_name=row["provider_name"] if "provider_name" in row.keys() else "",
            model_name=row["model_name"] if "model_name" in row.keys() else "",
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ------------------------------------------------------------------
    # Writing Assistant CRUD
    # ------------------------------------------------------------------

    def save_write_record(
        self,
        input_text: str,
        result: str,
        action_type: str,
        language: str,
        provider_name: str = "",
        model_name: str = "",
    ) -> int:
        """Insert a new writing record. Returns the new row id."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO write_history (input_text, result, action_type, language, provider_name, model_name, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (input_text, result, action_type, language, provider_name, model_name, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_all_write_records(self, limit: int = 50) -> List[WriteRecord]:
        """Return the most recent writing records (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM write_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_write_record(r) for r in rows]

    def get_write_record_by_id(self, record_id: int) -> Optional[WriteRecord]:
        """Fetch a single writing record by primary key."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM write_history WHERE id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_write_record(row)

    def delete_write_record(self, record_id: int) -> bool:
        """Delete a writing record."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM write_history WHERE id = ?",
                (record_id,),
            )
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_write_record(row: sqlite3.Row) -> WriteRecord:
        return WriteRecord(
            id=row["id"],
            input_text=row["input_text"],
            result=row["result"],
            action_type=row["action_type"],
            language=row["language"],
            provider_name=row["provider_name"] if "provider_name" in row.keys() else "",
            model_name=row["model_name"] if "model_name" in row.keys() else "",
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ------------------------------------------------------------------
    # Settings (key-value store)
    # ------------------------------------------------------------------

    def set_setting(self, key: str, value: str) -> None:
        """Insert or update a setting."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_setting(self, key: str, default: str = "") -> str:
        """Get a setting value, or *default* if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,),
            ).fetchone()
        return row["value"] if row else default

