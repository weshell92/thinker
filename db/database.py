"""SQLite persistence for analysis history."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import List, Optional

from analyzer.models import AnalysisRecord, AnalysisResult, ChatRecord, DatingChatRecord, GatewayQARecord, QARecord, TranslateRecord, WriteRecord

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

_CREATE_TRANSLATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS translate_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    input_text     TEXT    NOT NULL,
    result         TEXT    NOT NULL,
    source_lang    TEXT    NOT NULL DEFAULT '',
    target_lang    TEXT    NOT NULL DEFAULT '',
    provider_name  TEXT    NOT NULL DEFAULT '',
    model_name     TEXT    NOT NULL DEFAULT '',
    created_at     TEXT    NOT NULL
);
"""

_CREATE_DATING_CHAT_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS dating_chat_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_gender    TEXT    NOT NULL DEFAULT '',
    chat_stage     TEXT    NOT NULL DEFAULT '',
    scene_dialogue TEXT    NOT NULL,
    result         TEXT    NOT NULL,
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
            conn.execute(_CREATE_DATING_CHAT_TABLE_SQL)
            conn.execute(_CREATE_TRANSLATE_TABLE_SQL)
            # Migrate: add provider_name/model_name columns if missing
            for table in ("analysis_history", "qa_history", "chat_history", "gateway_qa_history", "write_history", "dating_chat_history", "translate_history"):
                for col in ("provider_name", "model_name"):
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
                    except sqlite3.OperationalError:
                        pass  # column already exists


    def ensure_soft_delete_schema(self) -> None:
        """Backfill soft-delete columns for legacy DBs."""
        for table in ("analysis_records", "qa_records", "write_records"):
            self._ensure_soft_delete_columns(table)

    def _ensure_soft_delete_columns(self, table_name: str) -> None:
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        existing_cols = {row[1] for row in cur.fetchall()}

        if "is_deleted" not in existing_cols:
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN is_deleted INTEGER DEFAULT 0")
        if "deleted_at" not in existing_cols:
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN deleted_at TEXT")
        self.conn.commit()

    # ---------- soft delete ----------
    def soft_delete_record(self, table: str, record_id: int) -> bool:
        cur = self.conn.cursor()
        cur.execute(
            f"UPDATE {table} SET is_deleted=1, deleted_at=? WHERE id=?",
            (datetime.utcnow().isoformat(), record_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def restore_record(self, table: str, record_id: int) -> bool:
        cur = self.conn.cursor()
        cur.execute(
            f"UPDATE {table} SET is_deleted=0, deleted_at=NULL WHERE id=?",
            (record_id,),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def soft_delete_analysis_record(self, record_id: int) -> bool:
        return self.soft_delete_record("analysis_records", record_id)

    def soft_delete_qa_record(self, record_id: int) -> bool:
        return self.soft_delete_record("qa_records", record_id)

    def soft_delete_write_record(self, record_id: int) -> bool:
        return self.soft_delete_record("write_records", record_id)

    def restore_analysis_record(self, record_id: int) -> bool:
        return self.restore_record("analysis_records", record_id)

    def restore_qa_record(self, record_id: int) -> bool:
        return self.restore_record("qa_records", record_id)

    def restore_write_record(self, record_id: int) -> bool:
        return self.restore_record("write_records", record_id)

    # ---------- admin read ----------
    def admin_list_table(self, table: str):
        cur = self.conn.cursor()
        cur.execute(f"SELECT * FROM {table} ORDER BY id DESC")
        rows = cur.fetchall()
        col_names = [d[0] for d in cur.description] if cur.description else []
        return col_names, rows
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
    # Dating Chat CRUD
    # ------------------------------------------------------------------

    def save_dating_chat_record(
        self,
        user_gender: str,
        chat_stage: str,
        scene_dialogue: str,
        result: str,
        language: str,
        provider_name: str = "",
        model_name: str = "",
    ) -> int:
        """Insert a new dating chat record. Returns the new row id."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO dating_chat_history (user_gender, chat_stage, scene_dialogue, result, language, provider_name, model_name, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_gender, chat_stage, scene_dialogue, result, language, provider_name, model_name, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_all_dating_chat_records(self, limit: int = 50) -> List[DatingChatRecord]:
        """Return the most recent dating chat records (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dating_chat_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dating_chat_record(r) for r in rows]

    def get_dating_chat_record_by_id(self, record_id: int) -> Optional[DatingChatRecord]:
        """Fetch a single dating chat record by primary key."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dating_chat_history WHERE id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dating_chat_record(row)

    def delete_dating_chat_record(self, record_id: int) -> bool:
        """Delete a dating chat record."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM dating_chat_history WHERE id = ?",
                (record_id,),
            )
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_dating_chat_record(row: sqlite3.Row) -> DatingChatRecord:
        return DatingChatRecord(
            id=row["id"],
            user_gender=row["user_gender"],
            chat_stage=row["chat_stage"],
            scene_dialogue=row["scene_dialogue"],
            result=row["result"],
            language=row["language"],
            provider_name=row["provider_name"] if "provider_name" in row.keys() else "",
            model_name=row["model_name"] if "model_name" in row.keys() else "",
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ------------------------------------------------------------------
    # Translation CRUD
    # ------------------------------------------------------------------

    def save_translate_record(
        self,
        input_text: str,
        result: str,
        source_lang: str,
        target_lang: str,
        provider_name: str = "",
        model_name: str = "",
    ) -> int:
        """Insert a new translation record. Returns the new row id."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO translate_history (input_text, result, source_lang, target_lang, provider_name, model_name, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (input_text, result, source_lang, target_lang, provider_name, model_name, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_all_translate_records(self, limit: int = 50) -> List[TranslateRecord]:
        """Return the most recent translation records (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM translate_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_translate_record(r) for r in rows]

    def get_translate_record_by_id(self, record_id: int) -> Optional[TranslateRecord]:
        """Fetch a single translation record by primary key."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM translate_history WHERE id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_translate_record(row)

    def delete_translate_record(self, record_id: int) -> bool:
        """Delete a translation record."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM translate_history WHERE id = ?",
                (record_id,),
            )
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_translate_record(row: sqlite3.Row) -> TranslateRecord:
        return TranslateRecord(
            id=row["id"],
            input_text=row["input_text"],
            result=row["result"],
            source_lang=row["source_lang"] if "source_lang" in row.keys() else "",
            target_lang=row["target_lang"] if "target_lang" in row.keys() else "",
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

