"""Persistent local storage for chats, controllable memories and the lexical RAG index.

All records stay in a per-user SQLite database. This module performs no network
requests and never copies the user's source documents; it stores extracted text
chunks and checks source file metadata before reusing an index after restart.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .rag import DocumentChunk, RAGIndex


DEFAULT_DB_NAME = "assistant_data.sqlite3"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def default_backup_directory() -> Path:
    """Return the in-project backup folder, creating it when possible.

    The user can still choose another destination. This folder is a convenient
    default, not an off-device disaster-recovery location.
    """
    return Path(__file__).resolve().parents[2] / "Sauvegardes"


def default_database_path() -> Path:
    """Return a writable user-local DB path, with an environment override for tests."""
    override = os.environ.get("BITNET_ASSISTANT_DATA_DIR")
    if override:
        return Path(override).expanduser() / DEFAULT_DB_NAME
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        base = Path(local_app_data)
    elif os.name == "nt":
        base = Path.home() / "AppData" / "Local"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "BitNetWindowsAssistant" / DEFAULT_DB_NAME


class StorageError(RuntimeError):
    """A local database operation failed and can be shown to the user."""


class LocalStore:
    """Small SQLite data store for conversations, approved memories and RAG chunks."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_database_path()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(f"Impossible de préparer la base locale {self.path}: {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def _db(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._db() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    display_name TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, id);
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'Général',
                    source TEXT NOT NULL DEFAULT 'manuel',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_updated
                    ON memories(enabled, updated_at DESC);
                CREATE TABLE IF NOT EXISTS rag_documents (
                    source_path TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    modified_ns INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    source_path TEXT NOT NULL REFERENCES rag_documents(source_path) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    locator TEXT NOT NULL,
                    text TEXT NOT NULL,
                    PRIMARY KEY(source_path, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_source
                    ON rag_chunks(source_path, chunk_index);
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                PRAGMA user_version = 1;
                """
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def create_conversation(self, title: str = "Nouvelle conversation") -> str:
        conversation_id = str(uuid.uuid4())
        stamp = _now()
        try:
            with self._db() as db:
                db.execute(
                    "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (conversation_id, title.strip() or "Nouvelle conversation", stamp, stamp),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de créer la conversation : {exc}") from exc
        return conversation_id

    def list_conversations(self, limit: int = 100) -> list[dict[str, Any]]:
        try:
            with self._db() as db:
                rows = db.execute(
                    "SELECT c.id, c.title, c.created_at, c.updated_at, "
                    "(SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS message_count "
                    "FROM conversations c ORDER BY c.updated_at DESC, c.created_at DESC LIMIT ?",
                    (max(1, min(int(limit), 500)),),
                ).fetchall()
            return [dict(row) for row in rows]
        except (sqlite3.Error, ValueError) as exc:
            raise StorageError(f"Impossible de lire les conversations : {exc}") from exc

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        try:
            with self._db() as db:
                return self._row_to_dict(db.execute(
                    "SELECT id, title, created_at, updated_at FROM conversations WHERE id=?",
                    (conversation_id,),
                ).fetchone())
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de lire la conversation : {exc}") from exc

    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        clean_title = " ".join(title.split()).strip()[:120] or "Nouvelle conversation"
        try:
            with self._db() as db:
                db.execute(
                    "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                    (clean_title, _now(), conversation_id),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible d'enregistrer le titre de la conversation : {exc}") from exc

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        timestamp: str | None = None,
        display_name: str | None = None,
    ) -> None:
        stamp = timestamp or _now()
        try:
            with self._db() as db:
                cursor = db.execute(
                    "INSERT INTO messages(conversation_id, role, content, timestamp, display_name) VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, role, content, stamp, display_name),
                )
                if cursor.rowcount != 1:
                    raise StorageError("Le message n'a pas été enregistré.")
                db.execute(
                    "UPDATE conversations SET updated_at=? WHERE id=?",
                    (_now(), conversation_id),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible d'enregistrer le message : {exc}") from exc

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        try:
            with self._db() as db:
                rows = db.execute(
                    "SELECT role, content, timestamp, display_name FROM messages "
                    "WHERE conversation_id=? ORDER BY id ASC",
                    (conversation_id,),
                ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de restaurer les messages : {exc}") from exc

    def clear_conversation(self, conversation_id: str) -> None:
        stamp = _now()
        try:
            with self._db() as db:
                db.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
                db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (stamp, conversation_id))
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible d'effacer cette conversation : {exc}") from exc

    def delete_conversation(self, conversation_id: str) -> None:
        try:
            with self._db() as db:
                db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de supprimer cette conversation : {exc}") from exc

    def create_memory(self, content: str, *, category: str = "Général", source: str = "manuel", enabled: bool = True) -> str:
        clean_content = content.strip()
        if not clean_content:
            raise StorageError("Un souvenir ne peut pas être vide.")
        if len(clean_content) > 6000:
            raise StorageError("Un souvenir est limité à 6 000 caractères.")
        memory_id = str(uuid.uuid4())
        stamp = _now()
        try:
            with self._db() as db:
                db.execute(
                    "INSERT INTO memories(id, content, category, source, enabled, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (memory_id, clean_content, category.strip() or "Général", source, int(enabled), stamp, stamp),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible d'enregistrer le souvenir : {exc}") from exc
        return memory_id

    def list_memories(self) -> list[dict[str, Any]]:
        try:
            with self._db() as db:
                rows = db.execute(
                    "SELECT id, content, category, source, enabled, created_at, updated_at "
                    "FROM memories ORDER BY enabled DESC, updated_at DESC"
                ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de lire les souvenirs : {exc}") from exc

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        try:
            with self._db() as db:
                return self._row_to_dict(db.execute(
                    "SELECT id, content, category, source, enabled, created_at, updated_at FROM memories WHERE id=?",
                    (memory_id,),
                ).fetchone())
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de lire le souvenir : {exc}") from exc

    def update_memory(self, memory_id: str, content: str, *, category: str | None = None) -> None:
        clean_content = content.strip()
        if not clean_content:
            raise StorageError("Un souvenir ne peut pas être vide.")
        if len(clean_content) > 6000:
            raise StorageError("Un souvenir est limité à 6 000 caractères.")
        try:
            with self._db() as db:
                if category is None:
                    db.execute("UPDATE memories SET content=?, updated_at=? WHERE id=?", (clean_content, _now(), memory_id))
                else:
                    db.execute(
                        "UPDATE memories SET content=?, category=?, updated_at=? WHERE id=?",
                        (clean_content, category.strip() or "Général", _now(), memory_id),
                    )
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de modifier le souvenir : {exc}") from exc

    def set_memory_enabled(self, memory_id: str, enabled: bool) -> None:
        try:
            with self._db() as db:
                db.execute("UPDATE memories SET enabled=?, updated_at=? WHERE id=?", (int(enabled), _now(), memory_id))
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de changer l'état du souvenir : {exc}") from exc

    def delete_memory(self, memory_id: str) -> None:
        try:
            with self._db() as db:
                db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de supprimer le souvenir : {exc}") from exc

    def enabled_memories_for_query(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Return enabled memories: prefer lexical matches, then recent approved items.

        The list is deliberately small so it does not consume much of BitNet's context.
        User-controlled preference memories remain eligible even when word overlap is low.
        """
        records = [item for item in self.list_memories() if item["enabled"]]
        if not records:
            return []
        from .rag import tokenize

        query_terms = set(tokenize(query))
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        global_preferences: list[dict[str, Any]] = []
        for order, memory in enumerate(records):
            words = set(tokenize(memory["content"]))
            overlap = len(query_terms & words)
            category = str(memory.get("category", "")).casefold()
            if any(label in category for label in ("préférence", "preference", "règle", "regle")):
                global_preferences.append(memory)
            if overlap:
                ranked.append((overlap, -order, memory))
        ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        maximum = max(0, min(int(limit), 10))
        if maximum == 0:
            return []
        chosen: list[dict[str, Any]] = []
        seen: set[str] = set()
        # Approved preferences and explicit rules apply across topics; other
        # memories are included only when their words relate to the current query.
        for memory in global_preferences + [entry[2] for entry in ranked]:
            memory_id = str(memory["id"])
            if memory_id not in seen:
                chosen.append(memory)
                seen.add(memory_id)
            if len(chosen) >= maximum:
                break
        return chosen

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        try:
            with self._db() as db:
                row = db.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
            return str(row["value"]) if row is not None else default
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de lire le paramètre local : {exc}") from exc

    def set_setting(self, key: str, value: str) -> None:
        try:
            with self._db() as db:
                db.execute(
                    "INSERT INTO app_settings(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, value),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible d'enregistrer le paramètre local : {exc}") from exc

    def save_rag_index(self, index: RAGIndex) -> list[str]:
        """Atomically replace persisted chunks for a new index; source files are read-only."""
        chunks_by_path: dict[str, list[DocumentChunk]] = {}
        for chunk in index.chunks:
            chunks_by_path.setdefault(str(Path(chunk.source_path)), []).append(chunk)
        indexed_rows: list[tuple[str, str, int, int, str]] = []
        persisted_chunks: list[tuple[str, int, str, str, str]] = []
        warnings: list[str] = []
        for source_path in index.indexed_files:
            key = str(Path(source_path))
            chunks = chunks_by_path.get(key, [])
            if not chunks:
                continue
            path = Path(source_path)
            try:
                stat = path.stat()
                if not path.is_file():
                    raise OSError("le chemin ne désigne pas un fichier")
            except OSError as exc:
                warnings.append(f"{path.name} n'a pas été conservé dans l'index persistant : {exc}.")
                continue
            filename = chunks[0].filename
            indexed_rows.append((key, filename, int(stat.st_mtime_ns), int(stat.st_size), _now()))
            seen_chunk_indices: set[int] = set()
            for chunk in chunks:
                chunk_index = int(chunk.chunk_index)
                if chunk_index in seen_chunk_indices:
                    raise StorageError(
                        f"L'index de {filename} contient le numéro de passage dupliqué {chunk_index}. "
                        "Aucune modification de l'index enregistré n'a été appliquée ; relance l'indexation du document."
                    )
                seen_chunk_indices.add(chunk_index)
                persisted_chunks.append((key, chunk_index, chunk.filename, chunk.locator, chunk.text))
        try:
            with self._db() as db:
                db.execute("DELETE FROM rag_documents")
                db.executemany(
                    "INSERT INTO rag_documents(source_path, filename, modified_ns, size_bytes, indexed_at) VALUES (?, ?, ?, ?, ?)",
                    indexed_rows,
                )
                db.executemany(
                    "INSERT INTO rag_chunks(source_path, chunk_index, filename, locator, text) VALUES (?, ?, ?, ?, ?)",
                    persisted_chunks,
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible d'enregistrer l'index documentaire : {exc}") from exc
        return warnings

    def save_rag_documents(self, index: RAGIndex) -> list[str]:
        """Atomically upsert only the documents supplied, preserving every other index."""
        chunks_by_path: dict[str, list[DocumentChunk]] = {}
        for chunk in index.chunks:
            chunks_by_path.setdefault(str(Path(chunk.source_path).expanduser().resolve()), []).append(chunk)

        rows: list[tuple[str, str, int, int, str]] = []
        persisted_chunks: list[tuple[str, int, str, str, str]] = []
        warnings: list[str] = []
        for source in index.indexed_files:
            key = str(Path(source).expanduser().resolve())
            chunks = chunks_by_path.get(key, [])
            if not chunks:
                continue
            seen: set[int] = set()
            for chunk in chunks:
                chunk_index = int(chunk.chunk_index)
                if chunk_index in seen:
                    raise StorageError(
                        f"L'index de {chunks[0].filename} contient le numéro de passage dupliqué {chunk_index}. "
                        "Aucune modification n'a été appliquée."
                    )
                seen.add(chunk_index)
            path = Path(key)
            try:
                stat = path.stat()
                if not path.is_file():
                    raise OSError("le chemin ne désigne pas un fichier")
            except OSError as exc:
                warnings.append(f"{path.name} n'a pas été conservé dans l'index persistant : {exc}.")
                continue
            rows.append((key, chunks[0].filename, int(stat.st_mtime_ns), int(stat.st_size), _now()))
            persisted_chunks.extend((key, int(c.chunk_index), c.filename, c.locator, c.text) for c in chunks)

        if not rows:
            return warnings
        try:
            with self._db() as db:
                for source_path, filename, modified_ns, size_bytes, indexed_at in rows:
                    db.execute("DELETE FROM rag_documents WHERE source_path=?", (source_path,))
                    db.execute(
                        "INSERT INTO rag_documents(source_path, filename, modified_ns, size_bytes, indexed_at) VALUES (?, ?, ?, ?, ?)",
                        (source_path, filename, modified_ns, size_bytes, indexed_at),
                    )
                db.executemany(
                    "INSERT INTO rag_chunks(source_path, chunk_index, filename, locator, text) VALUES (?, ?, ?, ?, ?)",
                    persisted_chunks,
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de mettre à jour les documents indexés : {exc}") from exc
        return warnings

    def delete_rag_documents(self, source_paths: Iterable[str | Path]) -> int:
        """Delete only selected documents and their chunks; never touch source files or chats."""
        paths = list(dict.fromkeys(str(Path(p).expanduser().resolve()) for p in source_paths))
        if not paths:
            return 0
        deleted = 0
        try:
            with self._db() as db:
                for source_path in paths:
                    cursor = db.execute("DELETE FROM rag_documents WHERE source_path=?", (source_path,))
                    deleted += cursor.rowcount
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de supprimer les index documentaires sélectionnés : {exc}") from exc
        return deleted

    def list_stale_rag_files(self) -> list[str]:
        """Return indexed source paths whose file metadata changed or whose file disappeared."""
        try:
            with self._db() as db:
                docs = db.execute("SELECT source_path, modified_ns, size_bytes FROM rag_documents").fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de vérifier l'actualité des documents : {exc}") from exc
        stale: list[str] = []
        for row in docs:
            path = Path(str(row["source_path"]))
            try:
                stat = path.stat()
                valid = path.is_file() and int(stat.st_mtime_ns) == int(row["modified_ns"]) and int(stat.st_size) == int(row["size_bytes"])
            except OSError:
                valid = False
            if not valid:
                stale.append(str(path))
        return stale

    def load_rag_index(self) -> RAGIndex:
        """Load only chunks whose original files still exist and have unchanged size/time."""
        result = RAGIndex()
        try:
            with self._db() as db:
                docs = db.execute(
                    "SELECT source_path, filename, modified_ns, size_bytes FROM rag_documents ORDER BY filename"
                ).fetchall()
                chunks = db.execute(
                    "SELECT source_path, chunk_index, filename, locator, text FROM rag_chunks ORDER BY source_path, chunk_index"
                ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de restaurer l'index documentaire : {exc}") from exc

        valid_paths: set[str] = set()
        for doc in docs:
            path = Path(doc["source_path"])
            try:
                stat = path.stat()
                valid = path.is_file() and int(stat.st_mtime_ns) == int(doc["modified_ns"]) and int(stat.st_size) == int(doc["size_bytes"])
            except OSError:
                valid = False
            if valid:
                valid_paths.add(str(path))
                result.indexed_files.append(str(path))
            else:
                # Keep stale/missing source paths visible after restart. The database
                # record and chunks remain untouched, but they are never returned as
                # active evidence until the original source is available and current.
                result.stale_files.append(str(path))
                if not path.is_file():
                    result.warnings.append(
                        f"{doc['filename']} est introuvable à son emplacement d'origine. Son index local est conservé, mais ses passages sont exclus du RAG. Restaure le fichier au même emplacement ou supprime explicitement son index."
                    )
                else:
                    result.warnings.append(
                        f"{doc['filename']} a changé depuis son indexation. Ses anciens passages sont exclus du RAG ; réindexe ce fichier pour actualiser l'index."
                    )

        for row in chunks:
            source_path = str(row["source_path"])
            if source_path not in valid_paths:
                continue
            result.chunks.append(DocumentChunk(
                source_path=source_path,
                filename=str(row["filename"]),
                locator=str(row["locator"]),
                text=str(row["text"]),
                chunk_index=int(row["chunk_index"]),
            ))
        return result

    def integrity_check(self) -> list[str]:
        """Run SQLite's built-in integrity check and return its diagnostic lines."""
        try:
            with self._db() as db:
                rows = db.execute("PRAGMA integrity_check").fetchall()
            return [str(row[0]) for row in rows] or ["No result returned"]
        except sqlite3.Error as exc:
            raise StorageError(f"Impossible de vérifier la base locale : {exc}") from exc

    def backup_to(self, destination: str | Path) -> Path:
        """Create a consistent, integrity-checked SQLite backup without copying live WAL files."""
        target = Path(destination).expanduser().resolve()
        source = self.path.expanduser().resolve()
        if target == source:
            raise StorageError("La sauvegarde doit être enregistrée dans un autre fichier que la base active.")
        if target.exists() and target.is_dir():
            raise StorageError("La destination de sauvegarde est un dossier, pas un fichier.")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                with self._db() as source_db:
                    destination_db = sqlite3.connect(temporary)
                    try:
                        source_db.backup(destination_db)
                        check = [str(row[0]) for row in destination_db.execute("PRAGMA integrity_check").fetchall()]
                        if check != ["ok"]:
                            raise StorageError("La vérification d'intégrité de la sauvegarde a échoué : " + "; ".join(check))
                    finally:
                        destination_db.close()
                os.replace(temporary, target)
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            return target
        except StorageError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(f"Impossible de créer la sauvegarde locale : {exc}") from exc

    def close(self) -> None:
        """Connections are short-lived; kept for symmetric lifecycle use in tests/UI."""
        return None
