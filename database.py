from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DATABASE_PATH = Path(os.getenv("RESUME_APP_DB", "resume_shortlisting.db"))
PBKDF2_ITERATIONS = 600_000


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    with connect() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                job_title TEXT NOT NULL,
                results_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)


def hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS).hex()


def create_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if len(username) < 3:
        return False, "Username must contain at least 3 characters."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."
    salt = secrets.token_bytes(32)
    try:
        with connect() as connection:
            connection.execute(
                "INSERT INTO users(username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (username, hash_password(password, salt), salt.hex(), datetime.now(timezone.utc).isoformat()),
            )
        return True, "Account created. You can now log in."
    except sqlite3.IntegrityError:
        return False, "That username is already registered."


def authenticate(username: str, password: str) -> dict | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT id, username, password_hash, salt FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
    if row is None:
        return None
    attempted = hash_password(password, bytes.fromhex(row["salt"]))
    if not hmac.compare_digest(attempted, row["password_hash"]):
        return None
    return {"id": row["id"], "username": row["username"]}


def save_analysis(user_id: int, job_title: str, results: pd.DataFrame) -> int:
    with connect() as connection:
        cursor = connection.execute(
            "INSERT INTO analyses(user_id, job_title, results_json, created_at) VALUES (?, ?, ?, ?)",
            (user_id, job_title.strip() or "Untitled role", results.to_json(orient="records"), datetime.now(timezone.utc).isoformat()),
        )
        return int(cursor.lastrowid)


def list_analyses(user_id: int) -> list[dict]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT id, job_title, created_at FROM analyses WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def load_analysis(user_id: int, analysis_id: int) -> pd.DataFrame | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT results_json FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id)
        ).fetchone()
    return pd.DataFrame(json.loads(row["results_json"])) if row else None
