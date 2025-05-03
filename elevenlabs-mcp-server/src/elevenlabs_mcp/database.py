import aiosqlite
import json
import os
from datetime import datetime
from typing import List, Optional

from .models import AudioJob

def get_database_path() -> str:
    """Get the database path, ensuring it's in the output directory."""
    # First try to get the output directory from the server instance
    output_dir = os.getenv("ELEVENLABS_OUTPUT_DIR")
    if not output_dir:
        # Fall back to default output directory in project root
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    return os.path.join(output_dir, "voiceover_history.db")

DATABASE_PATH = get_database_path()

CREATE_VOICES_TABLE = """
CREATE TABLE IF NOT EXISTS voices (
    voice_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    labels TEXT,  -- JSON string
    description TEXT,
    preview_url TEXT,
    high_quality_base_model_ids TEXT,  -- JSON string
    last_updated TEXT NOT NULL
)
"""

CREATE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS audio_jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    script_parts TEXT NOT NULL,  -- JSON string
    output_file TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    total_parts INTEGER NOT NULL DEFAULT 1,
    completed_parts INTEGER NOT NULL DEFAULT 0
)
"""

class Database:
    CACHE_DURATION_SECONDS = 24 * 60 * 60  # 24 hours
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    async def initialize(self):
        """Initialize database and create tables if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            # Create tables one at a time
            await db.execute(CREATE_VOICES_TABLE)
            await db.execute(CREATE_JOBS_TABLE)
            await db.commit()

    async def insert_job(self, job: AudioJob) -> None:
        """Insert a new audio job into the database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO audio_jobs 
                (id, status, script_parts, output_file, error, created_at, updated_at, total_parts, completed_parts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.status,
                    json.dumps(job.script_parts),
                    job.output_file,
                    job.error,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                    job.total_parts,
                    job.completed_parts
                )
            )
            await db.commit()

    async def update_job(self, job: AudioJob) -> None:
        """Update an existing audio job in the database."""
        job.updated_at = datetime.utcnow()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE audio_jobs 
                SET status = ?, script_parts = ?, output_file = ?, error = ?, 
                    updated_at = ?, total_parts = ?, completed_parts = ?
                WHERE id = ?
                """,
                (
                    job.status,
                    json.dumps(job.script_parts),
                    job.output_file,
                    job.error,
                    job.updated_at.isoformat(),
                    job.total_parts,
                    job.completed_parts,
                    job.id
                )
            )
            await db.commit()

    async def get_job(self, job_id: str) -> Optional[AudioJob]:
        """Get a specific audio job by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM audio_jobs WHERE id = ?", (job_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return AudioJob.from_dict({
                    "id": row["id"],
                    "status": row["status"],
                    "script_parts": json.loads(row["script_parts"]),
                    "output_file": row["output_file"],
                    "error": row["error"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "total_parts": row["total_parts"],
                    "completed_parts": row["completed_parts"]
                })

    async def get_all_jobs(self) -> List[AudioJob]:
        """Get all audio jobs."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM audio_jobs ORDER BY created_at DESC") as cursor:
                rows = await cursor.fetchall()
                return [AudioJob.from_dict({
                    "id": row["id"],
                    "status": row["status"],
                    "script_parts": json.loads(row["script_parts"]),
                    "output_file": row["output_file"],
                    "error": row["error"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "total_parts": row["total_parts"],
                    "completed_parts": row["completed_parts"]
                }) for row in rows]

    async def delete_job(self, job_id: str) -> bool:
        """Delete an audio job by ID. Returns True if job was deleted."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM audio_jobs WHERE id = ?", (job_id,))
            deleted = cursor.rowcount > 0
            await db.commit()
            return deleted

    async def cleanup(self) -> None:
        """Delete the database file. Useful for testing."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    async def upsert_voices(self, voices: List[dict]) -> None:
        """Insert or update voice data in the database."""
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.utcnow().isoformat()
            for voice in voices:
                await db.execute(
                    """
                    INSERT INTO voices 
                    (voice_id, name, category, labels, description, preview_url, 
                     high_quality_base_model_ids, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(voice_id) DO UPDATE SET
                        name = excluded.name,
                        category = excluded.category,
                        labels = excluded.labels,
                        description = excluded.description,
                        preview_url = excluded.preview_url,
                        high_quality_base_model_ids = excluded.high_quality_base_model_ids,
                        last_updated = excluded.last_updated
                    """,
                    (
                        voice["voice_id"],
                        voice["name"],
                        voice["category"],
                        json.dumps(voice["labels"]),
                        voice["description"],
                        voice["preview_url"],
                        json.dumps(voice["high_quality_base_model_ids"]),
                        now
                    )
                )
            await db.commit()

    async def get_voices(self, max_age_seconds: Optional[int] = None) -> tuple[List[dict], bool]:
        """
        Get all voices from the database.
        Returns tuple of (voices, needs_refresh) where needs_refresh indicates if cache is stale.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM voices ORDER BY name") as cursor:
                rows = await cursor.fetchall()
                
                voices = []
                needs_refresh = False
                
                if not rows:
                    needs_refresh = True
                else:
                    max_age = max_age_seconds or self.CACHE_DURATION_SECONDS
                    now = datetime.utcnow()
                    
                    for row in rows:
                        last_updated = datetime.fromisoformat(row["last_updated"])
                        age = (now - last_updated).total_seconds()
                        
                        if age > max_age:
                            needs_refresh = True
                            
                        voices.append({
                            "voice_id": row["voice_id"],
                            "name": row["name"],
                            "category": row["category"],
                            "labels": json.loads(row["labels"]),
                            "description": row["description"],
                            "preview_url": row["preview_url"],
                            "high_quality_base_model_ids": json.loads(row["high_quality_base_model_ids"])
                        })
                
                return voices, needs_refresh
