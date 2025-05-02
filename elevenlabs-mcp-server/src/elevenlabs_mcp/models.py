from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class ScriptPart:
    text: str
    voice_id: Optional[str] = None
    actor: Optional[str] = None

@dataclass
class AudioJob:
    id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    script_parts: List[Dict]
    output_file: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()
    total_parts: int = 1
    completed_parts: int = 0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "status": self.status,
            "script_parts": self.script_parts,
            "output_file": self.output_file,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_parts": self.total_parts,
            "completed_parts": self.completed_parts
        }

    @staticmethod
    def from_dict(data: Dict) -> "AudioJob":
        return AudioJob(
            id=data["id"],
            status=data["status"],
            script_parts=data["script_parts"],
            output_file=data.get("output_file"),
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"],
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"],
            total_parts=data.get("total_parts", 1),
            completed_parts=data.get("completed_parts", 0)
        )
