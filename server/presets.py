"""
Preset and parameter history data structures for enhanced generation controls.
"""

from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel, Field


class GenerationPreset(BaseModel):
    """Represents saved combinations of generation parameters for quick access."""
    
    name: str
    parameter_values: Dict[str, float]
    creation_date: datetime = Field(default_factory=datetime.now)


class ParameterHistory(BaseModel):
    """Tracks recent parameter changes for undo/redo functionality."""
    
    changes: List[Dict] = []
    max_entries: int = 50
    
    def add_change(self, parameter_name: str, old_value: float, new_value: float, source: str = "user"):
        """Add a parameter change to history."""
        change = {
            "timestamp": datetime.now(),
            "parameter_name": parameter_name,
            "old_value": old_value,
            "new_value": new_value,
            "source": source
        }
        self.changes.append(change)
        if len(self.changes) > self.max_entries:
            self.changes.pop(0)
    
    def get_recent_changes(self, limit: int = 10) -> List[Dict]:
        """Get recent parameter changes."""
        return self.changes[-limit:]
