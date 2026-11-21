import json
import os
import threading
from pathlib import Path


class SessionStore:
    """
    Keeps chat sessions in a JSON file, so History and open chats survive a restart.
    Only plain data is saved (titles, messages, sources); the LangChain chain is
    rebuilt from the messages when a session is used again.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self):
        """All saved sessions as {session_id: record}; {} if there is no file yet"""
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            # Keep the broken file for inspection rather than overwriting it
            broken = self.path.with_suffix('.broken.json')
            print(f"⚠️  Could not read {self.path} ({e}); moved it to {broken.name}")
            os.replace(self.path, broken)
            return {}
        sessions = data.get('sessions') if isinstance(data, dict) else None
        return sessions if isinstance(sessions, dict) else {}

    def save(self, records):
        """Writes every session at once; a temp file and rename keep it whole if Flask stops mid-write"""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps({'version': 1, 'sessions': records}, indent=1), encoding='utf-8')
            os.replace(tmp, self.path)