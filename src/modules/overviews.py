"""Overviews of sources: built in the background, kept in a JSON file, waited for by chats."""
import json
import os
import threading
from collections import deque
from pathlib import Path

from src import config
from src.modules.source_library import DOCUMENT_TYPES, display_name, now_iso, source_id
from src.modules.source_overview import pick_passages, write_overview

BUILDING, READY, FAILED = 'building', 'ready', 'failed'


class Overviews:
    """
    One overview per source, keyed by source id. A queued source is built on one
    worker thread, one at a time (Ollama runs one model at a time anyway).

    With inline=True a source is built inside queue() itself, which tests use.
    """

    def __init__(self, vector_store, llm, path, enabled=True, inline=False):
        self.vector_store = vector_store
        self.llm = llm
        self.path = Path(path)
        self.enabled = enabled
        self.inline = inline
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._queue = deque()  # sources (file names or URLs) waiting to be built
        self._thread = None
        self._entries = self._load()

    # ---------- reading ----------

    def get(self, sid):
        """{status, text, built_at, error} for a source, or None when it has none yet"""
        with self._lock:
            entry = self._entries.get(sid)
            return dict(entry) if entry else None

    def status(self, sid):
        entry = self.get(sid)
        return entry['status'] if entry else None

    def wait(self, sid, timeout):
        """The entry once it is no longer being built, or as it is when the timeout passes"""
        with self._changed:
            self._changed.wait_for(lambda: (self._entries.get(sid) or {}).get('status') != BUILDING, timeout)
            entry = self._entries.get(sid)
            return dict(entry) if entry else None

    # ---------- changing ----------

    def queue(self, source, front=False):
        """Builds (or rebuilds) a source's overview; front=True goes before everything waiting"""
        if not self.enabled:
            return
        sid = source_id(source)
        with self._changed:
            self._entries[sid] = {'status': BUILDING, 'text': None, 'built_at': None, 'error': None}
            if source in self._queue:
                self._queue.remove(source)
            if front:
                self._queue.appendleft(source)
            else:
                self._queue.append(source)
            self._changed.notify_all()
        if self.inline:
            self._build_next()
        else:
            self.start()

    def delete(self, sid):
        with self._changed:
            removed = self._entries.pop(sid, None)
            self._queue = deque(s for s in self._queue if source_id(s) != sid)
            self._save_locked()
            self._changed.notify_all()
        return removed is not None

    def start(self):
        """Starts the worker thread (once); nothing to start when inline or switched off"""
        if self.inline or not self.enabled or self._thread:
            return
        self._thread = threading.Thread(target=self._run, name='overviews', daemon=True)
        self._thread.start()

    # ---------- building ----------

    def _run(self):
        while True:
            with self._changed:
                self._changed.wait_for(lambda: self._queue)
            self._build_next()

    def _build_next(self):
        with self._lock:
            if not self._queue:
                return
            source = self._queue.popleft()
        sid = source_id(source)
        try:
            result = {'status': READY, 'text': self._build(source), 'built_at': now_iso(), 'error': None}
        except Exception as e:
            print(f"⚠️  Could not write the overview of {source}: {e}")
            result = {'status': FAILED, 'text': None, 'built_at': None, 'error': str(e)}
        with self._changed:
            # Deleted (or queued again) while it was being built: this result is stale
            if (self._entries.get(sid) or {}).get('status') == BUILDING and source not in self._queue:
                self._entries[sid] = result
                self._save_locked()
            self._changed.notify_all()

    def _build(self, source):
        stored = self.vector_store.vectorstore.get(
            where={'source': source}, include=['documents', 'metadatas', 'embeddings']
        )
        embeddings = stored.get('embeddings')
        embeddings = [] if embeddings is None else list(embeddings)
        chunks, name = [], source
        for i, (text, metadata) in enumerate(zip(stored['documents'], stored['metadatas'])):
            metadata = metadata or {}
            if metadata.get('type') not in DOCUMENT_TYPES:
                continue
            name = display_name(metadata)
            chunks.append({
                'text': text,
                'start_line': metadata.get('start_line'),
                'embedding': list(embeddings[i]) if i < len(embeddings) else [0.0],
            })
        if not chunks:
            raise ValueError('This source has no text to summarize')

        passages = pick_passages(chunks, config.OVERVIEW_WHOLE_CHARS, config.OVERVIEW_TOPICS)
        text = write_overview(self.llm, name, passages, config.OVERVIEW_BATCH_CHARS)
        if not text:
            raise ValueError('The model returned an empty overview')
        return text

    # ---------- the file ----------

    def _load(self):
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            entries = data.get('overviews', {}) if isinstance(data, dict) else {}
        except (OSError, ValueError) as e:
            print(f"⚠️  Could not read {self.path} ({e}); overviews start empty")
            return {}
        # One still being built when the app stopped is built again when it is needed
        return {sid: e for sid, e in entries.items() if isinstance(e, dict) and e.get('status') in (READY, FAILED)}

    def _save_locked(self):
        """Writes only finished overviews; a temp file and rename keep the file whole"""
        finished = {sid: e for sid, e in self._entries.items() if e['status'] != BUILDING}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'version': 1, 'overviews': finished}, indent=1), encoding='utf-8')
        os.replace(tmp, self.path)
