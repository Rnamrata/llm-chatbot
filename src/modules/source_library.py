import hashlib
import os
import threading
from datetime import datetime, timezone

# Chunk types that come from sources people add (code-review chunks are excluded)
DOCUMENT_TYPES = ('file_upload', 'youtube', 'web')

# Chunks stored by older versions of this app have no 'type' (their metadata is
# empty, or only a chunk index or markdown header), so they cannot be grouped by
# source. They are listed together under this id.
LEGACY_ID = 'legacy'


def is_legacy(metadata):
    return not (metadata or {}).get('type')


def source_id(source):
    """Short, stable, URL-safe id for a source (a file name or a URL)"""
    return hashlib.sha1(source.encode('utf-8')).hexdigest()[:12]


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def display_name(metadata):
    """What the Sources table shows: page title, video transcript name, or file name"""
    doc_type = metadata.get('type')
    if doc_type == 'web' and metadata.get('title'):
        return metadata['title']
    if doc_type == 'youtube' and metadata.get('filename'):
        name = os.path.splitext(metadata['filename'])[0]
        return name.removesuffix('_transcription')
    return metadata.get('source') or 'Untitled source'


class SourceLibrary:
    """
    Lists and deletes the sources people added (files, YouTube videos, web pages)

    Sources are not stored separately: they are worked out from the chunks in the
    vector store, grouped by their 'source' metadata. Uploads in progress are
    tracked in memory, so the list can show them before their chunks exist.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self._processing = {}
        self._lock = threading.Lock()

    # ---------- uploads in progress ----------

    def start(self, source, name, doc_type):
        with self._lock:
            self._processing[source] = {'name': name, 'type': doc_type, 'added_at': now_iso()}

    def finish(self, source):
        with self._lock:
            self._processing.pop(source, None)

    # ---------- list / get / delete ----------

    def list(self):
        result = self.vector_store.vectorstore.get(include=['metadatas'])
        groups = {}
        legacy = 0

        for metadata in result['metadatas']:
            if is_legacy(metadata):
                legacy += 1
                continue
            source = metadata.get('source')
            if metadata.get('type') not in DOCUMENT_TYPES or not source:
                continue
            item = groups.get(source)
            if item is None:
                item = groups[source] = {
                    'id': source_id(source),
                    'name': display_name(metadata),
                    'type': metadata['type'],
                    'source': source,
                    'chunks': 0,
                    'status': 'ready',
                    'added_at': metadata.get('added_at'),
                }
            item['chunks'] += 1
            if metadata.get('content_hash') and not item.get('content_hash'):
                item['content_hash'] = metadata['content_hash']

        with self._lock:
            processing = dict(self._processing)

        for source, job in processing.items():
            item = groups.get(source)
            if item is None:
                groups[source] = {
                    'id': source_id(source),
                    'name': job['name'],
                    'type': job['type'],
                    'source': source,
                    'chunks': 0,
                    'status': 'processing',
                    'added_at': job['added_at'],
                }
            else:
                item['status'] = 'processing'  # being added again

        # Newest first; sources stored before added_at existed go last
        sources = sorted(groups.values(), key=lambda s: s['added_at'] or '', reverse=True)

        if legacy:
            sources.append({
                'id': LEGACY_ID,
                'name': 'Older data without details',
                'type': 'unknown',
                'source': None,
                'chunks': legacy,
                'status': 'outdated',
                'added_at': None,
            })

        return {'success': True, 'sources': sources, 'total': len(sources)}

    def get(self, sid):
        return next((s for s in self.list()['sources'] if s['id'] == sid), None)

    def find(self, source):
        """The stored source with this file name or URL, or None"""
        return self.get(source_id(source))

    def resolve(self, ids):
        """
        The file names and URLs behind source ids, in the order given. Ids that are
        not in the library (removed, or never added) are left out.
        """
        by_id = {s['id']: s['source'] for s in self.list()['sources'] if s.get('source')}
        return [by_id[i] for i in dict.fromkeys(ids) if i in by_id]

    def chunk_ids(self, source):
        """The stored chunk ids of one source, e.g. to remove an older copy once a new one is in"""
        result = self.vector_store.vectorstore.get(where={'source': source}, include=['metadatas'])
        return [i for i, m in zip(result['ids'], result['metadatas']) if (m or {}).get('type') in DOCUMENT_TYPES]

    def delete(self, sid):
        """Removes every chunk of one source. Returns how many chunks were removed (0 if none)."""
        result = self.vector_store.vectorstore.get(include=['metadatas'])

        if sid == LEGACY_ID:
            ids = [i for i, m in zip(result['ids'], result['metadatas']) if is_legacy(m)]
        else:
            ids = [
                i for i, m in zip(result['ids'], result['metadatas'])
                if m and m.get('type') in DOCUMENT_TYPES and m.get('source') and source_id(m['source']) == sid
            ]

        if ids:
            self.vector_store.vectorstore.delete(ids=ids)
        return len(ids)