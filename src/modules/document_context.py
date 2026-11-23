"""What the model reads in a document chat: passages from added sources, each named.

Only sources people added (files, YouTube videos, web pages) are searched, so a
code review's chunks never turn up in a document answer. Each passage carries a
label with its source's name, kind and when it was added, so the model can say
where something comes from and tell "the PDF" from "the video".
"""
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src import config
from src.modules.source_library import DOCUMENT_TYPES, display_name

DOCUMENT_FILTER = {'type': {'$in': list(DOCUMENT_TYPES)}}

KINDS = {'youtube': 'YouTube video', 'web': 'web page'}
FILE_KINDS = {'.pdf': 'PDF', '.md': 'Markdown file', '.txt': 'text file'}


def source_label(metadata):
    """'Prompt_Engineering.pdf (PDF, added 2026-11-20 14:05 UTC)': the time too, so two
    sources added on the same day can still be told apart"""
    metadata = metadata or {}
    doc_type = metadata.get('type')
    if doc_type == 'file_upload':
        source = (metadata.get('source') or '').lower()
        kind = next((k for ext, k in FILE_KINDS.items() if source.endswith(ext)), 'file')
    else:
        kind = KINDS.get(doc_type, 'source')

    details = [kind]
    added = metadata.get('added_at') or ''
    if added[:10]:
        when = added[:10] + (f' {added[11:16]} UTC' if len(added) >= 16 else '')
        details.append(f'added {when}')
    return f"{display_name(metadata)} ({', '.join(details)})"


class DocumentRetriever(BaseRetriever):
    """Searches the added sources only, and names each passage's source"""
    vector_store: Any = None
    k: int = config.RETRIEVAL_K

    @property
    def filter(self):
        return DOCUMENT_FILTER

    def _get_relevant_documents(self, query, *, run_manager=None):
        found = self.vector_store.get_retriever(k=self.k, filter=self.filter).invoke(query)
        return [
            Document(page_content=d.page_content, metadata={**d.metadata, 'source_label': source_label(d.metadata)})
            for d in found
        ]
