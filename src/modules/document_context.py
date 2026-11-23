"""What the model reads in a document chat: passages from added sources, each named.

Only sources people added (files, YouTube videos, web pages) are searched, so a
code review's chunks never turn up in a document answer. Each passage carries a
label with its source's name, kind and when it was added, so the model can say
where something comes from and tell "the PDF" from "the video".
"""
import os
import re
from typing import Any, List

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


def _name_forms(name):
    """
    How an answer may refer to a source: the whole name, without its extension or
    underscores, the title before " | Site" or " - Author", and the first four words of
    a long name ("Prompt Engineering for LLMs" for a book's long file name)
    """
    base = os.path.splitext(name)[0].replace('_', ' ')
    title = base.split(' | ')[0].split(' - ')[0]
    author = base.split(' - ', 1)[1] if ' - ' in base else ''  # "Title - Author", as video names often are
    forms = {name, base, title, author, ' '.join(base.split()[:4])}
    return {f.strip().lower() for f in forms if len(f.strip()) >= 8}


# Words an answer uses for a kind of source, when it names none ("the video says...")
KIND_WORDS = {
    'youtube': re.compile(r'\b(?:youtube|video)\b', re.I),
    'web': re.compile(r'\b(?:web ?page|article|blog post)\b', re.I),
    'pdf': re.compile(r'\b(?:pdf|book)\b', re.I),
}


def _kind(metadata):
    if metadata.get('type') == 'file_upload':
        return 'pdf' if (metadata.get('source') or '').lower().endswith('.pdf') else 'file'
    return metadata.get('type')


def sources_named_in(answer, documents):
    """
    The passages to show under an answer: those from the sources the answer names; if
    it names none, those of the kind it mentions ("the video", "the book"); otherwise
    all of them. A chip for a book under an answer about a video suggested the answer
    came from the book.
    """
    text = (answer or '').lower()
    named = {
        display_name(d.metadata)
        for d in documents
        if any(form in text for form in _name_forms(display_name(d.metadata)))
    }
    if named:
        return [d for d in documents if display_name(d.metadata) in named]

    kinds = {kind for kind, words in KIND_WORDS.items() if words.search(text)}
    of_kind = [d for d in documents if _kind(d.metadata) in kinds]
    return of_kind or list(documents)


class DocumentRetriever(BaseRetriever):
    """
    Searches the added sources only, and names each passage's source. A chat that has
    its own sources searches just those; one without searches every source.
    """
    vector_store: Any = None
    k: int = config.RETRIEVAL_K
    sources: List[str] = []

    @property
    def filter(self):
        if not self.sources:
            return DOCUMENT_FILTER
        return {'$and': [DOCUMENT_FILTER, {'source': {'$in': list(self.sources)}}]}

    def _get_relevant_documents(self, query, *, run_manager=None):
        if len(self.sources) < 2:
            found = self.vector_store.get_retriever(k=self.k, filter=self.filter).invoke(query)
        else:
            found = self._from_each_source(query)
        return [
            Document(page_content=d.page_content, metadata={**d.metadata, 'source_label': source_label(d.metadata)})
            for d in found
        ]

    def _from_each_source(self, query):
        """
        A chat with several sources of its own reads from each of them: every source gets a
        share of the passages (at least one), and the rest go to the best matches overall.
        Ranked in one list, a 971-passage book filled all 8 places for a general question,
        and a 94-passage video chosen with it was never read.
        """
        store = self.vector_store.vectorstore
        share = max(1, self.k // len(self.sources))
        picked = []
        seen = set()

        def keep(hits):
            for doc, score in hits:
                key = (doc.metadata.get('source'), doc.page_content)
                if key not in seen:
                    seen.add(key)
                    picked.append((doc, score))

        for source in self.sources:
            keep(store.similarity_search_with_score(query, k=share, filter={'$and': [DOCUMENT_FILTER, {'source': source}]}))
        if len(picked) < self.k:
            keep(store.similarity_search_with_score(query, k=self.k, filter=self.filter)[: self.k])
            picked = picked[: max(self.k, len(self.sources))]

        picked.sort(key=lambda hit: hit[1])  # Chroma scores are distances: closest first
        return [doc for doc, _ in picked]
