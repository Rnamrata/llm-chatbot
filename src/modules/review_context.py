"""What the model reads in a code-review chat (Ask Lens).

The whole file, with a number at the start of every line, and every finding. A
search for "the 5 best-matching passages" left out the methods people asked about,
and code without line numbers made the model guess them. Only a file too long for
the model's context falls back to searching, and those passages are numbered too.
"""
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src import config

SEVERITY_ORDER = ['critical', 'high', 'medium', 'low', 'info']


def number_lines(code, start=1):
    """'35:     return total / len(products)': the number, a colon and a space, then the line as it is"""
    lines = code.split('\n')
    if lines and lines[-1] == '':
        lines.pop()  # a file's final newline is not a line of its own
    width = len(str(start + len(lines) - 1))
    return '\n'.join(f'{n:>{width}}: {line}' for n, line in enumerate(lines, start))


def findings_text(findings):
    """Every finding on its own line, most serious first, then by line"""
    if not findings:
        return 'The automated review found no issues.'

    rank = {severity: i for i, severity in enumerate(SEVERITY_ORDER)}
    ordered = sorted(findings, key=lambda f: (rank.get(f.get('severity'), len(rank)), f.get('line') or 0))

    rows = []
    for f in ordered:
        where = f"line {f['line']}" if f.get('line') else 'whole file'
        labels = ', '.join(x for x in (f.get('severity') or 'info', f.get('category'), f.get('finding_source')) if x)
        row = f"- {where} ({labels}): {f.get('message', '')}"
        if f.get('suggestion'):
            row += f" Suggested fix: {f['suggestion']}"
        rows.append(row)
    return '\n'.join(rows)


def _chunk_body(doc):
    """A stored code chunk without the 'File.java, lines 3-20:' header file_manager adds"""
    m = doc.metadata
    header = f"{m.get('source')}, lines {m.get('start_line')}-{m.get('end_line')}:\n"
    text = doc.page_content
    return text[len(header):] if text.startswith(header) else text


class ReviewContextRetriever(BaseRetriever):
    """
    Gives the chain one review's code and findings instead of a similarity search.
    Same interface as any LangChain retriever, so ChatSession uses it like the others.
    """
    vector_store: Any = None
    review_id: str = ''
    k: int = 5
    max_code_chars: int = config.REVIEW_FULL_CODE_CHARS

    @property
    def filter(self):
        """The same scope a search retriever for this review would use"""
        return {'review_id': self.review_id}

    def _get_relevant_documents(self, query, *, run_manager=None):
        stored = self.vector_store.vectorstore.get(where=self.filter, include=['documents', 'metadatas'])

        source = None
        findings = []
        for text, metadata in zip(stored['documents'], stored['metadatas']):
            metadata = metadata or {}
            if metadata.get('type') == 'code_review_source':
                source = Document(page_content=text, metadata=metadata)
            elif metadata.get('type') == 'review_finding':
                findings.append(metadata)

        if source is None:
            return []

        filename = source.metadata.get('source') or 'the file'
        language = source.metadata.get('language') or 'unknown'
        code = source.page_content

        if len(code) <= self.max_code_chars:
            code_doc = Document(
                page_content=(
                    f'File: {filename} (language: {language}). The complete file, '
                    f'each line starting with its line number:\n{number_lines(code)}'
                ),
                metadata={**source.metadata, 'type': 'code_review_source'},
            )
            code_docs = [code_doc]
        else:
            code_docs = self._passages(query, filename, language)

        findings_doc = Document(
            page_content=f'Findings of the automated review of {filename}:\n{findings_text(findings)}',
            metadata={'source': filename, 'type': 'review_findings', 'review_id': self.review_id},
        )
        return code_docs + [findings_doc]

    def _passages(self, query, filename, language):
        """For a file too long to send whole: the passages that best match the question, numbered"""
        found = self.vector_store.search(query, k=self.k * 3, filter=self.filter)
        chunks = [d for d in found if d.metadata.get('type') == 'code_review'][:self.k]
        chunks.sort(key=lambda d: d.metadata.get('start_line') or 0)

        docs = []
        for i, chunk in enumerate(chunks):
            start = chunk.metadata.get('start_line') or 1
            header = (
                f'File: {filename} (language: {language}). The file is too long to show whole; '
                'these are the parts that best match the question, each line starting with its line number:\n'
                if i == 0 else '...\n'
            )
            docs.append(Document(page_content=header + number_lines(_chunk_body(chunk), start), metadata=chunk.metadata))
        return docs
