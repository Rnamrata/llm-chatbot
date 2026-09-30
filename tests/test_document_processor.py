import pytest
from src.modules.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor()


def test_markdown_chunking_keeps_original_metadata(processor):
    markdown = "# Title\n\nSome intro text.\n\n## Section\n\nMore content here.\n"
    metadata = {'source': 'notes.md', 'type': 'file_upload'}

    chunks = processor.chunkDocument(markdown, metadata)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata['source'] == 'notes.md'
        assert chunk.metadata['type'] == 'file_upload'


def test_plain_text_chunking_keeps_metadata(processor):
    text = "Just a plain paragraph with no markdown headers at all.\n" * 5
    metadata = {'source': 'notes.txt', 'type': 'file_upload'}

    chunks = processor.chunkDocument(text, metadata)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata['source'] == 'notes.txt'
        assert chunk.metadata['type'] == 'file_upload'


def test_code_chunks_have_correct_line_ranges(processor):
    code = "\n".join(f"def func_{i}():\n    return {i}\n" for i in range(3))
    metadata = {'source': 'sample.py', 'type': 'code_review', 'review_id': 'r1'}

    chunks = processor.chunkCode(code, 'python', metadata)
    total_lines = code.count('\n') + 1

    assert len(chunks) > 0
    assert chunks[0].metadata['start_line'] == 1
    assert chunks[-1].metadata['end_line'] <= total_lines

    for prev, curr in zip(chunks, chunks[1:]):
        assert curr.metadata['start_line'] >= prev.metadata['start_line']
        assert curr.metadata['end_line'] >= curr.metadata['start_line']