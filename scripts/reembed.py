"""Re-embed everything in the vector store with the current embedding settings.

Run this once after changing EMBED_DOCUMENT_PREFIX / EMBED_QUERY_PREFIX (or the
embedding model): vectors made the old way do not match questions embedded the new
way, so search would quietly get worse. The text and metadata of each chunk stay
exactly as they are; only its vector is replaced, in place, under the same id.

Stop Flask first, and copy chroma_db/ somewhere as a backup.

    .venv/bin/python scripts/reembed.py --dry-run   # count what would be re-embedded
    .venv/bin/python scripts/reembed.py
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.modules.vector_store_and_embedding import VectorStoreAndEmbedding  # noqa: E402


def reembed(store, batch_size, dry_run=False, log=print):
    collection = store.vectorstore._collection
    stored = collection.get(include=['documents', 'metadatas'])
    ids, texts, metadatas = stored['ids'], stored['documents'], stored['metadatas']
    log(f'{len(ids)} chunks in "{config.CHROMA_COLLECTION}" '
        f'(document prefix {config.EMBED_DOCUMENT_PREFIX!r}, query prefix {config.EMBED_QUERY_PREFIX!r})')
    if dry_run or not ids:
        return len(ids)

    started = time.time()
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        vectors = store.embeddings.embed_documents(texts[start:end])
        collection.upsert(ids=ids[start:end], embeddings=vectors, documents=texts[start:end], metadatas=metadatas[start:end])
        log(f'  {min(end, len(ids))}/{len(ids)} ({time.time() - started:.0f}s)')
    return len(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true', help='only count the chunks')
    parser.add_argument('--batch-size', type=int, default=config.EMBED_BATCH_SIZE)
    args = parser.parse_args()

    count = reembed(VectorStoreAndEmbedding(), args.batch_size, dry_run=args.dry_run)
    print('Nothing changed (dry run).' if args.dry_run else f'Re-embedded {count} chunks.')


if __name__ == '__main__':
    main()
