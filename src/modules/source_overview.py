"""An overview of a whole source: what it is about and its main topics.

A book doesn't fit the model's context, and summarizing all of it would take 40-60
calls. Instead its chunks are grouped by topic using the embeddings already stored,
one passage per group is picked, and those few passages are summarized. A source
short enough to fit is summarized whole.
"""
import math
import os
import re

import numpy as np

# Fewer chunks than this are always used whole: there is nothing to group
MIN_TOPICS = 5
KMEANS_ITERATIONS = 30


def topic_count(n, topics):
    """How many groups a source of n chunks is split into"""
    return min(topics, max(MIN_TOPICS, round(math.sqrt(n))))


def _reading_order(chunks):
    """Chunks by their position in the source; those without one keep their stored order"""
    return sorted(range(len(chunks)), key=lambda i: (chunks[i].get('start_line') is None, chunks[i].get('start_line') or 0, i))


def _kmeans(points, k, seed):
    """Group labels and centres for unit vectors: k-means++ start, cosine distance"""
    rng = np.random.default_rng(seed)
    n = len(points)
    centres = [points[rng.integers(n)]]
    for _ in range(1, k):
        distance = np.min(1 - points @ np.array(centres).T, axis=1).clip(min=0)
        total = distance.sum()
        centres.append(points[rng.choice(n, p=distance / total)] if total > 0 else points[rng.integers(n)])
    centres = np.array(centres)

    labels = np.zeros(n, dtype=int)
    for _ in range(KMEANS_ITERATIONS):
        new_labels = np.argmax(points @ centres.T, axis=1)
        for g in range(k):
            members = points[new_labels == g]
            if len(members):
                centre = members.mean(axis=0)
                norm = np.linalg.norm(centre)
                centres[g] = centre / norm if norm else centre
            else:
                # An empty group starts again from the point farthest from its own centre
                far = np.argmin(np.max(points @ centres.T, axis=1))
                centres[g] = points[far]
                new_labels[far] = g
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels, centres


def pick_passages(chunks, whole_chars, topics, seed=42):
    """
    The passages an overview is written from, in reading order

    Args:
        chunks: [{'text', 'start_line', 'embedding'}] for one source
        whole_chars: A source with no more text than this is used whole
        topics: The most groups a long source is split into
    """
    order = _reading_order(chunks)
    total = sum(len(c['text']) for c in chunks)
    if total <= whole_chars or len(chunks) < MIN_TOPICS:
        return [chunks[i]['text'] for i in order]

    points = np.array([c['embedding'] for c in chunks], dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.where(norms == 0, 1, norms)

    k = topic_count(len(chunks), topics)
    labels, centres = _kmeans(points, k, seed)

    picked = set()
    for g in range(k):
        members = np.flatnonzero(labels == g)
        if len(members):
            picked.add(int(members[np.argmax(points[members] @ centres[g])]))
    rank = {index: position for position, index in enumerate(order)}
    return [chunks[i]['text'] for i in sorted(picked, key=rank.get)]


# The rules for every overview. No real source is named here: an example name in a
# prompt turned up in answers before.
_RULES = """Rules:
- Use only what the text below says. Never add facts, names or numbers it doesn't give.
- Start with 2 to 4 sentences: what the source is and its main argument or purpose.
- Then the line "Key topics:" and 5 to 7 lines, each starting with "- " and naming one topic in a few words.
- Write in the same language as the text.
- Don't mention "the text", "the passages" or "the excerpts"; write about the source itself.
- No title, no closing remarks."""

OVERVIEW_PROMPT = """Write a short overview of the source "{name}", from these passages taken from across it, in reading order.

""" + _RULES + """

Passages:
{passages}

Overview:"""

COMBINE_PROMPT = """Below are overviews of consecutive sections of the source "{name}". Combine them into one short overview of the whole source.

""" + _RULES + """

Section overviews:
{passages}

Overview:"""

TOPICS_PROMPT = """List the 5 to 7 key topics of the source "{name}", from these passages taken from across it.
Write one topic per line, each line starting with "- ", a few words each. Use only what the passages say. Nothing else.

Passages:
{passages}

Key topics:"""

_SEPARATOR = '\n\n---\n\n'
MAX_TOPICS = 7
# A line the model meant as a bullet: "-", "*", "•", "·", "1." and the like
_BULLET = re.compile(r'^\s*(?:[-*•·●▪‣]|\d+[.)])\s*(.*?)\s*$')


def readable_name(name):
    """A file name as a title ("LangChain_for_Life.pdf" → "LangChain for Life"); other names as they are"""
    base, ext = os.path.splitext(name)
    if ext and len(ext) <= 6 and ext[1:].isalnum() and ' ' not in ext:
        name = base
    return ' '.join(name.replace('_', ' ').split())


def tidy_overview(text):
    """Markdown bullets ("- "), no empty ones, at most MAX_TOPICS; the model used "•" and left some blank"""
    lines, topics = [], 0
    for line in str(text).strip().splitlines():
        bullet = _BULLET.match(line)
        if bullet:
            item = bullet.group(1)
            if not item:
                continue
            topics += 1
            if topics <= MAX_TOPICS:
                lines.append(f'- {item}')
            continue
        lines.append(line.rstrip())
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


def _has_topics(text):
    return any(line.startswith('- ') for line in text.splitlines())


def _batches(passages, batch_chars):
    """Passages kept whole, in order, grouped so each group stays within batch_chars where it can"""
    batches, current, size = [], [], 0
    for passage in passages:
        if current and size + len(passage) > batch_chars:
            batches.append(current)
            current, size = [], 0
        current.append(passage)
        size += len(passage)
    if current:
        batches.append(current)
    return batches


def write_overview(llm, name, passages, batch_chars):
    """
    The overview text for a source, from its picked passages

    One call when the passages fit batch_chars; otherwise one per batch, then one
    that combines the batch overviews.
    """
    name = readable_name(name)

    def ask(prompt, items):
        return str(llm.invoke(prompt.format(name=name, passages=_SEPARATOR.join(items)))).strip()

    batches = _batches(passages, batch_chars)
    if len(batches) == 1:
        final_items, text = batches[0], ask(OVERVIEW_PROMPT, batches[0])
    else:
        final_items = [ask(OVERVIEW_PROMPT, batch) for batch in batches]
        text = ask(COMBINE_PROMPT, final_items)

    text = tidy_overview(text)
    if text and not _has_topics(text):
        # The model sometimes leaves the topic list empty: ask for the topics on their own
        topics = tidy_overview(ask(TOPICS_PROMPT, final_items))
        bullets = '\n'.join(line for line in topics.splitlines() if line.startswith('- '))
        if bullets:
            text = f'{text}\n{bullets}' if text.endswith('Key topics:') else f'{text}\n\nKey topics:\n{bullets}'
    return text
