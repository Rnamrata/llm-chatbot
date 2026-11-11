"""Builds the opening message of a code review from the findings themselves.

This is done in code and not by asking a language model: a small model asked to
summarize the findings invented a count ("3 issues" for 2 findings) and repeated one.
"""

SEVERITY_ORDER = ['critical', 'high', 'medium', 'low', 'info']

# How a warning from the review service names the part that had a problem
SOURCE_LABELS = {
    'llm': 'The AI reviewer',
    'eslint': 'ESLint',
    'ruff': 'Ruff',
    'orchestrator': 'The review service',
}

MAX_MESSAGE_CHARS = 120
MAX_NOTE_CHARS = 200


def _severity_of(finding):
    return finding.get('severity') or 'info'


def _rank(severity):
    """Position in SEVERITY_ORDER; an unknown severity sorts after the known ones"""
    return SEVERITY_ORDER.index(severity) if severity in SEVERITY_ORDER else len(SEVERITY_ORDER)


def _shorten(text, limit):
    text = ' '.join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + '...'


def _count_text(findings):
    """'1 high, 2 medium': most serious first, severities with no findings left out"""
    counts = {}
    for finding in findings:
        severity = _severity_of(finding)
        counts[severity] = counts.get(severity, 0) + 1

    ordered = sorted(counts, key=lambda severity: (_rank(severity), severity))
    return ', '.join(f'{counts[severity]} {severity}' for severity in ordered)


def _most_serious_text(findings):
    """The most serious finding; among equals the one nearest the top of the file"""
    top = min(findings, key=lambda f: (_rank(_severity_of(f)), f.get('line') or 0))
    message = _shorten(top.get('message') or '', MAX_MESSAGE_CHARS)
    if not message:
        return ''

    line = top.get('line')
    where = f' (line {line})' if line else ''
    return f'Most serious: {message}{where}.'


def _warning_note(warnings):
    """One 'Note: ...' line naming every part of the review that reported a problem"""
    notes = []
    for warning in warnings or []:
        if not isinstance(warning, dict):
            continue

        message = _shorten(warning.get('message') or '', MAX_NOTE_CHARS)
        if not message:
            continue

        label = SOURCE_LABELS.get(warning.get('source'), 'A check')
        note = f'{label} reported a problem: {message}'
        notes.append(note if note.endswith('.') else note + '.')

    return 'Note: ' + ' '.join(notes) if notes else ''


def build_headline(findings, warnings=None, review_failed=False):
    """
    Opening chat message for a review

    Args:
        findings: List of finding dicts with file/line/severity/message
        warnings: Warnings the review service returned (dicts with source and message)
        review_failed: True when the review service could not be reached at all, so an
                       empty findings list must not be read as "no problems"

    Returns:
        str: e.g. "2 issues found: 1 high, 1 medium. Most serious: SQL injection (line 9)."
    """
    if review_failed:
        return (
            'The code could not be reviewed automatically because the review service was '
            'unavailable. It has been stored, so you can still ask questions about it.'
        )

    if findings:
        count = len(findings)
        plural = 's' if count != 1 else ''
        headline = f'{count} issue{plural} found: {_count_text(findings)}.'
        most_serious = _most_serious_text(findings)
        if most_serious:
            headline += f' {most_serious}'
    elif warnings:
        headline = 'No issues found by the checks that ran.'
    else:
        headline = 'No issues found - the code looks clean!'

    note = _warning_note(warnings)
    return f'{headline}\n\n{note}' if note else headline