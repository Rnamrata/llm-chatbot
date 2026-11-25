"""Whether a question asks for an overview of a whole source ("summarize this").

Decided in code, not by the model, so it is fast and the same every time. A question
that also asks about something in particular ("summarize what it says about agents")
is not an overview question: normal search answers it better.
"""
import re

OVERVIEW_WORDS = re.compile(
    r"\b(summar(?:y|ize|ise|ies|ized|ised)"
    r"|overview"
    r"|(?:main|key) (?:points?|ideas?|topics?|themes?|arguments?|takeaways?)"
    r"|takeaways?"
    r"|gist"
    r"|tl;?dr"
    r"|what(?:'s| is) (?:this|it|that|(?:the|this|that) \w+) about)\b",
    re.IGNORECASE,
)

# Something in particular: "about agents", "on fine-tuning", "chapter 4", "page 12"
SPECIFIC = re.compile(
    r"\b(?:about|on|regarding|concerning|of how|how)\s+(?!(?:this|it|that|these|them|the (?:book|video|page|article|file|document|source|notes?|paper|text))\b)\w+"
    r"|\b(?:chapter|section|page|part)s?\s*\d+"
    r"|\b(?:chapter|section|part) (?:on|about)\b",
    re.IGNORECASE,
)

# A longer question is asking about something in particular
MAX_WORDS = 12


def is_overview_question(question):
    text = (question or '').strip()
    if not text or not OVERVIEW_WORDS.search(text):
        return False
    if len(text.split()) > MAX_WORDS:
        return False
    return not SPECIFIC.search(text)
