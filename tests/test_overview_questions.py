import pytest

from src.modules.overview_questions import is_overview_question


@pytest.mark.parametrize('question', [
    'Summarize this', 'summarise the book', 'Give me a summary', 'Give me an overview', 'What is this about?',
    "what's it about", 'What are the main points?', 'key takeaways', 'tl;dr', 'TLDR please',
    'What is the gist of it?', 'Summarize b.txt', 'What are the main topics of the video?',
    'Summarize this for me', 'Summarize it in simple words', 'Give me a summary in 5 bullet points',
    'What is this book about?', 'Summarize the main ideas for a beginner', 'Summarize this video',
])
def test_overview_questions(question):
    assert is_overview_question(question)


@pytest.mark.parametrize('question', [
    'Summarize what it says about agents', 'What does chapter 3 say about RAG?', 'What is a transformer?',
    'main points about pricing in the video', 'Who wrote this?', 'Summarize the section on fine-tuning',
    'Summarize chapter 4', 'summarize page 12',
    'Can you summarize how the attention mechanism decides which earlier tokens matter most in a long sentence?',
    '',
])
def test_specific_questions_are_not(question):
    assert not is_overview_question(question)
