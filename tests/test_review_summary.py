from src.modules.review_summary import build_headline


def _finding(severity='low', message='a problem', line=1):
    return {'file': 'a.py', 'line': line, 'severity': severity, 'message': message}


def test_clean_code():
    assert build_headline([]) == 'No issues found - the code looks clean!'


def test_clean_code_but_a_check_reported_a_problem_is_not_called_clean():
    headline = build_headline([], [{'source': 'llm', 'message': 'Ollama is unreachable'}])

    assert 'looks clean' not in headline
    assert headline.startswith('No issues found by the checks that ran.')


def test_one_finding_is_singular():
    headline = build_headline([_finding('high', 'SQL injection', 9)])

    assert headline.startswith('1 issue found: 1 high.')


def test_counts_are_exact():
    headline = build_headline([_finding('high'), _finding('medium')])

    assert headline.startswith('2 issues found: 1 high, 1 medium.')


def test_counts_group_the_same_severity():
    headline = build_headline([_finding('low'), _finding('low'), _finding('low'), _finding('high')])

    assert headline.startswith('4 issues found: 1 high, 3 low.')


def test_severities_are_listed_most_serious_first_whatever_the_input_order():
    findings = [_finding('info'), _finding('low'), _finding('medium'), _finding('high'), _finding('critical')]

    assert '1 critical, 1 high, 1 medium, 1 low, 1 info' in build_headline(findings)


def test_severities_with_no_findings_are_left_out():
    headline = build_headline([_finding('high'), _finding('low')])

    assert 'medium' not in headline
    assert 'critical' not in headline


def test_a_missing_severity_counts_as_info():
    headline = build_headline([{'file': 'a.py', 'line': 1, 'message': 'x'}])

    assert headline.startswith('1 issue found: 1 info.')


def test_an_unknown_severity_is_listed_after_the_known_ones():
    headline = build_headline([_finding('weird'), _finding('high')])

    assert '1 high, 1 weird' in headline


def test_names_the_most_serious_finding_and_its_line():
    headline = build_headline([_finding('low', 'unused import', 1), _finding('high', 'SQL injection', 9)])

    assert 'Most serious: SQL injection (line 9).' in headline


def test_among_equally_serious_findings_the_first_in_the_file_is_named():
    headline = build_headline([_finding('high', 'later', 20), _finding('high', 'earlier', 4)])

    assert 'Most serious: earlier (line 4).' in headline


def test_leaves_out_the_line_when_there_is_none():
    headline = build_headline([{'file': 'a.py', 'severity': 'high', 'message': 'bad', 'line': 0}])

    assert 'Most serious: bad.' in headline
    assert '(line' not in headline


def test_a_long_message_is_shortened_and_whitespace_collapsed():
    headline = build_headline([_finding('high', 'word   ' * 60, 3)])

    named = headline.split('Most serious: ')[1]
    assert len(named) < 160
    assert named.count('   ') == 0
    assert '...' in named


def test_does_not_modify_its_inputs():
    findings = [_finding('low'), _finding('high')]
    warnings = [{'source': 'llm', 'message': 'x'}]

    build_headline(findings, warnings)

    assert [f['severity'] for f in findings] == ['low', 'high']
    assert warnings == [{'source': 'llm', 'message': 'x'}]


def test_names_the_part_of_the_review_that_had_a_problem():
    headline = build_headline([_finding('high')], [{'source': 'llm', 'message': 'Ollama is unreachable at http://x'}])

    assert 'Note: The AI reviewer reported a problem: Ollama is unreachable at http://x.' in headline


def test_the_note_is_on_its_own_paragraph_after_the_headline():
    headline = build_headline([_finding('high')], [{'source': 'llm', 'message': 'down'}])

    first, note = headline.split('\n\n')
    assert first.startswith('1 issue found')
    assert note.startswith('Note:')


def test_each_source_has_its_own_name():
    for source, label in [('llm', 'The AI reviewer'), ('eslint', 'ESLint'), ('ruff', 'Ruff'),
                          ('orchestrator', 'The review service'), ('mystery', 'A check'), (None, 'A check')]:
        assert f'{label} reported a problem' in build_headline([], [{'source': source, 'message': 'm'}])


def test_several_warnings_are_all_named():
    headline = build_headline([], [{'source': 'llm', 'message': 'one'}, {'source': 'ruff', 'message': 'two'}])

    assert 'The AI reviewer reported a problem: one.' in headline
    assert 'Ruff reported a problem: two.' in headline


def test_malformed_warnings_are_ignored():
    headline = build_headline([_finding('high')], ['text', None, 5, {'source': 'llm'}, {'source': 'llm', 'message': '   '}])

    assert 'Note' not in headline


def test_a_long_warning_is_shortened():
    headline = build_headline([], [{'source': 'llm', 'message': 'x' * 500}])

    assert len(headline) < 400
    assert '...' in headline


def test_a_warning_message_that_already_ends_with_a_period_gets_no_second_one():
    headline = build_headline([], [{'source': 'llm', 'message': 'It failed.'}])

    assert headline.endswith('It failed.')
    assert not headline.endswith('..')


def test_when_the_review_failed_it_never_says_the_code_is_clean():
    headline = build_headline([], review_failed=True)

    assert 'looks clean' not in headline
    assert 'No issues found' not in headline
    assert 'could not be reviewed' in headline


def test_when_the_review_failed_it_says_the_code_is_still_chattable():
    assert 'ask questions about it' in build_headline([], review_failed=True)