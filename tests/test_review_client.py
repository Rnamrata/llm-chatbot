import pytest
import requests as requests_lib
from unittest.mock import patch, MagicMock
from src.modules.review_client import ReviewClient


@pytest.fixture
def review_client():
    return ReviewClient(base_url='http://fake-review:4000', timeout=5)


def test_review_success(review_client):
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {
        'findings': [{'file': 'a.py', 'line': 1, 'severity': 'high', 'message': 'bad'}]
    }

    with patch('src.modules.review_client.requests.request', return_value=fake_response) as mock_request:
        result = review_client.review('a.py', 'python', 'print(1)')

    assert result['success'] is True
    assert result['findings'][0]['message'] == 'bad'
    _, kwargs = mock_request.call_args
    assert kwargs['json'] == {'filename': 'a.py', 'language': 'python', 'code': 'print(1)'}


def test_review_timeout(review_client):
    with patch('src.modules.review_client.requests.request', side_effect=requests_lib.exceptions.Timeout()):
        result = review_client.review('a.py', 'python', 'print(1)')

    assert result['success'] is False
    assert 'timed out' in result['error']


def test_review_node_down(review_client):
    with patch('src.modules.review_client.requests.request', side_effect=requests_lib.exceptions.ConnectionError()):
        result = review_client.review('a.py', 'python', 'print(1)')

    assert result['success'] is False
    assert 'unreachable' in result['error']