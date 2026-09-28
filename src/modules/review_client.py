import requests
from src import config

class ReviewClient:
    """Client for the code review service"""

    def __init__(self, base_url=None, timeout=None):
        """
        Initialize Review Client

        Args:
            base_url: Base URL of the review service (default: config.REVIEW_SERVICE_URL)
            timeout: Request timeout in seconds (default: config.REVIEW_TIMEOUT_SECONDS)
        """

        self.base_url = (base_url or config.REVIEW_SERVICE_URL).rstrip("/")
        self.timeout = timeout or config.REVIEW_TIMEOUT_SECONDS

    def _request(self, method, path, **kwargs):
        """
        Make a request to the review service, translating network failures
        into a consistent error shape instead of raising

        Args:
            method: HTTP method ("get" or "post")
            path: Path on the review service, e.g. "/review"
            **kwargs: Extra arguments passed to requests (e.g. json=...)

        Returns:
            dict: {'success': True, 'data': dict} or {'success': False, 'error': str}
        """

        try: 
            response = requests.request(
                method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
            )
        except requests.exceptions.Timeout:
            return {'success': False, 'error': f'Review service timed out after {self.timeout}s'}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': 'Review service is unreachable (connection refused)'}
        except requests.exceptions.RequestException as e:
            return {'success': False, 'error': f'Review service request failed: {str(e)}'}

        if response.status_code != 200:
            return {
                'success': False,
                'error': f'Review service returned {response.status_code}: {response.text[:200]}'
            }

        try:
            return {'success': True, 'data': response.json()}
        except ValueError:
            return {'success': False, 'error': 'Review service returned invalid JSON'}

    def review(self, filename, language, code):
        """
        Send code to the Node review service and get back findings

        Args:
            filename: Name of the file being reviewed
            language: Language identifier (e.g. "python", "js")
            code: Source code to review

        Returns:
            dict: {'success': True, 'findings': [...]} or {'success': False, 'error': str}
        """

        result = self._request(
            'post', '/review',
            json={'filename': filename, 'language': language, 'code': code}
        )

        if not result['success']:
            return result

        return {'success': True, 'findings': result['data'].get('findings', [])}

    def health(self):
        """
        Ping the Node review service's health endpoint

        Returns:
            dict: {'success': True, 'status': ...} or {'success': False, 'error': str}
        """

        result = self._request('get', '/health')

        if not result['success']:
            return result

        return {'success': True, 'status': result['data'].get('status', 'unknown')}