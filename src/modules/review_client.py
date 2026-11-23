import requests
from src import config

# Seconds a health check may take; a review gets config.REVIEW_TIMEOUT_SECONDS
HEALTH_TIMEOUT_SECONDS = 5


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

    def _request(self, method, path, timeout=None, **kwargs):
        """
        Make a request to the review service, translating network failures
        into a consistent error shape instead of raising

        Args:
            method: HTTP method ("get" or "post")
            path: Path on the review service, e.g. "/review"
            timeout: Seconds to wait (default: this client's review timeout)
            **kwargs: Extra arguments passed to requests (e.g. json=...)

        Returns:
            dict: {'success': True, 'data': dict} or {'success': False, 'error': str}
        """

        timeout = timeout or self.timeout
        try:
            response = requests.request(
                method, f"{self.base_url}{path}", timeout=timeout, **kwargs
            )
        except requests.exceptions.Timeout:
            return {'success': False, 'error': f'Review service timed out after {timeout}s'}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': 'Review service is unreachable (connection refused)'}
        except requests.exceptions.RequestException as e:
            return {'success': False, 'error': f'Review service request failed: {str(e)}'}

        if response.status_code != 200:
            return {
                'success': False,
                'error': self._error_message(response),
                # Callers use these to tell "the service refused this file" (400, 413)
                # or "too many reviews" (429) apart from "the service is down"
                'status': response.status_code,
                'retry_after': self._retry_after(response),
            }

        try:
            return {'success': True, 'data': response.json()}
        except ValueError:
            return {'success': False, 'error': 'Review service returned invalid JSON'}

    @staticmethod
    def _error_message(response):
        """The service's own message ({success, error, message}) or, failing that, the raw text"""
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            message = body.get('message') or body.get('error')
            if isinstance(message, str) and message.strip():
                return message.strip()
        return f'Review service returned {response.status_code}: {response.text[:200]}'

    @staticmethod
    def _retry_after(response):
        """Seconds from a Retry-After header, or None"""
        value = response.headers.get('Retry-After') if response.headers else None
        try:
            return max(0, int(value)) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clean_warnings(raw):
        """Keep only well-formed warnings (a dict with a text message); ignore anything else"""
        if not isinstance(raw, list):
            return []

        return [
            {'source': w.get('source'), 'message': w['message']}
            for w in raw
            if isinstance(w, dict) and isinstance(w.get('message'), str) and w['message'].strip()
        ]

    def review(self, filename, language, code):
        """
        Send code to the Node review service and get back findings

        Args:
            filename: Name of the file being reviewed
            language: Language identifier (e.g. "python", "js")
            code: Source code to review

        Returns:
            dict: {'success': True, 'findings': [...], 'warnings': [...]} or
                  {'success': False, 'error': str}. Warnings are problems one part of the
                  review had (for example the AI reviewer was unreachable) while the
                  rest still completed.
        """

        result = self._request(
            'post', '/review',
            json={'filename': filename, 'language': language, 'code': code}
        )

        if not result['success']:
            return result

        data = result['data']
        return {
            'success': True,
            'findings': data.get('findings', []),
            'warnings': self._clean_warnings(data.get('warnings'))
        }
    def health(self):
        """
        Ping the Node review service's health endpoint

        Returns:
            dict: {'success': True, 'status': ...} or {'success': False, 'error': str}
        """

        # A health check answers at once or not at all: it must not wait as long as a review
        result = self._request('get', '/health', timeout=HEALTH_TIMEOUT_SECONDS)

        if not result['success']:
            return result

        return {'success': True, 'status': result['data'].get('status', 'unknown')}