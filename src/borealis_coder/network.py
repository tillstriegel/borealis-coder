"""Shared HTTP redirect policy for credential-bearing standard-library clients."""

from __future__ import annotations

import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def same_origin_redirect_url(source_url: str, location: str) -> str:
    """Resolve one redirect and reject origin changes or credential-bearing URLs."""

    target_url = urllib.parse.urljoin(source_url, location).replace(" ", "%20")
    source_origin = _http_origin(source_url)
    target_origin = _http_origin(target_url)
    if source_origin is None or target_origin is None or source_origin != target_origin:
        raise ValueError("Cross-origin HTTP redirect is not allowed")
    return target_url


class SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow normal urllib redirects only while the HTTP origin stays unchanged."""

    def redirect_request(  # type: ignore[no-untyped-def]
        self,
        request: urllib.request.Request,
        response: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        try:
            safe_url = same_origin_redirect_url(request.full_url, new_url)
        except ValueError as error:
            raise urllib.error.HTTPError(
                new_url,
                code,
                str(error),
                headers,
                response,
            ) from error
        if code in {307, 308}:
            return urllib.request.Request(
                safe_url,
                data=request.data,
                headers={**request.headers, **request.unredirected_hdrs},
                origin_req_host=request.origin_req_host,
                unverifiable=True,
                method=request.get_method(),
            )
        return super().redirect_request(
            request,
            response,
            code,
            message,
            headers,
            safe_url,
        )


def open_same_origin(
    request: urllib.request.Request,
    *,
    timeout: float,
    context: ssl.SSLContext | None = None,
) -> Any:
    """Open a request with a redirect policy that applies to every hop."""

    handlers: list[Any] = [SameOriginRedirectHandler()]
    if context is not None:
        handlers.append(urllib.request.HTTPSHandler(context=context))
    return urllib.request.build_opener(*handlers).open(request, timeout=timeout)


def _http_origin(url: str) -> tuple[str, str, int] | None:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    host = parsed.hostname
    if (
        scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return scheme, host.lower(), port or (443 if scheme == "https" else 80)
