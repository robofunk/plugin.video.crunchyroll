# Crunchyroll
# Copyright (C) 2023 smirgol
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Local proxy that emulates HTTP byte ranges for media files.

Some network proxies strip the Range header for the Crunchyroll CDN, so every DASH
SegmentBase request returns the whole episode. This proxy turns those full responses back
into the requested ranges, reading each file forward over a single upstream connection.
"""

from __future__ import annotations

import html
import http.server
import os
import re
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request
from xml.sax.saxutils import escape

import xbmc
import xbmcvfs

from .utils.api_trace import redact_url
from .utils.logging import crunchy_log

CHUNK_SIZE = 1024 * 1024
REWIND_SIZE = 8 * CHUNK_SIZE
UPSTREAM_TIMEOUT = 30
PROXIED_HOST_SUFFIX = ".crunchyrollcdn.com"
KODI_CA_BUNDLE = "special://xbmc/system/certs/cacert.pem"

_RANGE_RE = re.compile(r"^bytes=(\d+)-(\d*)$")
_BASE_URL_RE = re.compile(r"<BaseURL>\s*(https://[^<]+?)\s*</BaseURL>")
_ADAPTATION_SET_RE = re.compile(r"<AdaptationSet\b[^>]*>.*?</AdaptationSet>", re.DOTALL)
_REPRESENTATION_RE = re.compile(r"\s*<Representation\b[^>]*>.*?</Representation>", re.DOTALL)
_PASS_THROUGH_HEADERS = ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges")
_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class _Stopped(Exception):
    """Raised inside request threads once the proxy is stopping."""


def parse_range(value: str | None) -> tuple[int, int | None] | None:
    """Parse a single 'bytes=start-[end]' range; returns None for anything else."""
    match = _RANGE_RE.match(value.strip()) if value else None
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else None
    if end is not None and end < start:
        return None
    return start, end


def is_proxied_url(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    return parts.scheme == "https" and (parts.hostname or "").endswith(PROXIED_HOST_SUFFIX)


def _int_attr(tag: str, name: str) -> int:
    match = re.search(rf'\b{name}="(\d+)"', tag)
    return int(match.group(1)) if match else 0


def keep_highest_video_quality(manifest: str) -> str:
    """Drop all but the highest-resolution video Representation of each video AdaptationSet."""

    def filter_set(match):
        adaptation_set = match.group(0)
        opening_tag = adaptation_set[: adaptation_set.index(">") + 1]
        if 'contentType="video"' not in opening_tag and 'mimeType="video/' not in opening_tag:
            return adaptation_set
        representations = list(_REPRESENTATION_RE.finditer(adaptation_set))
        if len(representations) < 2:
            return adaptation_set
        best = max(
            representations,
            key=lambda r: (_int_attr(r.group(0), "height"), _int_attr(r.group(0), "bandwidth")),
        )
        for representation in reversed(representations):
            if representation is not best:
                adaptation_set = adaptation_set[: representation.start()] + adaptation_set[representation.end() :]
        return adaptation_set

    return _ADAPTATION_SET_RE.sub(filter_set, manifest)


def create_ssl_context(kodi_bundle: str | None = None) -> ssl.SSLContext:
    """Trust the OS store plus Kodi's CA bundle, where users add certificates of TLS-inspecting proxies."""
    context = ssl.create_default_context()
    if kodi_bundle is None:
        kodi_bundle = xbmcvfs.translatePath(KODI_CA_BUNDLE)
    if isinstance(kodi_bundle, str) and os.path.isfile(kodi_bundle):
        try:
            context.load_verify_locations(cafile=kodi_bundle)
        except (OSError, ssl.SSLError) as e:
            crunchy_log(f"Could not load Kodi CA bundle: {e}", xbmc.LOGWARNING)
    return context


class _SequentialReader:
    """A full upstream response body that is only ever read forward."""

    def __init__(self, response, total: int, content_type: str, stopping: threading.Event | None = None):
        self.response = response
        self.total = total
        self.content_type = content_type
        self._stopping = stopping or threading.Event()
        # offset of the first byte not yet delivered; _pending holds bytes read from there on
        self.pos = 0
        self._pending = b""
        # the last bytes before pos, so a retry from slightly earlier can rewind
        self._history = b""

    def _consumed(self, data: bytes) -> None:
        self.pos += len(data)
        self._history = (self._history + data)[-REWIND_SIZE:]

    def rewind_to(self, offset: int) -> bool:
        back = self.pos - offset
        if back < 0 or back > len(self._history):
            return False
        if back:
            self._pending = self._history[-back:] + self._pending
            self._history = self._history[:-back]
            self.pos = offset
        return True

    def _next(self, limit: int) -> bytes:
        if self._stopping.is_set():
            raise _Stopped()
        if self._pending:
            data, self._pending = self._pending[:limit], self._pending[limit:]
            return data
        data = self.response.read(min(CHUNK_SIZE, limit))
        if not data:
            raise EOFError("upstream ended early")
        return data

    def skip_to(self, offset: int) -> None:
        while self.pos < offset:
            self._consumed(self._next(offset - self.pos))

    def copy_to(self, out, length: int) -> None:
        remaining = length
        while remaining > 0:
            data = self._next(remaining)
            try:
                out.write(data)
            except BaseException:
                self._pending = data + self._pending
                raise
            self._consumed(data)
            remaining -= len(data)

    def close(self) -> None:
        try:
            self.response.close()
        except Exception:  # noqa: BLE001 - closing a dead connection must not fail the request
            pass


class MediaRangeProxy:
    """Serves media through 127.0.0.1, emulating byte ranges when the upstream ignores them."""

    def __init__(self, user_agent: str, ssl_context: ssl.SSLContext | None = None, allowed=is_proxied_url):
        self.user_agent = user_agent
        self._allowed = allowed
        self._ssl_context = ssl_context or create_ssl_context()
        self._readers: dict[str, _SequentialReader] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._stopping = threading.Event()
        self.server = None
        self.port = None

    def start(self) -> None:
        if self.server:
            return
        self._stopping.clear()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                owner._handle_request(self)

            def log_message(self, format, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        crunchy_log(f"Media range proxy started on port {self.port}")

    def stop(self) -> None:
        self._stopping.set()
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            self.port = None
            crunchy_log("Media range proxy stopped")
        with self._guard:
            readers = list(self._readers.values())
            self._readers.clear()
        for reader in readers:
            reader.close()

    def proxied_url(self, url: str) -> str:
        self.start()
        return f"http://127.0.0.1:{self.port}/media?url={urllib.parse.quote(url, safe='')}"

    def rewrite_manifest(self, content: bytes) -> bytes:
        """Keep only the highest video quality and point the manifest's CDN BaseURLs at this proxy."""

        def replace(match):
            url = html.unescape(match.group(1))
            if not self._allowed(url):
                return match.group(0)
            return f"<BaseURL>{escape(self.proxied_url(url))}</BaseURL>"

        manifest = keep_highest_video_quality(content.decode("utf-8"))
        return _BASE_URL_RE.sub(replace, manifest).encode("utf-8")

    def _handle_request(self, handler) -> None:
        parsed = urllib.parse.urlsplit(handler.path)
        url = urllib.parse.parse_qs(parsed.query).get("url", [None])[0]
        if parsed.path != "/media" or not url:
            handler.send_error(404, "Not found")
            return
        if not self._allowed(url):
            handler.send_error(403, "Forbidden")
            return
        try:
            with self._lock_for(url):
                self._serve(handler, url, parse_range(handler.headers.get("Range")))
        except urllib.error.HTTPError as e:
            handler.send_error(e.code, "Upstream error")
        except (_Stopped, *_CLIENT_GONE):
            pass
        except Exception as e:  # noqa: BLE001 - report any upstream failure to the player
            crunchy_log(f"Media range proxy error for {redact_url(url)}: {e}", xbmc.LOGWARNING)
            try:
                handler.send_error(502, "Upstream failure")
            except Exception:  # noqa: BLE001 - headers may already be sent
                pass

    def _serve(self, handler, url: str, requested: tuple[int, int | None] | None) -> None:
        if requested is None:
            self._pass_through(handler, self._open(url, None))
            return

        start, end = requested
        reader = self._take_reader(url, start)
        if reader is not None:
            try:
                reader.skip_to(start)
            except _Stopped:
                reader.close()
                raise
            except Exception:  # noqa: BLE001 - an idle upstream connection may have been dropped
                reader.close()
                reader = None

        if reader is None:
            response = self._open(url, f"bytes={start}-{'' if end is None else end}")
            content_length = response.headers.get("Content-Length")
            if response.status == 206 or not content_length:
                self._pass_through(handler, response)
                return
            crunchy_log(f"Upstream ignored byte range, emulating it: {redact_url(url)}", xbmc.LOGDEBUG)
            reader = _SequentialReader(
                response,
                int(content_length),
                response.headers.get("Content-Type", "application/octet-stream"),
                self._stopping,
            )
            try:
                reader.skip_to(start)
            except BaseException:
                reader.close()
                raise

        end = reader.total - 1 if end is None else min(end, reader.total - 1)
        if start > end:
            reader.close()
            handler.send_error(416, "Range not satisfiable")
            return

        try:
            handler.send_response(206)
            handler.send_header("Content-Type", reader.content_type)
            handler.send_header("Content-Length", str(end - start + 1))
            handler.send_header("Content-Range", f"bytes {start}-{end}/{reader.total}")
            handler.send_header("Accept-Ranges", "bytes")
            handler.end_headers()
            reader.copy_to(handler.wfile, end - start + 1)
        except _CLIENT_GONE:
            # the player gave up, typically while we skipped ahead; keep the position for its retry
            self._park(url, reader)
            raise
        except BaseException:
            reader.close()
            raise

        self._park(url, reader)

    def _park(self, url: str, reader: _SequentialReader) -> None:
        if reader.pos >= reader.total or self._stopping.is_set():
            reader.close()
            return
        with self._guard:
            self._readers[url] = reader

    def _open(self, url: str, range_header: str | None):
        headers = {"User-Agent": self.user_agent}
        if range_header:
            headers["Range"] = range_header
        request = urllib.request.Request(url, headers=headers)
        return urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT, context=self._ssl_context)

    @staticmethod
    def _pass_through(handler, response) -> None:
        try:
            handler.send_response(response.status)
            for name in _PASS_THROUGH_HEADERS:
                if response.headers.get(name):
                    handler.send_header(name, response.headers[name])
            handler.end_headers()
            while data := response.read(CHUNK_SIZE):
                handler.wfile.write(data)
        finally:
            response.close()

    def _take_reader(self, url: str, start: int) -> _SequentialReader | None:
        with self._guard:
            reader = self._readers.pop(url, None)
        if reader is not None and not reader.rewind_to(min(start, reader.pos)):
            reader.close()
            return None
        return reader

    def _lock_for(self, url: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(url, threading.Lock())


_media_proxy: MediaRangeProxy | None = None


def get_media_range_proxy(user_agent: str) -> MediaRangeProxy:
    """Get the running media range proxy, starting it if needed."""
    global _media_proxy
    if _media_proxy is None:
        _media_proxy = MediaRangeProxy(user_agent)
    _media_proxy.user_agent = user_agent
    _media_proxy.start()
    return _media_proxy


def stop_media_range_proxy() -> None:
    """Stop the media range proxy, if one was started."""
    global _media_proxy
    if _media_proxy is not None:
        try:
            _media_proxy.stop()
        finally:
            _media_proxy = None
