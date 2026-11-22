"""Tests for the media range proxy in resources/lib/range_proxy.py."""

import http.server
import io
import socket
import threading
import time
import urllib.parse
import urllib.request

import pytest

import resources.lib.range_proxy as range_proxy
from resources.lib.range_proxy import (
    MediaRangeProxy,
    _SequentialReader,
    _Stopped,
    create_ssl_context,
    is_proxied_url,
    keep_highest_video_quality,
    parse_range,
)

PAYLOAD = bytes(range(256)) * 40  # 10240 bytes


class FakeCdn:
    """Local HTTP server that serves PAYLOAD and records each request."""

    def __init__(self, honor_ranges: bool, delay_per_kb: float = 0.0):
        self.honor_ranges = honor_ranges
        self.requests = []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                owner.requests.append(self.headers.get("Range"))
                requested = parse_range(self.headers.get("Range"))
                if owner.honor_ranges and requested:
                    start, end = requested
                    end = len(PAYLOAD) - 1 if end is None else end
                    body = PAYLOAD[start : end + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(PAYLOAD)}")
                else:
                    body = PAYLOAD
                    self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    if not delay_per_kb:
                        self.wfile.write(body)
                        return
                    for i in range(0, len(body), 1024):
                        time.sleep(delay_per_kb)
                        self.wfile.write(body[i : i + 1024])
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def log_message(self, format, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/video.mp4?t=exp=1~hmac=secret"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def media_proxy():
    # the fake CDN is plain http on 127.0.0.1, so allow every URL in these tests
    proxy = MediaRangeProxy(user_agent="TestAgent/1.0", allowed=lambda url: True)
    yield proxy
    proxy.stop()


def _cdn(request, honor_ranges, delay_per_kb=0.0):
    cdn = FakeCdn(honor_ranges, delay_per_kb)
    request.addfinalizer(cdn.close)
    return cdn


def _get(url, range_header=None, timeout=5):
    headers = {"Range": range_header} if range_header else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
        return resp.status, dict(resp.headers), resp.read()


class TestParseRange:
    @pytest.mark.parametrize(
        "value, expected",
        [("bytes=0-99", (0, 99)), ("bytes=1811-6102", (1811, 6102)), ("bytes=500-", (500, None))],
    )
    def test_valid(self, value, expected):
        assert parse_range(value) == expected

    @pytest.mark.parametrize("value", [None, "", "bytes=-500", "bytes=10-5", "bytes=0-1,5-9", "items=0-1"])
    def test_unsupported(self, value):
        assert parse_range(value) is None


class TestRangeEmulation:
    def test_emulates_range_when_upstream_ignores_it(self, request, media_proxy):
        cdn = _cdn(request, honor_ranges=False)

        status, headers, body = _get(media_proxy.proxied_url(cdn.url), "bytes=1811-6102")

        assert status == 206
        assert body == PAYLOAD[1811:6103]
        assert headers["Content-Range"] == f"bytes 1811-6102/{len(PAYLOAD)}"

    def test_open_ended_range_returns_rest_of_file(self, request, media_proxy):
        cdn = _cdn(request, honor_ranges=False)

        status, headers, body = _get(media_proxy.proxied_url(cdn.url), "bytes=10000-")

        assert status == 206
        assert body == PAYLOAD[10000:]

    def test_consecutive_ranges_reuse_one_upstream_connection(self, request, media_proxy):
        cdn = _cdn(request, honor_ranges=False)
        url = media_proxy.proxied_url(cdn.url)

        bodies = [_get(url, r)[2] for r in ("bytes=0-1810", "bytes=1811-6102", "bytes=8000-8999")]

        assert bodies == [PAYLOAD[0:1811], PAYLOAD[1811:6103], PAYLOAD[8000:9000]]
        assert len(cdn.requests) == 1

    def test_backward_seek_reopens_upstream(self, request, media_proxy, monkeypatch):
        monkeypatch.setattr(range_proxy, "REWIND_SIZE", 1024)
        cdn = _cdn(request, honor_ranges=False)
        url = media_proxy.proxied_url(cdn.url)

        _get(url, "bytes=5000-5999")
        _, _, body = _get(url, "bytes=100-199")

        assert body == PAYLOAD[100:200]
        assert len(cdn.requests) == 2

    def test_passes_through_when_upstream_honors_ranges(self, request, media_proxy):
        cdn = _cdn(request, honor_ranges=True)
        url = media_proxy.proxied_url(cdn.url)

        status, headers, body = _get(url, "bytes=1811-6102")
        _get(url, "bytes=7000-7099")

        assert status == 206
        assert body == PAYLOAD[1811:6103]
        assert cdn.requests == ["bytes=1811-6102", "bytes=7000-7099"]

    def test_request_without_range_returns_whole_file(self, request, media_proxy):
        cdn = _cdn(request, honor_ranges=False)

        status, _, body = _get(media_proxy.proxied_url(cdn.url))

        assert status == 200
        assert body == PAYLOAD

    def test_unknown_path_is_rejected(self, media_proxy):
        media_proxy.start()

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get(f"http://127.0.0.1:{media_proxy.port}/other")
        assert exc_info.value.code == 404

    def test_retry_after_client_timeout_continues_same_upstream_read(self, request, media_proxy, monkeypatch):
        """A player that gives up while the proxy skips ahead must not make the retry start from zero."""
        monkeypatch.setattr(range_proxy, "CHUNK_SIZE", 1024)
        monkeypatch.setattr(range_proxy, "REWIND_SIZE", 8 * 1024)
        cdn = _cdn(request, honor_ranges=False, delay_per_kb=0.05)
        url = media_proxy.proxied_url(cdn.url)

        with pytest.raises((socket.timeout, TimeoutError, urllib.error.URLError)):
            _get(url, "bytes=8000-8999", timeout=0.15)
        _, _, body = _get(url, "bytes=8000-8999")

        assert body == PAYLOAD[8000:9000]
        assert len(cdn.requests) == 1


class TestKeepHighestVideoQuality:
    MANIFEST = """<MPD><Period>
    <AdaptationSet contentType="video" mimeType="video/mp4">
      <ContentProtection value="cenc"/>
      <Representation id="720" bandwidth="3410957" width="1280" height="720"><BaseURL>a</BaseURL></Representation>
      <Representation id="1080" bandwidth="7255071" width="1920" height="1080"><BaseURL>b</BaseURL></Representation>
      <Representation id="480" bandwidth="1641933" width="854" height="480"><BaseURL>c</BaseURL></Representation>
    </AdaptationSet>
    <AdaptationSet contentType="audio" mimeType="audio/mp4">
      <Representation id="aac-hi" bandwidth="199315"><BaseURL>d</BaseURL></Representation>
      <Representation id="aac-lo" bandwidth="99315"><BaseURL>e</BaseURL></Representation>
    </AdaptationSet>
  </Period></MPD>"""

    def test_keeps_only_highest_resolution_video(self):
        result = keep_highest_video_quality(self.MANIFEST)

        assert 'id="1080"' in result
        assert 'id="720"' not in result
        assert 'id="480"' not in result
        assert '<ContentProtection value="cenc"/>' in result

    def test_audio_untouched(self):
        result = keep_highest_video_quality(self.MANIFEST)

        assert 'id="aac-hi"' in result
        assert 'id="aac-lo"' in result

    def test_equal_height_prefers_higher_bandwidth(self):
        manifest = (
            '<AdaptationSet mimeType="video/mp4">'
            '<Representation id="lo" bandwidth="5" height="1080"></Representation>'
            '<Representation id="hi" bandwidth="9" height="1080"></Representation>'
            "</AdaptationSet>"
        )

        result = keep_highest_video_quality(manifest)

        assert 'id="hi"' in result
        assert 'id="lo"' not in result

    def test_rewrite_manifest_applies_filter(self):
        media_proxy = MediaRangeProxy(user_agent="TestAgent/1.0")
        try:
            result = media_proxy.rewrite_manifest(self.MANIFEST.encode()).decode()
        finally:
            media_proxy.stop()

        assert 'id="1080"' in result
        assert 'id="720"' not in result


class TestSslContext:
    def test_loads_kodi_bundle_when_present(self, tmp_path, monkeypatch):
        bundle = tmp_path / "cacert.pem"
        bundle.write_text("placeholder")
        loaded = []
        monkeypatch.setattr(
            range_proxy.ssl.SSLContext,
            "load_verify_locations",
            lambda self, cafile=None, **kw: cafile and loaded.append(cafile),
        )

        create_ssl_context(str(bundle))

        assert loaded == [str(bundle)]

    def test_missing_bundle_falls_back_to_os_store(self, tmp_path):
        context = create_ssl_context(str(tmp_path / "missing.pem"))

        assert context.verify_mode == range_proxy.ssl.CERT_REQUIRED

    def test_invalid_bundle_is_ignored(self, tmp_path):
        bundle = tmp_path / "cacert.pem"
        bundle.write_text("not a certificate")

        context = create_ssl_context(str(bundle))

        assert context.verify_mode == range_proxy.ssl.CERT_REQUIRED


class TestSequentialReader:
    @staticmethod
    def _reader(stopping=None):
        return _SequentialReader(io.BytesIO(PAYLOAD), len(PAYLOAD), "video/mp4", stopping)

    def test_failed_write_keeps_bytes_for_next_copy(self):
        reader = self._reader()
        reader.skip_to(100)

        class GoneClient:
            def write(self, data):
                raise ConnectionAbortedError()

        with pytest.raises(ConnectionAbortedError):
            reader.copy_to(GoneClient(), 50)
        out = io.BytesIO()
        reader.copy_to(out, 50)

        assert reader.pos == 150
        assert out.getvalue() == PAYLOAD[100:150]

    def test_rewind_within_history(self):
        reader = self._reader()
        reader.skip_to(5000)

        assert reader.rewind_to(4000)
        out = io.BytesIO()
        reader.copy_to(out, 100)
        assert out.getvalue() == PAYLOAD[4000:4100]

    def test_rewind_beyond_history_is_refused(self, monkeypatch):
        monkeypatch.setattr(range_proxy, "REWIND_SIZE", 1000)
        reader = self._reader()
        reader.skip_to(5000)

        assert not reader.rewind_to(3000)
        assert reader.pos == 5000

    def test_stop_aborts_reads(self):
        stopping = threading.Event()
        reader = self._reader(stopping)
        stopping.set()

        with pytest.raises(_Stopped):
            reader.skip_to(100)


class TestRewriteManifest:
    def test_rewrites_only_cdn_base_urls(self):
        media_proxy = MediaRangeProxy(user_agent="TestAgent/1.0")
        cdn_url = "https://vod-fy-mod.crunchyrollcdn.com/static/video.mp4?t=exp=1~acl=/static/*~hmac=abc&x=1"
        manifest = (
            "<MPD><BaseURL>https://other.example.com/a.mp4</BaseURL>"
            f"<BaseURL>{cdn_url.replace('&', '&amp;')}</BaseURL></MPD>"
        ).encode()

        try:
            rewritten = media_proxy.rewrite_manifest(manifest).decode()
        finally:
            media_proxy.stop()

        assert "<BaseURL>https://other.example.com/a.mp4</BaseURL>" in rewritten
        proxied = rewritten.split("<BaseURL>")[2].split("</BaseURL>")[0].replace("&amp;", "&")
        assert proxied.startswith("http://127.0.0.1:")
        assert urllib.parse.parse_qs(urllib.parse.urlsplit(proxied).query)["url"] == [cdn_url]


class TestAllowedUrls:
    @pytest.mark.parametrize(
        "url, allowed",
        [
            ("https://vod-fy-mod.crunchyrollcdn.com/v.mp4", True),
            ("https://vod-gcp.crunchyrollcdn.com/v.mp4", True),
            ("http://vod-gcp.crunchyrollcdn.com/v.mp4", False),
            ("https://crunchyrollcdn.com.evil.example/v.mp4", False),
            ("https://example.com/v.mp4", False),
            ("file:///etc/passwd", False),
        ],
    )
    def test_is_proxied_url(self, url, allowed):
        assert is_proxied_url(url) is allowed

    def test_proxy_rejects_non_cdn_urls(self):
        media_proxy = MediaRangeProxy(user_agent="TestAgent/1.0")
        try:
            url = media_proxy.proxied_url("https://example.com/secret")
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                _get(url)
            assert exc_info.value.code == 403
        finally:
            media_proxy.stop()
