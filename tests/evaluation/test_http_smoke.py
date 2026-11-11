from argparse import Namespace

import httpx

from tools.evaluation.http_smoke import collect


def test_live_smoke_preserves_partial_rows_and_keeps_credentials_on_language_request(monkeypatch):
    seen = []
    secret = "model-server-only"
    monkeypatch.setenv("BIAS_LANGUAGE_API_KEY", secret)

    def handle(request):
        seen.append(request)
        if request.url.path in {"/health", "/api/health"}:
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            assert request.headers["Authorization"] == "Bearer " + secret
            return httpx.Response(200, json={"choices": [{"message": {"content": "Hello"}}]})
        assert "authorization" not in request.headers
        raise httpx.ReadError("The connection failed with a private endpoint detail", request=request)

    report = {"runs": []}
    args = Namespace(language_url="http://language", app_url="http://application", model="test")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        collect(args, report, client)
    assert len(report["runs"]) == 2
    assert report["runs"][0]["http_status"] == 200
    assert report["runs"][1]["kind"] == "application_sse"
    assert report["failure"]["type"] == "ReadError"
    assert secret not in str(report)
    assert "private endpoint" not in str(report)
