from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx


def validate_public_url(url: str) -> None:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not host or parsed.username or parsed.password or parsed.port not in {None, 443}:
        raise ValueError("Source acquisition requires an HTTPS URL without credentials or custom ports")
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Local-network source URLs are not supported")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if not address.is_global:
        raise ValueError("Non-public IP addresses are not supported")


def collect_source(source: dict, destination: Path, *, client: httpx.Client | None = None, maximum_bytes: int = 15_000_000) -> dict:
    identifier = source["id"]
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", identifier):
        raise ValueError("Invalid source identifier")
    if destination.is_symlink():
        raise ValueError("Acquisition destination cannot be a symbolic link")
    if (destination / identifier).is_symlink():
        raise ValueError("Source directories cannot be symbolic links")
    url = source.get("acquisition_url") or source["url"]
    validate_public_url(url)
    own_client = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(30, connect=10), follow_redirects=False, headers={"User-Agent": "who-is-your-bias-research/0.1", "Accept": "text/html,text/plain,application/pdf,application/json"})
    try:
        current = url
        data = bytearray()
        mime = ""
        for _ in range(6):
            validate_public_url(current)
            with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Source redirect has no destination")
                    current = str(response.url.join(location))
                    continue
                response.raise_for_status()
                mime = response.headers.get("content-type", "").split(";")[0].lower()
                if not (mime.startswith("text/") or mime in {"application/pdf", "application/json", "application/xhtml+xml"}):
                    raise ValueError("Use the media pipeline for binary recordings; this collector accepts source documents only")
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > maximum_bytes:
                        raise ValueError("Source document exceeds the acquisition size limit")
                break
        else:
            raise ValueError("Too many source redirects")
        if not data:
            raise ValueError("Source document is empty")
        sha = hashlib.sha256(data).hexdigest()
        root = destination.resolve()
        directory = root / identifier
        if not directory.resolve().is_relative_to(root):
            raise ValueError("Source directory must remain inside the acquisition destination")
        directory.mkdir(parents=True, exist_ok=True)
        extension = ".pdf" if mime == "application/pdf" else ".json" if mime == "application/json" else ".html" if "html" in mime else ".txt"
        artifact = directory / (sha[:20] + extension)
        if artifact.is_symlink():
            raise ValueError("Acquisition artifacts cannot be symbolic links")
        if artifact.exists():
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != sha:
                raise ValueError("Existing artifact does not match its recorded content hash")
        else:
            with artifact.open("xb") as output:
                output.write(data)
        record = {"source_id": identifier, "requested_url": url, "final_url": current, "retrieved_at": datetime.now(timezone.utc).isoformat(), "mime_type": mime, "bytes": len(data), "sha256": sha, "artifact": artifact.name, "review_status": "unreviewed", "target_speaker_verified": False, "indexed": False, "kind": "publisher_document_snapshot", "note": "A downloaded page is not necessarily a clean or complete transcript. Verify speaker attribution, publisher access, event identity, and the media timeline before ingestion."}
        # Identical documents can be linked from distinct component URLs. Keep
        # each acquisition's provenance rather than overwriting the first URL.
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        manifest = directory / (sha[:20] + "-" + url_hash + ".manifest.json")
        if manifest.is_symlink():
            raise ValueError("Acquisition manifests cannot be symbolic links")
        manifest.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        return record
    finally:
        if own_client:
            client.close()
