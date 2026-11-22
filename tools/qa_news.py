"""Inspect the reviewed digest or collect unreviewed official RSS candidates.

Collection deliberately cannot create a runtime QAEntry. A reviewer must read
the actual release, write an attributed summary, choose specific cues and set
a new <=48-hour review/expiry window before copying a record into data/qa/news.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias.qa_knowledge import QAKnowledge  # noqa: E402

FEED = "https://nvidianews.nvidia.com/cats/press_release.xml"
MAX_BYTES = 1_000_000


def parse_candidates(raw: bytes, *, now: datetime) -> dict:
    if len(raw) > MAX_BYTES or b"<!doctype" in raw.lower() or b"<!entity" in raw.lower():
        raise ValueError("RSS exceeds bounds or contains unsupported declarations")
    root = ET.fromstring(raw)
    rows = []
    seen = set()
    for item in root.findall("./channel/item")[:40]:
        title = " ".join((item.findtext("title") or "").split())
        url = (item.findtext("link") or "").strip()
        parsed = urlsplit(url)
        if (not title or parsed.scheme != "https" or parsed.hostname != "nvidianews.nvidia.com"
                or parsed.username or parsed.password or parsed.port is not None or url in seen):
            continue
        seen.add(url)
        published = None
        date_text = item.findtext("pubDate")
        if date_text:
            try:
                stamp = parsedate_to_datetime(date_text)
                if stamp.tzinfo is not None:
                    published = stamp.astimezone(timezone.utc).isoformat()
            except (ValueError, TypeError):
                pass
        rows.append({"title": title[:200], "url": url, "feed_published_at": published,
                     "review_status": "unreviewed", "article_read": False})
    return {"schema_version": "qa-news-candidates.v1", "feed_url": FEED,
            "collected_at": now.astimezone(timezone.utc).isoformat(),
            "runtime_eligible": False, "entries": rows}


class _OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        parsed = urlsplit(newurl)
        if (parsed.scheme != "https" or parsed.hostname != "nvidianews.nvidia.com"
                or parsed.username or parsed.password or parsed.port is not None):
            raise ValueError("RSS redirect left the official HTTPS newsroom")
        return super().redirect_request(request, fp, code, message, headers, newurl)


def collect() -> dict:
    request = urllib.request.Request(FEED, headers={"User-Agent": "YAICON-reviewed-news-candidates/1.0"})
    opener = urllib.request.build_opener(_OfficialRedirect())
    with opener.open(request, timeout=10) as response:
        raw = response.read(MAX_BYTES + 1)
    return parse_candidates(raw, now=datetime.now(timezone.utc))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate reviewed JSON and show freshness counts")
    validate.add_argument("path", type=Path)
    validate.add_argument("--at", help="Timezone-aware ISO datetime; defaults to now")
    candidates = sub.add_parser("candidates", help="Fetch official feed metadata; never auto-approve news")
    candidates.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        now = datetime.fromisoformat(args.at) if args.at else datetime.now(timezone.utc)
        if now.tzinfo is None:
            parser.error("--at must include a timezone")
        store = QAKnowledge.load(args.path)
        active = [entry for entry in store.entries if entry.reviewed_at <= now < entry.expires_at]
        print(json.dumps({"schema_valid": True, "entries": len(store.entries), "within_review_window": len(active),
                          "not_within_review_window": len(store.entries) - len(active),
                          "as_of": now.isoformat(), "claim_entailment_checked": False}, indent=2))
    else:
        if args.output.exists():
            parser.error("Output exists; choose a new candidate filename")
        result = collect()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create avoids overwriting another collector's output.
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(f"Saved {len(result['entries'])} unreviewed candidates: {args.output}")


if __name__ == "__main__":
    main()
