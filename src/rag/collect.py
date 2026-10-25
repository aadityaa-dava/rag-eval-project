"""Download source documents into data/raw, validate them, and record a manifest."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
MANIFEST = Path("data/manifest.json")  # committed, unlike data/raw
HEADERS = {"User-Agent": "rag-eval-project/0.1 (portfolio research; https://github.com/aadityaa-dava/rag-eval-project)"}

# Fetched by script
SOURCES = [
    {"name": "fr_pccp_final_notice_2024.html",
     "url": "https://www.govinfo.gov/content/pkg/FR-2024-12-04/html/2024-28361.htm"},
    {"name": "fr_ai_lifecycle_draft_notice_2025.pdf",
     "url": "https://www.govinfo.gov/content/pkg/FR-2025-01-07/pdf/2024-31543.pdf"},
    {"name": "fr_cybersecurity_notice_2023.html",
     "url": "https://www.govinfo.gov/content/pkg/FR-2023-09-27/html/2023-20955.htm"},
    {"name": "fr_cybersecurity_notice_2025.html",
     "url": "https://www.govinfo.gov/content/pkg/FR-2025-06-27/html/2025-11669.htm"},
    {"name": "fr_human_factors_notice_2016.html",
     "url": "https://www.govinfo.gov/content/pkg/FR-2016-02-03/html/2016-01887.htm"},
]

# Downloaded by hand in a browser (the host blocks scripts); must already be in data/raw
MANUAL = [
    {"name": "fda_ai_lifecycle_draft_2025.pdf",
     "url": "https://www.fda.gov/media/184856/download"},
    {"name": "fda_premarket_software_content_2023.pdf",
     "url": "https://www.fda.gov/media/153781/download"},
    {"name": "fda_human_factors_2016.pdf",
     "url": "https://fda.gov/media/80481/download"},
    {"name": "fda_cybersecurity_premarket_2026.pdf",
     "url": "https://www.fda.gov/media/119933/download"},
    {"name": "fda_pccp_webinar_transcript_2025.pdf",
     "url": "https://www.fda.gov/media/187905/download"},
    {"name": "fda_cybersecurity_webinar_transcript_2023.pdf",
     "url": "https://fda.gov/media/173984/download"},
]


def validate(name: str, content: bytes):
    """Return an error string if content is not the real document, else None."""
    if name.endswith(".pdf"):
        return None if content[:5] == b"%PDF-" else f"not a PDF (starts with {content[:20]!r})"
    html = content.decode("utf-8", errors="ignore")
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    title = m.group(1).strip() if m else ""
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    if "request access" in title.lower() or "access denied" in title.lower():
        return f"looks like an access gate: title={title!r}"
    if len(text) < 5000:
        return f"too little text ({len(text)} chars)"
    return None


def make_record(src: dict, content: bytes, **extra) -> dict:
    return {
        **src,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for src in SOURCES:
        try:
            resp = requests.get(src["url"], headers=HEADERS, timeout=60)
            resp.raise_for_status()
            err = validate(src["name"], resp.content)
            if err:
                print(f"REJECTED {src['name']}  {err}")
                continue
            (RAW_DIR / src["name"]).write_bytes(resp.content)
            records.append(make_record(src, resp.content, acquired="script"))
            print(f"OK       {src['name']}  {len(resp.content):,} bytes")
        except Exception as exc:
            print(f"FAILED   {src['name']}  {exc}")

    for src in MANUAL:
        path = RAW_DIR / src["name"]
        if not path.exists():
            print(f"MISSING  {src['name']}  (download it manually from {src['url']})")
            continue
        content = path.read_bytes()
        err = validate(src["name"], content)
        if err:
            print(f"REJECTED {src['name']}  {err}")
            continue
        records.append(make_record(src, content, acquired="manual"))
        print(f"OK       {src['name']}  {len(content):,} bytes (manual)")

    MANIFEST.write_text(json.dumps(records, indent=2))
    print(f"Wrote {MANIFEST} with {len(records)} entries")


if __name__ == "__main__":
    main()
