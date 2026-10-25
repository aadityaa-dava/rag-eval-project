"""Parse every raw document into page records and write data/processed/pages.jsonl."""
import json
from pathlib import Path

from rag.parse_html import parse_html
from rag.parse_pdf import parse_pdf

RAW = Path("data/raw")
OUT = Path("data/processed/pages.jsonl")
MANIFEST = Path("data/manifest.json")


def main():
    urls = {Path(r["name"]).stem: r["url"] for r in json.loads(MANIFEST.read_text())}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_pages = 0
    with OUT.open("w") as f:
        for path in sorted(RAW.iterdir()):
            if path.suffix == ".pdf":
                pages = parse_pdf(path)
            elif path.suffix == ".html":
                pages = parse_html(path)
            else:
                continue
            for p in pages:
                p["source_url"] = urls.get(path.stem)
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
            n_pages += len(pages)
            body = sum(len(p["text"]) for p in pages)
            empty = sum(1 for p in pages if not p["text"].strip())
            print(f"{path.stem[:44]:44} {len(pages):3d} pages {body:8,d} chars  empty pages: {empty}")
    print(f"\nwrote {n_pages} page records to {OUT}")


if __name__ == "__main__":
    main()
