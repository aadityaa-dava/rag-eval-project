"""Baseline chunker: paragraph-aware packing with character overlap and page-aware metadata."""
import json
import re
import statistics
import sys
from pathlib import Path

PAGES = Path("data/processed/pages.jsonl")
LEADERS = re.compile(r"\.{5,}")                 # dotted leaders: table of contents and table rows
TERMINAL = re.compile(r"[.!?:;\u201d\")\]]$")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def load_pages(path=PAGES):
    docs = {}
    for line in path.open():
        r = json.loads(line)
        docs.setdefault(r["doc"], []).append(r)
    return docs


def page_label(r):
    if r["printed_page"] is not None:
        return r["printed_page"], "printed"
    return r["pdf_page"], "pdf"


def paragraph_stream(pages):
    """Paragraphs with page info; a paragraph that continues across a page break is merged."""
    paras = []
    for r in pages:
        page, basis = page_label(r)
        texts = [t.strip() for t in r["text"].split("\n\n") if t.strip() and not LEADERS.search(t)]
        for i, t in enumerate(texts):
            is_head = t.startswith("## ")
            t = t[3:] if is_head else t
            if i == 0 and paras and not is_head and not paras[-1]["heading"] \
                    and not TERMINAL.search(paras[-1]["text"]):
                paras[-1]["text"] += " " + t           # sentence continues from the previous page
                paras[-1]["end_page"] = page
                continue
            paras.append({"text": t, "page": page, "end_page": page, "basis": basis, "heading": is_head})
    return paras


def split_long(text, limit):
    """Split text longer than limit on sentence boundaries, then on words."""
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for sent in SENTENCE.split(text):
        while len(sent) > limit:
            cut = sent.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if cur:
                out.append(cur)
                cur = ""
            out.append(sent[:cut])
            sent = sent[cut:].lstrip()
        if cur and len(cur) + 1 + len(sent) > limit:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def tail(text, overlap):
    """Last ~overlap characters of text, starting on a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    cut = text.find(" ", len(text) - overlap)
    return text[cut + 1:] if cut != -1 else ""


def chunk_doc(doc, pages, size, overlap):
    limit = size - overlap - 2                     # a piece plus carried overlap never exceeds size
    chunks, buf, meta, section = [], "", None, ""

    def flush():
        nonlocal buf, meta
        if meta is None:
            return
        chunks.append({
            "chunk_id": f"{doc}::{len(chunks):04d}", "doc": doc, "section": meta["section"],
            "start_page": meta["start_page"], "end_page": meta["end_page"],
            "page_basis": meta["basis"], "text": buf, "n_chars": len(buf),
        })
        buf, meta = tail(buf, overlap), None

    for p in paragraph_stream(pages):
        if p["heading"]:
            section = p["text"]
        for piece in split_long(p["text"], limit):
            if meta is not None and len(buf) + 2 + len(piece) > size:
                flush()
            if meta is None:
                meta = {"section": section, "start_page": p["page"], "end_page": p["end_page"], "basis": p["basis"]}
            buf = f"{buf}\n\n{piece}" if buf else piece
            meta["end_page"] = p["end_page"]
    flush()
    return chunks


def main(size=1000, overlap=150):
    docs = load_pages()
    out = Path(f"data/processed/chunks_{size}_{overlap}.jsonl")
    all_chunks = []
    for doc, pages in docs.items():
        cs = chunk_doc(doc, pages, size, overlap)
        all_chunks += cs
        print(f"{doc[:44]:44} {len(cs):4d} chunks")
    with out.open("w") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    lens = [c["n_chars"] for c in all_chunks]
    src = sum(len(p["text"]) for ps in docs.values() for p in ps)
    print(f"\nwrote {len(all_chunks)} chunks to {out}")
    print(f"chars min/median/max: {min(lens)}/{int(statistics.median(lens))}/{max(lens)} | "
          f"under 200 chars: {sum(l < 200 for l in lens)} | over {size}: {sum(l > size for l in lens)}")
    print(f"chunk chars / source chars: {sum(lens) / src:.2f}  (about 1.15 expected: overlap, minus dropped TOC)")
    print("page basis:", {b: sum(c['page_basis'] == b for c in all_chunks) for b in ('printed', 'pdf')})
    hit = next(c for c in all_chunks if c["doc"] == "fda_ai_lifecycle_draft_2025" and "data drift" in c["text"])
    print(f"\n--- sample: {hit['chunk_id']} | pages {hit['start_page']}-{hit['end_page']} ({hit['page_basis']}) "
          f"| section: {hit['section']!r} ---\n{hit['text']}")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:3]]
    main(*args)
