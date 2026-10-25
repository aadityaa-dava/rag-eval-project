"""Parse govinfo Federal Register HTML notices into page-aware text."""
import re
from pathlib import Path

from bs4 import BeautifulSoup

DASH = re.compile(r"^\s*-{20,}\s*$")
PAGE_MARK = re.compile(r"^\s*\[\[Page (\d+)\]\]\s*$")
FOOTNOTE_OPEN = re.compile(r"^\s*\\\d+\\")   # footnote regions start with a \1\ style marker
FOOTNOTE_STRIP = re.compile(r"^\s*\\\d+\\\s*")
HEADING = re.compile(r"^[IVX]+\.\s+\S.{0,80}$")
BOILERPLATE = re.compile(r"^\s*(BILLING CODE|\[FR Doc\. .*Filed)")


def to_paragraphs(lines):
    """Join hard-wrapped lines into paragraphs (blank line = break); roman-numeral lines become headings."""
    paras, cur = [], []

    def flush():
        if not cur:
            return
        text = ""
        for line in cur:
            s = line.strip()
            text = s if not text else text + ("" if text.endswith("-") else " ") + s
        if len(cur) == 1 and HEADING.match(text):
            text = "## " + text
        paras.append(text)
        cur.clear()

    for line in lines:
        if line.strip():
            cur.append(line)
        else:
            flush()
    flush()
    return "\n\n".join(paras)


def clean(line):
    return line.replace("``", "\u201c").replace("''", "\u201d")


def next_nonblank(lines, i):
    return next((l for l in lines[i + 1:i + 6] if l.strip()), "")


def parse_html(path):
    path = Path(path)
    soup = BeautifulSoup(path.read_text(errors="ignore"), "lxml")
    lines = soup.find("pre").get_text().splitlines()

    m = re.search(r"\[Pages (\d+)-(\d+)\]", "\n".join(lines[:10]))
    page = int(m.group(1)) if m else None
    first_sep = next(i for i, l in enumerate(lines) if DASH.match(l))

    pages = {page: {"body": [], "notes": []}}
    in_note = False
    for i in range(first_sep + 1, len(lines)):
        raw = lines[i]
        pm = PAGE_MARK.match(raw)
        if pm:
            page = int(pm.group(1))
            pages.setdefault(page, {"body": [], "notes": []})
            continue
        if DASH.match(raw):
            if in_note:
                in_note = False                                   # closing fence
            elif FOOTNOTE_OPEN.match(next_nonblank(lines, i)):
                in_note = True                                    # opening fence
            continue                                              # separators and table borders: dropped
        if BOILERPLATE.match(raw):
            continue
        line = clean(FOOTNOTE_STRIP.sub("", raw) if in_note else raw)
        pages[page]["notes" if in_note else "body"].append(line)
    if in_note:
        raise ValueError(f"{path.name}: unclosed footnote region")

    return [
        {"doc": path.stem, "profile": "fr_html", "pdf_page": None, "printed_page": pg,
         "text": to_paragraphs(d["body"]), "footnotes": to_paragraphs(d["notes"])}
        for pg, d in sorted(pages.items())
    ]
