"""Extract clean, page-aware text from the FDA guidance PDFs."""
import re
import statistics
from pathlib import Path

import pymupdf

HEADER_MAX_Y = 75         # running header sits above this (points)
PAGE_NO_MIN_Y = 740       # printed page number sits below this
LINE_NO_MAX_X = 60        # margin line numbers sit left of this
MARKER_MAX_SIZE = 9       # superscript footnote markers are smaller than this
FOOTNOTE_MAX_SIZE = 10.5  # footnote text is ~10pt, body text is ~12pt
HEADING_MIN_SIZE = 14
BULLET = re.compile(r"^([•▪\uf000-\uf8ff]|o\s)")


def parse_page(page):
    """Return (printed_page, body_lines, footnote_lines) for one page."""
    printed = None
    body, notes = [], []
    half = page.rect.height / 2
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            kept = []
            for s in line["spans"]:
                t = s["text"].strip()
                if not t:
                    continue
                x0, y0, size = s["bbox"][0], s["bbox"][1], s["size"]
                if y0 < HEADER_MAX_Y:
                    continue                      # running header
                if y0 > PAGE_NO_MIN_Y and re.fullmatch(r"\d{1,3}", t):
                    printed = int(t)              # printed page number
                    continue
                if x0 < LINE_NO_MAX_X and re.fullmatch(r"\d{1,4}", t):
                    continue                      # margin line number
                if size < MARKER_MAX_SIZE:
                    continue                      # superscript footnote marker
                kept.append(s)
            if not kept:
                continue
            text = "".join(s["text"] for s in kept).strip()
            y0 = min(s["bbox"][1] for s in kept)
            x0 = min(s["bbox"][0] for s in kept)
            size = max(s["size"] for s in kept)
            is_note = size <= FOOTNOTE_MAX_SIZE and y0 > half
            (notes if is_note else body).append((y0, x0, size, text))
    return printed, body, notes


def assemble(lines):
    """Join visual lines into paragraphs; headings become '## ' lines."""
    if not lines:
        return ""
    lines = sorted(lines, key=lambda l: (round(l[0]), l[1]))
    gaps = [b[0] - a[0] for a, b in zip(lines, lines[1:]) if 5 < b[0] - a[0] < 40]
    pitch = statistics.median(gaps) if gaps else 14.0
    paras, prev_y = [], None                      # paras: [is_heading, text]
    for y0, _x0, size, text in lines:
        heading = size >= HEADING_MIN_SIZE
        new = (
            not paras
            or paras[-1][0] != heading
            or (not heading and (y0 - prev_y) > 1.5 * pitch)
            or (not heading and BULLET.match(text))
        )
        if new:
            paras.append([heading, text])
        else:
            joiner = "" if paras[-1][1].endswith("-") else " "
            paras[-1][1] += joiner + text
        prev_y = y0
    return "\n\n".join(("## " if h else "") + t for h, t in paras)


def parse_pdf(path):
    path = Path(path)
    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            printed, body, notes = parse_page(page)
            pages.append({
                "doc": path.stem,
                "pdf_page": i + 1,
                "printed_page": printed,
                "text": assemble(body),
                "footnotes": assemble(notes),
            })
    return pages
