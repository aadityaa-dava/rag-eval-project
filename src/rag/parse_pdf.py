"""Extract clean, page-aware text from the PDF documents, using per-family profiles."""
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import pymupdf

BULLET = re.compile(r"^([•▪\uf000-\uf8ff]|o\s)")


@dataclass(frozen=True)
class Profile:
    name: str
    header_max_y: float = 75.0        # running header sits above this
    columns: int = 1
    left: float = 36.0                # text-area bounds, used to assign columns
    right: float = 576.0
    line_numbers: bool = True         # strip digit-only margin line numbers
    marker_max_size: float = 9.0      # drop superscript markers below this (0 = off)
    footnote_max_size: float = 10.5   # small text in lower half -> footnotes (0 = off)
    heading_min_size: float = 14.0    # 0 = no heading detection
    page_no_in_header: bool = False   # printed page number lives in the header


FDA_GUIDANCE = Profile("fda_guidance")
TRANSCRIPT = Profile("transcript", line_numbers=False, marker_max_size=0,
                     footnote_max_size=0, heading_min_size=0)
FR_NOTICE = Profile("fr_notice", header_max_y=50.0, columns=3, left=45.0, right=567.0,
                    line_numbers=False, marker_max_size=0, footnote_max_size=0,
                    heading_min_size=0, page_no_in_header=True)


def profile_for(path) -> Profile:
    stem = Path(path).stem
    if stem.startswith("fr_"):
        return FR_NOTICE
    if "transcript" in stem:
        return TRANSCRIPT
    return FDA_GUIDANCE


def parse_page(page, prof: Profile):
    """Return (printed_page, body_lines, footnote_lines). A line is (col, y0, x0, size, text)."""
    printed = None
    body, notes = [], []
    height = page.rect.height
    col_w = (prof.right - prof.left) / prof.columns
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            kept = []
            for s in line["spans"]:
                t = s["text"].strip()
                if not t:
                    continue
                x0, y0, size = s["bbox"][0], s["bbox"][1], s["size"]
                if y0 < prof.header_max_y:                       # running header
                    if prof.page_no_in_header and printed is None and re.fullmatch(r"\d{3,5}", t):
                        printed = int(t)
                    continue
                if y0 > height - 100 and x0 > 480 and re.fullmatch(r"\d{1,3}", t):
                    printed = int(t)                             # bottom-right page number
                    continue
                if prof.line_numbers and x0 < 60 and re.fullmatch(r"\d{1,4}", t):
                    continue                                     # margin line number
                if size < prof.marker_max_size:
                    continue                                     # superscript marker
                kept.append(s)
            if not kept:
                continue
            text = "".join(s["text"] for s in kept).strip()
            y0 = min(s["bbox"][1] for s in kept)
            x0 = min(s["bbox"][0] for s in kept)
            size = max(s["size"] for s in kept)
            col = max(0, min(int((x0 - prof.left) // col_w), prof.columns - 1))
            is_note = prof.footnote_max_size > 0 and size <= prof.footnote_max_size and y0 > height / 2
            (notes if is_note else body).append((col, y0, x0, size, text))
    return printed, body, notes


def assemble(lines, prof: Profile) -> str:
    """Order lines (column, then top-to-bottom), join into paragraphs, mark headings with '## '."""
    if not lines:
        return ""
    lines = sorted(lines, key=lambda l: (l[0], round(l[1]), l[2]))
    gaps = [b[1] - a[1] for a, b in zip(lines, lines[1:])
            if b[0] == a[0] and 3 < b[1] - a[1] < 40]
    pitch = statistics.median(gaps) if gaps else 14.0
    paras, prev = [], None                                       # paras: [is_heading, text]
    for col, y0, _x0, size, text in lines:
        heading = prof.heading_min_size > 0 and size >= prof.heading_min_size
        new = (
            not paras
            or paras[-1][0] != heading
            or (not heading and prev is not None and prev[0] == col and (y0 - prev[1]) > 1.5 * pitch)
            or (not heading and bool(BULLET.match(text)))
        )
        if new:
            paras.append([heading, text])
        else:
            joiner = "" if paras[-1][1].endswith("-") else " "
            paras[-1][1] += joiner + text
        prev = (col, y0)
    return "\n\n".join(("## " if h else "") + t for h, t in paras)


def parse_pdf(path, prof: Profile = None):
    path = Path(path)
    prof = prof or profile_for(path)
    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            printed, body, notes = parse_page(page, prof)
            pages.append({
                "doc": path.stem,
                "profile": prof.name,
                "pdf_page": i + 1,
                "printed_page": printed,
                "text": assemble(body, prof),
                "footnotes": assemble(notes, prof),
            })
    return pages
