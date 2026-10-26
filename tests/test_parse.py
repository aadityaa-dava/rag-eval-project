from rag.parse_pdf import clean_glyphs


def test_private_use_bullets_become_bullets():
    assert clean_glyphs("such as: \uf0a7 Changes in data; \uf0b7 Shifts") == "such as: \u2022 Changes in data; \u2022 Shifts"


def test_normal_text_is_unchanged():
    t = "Section 524B, 21 CFR 820.30(g) \u2014 \u201cquoted\u201d caf\u00e9"
    assert clean_glyphs(t) == t
