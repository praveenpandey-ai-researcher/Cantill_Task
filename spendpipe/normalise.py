"""Turn messy identifiers and names into comparable keys.

Used by matching (stage 3) and supplier/item de-duplication (stage 5).
All functions take one value and return a string ('' when nothing usable).
"""
from __future__ import annotations

import re
import unicodedata

# Legal-entity words removed from supplier names. Generic, not company data;
# a config can add more via dedup.supplier.extra_legal_suffixes.
LEGAL_SUFFIXES = {
    "PTY", "PTY LTD", "PROPRIETARY", "LTD", "LIMITED", "CC", "INC", "INCORPORATED", "LLC",
    "LLP", "PLC", "CORP", "CORPORATION", "CO", "COMPANY", "GMBH", "AG", "KG", "SA", "SAS",
    "SARL", "SRL", "SPA", "SL", "SAU", "SA DE CV", "DE CV", "CV", "BV", "NV", "AB", "AS",
    "OY", "SRO", "SP ZOO", "ZOO", "KFT", "DOO", "OOO", "ZAO", "PAO", "AO", "LTDA", "EIRELI",
    "ME", "PVT", "PRIVATE", "HOLDINGS", "GROUP", "THE",
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_po(value, pattern: str | None = None) -> str:
    """'PO#4510008616', 'PO 4510008616', '4510008616/10.0' -> '4510008616'.
    Default: the longest run of >= 5 digits, leading zeros removed.
    `pattern` (regex with one group) can be set per company in the config."""
    s = str(value or "")
    if pattern:
        m = re.search(pattern, s)
        return m.group(1).lstrip("0") if m else ""
    runs = re.findall(r"\d{5,}", s)
    return max(runs, key=len).lstrip("0") if runs else ""


def norm_po_line(value) -> str:
    """'10.0' -> '10', '4510008616/20.0' -> '20', '' -> ''."""
    s = str(value or "").strip()
    if "/" in s:
        s = s.split("/")[-1]
    m = re.match(r"^0*(\d+)(?:\.0+)?$", s.strip())
    return m.group(1) if m else ""


def norm_invoice(value) -> str:
    """Invoice number core = last group of digits without leading zeros.
    'INV/2023/759857', 'SI-2023-759857', 'IN0759857', 'TAX INV 759857' -> '759857'.
    The year in the middle is ignored (the same invoice appears with two years)."""
    runs = re.findall(r"\d+", str(value or ""))
    return runs[-1].lstrip("0") if runs else ""


def norm_supplier(value, extra_suffixes: set[str] | None = None) -> str:
    """Upper-case, accents and punctuation removed, legal suffixes stripped.
    'Kinora  Industrial (Pty) Ltd.' -> 'KINORA INDUSTRIAL'."""
    s = strip_accents(str(value or "")).upper()
    s = s.replace("&", " AND ")
    s = re.sub(r"\b([A-Z])\.", r"\1", s)      # dotted initials: 'S.A.' -> 'SA', 'C.V.' -> 'CV'
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    words = s.split()
    suffixes = LEGAL_SUFFIXES | (extra_suffixes or set())
    # drop suffix words (single and two-word) from the END only, repeatedly,
    # so 'SA' inside a name is kept but '... MINING SA PTY LTD' -> '... MINING'
    changed = True
    while changed and len(words) > 1:
        changed = False
        for n in (3, 2, 1):
            if len(words) > n and " ".join(words[-n:]) in suffixes:
                words = words[:-n]
                changed = True
                break
    # also drop a leading 'THE'
    if len(words) > 1 and words[0] == "THE":
        words = words[1:]
    return " ".join(words)


def squash(s: str) -> str:
    """Remove all spaces: catches 'YARF IELD' vs 'YARFIELD'."""
    return s.replace(" ", "")


def norm_text(value) -> str:
    """Item text for comparison: upper-case, decimal comma -> dot ('3,5M' == '3.5M'),
    punctuation to spaces, whitespace collapsed."""
    s = strip_accents(str(value or "")).upper()
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    s = re.sub(r"[^A-Z0-9.]+", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)   # keep dots only inside numbers
    return " ".join(s.split())
