"""Load a category tree from a file at run time. No category names live in code.

Supports trees of 1-4 (or more) levels, branches of different depth (blank
cells below a branch's last level), repeated names across rows, and an
optional description/keywords column per row.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from spendpipe.config import Config
from spendpipe.ingest import read_table

log = logging.getLogger(__name__)


@dataclass
class Leaf:
    path: tuple[str, ...]          # exact names from the file, top to bottom
    text: str                      # text used to embed this leaf
    description: str = ""


@dataclass
class Tree:
    levels: int                    # number of level columns in the file
    leaves: list[Leaf] = field(default_factory=list)

    def nodes_at(self, depth: int) -> list[tuple[str, ...]]:
        """Distinct path prefixes of length `depth` (1 = L1)."""
        return sorted({lf.path[:depth] for lf in self.leaves if len(lf.path) >= depth})


def _level_columns(df: pd.DataFrame, configured: list[str] | None) -> list[str]:
    if configured:
        return configured
    # columns named like level/l1/category_l2/lvl3... in numeric order; else all
    # columns except obvious description columns
    num = []
    for c in df.columns:
        m = re.search(r"(?:^|[^a-z])(?:l|lvl|level)[ _]?(\d)\b", c.lower())
        if m:
            num.append((int(m.group(1)), c))
    if num:
        return [c for _, c in sorted(num)]
    return [c for c in df.columns if not re.search(r"desc|keyword|note|comment|example", c.lower())]


def path_text(path: tuple[str, ...], description: str = "") -> str:
    """'A > B > C' with consecutive repeated names dropped (A's L3 often == L2)."""
    parts: list[str] = []
    for p in path:
        if not parts or p.strip().lower() != parts[-1].strip().lower():
            parts.append(p.strip())
    text = " > ".join(parts)
    return f"{text}: {description}" if description else text


def load_tree(cfg: Config) -> Tree:
    tc = cfg.section("taxonomy")
    df = read_table(cfg.file(tc["file"], fallback=True))
    cols = _level_columns(df, tc.get("level_columns"))
    desc_col = tc.get("description_column")
    if desc_col is None:
        cand = [c for c in df.columns if c not in cols and re.search(r"desc|keyword|example", c.lower())]
        desc_col = cand[0] if cand else None

    leaves: dict[tuple, Leaf] = {}
    for _, row in df.iterrows():
        vals = [str(row[c]).strip() for c in cols]
        while vals and not vals[-1]:          # branch ends early: trailing blanks
            vals.pop()
        if not vals or any(not v for v in vals):
            if vals:
                log.warning("taxonomy row with a gap skipped: %s", vals)
            continue
        path = tuple(vals)
        desc = str(row[desc_col]).strip() if desc_col else ""
        if path not in leaves:
            leaves[path] = Leaf(path=path, text=path_text(path, desc), description=desc)
        elif desc and desc not in leaves[path].description:
            leaves[path].description += "; " + desc
            leaves[path].text = path_text(path, leaves[path].description)

    # a path that is a prefix of a longer path is an inner node, not a leaf
    paths = set(leaves)
    inner = {p for p in paths if any(q != p and q[:len(p)] == p for q in paths)}
    tree = Tree(levels=len(cols), leaves=[lf for p, lf in leaves.items() if p not in inner])
    log.info("taxonomy %s: %d levels, %d leaves, L1 nodes %d", tc["file"], tree.levels,
             len(tree.leaves), len(tree.nodes_at(1)))
    return tree
