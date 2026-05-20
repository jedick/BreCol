#!/usr/bin/env python3
"""Publish the manuscript as a Jekyll post in a Minimal Mistakes blog repository.

Reads ``manuscript/build/manuscript.jekyll.md`` (produced by ``make manuscript_jekyll``),
prepends Jekyll front matter, rewrites figure and table paths to ``/assets/brecol/...``,
copies figures and table HTML fragments into the blog's assets directory, and writes the
result to ``_posts/YYYY-MM-DD-brecol.md`` in the target blog repository.

The blog repository path is read from the ``BRECOL_BLOG_REPO`` environment variable or
``--blog-repo``. The post date defaults to today and may be overridden with ``--date``.
A trailing ``--push`` flag commits and pushes the change.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_MD = REPO_ROOT / "manuscript" / "build" / "manuscript.jekyll.md"
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"

POST_SLUG = "brecol"
ASSET_SUBDIR = "brecol"

FRONT_MATTER_TEMPLATE = """---
title: "BreCol: Cancer classification benchmark using gut microbiome data"
date: {date}
layout: single
classes: wide
toc: true
toc_sticky: true
categories:
  - research
tags:
  - microbiome
  - cancer
  - hyenadna
  - benchmark
---

"""

FIGURE_PATTERNS = ("figure*.svg", "figure*.png")
TABLE_PATTERNS = ("table*.html",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blog-repo",
        type=Path,
        default=os.environ.get("BRECOL_BLOG_REPO"),
        help="Path to the Minimal Mistakes blog repository "
        "(or set the BRECOL_BLOG_REPO environment variable).",
    )
    parser.add_argument(
        "--date",
        default=_dt.date.today().isoformat(),
        help="Post date in YYYY-MM-DD form (default: today).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="git add / commit / push the result in the blog repository.",
    )
    return parser.parse_args()


def rewrite_asset_paths(body: str) -> str:
    """Rewrite same-directory asset links and ``<img src>`` to ``/assets/<ASSET_SUBDIR>/...``.

    Pandoc's GFM writer keeps markdown image syntax ``![alt](figure.svg)`` for some
    inputs but emits raw ``<img src="figure.svg">`` HTML when attributes (alt text,
    sizing) need to round-trip; cover both forms plus standalone link references to
    the table HTML fragments.
    """
    asset_root = f"/assets/{ASSET_SUBDIR}"
    body = re.sub(
        r"\]\((figure[^)\s]*\.(?:svg|png))\)",
        rf"]({asset_root}/\1)",
        body,
    )
    body = re.sub(
        r"\]\((table[^)\s]*\.html)\)",
        rf"]({asset_root}/\1)",
        body,
    )
    body = re.sub(
        r'(src=")(figure[^"\s]*\.(?:svg|png))(")',
        rf"\1{asset_root}/\2\3",
        body,
    )
    return body


def copy_assets(blog_assets_dir: Path) -> list[Path]:
    """Copy figures and table HTML fragments into the blog's assets directory."""
    blog_assets_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for pattern in FIGURE_PATTERNS + TABLE_PATTERNS:
        for src in MANUSCRIPT_DIR.glob(pattern):
            dst = blog_assets_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def git_publish(blog_repo: Path, files: list[Path], date: str) -> None:
    """``git add`` the touched paths in the blog repo and commit + push."""
    rels = [str(p.relative_to(blog_repo)) for p in files]
    subprocess.run(["git", "-C", str(blog_repo), "add", "--", *rels], check=True)
    status = subprocess.run(
        ["git", "-C", str(blog_repo), "status", "--porcelain", "--", *rels],
        check=True,
        capture_output=True,
        text=True,
    )
    if not status.stdout.strip():
        print("No changes to publish.")
        return
    subprocess.run(
        ["git", "-C", str(blog_repo), "commit", "-m", f"BreCol manuscript update ({date})"],
        check=True,
    )
    subprocess.run(["git", "-C", str(blog_repo), "push"], check=True)


def main() -> int:
    args = parse_args()

    if not BUILD_MD.exists():
        sys.exit(
            f"Missing {BUILD_MD.relative_to(REPO_ROOT)}. "
            "Run `make manuscript_jekyll` first."
        )
    if args.blog_repo is None:
        sys.exit(
            "Blog repo path required. Set --blog-repo or BRECOL_BLOG_REPO."
        )

    blog_repo = Path(args.blog_repo).expanduser().resolve()
    if not (blog_repo / ".git").exists():
        sys.exit(f"{blog_repo} does not look like a git repository.")

    try:
        _dt.date.fromisoformat(args.date)
    except ValueError:
        sys.exit(f"--date {args.date!r} is not in YYYY-MM-DD form.")

    body = BUILD_MD.read_text(encoding="utf-8")
    body = rewrite_asset_paths(body)
    front = FRONT_MATTER_TEMPLATE.format(date=args.date)

    post_path = blog_repo / "_posts" / f"{args.date}-{POST_SLUG}.md"
    post_path.parent.mkdir(parents=True, exist_ok=True)
    post_path.write_text(front + body, encoding="utf-8")

    blog_assets_dir = blog_repo / "assets" / ASSET_SUBDIR
    copied_assets = copy_assets(blog_assets_dir)

    print(f"Wrote {post_path.relative_to(blog_repo)}")
    for p in copied_assets:
        print(f"Copied {p.relative_to(blog_repo)}")

    if args.push:
        git_publish(blog_repo, [post_path, *copied_assets], args.date)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
