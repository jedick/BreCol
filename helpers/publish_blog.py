#!/usr/bin/env python3
"""Publish the manuscript as a Jekyll post in a Minimal Mistakes blog repository.

Reads ``manuscript/build/manuscript.jekyll.md`` (produced by ``make manuscript_jekyll``),
prepends Jekyll front matter, rewrites figure ``src`` URLs to
``/assets/images/<POST_DATE>-<POST_SLUG>/...``, copies SVG figures into the blog's
assets directory, and writes the result to ``_posts/<POST_DATE>-<POST_SLUG>.md`` in the
target blog repository. Tables are inlined as raw HTML in the rendered post (by the
manuscript_jekyll Pandoc pipeline), so no table fragments are copied as assets.

``POST_DATE`` is fixed (the post's original publication date used by Jekyll's filename
convention and the ``date:`` front-matter field). Each invocation stamps the front
matter with ``last_modified_at: <today>`` so the rendered post records when it was
last regenerated.

The blog repository path is read from the ``BRECOL_BLOG_REPO`` environment variable or
``--blog-repo``. A trailing ``--push`` flag commits and pushes the change.
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

POST_DATE = "2026-05-22"
POST_SLUG = "BreCol-cancer-classification-benchmark-using-gut-microbiome-data"

FRONT_MATTER_TEMPLATE = """---
title: "BreCol: Cancer classification benchmark using gut microbiome data"
date: {date}
last_modified_at: {last_modified_at}
layout: single
classes: wide
category: Blog
tags:
  - Microbiome
  - Cancer
  - HyenaDNA
  - Benchmark
header:
  teaser: /assets/siteimages/BreCol_banner.svg
  header: /assets/siteimages/BreCol_banner.svg
  og_image: /assets/siteimages/BreCol_banner.svg
excerpt: "Microbiome-based cancer prediction benchmarks sometimes overestimate real-world performance
  because test samples are drawn from the same studies used for training."
---

"""

FIGURE_GLOB = "figure*.svg"


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
        "--push",
        action="store_true",
        help="git add / commit / push the result in the blog repository.",
    )
    return parser.parse_args()


def enable_markdown_in_csl_divs(body: str) -> str:
    """Add ``markdown="1"`` to citeproc reference ``<div>`` wrappers.

    Pandoc's GFM citeproc output wraps the bibliography in
    ``<div id="refs">`` and each entry in ``<div id="ref-...">``. Jekyll's
    Kramdown processor treats content inside block-level HTML as raw HTML by
    default, which leaves the inline ``*italics*``, ``**bold**``, and
    ``[text](url)`` markdown inside each entry unrendered. The Kramdown
    ``markdown="1"`` extension attribute opts those divs back into markdown
    parsing without touching the surrounding HTML.
    """
    return re.sub(
        r'(<div\s+id="(?:refs|ref-[^"]+)"[^>]*?)(>)',
        r'\1 markdown="1"\2',
        body,
    )


def rewrite_asset_paths(body: str, asset_root: str) -> str:
    """Rewrite same-directory ``<img src="figure*.svg">`` URLs to ``<asset_root>/...``.

    The manuscript_jekyll Pandoc pipeline wraps every figure in a
    ``<figure><img .../></figure>`` block (so number_figures.lua can attach a
    numbered caption), which means the only references to ``figure*.svg`` in
    the rendered body are HTML ``src`` attributes. Tables are inlined as raw
    ``<table>`` HTML, so no ``table*.html`` link rewriting is needed either.
    """
    return re.sub(
        r'(src=")(figure[^"\s]*\.svg)(")',
        rf"\1{asset_root}/\2\3",
        body,
    )


def copy_assets(blog_assets_dir: Path) -> list[Path]:
    """Copy SVG figures into the blog's assets directory."""
    blog_assets_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in MANUSCRIPT_DIR.glob(FIGURE_GLOB):
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

    last_modified_at = _dt.date.today().isoformat()

    post_stem = f"{POST_DATE}-{POST_SLUG}"
    asset_subdir = Path("images") / post_stem
    asset_root = f"/assets/{asset_subdir.as_posix()}"

    body = BUILD_MD.read_text(encoding="utf-8")
    body = rewrite_asset_paths(body, asset_root)
    body = enable_markdown_in_csl_divs(body)
    front = FRONT_MATTER_TEMPLATE.format(
        date=POST_DATE, last_modified_at=last_modified_at
    )

    post_path = blog_repo / "_posts" / f"{post_stem}.md"
    post_path.parent.mkdir(parents=True, exist_ok=True)
    post_path.write_text(front + body, encoding="utf-8")

    blog_assets_dir = blog_repo / "assets" / asset_subdir
    copied_assets = copy_assets(blog_assets_dir)

    print(f"Wrote {post_path.relative_to(blog_repo)}")
    for p in copied_assets:
        print(f"Copied {p.relative_to(blog_repo)}")

    if args.push:
        git_publish(blog_repo, [post_path, *copied_assets], last_modified_at)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
