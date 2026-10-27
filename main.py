#!/usr/bin/env python3
"""main.py - command line entry point.

Examples
    python main.py owner/repo1 owner/repo2 --hf-repo myuser/rust-qa
    python main.py https://github.com/BurntSushi/ripgrep --no-upload --max-chunks-per-repo 200
    python main.py --repos-file repos.txt --hf-repo myuser/rust-qa --private
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from core import MAX_REPOS, Config, FatalLLMError, log, setup_logging


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Build a Rust question->code dataset from 1-{MAX_REPOS} GitHub repos and upload it to Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("repos", nargs="*", help="owner/name or GitHub URL (1-10)")
    p.add_argument("--repos-file", type=Path, help="text file with one repo per line (# comments allowed)")

    g = p.add_argument_group("LLM (any OpenAI-compatible endpoint)")
    g.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    g.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    g.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--concurrency", type=int, default=8, help="parallel API calls")
    g.add_argument("--questions-per-chunk", type=int, default=1)

    g = p.add_argument_group("chunking")
    g.add_argument("--max-chunk-chars", type=int, default=6000)
    g.add_argument("--min-chunk-lines", type=int, default=3)
    g.add_argument("--include-tests", action="store_true", help="keep test code")
    g.add_argument("--max-chunks-per-repo", type=int, default=None, help="random sample, to cap API cost")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("output")
    g.add_argument("--workdir", type=Path, default=Path("work"), help="clones + resume files")
    g.add_argument("--out-dir", type=Path, default=Path("out"), help="Parquet + README.md")
    g.add_argument("--keep-clones", action="store_true")

    g = p.add_argument_group("Hugging Face")
    g.add_argument("--hf-repo", default=os.getenv("HF_REPO"), help="user/dataset-name")
    g.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    g.add_argument("--private", action="store_true", help="create the dataset as private")
    g.add_argument("--no-upload", action="store_true", help="only write files locally")

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def collect_specs(args: argparse.Namespace) -> list[str]:
    raw: list[str] = []
    for item in args.repos:
        raw.extend(item.split(","))
    if args.repos_file:
        for line in args.repos_file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                raw.append(line)
    if not raw and sys.stdin.isatty():
        answer = input(f"GitHub repos (1-{MAX_REPOS}, separated by comma or space): ")
        raw = answer.replace(",", " ").split()
    specs = list(dict.fromkeys(s.strip() for s in raw if s.strip()))  # de-dupe, keep order
    return specs


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    # import here so `--help` works even before dependencies are installed
    from pip import Pipeline  # local pipeline module (see note in pip.py)
    from repo import parse_repo

    specs = collect_specs(args)
    if not 1 <= len(specs) <= MAX_REPOS:
        parser.error(f"give between 1 and {MAX_REPOS} repositories (got {len(specs)})")
    for s in specs:
        try:
            parse_repo(s)
        except ValueError as exc:
            parser.error(str(exc))
    if not args.no_upload and not args.hf_repo:
        log.warning("No --hf-repo given: files will be written locally and NOT uploaded.")
    if args.hf_repo and not args.no_upload and not args.hf_token:
        parser.error("uploading needs a Hugging Face token: set HF_TOKEN or pass --hf-token (or use --no-upload)")

    cfg = Config(
        api_key=args.api_key, base_url=args.base_url, model=args.model,
        temperature=args.temperature, concurrency=args.concurrency,
        questions_per_chunk=max(1, args.questions_per_chunk),
        max_chunk_chars=args.max_chunk_chars, min_chunk_lines=args.min_chunk_lines,
        include_tests=args.include_tests, max_chunks_per_repo=args.max_chunks_per_repo,
        seed=args.seed, workdir=args.workdir, out_dir=args.out_dir,
        keep_clones=args.keep_clones, hf_repo=args.hf_repo, hf_token=args.hf_token,
        private=args.private, upload=not args.no_upload,
    )

    try:
        summary = asyncio.run(Pipeline(cfg).run(specs))
    except FatalLLMError as exc:
        log.error("Stopped: %s", exc)
        log.error("Progress is saved - fix the problem and run the same command again to resume.")
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted. Progress is saved - run the same command again to resume.")
        return 130

    pt, ct = summary["tokens"]
    log.info("Done: %d rows | tokens in/out: %d / %d", summary["rows"], pt, ct)
    log.info("Parquet: %s", summary["parquet"])
    if summary["url"]:
        log.info("Dataset: %s", summary["url"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
