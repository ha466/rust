"""pip.py - the pipeline (orchestrates everything).

    for each repo:  clone -> metadata -> clean -> chunk -> dedupe -> generate questions
    then:           Parquet -> dataset card (README.md) -> upload to Hugging Face

NOTE: this file is called pip.py as requested. Because of that, run
``pip install ...`` (the command) instead of ``python -m pip install ...``
from inside this folder - Python would pick up this file instead of real pip.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from convert import chunk_repo
from core import Config, FatalLLMError, LLMClient, RepoMeta, log, sha1
from loop import JsonlStore, LoopStats, run_loop
from repo import RepoRef, _rmtree, clean_repo, clone_repo, parse_repo, read_metadata

PARQUET_NAME = "train-00000-of-00001.parquet"

# SPDX id -> id accepted in the Hugging Face dataset card
HF_LICENSES = {
    "MIT": "mit", "Apache-2.0": "apache-2.0", "BSD-3-Clause": "bsd-3-clause",
    "BSD-2-Clause": "bsd-2-clause", "MPL-2.0": "mpl-2.0", "GPL-3.0": "gpl-3.0",
    "GPL-2.0": "gpl-2.0", "LGPL-3.0": "lgpl-3.0", "AGPL-3.0": "agpl-3.0",
    "Unlicense": "unlicense", "ISC": "isc",
}


@dataclass
class RepoResult:
    meta: RepoMeta
    files: int = 0
    chunks: int = 0
    chunk_stats: Counter = field(default_factory=Counter)
    loop: LoopStats = field(default_factory=LoopStats)
    store: JsonlStore | None = None
    valid_ids: set[str] = field(default_factory=set)
    rows: int = 0


def _schema():
    import pyarrow as pa
    return pa.schema([
        ("id", pa.string()), ("chunk_id", pa.string()), ("repo", pa.string()),
        ("repo_url", pa.string()), ("commit", pa.string()), ("license", pa.string()),
        ("path", pa.string()), ("kind", pa.string()), ("name", pa.string()),
        ("context", pa.string()), ("start_line", pa.int32()), ("end_line", pa.int32()),
        ("num_lines", pa.int32()), ("question", pa.string()), ("answer", pa.string()),
        ("model", pa.string()),
    ])


class Pipeline:
    def __init__(self, cfg: Config, llm: LLMClient | None = None):
        self.cfg = cfg
        self.llm = llm or LLMClient(cfg)
        self._seen_code: set[str] = set()  # cross-repo de-duplication

    # ------------------------------------------------------------------ #
    # one repository
    # ------------------------------------------------------------------ #
    async def process_repo(self, ref: RepoRef, repo_dir: Path | None = None) -> RepoResult:
        cfg = self.cfg
        # 1) clone (repo_dir can be passed in for tests)
        repo_dir = repo_dir or clone_repo(ref, cfg.workdir / "clones")
        # 2) metadata first - cleaning deletes .git, Cargo.toml and LICENSE
        meta = read_metadata(ref, repo_dir)
        # 3) keep only code
        kept = clean_repo(repo_dir, cfg)
        # 4) small chunks
        chunks, cstats = chunk_repo(repo_dir, ref.full_name, cfg)
        # 5) drop duplicates (vendored copies, identical helpers)
        unique = []
        for c in chunks:
            h = sha1("".join(c.code.split()))
            if h not in self._seen_code:
                self._seen_code.add(h)
                unique.append(c)
        if len(unique) < len(chunks):
            log.info("Removed %d duplicate chunks", len(chunks) - len(unique))
        # 6) optional cost cap
        if cfg.max_chunks_per_repo and len(unique) > cfg.max_chunks_per_repo:
            unique = random.Random(cfg.seed).sample(unique, cfg.max_chunks_per_repo)
            unique.sort(key=lambda c: (c.path, c.start_line))
            log.info("Sampled %d chunks (--max-chunks-per-repo)", len(unique))
        log.info(
            "%s: %d files -> %d chunks %s",
            ref.full_name, len(kept), len(unique),
            dict(cstats) if cstats else "",
        )

        result = RepoResult(meta=meta, files=len(kept), chunks=len(unique), chunk_stats=cstats)
        result.valid_ids = {c.id for c in unique}
        # 7) questions
        result.store = JsonlStore(cfg.workdir / "generated" / f"{ref.slug}.jsonl")
        if unique:
            result.loop = await run_loop(unique, meta, self.llm, cfg, result.store)
        result.rows = sum(1 for _ in result.store.iter_rows(result.valid_ids))

        if not cfg.keep_clones:
            _rmtree(repo_dir)
        return result

    # ------------------------------------------------------------------ #
    # Parquet
    # ------------------------------------------------------------------ #
    def write_parquet(self, results: list[RepoResult]) -> tuple[Path, dict]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        data_dir = self.cfg.out_dir / "data"
        if data_dir.exists():
            for old in data_dir.glob("*.parquet"):
                old.unlink()
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / PARQUET_NAME

        schema = _schema()
        kinds: Counter = Counter()
        example: dict | None = None
        total = 0
        batch: list[dict] = []
        with pq.ParquetWriter(path, schema, compression="zstd") as writer:
            for res in results:
                for row in res.store.iter_rows(res.valid_ids):
                    batch.append(row)
                    kinds[row["kind"]] += 1
                    total += 1
                    if example is None and row["num_lines"] <= 40:
                        example = row
                    if len(batch) >= 2000:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch = []
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        log.info("Wrote %d rows -> %s (%.1f MB)", total, path, path.stat().st_size / 1e6)
        return path, {"total": total, "kinds": kinds, "example": example}

    # ------------------------------------------------------------------ #
    # README / dataset card
    # ------------------------------------------------------------------ #
    def write_readme(self, results: list[RepoResult], info: dict) -> Path:
        cfg = self.cfg
        total = info["total"]

        spdx = set()
        for r in results:
            spdx.update(t for t in re.split(r"\s+(?:OR|AND|WITH)\s+|[ ()/]+", r.meta.license) if t)
        hf_license = HF_LICENSES.get(next(iter(spdx))) if len(spdx) == 1 else None
        size = (
            "n<1K" if total < 1_000 else "1K<n<10K" if total < 10_000
            else "10K<n<100K" if total < 100_000 else "100K<n<1M"
        )

        repo_rows = "\n".join(
            f"| [{r.meta.full_name}]({r.meta.url}) | `{r.meta.commit[:10] or '-'}` | "
            f"{r.meta.license} | {r.files} | {r.chunks} | {r.rows} |"
            for r in results
        )
        kind_rows = "\n".join(f"| {k} | {v} |" for k, v in info["kinds"].most_common())

        example = ""
        if info["example"]:
            ex = info["example"]
            example = (
                "## Example\n\n"
                f"**Question**\n\n> {ex['question']}\n\n"
                f"**Answer** (`{ex['repo']}` - `{ex['path']}`)\n\n```rust\n{ex['answer']}\n```\n"
            )

        card = f"""---
license: {hf_license or "other"}
task_categories:
- text-generation
language:
- en
tags:
- rust
- code
- instruction-tuning
- synthetic
size_categories:
- {size}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# Rust code instruction dataset

{total:,} pairs of **(question, real Rust code)** built from {len(results)} open-source GitHub {'repository' if len(results)==1 else 'repositories'}.

* **answer** - real, unmodified code from the repository (one function, method, struct, enum, trait or macro per row).
* **question** - the request a developer would send to an AI coding assistant to get that code. Written by `{self.llm.model}` after reading the code; the code itself is never generated.

## How it was built

1. Shallow `git clone` of each repository (commit is stored in the `commit` column).
2. Everything that is not Rust source is deleted: README/Markdown, images, docs, configs, lockfiles, tests (unless enabled), build output, generated and oversized files.
3. Each `.rs` file is split into small chunks - one per item, with its doc comments and attributes. Methods keep their `impl` header in the `context` column. Chunks shorter than {cfg.min_chunk_lines} lines or longer than {cfg.max_chunk_chars} characters are dropped, as are exact duplicates.
4. An OpenAI-compatible LLM writes {cfg.questions_per_chunk} question(s) per chunk.

## Columns

| column | description |
|---|---|
| `id`, `chunk_id` | row id / id of the code chunk |
| `question` | generated request (prompt) |
| `answer` | the original Rust code |
| `repo`, `repo_url`, `commit`, `license` | provenance of the code |
| `path`, `start_line`, `end_line`, `num_lines` | location in the repository |
| `kind`, `name`, `context` | item type, item name, enclosing `impl`/`trait` header |
| `model` | model that wrote the question |

## Source repositories

| repository | commit | license | files | chunks | rows |
|---|---|---|---|---|---|
{repo_rows}

## Rows by kind

| kind | rows |
|---|---|
{kind_rows}

{example}
## Usage

```python
from datasets import load_dataset

ds = load_dataset("{cfg.hf_repo or "<user>/<dataset>"}", split="train")
print(ds[0]["question"])
print(ds[0]["answer"])
```

## License and attribution

The code in `answer` stays under the license of the repository it came from (see the `license`
and `repo_url` columns). If you redistribute or train on it, comply with those licenses -
for example attribution for MIT/Apache-2.0, and the stronger terms of GPL/AGPL repositories.
The questions are synthetic; check the terms of the model provider you use them with.
"""
        if cfg.readme_template:
            values = {
                "dataset_repo": cfg.hf_repo or "<user>/<dataset>",
                "model": self.llm.model,
                "row_count": f"{total:,}",
                "repo_count": str(len(results)),
                "license": hf_license or "other",
                "size_category": size,
                "source_repositories": repo_rows,
                "rows_by_kind": kind_rows,
                "example": example,
            }
            card = cfg.readme_template
            for key, value in values.items():
                card = card.replace("{{" + key + "}}", value)
        path = cfg.out_dir / "README.md"
        path.write_text(card, encoding="utf-8")
        log.info("Wrote %s", path)
        return path

    # ------------------------------------------------------------------ #
    # Hugging Face
    # ------------------------------------------------------------------ #
    def upload(self) -> str:
        from huggingface_hub import HfApi

        cfg = self.cfg
        api = HfApi(token=cfg.hf_token)
        repo_id = cfg.hf_repo
        if "/" not in repo_id:  # allow just "my-dataset"
            repo_id = f"{api.whoami()['name']}/{repo_id}"
        api.create_repo(repo_id, repo_type="dataset", private=cfg.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(cfg.out_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add Rust question/code dataset",
            allow_patterns=["README.md", "data/*.parquet"],
        )
        url = f"https://huggingface.co/datasets/{repo_id}"
        log.info("Uploaded -> %s", url)
        return url

    # ------------------------------------------------------------------ #
    # everything
    # ------------------------------------------------------------------ #
    async def run(self, specs: list[str]) -> dict:
        refs = [parse_repo(s) for s in specs]
        results: list[RepoResult] = []
        for ref in refs:
            try:
                results.append(await self.process_repo(ref))
            except FatalLLMError:
                raise
            except Exception as exc:
                log.error("Repo %s failed and was skipped: %s", ref.full_name, exc)

        if not results or not any(r.rows for r in results):
            raise SystemExit("No samples were produced - nothing to save.")

        parquet, info = self.write_parquet(results)
        readme = self.write_readme(results, info)
        url = None
        if self.cfg.upload and self.cfg.hf_repo:
            url = self.upload()
        else:
            log.info("Upload skipped. Files are in %s", self.cfg.out_dir)
        return {"parquet": parquet, "readme": readme, "rows": info["total"], "url": url,
                "tokens": (self.llm.prompt_tokens, self.llm.completion_tokens)}
