"""loop.py - the generation loop.

For every chunk it asks the LLM for question(s) and stores one row per
question (question + the real code as the answer).

* N async workers share the endpoint (``--concurrency``)
* every finished chunk is appended to a JSONL file immediately, so a crash or
  Ctrl+C loses nothing and a re-run skips what is already done (resume)
* the loop stops early if the endpoint is clearly broken (bad key, 25 failures
  in a row) instead of burning time and money
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from convert import SYSTEM_PROMPT, build_user_prompt, is_valid_question, parse_questions
from core import Chunk, Config, FatalLLMError, LLMClient, RepoMeta, log

try:  # progress bar is optional
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# --------------------------------------------------------------------------- #
# Storage (append-only JSONL, one file per repo)
# --------------------------------------------------------------------------- #
class JsonlStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = {row["chunk_id"] for row in self.iter_rows()}

    def append(self, rows: list[dict]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
        self.done.add(rows[0]["chunk_id"])

    def iter_rows(self, only_chunk_ids: Optional[Iterable[str]] = None) -> Iterator[dict]:
        if not self.path.exists():
            return
        keep = set(only_chunk_ids) if only_chunk_ids is not None else None
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:  # half-written last line after a crash
                    continue
                if keep is None or row["chunk_id"] in keep:
                    yield row


# --------------------------------------------------------------------------- #
# Row format (must match SCHEMA in pip.py)
# --------------------------------------------------------------------------- #
def make_row(chunk: Chunk, meta: RepoMeta, question: str, idx: int, model: str) -> dict:
    return {
        "id": f"{chunk.id}-{idx}",
        "chunk_id": chunk.id,
        "repo": meta.full_name,
        "repo_url": meta.url,
        "commit": meta.commit,
        "license": meta.license,
        "path": chunk.path,
        "kind": chunk.kind,
        "name": chunk.name,
        "context": chunk.context,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "num_lines": chunk.end_line - chunk.start_line + 1,
        "question": question,
        "answer": chunk.code,
        "model": model,
    }


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
@dataclass
class LoopStats:
    total: int = 0
    resumed: int = 0     # already done in an earlier run
    ok: int = 0
    bad_reply: int = 0   # model answered but nothing usable
    failed: int = 0      # API errors


async def run_loop(
    chunks: list[Chunk], meta: RepoMeta, llm: LLMClient, cfg: Config, store: JsonlStore
) -> LoopStats:
    todo = [c for c in chunks if c.id not in store.done]
    stats = LoopStats(total=len(chunks), resumed=len(chunks) - len(todo))
    if stats.resumed:
        log.info("%s: resuming - %d chunks already done, %d left", meta.full_name, stats.resumed, len(todo))
    if not todo:
        return stats

    queue: asyncio.Queue[Chunk] = asyncio.Queue()
    for c in todo:
        queue.put_nowait(c)

    bar = tqdm(total=len(todo), desc=meta.full_name, unit="chunk") if tqdm else None
    fatal: list[Exception] = []
    streak = 0  # consecutive API failures

    async def worker() -> None:
        nonlocal streak
        while not fatal:
            try:
                chunk = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                raw = await llm.complete(SYSTEM_PROMPT, build_user_prompt(chunk, meta, cfg.questions_per_chunk))
                questions: list[str] = []
                for q in parse_questions(raw):
                    if is_valid_question(q, chunk) and q not in questions:
                        questions.append(q)
                questions = questions[: cfg.questions_per_chunk]
                streak = 0
                if questions:
                    store.append([make_row(chunk, meta, q, i, llm.model) for i, q in enumerate(questions)])
                    stats.ok += 1
                else:
                    stats.bad_reply += 1
            except FatalLLMError as exc:
                fatal.append(exc)
            except Exception as exc:
                stats.failed += 1
                streak += 1
                log.warning("chunk %s (%s) failed: %s", chunk.id, chunk.name, str(exc)[:200])
                if streak >= cfg.max_consecutive_failures:
                    fatal.append(FatalLLMError(
                        f"{streak} API calls failed in a row - check the endpoint, key, model and quota. Last error: {exc}"
                    ))
            finally:
                if bar:
                    bar.update(1)

    try:
        await asyncio.gather(*(worker() for _ in range(max(1, cfg.concurrency))))
    finally:
        if bar:
            bar.close()
    if fatal:
        raise fatal[0]

    log.info(
        "%s: %d ok, %d unusable replies, %d API failures (re-run to retry the failed ones)",
        meta.full_name, stats.ok, stats.bad_reply, stats.failed,
    )
    return stats
