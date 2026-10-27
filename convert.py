"""convert.py - turn Rust source into small chunks, and chunks into prompts.

Part 1 (chunking)
    Every function / method / struct / enum / trait / macro becomes its own
    chunk, so a file with three functions gives three chunks. The parser is a
    small dependency-free scanner: it blanks out comments and strings, matches
    braces, and cuts each item out of the ORIGINAL text, so the answer is real,
    unmodified code (only de-indented).

Part 2 (prompting)
    ``build_user_prompt`` asks the LLM to write the request a developer would
    send to an AI assistant to get exactly this code back.
"""
from __future__ import annotations

import json
import re
import textwrap
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from core import Chunk, Config, RepoMeta, collapse_ws, log, sha1

# =========================================================================== #
# Part 1 - chunking
# =========================================================================== #

_RAW_STR_RE = re.compile(r'b?r(#*)"')

# an item keyword at the start of a line, with optional visibility/qualifiers
_ITEM_RE = re.compile(
    r'^[ \t]*(?:pub(?:[ \t]*\([^)\n]*\))?[ \t]+)?(?:default[ \t]+)?(?:const[ \t]+)?'
    r'(?:async[ \t]+)?(?:unsafe[ \t]+)?(?:extern[ \t]+(?:"[^"\n]*"[ \t]+)?)?'
    r'(?P<kind>fn|struct|enum|trait|impl|mod|macro_rules!)(?=[\s<(])',
    re.M,
)
_NAME_RE = re.compile(r"\s+(?:r#)?(\w+)")
_TEST_ATTR_RE = re.compile(r"#\[\s*(?:\w+::)*test\b|#\[\s*cfg\s*\([^\]]*\btest\b")


def mask_rust(src: str) -> str:
    """Return ``src`` with comments and string/char contents replaced by spaces
    (newlines and length preserved) so braces can be matched safely."""
    out = list(src)
    n = len(src)

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    i = 0
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":                       # line comment
            j = src.find("\n", i)
            j = n if j == -1 else j
            blank(i, j)
            i = j
        elif c == "/" and nxt == "*":                     # block comment (nested)
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif c in "rb" and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")) \
                and (m := _RAW_STR_RE.match(src, i)):     # raw string r#"..."#
            marker = '"' + m.group(1)
            j = src.find(marker, m.end())
            j = n if j == -1 else j + len(marker)
            blank(m.end(), max(m.end(), j - len(marker)))
            i = j
        elif c == '"':                                    # normal string
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            blank(i + 1, min(j, n))
            i = j + 1
        elif c == "'":                                    # char literal or lifetime
            if nxt == "\\":
                j = src.find("'", i + 3)
                j = n if j == -1 else j
                blank(i + 1, j)
                i = j + 1
            elif i + 2 < n and src[i + 2] == "'":
                blank(i + 1, i + 2)
                i += 3
            else:                                         # lifetime like 'a
                i += 1
        else:
            i += 1
    return "".join(out)


def _match_braces(masked: str) -> dict[int, int]:
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for m in re.finditer(r"[{}]", masked):
        if m.group() == "{":
            stack.append(m.start())
        elif stack:
            pairs[stack.pop()] = m.start()
    return pairs


def _scan_header(masked: str, pos: int, pairs: dict[int, int]):
    """From the end of the keyword find the `{` (body) or `;` that ends the
    signature, ignoring anything inside () and []."""
    depth, n = 0, len(masked)
    for i in range(pos, n):
        ch = masked[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth <= 0:
            if ch == "{":
                return i, pairs.get(i)
            if ch == ";":
                return i, None
    return None, None


@dataclass
class Item:
    kind: str                 # fn | struct | enum | trait | impl | mod | macro_rules
    name: str
    sig_start: int            # offset of the line containing the keyword
    header_end: int           # offset of '{' or ';'
    end: int                  # offset just after '}' or ';'
    has_body: bool
    start_line: int = 0       # 0-based, includes doc comments + attributes
    is_test: bool = False
    parents: list["Item"] = field(default_factory=list)


def extract_items(masked: str) -> list[Item]:
    pairs = _match_braces(masked)
    items: list[Item] = []
    for m in _ITEM_RE.finditer(masked):
        kind = m.group("kind")
        name_m = _NAME_RE.match(masked, m.end())
        if kind != "impl" and not name_m:
            continue
        header_end, close = _scan_header(masked, m.end(), pairs)
        if header_end is None:
            continue
        has_body = masked[header_end] == "{"
        if has_body and close is None:
            continue
        items.append(Item(
            kind=kind.rstrip("!"),
            name=name_m.group(1) if name_m else "",
            sig_start=m.start(),
            header_end=header_end,
            end=(close if has_body else header_end) + 1,
            has_body=has_body,
        ))
    return items


def _attr_start(lines: list[str], end_idx: int):
    """If lines[end_idx] closes a multi-line #[...] attribute, return the index
    of the line where it starts."""
    for j in range(end_idx - 1, max(end_idx - 25, -1), -1):
        s = lines[j].strip()
        if s.startswith("#["):
            return j
        if not s or any(ch in s for ch in ";{}"):
            return None
    return None


def _extend_up(lines: list[str], i: int) -> int:
    """Move the start line up to include doc comments and attributes."""
    while i > 0:
        prev = lines[i - 1].strip()
        if prev.startswith(("///", "#[")):
            i -= 1
        elif prev.endswith("]") and not prev.startswith("//"):
            j = _attr_start(lines, i - 1)
            if j is None:
                break
            i = j
        else:
            break
    return i


def _is_test_attr(attrs: str) -> bool:
    attrs = re.sub(r"not\s*\(\s*test\s*\)", "", attrs)
    return bool(_TEST_ATTR_RE.search(attrs))


def chunk_source(src: str, rel_path: str, repo: str, cfg: Config) -> tuple[list[Chunk], Counter]:
    """Split one Rust file into chunks."""
    stats: Counter = Counter()
    src = src.replace("\r\n", "\n")
    masked = mask_rust(src)
    items = extract_items(masked)
    if not items:
        return [], stats

    lines = src.split("\n")
    line_starts = [0] + [m.end() for m in re.finditer("\n", src)]

    def line_of(pos: int) -> int:
        return bisect_right(line_starts, pos) - 1

    items.sort(key=lambda it: it.sig_start)
    open_items: list[Item] = []
    for it in items:
        it.start_line = _extend_up(lines, line_of(it.sig_start))
        attrs = src[line_starts[it.start_line]:it.sig_start]
        it.is_test = _is_test_attr(attrs)
        while open_items and open_items[-1].end <= it.sig_start:
            open_items.pop()
        it.parents = list(open_items)
        if it.has_body:
            open_items.append(it)

    # traits that contain default method bodies are represented by their methods
    traits_with_bodies = {
        id(p) for it in items if it.kind == "fn" and it.has_body
        for p in it.parents if p.kind == "trait"
    }

    chunks: list[Chunk] = []
    for it in items:
        if it.kind not in ("fn", "struct", "enum", "trait", "macro_rules"):
            continue
        parent = it.parents[-1] if it.parents else None
        if parent is not None and parent.kind not in ("mod", "impl", "trait"):
            continue  # nested inside a fn or macro body: part of its parent
        if it.is_test or any(p.is_test for p in it.parents):
            stats["skipped_test"] += 1
            continue

        context = ""
        if it.kind == "fn":
            if not it.has_body:
                continue  # trait method declaration / extern fn
            if parent is not None and parent.kind in ("impl", "trait"):
                kind = "method"
                context = collapse_ws(src[parent.sig_start:parent.header_end])[:200]
            else:
                kind = "function"
        elif it.kind == "trait":
            if id(it) in traits_with_bodies:
                continue
            kind = "trait"
        elif it.kind == "macro_rules":
            kind = "macro"
        else:
            kind = it.kind

        code = textwrap.dedent(src[line_starts[it.start_line]:it.end]).rstrip()
        n_lines = code.count("\n") + 1
        if n_lines < cfg.min_chunk_lines:
            stats["skipped_small"] += 1
            continue
        if len(code) > cfg.max_chunk_chars:
            stats["skipped_large"] += 1
            continue

        start, end = it.start_line + 1, it.start_line + n_lines
        chunks.append(Chunk(
            id=sha1(f"{repo}:{rel_path}:{start}:{code}")[:16],
            repo=repo, path=rel_path, kind=kind, name=it.name, context=context,
            code=code, start_line=start, end_line=end,
        ))
        stats[kind] += 1
    return chunks, stats


def chunk_repo(repo_dir: Path, repo: str, cfg: Config) -> tuple[list[Chunk], Counter]:
    """Chunk every .rs file left in a cleaned repo."""
    all_chunks: list[Chunk] = []
    total: Counter = Counter()
    for path in sorted(repo_dir.rglob("*.rs")):
        try:
            src = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            total["unreadable_files"] += 1
            continue
        rel = path.relative_to(repo_dir).as_posix()
        try:
            chunks, stats = chunk_source(src, rel, repo, cfg)
        except Exception as exc:  # one weird file must not kill the run
            log.warning("Skipping %s/%s (%s)", repo, rel, exc)
            total["parse_errors"] += 1
            continue
        all_chunks.extend(chunks)
        total.update(stats)
    return all_chunks, total


# =========================================================================== #
# Part 2 - prompts
# =========================================================================== #

SYSTEM_PROMPT = """You create instruction data for training a Rust code-generation model.

You receive a piece of Rust code taken from an open-source project. Write the message a developer would type to an AI coding assistant so that the assistant's best possible reply is exactly this code.

Rules:
- Write as the developer, in English, first person ("Write a function that...", "I need...", "How do I...").
- Be specific enough that this code is the natural answer: say what it does, its inputs, its output / return type, error handling, and any algorithm, trait or constraint that is visible in the code. Describe behaviour, not line by line.
- Use the real names of the function / type and of the types it depends on when that is needed for the code to fit.
- NEVER paste the code or use code fences. Short inline identifiers in backticks are fine.
- Do not write "the code above", "this snippet" or "the given code". Do not mention file paths or the repository name.
- Do not invent behaviour that is not in the code.
- If you are asked for several requests, make them genuinely different: one short and direct, one detailed with requirements, one phrased as a problem or question.
- Answer with JSON only, exactly: {"questions": ["...", "..."]}"""

KIND_HINT = {
    "function": "a free function: ask for it to be implemented",
    "method": "a method inside an impl/trait block: ask for it to be implemented for that type",
    "struct": "a struct definition: ask for the data type to be defined",
    "enum": "an enum definition: ask for the enum to be defined",
    "trait": "a trait definition: ask for the trait to be defined",
    "macro": "a macro_rules! macro: ask for the macro to be written",
}


def build_user_prompt(chunk: Chunk, meta: RepoMeta, n: int = 1) -> str:
    about = meta.full_name + (f" - {meta.description}" if meta.description else "")
    parts = [
        f"Project (for background only): {about}",
        f"Kind: {chunk.kind} - {KIND_HINT.get(chunk.kind, '')}",
        f"Name: {chunk.name}",
    ]
    if chunk.context:
        parts.append(f"Enclosing block: {chunk.context}")
    parts.append(f"Number of requests to write: {n}")
    parts.append(f"\n```rust\n{chunk.code}\n```")
    return "\n".join(parts)


def parse_questions(raw: str) -> list[str]:
    """Pull the list of questions out of the model reply (tolerates extra text)."""
    raw = raw.strip()
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", raw, re.S)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                data = None
    if isinstance(data, dict):
        data = data.get("questions", data.get("question"))
    if isinstance(data, str):
        data = [data]
    if not isinstance(data, list):
        return []
    return [collapse_ws(q) for q in data if isinstance(q, str) and q.strip()]


def is_valid_question(q: str, chunk: Chunk) -> bool:
    if not 20 <= len(q) <= 1500:
        return False
    if "```" in q:
        return False
    # the model must not leak the answer
    first_line = chunk.code.split("\n", 1)[0].strip()
    if len(first_line) > 25 and first_line in q:
        return False
    return True
