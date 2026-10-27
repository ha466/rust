"""repo.py - everything that touches a repository on disk.

    parse_repo()     "owner/name" or a GitHub URL  -> RepoRef
    clone_repo()     shallow `git clone`
    read_metadata()  commit, description and license (read BEFORE cleaning)
    clean_repo()     delete everything that is not Rust source code
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from core import (
    ALWAYS_SKIP_DIRS, CODE_EXTS, GENERATED_MARKERS, TEST_DIRS,
    Config, RepoMeta, log,
)

_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", re.I
)
_SHORT_RE = re.compile(r"^([\w.-]+)/([\w.-]+?)(?:\.git)?$")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def web_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"

    @property
    def clone_url(self) -> str:
        return self.web_url + ".git"

    @property
    def slug(self) -> str:
        return f"{self.owner}__{self.name}"


def parse_repo(spec: str) -> RepoRef:
    """Accepts 'owner/name', 'github.com/owner/name' or a full https URL."""
    spec = spec.strip()
    m = _URL_RE.match(spec) or _SHORT_RE.match(spec)
    if not m:
        raise ValueError(f"Not a valid GitHub repo: {spec!r} (use 'owner/name' or a github.com URL)")
    return RepoRef(m.group(1), m.group(2))


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def _rmtree(path: Path) -> None:
    """rmtree that also works for read-only files (.git objects on Windows)."""
    if path.is_symlink():
        path.unlink()
        return

    def handler(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=handler)
    else:
        shutil.rmtree(path, onerror=handler)


# --------------------------------------------------------------------------- #
# Clone
# --------------------------------------------------------------------------- #
def git_clone(url: str, dest: Path, timeout: int = 900) -> Path:
    if dest.exists():
        _rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", "--single-branch", "--quiet", url, str(dest)]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # never hang on a password prompt
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except FileNotFoundError as exc:
        raise RuntimeError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"git clone timed out after {timeout}s") from exc
    if res.returncode != 0:
        raise RuntimeError(f"git clone failed: {res.stderr.strip()[:300]}")
    return dest


def clone_repo(ref: RepoRef, dest_root: Path) -> Path:
    log.info("Cloning %s ...", ref.full_name)
    return git_clone(ref.clone_url, dest_root / ref.slug)


# --------------------------------------------------------------------------- #
# Metadata (must run before clean_repo, which deletes .git / Cargo.toml / LICENSE)
# --------------------------------------------------------------------------- #
def _parse_cargo(text: str) -> dict:
    try:
        import tomllib  # Python 3.11+
        return tomllib.loads(text)
    except Exception:
        return {}


def _str(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _license_from_files(repo_dir: Path) -> str:
    found = set()
    for f in repo_dir.iterdir():
        if not (f.is_file() and re.match(r"(?i)(licen[sc]e|copying|unlicense)", f.name)):
            continue
        t = f.read_text(encoding="utf-8", errors="ignore")[:8000].lower()
        if "apache license" in t and "version 2.0" in t:
            found.add("Apache-2.0")
        elif "permission is hereby granted, free of charge" in t:
            found.add("MIT")
        elif "mozilla public license" in t:
            found.add("MPL-2.0")
        elif "gnu affero general public" in t:
            found.add("AGPL-3.0")
        elif "gnu lesser general public" in t:
            found.add("LGPL-3.0")
        elif "gnu general public license" in t:
            found.add("GPL-3.0" if "version 3" in t else "GPL-2.0")
        elif "redistribution and use in source and binary forms" in t:
            found.add("BSD-3-Clause" if "neither the name" in t else "BSD-2-Clause")
        elif "free and unencumbered software" in t:
            found.add("Unlicense")
        elif "isc license" in t:
            found.add("ISC")
    return " OR ".join(sorted(found)) if found else "unknown"


def read_metadata(ref: RepoRef, repo_dir: Path) -> RepoMeta:
    meta = RepoMeta(full_name=ref.full_name, url=ref.web_url)

    try:
        res = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        if res.returncode == 0:
            meta.commit = res.stdout.strip()
    except Exception:
        pass

    cargo = repo_dir / "Cargo.toml"
    if cargo.exists():
        data = _parse_cargo(cargo.read_text(encoding="utf-8", errors="ignore"))
        pkg = data.get("package") or (data.get("workspace") or {}).get("package") or {}
        meta.crate_name = _str(pkg.get("name"))
        meta.description = _str(pkg.get("description"))
        meta.license = _str(pkg.get("license")) or "unknown"

    if meta.license == "unknown":
        meta.license = _license_from_files(repo_dir)
    return meta


# --------------------------------------------------------------------------- #
# Cleaning: keep only Rust source files
# --------------------------------------------------------------------------- #
def _is_good_code_file(path: Path, cfg: Config) -> bool:
    try:
        size = path.stat().st_size
        if size == 0 or size > cfg.max_file_bytes:
            return False
        with open(path, "rb") as fh:
            head = fh.read(1500).decode("utf-8", errors="ignore").lower()
    except OSError:
        return False
    return not any(marker in head for marker in GENERATED_MARKERS)


def clean_repo(repo_dir: Path, cfg: Config) -> list[Path]:
    """Delete READMEs, images, docs, configs, tests, build output, generated and
    oversized files. Only ``.rs`` source files stay. Returns the kept files."""
    skip_dirs = set(ALWAYS_SKIP_DIRS)
    if not cfg.include_tests:
        skip_dirs |= TEST_DIRS

    kept: list[Path] = []
    removed = 0
    for root, dirs, files in os.walk(repo_dir, topdown=True):
        root_p = Path(root)
        for d in list(dirs):
            if d in skip_dirs or d.startswith("."):  # .git, .github, .vscode, ...
                _rmtree(root_p / d)
                dirs.remove(d)
                removed += 1
        for name in files:
            p = root_p / name
            if p.suffix in CODE_EXTS and not p.is_symlink() and _is_good_code_file(p, cfg):
                kept.append(p)
            else:
                try:
                    p.unlink()
                except OSError:
                    pass
                removed += 1

    # remove directories that became empty
    for root, _dirs, _files in os.walk(repo_dir, topdown=False):
        if Path(root) != repo_dir and not os.listdir(root):
            os.rmdir(root)

    log.info("Cleaned: kept %d .rs files, removed %d other files/dirs", len(kept), removed)
    return kept
