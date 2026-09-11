"""Video sources: a local path, or an http(s) URL fetched once into the user cache."""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

CACHE_ENV = "ACE_EGO_HAND_CACHE"


def is_url(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https")


def cache_dir() -> Path:
    """`$ACE_EGO_HAND_CACHE`, else `$XDG_CACHE_HOME/ace-ego-hand`, else `~/.cache/ace-ego-hand`."""

    if (override := os.environ.get(CACHE_ENV)) is not None:
        return Path(override)
    base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return base / "ace-ego-hand"


def fetch_video(source: str) -> Path:
    """The local file for a source: the path itself, or the URL downloaded into the cache."""

    if not is_url(source):
        return Path(source)
    name = Path(urlparse(source).path).name or "video"
    target = cache_dir() / "videos" / hashlib.sha1(source.encode()).hexdigest()[:12] / name
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(source) as response, partial.open("wb") as file:
        shutil.copyfileobj(response, file)
    partial.replace(target)
    return target


def output_dir(source: str, video: Path, override: Path | None) -> Path:
    """Where a source's outputs go: `override`, else next to a local file, else the working directory."""

    if override is not None:
        override.mkdir(parents=True, exist_ok=True)
        return override
    return Path.cwd() if is_url(source) else video.parent
