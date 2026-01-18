"""URL -> local path resolver for DRAMA-X.

Fast supervision JSONL stores `image` as an S3 URL like:
  https://.../data/drama/combined/titan/clip_305_000786/frame_000786.png

On your machine, you said the dataset root is:
  /data2/automan/data/drama_data
which contains `combined/` at top level. So the local path is:
  {DRAMA_DATA_ROOT}/combined/titan/clip_305_000786/frame_000786.png

This module centralizes that mapping, so dataset / visualization / training
all use the same logic.
"""

import os
import re
from urllib.parse import urlparse
from typing import Optional, Tuple


# Typical markers in URL paths
_DRAMA_MARKERS = [
    "/data/drama/",  # https://.../data/drama/combined/...
    "/drama/",       # sometimes .../drama/combined/...
]

_FRAME_RE = re.compile(r"(frame[_-]?)(\d{3,})", re.IGNORECASE)


def strip_query_fragment(url: str) -> str:
    """Remove ?query and #fragment."""
    return url.split("?", 1)[0].split("#", 1)[0]


def parse_relative_path_from_url(image_url: str) -> str:
    """Extract DRAMA relative path like `combined/.../frame_xxxxx.png` from a URL.

    Raises:
        ValueError if marker not found.
    """
    u = strip_query_fragment(image_url)
    path = urlparse(u).path  # only the path part

    for m in _DRAMA_MARKERS:
        if m in path:
            rel = path.split(m, 1)[1]
            return rel.lstrip("/")

    # If no marker, assume the URL already ends with combined/.../frame...
    # Try to find "combined/" as a fallback.
    idx = path.find("/combined/")
    if idx != -1:
        return path[idx + 1 :].lstrip("/")

    raise ValueError(f"Unrecognized DRAMA url path (can't find marker): {image_url}")


def resolve_local_image_path(image_url: str, data_root: Optional[str] = None) -> str:
    """Map an image URL to local absolute path.

    data_root defaults to env DRAMA_DATA_ROOT, else `/data2/automan/data/drama_data`.
    """
    if data_root is None:
        data_root = os.getenv("DRAMA_DATA_ROOT", "/data2/automan/data/drama_data")

    rel = parse_relative_path_from_url(image_url)
    return os.path.join(data_root, rel)


def resolve_clip_dir(image_url: str, data_root: Optional[str] = None) -> str:
    """Return directory that contains frames for this sample."""
    return os.path.dirname(resolve_local_image_path(image_url, data_root=data_root))


def extract_frame_index(path_or_name: str) -> Optional[int]:
    """Extract numeric frame index from filename like frame_000786.png.

    Returns None if not found.
    """
    name = os.path.basename(path_or_name)
    m = _FRAME_RE.search(name)
    if not m:
        return None
    return int(m.group(2))


def find_keyframe_file(clip_dir: str, keyframe_index: Optional[int]) -> Tuple[Optional[str], list]:
    """List frame files in clip_dir and (optionally) find best match keyframe.

    Returns:
        keyframe_path: best matching frame path (or None if directory empty)
        files_sorted: list of (frame_idx, filepath) sorted by frame_idx
    """
    if not os.path.isdir(clip_dir):
        return None, []

    candidates = []
    for fn in os.listdir(clip_dir):
        if not (fn.lower().endswith(".jpg") or fn.lower().endswith(".jpeg") or fn.lower().endswith(".png")):
            continue
        idx = extract_frame_index(fn)
        if idx is None:
            continue
        candidates.append((idx, os.path.join(clip_dir, fn)))

    candidates.sort(key=lambda x: x[0])
    if not candidates:
        return None, []

    if keyframe_index is None:
        return candidates[-1][1], candidates

    # exact match preferred; otherwise nearest index
    best = min(candidates, key=lambda x: abs(x[0] - keyframe_index))
    return best[1], candidates
