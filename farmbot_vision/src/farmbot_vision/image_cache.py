"""Small on-disk cache for processed images shown by the web UI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CachedImage:
    path: Path
    width: int
    height: int
    oriented_width: int | None = None
    oriented_height: int | None = None


class ImageFileCache:
    """Persist UI JPEGs so repeated views do not call Home Assistant again."""

    def __init__(self, directory: Path, *, maximum_files: int = 256):
        self.directory = directory
        self.maximum_files = maximum_files

    @staticmethod
    def _entry_key(entry_id: str) -> str:
        return hashlib.sha256(entry_id.encode("utf-8")).hexdigest()[:16]

    def _paths(
        self, entry_id: str, image_id: int, max_width: int, max_height: int
    ) -> tuple[Path, Path]:
        stem = f"{self._entry_key(entry_id)}-{image_id}-{max_width}x{max_height}"
        return self.directory / f"{stem}.jpg", self.directory / f"{stem}.json"

    def get(
        self, entry_id: str, image_id: int, max_width: int, max_height: int
    ) -> CachedImage | None:
        image_path, metadata_path = self._paths(entry_id, image_id, max_width, max_height)
        if not image_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = CachedImage(
                path=image_path,
                width=int(metadata["width"]),
                height=int(metadata["height"]),
                oriented_width=(
                    int(metadata["oriented_width"])
                    if metadata.get("oriented_width") is not None
                    else None
                ),
                oriented_height=(
                    int(metadata["oriented_height"])
                    if metadata.get("oriented_height") is not None
                    else None
                ),
            )
            # Access time is not reliable on every Home Assistant filesystem.
            # Touch both files so mtime provides a portable LRU approximation.
            image_path.touch()
            metadata_path.touch()
            return cached
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            image_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            return None

    def put(
        self,
        entry_id: str,
        image_id: int,
        max_width: int,
        max_height: int,
        jpeg: bytes,
        *,
        width: int,
        height: int,
        oriented_width: int | None,
        oriented_height: int | None,
    ) -> CachedImage:
        self.directory.mkdir(parents=True, exist_ok=True)
        image_path, metadata_path = self._paths(entry_id, image_id, max_width, max_height)
        image_tmp = image_path.with_suffix(".jpg.tmp")
        metadata_tmp = metadata_path.with_suffix(".json.tmp")
        image_tmp.write_bytes(jpeg)
        metadata_tmp.write_text(
            json.dumps(
                {
                    "width": width,
                    "height": height,
                    "oriented_width": oriented_width,
                    "oriented_height": oriented_height,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        image_tmp.replace(image_path)
        metadata_tmp.replace(metadata_path)
        self._trim()
        return CachedImage(
            path=image_path,
            width=width,
            height=height,
            oriented_width=oriented_width,
            oriented_height=oriented_height,
        )

    def _trim(self) -> None:
        images = sorted(
            self.directory.glob("*.jpg"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in images[self.maximum_files :]:
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
