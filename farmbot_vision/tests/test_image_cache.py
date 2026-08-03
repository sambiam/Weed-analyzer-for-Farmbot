from __future__ import annotations

from pathlib import Path

from farmbot_vision.image_cache import ImageFileCache


def test_trim_ignores_entry_removed_between_glob_and_stat(tmp_path, monkeypatch):
    cache = ImageFileCache(tmp_path, maximum_files=1)
    disappearing = tmp_path / "disappearing.jpg"
    surviving = tmp_path / "surviving.jpg"
    disappearing.write_bytes(b"old")
    surviving.write_bytes(b"new")

    original_stat = Path.stat
    raised = False

    def racing_stat(path, *args, **kwargs):
        nonlocal raised
        if path == disappearing and not raised:
            raised = True
            disappearing.unlink(missing_ok=True)
            raise FileNotFoundError(disappearing)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", racing_stat)

    cache._trim()

    assert raised
    assert surviving.is_file()
