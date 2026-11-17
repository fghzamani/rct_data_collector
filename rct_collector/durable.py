"""Crash-safe file writes.

A write that is only renamed into place (tmp -> os.replace) is not durable:
after a power loss the rename can reach the disk before the data, leaving a
0-byte file under the final name. This is what happened to probe 109 when the
laptop was force-shut-down on 2026-11-13. The helpers here fsync the data
before the rename and the directory after it, so a file is either the old
version, or the complete new one.
"""
import json
import os


def fsync_dir(path: str) -> None:
    """fsync the directory holding `path` so a rename/create in it is durable."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def fsync_path(path: str) -> None:
    """fsync an already-written file by path."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def replace_durable(tmp: str, path: str) -> None:
    """fsync `tmp`, atomically rename it to `path`, fsync the directory."""
    fsync_path(tmp)
    os.replace(tmp, path)
    fsync_dir(path)


def write_text_durable(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path)


def write_json_durable(path: str, obj, **kw) -> None:
    kw.setdefault("indent", 2)
    write_text_durable(path, json.dumps(obj, **kw))
