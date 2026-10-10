"""Bounded-memory extraction of the local VGGFace2 training archive."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import os
import shutil
import tarfile
import time
import uuid


def extract_train_archive(archive, destination, progress_seconds=10):
    """Extract train/ only; compare existing files and never overwrite differences.

    Restart scans the gzip stream again, but writes only missing files. This is
    archive-byte preparation, not face alignment or full image decoding.
    """
    archive, root = Path(archive).resolve(), Path(destination).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Training archive is missing: {archive}")
    root.mkdir(parents=True, exist_ok=True)
    result = {"archive": str(archive), "destination": str(root),
              "written": 0, "existing_verified": 0, "file_bytes": 0,
              "all_images_decode": "not_checked"}
    print(f"Extracting {archive} -> {root}; free={shutil.disk_usage(root).free / 2**30:.1f} GiB", flush=True)
    last_report = time.monotonic()
    # r|gz avoids seeking and storing an index of millions of archive members.
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            tar.members.clear()
            name = member.name.replace("\\", "/")
            rel = PurePosixPath(name)
            if (rel.is_absolute() or ".." in rel.parts or not rel.parts
                    or rel.parts[0] != "train"
                    or any(":" in p or p.endswith((".", " ")) or PureWindowsPath(p).is_reserved() for p in rel.parts)):
                raise ValueError(f"Unsafe or unexpected training archive path: {member.name}")
            target = root.joinpath(*rel.parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"Archive path escapes destination: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issparse() or len(rel.parts) < 3:
                raise ValueError(f"Unsupported archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source:
                if target.exists():
                    if not target.is_file() or target.stat().st_size != member.size:
                        raise FileExistsError(f"Existing image differs; preserve it and investigate: {target}")
                    with target.open("rb") as old:
                        while chunk := source.read(1024 * 1024):
                            if old.read(len(chunk)) != chunk:
                                raise FileExistsError(f"Existing image differs; preserve it and investigate: {target}")
                    result["existing_verified"] += 1
                else:
                    if shutil.disk_usage(root).free < member.size + 1024**3:
                        raise OSError("Insufficient free space (requires 1 GiB reserve); extraction can be resumed.")
                    temp = target.with_name(target.name + f".{uuid.uuid4().hex}.partial")
                    try:
                        with temp.open("xb") as out:
                            shutil.copyfileobj(source, out, length=1024 * 1024)
                        if temp.stat().st_size != member.size:
                            raise OSError(f"Incomplete archive member: {member.name}")
                        # On Windows rename fails if another process created target.
                        if target.exists():
                            raise FileExistsError(f"Image appeared during extraction: {target}")
                        os.rename(temp, target)
                    finally:
                        temp.unlink(missing_ok=True)
                    result["written"] += 1
            result["file_bytes"] += member.size
            if time.monotonic() - last_report >= progress_seconds:
                print(f"written={result['written']:,}, existing verified={result['existing_verified']:,}, "
                      f"uncompressed={result['file_bytes'] / 2**30:.2f} GiB", flush=True)
                last_report = time.monotonic()
    if result["written"] + result["existing_verified"] == 0:
        raise ValueError("Archive contained no training files")
    result["status"] = "extraction_completed"
    print(result, flush=True)
    return result
