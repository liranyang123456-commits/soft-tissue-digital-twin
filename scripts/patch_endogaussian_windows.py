"""Apply minimal idempotent Windows compatibility fixes to EndoGaussian."""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(r"E:\third_party\EndoGaussian"),
    )
    args = parser.parse_args()
    changed = []
    render = args.repo / "render.py"
    if replace_once(
        render,
        "os.sched_setaffinity(0, cpu_list)",
        "if hasattr(os, 'sched_setaffinity'):\n    os.sched_setaffinity(0, cpu_list)",
    ):
        changed.append(str(render))
    print("patched" if changed else "already patched", *changed)


if __name__ == "__main__":
    main()
