"""Resumable multi-connection downloader for large public benchmark archives."""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import urllib.request
from pathlib import Path


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download(
    url: str,
    output: Path,
    connections: int,
    chunk_mib: int,
    request_chunks: int = 1,
) -> None:
    if connections <= 0 or chunk_mib <= 0 or request_chunks <= 0:
        raise ValueError("connections, chunk_mib, and request_chunks must be positive")
    total = remote_size(url)
    chunk_size = chunk_mib * 1024 * 1024
    chunks = [
        (start, min(start + chunk_size, total) - 1)
        for start in range(0, total, chunk_size)
    ]
    state_path = output.with_suffix(output.suffix + ".ranges.json")
    initial_size = output.stat().st_size if output.exists() else 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("url") != url or state.get("total") != total:
            raise ValueError("range state belongs to a different download")
        if state.get("chunk_size", chunk_size) != chunk_size:
            raise ValueError("range state uses a different chunk size")
        completed = set(map(int, state.get("completed", [])))
    else:
        completed = {
            index for index, (_, end) in enumerate(chunks) if end < initial_size
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    if not state_path.exists():
        state_path.write_text(
            json.dumps(
                {
                    "url": url,
                    "total": total,
                    "chunk_size": chunk_size,
                    "completed": sorted(completed),
                }
            ),
            encoding="utf-8",
        )
    with output.open("ab") as handle:
        handle.truncate(total)

    pending: queue.Queue[list[int]] = queue.Queue()
    group: list[int] = []
    for index in range(len(chunks)):
        if index in completed:
            if group:
                pending.put(group)
                group = []
            continue
        if group and (index != group[-1] + 1 or len(group) >= request_chunks):
            pending.put(group)
            group = []
        group.append(index)
    if group:
        pending.put(group)
    lock = threading.Lock()
    failed: list[BaseException] = []

    def save_state() -> None:
        state = {
            "url": url,
            "total": total,
            "chunk_size": chunk_size,
            "completed": sorted(completed),
        }
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        temporary.replace(state_path)

    def worker() -> None:
        while failed == []:
            try:
                indices = pending.get_nowait()
            except queue.Empty:
                return
            start = chunks[indices[0]][0]
            end = chunks[indices[-1]][1]
            for attempt in range(20):
                try:
                    request = urllib.request.Request(
                        url,
                        headers={
                            "Range": f"bytes={start}-{end}",
                            "User-Agent": "MVBRDF-SHR benchmark downloader",
                        },
                    )
                    with urllib.request.urlopen(request, timeout=120) as response:
                        if response.status != 206:
                            raise RuntimeError(
                                f"server ignored range {start}-{end}: {response.status}"
                            )
                        with output.open("r+b", buffering=0) as handle:
                            handle.seek(start)
                            remaining = end - start + 1
                            while remaining:
                                block = response.read(min(4 * 1024 * 1024, remaining))
                                if not block:
                                    raise EOFError(f"short range {start}-{end}")
                                handle.write(block)
                                remaining -= len(block)
                    with lock:
                        completed.update(indices)
                        save_state()
                    break
                except (OSError, EOFError, RuntimeError) as error:
                    if attempt == 19:
                        failed.append(error)
                        return
                    time.sleep(min(30, 2 + attempt))
            pending.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(connections)]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        with lock:
            downloaded = sum(chunks[index][1] - chunks[index][0] + 1 for index in completed)
        print(
            f"downloaded={downloaded / 2**30:.2f}/{total / 2**30:.2f} GiB "
            f"({100 * downloaded / total:.1f}%)",
            flush=True,
        )
        time.sleep(30)
    for thread in threads:
        thread.join()
    if failed:
        raise failed[0]
    state_path.unlink(missing_ok=True)
    print(f"Download complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--connections", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument(
        "--request-chunks",
        type=int,
        default=1,
        help="coalesce this many adjacent state chunks into one HTTP range",
    )
    args = parser.parse_args()
    download(
        args.url,
        args.output,
        args.connections,
        args.chunk_mib,
        args.request_chunks,
    )


if __name__ == "__main__":
    main()
