"""Apply idempotent modern-PyTorch and strict-holdout fixes to EndoNeRF."""
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
    parser.add_argument("--repo", type=Path, default=Path(r"E:\third_party\EndoNeRF"))
    args = parser.parse_args()
    changed = []
    training = args.repo / "run_endonerf.py"
    if replace_once(
        training,
        "i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in args.skip_frames)])  # use all frames for reconstruction",
        "i_train = np.array([i for i in np.arange(int(images.shape[0])) "
        "if (i not in i_test and i not in i_val and i not in args.skip_frames)]) "
        "# strict common holdout",
    ):
        changed.append(str(training))

    helper = args.repo / "run_endonerf_helpers.py"
    old_import = "from torchsearchsorted import searchsorted"
    new_import = """try:
    from torchsearchsorted import searchsorted
except ImportError:
    def searchsorted(a, v, out=None, side='left'):
        result = torch.searchsorted(
            a.contiguous(), v.contiguous(), right=(side == 'right')
        )
        if out is not None:
            out.copy_(result)
            return out
        return result"""
    if replace_once(helper, old_import, new_import):
        changed.append(str(helper))
    print("patched" if changed else "already patched", *changed)


if __name__ == "__main__":
    main()
