"""Run official EndoGaussian training on a STIR clip with every frame as training frame.

Executed from the EndoGaussian repository root. The EndoNeRF loader holds out
every eighth frame by default; for point tracking the first and the last
frame must be observed, as they are by the fixed-identity tracker, so the
training split is set to all frames. All other code and settings are the
official ones.
"""
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())

import scene.endo_loader as endo_loader  # noqa: E402

_original_init = endo_loader.EndoNeRF_Dataset.__init__


def _all_frames_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.train_idxs = list(range(len(self.image_paths)))


endo_loader.EndoNeRF_Dataset.__init__ = _all_frames_init
sys.argv = ["train.py"] + sys.argv[1:]
runpy.run_path("train.py", run_name="__main__")
