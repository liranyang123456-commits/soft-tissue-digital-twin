# Medical & Biological Engineering & Computing submission

Target journal: **Medical & Biological Engineering & Computing (MBEC)**,
Springer Nature / IFMBE, hybrid publishing.

Official Original Research requirements:

- abstract no longer than 200 words;
- main sections: Introduction, Methods, Results, Discussion, Conclusion;
- manuscript no longer than 18 pages;
- scientific rigor, methodological contribution, and practical biomedical
  relevance must be explicit.

## Build status

- `main_full.tex` is the uncondensed source used only to verify the MBEC
  Springer template; it is not a submission file.
- `main.tex` is generated from verified experiment summaries by
  `scripts/build_mbec_submission.py`.
- The 60-scenario FEM campaign, strict EndoGaussian holdout, and
  Gaussian-to-volume mapping are complete.
- Tissue SSIM uses the local SSIM map averaged only over tissue pixels;
  masked-out black regions are excluded.
- New force-instrumented endoscopic phantom acquisition is outside the
  available experimental scope and will remain a stated limitation.

## Required final uploads

- `main.pdf`
- `manuscript_source.zip`
- `cover_letter.txt`
- optional supplementary experiment archive

## Final verified results

- FEM60 background-modulus error: median 9.59%, mean 9.95%.
- FEM60 inclusion-modulus error: median 32.02%, mean 30.08%.
- FEM60 contrast error: median 32.44%, mean 33.83%.
- EndoGaussian common holdout: tissue PSNR 36.589 dB, SSIM 0.9386.
- Present method common holdout: tissue PSNR 16.714 dB, SSIM 0.5640.
- Gaussian-to-volume mapping: 28,523 nodes, 149,028 tetrahedra, and
  38.743% covered boundary nodes in the cross-subject interface test.

## Required author confirmation before submission

The responsible ethics committee name and approval or waiver reference
for the de-identified hospital CT case are not available in the repository.
They must be inserted before the manuscript is uploaded. No identifier has
been inferred or invented.
