#!/bin/bash
EG=/root/src/EndoGaussian
PY=/root/venvs/endogaussian/bin/python
SCRIPTS="/mnt/e/Digital twin of soft tissue/mvbrdf_shr_next/scripts"
DATA=/mnt/g/E_ZhiTai1_MovedData/MIS_Datasets/STIRC2024/endonerf_format
WORK=/root/stir_eg
OUT="/mnt/e/Digital twin of soft tissue/mvbrdf_shr_next/outputs/stir_tracking/endogaussian"
cd $EG
CLIPS=$($PY "$SCRIPTS/stir_clip_names.py" "$DATA/${CLIP_LIST:-subset20.json}")
for clip in $CLIPS; do
  if [ -f "$OUT/$clip/tracks.npz" ]; then continue; fi
  mkdir -p "$OUT/$clip" "$WORK/$clip"
  start=$(date +%s)
  $PY "$SCRIPTS/endogaussian_stir_train.py" -s "$DATA/$clip" --model_path "$WORK/$clip" --port 6031 --configs arguments/endonerf/pulling.py > "$WORK/$clip/train.log" 2>&1
  $PY "$SCRIPTS/endogaussian_stir_tracks.py" --model_path "$WORK/$clip" --configs arguments/endonerf/pulling.py --out "$OUT/$clip/tracks.npz" > "$WORK/$clip/tracks.log" 2>&1
  echo "$clip $(( $(date +%s) - start ))s $(ls "$OUT/$clip" 2>/dev/null)"
done
echo EG_STIR_DONE
