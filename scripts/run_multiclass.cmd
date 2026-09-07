@echo off
rem One-shot multilingual training + eval + export, run AFTER fetch_mswc completes.
rem Uses the --multiclass path (one class per greeting + "other", OR at inference)
rem — the fix RESULTS.md planned for the Run 1 binary-class failure.
set PYTHONPATH=E:\O1-Sound
cd /d E:\O1-Sound

rem Languages on disk: en de fr es pt pl ru cs fa sv-SE (fetch_mswc set).
rem Training: 20 epochs, weighted CE, cosine schedule (train.py defaults).
py -3.11 -X utf8 train.py --root data/mswc --epochs 20 ^
  --multiclass --hidden 640 --layers 2 --negatives 400 ^
  --window 1.0 --seed 0 --out checkpoints/o1sound_multiclass.pt ^
  > data/train_multiclass.log 2>&1

if errorlevel 1 (
  echo TRAIN FAILED - see data\train_multiclass.log
  exit /b 1
)

rem Per-language FRR at fixed FAR on the held-out test split.
py -3.11 -X utf8 eval.py --ckpt checkpoints/o1sound_multiclass.pt ^
  --root data/mswc --split test --out results/test_multiclass.json ^
  > data/eval_multiclass.log 2>&1

rem Export C-kernel weights (head is generic over n_classes).
py -3.11 -X utf8 scripts\export_c_kernel.py ^
  --ckpt checkpoints/o1sound_multiclass.pt ^
  --out freertos\generated\weights.h ^
  >> data/eval_multiclass.log 2>&1

echo DONE - see data/train_multiclass.log, results/test_multiclass.json
