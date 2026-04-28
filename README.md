# Mini Project 2 — Traffic Vehicle Recognition (UA-DETRAC)

This repository provides a complete **starter pipeline** for the two required tasks:

1. **Max traffic flow in a period of time**
2. **Maximum road load in a frame**

It includes:
- `train.py`: training pipeline with loss curves + validation F1 curve
- `test.py`: evaluation pipeline (Precision / Recall / F1 / mAP)
- `inference.py`: standalone video inference that writes annotated output video and reports flow + max load
- `src/`: reusable dataset, model, and metrics modules
- `report_template.md`: report structure matching assignment sections

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Dataset layout expected

```text
data/
  DETRAC-Train-Annotations-XML/
    MVI_20011.xml
    ...
  Insight-MVT_Annotation_Train/
    MVI_20011/
      img00001.jpg
      ...
  DETRAC-Test-Annotations-XML/
  Insight-MVT_Annotation_Test/
```

You can point scripts to your own paths using CLI args.

## 3) Train

```bash
python train.py \
  --train-xml-dir data/DETRAC-Train-Annotations-XML \
  --train-img-root data/Insight-MVT_Annotation_Train \
  --epochs 10 \
  --batch-size 2 \
  --output-dir outputs
```

Outputs:
- `outputs/model.pt`
- `outputs/loss_curves.png`
- `outputs/val_f1_curve.png`

## 4) Test

```bash
python test.py \
  --model-path outputs/model.pt \
  --test-xml-dir data/DETRAC-Test-Annotations-XML \
  --test-img-root data/Insight-MVT_Annotation_Test
```

Outputs precision/recall/f1/mAP to console and `outputs/test_metrics.json`.

## 5) Inference (standalone video)

```bash
python inference.py \
  --model-path outputs/model.pt \
  --video-path your_test_video.mp4 \
  --output-video outputs/annotated.mp4 \
  --output-json outputs/traffic_stats.json
```

`inference.py` reports:
- `max_road_load`: max vehicles visible in a single frame
- `max_traffic_flow_window`: max counted objects in a configurable time window (default 60s)

## 6) Notes

- This is a strong baseline built on `fasterrcnn_resnet50_fpn` (PyTorch / torchvision).
- For better grading performance, tune confidence / NMS / training epochs and optionally add tracking for de-duplicated flow counting.
