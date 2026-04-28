from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.detrac_dataset import DetracDetectionDataset, collate_fn
from src.metrics import approximate_map_from_pr, detection_prf1
from src.modeling import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--test-xml-dir", type=str, required=True)
    p.add_argument("--test-img-root", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--output-dir", type=str, default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = DetracDetectionDataset(args.test_xml_dir, args.test_img_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=5)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Testing"):
            images = [img.to(device) for img in images]
            preds = model(images)
            preds = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
            tgts = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
            all_preds.extend(preds)
            all_targets.extend(tgts)

    metrics = detection_prf1(all_preds, all_targets)
    metrics["mAP"] = approximate_map_from_pr(metrics["precision"], metrics["recall"])

    print("=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
