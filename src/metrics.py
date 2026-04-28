from __future__ import annotations

from typing import Dict, List

import torch
from torchvision.ops import box_iou


def detection_prf1(
    preds: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_thresh: float = 0.5,
    score_thresh: float = 0.5,
):
    tp = fp = fn = 0

    for pred, tgt in zip(preds, targets):
        pboxes = pred["boxes"]
        pscores = pred["scores"]
        plabels = pred["labels"]

        keep = pscores >= score_thresh
        pboxes = pboxes[keep]
        plabels = plabels[keep]

        tboxes = tgt["boxes"]
        tlabels = tgt["labels"]

        matched_targets = set()

        for pb, pl in zip(pboxes, plabels):
            same_class_idx = (tlabels == pl).nonzero(as_tuple=False).flatten()
            if len(same_class_idx) == 0:
                fp += 1
                continue

            sb = tboxes[same_class_idx]
            ious = box_iou(pb.unsqueeze(0), sb).squeeze(0)
            best_iou, best_j = torch.max(ious, dim=0)
            target_global_idx = same_class_idx[best_j].item()

            if best_iou.item() >= iou_thresh and target_global_idx not in matched_targets:
                tp += 1
                matched_targets.add(target_global_idx)
            else:
                fp += 1

        fn += len(tboxes) - len(matched_targets)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def approximate_map_from_pr(precision: float, recall: float) -> float:
    # Placeholder coarse approximation for assignment baseline.
    return float(precision * recall)
