from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from torch.utils.data import Dataset

CLASS_TO_ID = {
    "car": 1,
    "van": 2,
    "bus": 3,
    "others": 4,
}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


@dataclass
class DetracRecord:
    image_path: Path
    boxes: List[Tuple[float, float, float, float]]  # x1, y1, x2, y2
    labels: List[int]


def _parse_xml(xml_path: Path, img_root: Path) -> List[DetracRecord]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    video_name = xml_path.stem
    video_img_dir = img_root / video_name

    records: List[DetracRecord] = []
    for frame in root.findall(".//frame"):
        fid = int(frame.attrib["num"])
        img_path = video_img_dir / f"img{fid:05d}.jpg"
        if not img_path.exists():
            continue

        frame_boxes: List[Tuple[float, float, float, float]] = []
        frame_labels: List[int] = []

        target_list = frame.find("target_list")
        if target_list is None:
            continue

        for target in target_list.findall("target"):
            attr = target.find("attribute")
            box = target.find("box")
            if box is None:
                continue

            vehicle_type = "others"
            if attr is not None:
                vehicle_type = attr.get("vehicle_type", "others")
            label = CLASS_TO_ID.get(vehicle_type, CLASS_TO_ID["others"])

            left = float(box.attrib["left"])
            top = float(box.attrib["top"])
            width = float(box.attrib["width"])
            height = float(box.attrib["height"])
            x1, y1 = left, top
            x2, y2 = left + width, top + height

            frame_boxes.append((x1, y1, x2, y2))
            frame_labels.append(label)

        if frame_boxes:
            records.append(DetracRecord(img_path, frame_boxes, frame_labels))

    return records


class DetracDetectionDataset(Dataset):
    def __init__(self, xml_dir: str | Path, img_root: str | Path):
        self.xml_dir = Path(xml_dir)
        self.img_root = Path(img_root)
        self.records: List[DetracRecord] = []

        xml_files = sorted(self.xml_dir.glob("*.xml"))
        for xml_path in xml_files:
            self.records.extend(_parse_xml(xml_path, self.img_root))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = cv2.imread(str(rec.image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {rec.image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        boxes = torch.tensor(rec.boxes, dtype=torch.float32)
        labels = torch.tensor(rec.labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
        }

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
