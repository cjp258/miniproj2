from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.detrac_dataset import DetracDetectionDataset, collate_fn
from src.metrics import detection_prf1
from src.modeling import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-xml-dir", type=str, required=True)
    p.add_argument("--train-img-root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--output-dir", type=str, default="outputs")
    return p.parse_args()


def evaluate_f1(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            preds = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
            tgts = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
            all_preds.extend(preds)
            all_targets.extend(tgts)

    return detection_prf1(all_preds, all_targets)["f1"]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = DetracDetectionDataset(args.train_xml_dir, args.train_img_root)
    if len(dataset) < 10:
        raise ValueError("Dataset seems too small. Check paths to UA-DETRAC XML/images.")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=5).to(device)  # background + 4 classes

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    train_losses = []
    val_losses = []
    val_f1s = []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running / len(train_loader)
        train_losses.append(avg_train_loss)

        model.train()
        val_running = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_running += sum(loss_dict.values()).item()

        avg_val_loss = val_running / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        f1 = evaluate_f1(model, val_loader, device)
        val_f1s.append(f1)

        print(
            f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, val_f1={f1:.4f}"
        )

    torch.save(model.state_dict(), out_dir / "model.pt")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_f1s, label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_f1_curve.png", dpi=150)
    plt.close()

    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses, "val_f1": val_f1s}, f, indent=2)

    print(f"Saved model and curves to: {out_dir}")


if __name__ == "__main__":
    main()
