#!/usr/bin/env python3
import argparse, os, random, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="Root containing images/ and labels/")
    ap.add_argument("--ratio", type=float, default=0.85, help="Train split ratio")
    args = ap.parse_args()

    root = Path(args.root)
    imgs = sorted((root/"images").glob("*/*.*")) if (root/"images/train").exists() else sorted((root/"images").glob("*.*"))
    # If already split, do nothing
    if (root/"images/train").exists():
        print("Looks already split. Nothing to do.")
        return

    (root/"images/train").mkdir(parents=True, exist_ok=True)
    (root/"images/val").mkdir(parents=True, exist_ok=True)
    (root/"labels/train").mkdir(parents=True, exist_ok=True)
    (root/"labels/val").mkdir(parents=True, exist_ok=True)

    pairs = []
    for img in imgs:
        stem = img.stem
        lbl = root/"labels"/f"{stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))

    random.shuffle(pairs)
    cut = int(len(pairs)*args.ratio)
    train = pairs[:cut]
    val = pairs[cut:]

    def move_pair(img, lbl, subset):
        shutil.copy2(img, root/f"images/{subset}"/img.name)
        shutil.copy2(lbl, root/f"labels/{subset}"/lbl.name)

    for img,lbl in train:
        move_pair(img,lbl,"train")
    for img,lbl in val:
        move_pair(img,lbl,"val")

    print(f"Split {len(train)} train and {len(val)} val samples.")

if __name__ == "__main__":
    main()
