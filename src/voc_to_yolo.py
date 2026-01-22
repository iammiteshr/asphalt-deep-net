#!/usr/bin/env python3
"""
Convert Pascal VOC XML annotations to YOLO (box) format for a single or multiple classes.
Default behavior: map everything to class 0 ('pothole').
If your XML has specific names (e.g., 'pothole', 'crack'), you can filter by --only pothole.
"""

import argparse, os, glob, shutil
from pathlib import Path
from lxml import etree
from tqdm import tqdm
import random

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder with images")
    ap.add_argument("--ann", required=True, help="Folder with VOC XML annotations")
    ap.add_argument("--out", default="data", help="Output root (will create images/ and labels/)")
    ap.add_argument("--split", type=float, default=0.85, help="Train split ratio (rest goes to val)")
    ap.add_argument("--only", nargs="*", default=None, help="If set, only keep these class names (e.g., --only pothole)")
    ap.add_argument("--copy", action="store_true", help="Copy images instead of symlink")
    return ap.parse_args()

def voc_to_yolo_bbox(box, w, h):
    # VOC: xmin, ymin, xmax, ymax  -> YOLO: cx, cy, w, h (normalized)
    xmin, ymin, xmax, ymax = box
    bw = xmax - xmin
    bh = ymax - ymin
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return (cx / w, cy / h, bw / w, bh / h)

def main():
    args = parse_args()
    images = sorted(glob.glob(os.path.join(args.images, "*.*")))
    xmls = sorted(glob.glob(os.path.join(args.ann, "*.xml")))
    assert xmls, f"No XML files found under {args.ann}"
    out_root = Path(args.out)
    for p in [out_root/"images/train", out_root/"images/val", out_root/"labels/train", out_root/"labels/val"]:
        p.mkdir(parents=True, exist_ok=True)

    # Match XML to image by stem
    xml_by_stem = {Path(x).stem: x for x in xmls}
    pairs = []
    for img in images:
        stem = Path(img).stem
        if stem in xml_by_stem:
            pairs.append((img, xml_by_stem[stem]))

    random.shuffle(pairs)
    split_idx = int(len(pairs) * args.split)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    def handle_pair(img_path, xml_path, subset):
        # Parse VOC xml
        tree = etree.parse(xml_path)
        root = tree.getroot()
        w = int(root.findtext("size/width"))
        h = int(root.findtext("size/height"))
        objects = root.findall("object")
        yolo_lines = []
        for obj in objects:
            name = obj.findtext("name").strip().lower()
            if args.only and name not in [c.lower() for c in args.only]:
                continue
            bnd = obj.find("bndbox")
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
            cx, cy, bw, bh = voc_to_yolo_bbox((xmin, ymin, xmax, ymax), w, h)
            # class 0 = pothole by default
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            return  # skip images without (selected) objects

        # Write label
        out_label = out_root / "labels" / subset / f"{Path(img_path).stem}.txt"
        out_label.write_text("\n".join(yolo_lines))

        # Copy/symlink image
        out_img = out_root / "images" / subset / Path(img_path).name
        if args.copy:
            shutil.copy2(img_path, out_img)
        else:
            # Try to symlink; if fails on Windows, fallback to copy
            try:
                if out_img.exists():
                    out_img.unlink()
                os.symlink(os.path.abspath(img_path), out_img)
            except Exception:
                shutil.copy2(img_path, out_img)

    for (img, xml) in tqdm(train_pairs, desc="Train pairs"):
        handle_pair(img, xml, "train")
    for (img, xml) in tqdm(val_pairs, desc="Val pairs"):
        handle_pair(img, xml, "val")

    print("Done. YOLO dataset at:", out_root.resolve())

if __name__ == "__main__":
    main()
