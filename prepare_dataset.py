"""
EcoTex Sorter - dataset preparation
-----------------------------------
Merges roboflow_1 (YOLO detection), roboflow_2 (classification folders),
textileclass7 (classification folders) and optional my_photos into ONE clean
5-class dataset with train / val / test splits.

Usage (Windows):
    pip install pillow
    python prepare_dataset.py
    python prepare_dataset.py "D:\\FYP EcoTex"      (agar folder alag ho)

Output:
    <ROOT>\\dataset\\train|val|test\\<class>\\*.jpg
    <ROOT>\\dataset\\manifest.csv   (har image kahan se aayi, kis split mein gayi)

Note: <ROOT>\\dataset folder har run par dobara banta hai (purana delete hota hai).
"""
import csv
import hashlib
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict

from PIL import Image, ImageOps

# ----------------------------- SETTINGS -----------------------------------
ROOT = sys.argv[1] if len(sys.argv) > 1 else r"D:\FYP EcoTex"
RAW = os.path.join(ROOT, "raw_data")
OUT = os.path.join(ROOT, "dataset")
MY_PHOTOS = os.path.join(ROOT, "my_photos")  # optional: my_photos\cotton\shirt1_01.jpg

SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_SIDE = 512                 # images ko isse bada nahi rakhte (size kam rakhne ke liye)
KEEP_ONE_PER_GROUP = True      # augmented copies hata kar har asal image ki ek copy rakho
CROP_WHITE_BORDER = {"roboflow_2", "textileclass7"}  # Roboflow ke white padding wale datasets
MINE_SPLIT = {"train": 0.50, "val": 0.20, "test": 0.30}  # apni photos: test ko zyada hissa (real-world number)
MINE_TRAIN_REPEAT = 3   # train mein apni photos 3 baar (chhoti tadaad, domain match)
GARMENT_BLOCK = 4       # agar file ke naam garment_NN nahi, to itni consecutive photos = ek kapra
EXCLUDE_SOURCES = {"roboflow_1"}  # detection dataset, test accuracy sirf ~38% (label noise) -> hata diya

TARGET_CLASSES = ["cotton", "denim", "wool", "polyester", "silk"]
ROBOFLOW1_NAMES = ["cotton", "denim", "linen", "polyester", "silk", "wool"]  # data.yaml ka order
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ---------------------------------------------------------------------------


def norm(s):
    """'Cotton MIxed' -> 'cottonmixed'"""
    return re.sub(r"[^a-z]", "", s.lower())


def base_name(fname):
    """'Silk_33_jpg.rf.abc123.jpg' -> 'Silk_33' (augmented copies ka same base hota hai)"""
    stem = os.path.splitext(fname)[0].split(".rf.")[0]
    return re.sub(r"_(jpg|jpeg|png|bmp|webp)$", "", stem, flags=re.I)


def is_img(fname):
    return os.path.splitext(fname)[1].lower() in IMG_EXT


def find_root(path):
    """Wo folder dhoondo jis ke andar 'train' folder ho (extra nested folders ke liye)."""
    for dirpath, dirnames, _ in os.walk(path):
        if "train" in dirnames:
            return dirpath
    return None


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------- COLLECTORS ----------------------------------
def collect_yolo(source, folder, names, log):
    """Detection dataset: class filename prefix se, label file se cross-check."""
    items = []
    root = find_root(folder)
    if not root:
        log.append(f"[!] {source}: 'train' folder nahi mila in {folder}")
        return items
    for split in ("train", "valid", "test"):
        img_dir = os.path.join(root, split, "images")
        lbl_dir = os.path.join(root, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            if not is_img(fn):
                continue
            base = base_name(fn)
            m = re.match(r"([A-Za-z]+)", base)
            prefix_cls = norm(m.group(1)) if m else None
            if prefix_cls not in names:
                prefix_cls = None
            label_cls = None
            lbl = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
            if os.path.isfile(lbl):
                ids = []
                with open(lbl) as f:
                    for line in f:
                        parts = line.split()
                        if parts and parts[0].isdigit() and int(parts[0]) < len(names):
                            ids.append(int(parts[0]))
                if ids:
                    label_cls = names[Counter(ids).most_common(1)[0][0]]
            if prefix_cls and label_cls and prefix_cls != label_cls:
                log.append(f"[conflict] {source}: {fn} filename={prefix_cls} label={label_cls} -> skip")
                continue
            cls = prefix_cls or label_cls
            if not cls:
                log.append(f"[no class] {source}: {fn} -> skip")
                continue
            items.append(dict(path=os.path.join(img_dir, fn), cls=cls, source=source,
                              group=(source, base)))
    return items


def collect_folders(source, folder, log):
    """Classification dataset: train|valid|test / <class> / images"""
    items = []
    root = find_root(folder)
    if not root:
        log.append(f"[!] {source}: 'train' folder nahi mila in {folder}")
        return items
    for split in ("train", "valid", "test"):
        sdir = os.path.join(root, split)
        if not os.path.isdir(sdir):
            continue
        for cname in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, cname)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                if is_img(fn):
                    items.append(dict(path=os.path.join(cdir, fn), cls=norm(cname),
                                      source=source, group=(source, base_name(fn))))
    return items


GARMENT_RE = re.compile(r"^[A-Za-z]+\d*_\d+$")  # shirt1_01
PHONE_PREFIX = ("IMG", "PXL", "DSC", "WP", "VID", "PHOTO", "SAM")


def collect_mine(log):
    """my_photos/<class>/<garment>_<n>.jpg. Phone ke default naam ho to consecutive blocks."""
    items = []
    if not os.path.isdir(MY_PHOTOS):
        return items
    for cname in sorted(os.listdir(MY_PHOTOS)):
        cdir = os.path.join(MY_PHOTOS, cname)
        if not os.path.isdir(cdir):
            continue
        if norm(cname) not in TARGET_CLASSES:
            log.append(f"[!] my_photos: folder '{cname}' ka naam {TARGET_CLASSES} mein se nahi -> skip")
            continue
        files = sorted(os.listdir(cdir))
        imgs = [f for f in files if is_img(f)]
        bad = [f for f in files if not is_img(f)]
        if bad:
            log.append(f"[!] my_photos/{cname}: {len(bad)} file(s) supported nahi (jaise .heic), skip: {bad[:3]}")
        stems = [os.path.splitext(f)[0] for f in imgs]
        named = bool(imgs) and all(GARMENT_RE.match(t) and not t.upper().startswith(PHONE_PREFIX) for t in stems)
        for k, (fn, stem) in enumerate(zip(imgs, stems)):
            garment = stem.rsplit("_", 1)[0] if named else f"block{k // GARMENT_BLOCK}"
            items.append(dict(path=os.path.join(cdir, fn), cls=norm(cname), source="mine",
                              group=("mine", norm(cname) + "/" + garment)))
        n_groups = len({i["group"] for i in items if i["cls"] == norm(cname) and i["source"] == "mine"})
        print(f"my_photos/{cname}: {len(imgs)} photos, ~{n_groups} kapre, grouping: "
              + ("file ke naam se" if named else f"{GARMENT_BLOCK}-{GARMENT_BLOCK} consecutive photos"))
    return items


# ----------------------------- GROUPING ------------------------------------
class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def crop_white(img, thr=240):
    gray = img.convert("L")
    bbox = gray.point(lambda p: 255 if p < thr else 0).getbbox()
    if not bbox:
        return img, False
    w, h = img.size
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if (bw, bh) == (w, h) or bw < 0.5 * w or bh < 0.5 * h:
        return img, False
    return img.crop(bbox), True


def main():
    random.seed(SEED)
    log = []

    # 1) collect
    items = []
    items += collect_yolo("roboflow_1", os.path.join(RAW, "roboflow_1"), ROBOFLOW1_NAMES, log)
    items += collect_folders("roboflow_2", os.path.join(RAW, "roboflow_2"), log)
    items += collect_folders("textileclass7", os.path.join(RAW, "textileclass7"), log)
    items += collect_mine(log)
    mine_cls = {i["cls"] for i in items if i["source"] == "mine"}
    if os.path.isdir(MY_PHOTOS):
        miss = [c for c in TARGET_CLASSES if c not in mine_cls]
        if miss:
            print("[!] my_photos mein in classes ki photos NAHI hain:", miss)
    items = [i for i in items if i["source"] not in EXCLUDE_SOURCES]
    if EXCLUDE_SOURCES:
        print("Excluded sources:", sorted(EXCLUDE_SOURCES))
    print("\nFound images per source (before cleaning):")
    for s, n in Counter(i["source"] for i in items).items():
        print(f"  {s:15s} {n}")
    if not items:
        print("Koi image nahi mili. ROOT path check karo:", ROOT)
        return

    # 2) drop classes we don't use
    dropped = Counter(i["cls"] for i in items if i["cls"] not in TARGET_CLASSES)
    items = [i for i in items if i["cls"] in TARGET_CLASSES]
    print("\nDropped (not in the 5 target classes):", dict(dropped) or "none")

    # 3) groups: same base name (augmentations) + identical file content (exact duplicates)
    uf = UF()
    seen_hash = {}
    for it in items:
        uf.find(it["group"])
        h = md5(it["path"])
        if h in seen_hash:
            uf.union(it["group"], seen_hash[h])
        else:
            seen_hash[h] = it["group"]
    comps = defaultdict(list)
    for it in items:
        comps[uf.find(it["group"])].append(it)

    # 4) class conflicts -> drop whole component
    kept_comps = []
    for key, members in comps.items():
        classes = {m["cls"] for m in members}
        if len(classes) > 1:
            log.append(f"[label conflict] {key}: {sorted(classes)} -> {len(members)} images dropped")
            continue
        members = sorted(members, key=lambda m: m["path"])
        if KEEP_ONE_PER_GROUP and members[0]["source"] != "mine":
            members = members[:1]
        kept_comps.append((members[0]["cls"], members))
    n_before = len(items)
    n_after = sum(len(m) for _, m in kept_comps)
    print(f"\nAfter removing augmented copies / duplicates: {n_before} -> {n_after} images")

    # 5) stratified split at group level (no leakage between splits)
    assignment = {}
    by_class = defaultdict(list)
    for cls, members in kept_comps:
        by_class[(cls, members[0]["source"] == "mine")].append(members)
    for (cls, is_mine), groups in by_class.items():
        ratios = MINE_SPLIT if is_mine else SPLIT_RATIOS
        random.shuffle(groups)
        groups.sort(key=len, reverse=True)  # bade groups pehle, phir deficit-based assignment
        total = sum(len(g) for g in groups)
        target = {s: r * total for s, r in ratios.items()}
        got = {s: 0 for s in SPLIT_RATIOS}
        for g in groups:
            s = max(SPLIT_RATIOS, key=lambda k: target[k] - got[k])
            got[s] += len(g)
            for m in g:
                assignment[m["path"]] = s

    # 6) write output
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    counters = Counter()
    rows, skipped, cropped = [], 0, 0
    for cls, members in kept_comps:
        for m in members:
            split = assignment[m["path"]]
            try:
                img = ImageOps.exif_transpose(Image.open(m["path"])).convert("RGB")
                if m["source"] in CROP_WHITE_BORDER:
                    img, did = crop_white(img)
                    cropped += did
                if max(img.size) > MAX_SIDE:
                    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            except Exception as e:  # corrupt image
                skipped += 1
                log.append(f"[corrupt] {m['path']}: {e}")
                continue
            counters[m["source"]] += 1
            out_dir = os.path.join(OUT, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{m['source']}_{counters[m['source']]:05d}.jpg")
            img.save(out_path, "JPEG", quality=95)
            rows.append([split, cls, m["source"], m["path"], out_path])
            if m["source"] == "mine" and split == "train":
                for _ in range(MINE_TRAIN_REPEAT - 1):
                    counters["mine"] += 1
                    p2 = os.path.join(out_dir, f"mine_{counters['mine']:05d}.jpg")
                    img.save(p2, "JPEG", quality=95)
                    rows.append([split, cls, "mine", m["path"], p2])

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "source", "original_path", "output_path"])
        w.writerows(rows)

    # 7) summary
    print("\n=== FINAL DATASET ===")
    print(f"{'class':12s}" + "".join(f"{s:>8s}" for s in SPLIT_RATIOS) + f"{'total':>8s}")
    table = Counter((r[0], r[1]) for r in rows)
    for cls in TARGET_CLASSES:
        vals = [table[(s, cls)] for s in SPLIT_RATIOS]
        print(f"{cls:12s}" + "".join(f"{v:8d}" for v in vals) + f"{sum(vals):8d}")
    tot = [sum(table[(s, c)] for c in TARGET_CLASSES) for s in SPLIT_RATIOS]
    print(f"{'TOTAL':12s}" + "".join(f"{v:8d}" for v in tot) + f"{sum(tot):8d}")
    print("\nBy source:", dict(counters))
    print(f"White borders cropped: {cropped} | corrupt skipped: {skipped}")
    if log:
        print(f"\n--- {len(log)} warnings (pehli 25) ---")
        for line in log[:25]:
            print(line)
    print(f"\nDone. Output: {OUT}")


if __name__ == "__main__":
    main()
