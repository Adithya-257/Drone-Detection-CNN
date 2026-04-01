import os
import shutil
import random
import glob
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
VALID_SPLIT  = 0.1
TEST_SPLIT   = 0.1

OUTPUT_DIR = r"C:\Users\Adithya KL\Desktop\CAIS\Drone Detection Using CNN\Data\merged"

# Dataset paths — adjust if your folder names differ
DATASETS = {
    "drone7k": {
        "path": r"C:\Users\Adithya KL\Desktop\CAIS\Drone Detection Using CNN\Data\Drone Dataset(7k)",
        "splits": ["train"],  # already split
        "class_remap": {
            0: 0,    # drone → drone
            1: 1,    # not drone → bird
        },
        "skip_classes": [],
    },
    "drone5k": {
        "path": r"C:\Users\Adithya KL\Desktop\CAIS\Drone Detection Using CNN\Data\DroneorBird(5k)",
        "splits": ["train"],  # only train, will auto-split
        "class_remap": {
            0: None, # garbage label → skip
            1: 1,    # birds → bird
            2: 0,    # drones → drone
            3: 2,    # helicopter → plane
            4: 2,    # plane → plane
        },
        "skip_classes": [0],
    },
    "drone600": {
        "path": r"C:\Users\Adithya KL\Desktop\CAIS\Drone Detection Using CNN\Data\Drone or kite(600)",
        "splits": ["train"],  # only train, will auto-split
        "class_remap": {
            0: 1,    # bird → bird
            1: 0,    # drone → drone
            2: 3,    # kite → kite
            3: 2,    # plane → plane
        },
        "skip_classes": [],
    },
}

# Final class names
CLASS_NAMES = ["drone", "bird", "plane", "kite"]

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def remap_label_file(src_label, dst_label, class_remap, skip_classes):
    """Read a label file, remap class IDs, skip unwanted classes, write output."""
    lines_out = []

    with open(src_label, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        class_id = int(parts[0])

        # Skip garbage classes
        if class_id in skip_classes:
            continue

        # Remap class ID
        new_class_id = class_remap.get(class_id, None)
        if new_class_id is None:
            continue

        lines_out.append(f"{new_class_id} {' '.join(parts[1:])}")

    # Write even if empty (image has no valid annotations)
    os.makedirs(os.path.dirname(dst_label), exist_ok=True)
    with open(dst_label, 'w') as f:
        f.write('\n'.join(lines_out) + '\n' if lines_out else '')


def copy_image(src_image, dst_image):
    os.makedirs(os.path.dirname(dst_image), exist_ok=True)
    shutil.copy2(src_image, dst_image)


def get_image_label_pairs(dataset_path, split):
    """Return list of (image_path, label_path) pairs for a given split."""
    images_dir = os.path.join(dataset_path, split, "images")
    labels_dir = os.path.join(dataset_path, split, "labels")

    if not os.path.exists(images_dir):
        return []

    pairs = []
    for img_path in glob.glob(os.path.join(images_dir, "*")):
        ext = Path(img_path).suffix
        stem = Path(img_path).stem
        label_path = os.path.join(labels_dir, stem + ".txt")
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))

    return pairs


def split_pairs(pairs, valid_ratio, test_ratio, seed):
    """Split a list of pairs into train/valid/test."""
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_test  = int(n * test_ratio)
    n_valid = int(n * valid_ratio)

    test  = pairs[:n_test]
    valid = pairs[n_test:n_test + n_valid]
    train = pairs[n_test + n_valid:]

    return train, valid, test


def add_to_split(pairs, split_name, output_dir, dataset_name,
                 class_remap, skip_classes, counters):
    """Process and copy image+label pairs into the merged output split."""
    img_out_dir   = os.path.join(output_dir, split_name, "images")
    label_out_dir = os.path.join(output_dir, split_name, "labels")
    os.makedirs(img_out_dir,   exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    for img_path, label_path in pairs:
        stem = Path(img_path).stem
        ext  = Path(img_path).suffix

        # Prefix filename with dataset name to avoid collisions
        new_name  = f"{dataset_name}_{stem}"
        dst_image = os.path.join(img_out_dir,   new_name + ext)
        dst_label = os.path.join(label_out_dir, new_name + ".txt")

        copy_image(img_path, dst_image)
        remap_label_file(label_path, dst_label, class_remap, skip_classes)

        counters[split_name] = counters.get(split_name, 0) + 1

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Drone Detection — Dataset Merge Script")
    print("=" * 60)

    # Clean output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"\nRemoving existing output: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    counters = {}

    for dataset_name, cfg in DATASETS.items():
        print(f"\nProcessing: {dataset_name}")
        print(f"  Path: {cfg['path']}")

        class_remap  = cfg["class_remap"]
        skip_classes = cfg["skip_classes"]
        splits       = cfg["splits"]

        if len(splits) > 1:
            # Dataset already has train/valid/test splits
            for split in splits:
                pairs = get_image_label_pairs(cfg["path"], split)
                print(f"  {split}: {len(pairs)} pairs")
                add_to_split(pairs, split, OUTPUT_DIR, dataset_name,
                             class_remap, skip_classes, counters)
        else:
            # Only train split — auto split into train/valid/test
            pairs = get_image_label_pairs(cfg["path"], "train")
            print(f"  Total pairs found: {len(pairs)}")
            train, valid, test = split_pairs(
                pairs, VALID_SPLIT, TEST_SPLIT, RANDOM_SEED
            )
            print(f"  Auto-split → train:{len(train)} valid:{len(valid)} test:{len(test)}")
            for split_name, split_pairs_list in [("train", train), ("valid", valid), ("test", test)]:
                add_to_split(split_pairs_list, split_name, OUTPUT_DIR,
                             dataset_name, class_remap, skip_classes, counters)

    # Write data.yaml
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.join(OUTPUT_DIR, 'train', 'images')}\n")
        f.write(f"val:   {os.path.join(OUTPUT_DIR, 'valid', 'images')}\n")
        f.write(f"test:  {os.path.join(OUTPUT_DIR, 'test',  'images')}\n")
        f.write(f"\nnc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print("\n" + "=" * 60)
    print("Merge complete!")
    print(f"  Train : {counters.get('train', 0)} images")
    print(f"  Valid : {counters.get('valid', 0)} images")
    print(f"  Test  : {counters.get('test',  0)} images")
    print(f"  Total : {sum(counters.values())} images")
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"YAML  : {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()