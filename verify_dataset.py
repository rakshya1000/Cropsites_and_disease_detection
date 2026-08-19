"""
AgroSmart Disease Detection Dataset Verification Script
=========================================================
Run this against your local dataset folder before starting CNN development.

USAGE:
    python verify_dataset.py [path_to_dataset_root]

If no path is given, it defaults to DATASET_ROOT below.

Expected folder layout:
    dataset/
        Maize/
            Common_Rust/
            Gray_Leaf_Spot/
            Northern_Leaf_Blight/
            Healthy/
        Potato/
            Early_Blight/
            Late_Blight/
            Healthy/
        Rice/
            Bacterial_Leaf_Blight/
            Brown_Spot/
            Blast/
            Healthy/
        Wheat/
            Healthy/
            Stripe_Rust/
            Septoria/
        Sugarcane/
            Healthy/
            Red_Rot/
            Red_Rust/
        Mango/
            Anthracnose/
            Bacterial_Canker/
            Cutting_Weevil/
            Die_Back/
            Gall_Midge/
            Healthy/
            Powdery_Mildew/
            Sooty_Mould/
        Banana/
            Sigatoka/
            Cordana/
            Pestalotiopsis/
            Healthy/

Checks performed:
    1. Image counts per class + summary table (printed + CSV)
    2. Empty folders (crop dirs with no class subfolders, or class dirs with 0 images)
    3. Corrupted images (fails PIL verify())
    4. Inconsistent file formats within a class folder (e.g. mixed JPEG/PNG)
    5. Duplicate/near-duplicate class folder names within the same crop
       (e.g. "Healthy" and "healthy" both existing — a filesystem case-bug)
    6. Folder structure vs. the finalized 7-crop / 29-class plan
       (missing crops, missing classes, unexpected extra classes)
"""

import os
import sys
import csv
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install it with: pip install Pillow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# CONFIG — change this if you don't pass a path on the command line
# ---------------------------------------------------------------------------
DATASET_ROOT = "dataset"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# The finalized dataset plan from our crop-by-crop review
EXPECTED_STRUCTURE = {
    "Maize": {"Common_Rust", "Gray_Leaf_Spot", "Northern_Leaf_Blight", "Healthy"},
    "Potato": {"Early_Blight", "Late_Blight", "Healthy"},
    "Rice": {"Bacterial_Leaf_Blight", "Brown_Spot", "Blast", "Healthy"},
    "Wheat": {"Healthy", "Stripe_Rust", "Septoria"},
    "Sugarcane": {"Healthy", "Red_Rot", "Red_Rust"},
    "Mango": {
        "Anthracnose", "Bacterial_Canker", "Cutting_Weevil", "Die_Back",
        "Gall_Midge", "Healthy", "Powdery_Mildew", "Sooty_Mould",
    },
    "Banana": {"Sigatoka", "Cordana", "Pestalotiopsis", "Healthy"},
}


def is_image_corrupted(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return False
    except Exception:
        return True


def get_image_format(filepath):
    try:
        with Image.open(filepath) as img:
            return img.format
    except Exception:
        return None


def scan_dataset(root):
    root_path = Path(root)
    if not root_path.exists():
        print(f"ERROR: dataset root '{root}' does not exist.")
        sys.exit(1)

    results = []
    empty_folders = []
    corrupted_files = []
    format_issues = []
    # normalized_class_name -> list of (crop, actual_folder_name)
    class_name_seen = defaultdict(list)
    found_structure = {}

    crop_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    for crop_dir in crop_dirs:
        crop_name = crop_dir.name
        class_dirs = sorted([d for d in crop_dir.iterdir() if d.is_dir()])
        found_structure[crop_name] = set(d.name for d in class_dirs)

        if not class_dirs:
            empty_folders.append(f"{crop_name} (no class subfolders found)")
            continue

        for class_dir in class_dirs:
            class_name = class_dir.name
            normalized = class_name.strip().lower()
            class_name_seen[normalized].append((crop_name, class_name))

            image_files = [f for f in class_dir.iterdir() if f.is_file()]

            if not image_files:
                empty_folders.append(f"{crop_name}/{class_name}")
                results.append({"crop": crop_name, "class": class_name, "count": 0})
                continue

            valid_count = 0
            formats_in_folder = set()

            for f in image_files:
                ext = f.suffix.lower()
                if ext not in VALID_EXTENSIONS:
                    format_issues.append(
                        f"{crop_name}/{class_name}/{f.name} — unexpected extension '{ext}'"
                    )
                    continue

                if is_image_corrupted(f):
                    corrupted_files.append(f"{crop_name}/{class_name}/{f.name}")
                    continue

                fmt = get_image_format(f)
                if fmt:
                    formats_in_folder.add(fmt)

                valid_count += 1

            if len(formats_in_folder) > 1:
                format_issues.append(
                    f"{crop_name}/{class_name} — mixed formats: {', '.join(sorted(formats_in_folder))}"
                )

            results.append({"crop": crop_name, "class": class_name, "count": valid_count})

    return results, empty_folders, corrupted_files, format_issues, class_name_seen, found_structure


def print_summary_table(results):
    print("\n" + "=" * 60)
    print(f"{'Crop':<15}{'Class':<25}{'Image Count':>15}")
    print("=" * 60)

    current_crop = None
    crop_total = 0
    grand_total = 0
    crop_totals = {}

    for row in results:
        if row["crop"] != current_crop:
            if current_crop is not None:
                print("-" * 60)
                print(f"{'':<15}{'Subtotal':<25}{crop_total:>15}")
                crop_totals[current_crop] = crop_total
            current_crop = row["crop"]
            crop_total = 0
        print(f"{row['crop']:<15}{row['class']:<25}{row['count']:>15}")
        crop_total += row["count"]
        grand_total += row["count"]

    if current_crop is not None:
        print("-" * 60)
        print(f"{'':<15}{'Subtotal':<25}{crop_total:>15}")
        crop_totals[current_crop] = crop_total

    print("=" * 60)
    print(f"{'GRAND TOTAL':<40}{grand_total:>15}")
    print("=" * 60 + "\n")

    return crop_totals, grand_total


def write_csv(results, crop_totals, grand_total, out_path="dataset_summary.csv"):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Crop", "Class", "Image Count"])
        for row in results:
            writer.writerow([row["crop"], row["class"], row["count"]])
        writer.writerow([])
        for crop, total in crop_totals.items():
            writer.writerow([crop, "TOTAL", total])
        writer.writerow([])
        writer.writerow(["GRAND TOTAL", "", grand_total])
    print(f"Summary written to: {out_path}")


def check_duplicate_class_names(class_name_seen):
    duplicates = []
    for normalized, occurrences in class_name_seen.items():
        crops_involved = set(c for c, _ in occurrences)
        # Flag only when ALL occurrences of this normalized name sit under
        # a single crop but use more than one distinct raw folder name
        # (e.g. "Healthy" and "healthy" both existing under Maize/).
        if len(crops_involved) == 1 and len(occurrences) > 1:
            names = [n for _, n in occurrences]
            if len(set(names)) > 1:
                duplicates.append((occurrences[0][0], names))
    return duplicates


def check_structure_vs_plan(found_structure):
    missing_crops = []
    extra_crops = []
    class_mismatches = []  # (crop, missing_classes, extra_classes)

    expected_crops = set(EXPECTED_STRUCTURE.keys())
    found_crops = set(found_structure.keys())

    missing_crops = sorted(expected_crops - found_crops)
    extra_crops = sorted(found_crops - expected_crops)

    for crop in sorted(expected_crops & found_crops):
        expected_classes = EXPECTED_STRUCTURE[crop]
        found_classes = found_structure[crop]
        missing = sorted(expected_classes - found_classes)
        extra = sorted(found_classes - expected_classes)
        if missing or extra:
            class_mismatches.append((crop, missing, extra))

    return missing_crops, extra_crops, class_mismatches


def print_issues(empty_folders, corrupted_files, format_issues, class_name_seen, found_structure):
    print("\n" + "#" * 60)
    print("DATASET HEALTH CHECK")
    print("#" * 60)

    print(f"\n[1] Empty folders: {len(empty_folders)}")
    for item in empty_folders:
        print(f"    - {item}")
    if not empty_folders:
        print("    - None found")

    print(f"\n[2] Corrupted images: {len(corrupted_files)}")
    for item in corrupted_files:
        print(f"    - {item}")
    if not corrupted_files:
        print("    - None found")

    print(f"\n[3] Format inconsistencies: {len(format_issues)}")
    for item in format_issues:
        print(f"    - {item}")
    if not format_issues:
        print("    - None found")

    duplicates = check_duplicate_class_names(class_name_seen)
    print(f"\n[4] Duplicate/near-duplicate class folder names: {len(duplicates)}")
    for crop, names in duplicates:
        print(f"    - Under {crop}/: {names}")
    if not duplicates:
        print("    - None found")

    missing_crops, extra_crops, class_mismatches = check_structure_vs_plan(found_structure)
    print(f"\n[5] Folder structure vs. finalized plan:")
    if missing_crops:
        print(f"    - Missing crop folders: {missing_crops}")
    if extra_crops:
        print(f"    - Unexpected extra crop folders: {extra_crops}")
    for crop, missing, extra in class_mismatches:
        if missing:
            print(f"    - {crop}: missing expected class(es): {missing}")
        if extra:
            print(f"    - {crop}: unexpected extra class(es): {extra}")
    if not missing_crops and not extra_crops and not class_mismatches:
        print("    - Matches finalized plan exactly")

    print("\n" + "#" * 60)

    total_issues = (
        len(empty_folders) + len(corrupted_files) + len(format_issues)
        + len(duplicates) + len(missing_crops) + len(extra_crops) + len(class_mismatches)
    )
    if total_issues == 0:
        print("\n✅ Dataset passed all checks. Ready to proceed to CNN implementation.\n")
    else:
        print(f"\n⚠️  {total_issues} issue(s) found. Please resolve before proceeding to CNN implementation.\n")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DATASET_ROOT
    print(f"Scanning dataset at: {root}\n")

    results, empty_folders, corrupted_files, format_issues, class_name_seen, found_structure = scan_dataset(root)
    crop_totals, grand_total = print_summary_table(results)
    write_csv(results, crop_totals, grand_total)
    print_issues(empty_folders, corrupted_files, format_issues, class_name_seen, found_structure)


if __name__ == "__main__":
    main()