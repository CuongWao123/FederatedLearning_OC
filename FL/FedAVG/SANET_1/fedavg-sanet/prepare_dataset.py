"""
Prepare CSV dataset file for ShanghaiTech crowd counting dataset.

Creates a CSV file mapping image paths to ground truth annotation paths.
Required for Flower FederatedDataset with custom partitioning.
"""

import pandas as pd
from pathlib import Path
import argparse


def create_shanghaitech_csv(
    data_root: str,
    output_csv: str,
    part: str = "A",
    split: str = "train"
):
    """
    Create CSV file mapping images to ground truth annotations.
    
    Expected ShanghaiTech structure:
    data_root/
        ShanghaiTech/
            part_A/  (or part_B)
                train_data/
                    images/
                        IMG_1.jpg
                        IMG_2.jpg
                        ...
                    ground-truth/
                        GT_IMG_1.mat
                        GT_IMG_2.mat
                        ...
                test_data/
                    images/
                    ground-truth/
    
    Args:
        data_root: Root directory containing ShanghaiTech dataset
        output_csv: Output CSV file path
        part: "A" or "B" for ShanghaiTech Part A or B
        split: "train" or "test"
    """
    
    data_root = Path(data_root)
    part_dir = data_root / "ShanghaiTech" / f"part_{part}"
    split_dir = part_dir / f"{split}_data"
    
    images_dir = split_dir / "images"
    gt_dir = split_dir / "ground-truth"
    
    print(f"\n{'='*70}")
    print(f"Creating CSV for ShanghaiTech Part {part} - {split.capitalize()}")
    print(f"{'='*70}")
    print(f"\nSearching in:")
    print(f"  Images:       {images_dir}")
    print(f"  Ground truth: {gt_dir}")
    
    # Check directories exist
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    image_files = sorted(image_files)
    
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")
    
    print(f"\nFound {len(image_files)} images")
    
    # Match ground truth files
    pairs = []
    missing_gt = []
    
    for img_path in image_files:
        # ShanghaiTech format: IMG_1.jpg → GT_IMG_1.mat
        gt_name = f"GT_{img_path.stem}.mat"
        gt_path = gt_dir / gt_name
        
        if not gt_path.exists():
            missing_gt.append(img_path.name)
            continue
        
        pairs.append({
            "image_path": str(img_path.absolute()),
            "gt_path": str(gt_path.absolute())
        })
    
    # Report missing ground truth files
    if missing_gt:
        print(f"\n⚠️  Warning: {len(missing_gt)} images missing ground truth:")
        for name in missing_gt[:5]:  # Show first 5
            print(f"    - {name}")
        if len(missing_gt) > 5:
            print(f"    ... and {len(missing_gt) - 5} more")
    
    # Create DataFrame and save
    if not pairs:
        raise ValueError("No valid image-GT pairs found!")
    
    df = pd.DataFrame(pairs)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Successfully created CSV:")
    print(f"  Output file:  {output_path}")
    print(f"  Total pairs:  {len(df)}")
    print(f"  File size:    {output_path.stat().st_size / 1024:.2f} KB")
    
    # Show sample
    print(f"\nSample entries:")
    print(df.head(3).to_string(index=False))
    print(f"{'='*70}\n")
    
    return output_path


def main():
    """Main entry point with CLI arguments."""
    
    parser = argparse.ArgumentParser(
        description="Prepare ShanghaiTech dataset CSV for Federated Learning"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets",
        help="Root directory containing ShanghaiTech folder"
    )
    parser.add_argument(
        "--part",
        type=str,
        choices=["A", "B"],
        default="A",
        help="ShanghaiTech part (A or B)"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Dataset split (train or test)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if args.output is None:
        args.output = f"shanghaitech_part{args.part}_{args.split}.csv"
    
    # Create CSV
    create_shanghaitech_csv(
        data_root=args.data_root,
        output_csv=args.output,
        part=args.part,
        split=args.split
    )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) == 1:
        # No arguments - run with defaults for debugging
        print("Running with default parameters...")
        print("Use --help for options\n")
        
        # Create CSV for Part A training data (debug version)
        try:
            create_shanghaitech_csv(
                data_root=r"d:\Research\FL_experiments",
                output_csv="shanghaitech_train_debug.csv",
                part="A",
                split="train"
            )
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nTip: Check that your dataset path is correct.")
            print("Expected structure:")
            print("  datasets/")
            print("    ShanghaiTech/")
            print("      part_A/")
            print("        train_data/")
            print("          images/")
            print("          ground-truth/")
    else:
        # Run with CLI arguments
        main()
