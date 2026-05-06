import sys
import argparse
import cv2
import numpy as np
import os

# Добавляем путь к родительской директории для импорта из lab02
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from lab02.segment_walnut import SegmentWalnut
from lab03.overlay import Overlay


def parse_arguments():
    """Parse command line arguments for walnut segmentation and visualization."""
    parser = argparse.ArgumentParser(
        description="Segmentation and visualization of walnut image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lab03.py "C:\\Walnut_Test\\Test_Walnut.tiff"
  python lab03.py path/to/image.tiff
        """
    )
    parser.add_argument(
        "image_path",
        help="Path to the walnut image file (e.g., Test_Walnut.tiff)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Transparency level for overlay (0.0-1.0, default: 0.45)"
    )
    parser.add_argument(
        "--bright-quantile",
        type=float,
        default=99.5,
        help="Percentile used for shell thresholding (default: 99.5)"
    )
    parser.add_argument(
        "--step-fraction",
        type=float,
        default=0.03,
        help="Step fraction for adaptive shell thresholding (default: 0.03)"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Выполнение сегментации (внутри метода уже есть загрузка и нормализация)
    print("Performing segmentation...")
    segmenter = SegmentWalnut()

    img, normalized, shell_mask, kernel_mask, partitions_mask = segmenter.segment_walnut(
        args.image_path,
        bright_quantile=args.bright_quantile,
        step_fraction=args.step_fraction
    )
    
    # Создание оверлея
    print("Creating overlay visualization...")
    overlay = Overlay(
        original_img=img,
        normalized_img=normalized,
        shell_mask=shell_mask,
        kernel_mask=kernel_mask,
        partitions_mask=partitions_mask
    )
    
    # Показ панели визуализации
    print("Displaying segmentation dashboard...")
    overlay.plot_dashboard(alpha=args.alpha)


if __name__ == "__main__":
    main()
