import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from segment_walnut import SegmentWalnut

def parse_arguments():
    """Parse command line arguments for walnut segmentation."""
    parser = argparse.ArgumentParser(
        description="Segmentation of walnut image into shell, kernel, and partitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lab02.py "C:\\Users\\Пользователь\\Desktop\\Walnut_Test\\Test_Walnut.tiff"
  python lab02.py path/to/image.tiff
        """
    )
    parser.add_argument(
        "image_path",
        help="Path to the walnut image file (e.g., Test_Walnut.tiff)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # 1. Создаём экземпляр класса
    segmenter = SegmentWalnut()
    
    # 2. Вызываем метод у экземпляра
    segmenter.segment_walnut(
        image_path=args.image_path,
        bright_quantile=95.0,  # Было 99.5 → опускаем планку, чтобы захватить менее яркие участки скорлупы
        step_fraction=0.10     # Было 0.03 → увеличиваем отступ, порог станет ниже
    )
