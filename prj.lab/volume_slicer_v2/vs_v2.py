import cv2
import numpy as np
from pathlib import Path


def slice_volume(image_paths):
    out_path = Path(r"C://masks_test//output_vs")

    # Четыре среза 501x501x3 (BGR), заполнены нулями
    yz_slice_239 = np.zeros((501, 501, 3), dtype=np.uint8)
    xz_slice_239 = np.zeros((501, 501, 3), dtype=np.uint8)
    yz_slice_259 = np.zeros((501, 501, 3), dtype=np.uint8)
    xz_slice_259 = np.zeros((501, 501, 3), dtype=np.uint8)

    z = 0
    loaded = 0

    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[!] empty: {path}")
            continue

        h, w = img.shape[:2]
        if h != 501 or w != 501:
            print(f"[!] wrong size: {path} {w}x{h}")
            continue

        if loaded == 0:
            print(f"[INFO] first file: {path}")
            print(f"[INFO] channels={img.shape[2]} size={w}x{h}")

        if z >= 501:
            break  # не выходим за пределы массива

        # img.row(241) -> img[241, :, :]   (форма (501, 3))
        # img.col(241) -> img[:, 241, :]   (форма (501, 3))
        yz_slice_239[z, :, :] = img[241, :, :]
        xz_slice_239[:, z, :] = img[:, 241, :]
        yz_slice_259[z, :, :] = img[261, :, :]
        xz_slice_259[:, z, :] = img[:, 261, :]

        z += 1
        loaded += 1

    print(f"[INFO] loaded: {loaded}")

    def check(m, name):
        # cv::mean возвращает среднее по каждому каналу: B, G, R
        mean = cv2.mean(m)
        print(f"[INFO] {name}: meanB={mean[0]} meanG={mean[1]} meanR={mean[2]}")

    check(yz_slice_239, "yz_x239")
    check(xz_slice_239, "xz_y239")
    check(yz_slice_259, "yz_x259")
    check(xz_slice_259, "xz_y259")

    out_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path / "yz_x239.png"), yz_slice_239)
    cv2.imwrite(str(out_path / "xz_y239.png"), xz_slice_239)
    cv2.imwrite(str(out_path / "yz_x259.png"), yz_slice_259)
    cv2.imwrite(str(out_path / "xz_y259.png"), xz_slice_259)


def main():
    input_dir = Path(r"C:\masks_test\out")

    image_paths = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    ]

    # сортировка по имени (overlay_si049.png, overlay_si050.png, ...)
    image_paths.sort()

    print(f"[INFO] found PNG files: {len(image_paths)}")
    if image_paths:
        print(f"[INFO] first: {image_paths[0]}")
        print(f"[INFO] last:  {image_paths[-1]}")

    slice_volume(image_paths)


if __name__ == "__main__":
    main()