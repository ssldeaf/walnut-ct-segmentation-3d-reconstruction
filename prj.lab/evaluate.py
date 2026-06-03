import os
import numpy as np
from pathlib import Path
from tifffile import TiffFile
import csv

etalon_dir = Path(r"C:\Users\Пользователь\source\repos\misis2026s-23-03-lelikov-s-s\etalon_masks")
pred_dir   = Path(r"C:\Users\Пользователь\source\repos\misis2026s-23-03-lelikov-s-s\etalon_masks\segmentation_masks")
output_csv = Path(r"C:\Users\Пользователь\source\repos\misis2026s-23-03-lelikov-s-s\notes\segmentation_metrics.csv")
thresh  = 127

# Срезы, исключённые из анализа (формат: base_stem без _etalon)
EXCLUDE = {"full_AGD_50_000383_kernel"}

# Чтение маски (с учётом палитрового формата)
def read_mask(path: Path) -> np.ndarray:
    """
    Читает TIFF-маску корректно для двух форматов:
    - photometric=1 (grayscale): возвращает массив как есть
    - photometric=3 (palette/indexed): применяет цветовую таблицу
      получая реальные значения яркости (эталоны сохранены в этом формате
      с инвертированной палитрой: индекс 0 = белый, 255 = чёрный)
    """
    with TiffFile(str(path)) as tif:
        page = tif.pages[0]
        raw = page.asarray()
        cmap_tag = page.tags.get('ColorMap')
        if cmap_tag:
            cmap = np.array(cmap_tag.value)
            # Палитра TIFF: значения 0–65535, нормализуем в 0–255
            if cmap.ndim == 1:
                r = cmap[: len(cmap) // 3]
            else:
                r = cmap[0]
            return (r[raw] >> 8).astype(np.uint8)
        return raw

# метрики
def dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    intersection = np.logical_and(y_true, y_pred).sum()
    denom = y_true.sum() + y_pred.sum()
    return 1.0 if denom == 0 else 2.0 * intersection / denom

def iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return 1.0 if union == 0 else intersection / union


def find_pairs(etalon_dir: Path, pred_dir: Path):
    pairs, missing = [], []
    for etalon_path in sorted(etalon_dir.glob("*.tif")):
        stem = etalon_path.stem
        if not stem.endswith("_etalon"):
            print(f"[WARN] Пропускаю файл без суффикса _etalon: {etalon_path.name}")
            continue
        base_stem = stem[: -len("_etalon")]
        pred_path = pred_dir / (base_stem + ".tif")
        if pred_path.exists():
            pairs.append((etalon_path, pred_path, base_stem))
        else:
            missing.append(base_stem)
    return pairs, missing

def main():
    print(f"Папка эталонов : {etalon_dir}")
    print(f"Папка срезов   : {pred_dir}\n")

    pairs, missing = find_pairs(etalon_dir, pred_dir)

    if not pairs:
        print("Не найдено ни одной пары. Проверьте пути и имена файлов.")
        return

    if missing:
        print(f"Нет срезов алгоритма для {len(missing)} эталон(ов):")
        for m in missing:
            print(f"    {m}.tif")
        print()

    results = []

    # Шапка таблицы
    col = 47
    print(f"{'Срез':<{col}} {'Dice':>8} {'IoU':>8}")
    print("-" * (col + 18))

    for etalon_path, pred_path, name in pairs:
        if name in EXCLUDE:
            print(f"[SKIP] {name}")
            continue

        et = read_mask(etalon_path)
        pr = read_mask(pred_path)

        if et.shape != pr.shape:
            print(f"Размеры не совпадают для {name}: "
                  f"{et.shape} vs {pr.shape} — пропускаю.")
            continue

        et_bin = et > thresh
        pr_bin = pr > thresh

        d = dice(et_bin, pr_bin)
        j = iou(et_bin, pr_bin)

        results.append({"name": name, "dice": d, "iou": j})
        print(f"{name:<{col}} {d:>8.3f} {j:>8.3f}")

    if not results:
        print("\nНи одной пары не удалось обработать.")
        return

    # Статистика
    print("-" * (col + 18))

    for structure in ["kernel", "septa", "shell"]:
        subset = [r for r in results if r["name"].endswith(f"_{structure}")]
        if not subset:
            continue
        d_arr = np.array([r["dice"] for r in subset])
        j_arr = np.array([r["iou"]  for r in subset])
        print(f"\n{structure} ({len(subset)} срезов):")
        print(f"  Dice — mean={d_arr.mean():.3f}  std={d_arr.std():.3f}  "
              f"min={d_arr.min():.3f}  max={d_arr.max():.3f}")
        print(f"  IoU  — mean={j_arr.mean():.3f}  std={j_arr.std():.3f}  "
              f"min={j_arr.min():.3f}  max={j_arr.max():.3f}")


    d_all = np.array([r["dice"] for r in results])
    j_all = np.array([r["iou"]  for r in results])
    print(f"\nВсего ({len(results)} масок):")
    print(f"  Dice — mean={d_all.mean():.3f}  std={d_all.std():.3f}  "
          f"min={d_all.min():.3f}  max={d_all.max():.3f}")
    print(f"  IoU  — mean={j_all.mean():.3f}  std={j_all.std():.3f}  "
          f"min={j_all.min():.3f}  max={j_all.max():.3f}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "dice", "iou"])
        writer.writeheader()
        formatted_results = [
            {
                "name": r["name"],
                "dice": f"{float(r['dice']):.3f}",
                "iou": f"{float(r['iou']):.3f}"
            }
            for r in results
        ]
        writer.writerows(formatted_results)

        writer.writerow({})
        for structure in ["kernel", "septa", "shell"]:
            subset = [r for r in results if r["name"].endswith(f"_{structure}")]
            if not subset:
                continue
            d_arr = np.array([r["dice"] for r in subset])
            j_arr = np.array([r["iou"]  for r in subset])
            writer.writerow({"name": f"{structure}_mean", "dice": f"{d_arr.mean():.3f}", "iou": f"{j_arr.mean():.3f}"})
            writer.writerow({"name": f"{structure}_std",  "dice": f"{d_arr.std():.3f}",  "iou": f"{j_arr.std():.3f}"})
        writer.writerow({"name": "total_mean", "dice": f"{d_all.mean():.3f}", "iou": f"{j_all.mean():.3f}"})
        writer.writerow({"name": "total_std",  "dice": f"{d_all.std():.3f}",  "iou": f"{j_all.std():.3f}"})


if __name__ == "__main__":
    main()