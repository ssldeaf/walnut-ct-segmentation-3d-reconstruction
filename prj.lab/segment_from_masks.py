"""
Walnut X-ray Segmentation from Pre-built Masks
===============================================
Читает готовые маски из masks_dir (shell / kernel / septa),
накладывает overlay на исходные срезы из images_dir.

Цветовая палитра и прозрачность идентичны segment_walnut_v2.py:
  shell      → зелёный  (0, 180, 0)   α = 0.50
  kernel     → красный  (0, 0, 210)   α = 0.45
  partitions → жёлтый   (0, 200, 200) α = 0.45

Маски хранятся в инвертированном формате: 0 = сегмент, 255 = фон.
"""

import os
import re
import cv2
import numpy as np
import tifffile


# ── Цветовая палитра (BGR, как в segment_walnut_v2.py) ────────────────────
COLOR_SHELL      = (0, 180,   0)   # зелёный
COLOR_KERNEL     = (0,   0, 210)   # красный
COLOR_PARTITIONS = (0, 200, 200)   # жёлтый
ALPHA_SHELL      = 0.50
ALPHA_KERNEL     = 0.45
ALPHA_PARTITIONS = 0.45


# ── Загрузка среза ─────────────────────────────────────────────────────────

def load_slice(path: str) -> np.ndarray:
    """Загружает TIFF-срез и нормализует в uint8 [0..255]."""
    raw = tifffile.imread(path).astype(np.float32)
    p_low  = np.percentile(raw, 1)
    p_high = np.percentile(raw, 99)
    raw = np.clip(raw, p_low, p_high)
    img = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


# ── Загрузка маски (инвертированный формат) ────────────────────────────────

def load_mask(path: str) -> np.ndarray:
    """
    Загружает маску-эталон: пиксели == 0 → сегмент (возвращаем 255),
    пиксели == 255 → фон (возвращаем 0).
    """
    if not os.path.exists(path):
        return None
    raw = tifffile.imread(path)
    # инвертируем: 0 → сегмент → 255
    mask = np.where(raw == 0, np.uint8(255), np.uint8(0))
    return mask


# ── Overlay (идентично make_overlay из segment_walnut_v2.py) ───────────────

def make_overlay(img_gray: np.ndarray,
                 shell: np.ndarray,
                 kernel: np.ndarray,
                 partitions: np.ndarray) -> np.ndarray:
    """
    Строит цветной overlay поверх серого среза.
    Порядок и веса точно совпадают с segment_walnut_v2.py::make_overlay().
    """
    ov = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    gl = np.zeros_like(ov)
    rl = np.zeros_like(ov)
    yl = np.zeros_like(ov)

    if shell is not None:
        gl[shell > 0] = COLOR_SHELL
    if kernel is not None:
        rl[kernel > 0] = COLOR_KERNEL
    if partitions is not None:
        yl[partitions > 0] = COLOR_PARTITIONS

    ov = cv2.addWeighted(ov, 1.0 - ALPHA_SHELL,      gl, ALPHA_SHELL,      0)
    ov = cv2.addWeighted(ov, 1.0,                     rl, ALPHA_KERNEL,     0)
    ov = cv2.addWeighted(ov, 1.0,                     yl, ALPHA_PARTITIONS, 0)
    return ov


# ── Извлечение si из имени файла ───────────────────────────────────────────

def si_from_name(fname: str) -> int:
    m = re.search(r'(\d+)(?=\.tiff?$)', fname, re.IGNORECASE)
    return int(m.group(1)) if m else -1


# ── Основная функция ───────────────────────────────────────────────────────

def process_batch(images_dir: str,
                  masks_dir: str,
                  output_dir: str = 'out_masks') -> None:
    """
    Для каждого TIFF-среза в images_dir ищет соответствующие маски в masks_dir
    (по номеру среза), строит overlay и сохраняет в output_dir.

    Ожидаемые имена масок:
        *_<si>_shell_etalon.tif
        *_<si>_kernel_etalon.tif
        *_<si>_septa_etalon.tif   (перегородки)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Собираем маски в словарь: si → {shell, kernel, septa}
    mask_index: dict[int, dict[str, str]] = {}
    for fname in os.listdir(masks_dir):
        low = fname.lower()
        m = re.search(r'(\d{6})', fname)
        if not m:
            continue
        si = int(m.group(1))
        if si not in mask_index:
            mask_index[si] = {}
        if 'shell' in low:
            mask_index[si]['shell'] = os.path.join(masks_dir, fname)
        elif 'kernel' in low:
            mask_index[si]['kernel'] = os.path.join(masks_dir, fname)
        elif 'septa' in low:
            mask_index[si]['septa'] = os.path.join(masks_dir, fname)

    # Обрабатываем срезы
    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith('.tiff') or f.lower().endswith('.tif')
    )
    print(f'[INFO] Срезов найдено: {len(image_files)}')

    matched = 0
    for fname in image_files:
        si = si_from_name(fname)
        img_path = os.path.join(images_dir, fname)

        try:
            img = load_slice(img_path)
        except Exception as e:
            print(f'  [!] Ошибка загрузки {fname}: {e}')
            continue

        segs = mask_index.get(si, {})
        shell      = load_mask(segs['shell'])      if 'shell'  in segs else None
        kernel     = load_mask(segs['kernel'])     if 'kernel' in segs else None
        partitions = load_mask(segs['septa'])      if 'septa'  in segs else None

        if not segs:
            print(f'  [~] si={si:03d}: маски не найдены, сохраняем без overlay')
        else:
            matched += 1

        ov = make_overlay(img, shell, kernel, partitions)

        # si-метка на изображении
        cv2.putText(ov, f'si={si}', (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(output_dir, f'overlay_si{si:03d}.png')
        cv2.imwrite(out_path, ov)
        print(f'  [OK] overlay_si{si:03d}.png  (shell={shell is not None}, '
              f'kernel={kernel is not None}, septa={partitions is not None})')

    print(f'\nГотово. Результаты: {output_dir}')
    print(f'Срезов обработано: {len(image_files)}, из них с масками: {matched}')


# ── Запуск ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    # ── Пути — измените под свою структуру папок ──────────────────────────
    IMAGES_DIR = r'C:\m_result_overlays\tiff_images'          # папка с .tiff срезами
    MASKS_DIR  = r'C:\m_result_overlays\masks_for_build'     # папка с масками-эталонами
    OUTPUT_DIR = r'C:\m_result_overlays'     # куда сохранять результаты

    # Поддержка аргументов командной строки:
    #   python segment_from_masks.py <images_dir> <masks_dir> [output_dir]
    if len(sys.argv) >= 3:
        IMAGES_DIR = sys.argv[1]
        MASKS_DIR  = sys.argv[2]
        if len(sys.argv) >= 4:
            OUTPUT_DIR = sys.argv[3]

    print(f'[INFO] images_dir : {IMAGES_DIR}')
    print(f'[INFO] masks_dir  : {MASKS_DIR}')
    print(f'[INFO] output_dir : {OUTPUT_DIR}')

    process_batch(IMAGES_DIR, MASKS_DIR, OUTPUT_DIR)
