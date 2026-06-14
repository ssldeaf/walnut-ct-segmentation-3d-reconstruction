"""
recolor_mask.py
---------------
Перекрашивает пиксели оверлея на основе бинарной маски.

Два режима:
  - partition  : маска перегородок → пиксели под маской → цвет ЖЁЛТЫЙ (взятый с оверлея)
  - kernel     : маска ядер        → пиксели под маской → цвет КРАСНЫЙ (взятый с оверлея)

Пути и параметры можно задать прямо в блоке CONFIG ниже —
тогда запуск без аргументов: python recolor_mask.py

Или передать через консоль (аргументы имеют приоритет над CONFIG).
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image


# =============================================================================
#  CONFIG — меняйте здесь, запускайте без аргументов
# =============================================================================

CONFIG = {
    # ── Режим работы ──────────────────────────────────────────────────────────
    # "partition"  →  перекрасить в ЖЁЛТЫЙ (маски перегородок)
    # "kernel"     →  перекрасить в КРАСНЫЙ (маски ядер)
    "mode": "partition",

    # ── Степень смешивания ────────────────────────────────────────────────────
    # 0.0 = чистый целевой цвет,  1.0 = оригинал без изменений
    "blend": 0.0,

    # ── Батч-режим ────────────────────────────────────────────────────────────
    "batch": True,

    "overlay_dir": r"C:\Users\Пользователь\Desktop\319349\For_P",
    "overlay_tpl": "overlay_si{n}_K.png",       # {n} — номер без нулей

    "mask_dir":    r"C:\Users\Пользователь\Desktop\for_segm",
    "mask_tpl":    "slice_{n}_P.png",

    "output_dir":  r"C:\Users\Пользователь\Desktop\for_segm\results",
    "output_tpl":  "overlay_si{n}_P.png",        # None = совпадает с overlay_tpl

    "start": 319,
    "end":   329,

    # ── Одиночный режим (batch: False) ────────────────────────────────────────
    "overlay": "",   # путь к оверлею
    "mask":    "",   # путь к маске
    "output":  "",   # путь к результату
}

# =============================================================================


# ─── Цветовой сэмплинг ────────────────────────────────────────────────────────

def sample_color(overlay_rgb: np.ndarray, mode: str) -> np.ndarray:
    """Берёт характерный цвет прямо с оверлея."""
    r = overlay_rgb[:, :, 0].astype(int)
    g = overlay_rgb[:, :, 1].astype(int)
    b = overlay_rgb[:, :, 2].astype(int)

    if mode == "partition":
        color_mask = (r > 100) & (g > 100) & (b < 100) & (np.abs(r - g) < 60)
        label = "жёлтый"
    else:
        color_mask = (r > 150) & (g < 100) & (b < 100)
        label = "красный"

    count = int(color_mask.sum())
    if count == 0:
        raise ValueError(
            f"Не удалось найти {label} цвет на оверлее. "
            f"Попробуйте скорректировать диапазоны в sample_color()."
        )

    sampled = overlay_rgb[color_mask].mean(axis=0).astype(np.uint8)
    print(f"    Целевой цвет ({label}): RGB{tuple(int(x) for x in sampled)}  [{count} образцов]")
    return sampled


# ─── Основная функция перекраски ──────────────────────────────────────────────

def recolor(overlay_path: str, mask_path: str, mode: str,
            output_path: str, blend: float = 0.0) -> bool:
    """
    Возвращает True при успехе, False при ошибке.
    """
    if not os.path.exists(overlay_path):
        print(f"  ПРОПУСК: оверлей не найден — {overlay_path}")
        return False
    if not os.path.exists(mask_path):
        print(f"  ПРОПУСК: маска не найдена — {mask_path}")
        return False

    overlay_img = Image.open(overlay_path).convert("RGB")
    mask_img    = Image.open(mask_path).convert("L")

    overlay = np.array(overlay_img, dtype=np.uint8)
    mask    = np.array(mask_img,    dtype=np.uint8)

    binary_mask = mask > 128
    pixel_count = int(binary_mask.sum())

    print(f"    Оверлей : {overlay_path}")
    print(f"    Маска   : {mask_path}")
    print(f"    Пикселей под маской: {pixel_count}")

    if pixel_count == 0:
        print("    ПРЕДУПРЕЖДЕНИЕ: маска пустая — файл скопирован без изменений.")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        overlay_img.save(output_path)
        return True

    try:
        target_color = sample_color(overlay, mode)
    except ValueError as e:
        print(f"    ОШИБКА: {e}")
        return False

    result = overlay.copy().astype(float)
    for c in range(3):
        orig   = result[:, :, c]
        filled = np.full_like(orig, float(target_color[c]))
        result[:, :, c] = np.where(
            binary_mask,
            orig * blend + filled * (1.0 - blend),
            orig
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(result.astype(np.uint8)).save(output_path)
    print(f"    Сохранено: {output_path}")
    return True


# ─── Шаблон → имя файла ───────────────────────────────────────────────────────

def apply_template(template: str, n: int) -> str:
    """
    Подставляет номер в шаблон.
    Поддерживает {n} и {n:04d} / {n:03d} и т.п.
    """
    return template.format(n=n)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Перекраска пикселей оверлея по бинарной маске.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Все аргументы необязательны — дефолты берутся из CONFIG
    parser.add_argument("--mode", choices=["partition", "kernel"])
    parser.add_argument("--blend", type=float)
    parser.add_argument("--overlay")
    parser.add_argument("--mask")
    parser.add_argument("--output")
    parser.add_argument("--batch", action="store_true", default=None)
    parser.add_argument("--overlay-dir")
    parser.add_argument("--overlay-tpl")
    parser.add_argument("--mask-dir")
    parser.add_argument("--mask-tpl")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-tpl")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end",   type=int)

    args = parser.parse_args()

    # Аргументы консоли перекрывают CONFIG, None-значения берутся из CONFIG
    def get(cli_val, config_key):
        return cli_val if cli_val is not None else CONFIG.get(config_key)

    mode        = get(args.mode,        "mode")        or "kernel"
    blend       = get(args.blend,       "blend")       or 0.0
    batch       = args.batch or CONFIG.get("batch", False)
    overlay_dir = get(args.overlay_dir, "overlay_dir") or "."
    overlay_tpl = get(args.overlay_tpl, "overlay_tpl")
    mask_dir    = get(args.mask_dir,    "mask_dir")    or "."
    mask_tpl    = get(args.mask_tpl,    "mask_tpl")
    output_dir  = get(args.output_dir,  "output_dir")  or "output"
    output_tpl  = get(args.output_tpl,  "output_tpl")
    start       = get(args.start,       "start")
    end         = get(args.end,         "end")
    overlay     = get(args.overlay,     "overlay")
    mask        = get(args.mask,        "mask")
    output      = get(args.output,      "output")

    # ── Одиночный режим ──
    if not batch:
        if not all([overlay, mask, output]):
            parser.error(
                "Для одиночного режима заполните overlay / mask / output в CONFIG "
                "или передайте --overlay --mask --output."
            )
        ok = recolor(overlay, mask, mode, output, blend)
        sys.exit(0 if ok else 1)

    # ── Батч-режим ──
    if not overlay_tpl:
        parser.error("Укажите overlay_tpl в CONFIG или --overlay-tpl.")
    if not mask_tpl:
        parser.error("Укажите mask_tpl в CONFIG или --mask-tpl.")
    if start is None or end is None:
        parser.error("Укажите start / end в CONFIG или --start / --end.")

    out_tpl = output_tpl or overlay_tpl

    total   = 0
    success = 0
    skipped = 0

    print(f"\n{'─'*60}")
    print(f"  БАТЧ: слайсы {start}–{end}  |  режим: {mode}")
    print(f"  Оверлеи : {overlay_dir}")
    print(f"  Маски   : {mask_dir}")
    print(f"  Выход   : {output_dir}")
    print(f"{'─'*60}")

    for n in range(start, end + 1):
        total += 1
        overlay_path = os.path.join(overlay_dir, apply_template(overlay_tpl, n))
        mask_path    = os.path.join(mask_dir,    apply_template(mask_tpl, n))
        output_path  = os.path.join(output_dir,  apply_template(out_tpl, n))

        print(f"\n  [{n}]")
        ok = recolor(overlay_path, mask_path, mode, output_path, blend)
        if ok:
            success += 1
        else:
            skipped += 1

    print(f"\n{'─'*60}")
    print(f"  Готово: {success}/{total} слайсов обработано, {skipped} пропущено.")
    print(f"{'─'*60}\n")
    sys.exit(0 if skipped == 0 else 1)


if __name__ == "__main__":
    main()