"""
masks_check.py  —  анализ консистентности масок из overlay-PNG.

Читает файлы вида:
    overlay_si049.png, overlay_si050.png, ...  (из segment_walnut.py)

Цвета overlay (BGR, после addWeighted):
    shell      — зелёный   (0, 180, 0)   blend 0.50
    kernel     — красный   (0, 0,   210) add   0.45
    partitions — жёлтый    (0, 200, 200) add   0.45

Определение класса пикселя (в BGR):
    shell      : G - R > TRESH  и  G - B > TRESH  (зелёный доминирует)
    kernel     : R - G > TRESH  и  R - B > TRESH  (красный доминирует)
    partitions : G - B > TRESH  и  R - B > TRESH  и  |G - R| < SAME   (оба G и R подняты)
"""

import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Настройки ────────────────────────────────────────────────────────────────
INPUT_DIR   = Path(r"C:\masks_test\out")   # папка с overlay_siXXX.png
OUTPUT_FILE = INPUT_DIR / "consistency_plot.png"
THRESHOLD   = 0.15   # порог аномалии: изменение площади >15% между соседними срезами
SHOW        = True   # показывать окно matplotlib

# Диапазон срезов (включительно). None — без ограничения.
SI_START = 70    # первый срез (например, 40)
SI_END   = 400   # последний срез (например, 450)

# Порог «цветового подъёма» для детектирования масок из overlay
COLOR_THRESH = 40    # насколько доминирующий канал должен быть ярче остальных
SAME_THRESH  = 60    # max разница G и R для пикселей partitions (они одинаковы)
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "shell":  "#185FA5",
    "kernel": "#1D9E75",
    "septa":  "#BA7517",
}
LABELS = {
    "shell":  "Скорлупа (shell)",
    "kernel": "Ядро (kernel)",
    "septa":  "Перегородки (septa)",
}
ANOMALY_COLOR = "#E24B4A"


def extract_si(filename: str) -> int | None:
    """Извлечь номер среза из имени файла overlay_siXXX.png."""
    m = re.search(r'si(\d+)', filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


def count_pixels_from_overlay(path: Path) -> dict[str, int]:
    """
    Читает overlay PNG (BGR) и считает пиксели каждого класса по цвету.
    Возвращает {'shell': N, 'kernel': M, 'septa': K}.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Не удалось прочитать: {path}")

    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)

    # shell: зелёный канал доминирует над R и B
    shell_mask = ((g - r) > COLOR_THRESH) & ((g - b) > COLOR_THRESH)

    # kernel: красный канал доминирует над G и B
    kernel_mask = ((r - g) > COLOR_THRESH) & ((r - b) > COLOR_THRESH)

    # partitions: оба G и R подняты над B, и они близки друг к другу
    septa_mask = (
        ((g - b) > COLOR_THRESH) &
        ((r - b) > COLOR_THRESH) &
        (np.abs(g - r) < SAME_THRESH) &
        ~shell_mask &
        ~kernel_mask
    )

    return {
        "shell":  int(np.count_nonzero(shell_mask)),
        "kernel": int(np.count_nonzero(kernel_mask)),
        "septa":  int(np.count_nonzero(septa_mask)),
    }


def load_mask_data(
    input_dir: Path,
    si_start: int | None = None,
    si_end:   int | None = None,
) -> dict[str, dict[int, int]]:
    """
    Сканирует папку на overlay_si*.png, возвращает:
        {'shell': {si: px_count}, 'kernel': {...}, 'septa': {...}}

    si_start / si_end — включительные границы диапазона срезов (None = без ограничения).
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("overlay_si*.png"))
    if not files:
        files = sorted(input_dir.glob("overlay_si*.PNG"))
    if not files:
        raise FileNotFoundError(
            f"Файлы overlay_si*.png не найдены в {input_dir}\n"
            f"Убедитесь, что папка содержит результаты segment_walnut.py"
        )

    data: dict[str, dict[int, int]] = {"shell": {}, "kernel": {}, "septa": {}}
    skipped = 0

    for path in files:
        si = extract_si(path.name)
        if si is None:
            print(f"[!] не удалось извлечь номер среза из '{path.name}', пропуск")
            continue
        if si_start is not None and si < si_start:
            skipped += 1
            continue
        if si_end is not None and si > si_end:
            skipped += 1
            continue
        counts = count_pixels_from_overlay(path)
        for cls, cnt in counts.items():
            data[cls][si] = cnt

    loaded = len(data["shell"])
    rng = ""
    if si_start is not None or si_end is not None:
        a = si_start if si_start is not None else "начало"
        b = si_end   if si_end   is not None else "конец"
        rng = f"  (диапазон si: {a} … {b},  пропущено: {skipped})"
    found = {s: len(v) for s, v in data.items()}
    print(
        f"Загружено срезов: {loaded}{rng}\n"
        f"  shell={found['shell']}  kernel={found['kernel']}  septa={found['septa']}"
    )
    return data


def find_anomalies(
    values: list[float], indices: list[int], threshold: float
) -> list[int]:
    anomalies = []
    for i in range(1, len(values)):
        prev, cur = values[i - 1], values[i]
        if prev is None or cur is None or prev == 0:
            continue
        if abs(cur - prev) / prev > threshold:
            anomalies.append(indices[i])
    return anomalies


def plot_consistency(
    data: dict[str, dict[int, int]],
    threshold: float = 0.15,
    output: Path | None = None,
    show: bool = True,
) -> None:

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "Геометрическая консистентность масок сегментации",
        fontsize=14, fontweight="bold", y=0.98
    )

    for ax, mask_type in zip(axes, ("shell", "kernel", "septa")):
        mapping = data[mask_type]
        if not mapping:
            ax.set_ylabel(LABELS[mask_type], fontsize=11)
            ax.text(0.5, 0.5, "нет данных", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=12)
            continue

        indices = sorted(mapping.keys())
        values  = [mapping[i] for i in indices]
        anomaly_slices = set(find_anomalies(values, indices, threshold))
        color = COLORS[mask_type]

        ax.plot(indices, values, color=color, linewidth=1.5, zorder=2)
        ax.fill_between(indices, values, alpha=0.15, color=color)

        if anomaly_slices:
            ax_idx = [i for i in indices if i in anomaly_slices]
            ax_val = [mapping[i] for i in ax_idx]
            ax.scatter(ax_idx, ax_val, color=ANOMALY_COLOR, s=50, zorder=4,
                       label=f"аномалии ({len(anomaly_slices)})",
                       marker="o", linewidths=0)
            ax.legend(loc="upper right", fontsize=9, framealpha=0.7,
                      markerscale=1.2, handletextpad=0.4)

        ax.set_ylabel("пикселей", fontsize=10)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
        )
        ax.grid(axis="y", color="gray", alpha=0.2, linewidth=0.7)
        ax.grid(axis="x", color="gray", alpha=0.12, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(LABELS[mask_type], fontsize=11, loc="left", pad=6)

        mean_v = np.mean(values)
        std_v  = np.std(values)
        cv     = std_v / mean_v * 100 if mean_v > 0 else 0
        ax.text(0.01, 0.97, f"μ={mean_v:,.0f}  σ={std_v:,.0f}  CV={cv:.1f}%",
                transform=ax.transAxes, fontsize=9, va="top", color="gray")

    axes[-1].set_xlabel("номер среза (si)", fontsize=10)
    fig.text(
        0.5, 0.01,
        f"Порог аномалии: {threshold*100:.0f}%  |  "
        "Аномалия = изменение площади между соседними срезами превышает порог",
        ha="center", fontsize=9, color="gray"
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if output:
        fig.savefig(str(output), dpi=150, bbox_inches="tight")
        print(f"График сохранён: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    data = load_mask_data(INPUT_DIR, si_start=SI_START, si_end=SI_END)
    plot_consistency(data=data, threshold=THRESHOLD, output=OUTPUT_FILE, show=SHOW)


if __name__ == "__main__":
    main()