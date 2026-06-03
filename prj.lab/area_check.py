"""
area_check.py — контроль качества сегментации по динамике площадей масок.

Идея: площадь каждого класса (скорлупа, ядро, перегородки) меняется плавно
вдоль оси z. Резкий скачок вверх или вниз — признак ошибки сегментации
или артефакта на конкретном срезе.

Что делает скрипт:
  1. Прогоняет пайплайн по всем TIFF-срезам в INPUT_DIR.
  2. Собирает площадь каждого класса (px) для каждого среза.
  3. Строит три графика (shell / kernel / septa) с выделением аномалий.
  4. Выводит список подозрительных срезов в консоль.

Настройка:
  INPUT_DIR  — папка с TIFF-срезами
  OUTPUT_DIR — куда сохранить plot.png
  WINDOW     — размер окна скользящего медианного фильтра для baseline
  THRESHOLD  — порог аномалии: отклонение от baseline в долях (0.3 = 30%)
"""

import os, re, sys
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import deque
from pathlib import Path

# ── Настройка ─────────────────────────────────────────────────────────────────
INPUT_DIR  = r'C:\masks_test'          # ← папка с TIFF-срезами
OUTPUT_DIR = r'C:\masks_test\out'      # ← куда сохранить результат
WINDOW     = 15     # окно скользящего медиана для baseline
THRESHOLD  = 0.35   # порог аномалии: отклонение > 35% от baseline → флаг

# ── Параметры пайплайна (копия из segment_walnut.py) ──────────────────────────
TOLERANCE      = 20
HOLE_FRAC_MAX  = 0.10
KERNEL_MIN_SI  = 84
KERNEL_MAX_SI  = 383
SHELL_CLOSE_PX = 5

SEPTA_KERNEL_RATIO   = 0.80
SEPTA_THIN_Q_OFFSET  = -8.0
SEPTA_THIN_MAX_R     = 1
SEPTA_MIN_AREA       = 15
SEPTA_MAX_AREA       = 5000
SEPTA_CLOSE_PX       = 3
SEPTA_KERNEL_GUARD_R = 2
SEPTA_SHELL_GUARD_R  = 5

PARAM_ANCHORS = {
    'shell_pct':   [(49,26),(51,30),(69,55),(70,69),(79,73),(259,88),
                    (396,80),(408,74),(413,65),(420,45),(440,50)],
    'seed_pct':    [(0,50),(83,50),(84,55),(89,55),(90,65),(94,65),(95,70),
                    (96,65),(99,65),(100,85),(370,85),(371,92),(384,96),(500,99)],
    'area_frac':   [(0,0.015),(83,0.015),(84,0.02),(96,0.025),(87,0.02),
                    (130,0.025),(131,0.015),(500,0.015)],
    'solidity_min':[(0,0.60),(79,0.50),(80,0.55),(87,0.57),(100,0.59),
                    (130,0.65),(131,0.55),(500,0.55)],
    'aspect_min':  [(0,0.40),(79,0.40),(80,0.35),(87,0.50),(130,0.55),
                    (131,0.35),(210,0.35),(211,0.20),(215,0.25),(216,0.35),
                    (229,0.35),(230,0.25),(269,0.30),(270,0.35),(286,0.35),
                    (287,0.20),(305,0.16),(317,0.20),(318,0.30),(500,0.35)],
    'mask_pct':    [(0,96),(37,99),(38,98),(49,96),(50,0),(500,0)],
}

# ── Вспомогательные функции ────────────────────────────────────────────────────

def interp_param(name, si):
    anchors = sorted(PARAM_ANCHORS[name])
    if si <= anchors[0][0]:  return anchors[0][1]
    if si >= anchors[-1][0]: return anchors[-1][1]
    for (xa, pa), (xb, pb) in zip(anchors[:-1], anchors[1:]):
        if xa <= si <= xb:
            return pa + (pb - pa) * (si - xa) / (xb - xa)
    return anchors[-1][1]


def region_growing(bl, interior, seed_pct):
    H, W = bl.shape
    nz = bl[interior > 0]
    if len(nz) == 0:
        return np.zeros_like(bl)
    seed_t = int(np.percentile(nz, seed_pct))
    visited = np.zeros((H, W), dtype=bool)
    result  = np.zeros((H, W), dtype=np.uint8)
    for sy, sx in zip(*np.where((bl >= seed_t) & (interior > 0))):
        if visited[sy, sx]:
            continue
        rsum = float(bl[sy, sx]); rcnt = 1
        visited[sy, sx] = True
        q = deque([(sy, sx)]); pxs = [(sy, sx)]
        while q:
            y, x = q.popleft(); rm = rsum / rcnt
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < H and 0 <= nx < W
                        and not visited[ny, nx] and interior[ny, nx] > 0):
                    nb = float(bl[ny, nx])
                    if abs(nb - rm) <= TOLERANCE:
                        visited[ny, nx] = True
                        pxs.append((ny, nx)); rsum += nb; rcnt += 1
                        q.append((ny, nx))
        for py, px in pxs:
            result[py, px] = 255
    return cv2.bitwise_and(result, interior)


def shape_ok(comp_mask, sol_min, asp_min):
    a = float((comp_mask > 0).sum())
    cs, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return False
    cnt = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(cnt); ha = cv2.contourArea(hull)
    sol = a / ha if ha > 0 else 0
    asp = 0.0
    if len(cnt) >= 5:
        (_, (ma, Mi), _) = cv2.fitEllipse(cnt)
        if max(ma, Mi) > 0:
            asp = min(ma, Mi) / max(ma, Mi)
    cnts2, hier = cv2.findContours(comp_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hf = 0.0
    if hier is not None:
        ha2 = sum(cv2.contourArea(cnts2[ci])
                  for ci in range(len(cnts2)) if hier[0][ci][3] != -1)
        hf = ha2 / a if a > 0 else 0
    return sol >= sol_min and asp >= asp_min and hf <= HOLE_FRAC_MAX


def _otsu(pixels):
    hist, edges = np.histogram(pixels, bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0: return 128.0
    sum_total = np.dot(np.arange(256), hist)
    sum_b = w_b = max_var = 0.0; threshold = 128.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += t * hist[t]
        m_b = sum_b / w_b; m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var, threshold = var, edges[t]
    return float(threshold)


def segment_slice(img, bl, si):
    """Возвращает (shell, kernel, septa) для одного среза."""

    # ── Маска ореха ───────────────────────────────────────────────────────────
    mask_pct = interp_param('mask_pct', si)
    if mask_pct > 0:
        ke_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        bl_clean = cv2.morphologyEx(bl, cv2.MORPH_OPEN, ke_open)
        nz = bl_clean[bl_clean > 0]
        t = 255
        for try_pct in range(int(mask_pct), 50, -1):
            t_try = int(np.percentile(nz, try_pct)) if len(nz) > 0 else 0
            if t_try < 250:
                t = t_try; break
        _, bn = cv2.threshold(bl_clean, t, 255, cv2.THRESH_BINARY)
    else:
        _, bn = cv2.threshold(bl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cl = cv2.morphologyEx(bn, cv2.MORPH_CLOSE, k7)
    cs, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = bl.shape; cx_img, cy_img = W // 2, H // 2
    max_dist = min(H, W) * 0.4
    min_area = 500 if mask_pct > 0 else 5000
    valid = [c for c in cs if cv2.contourArea(c) >= min_area and
             np.sqrt((int(cv2.moments(c)['m10'] / (cv2.moments(c)['m00'] or 1)) - cx_img) ** 2 +
                     (int(cv2.moments(c)['m01'] / (cv2.moments(c)['m00'] or 1)) - cy_img) ** 2) <= max_dist]
    if not valid: valid = [c for c in cs if cv2.contourArea(c) >= min_area]
    mask = np.zeros_like(bl)
    if valid:
        cv2.drawContours(mask, [max(valid, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # ── Скорлупа ──────────────────────────────────────────────────────────────
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    me = cv2.erode(mask, k3); outer = cv2.subtract(mask, me)
    roi_e = cv2.bitwise_and(bl, bl, mask=me); nz = roi_e[me > 0]
    p_sh = int(round(interp_param('shell_pct', si)))
    t = int(np.percentile(nz, p_sh)) if len(nz) > 0 else 0
    _, bb = cv2.threshold(roi_e, t, 255, cv2.THRESH_BINARY)
    bb = cv2.bitwise_and(bb, me)
    nl, lb, st, _ = cv2.connectedComponentsWithStats(bb, connectivity=8)
    areas = sorted([(st[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)], reverse=True)
    shi = np.zeros_like(bl)
    if areas: shi[lb == areas[0][1]] = 255
    shell = cv2.bitwise_or(shi, outer)
    if si <= 69:
        k_cl = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * SHELL_CLOSE_PX + 1,) * 2)
        shell = cv2.bitwise_and(cv2.morphologyEx(shell, cv2.MORPH_CLOSE, k_cl), mask)
    cs2, _ = cv2.findContours(shell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sf = np.zeros_like(bl)
    if cs2: cv2.drawContours(sf, [max(cs2, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    interior = cv2.bitwise_and(cv2.subtract(mask, shell), sf)

    # ── Ядро ──────────────────────────────────────────────────────────────────
    kernel = np.zeros_like(bl)
    if KERNEL_MIN_SI <= si <= KERNEL_MAX_SI:
        int_area = int((interior > 0).sum())
        if int_area > 0:
            area_frac = interp_param('area_frac', si)
            min_lobe  = max(50, int(int_area * area_frac))
            seed_pct  = interp_param('seed_pct', si)
            sol_min   = interp_param('solidity_min', si)
            asp_min   = interp_param('aspect_min', si)
            kernel_raw = region_growing(bl, interior, seed_pct)
            nl, lb, st, _ = cv2.connectedComponentsWithStats(kernel_raw, connectivity=8)
            out = np.zeros_like(bl)
            for i in range(1, nl):
                if st[i, cv2.CC_STAT_AREA] < min_lobe: continue
                comp = (lb == i).astype(np.uint8) * 255
                if shape_ok(comp, sol_min, asp_min):
                    out[lb == i] = 255
            kernel = cv2.bitwise_and(out, interior)

    # ── Перегородки ───────────────────────────────────────────────────────────
    septa = np.zeros_like(bl)
    shell_exp = cv2.dilate(
        shell, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (2 * SEPTA_SHELL_GUARD_R + 1,) * 2))
    roi_int = cv2.bitwise_and(interior, cv2.bitwise_not(shell_exp))
    valid_pix = bl[roi_int > 0]
    if len(valid_pix) > 0:
        thresh_k = _otsu(valid_pix)
        thresh_s = thresh_k * SEPTA_KERNEL_RATIO
        thresh_thin = max(0.0, thresh_s + SEPTA_THIN_Q_OFFSET)
        sc = np.zeros_like(bl); sc[(roi_int > 0) & (bl >= thresh_s)] = 255
        tb = np.zeros_like(bl); tb[(roi_int > 0) & (bl >= thresh_thin)] = 255
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SEPTA_THIN_MAX_R + 1,) * 2)
        thin = cv2.subtract(tb, cv2.morphologyEx(tb, cv2.MORPH_OPEN, se))
        sa = cv2.bitwise_or(sc, thin)
        septa = cv2.bitwise_and(sa, cv2.bitwise_not(kernel))
        septa = cv2.bitwise_and(septa, roi_int)
        nl, lb2, st2, _ = cv2.connectedComponentsWithStats(septa, connectivity=8)
        clean = np.zeros_like(bl)
        for i in range(1, nl):
            a = st2[i, cv2.CC_STAT_AREA]
            if SEPTA_MIN_AREA <= a <= SEPTA_MAX_AREA:
                clean[lb2 == i] = 255
        septa = clean
        ke_cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SEPTA_CLOSE_PX + 1,) * 2)
        septa = cv2.morphologyEx(septa, cv2.MORPH_CLOSE, ke_cl)
        kg = cv2.dilate(kernel, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * SEPTA_KERNEL_GUARD_R + 1,) * 2))
        sg = cv2.dilate(shell, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * SEPTA_SHELL_GUARD_R + 1,) * 2))
        septa = cv2.bitwise_and(septa, cv2.bitwise_not(kg))
        septa = cv2.bitwise_and(septa, cv2.bitwise_not(sg))

    return shell, kernel, septa


# ── Детектор аномалий ─────────────────────────────────────────────────────────

def detect_anomalies(si_list, areas, window=WINDOW, threshold=THRESHOLD):
    """
    Сравниваем каждое значение с медианой в окне соседних срезов.
    Отклонение > threshold от медианы → аномалия.
    Возвращает список (si, area, baseline, отклонение).
    """
    areas = np.array(areas, dtype=float)
    n = len(areas)
    anomalies = []
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        # медиана из окна, исключая сам элемент
        neighbors = np.concatenate([areas[lo:i], areas[i+1:hi]])
        if len(neighbors) == 0:
            continue
        baseline = np.median(neighbors)
        if baseline == 0:
            continue
        deviation = abs(areas[i] - baseline) / baseline
        if deviation > threshold:
            anomalies.append((si_list[i], int(areas[i]), int(baseline), round(deviation, 2)))
    return anomalies


# ── Построение графика ────────────────────────────────────────────────────────

def plot_areas(records, anomalies_dict, out_path):
    si_vals    = [r[0] for r in records]
    shell_a    = [r[1] for r in records]
    kernel_a   = [r[2] for r in records]
    septa_a    = [r[3] for r in records]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Динамика площадей масок по срезам', fontsize=14, fontweight='bold')

    datasets = [
        (axes[0], shell_a,  anomalies_dict['shell'],  'Скорлупа (shell)',       '#2196F3'),
        (axes[1], kernel_a, anomalies_dict['kernel'], 'Ядро (kernel)',           '#F44336'),
        (axes[2], septa_a,  anomalies_dict['septa'],  'Перегородки (septa)',     '#FF9800'),
    ]

    for ax, areas, anomalies, label, color in datasets:
        ax.plot(si_vals, areas, color=color, linewidth=1.2, label=label)

        # Скользящий медиан как baseline
        areas_arr = np.array(areas, dtype=float)
        from scipy.ndimage import median_filter
        try:
            baseline = median_filter(areas_arr, size=WINDOW, mode='nearest')
            ax.plot(si_vals, baseline, color='gray', linewidth=0.8,
                    linestyle='--', alpha=0.7, label='медианный baseline')
        except ImportError:
            pass

        # Отмечаем аномалии красными точками
        anom_si = [a[0] for a in anomalies]
        anom_val = [a[1] for a in anomalies]
        if anom_si:
            ax.scatter(anom_si, anom_val, color='red', zorder=5,
                       s=40, label=f'аномалии ({len(anom_si)} шт.)')

        ax.set_ylabel('Площадь, px', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{int(x):,}'.replace(',', ' ')))

    axes[2].set_xlabel('Номер среза (si)', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[OK] График сохранён: {out_path}')


# ── Главный цикл ──────────────────────────────────────────────────────────────

def main():
    input_path  = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob('*.tiff')) + sorted(input_path.glob('*.tif'))
    if not files:
        print(f'[ERROR] TIFF-файлы не найдены в {INPUT_DIR}')
        sys.exit(1)

    print(f'Найдено срезов: {len(files)}')

    # Глобальные перцентили для нормализации
    print('Вычисление глобальных перцентилей...')
    all_vals = []
    for f in files:
        raw = tifffile.imread(str(f)).astype(np.float32)
        all_vals.append(raw.ravel())
    all_vals = np.concatenate(all_vals)
    p1, p99 = np.percentile(all_vals, [1, 99])
    print(f'  p1={p1:.5f}  p99={p99:.5f}')

    records = []   # (si, shell_px, kernel_px, septa_px)
    errors  = 0

    for i, fpath in enumerate(files, 1):
        m = re.search(r'(\d+)(?=\.tiff?$)', fpath.name, re.I)
        si = int(m.group(1)) if m else i

        try:
            raw = tifffile.imread(str(fpath)).astype(np.float32)
            raw_c = np.clip(raw, p1, p99)
            img = cv2.normalize(raw_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            bl  = cv2.medianBlur(img, 5)

            shell, kernel, septa = segment_slice(img, bl, si)

            shell_px  = int((shell  > 0).sum())
            kernel_px = int((kernel > 0).sum())
            septa_px  = int((septa  > 0).sum())
            records.append((si, shell_px, kernel_px, septa_px))

        except Exception as e:
            print(f'  [!] Ошибка на {fpath.name}: {e}')
            errors += 1
            continue

        if i % 20 == 0 or i == len(files):
            print(f'  [{i}/{len(files)}] si={si:4d}  '
                  f'shell={shell_px:6,}  kernel={kernel_px:6,}  septa={septa_px:5,}')

    # Сортируем по si на случай несортированных файлов
    records.sort(key=lambda r: r[0])

    si_list   = [r[0] for r in records]
    shell_a   = [r[1] for r in records]
    kernel_a  = [r[2] for r in records]
    septa_a   = [r[3] for r in records]

    # ── Детекция аномалий ─────────────────────────────────────────────────────
    anom_shell  = detect_anomalies(si_list, shell_a)
    anom_kernel = detect_anomalies(si_list, kernel_a)
    anom_septa  = detect_anomalies(si_list, septa_a)

    print('\n═══ АНОМАЛЬНЫЕ СРЕЗЫ ════════════════════════════════')
    for label, anomalies in [('Shell', anom_shell), ('Kernel', anom_kernel), ('Septa', anom_septa)]:
        if anomalies:
            print(f'\n  {label}:')
            for si, area, baseline, dev in anomalies:
                direction = '▲' if area > baseline else '▼'
                print(f'    si={si:4d}  площадь={area:7,}  baseline≈{baseline:7,}  '
                      f'отклонение={dev*100:.0f}% {direction}')
        else:
            print(f'\n  {label}: аномалий не обнаружено')

    print(f'\nВсего срезов обработано: {len(records)}, ошибок: {errors}')

    # ── График ────────────────────────────────────────────────────────────────
    anomalies_dict = {'shell': anom_shell, 'kernel': anom_kernel, 'septa': anom_septa}
    plot_path = str(output_path / 'area_dynamics.png')
    plot_areas(records, anomalies_dict, plot_path)

    # ── CSV для дополнительного анализа ──────────────────────────────────────
    csv_path = str(output_path / 'area_dynamics.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('si,shell_px,kernel_px,septa_px\n')
        for r in records:
            f.write(f'{r[0]},{r[1]},{r[2]},{r[3]}\n')
    print(f'[OK] CSV сохранён: {csv_path}')


if __name__ == '__main__':
    main()