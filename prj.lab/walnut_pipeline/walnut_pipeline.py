"""
Walnut X-ray CT Slice Segmentation Pipeline v3
Stages: A (walnut mask) → C (shell) → D (kernel) → E (partitions)
"""

import cv2
import numpy as np
import tifffile
import re
from collections import deque
from skimage.filters import frangi

# ── Параметры ──────────────────────────────────────────────────────────────

# ── si-зависимые параметры (линейная интерполяция между анкорами) ──────────
# Формат: (si, value). Вне диапазона анкоров — берётся ближайшее значение.
PARAM_ANCHORS = {
    # Скорлупа: процентиль яркости для порога
    'shell_pct':        [(49, 26),   (79, 73),   (259, 88)],

    # Ядро: процентиль яркости для seeds
    'seed_pct':         [(0, 92),    (69, 92),   (70, 85),   (500, 85)],

    # Ядро: фильтр формы
    'solidity_min':     [(0, 0.60),  (79, 0.60), (80, 0.55), (500, 0.55)],
    'aspect_min':       [(0, 0.40),  (79, 0.40), (80, 0.35), (500, 0.35)],

    # Перегородки: пороги
    'frangi_pct':       [(49, 85),   (259, 85)],
    'ring_bright_pct':  [(49, 70),   (259, 70)],
    'bright_filter_pct':[(49, 50),   (259, 50)],
}

# ── si-независимые параметры ───────────────────────────────────────────────
# Kernel
TOLERANCE      = 20    # region growing tolerance
DT_PCT         = 80    # distance transform percentile (si <= 69)
AREA_FRAC      = 0.005 # min lobe area fraction
EXPAND_TOL     = 25    # BFS expand tolerance (si <= 69)
SHELL_CLOSE_PX = 5     # morph close для si == 69
HOLE_FRAC_MAX  = 0.10
KERNEL_MIN_SI  = 50    # срезы ниже этого значения не имеют ядра (si <= 49)

# Partitions (стадия E) — структурные параметры (не зависят от si)
FRANGI_INTERIOR_ERODE = 3
FRANGI_KERNEL_DILATE  = 2
FRANGI_SIGMAS         = range(1, 4)
RING_PX               = 8
KERNEL_EXPAND_PX      = 2

# Shell zone (перегородки в области скорлупы)
SHELL_ZONE_DILATE_PX  = 3
SHELL_ZONE_BRIGHT_T   = 50
SHELL_ZONE_MIN_AREA   = 100
SHELL_ZONE_MAX_AREA   = 1000


# ── Вспомогательные функции ────────────────────────────────────────────────

def interp_param(name, si):
    """Линейная интерполяция значения параметра по анкорам для среза si."""
    anchors = sorted(PARAM_ANCHORS[name])
    if si <= anchors[0][0]:
        return anchors[0][1]
    if si >= anchors[-1][0]:
        return anchors[-1][1]
    for (xa, pa), (xb, pb) in zip(anchors[:-1], anchors[1:]):
        if xa <= si <= xb:
            return pa + (pb - pa) * (si - xa) / (xb - xa)
    return anchors[-1][1]


def load_slice(path):
    raw = tifffile.imread(path).astype(np.float32)
    img = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bl  = cv2.medianBlur(img, 5)
    fname = path.split('/')[-1]
    m = re.search(r'(\d+)(?=\.tiff?$)', fname, re.IGNORECASE)
    si = int(m.group(1)) if m else 0
    return img, bl, si


# ── Стадия A: маска ореха ──────────────────────────────────────────────────

def build_walnut_mask(bl, save=None):
    _, bn = cv2.threshold(bl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if save: save('threshold_otsu', bn)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cl = cv2.morphologyEx(bn, cv2.MORPH_CLOSE, k7)
    if save: save('morphologyEx_close', cl)
    cs, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cs if cv2.contourArea(c) >= 5000]
    mask = np.zeros_like(bl)
    if valid:
        cv2.drawContours(mask, [max(valid, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    if save: save('drawContours_largest', mask)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if save: save('dilate', mask)
    return mask


# ── Стадия C: маска скорлупы ───────────────────────────────────────────────

def build_shell(bl, mask, si, save=None):
    p = int(round(interp_param('shell_pct', si)))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    me = cv2.erode(mask, k3)
    if save: save('erode_mask', me)
    outer = cv2.subtract(mask, me)
    if save: save('subtract_outer', outer)
    roi_e = cv2.bitwise_and(bl, bl, mask=me)
    if save: save('bitwise_and_roi', roi_e)
    nz = roi_e[me > 0]
    t = int(np.percentile(nz, p)) if len(nz) > 0 else 0
    _, bb = cv2.threshold(roi_e, t, 255, cv2.THRESH_BINARY)
    if save: save(f'threshold_p{p}_t{t}', bb)
    bb = cv2.bitwise_and(bb, bb, mask=me)
    nl, lb, st, _ = cv2.connectedComponentsWithStats(bb, connectivity=8)
    areas = sorted([(st[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)], reverse=True)
    if not areas:
        return np.zeros_like(bl), np.zeros_like(bl)

    max_area = areas[0][0]
    shi = np.zeros_like(bl)

    if si > 69:
        shi[lb == areas[0][1]] = 255
        shell = cv2.bitwise_or(shi, outer)

    elif si == 69:
        for a, idx in areas:
            if a >= max_area * 0.9:
                shi[lb == idx] = 255
        shell = shi

    else:  # si < 69
        candidates = {}
        for a, idx in areas:
            mb = float(bl[lb == idx].mean())
            if a >= max_area * 0.05 or mb >= 135:
                candidates[idx] = a
        if len(candidates) > 1:
            filled = {}
            for idx in candidates:
                comp = (lb == idx).astype(np.uint8) * 255
                cs2, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                f = np.zeros_like(bl)
                if cs2:
                    cv2.drawContours(f, [max(cs2, key=cv2.contourArea)], -1, 255, cv2.FILLED)
                filled[idx] = f
            to_remove = set()
            for idx_a in list(candidates.keys()):
                for idx_b in list(candidates.keys()):
                    if idx_a == idx_b or idx_b in to_remove:
                        continue
                    comp_a = (lb == idx_a).astype(np.uint8) * 255
                    if (cv2.bitwise_and(comp_a, filled[idx_b]) > 0).sum() == (comp_a > 0).sum():
                        to_remove.add(idx_a)
            for idx in to_remove:
                del candidates[idx]
        for idx in candidates:
            shi[lb == idx] = 255
        shell = shi

    if si <= 69:
        k_cl = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * SHELL_CLOSE_PX + 1, 2 * SHELL_CLOSE_PX + 1))
        shell = cv2.bitwise_and(cv2.morphologyEx(shell, cv2.MORPH_CLOSE, k_cl), mask)
        if save: save('morphologyEx_close_shell', shell)

    if save: save('shell_result', shell)
    cs2, _ = cv2.findContours(shell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sf = np.zeros_like(bl)
    if cs2:
        cv2.drawContours(sf, [max(cs2, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    if save: save('drawContours_shell_filled', sf)
    interior = cv2.bitwise_and(cv2.subtract(mask, shell), sf)
    if save: save('interior_result', interior)
    return shell, interior


# ── Стадия D: маска ядра ───────────────────────────────────────────────────

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
                if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and interior[ny, nx] > 0:
                    nb = float(bl[ny, nx])
                    if abs(nb - rm) <= TOLERANCE:
                        visited[ny, nx] = True
                        pxs.append((ny, nx)); rsum += nb; rcnt += 1; q.append((ny, nx))
        for py, px in pxs:
            result[py, px] = 255
    return cv2.bitwise_and(result, interior)


def rg_from_seeds(bl, seeds, interior):
    H, W = bl.shape
    visited = seeds.astype(bool).copy()
    result  = seeds.copy()
    cv_ = bl[seeds > 0]
    if len(cv_) == 0:
        return result
    gm = float(cv_.mean())
    queue = deque(zip(*np.where(seeds > 0)))
    while queue:
        y, x = queue.popleft()
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and interior[ny, nx] > 0:
                if abs(float(bl[ny, nx]) - gm) <= EXPAND_TOL:
                    visited[ny, nx] = True; result[ny, nx] = 255; queue.append((ny, nx))
    return cv2.bitwise_and(result, interior)


def shape_ok(comp_mask, sol_min, asp_min):
    a = float((comp_mask > 0).sum())
    cs, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return False
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
        ha2 = sum(cv2.contourArea(cnts2[ci]) for ci in range(len(cnts2)) if hier[0][ci][3] != -1)
        hf = ha2 / a if a > 0 else 0
    return sol >= sol_min and asp >= asp_min and hf <= HOLE_FRAC_MAX


def build_kernel(bl, interior, si, save=None):
    if si < KERNEL_MIN_SI:  # si <= 49 — нет ядра
        return np.zeros_like(bl)
    int_area = int((interior > 0).sum())
    if int_area == 0:
        return np.zeros_like(bl)
    min_lobe = max(50, int(int_area * AREA_FRAC))

    seed_pct = interp_param('seed_pct', si)
    sol_min = interp_param('solidity_min', si)
    asp_min = interp_param('aspect_min', si)

    if si <= 69:
        kernel_raw = region_growing(bl, interior, seed_pct)
        if save: save('region_growing', kernel_raw)
        dist = cv2.distanceTransform(kernel_raw, cv2.DIST_L2, 5)
        dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if save: save('distanceTransform', dist_vis)
        nz_d = dist[kernel_raw > 0]
        dt_t = np.percentile(nz_d, DT_PCT) if len(nz_d) > 0 else 0
        kernel_dt = cv2.bitwise_and((dist >= dt_t).astype(np.uint8) * 255, interior)
        if save: save(f'threshold_dt_p{DT_PCT}', kernel_dt)
        nl, lb, st, _ = cv2.connectedComponentsWithStats(kernel_dt, connectivity=8)
        centers = np.zeros_like(bl)
        for i in range(1, nl):
            if st[i, cv2.CC_STAT_AREA] >= min_lobe:
                centers[lb == i] = 255
        if save: save('centers_min_lobe', centers)
        result = rg_from_seeds(bl, centers, interior)
        if save: save('rg_from_seeds', result)
        return result

    else:
        kernel_raw = region_growing(bl, interior, seed_pct)
        if save: save('region_growing', kernel_raw)
        nl, lb, st, _ = cv2.connectedComponentsWithStats(kernel_raw, connectivity=8)
        out = np.zeros_like(bl)
        for i in range(1, nl):
            if st[i, cv2.CC_STAT_AREA] < min_lobe:
                continue
            comp = (lb == i).astype(np.uint8) * 255
            if shape_ok(comp, sol_min, asp_min):
                out[lb == i] = 255
        if save: save('shape_filter', out)
        result = cv2.bitwise_and(out, interior)
        if save: save('kernel_result', result)
        return result


# ── Стадия E: маска перегородок ────────────────────────────────────────────

def build_partitions(bl, interior, kernel, shell=None, mask=None, si=259, save=None):
    if shell is None: shell = np.zeros_like(bl)
    if mask is None: mask = interior.copy()
    frangi_pct = interp_param('frangi_pct', si)
    ring_bright_pct = interp_param('ring_bright_pct', si)
    bright_filter_pct = interp_param('bright_filter_pct', si)
    # — Frangi для центральных перегородок —
    ke3 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * FRANGI_INTERIOR_ERODE + 1, 2 * FRANGI_INTERIOR_ERODE + 1))
    interior_eroded = cv2.erode(interior, ke3)
    if save: save('erode_interior', interior_eroded)

    ke2 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * FRANGI_KERNEL_DILATE + 1, 2 * FRANGI_KERNEL_DILATE + 1))
    kernel_d = cv2.dilate(kernel, ke2)
    if save: save('dilate_kernel', kernel_d)

    work = cv2.bitwise_and(bl, bl, mask=interior_eroded).astype(np.float32) / 255.0
    fo   = frangi(work, sigmas=FRANGI_SIGMAS, black_ridges=False)
    fo[interior_eroded == 0] = 0
    fn = cv2.normalize(fo, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if save: save('frangi_response', fn)
    ft = int(np.percentile(fn[interior_eroded > 0], frangi_pct))
    _, fb = cv2.threshold(fn, ft, 255, cv2.THRESH_BINARY)
    if save: save(f'threshold_frangi_p{frangi_pct}', fb)
    fb = cv2.bitwise_and(fb, interior_eroded)
    frangi_part = cv2.bitwise_and(fb, cv2.bitwise_not(kernel_d))
    frangi_part = cv2.bitwise_and(frangi_part, interior)
    if save: save('frangi_part', frangi_part)

    # — Яркостный порог в кольце вокруг ядра —
    ke_r = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * RING_PX + 1, 2 * RING_PX + 1))
    kernel_outer = cv2.dilate(kernel, ke_r)
    ring_zone    = cv2.bitwise_and(cv2.subtract(kernel_outer, kernel), interior)
    if save: save('ring_zone', ring_zone)

    ring_pixels = cv2.bitwise_and(bl, bl, mask=ring_zone)
    nz_ring     = ring_pixels[ring_zone > 0]
    bt = int(np.percentile(nz_ring, ring_bright_pct)) if len(nz_ring) > 0 else 0
    _, bright_ring_raw = cv2.threshold(ring_pixels, bt, 255, cv2.THRESH_BINARY)
    bright_ring_raw = cv2.bitwise_and(bright_ring_raw, ring_zone)
    if save: save(f'threshold_ring_p{ring_bright_pct}', bright_ring_raw)

    # — Фильтр мост: только компоненты касающиеся ядра —
    kernel_touch = cv2.dilate(kernel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    nl_r, lb_r, _, _ = cv2.connectedComponentsWithStats(bright_ring_raw, connectivity=8)
    bridge_mask = np.zeros_like(bl)
    for i in range(1, nl_r):
        comp = (lb_r == i).astype(np.uint8) * 255
        if (cv2.bitwise_and(comp, kernel_touch) > 0).sum() > 0:
            bridge_mask[lb_r == i] = 255
    if save: save('bridge_mask', bridge_mask)

    # — Эрозия 1px убирает тонкий контур ядра —
    ke_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bridge_eroded = cv2.erode(bridge_mask, ke_e)
    bridge_eroded = cv2.bitwise_and(bridge_eroded, ring_zone)
    if save: save('erode_bridge', bridge_eroded)

    # — Расширяем ядро на KERNEL_EXPAND_PX перед p-фильтром —
    ke_exp = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * KERNEL_EXPAND_PX + 1, 2 * KERNEL_EXPAND_PX + 1))
    kernel_expanded = cv2.dilate(kernel, ke_exp)
    bridge_eroded   = cv2.bitwise_and(bridge_eroded, cv2.bitwise_not(kernel_expanded))
    bridge_eroded   = cv2.bitwise_and(bridge_eroded, ring_zone)

    # — Для каждой компоненты оставляем только пиксели ярче p(bright_filter_pct) —
    nl_b, lb_b, _, _ = cv2.connectedComponentsWithStats(bridge_eroded, connectivity=8)
    bridge_bright = np.zeros_like(bl)
    for i in range(1, nl_b):
        comp_mask = (lb_b == i).astype(np.uint8) * 255
        vals = bl[comp_mask > 0]
        if len(vals) == 0:
            continue
        thresh = np.percentile(vals, bright_filter_pct)
        bridge_bright[(bl > thresh) & (comp_mask > 0)] = 255
    if save: save(f'bridge_bright_p{bright_filter_pct}', bridge_bright)

    # — Shell zone: перегородки в области скорлупы —
    ke_sd = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*SHELL_ZONE_DILATE_PX+1, 2*SHELL_ZONE_DILATE_PX+1))
    shell_d = cv2.dilate(shell, ke_sd)
    shell_zone = cv2.bitwise_and(cv2.subtract(mask, interior), cv2.subtract(mask, kernel))
    shell_zone = cv2.bitwise_and(shell_zone, mask)
    shell_zone_clean = cv2.bitwise_and(shell_zone, cv2.bitwise_not(shell_d))
    if save: save('shell_zone_clean', shell_zone_clean)
    sz_pixels = cv2.bitwise_and(bl, bl, mask=shell_zone_clean)
    _, sz_bright = cv2.threshold(sz_pixels, SHELL_ZONE_BRIGHT_T, 255, cv2.THRESH_BINARY)
    sz_bright = cv2.bitwise_and(sz_bright, shell_zone_clean)
    if save: save(f'threshold_shellzone_t{SHELL_ZONE_BRIGHT_T}', sz_bright)
    nl_sz, lb_sz, st_sz, _ = cv2.connectedComponentsWithStats(sz_bright, connectivity=8)
    shell_zone_part = np.zeros_like(bl)
    for i in range(1, nl_sz):
        a = st_sz[i, cv2.CC_STAT_AREA]
        if SHELL_ZONE_MIN_AREA <= a <= SHELL_ZONE_MAX_AREA:
            shell_zone_part[lb_sz == i] = 255
    if save: save(f'shell_zone_part_area{SHELL_ZONE_MIN_AREA}-{SHELL_ZONE_MAX_AREA}', shell_zone_part)

    # — Финальные перегородки = Frangi + ring bridge + shell zone —
    partitions = cv2.bitwise_or(frangi_part, cv2.bitwise_or(bridge_bright, shell_zone_part))
    partitions  = cv2.bitwise_and(partitions, mask)
    partitions  = cv2.bitwise_and(partitions, cv2.bitwise_not(shell))
    return partitions


# ── Полный пайплайн для одного среза ──────────────────────────────────────

def process_slice(path, debug=False, output_dir=None):
    """
    Обработка одного среза.
    debug=True — сохранять промежуточные результаты каждой операции.
    output_dir — куда сохранять (по умолчанию /mnt/user-data/outputs/debug_steps).
    """
    import os
    img, bl, si = load_slice(path)
    call_idx = [0]

    if debug and output_dir is None:
        output_dir = '/mnt/user-data/outputs/debug_steps'

    def save(name, image):
        """Сохраняем промежуточный результат — только если debug=True."""
        if not debug or output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        call_idx[0] += 1
        fname = f"{call_idx[0]:03d}_{name}_si{si}.png"
        cv2.imwrite(os.path.join(output_dir, fname), image)

    # ── Стадия A ──────────────────────────────────────────────────────────
    save('load_slice_img', img)
    save('load_slice_blurred', bl)

    def make_saver(func_name):
        """Создаём save-callback с префиксом имени функции."""
        def _save(step_name, image):
            save(f'{func_name}__{step_name}', image)
        return _save

    mask = build_walnut_mask(bl, save=make_saver('build_walnut_mask'))

    # ── Стадия C ──────────────────────────────────────────────────────────
    shell, interior = build_shell(bl, mask, si, save=make_saver('build_shell'))

    # ── Стадия D ──────────────────────────────────────────────────────────
    kernel = build_kernel(bl, interior, si, save=make_saver('build_kernel'))

    # ── Стадия E ──────────────────────────────────────────────────────────
    partitions = build_partitions(bl, interior, kernel, shell=shell, mask=mask, si=si,
                                  save=make_saver('build_partitions'))

    # финальный overlay
    r = {
        'img':        img,
        'bl':         bl,
        'si':         si,
        'mask':       mask,
        'shell':      shell,
        'interior':   interior,
        'kernel':     kernel,
        'partitions': partitions,
    }
    save('overlay', make_overlay(r))

    return r


# ── Визуализация ───────────────────────────────────────────────────────────

def make_overlay(r):
    ov = cv2.cvtColor(r['img'], cv2.COLOR_GRAY2BGR)
    gl = np.zeros_like(ov); gl[r['shell'] > 0]      = (0, 180, 0)
    rl = np.zeros_like(ov); rl[r['kernel'] > 0]     = (0, 0, 210)
    yl = np.zeros_like(ov); yl[r['partitions'] > 0] = (0, 200, 200)
    ov = cv2.addWeighted(ov, 0.50, gl, 0.50, 0)
    ov = cv2.addWeighted(ov, 1.0,  rl, 0.45, 0)
    ov = cv2.addWeighted(ov, 1.0,  yl, 0.45, 0)
    return ov


# ── Запуск ─────────────────────────────────────────────────────────────────

def process_all(input_dir, output_dir='/mnt/user-data/outputs', debug=False,
                progress_every=50):
    """
    Обработать все .tiff срезы в input_dir.
    progress_every — выводить прогресс каждые N срезов (по умолчанию 50).
    debug=True — для каждого среза сохранить промежуточные шаги.
    """
    import os
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.tiff'))
    total = len(files)
    print(f'Найдено {total} срезов в {input_dir}')

    os.makedirs(output_dir, exist_ok=True)
    for i, fname in enumerate(files, start=1):
        path = os.path.join(input_dir, fname)
        debug_dir = os.path.join(output_dir, f'debug_{fname[:-5]}') if debug else None
        r = process_slice(path, debug=debug, output_dir=debug_dir)
        ov = make_overlay(r)
        out_path = os.path.join(output_dir, f'overlay_si{r["si"]:03d}.png')
        cv2.imwrite(out_path, ov)

        if i % progress_every == 0 or i == total:
            print(f'  обработано {i}/{total}  (последний si={r["si"]})')

    print(f'Готово. Результаты: {output_dir}')


if __name__ == '__main__':
    import sys, os

    # Если есть аргумент 'debug' — включаем режим отладки на тестовых срезах
    debug_mode = 'debug' in sys.argv

    BASE = 'C:\\masks_test'
    test_slices = [
        'full_AGD_50_000077.tiff',
        'full_AGD_50_000087.tiff',
        'full_AGD_50_000097.tiff',
        'full_AGD_50_000107.tiff',
        'full_AGD_50_000117.tiff',
        'full_AGD_50_000127.tiff',
        'full_AGD_50_000137.tiff',
        'full_AGD_50_000147.tiff',
        'full_AGD_50_000157.tiff',
        'full_AGD_50_000167.tiff',
        'full_AGD_50_000177.tiff',
        'full_AGD_50_000187.tiff',
        'full_AGD_50_000197.tiff',
        'full_AGD_50_000207.tiff',
        'full_AGD_50_000217.tiff',
        'full_AGD_50_000227.tiff',
        'full_AGD_50_000237.tiff',
        'full_AGD_50_000247.tiff',
        'full_AGD_50_000257.tiff',
        'full_AGD_50_000259.tiff',
        'full_AGD_50_000267.tiff',
        'full_AGD_50_000277.tiff',
        'full_AGD_50_000287.tiff',
        'full_AGD_50_000297.tiff',
        'full_AGD_50_000309.tiff',
        'full_AGD_50_000319.tiff',
        'full_AGD_50_000329.tiff',
        'full_AGD_50_000339.tiff',
        'full_AGD_50_000349.tiff',
        'full_AGD_50_000359.tiff',
        'full_AGD_50_000369.tiff',
        'full_AGD_50_000379.tiff',
        'full_AGD_50_000389.tiff',
        'full_AGD_50_000399.tiff',
    ]

    panels = []
    for fname in test_slices:
        path = f'{BASE}/{fname}'
        if not os.path.exists(path):
            continue
        print(f'Processing {fname} (debug={debug_mode})...')
        debug_dir = f'C:\\masks_test\\output\\debug_{fname[:-5]}' if debug_mode else None
        r = process_slice(path, debug=debug_mode, output_dir=debug_dir)
        ov = make_overlay(r)
        cv2.putText(ov, f"si={r['si']}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(ov)

    if panels:
        grid = cv2.hconcat(panels)
        out_path = "C:\\masks_test\\output\\output.png"
        cv2.imwrite(out_path, grid)
        print(f'Saved → {out_path}')
