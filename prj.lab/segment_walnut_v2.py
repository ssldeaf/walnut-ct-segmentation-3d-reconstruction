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
    'shell_pct':        [(69, 65),   (79, 75), (259, 88), (396, 80),  (408, 74),  (413, 65),  (420, 45),  (440, 50)],

    # Ядро: процентиль яркости для seeds
    'seed_pct':         [(84, 55),   (89, 55), (90, 65), (94,65), (95, 70), (96, 65), (99, 65), (100, 85), (269, 85), (270, 70), (290, 70), (291, 85), (305, 70), (365, 70), (370, 85), (371, 92), (384, 96), (500, 99)],

    # Ядро: минимальная площадь лоба как доля от площади interior.
    # Анкоры сейчас даёт постоянное 0.015 — добавь точки чтобы настроить per-si.
    'area_frac':        [(0, 0.015), (83, 0.015), (84, 0.02), (96, 0.025), (87, 0.02), (130, 0.025), (131,0.015), (270, 0.015), (271, 0.005),(296, 0.005), (297, 0.015), (306, 0.015), (307, 0.006), (362, 0.006), (363, 0.015), (372, 0.010), (380, 0.010), (381, 0.010), (385, 0.015), (500, 0.015)],

    # Ядро: фильтр формы
    'solidity_min':     [(0, 0.60),  (79, 0.50), (80, 0.55), (87, 0.57), (100, 0.59), (130, 0.65), (131, 0.55), (299, 0.55), (300, 0.35), (400, 0.55), (500, 0.55)],
    'aspect_min':       [(0, 0.40),  (79, 0.40), (80, 0.35), (87, 0.50), (130, 0.55), (131, 0.35), (210, 0.35), (211, 0.20), (215, 0.25), (216,0.35), (229, 0.35), (230, 0.25), (305, 0.15), (400, 0.15), (500, 0.35)],

    # Перегородки: пороги
    'frangi_pct':       [(65, 75),   (120, 20),  (259, 15), (300, 10)],
    'ring_bright_pct':  [(65, 70),   (259, 70)],
    'bright_filter_pct':[(65, 50),   (259, 50)],

    # Маска ореха: процентиль для малых si (вместо OTSU)
    'mask_pct':          [(0, 96),    (37, 99),   (38, 98),   (65, 96),   (70, 0),    (500, 0)],
}

# ── si-независимые параметры ───────────────────────────────────────────────
# Kernel
TOLERANCE      = 20    # region growing tolerance
DT_PCT         = 80    # distance transform percentile (si <= 69)
EXPAND_TOL     = 25    # BFS expand tolerance (si <= 69)
SHELL_CLOSE_PX = 5     # morph close для si == 69
HOLE_FRAC_MAX  = 0.10
KERNEL_MIN_SI  = 70    # срезы ниже этого значения не имеют ядра (si <= 49)
KERNEL_MAX_SI  = 383   # срезы выше этого значения не имеют ядра (si >= 384)

# Partitions (стадия E) — структурные параметры (не зависят от si)
FRANGI_INTERIOR_ERODE = 3
FRANGI_KERNEL_DILATE  = 1
FRANGI_SIGMAS         = range(1, 4)
RING_PX               = 15
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
    # перцентильное клиппирование — убираем выбросы (топ и низ 1%)
    # защищает от артефактов которые сжимают орех в серый диапазон
    p_low  = np.percentile(raw, 1)
    p_high = np.percentile(raw, 99)
    raw = np.clip(raw, p_low, p_high)
    img = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bl  = img
    fname = path.replace(chr(92), '/').split('/')[-1]
    m = re.search(r'(\d+)(?=\.tiff?$)', fname, re.IGNORECASE)
    si = int(m.group(1)) if m else 0
    return img, bl, si
# ── Стадия A: маска ореха ──────────────────────────────────────────────────

def build_walnut_mask(bl, si=0, save=None):
    # для малых si (<=65): opening r=5 убирает шум + p99 вместо OTSU
    mask_pct = interp_param('mask_pct', si)
    if mask_pct > 0:
        # очистка от шума через opening
        ke_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # r=5
        bl_clean = cv2.morphologyEx(bl, cv2.MORPH_OPEN, ke_open)
        if save: save('morphOpen_denoise', bl_clean)
        nz = bl_clean[bl_clean > 0]
        # ищем наименьший pct начиная с mask_pct который даёт t < 255
        t = 255
        actual_pct = mask_pct
        for try_pct in range(int(mask_pct), 50, -1):
            t_try = int(np.percentile(nz, try_pct)) if len(nz) > 0 else 0
            if t_try < 250:
                t = t_try
                actual_pct = try_pct
                break
        _, bn = cv2.threshold(bl_clean, t, 255, cv2.THRESH_BINARY)
        if save: save(f'threshold_p{actual_pct:.0f}_t{t}', bn)
    else:
        # обычный OTSU для больших si
        _, bn = cv2.threshold(bl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if save: save('threshold_otsu', bn)

    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cl = cv2.morphologyEx(bn, cv2.MORPH_CLOSE, k7)
    if save: save('morphologyEx_close', cl)
    cs, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = bl.shape
    cx_img, cy_img = W // 2, H // 2
    max_dist = min(H, W) * 0.4
    min_area = 500 if mask_pct > 0 else 5000  # меньший порог для малых si
    valid = []
    for c in cs:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        dist = np.sqrt((cx - cx_img)**2 + (cy - cy_img)**2)
        if dist <= max_dist:
            valid.append(c)
    if not valid:
        valid = [c for c in cs if cv2.contourArea(c) >= min_area]
    mask = np.zeros_like(bl)
    if valid:
        cv2.drawContours(mask, [max(valid, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    if save: save('drawContours_largest', mask)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if save: save('dilate', mask)
    return mask


# ── Стадия B: маска скорлупы ───────────────────────────────────────────────

def build_shell(bl, mask, si, save=None):
    # для малых si (нет ядра) — вся маска является скорлупой
    if si < KERNEL_MIN_SI:
        if save: save('shell_result', mask)
        interior = np.zeros_like(bl)
        if save: save('interior_result', interior)
        return mask.copy(), interior

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

    if si > 70:
        shi[lb == areas[0][1]] = 255
        shell = cv2.bitwise_or(shi, outer)

    elif si >= 50:  # si 50-70: берём все компоненты с ratio >= 0.9 (скорлупа может быть в двух частях)
        for a, idx in areas:
            if a >= max_area * 0.2:
                shi[lb == idx] = 255
        shell = cv2.bitwise_or(shi, outer)

    else:  # si < 50
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


# ── Стадия C: маска ядра ───────────────────────────────────────────────────

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
    if si < KERNEL_MIN_SI or si > KERNEL_MAX_SI:  # нет ядра вне диапазона
        return np.zeros_like(bl)
    int_area = int((interior > 0).sum())
    if int_area == 0:
        return np.zeros_like(bl)
    area_frac = interp_param('area_frac', si)
    min_lobe = max(50, int(int_area * area_frac))

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


# ── Стадия D: маска перегородок ────────────────────────────────────────────

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
    if interior_eroded.sum() == 0:
        # пустой interior — нет перегородок
        return np.zeros_like(bl)
    fo   = frangi(work, sigmas=FRANGI_SIGMAS, black_ridges=False)
    fo[interior_eroded == 0] = 0
    fn = cv2.normalize(fo, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if save: save('frangi_response', fn)
    fn_vals = fn[interior_eroded > 0]
    ft = int(np.percentile(fn_vals, frangi_pct)) if len(fn_vals) > 0 else 0
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

    mask = build_walnut_mask(bl, si=si, save=make_saver('build_walnut_mask'))

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

def process_all(input_dir, output_dir='out', debug=False, progress_every=50):
    """
    Обработать все .tiff/.tif срезы в input_dir.
    progress_every — выводить прогресс каждые N срезов.
    debug=True — для каждого среза сохранить промежуточные шаги.
    """
    import os
    print(f'[INFO] input_dir = {input_dir}')
    print(f'[INFO] output_dir = {output_dir}')
    print(f'[INFO] папка существует: {os.path.isdir(input_dir)}')

    if not os.path.isdir(input_dir):
        print(f'[ERROR] Папка не найдена: {input_dir}')
        return

    all_files = os.listdir(input_dir)
    print(f'[INFO] всего файлов в папке: {len(all_files)}')
    if all_files:
        print(f'[INFO] первые 5 файлов: {all_files[:5]}')

    files = sorted(f for f in all_files
                   if f.lower().endswith('.tiff') or f.lower().endswith('.tif'))
    total = len(files)
    print(f'[INFO] найдено TIFF срезов: {total}')

    if total == 0:
        print('[ERROR] TIFF файлы не найдены. Проверь расширение и путь.')
        return

    os.makedirs(output_dir, exist_ok=True)
    errors = 0

    for i, fname in enumerate(files, start=1):
        path = os.path.join(input_dir, fname)
        debug_dir = os.path.join(output_dir, f'debug_{fname[:-5]}') if debug else None
        try:
            r = process_slice(path, debug=debug, output_dir=debug_dir)
            ov = make_overlay(r)
            out_path = os.path.join(output_dir, f'overlay_si{r["si"]:03d}.png')
            cv2.imwrite(out_path, ov)
        except Exception as e:
            errors += 1
            print(f'  [!] ошибка на {fname}: {e}')
            continue

        if i % progress_every == 0 or i == total:
            print(f'  обработано {i}/{total}  (последний si={r["si"]})')

    print(f'Готово. Результаты: {output_dir}  Ошибок: {errors}')


if __name__ == '__main__':
    import sys
    import os

    debug_mode = 'debug' in sys.argv

    # === НАСТРОЙКА ПУТЕЙ ===
    INPUT_DIR  = r'C:\masks_test'
    OUTPUT_DIR = r'C:\masks_test\out'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Парсим числовые аргументы командной строки как диапазон si (например: 40 50)
    numeric_args = [a for a in sys.argv[1:] if a.isdigit()]
    si_range = None
    if len(numeric_args) >= 2:
        si_range = (int(numeric_args[0]), int(numeric_args[1]))
    elif len(numeric_args) == 1:
        si_range = (int(numeric_args[0]), int(numeric_args[0]))

    # Режим: 'all' — все файлы, 'test' — только выбранные срезы
    MODE = 'all'  # 'all' или 'test'

    if si_range is not None:
        # Обработать только срезы с si в диапазоне [si_range[0], si_range[1]]
        si_lo, si_hi = si_range
        print(f'[INFO] Диапазон si: {si_lo} — {si_hi}')
        all_files = sorted(f for f in os.listdir(INPUT_DIR)
                           if f.lower().endswith('.tiff') or f.lower().endswith('.tif'))
        selected = []
        for fname in all_files:
            m = re.search(r'(\d+)(?=\.tiff?$)', fname, re.IGNORECASE)
            if m and si_lo <= int(m.group(1)) <= si_hi:
                selected.append(fname)
        print(f'[INFO] Найдено срезов в диапазоне: {len(selected)}')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        errors = 0
        for fname in selected:
            path = os.path.join(INPUT_DIR, fname)
            debug_dir = os.path.join(OUTPUT_DIR, f'debug_{fname[:-5]}') if debug_mode else None
            try:
                r = process_slice(path, debug=debug_mode, output_dir=debug_dir)
                ov = make_overlay(r)
                out_path = os.path.join(OUTPUT_DIR, f'overlay_si{r["si"]:03d}.png')
                cv2.imwrite(out_path, ov)
                print(f'  [OK] overlay_si{r["si"]:03d}.png')
            except Exception as e:
                errors += 1
                print(f'  [!] ошибка на {fname}: {e}')
        print(f'Готово. Результаты: {OUTPUT_DIR}  Ошибок: {errors}')

    elif MODE == 'all':
        # Обработать ВСЕ файлы в папке
        process_all(INPUT_DIR, OUTPUT_DIR, debug=debug_mode, progress_every=10)

    elif MODE == 'test':
        # Обработать ТОЛЬКО ВЫБРАННЫЕ файлы
        test_slices = [
            'full_AGD_50_000259.tiff',
        ]
        panels = []
        for fname in test_slices:
            path = os.path.join(INPUT_DIR, fname)
            if not os.path.exists(path):
                print(f'[!] Файл не найден: {path}')
                continue

            print(f'Processing {fname} (debug={debug_mode})...')
            debug_dir = os.path.join(OUTPUT_DIR, f'debug_{fname[:-5]}') if debug_mode else None
            r = process_slice(path, debug=debug_mode, output_dir=debug_dir)
            ov = make_overlay(r)
            cv2.putText(ov, f"si={r['si']}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            panels.append(ov)

            cv2.imwrite(os.path.join(OUTPUT_DIR, f'overlay_si{r["si"]:03d}.png'), ov)

        if panels and all(p.shape[0] == panels[0].shape[0] for p in panels):
            grid = cv2.hconcat(panels)
            out_path = os.path.join(OUTPUT_DIR, 'walnut_v3_result.png')
            cv2.imwrite(out_path, grid)
            print(f'[OK] Saved -> {out_path}')