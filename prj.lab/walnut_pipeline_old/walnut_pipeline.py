"""
Walnut X-ray CT Slice Segmentation Pipeline
Stages: A (walnut mask) → C (shell) → D (kernel) → E (partitions)

Два независимых формата автоматической трассировки (можно включать оба):

  --trace          (trace=True) — пары input/output для каждого вызова
                                  cv2/skimage:
    NNN_<func>_si<si>_input.png    — вход с впечатанным '#NNN'
    NNN_<func>_si<si>_output.png   — результат

  --trace-labeled  (trace_labeled=True) — один файл на вызов:
    from<global_id>_<func>.png     — результат вызова
    где <global_id> — устойчивый ID входной картинки на основе id(ndarray).
    Если та же картинка приходит на вход в несколько разных методов,
    у всех этих файлов будет одинаковый <global_id> в имени.
"""

import cv2
import numpy as np

_orig_percentile = np.percentile

def _safe_percentile(a, q, *args, **kwargs):
    a = np.asarray(a)
    if a.size == 0:
        return 0.0  # int(0.0) = 0, порог сработает безопасно
    return _orig_percentile(a, q, *args, **kwargs)

np.percentile = _safe_percentile

import os
import sys
import tifffile
import re
from collections import deque
from skimage.filters import frangi

# ── Параметры ──────────────────────────────────────────────────────────────

# ── si-зависимые параметры (линейная интерполяция между анкорами) ──────────
# Формат: (si, value). Вне диапазона анкоров — берётся ближайшее значение.
PARAM_ANCHORS = {
    # Скорлупа: процентиль яркости для порога
    'shell_pct':        [(69, 26),   (79, 36), (88, 46),  (259, 88), (390, 88), (391, 65), (407, 32), (433, 20)],

    # Ядро: процентиль яркости для seeds
    'seed_pct':         [(0, 92),    (69, 92),   (70, 85), (391, 85), (500, 85)],

    # Ядро: фильтр формы
    'solidity_min':     [(0, 0.60),  (79, 0.60), (80, 0.55), (500, 0.55)],
    'aspect_min':       [(0, 0.40),  (79, 0.40), (80, 0.35), (500, 0.35)],

    # Перегородки: пороги
    'frangi_pct':       [(69, 85),   (259, 85)],
    'ring_bright_pct':  [(69, 70),   (259, 70)],
    'bright_filter_pct':[(69, 50),   (259, 50)],
}

# ── si-независимые параметры ───────────────────────────────────────────────
# Kernel
TOLERANCE      = 20    # region growing tolerance
DT_PCT         = 80    # distance transform percentile (si <= 69)
AREA_FRAC      = 0.005 # min lobe area fraction
EXPAND_TOL     = 25    # BFS expand tolerance (si <= 69)
SHELL_CLOSE_PX = 5     # morph close для si == 69
HOLE_FRAC_MAX  = 0.10
KERNEL_MIN_SI  = 83    # срезы ниже этого значения не имеют ядра (si <= 83)
KERNEL_MAX_SI  = 383   # срезы выше этого значения не имеют ядра (si <= 390)

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


# ── ImageTracer: автоматический лог всех преобразований изображения ────────

class ImageTracer:
    """
    Monkey-patches функции cv2 и skimage.frangi: каждый вызов, который
    принимает на вход изображение и возвращает изображение, логируется как
    пара файлов:
        NNN_<имя_функции>_input_si<si>.png   — вход с впечатанным номером
        NNN_<имя_функции>_output_si<si>.png  — результат

    Использование:
        tracer = ImageTracer(output_dir, si=si)
        tracer.install()
        try:
            ...  # работа пайплайна
        finally:
            tracer.uninstall()
    """

    # Функции cv2, у которых вход-картинка → выход-картинка.
    # findContours и connectedComponentsWithStats намеренно исключены
    # (возвращают структуры данных / служебные labels).
    CV2_FUNCS = [
        'medianBlur', 'normalize', 'threshold', 'morphologyEx',
        'dilate', 'erode', 'subtract', 'bitwise_and', 'bitwise_or',
        'bitwise_not', 'distanceTransform', 'cvtColor', 'addWeighted',
    ]

    def __init__(self, output_dir, si, trace_pairs=True, trace_labeled=False):
        """
        trace_pairs   — старый формат: NNN_<func>_si<si>_input.png / _output.png.
        trace_labeled — новый формат: from<global_id>_<func>.png (только результат).
                        <global_id> — устойчивый ID картинки на основе id(ndarray):
                        если одна и та же картинка приходит на вход к разным
                        методам, у всех будет одинаковый ID.
        Можно включать оба флага одновременно.
        """
        self.output_dir = output_dir
        self.si = si
        self.call_idx = 0
        self.trace_pairs = trace_pairs
        self.trace_labeled = trace_labeled
        # Глобальные ID картинок: id(ndarray) -> int
        self._image_ids = {}
        self._next_image_id = 0
        self._cv2_originals = {}
        self._frangi_original = None
        # Снимок оригиналов cv2-функций, которые мы используем ВНУТРИ хелперов
        # _to_uint8 / _stamp_number / _log. Берём их заранее, чтобы
        # внутренние вызовы не уходили через наши же trace-обёртки
        # (иначе — бесконечная рекурсия).
        self._cv_normalize = cv2.normalize
        self._cv_cvtColor  = cv2.cvtColor
        self._cv_imwrite   = cv2.imwrite
        self._cv_rectangle = cv2.rectangle
        self._cv_putText   = cv2.putText
        self._cv_getTextSize = cv2.getTextSize

    def _get_id(self, arr):
        """Получить (и при необходимости назначить) глобальный ID картинки."""
        aid = id(arr)
        nid = self._image_ids.get(aid)
        if nid is None:
            nid = self._next_image_id
            self._image_ids[aid] = nid
            self._next_image_id += 1
        return nid

    # — Вспомогательные —

    @staticmethod
    def _looks_like_image(x):
        """Эвристика: ndarray с разумными размерами (>=10x10), 1/3/4 канала или grayscale."""
        if not isinstance(x, np.ndarray):
            return False
        if x.dtype.kind not in ('u', 'i', 'f', 'b'):
            return False
        if x.ndim == 2:
            return x.shape[0] >= 10 and x.shape[1] >= 10
        if x.ndim == 3:
            return (x.shape[0] >= 10 and x.shape[1] >= 10
                    and x.shape[2] in (1, 3, 4))
        return False

    def _to_uint8(self, img):
        """Привести к uint8 для записи через cv2.imwrite."""
        if img.dtype == np.uint8:
            return img
        if img.dtype == bool:
            return (img.astype(np.uint8)) * 255
        # float / int → нормализация в 0..255 (через ОРИГИНАЛЬНЫЙ normalize)
        out = self._cv_normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        return out.astype(np.uint8)

    def _stamp_number(self, img, num):
        """На копии входной картинки рисуем '#NNN' в левом верхнем углу."""
        u8 = self._to_uint8(img)
        bgr = u8 if u8.ndim == 3 else self._cv_cvtColor(u8, cv2.COLOR_GRAY2BGR)
        out = bgr.copy()
        text = f"#{num}"
        # Подложка для читаемости поверх любого фона
        (tw, th), _ = self._cv_getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        self._cv_rectangle(out, (4, 4), (8 + tw, 12 + th), (0, 0, 0), -1)
        self._cv_putText(out, text, (6, 8 + th), cv2.FONT_HERSHEY_SIMPLEX,
                         0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return out

    def _log(self, fn_name, input_img, output_img):
        """Логируем шаг в один или оба формата (в зависимости от флагов)."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.call_idx += 1
        idx = self.call_idx

        # ID назначаем всегда (нужны для labeled-формата и стабильны для prev-output → next-input)
        in_id  = self._get_id(input_img)
        out_id = self._get_id(output_img)

        out_u8 = self._to_uint8(output_img)

        # 1) Старый формат: пары input/output с впечатанным NNN
        if self.trace_pairs:
            base = f"{idx:03d}_{fn_name}_si{self.si:03d}"
            in_path  = os.path.join(self.output_dir, f"{base}_input.png")
            out_path = os.path.join(self.output_dir, f"{base}_output.png")
            self._cv_imwrite(in_path,  self._stamp_number(input_img, idx))
            self._cv_imwrite(out_path, out_u8)

        # 2) Новый формат: from<input_id>_<method>.png
        #    Имя кодирует пару (id входа, метод). Если такое имя уже занято
        #    (например, тот же вход + тот же метод были вызваны повторно с
        #    другими kwargs), добавляем суффикс __v2, __v3, ...
        if self.trace_labeled:
            stem = f"from{in_id}_{fn_name}"
            target = os.path.join(self.output_dir, f"{stem}.png")
            n = 2
            while os.path.exists(target):
                target = os.path.join(self.output_dir, f"{stem}__v{n}.png")
                n += 1
            self._cv_imwrite(target, out_u8)

    # — Обёртка вокруг функции —

    def _wrap(self, original, fn_name):
        tracer = self

        def wrapped(*args, **kwargs):
            # 1) Найти первый аргумент-изображение (среди позиционных и kwargs)
            input_img = None
            for a in args:
                if tracer._looks_like_image(a):
                    input_img = a
                    break
            if input_img is None:
                for v in kwargs.values():
                    if tracer._looks_like_image(v):
                        input_img = v
                        break

            # 2) Вызвать оригинал
            result = original(*args, **kwargs)

            # 3) Найти выходное изображение
            output_img = None
            if tracer._looks_like_image(result):
                output_img = result
            elif isinstance(result, tuple):
                for r in result:
                    if tracer._looks_like_image(r):
                        output_img = r
                        break

            # 4) Если оба есть — логируем (и они должны хоть как-то пересекаться по форме,
            #    чтобы отсечь ложные срабатывания вроде getStructuringElement)
            if input_img is not None and output_img is not None:
                tracer._log(fn_name, input_img, output_img)

            return result

        wrapped.__name__ = f"traced_{fn_name}"
        return wrapped

    # — Установка / снятие —

    def install(self):
        """Подменить cv2.* и frangi на trace-обёртки."""
        for fn_name in self.CV2_FUNCS:
            orig = getattr(cv2, fn_name)
            self._cv2_originals[fn_name] = orig
            setattr(cv2, fn_name, self._wrap(orig, fn_name))

        # frangi импортирован в этот модуль из skimage.filters — заменяем его
        # прямо в globals() этого модуля (надёжнее, чем sys.modules[__name__]:
        # последнее не работает при загрузке модуля через importlib.util).
        g = globals()
        self._frangi_original = g['frangi']
        g['frangi'] = self._wrap(self._frangi_original, 'frangi')

    def uninstall(self):
        """Восстановить оригиналы. Безопасно вызывать многократно."""
        for fn_name, orig in self._cv2_originals.items():
            setattr(cv2, fn_name, orig)
        self._cv2_originals.clear()
        if self._frangi_original is not None:
            globals()['frangi'] = self._frangi_original
            self._frangi_original = None


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

    if si > 88:
        shi[lb == areas[0][1]] = 255
        shell = cv2.bitwise_or(shi, outer)

    elif si == 88:
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

    if si <= 88:
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
    if si > KERNEL_MAX_SI:   # срез слишком высокий — ядра уже нет
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

def process_slice(path, debug=False, output_dir=None, trace=False, trace_labeled=False):
    """
    Обработка одного среза.
    debug=True         — сохранять ключевые промежуточные результаты (старый save-механизм).
    trace=True         — автоматически логировать КАЖДЫЙ вызов cv2/skimage,
                         сохраняя пары NNN_<функция>_si<si>_input.png /
                                       NNN_<функция>_si<si>_output.png.
    trace_labeled=True — параллельный формат: для каждого вызова сохранять
                         один файл from<global_id>_<функция>.png, где
                         <global_id> — устойчивый ID входного изображения.
    output_dir — куда сохранять (по умолчанию /mnt/user-data/outputs/debug_steps).
    """
    img, bl, si = load_slice(path)
    call_idx = [0]

    if (debug or trace or trace_labeled) and output_dir is None:
        output_dir = '/mnt/user-data/outputs/debug_steps'

    def save(name, image):
        """Сохраняем промежуточный результат — только если debug=True."""
        if not debug or output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        call_idx[0] += 1
        fname = f"{call_idx[0]:03d}_{name}_si{si}.png"
        cv2.imwrite(os.path.join(output_dir, fname), image)

    def make_saver(func_name):
        """Создаём save-callback с префиксом имени функции."""
        def _save(step_name, image):
            save(f'{func_name}__{step_name}', image)
        return _save

    # Активируем автоматический tracer (если запрошен), оборачиваем
    # ВЕСЬ блок обработки в try/finally — чтобы гарантированно снять
    # monkey-patches, даже если что-то упадёт.
    tracer = None
    if trace or trace_labeled:
        tracer = ImageTracer(output_dir, si=si,
                             trace_pairs=trace,
                             trace_labeled=trace_labeled)
        tracer.install()

    try:
        # ── Стадия A ──────────────────────────────────────────────────────
        save('load_slice_img', img)
        save('load_slice_blurred', bl)

        mask = build_walnut_mask(bl, save=make_saver('build_walnut_mask'))

        # ── Стадия C ──────────────────────────────────────────────────────
        shell, interior = build_shell(bl, mask, si, save=make_saver('build_shell'))

        # ── Стадия D ──────────────────────────────────────────────────────
        kernel = build_kernel(bl, interior, si, save=make_saver('build_kernel'))

        # ── Стадия E ──────────────────────────────────────────────────────
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
    finally:
        if tracer is not None:
            tracer.uninstall()

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


# ── Ручное редактирование overlay-масок: export / import ──────────────────
#
# Workflow:
#   Фаза 1 (--export-masks):  для каждого среза в masks/si<NNN>/ сохраняются
#       img.png                — исходный grayscale срез (для справки)
#       overlay_shell.png      — серый срез + ЗЕЛЁНЫЕ зоны скорлупы
#       overlay_kernel.png     — серый срез + КРАСНЫЕ зоны ядра
#       overlay_partitions.png — серый срез + ЖЁЛТЫЕ зоны перегородок
#       Каждая overlay-маска редактируется ОТДЕЛЬНО.
#
#   Фаза 2 (--import-masks):  парсер берёт из каждого overlay только «свой»
#       цветовой канал и восстанавливает соответствующую бинарную маску,
#       затем make_overlay() склеивает финальную картинку.
#
# Парсер устойчив и к полупрозрачным цветам make_overlay,
# и к чистым цветам кисти ((0,255,0) / (0,0,255) / (0,255,255)).

# Цвета и режимы наложения (один-в-один с make_overlay для согласованного вида).
_SHELL_COLOR_BGR      = (0, 180, 0)
_KERNEL_COLOR_BGR     = (0, 0, 210)
_PARTITIONS_COLOR_BGR = (0, 200, 200)

# Порог классификации пикселя по «эксцессу» канала.
_OVERLAY_PARSE_THRESHOLD = 25


def _mask_subdir(masks_root, si):
    """Подпапка для одного среза: <masks_root>/si<NNN>/."""
    return os.path.join(masks_root, f'si{si:03d}')


def _overlay_single(img_gray, mask, color_bgr, mode):
    """
    Полупрозрачное наложение ОДНОЙ маски на серое изображение в стиле make_overlay.
    mode='blend' — 50/50 (как shell в make_overlay)
    mode='add'   — base + 0.45*color (как kernel и partitions)
    """
    ov = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    layer = np.zeros_like(ov)
    layer[mask > 0] = color_bgr
    if mode == 'blend':
        return cv2.addWeighted(ov, 0.5, layer, 0.5, 0)
    else:
        return cv2.addWeighted(ov, 1.0, layer, 0.45, 0)


def _parse_overlay_channel(overlay_bgr, channel, threshold=_OVERLAY_PARSE_THRESHOLD):
    """
    Восстановить бинарную маску из overlay одного цвета.
      channel='green'  — для overlay_shell.png  → ловит зелёные зоны
      channel='red'    — для overlay_kernel.png → ловит красные зоны
      channel='yellow' — для overlay_partitions.png → ловит жёлтые зоны
    """
    bgr = overlay_bgr.astype(np.int16)
    B, G, R = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    if channel == 'green':
        excess = G - np.maximum(B, R)
    elif channel == 'red':
        excess = R - np.maximum(B, G)
    elif channel == 'yellow':
        excess = np.minimum(G, R) - B
    else:
        raise ValueError(f"channel must be 'green'/'red'/'yellow', got {channel!r}")
    return ((excess > threshold).astype(np.uint8)) * 255


def parse_overlay_to_masks(overlay_bgr, threshold=_OVERLAY_PARSE_THRESHOLD):
    """
    Совместимый комбинированный парсер: восстановить (shell, kernel, partitions)
    из одной BGR overlay-картинки (если все три цвета на одном изображении).
    В новом workflow НЕ нужен — каждая маска парсится из своего файла, — но
    оставлен для обратной совместимости.
    Приоритет при перекрытии: kernel > partitions > shell.
    """
    s = _parse_overlay_channel(overlay_bgr, 'green',  threshold)
    k = _parse_overlay_channel(overlay_bgr, 'red',    threshold)
    p = _parse_overlay_channel(overlay_bgr, 'yellow', threshold)
    # Приоритет: kernel > partitions > shell
    p = cv2.bitwise_and(p, cv2.bitwise_not(k))
    s = cv2.bitwise_and(s, cv2.bitwise_not(cv2.bitwise_or(k, p)))
    return s, k, p


def export_masks(r, masks_root):
    """
    Фаза 1: сохранить исходник и ТРИ редактируемые overlay-маски.

    В masks_root/si<NNN>/ создаются:
      img.png                — исходный grayscale срез (для справки)
      overlay_shell.png      — серый срез + ЗЕЛЁНАЯ скорлупа (правьте только её)
      overlay_kernel.png     — серый срез + КРАСНОЕ ядро       (правьте только его)
      overlay_partitions.png — серый срез + ЖЁЛТЫЕ перегородки (правьте только их)

    Возвращает путь к созданной подпапке.
    """
    folder = _mask_subdir(masks_root, r['si'])
    os.makedirs(folder, exist_ok=True)
    img = r['img']
    cv2.imwrite(os.path.join(folder, 'img.png'), img)
    cv2.imwrite(os.path.join(folder, 'overlay_shell.png'),
                _overlay_single(img, r['shell'],      _SHELL_COLOR_BGR,      'blend'))
    cv2.imwrite(os.path.join(folder, 'overlay_kernel.png'),
                _overlay_single(img, r['kernel'],     _KERNEL_COLOR_BGR,     'add'))
    cv2.imwrite(os.path.join(folder, 'overlay_partitions.png'),
                _overlay_single(img, r['partitions'], _PARTITIONS_COLOR_BGR, 'add'))
    return folder


def import_masks(masks_root, si):
    """
    Фаза 2: прочитать три (возможно правленые) overlay-маски и восстановить
    три бинарные маски из соответствующих цветовых каналов.

    Возвращает dict, совместимый с make_overlay():
        {'img', 'bl', 'si', 'shell', 'kernel', 'partitions',
         'mask', 'interior'}  (последние два — None, для make_overlay не нужны)
    """
    folder = _mask_subdir(masks_root, si)
    img_path = os.path.join(folder, 'img.png')
    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Не найден {img_path}. Сначала запустите --export-masks.")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    shape = img.shape

    def _read_layer(name, channel):
        p = os.path.join(folder, name)
        if not os.path.exists(p):
            # Если файла нет — считаем что маска пустая (пользователь стёр её)
            return np.zeros(shape, dtype=np.uint8)
        ov = cv2.imread(p, cv2.IMREAD_COLOR)
        if ov.shape[:2] != shape:
            ov = cv2.resize(ov, (shape[1], shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        return _parse_overlay_channel(ov, channel)

    return {
        'img':        img,
        'bl':         img,
        'si':         si,
        'mask':       None,
        'interior':   None,
        'shell':      _read_layer('overlay_shell.png',      'green'),
        'kernel':     _read_layer('overlay_kernel.png',     'red'),
        'partitions': _read_layer('overlay_partitions.png', 'yellow'),
    }


def process_all_export(input_dir, masks_root, progress_every=50):
    """
    Фаза 1 для ручного редактирования.
    Прогоняет пайплайн по всем TIFF в input_dir, для каждого среза
    сохраняет в masks_root/si<NNN>/ исходник + 3 редактируемых overlay.
    Финальные overlay-изображения НЕ создаёт.
    """
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.tiff'))
    total = len(files)
    print(f'[EXPORT] Найдено {total} срезов в {input_dir}')
    os.makedirs(masks_root, exist_ok=True)
    for i, fname in enumerate(files, start=1):
        path = os.path.join(input_dir, fname)
        r = process_slice(path)
        export_masks(r, masks_root)
        if i % progress_every == 0 or i == total:
            print(f'  [EXPORT] {i}/{total}  (последний si={r["si"]})')
    print(f'[EXPORT] Готово. Маски в: {masks_root}')
    print(f'[EXPORT] Правьте overlay_shell.png / overlay_kernel.png / overlay_partitions.png,')
    print(f'[EXPORT] затем запустите --import-masks для финализации.')


def process_all_import(masks_root, output_dir, progress_every=50):
    """
    Фаза 2: после ручной правки.
    Идёт по подпапкам masks_root/si<NNN>/, парсит три overlay-маски,
    восстанавливает 3 бинарные маски и склеивает финальный overlay в output_dir.
    """
    subdirs = sorted(d for d in os.listdir(masks_root)
                     if d.startswith('si') and
                        os.path.isdir(os.path.join(masks_root, d)))
    total = len(subdirs)
    print(f'[IMPORT] Найдено {total} срезов с масками в {masks_root}')
    os.makedirs(output_dir, exist_ok=True)
    for i, d in enumerate(subdirs, start=1):
        try:
            si = int(d[2:])
        except ValueError:
            print(f'  [IMPORT] Пропуск {d} (не удалось распарсить si)')
            continue
        r = import_masks(masks_root, si)
        ov = make_overlay(r)
        out_path = os.path.join(output_dir, f'overlay_si{si:03d}.png')
        cv2.imwrite(out_path, ov)
        if i % progress_every == 0 or i == total:
            print(f'  [IMPORT] {i}/{total}  (si={si})')
    print(f'[IMPORT] Готово. Overlay-изображения в: {output_dir}')


# ── Запуск ─────────────────────────────────────────────────────────────────

def process_all(input_dir, output_dir='C:\\masks_test\\output', debug=False,
                trace=False, trace_labeled=False, progress_every=50):
    """
    Обработать все .tiff срезы в input_dir.
    progress_every     — выводить прогресс каждые N срезов (по умолчанию 50).
    debug=True         — ключевые промежуточные шаги.
    trace=True         — пары input/output для каждого вызова cv2/skimage.
    trace_labeled=True — формат from<global_id>_<func>.png (только результат).
    """
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.tiff'))
    total = len(files)
    print(f'Найдено {total} срезов в {input_dir}')

    os.makedirs(output_dir, exist_ok=True)
    for i, fname in enumerate(files, start=1):
        path = os.path.join(input_dir, fname)
        sub_dir = None
        if debug or trace or trace_labeled:
            sub_dir = os.path.join(output_dir, f'debug_{fname[:-5]}')
        r = process_slice(path, debug=debug, output_dir=sub_dir,
                          trace=trace, trace_labeled=trace_labeled)
        ov = make_overlay(r)
        out_path = os.path.join(output_dir, f'overlay_si{r["si"]:03d}.png')
        cv2.imwrite(out_path, ov)

        if i % progress_every == 0 or i == total:
            print(f'  обработано {i}/{total}  (последний si={r["si"]})')

    print(f'Готово. Результаты: {output_dir}')


if __name__ == '__main__':
    import sys
    import argparse
    import traceback

    # === НАСТРОЙКА АРГУМЕНТОВ ===
    parser = argparse.ArgumentParser(
        description="Обработка TIFF-срезов: режим all (все файлы) или test (выборочные)"
    )
    parser.add_argument('--mode', choices=['all', 'test'], default='test',
                        help="Режим: 'all' — все файлы, 'test' — выборочные срезы")
    parser.add_argument('--input-dir', default=r'C:\masks_test',
                        help="Путь к папке с входными TIFF-файлами")
    parser.add_argument('--output-dir', default=r'C:\masks_test\output',
                        help="Путь к папке для сохранения результатов")
    parser.add_argument('--debug', action='store_true',
                        help="Включить режим отладки (сохранение ключевых промежуточных шагов)")
    parser.add_argument('--trace', action='store_true',
                        help="Логировать КАЖДЫЙ вызов cv2/skimage: "
                             "пары NNN_<func>_input_si<si>.png / NNN_<func>_output_si<si>.png")
    parser.add_argument('--trace-labeled', action='store_true',
                        help="Параллельный формат: один файл на вызов, "
                             "from<global_id>_<func>.png — где global_id "
                             "это стабильный ID входного изображения. "
                             "Можно комбинировать с --trace.")
    parser.add_argument('--export-masks', action='store_true',
                        help="Фаза 1: сохранить shell/kernel/partitions/img в подпапки "
                             "masks-dir/si<NNN>/ для ручного редактирования. "
                             "Финальные overlay НЕ создаёт.")
    parser.add_argument('--import-masks', action='store_true',
                        help="Фаза 2: прочитать (возможно отредактированные) маски "
                             "из masks-dir/si<NNN>/ и собрать финальные overlay.")
    parser.add_argument('--masks-dir', default=None,
                        help="Папка с масками для --export-masks / --import-masks "
                             "(по умолчанию <output-dir>/masks/).")
    parser.add_argument('--skip-errors', action='store_true',
                        help="Продолжать обработку при ошибке на отдельном срезе")
    args = parser.parse_args()

    INPUT_DIR  = args.input_dir
    OUTPUT_DIR = args.output_dir
    debug_mode = args.debug
    trace_mode = args.trace
    trace_labeled_mode = args.trace_labeled
    skip_errors = args.skip_errors
    debug_any = debug_mode or trace_mode or trace_labeled_mode

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    masks_dir = args.masks_dir or os.path.join(OUTPUT_DIR, 'masks')

    # === ФАЗА 1: ЭКСПОРТ МАСОК ДЛЯ РУЧНОГО РЕДАКТИРОВАНИЯ ===
    if args.export_masks:
        print(f"\n[PHASE 1] Экспорт масок в '{masks_dir}'...")
        try:
            process_all_export(INPUT_DIR, masks_dir, progress_every=10)
        except Exception as e:
            print(f"[ERR] Ошибка экспорта: {e}")
            if debug_any: traceback.print_exc()
            if not skip_errors: sys.exit(1)
        sys.exit(0)

    # === ФАЗА 2: ИМПОРТ ОТРЕДАКТИРОВАННЫХ МАСОК И ФИНАЛИЗАЦИЯ ===
    if args.import_masks:
        print(f"\n[PHASE 2] Финализация overlay из '{masks_dir}'...")
        try:
            process_all_import(masks_dir, OUTPUT_DIR, progress_every=10)
        except Exception as e:
            print(f"[ERR] Ошибка импорта: {e}")
            if debug_any: traceback.print_exc()
            if not skip_errors: sys.exit(1)
        sys.exit(0)

    # === ЛОГИКА ОБРАБОТКИ ===
    if args.mode == 'all':
        print(f"\n[RUN] Режим: ALL. Обработка всех файлов из '{INPUT_DIR}'...")
        try:
            process_all(INPUT_DIR, OUTPUT_DIR, debug=debug_mode, trace=trace_mode,
                        trace_labeled=trace_labeled_mode, progress_every=10)
            print("[OK] Пакетная обработка завершена.")
        except Exception as e:
            print(f"[ERR] Критическая ошибка при пакетной обработке: {e}")
            if debug_any:
                traceback.print_exc()
            if not skip_errors:
                print("[STOP] Прерывание. Используйте --skip-errors для продолжения.")
                sys.exit(1)

    elif args.mode == 'test':
        print(f"\n[RUN] Режим: TEST. Обработка выбранных срезов...")
        test_slices = [
            'full_AGD_50_000077.tiff',
            'full_AGD_50_000087.tiff',
            'full_AGD_50_000097.tiff',
            'full_AGD_50_000107.tiff',
            'full_AGD_50_000117.tiff',
            'full_AGD_50_000127.tiff',
            'full_AGD_50_000137.tiff',
        ]

        panels = []
        failed = []
        for fname in test_slices:
            path = os.path.join(INPUT_DIR, fname)
            if not os.path.exists(path):
                print(f"[WARN] Файл не найден: {path}")
                continue

            print(f"[..] Processing {fname} (debug={debug_mode}, trace={trace_mode}, "
                  f"trace_labeled={trace_labeled_mode})...")
            try:
                sub_dir = None
                if debug_any:
                    sub_dir = os.path.join(OUTPUT_DIR, f'debug_{fname[:-5]}')
                r = process_slice(path, debug=debug_mode, output_dir=sub_dir,
                                  trace=trace_mode, trace_labeled=trace_labeled_mode)

                ov = make_overlay(r)
                cv2.putText(ov, f"si={r['si']}", (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panels.append(ov)

                cv2.imwrite(os.path.join(OUTPUT_DIR, f'overlay_si{r["si"]:03d}.png'), ov)
                print(f"[OK] {fname} обработан успешно.")

            except Exception as e:
                print(f"[ERR] Ошибка при обработке {fname}: {e}")
                if debug_any:
                    traceback.print_exc()
                failed.append(fname)
                if not skip_errors:
                    print("[STOP] Прерывание. Используйте --skip-errors для продолжения.")
                    sys.exit(1)
                continue

        # === Сборка сетки результатов ===
        if panels:
            if all(p.shape[0] == panels[0].shape[0] for p in panels):
                grid = cv2.hconcat(panels)
                out_path = os.path.join(OUTPUT_DIR, 'output.png')
                cv2.imwrite(out_path, grid)
                print(f"[OK] Сетка сохранена -> {out_path}")
            else:
                print("[WARN] Не удалось собрать сетку: изображения имеют разную высоту.")
        else:
            print("[INFO] Ни один файл не был успешно обработан.")

        if failed:
            print(f"\n[STAT] Итог: успешно {len(panels)}, ошибок {len(failed)}.")
            print("Проблемные файлы:", failed)