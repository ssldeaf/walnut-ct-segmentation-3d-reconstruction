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
from collections import deque
from skimage.filters import frangi


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