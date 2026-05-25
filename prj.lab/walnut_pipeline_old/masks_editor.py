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
from skimage.filters import frangi
from image_tracer import ImageTracer

class MaskEditor:
    """
    Ручное редактирование масок: экспортируем их в виде PNG, правим в любом
    графическом редакторе, импортируем обратно. См. функции export_masks() и
    import_masks() ниже.
    """
    pass

    # ── Ручное редактирование масок: export / import ───────────────────────────

    def _mask_subdir(masks_root, si):
        """Подпапка для одного среза: <masks_root>/si<NNN>/."""
        return os.path.join(masks_root, f'si{si:03d}')


    def export_masks(r, masks_root):
        """
        Сохранить три маски (shell, kernel, partitions) и исходный срез
        в подпапку masks_root/si<NNN>/. После этого пользователь может
        вручную править маски в любом графическом редакторе.
        Возвращает путь к созданной подпапке.
        """
        folder = _mask_subdir(masks_root, r['si'])
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, 'img.png'),        r['img'])
        cv2.imwrite(os.path.join(folder, 'shell.png'),      r['shell'])
        cv2.imwrite(os.path.join(folder, 'kernel.png'),     r['kernel'])
        cv2.imwrite(os.path.join(folder, 'partitions.png'), r['partitions'])
        return folder


    def _read_mask_file(path, shape):
        """Прочитать маску с диска и бинаризовать (0/255 uint8).
        Если файла нет — вернуть пустую маску нужной формы."""
        if not os.path.exists(path):
            return np.zeros(shape, dtype=np.uint8)
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return np.zeros(shape, dtype=np.uint8)
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        # Бинаризация (порог 127) — на случай если правили мягкой кистью
        return ((m >= 127).astype(np.uint8)) * 255


    def import_masks(masks_root, si):
        """
        Прочитать (возможно отредактированные) маски из masks_root/si<NNN>/.
        Возвращает dict в формате, совместимом с make_overlay():
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
        return {
            'img':        img,
            'bl':         img,   # make_overlay использует только img
            'si':         si,
            'mask':       None,
            'shell':      _read_mask_file(os.path.join(folder, 'shell.png'),      shape),
            'interior':   None,
            'kernel':     _read_mask_file(os.path.join(folder, 'kernel.png'),     shape),
            'partitions': _read_mask_file(os.path.join(folder, 'partitions.png'), shape),
        }


    def process_all_export(input_dir, masks_root, progress_every=50):
        """
        Фаза 1 для ручного редактирования.
        Прогоняет пайплайн по всем TIFF в input_dir, для каждого среза
        сохраняет в masks_root/si<NNN>/ четыре файла:
            img.png, shell.png, kernel.png, partitions.png
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
        print(f'[EXPORT] Отредактируйте их и запустите --import-masks для финализации.')


    def process_all_import(masks_root, output_dir, progress_every=50):
        """
        Фаза 2: после ручной правки.
        Идёт по подпапкам masks_root/si<NNN>/, склеивает финальный overlay
        из (возможно изменённых) масок и сохраняет его в output_dir.
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
