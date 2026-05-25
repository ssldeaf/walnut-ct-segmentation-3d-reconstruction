"""
Редактор overlay-масок для walnut_pipeline.

Принимает на вход путь к одному из overlay-файлов, созданных
walnut_pipeline.export_masks():
    .../masks/si<NNN>/overlay_shell.png
    .../masks/si<NNN>/overlay_kernel.png
    .../masks/si<NNN>/overlay_partitions.png

Серый фон берётся автоматически из img.png в той же папке.
По кнопке S сохранение пишет ПЕРЕРИСОВАННЫЙ overlay поверх исходного файла
(тем же цветом и в том же режиме, что и export_masks → import_masks
видит правки корректно).

Запуск:
    python editor.py path/to/masks/si087/overlay_kernel.png
"""

import os
import cv2
import sys
import numpy as np


# Цвета и режимы наложения — должны 1-в-1 совпадать с walnut_pipeline.
_LAYER_SPECS = {
    'shell':      {'color': (0, 180, 0),   'mode': 'blend', 'channel': 'green'},
    'kernel':     {'color': (0, 0, 210),   'mode': 'add',   'channel': 'red'},
    'partitions': {'color': (0, 200, 200), 'mode': 'add',   'channel': 'yellow'},
}

_PARSE_THRESHOLD = 25


def _detect_layer(overlay_path):
    """По имени файла определить тип слоя (shell/kernel/partitions)."""
    name = os.path.basename(overlay_path).lower()
    for layer in _LAYER_SPECS:
        if f'overlay_{layer}' in name:
            return layer
    raise ValueError(
        f"Не удалось определить тип слоя по имени '{name}'. "
        f"Ожидается overlay_shell / overlay_kernel / overlay_partitions.")


def _parse_overlay_channel(overlay_bgr, channel, threshold=_PARSE_THRESHOLD):
    """Бинарная маска из BGR overlay по «эксцессу» нужного канала.
    Та же логика, что walnut_pipeline._parse_overlay_channel."""
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


def _render_overlay(img_gray, mask, color_bgr, mode):
    """Полупрозрачное наложение одной маски на серый фон.
    Та же логика, что walnut_pipeline._overlay_single."""
    ov = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    layer = np.zeros_like(ov)
    layer[mask > 0] = color_bgr
    if mode == 'blend':
        return cv2.addWeighted(ov, 0.5, layer, 0.5, 0)
    else:
        # 0.55+0.45=1.0 — без клиппинга, парсер увидит цвет на любом фоне
        return cv2.addWeighted(ov, 0.55, layer, 0.45, 0)


def edit_overlay_mask(overlay_path, img_path=None):
    """
    Загрузить overlay-маску → редактировать → сохранить обратно.

    overlay_path — путь к overlay_{shell|kernel|partitions}.png.
    img_path     — путь к серому фону (по умолчанию img.png в той же папке).
    """
    layer = _detect_layer(overlay_path)
    spec = _LAYER_SPECS[layer]

    if img_path is None:
        img_path = os.path.join(os.path.dirname(overlay_path), 'img.png')
    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Не найден серый фон: {img_path}\n"
            f"Положите рядом img.png или передайте --img.")
    img8 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    overlay = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
    if overlay is None:
        raise FileNotFoundError(f"Не удалось прочитать overlay: {overlay_path}")
    if overlay.shape[:2] != img8.shape:
        overlay = cv2.resize(overlay, (img8.shape[1], img8.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    # Восстанавливаем бинарную маску из overlay (этим же делает import_masks)
    mask = _parse_overlay_channel(overlay, spec['channel'])

    h, w = img8.shape
    state = {
        'drawing': False,
        'panning': False,
        'brush_size': 3,
        'zoom': 1.0,
        'offset_x': 0.0,
        'offset_y': 0.0,
        'last_mouse': [0, 0],
        'dirty': False,     # есть несохранённые правки?
    }

    def screen_to_img(sx, sy):
        return int(sx / state['zoom'] + state['offset_x']), \
               int(sy / state['zoom'] + state['offset_y'])

    def make_display():
        vis = _render_overlay(img8, mask, spec['color'], spec['mode'])
        z = state['zoom']
        x1, y1 = int(state['offset_x']), int(state['offset_y'])
        x2 = min(w, int(state['offset_x'] + w / z))
        y2 = min(h, int(state['offset_y'] + h / z))
        x1, y1 = max(0, x1), max(0, y1)
        cropped = vis[y1:y2, x1:x2]
        out_w = int((x2 - x1) * z)
        out_h = int((y2 - y1) * z)
        if out_w <= 0 or out_h <= 0:
            return vis
        return cv2.resize(cropped, (out_w, out_h),
                          interpolation=cv2.INTER_NEAREST)

    def mouse_callback(event, x, y, flags, param):
        ix, iy = screen_to_img(x, y)
        z = state['zoom']
        if event == cv2.EVENT_LBUTTONDOWN:
            state['drawing'] = True
            r = max(1, int(state['brush_size'] / z))
            cv2.circle(mask, (ix, iy), r, 255, -1)
            state['dirty'] = True
        elif event == cv2.EVENT_LBUTTONUP:
            state['drawing'] = False
        elif event == cv2.EVENT_MBUTTONDOWN:
            state['panning'] = True
            state['last_mouse'] = [x, y]
        elif event == cv2.EVENT_MBUTTONUP:
            state['panning'] = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if state['drawing']:
                r = max(1, int(state['brush_size'] / z))
                cv2.circle(mask, (ix, iy), r, 255, -1)
                state['dirty'] = True
            if state['panning']:
                dx = (x - state['last_mouse'][0]) / z
                dy = (y - state['last_mouse'][1]) / z
                state['offset_x'] = max(0, min(w - w / z, state['offset_x'] - dx))
                state['offset_y'] = max(0, min(h - h / z, state['offset_y'] - dy))
                state['last_mouse'] = [x, y]
        elif event == cv2.EVENT_RBUTTONDOWN:
            r = max(1, int(state['brush_size'] / z))
            cv2.circle(mask, (ix, iy), r, 0, -1)
            state['dirty'] = True
        elif event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = z
            if flags > 0:
                state['zoom'] = min(10.0, z * 1.2)
            else:
                state['zoom'] = max(1.0, z / 1.2)
            z = state['zoom']
            state['offset_x'] += x / old_zoom - x / z
            state['offset_y'] += y / old_zoom - y / z
            state['offset_x'] = max(0, min(w - w / z, state['offset_x']))
            state['offset_y'] = max(0, min(h - h / z, state['offset_y']))

    title = f"Edit {layer} mask — {os.path.basename(overlay_path)}"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse_callback)
    print(f"Слой: {layer}    цвет BGR={spec['color']}    файл: {overlay_path}")
    print("ЛКМ — рисовать,  ПКМ — стирать")
    print("Колёсико — зум,  СКМ (зажать) — перемещение")
    print("[ / ]  — уменьшить / увеличить кисть")
    print("S — сохранить,  Q — выйти")

    while True:
        cv2.imshow(title, make_display())
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            new_overlay = _render_overlay(img8, mask, spec['color'], spec['mode'])
            cv2.imwrite(overlay_path, new_overlay)
            state['dirty'] = False
            print(f"Сохранено -> {overlay_path}")
        elif key == ord('q'):
            if state['dirty']:
                print("[WARN] Есть несохранённые правки. Нажмите S чтобы сохранить, "
                      "или Q ещё раз чтобы выйти без сохранения.")
                state['dirty'] = False  # следующее Q выйдет
                continue
            break
        elif key == ord('['):
            state['brush_size'] = max(1, state['brush_size'] - 1)
            print(f"Кисть: {state['brush_size']}")
        elif key == ord(']'):
            state['brush_size'] = min(100, state['brush_size'] + 1)
            print(f"Кисть: {state['brush_size']}")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description="Редактор overlay-масок walnut_pipeline.")
    ap.add_argument('overlay',
                    help="Путь к overlay_{shell|kernel|partitions}.png "
                         "(например masks/si087/overlay_kernel.png).")
    ap.add_argument('--img', default=None,
                    help="Путь к серому фону. По умолчанию — img.png "
                         "в той же папке, что и overlay.")
    args = ap.parse_args()
    edit_overlay_mask(args.overlay, args.img)
