"""
build_mesh.py — строит 3D mesh из overlay_si***.png срезов.

Использование:
    python build_mesh.py                        # все срезы в INPUT_DIR
    python build_mesh.py 100 300               # только si 100..300
    python build_mesh.py --step 2              # каждый 2-й срез
    python build_mesh.py 100 300 --step 2

Выходные файлы (в OUTPUT_DIR):
    shell.obj / shell.ply
    kernel.obj / kernel.ply
    partitions.obj / partitions.ply
"""

import os
import sys
import re
import time
import argparse
import numpy as np
import cv2
from skimage.measure import marching_cubes
import trimesh

# ── Пути ──────────────────────────────────────────────────────────────────────
INPUT_DIR  = r'C:\masks_test\out'
OUTPUT_DIR = r'C:\masks_test\mesh'

# ── Масштаб вокселя ────────────────────────────────────────────────────────────
VOXEL_SPACING = (1.0, 1.0, 1.0)

# ── Классы ────────────────────────────────────────────────────────────────────
LABEL_BG         = 0
LABEL_SHELL      = 1
LABEL_KERNEL     = 2
LABEL_PARTITIONS = 3

# ── Параметры очистки ─────────────────────────────────────────────────────────
# Минимальная площадь компоненты на срезе (в пикселях).
# Компоненты меньше этого значения удаляются как шум.
MIN_COMPONENT_AREA = 100

# Минимальный размер 3D компоненты (в вокселях).
# Изолированные облака меньше этого значения удаляются из финального volume.
MIN_3D_COMPONENT_VOXELS = 500

# Отступ от края изображения (в пикселях).
# Пиксели ближе этого расстояния к краю считаются обрезанными и удаляются.
EDGE_MARGIN = 5


def decode_labels(img_bgr: np.ndarray) -> np.ndarray:
    """BGR image → 2D uint8 массив меток."""
    b = img_bgr[:, :, 0].astype(np.int16)
    g = img_bgr[:, :, 1].astype(np.int16)
    r = img_bgr[:, :, 2].astype(np.int16)

    ACTIVE = 40

    is_shell = (
        (g - r > 35) &
        (g - b > 35) &
        (np.abs(r - b) < 40) &
        (g > ACTIVE)
    )
    is_kernel = (
        (r - g > 30) &
        (r - b > 30) &
        (r > ACTIVE)
    )
    is_partitions = (
        (g > ACTIVE) &
        (r > ACTIVE) &
        (np.abs(g - r) < 45) &
        (g - b > 30) &
        (~is_shell) &
        (~is_kernel)
    )

    labels = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    labels[is_shell]      = LABEL_SHELL
    labels[is_kernel]     = LABEL_KERNEL
    labels[is_partitions] = LABEL_PARTITIONS
    return labels


def clean_slice_2d(binary: np.ndarray, min_area: int, edge_margin: int) -> np.ndarray:
    """
    Очищает бинарный срез:
    1. Удаляет маленькие изолированные компоненты (шум → артефакты в 3D).
    2. Удаляет пиксели у края кадра (обрезанные объекты → шипы вниз/вверх).
    3. Оставляет только наибольшую связную компоненту.
    """
    binary = binary.astype(np.uint8)
    H, W = binary.shape

    # Убираем пиксели у края
    if edge_margin > 0:
        mask = np.ones_like(binary)
        mask[:edge_margin, :]  = 0
        mask[-edge_margin:, :] = 0
        mask[:, :edge_margin]  = 0
        mask[:, -edge_margin:] = 0
        binary = binary & mask

    if binary.sum() == 0:
        return binary

    # Связные компоненты
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels <= 1:
        return binary

    # Убираем компоненты меньше min_area
    result = np.zeros_like(binary)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            result[labels == lbl] = 1

    return result


def remove_small_3d_components(volume: np.ndarray, label: int,
                                min_voxels: int) -> np.ndarray:
    """
    Убирает из volume изолированные 3D-компоненты меньше min_voxels вокселей.
    Работает непосредственно с volume (заменяет метку на 0).
    """
    binary = (volume == label).astype(np.uint8)

    # cv2 работает только в 2D — используем scipy для 3D связности
    from scipy import ndimage
    labeled_3d, n = ndimage.label(binary)
    print(f"  [3D clean] класс {label}: {n} компонент в 3D")

    if n <= 1:
        return volume

    sizes = ndimage.sum(binary, labeled_3d, range(1, n + 1))
    main_label = np.argmax(sizes) + 1
    main_size  = int(sizes[main_label - 1])
    print(f"  [3D clean] главная компонента: {main_size:,} вокселей")

    removed = 0
    for i, size in enumerate(sizes):
        lbl = i + 1
        if lbl == main_label:
            continue
        if int(size) < min_voxels:
            volume[labeled_3d == lbl] = LABEL_BG
            removed += int(size)

    print(f"  [3D clean] удалено вокселей: {removed:,}")
    return volume


def si_from_filename(fname: str):
    m = re.search(r'si(\d+)', fname, re.IGNORECASE)
    return int(m.group(1)) if m else None


def load_slices(input_dir: str, si_lo: int, si_hi: int, step: int):
    """Читает PNG срезы, возвращает очищенный 3D массив меток."""
    all_files = sorted(
        f for f in os.listdir(input_dir)
        if re.search(r'overlay_si\d+\.png$', f, re.IGNORECASE)
    )

    selected = []
    for f in all_files:
        si = si_from_filename(f)
        if si is None:
            continue
        if si_lo <= si <= si_hi and (si - si_lo) % step == 0:
            selected.append((si, f))

    if not selected:
        raise RuntimeError(f'Нет файлов в диапазоне {si_lo}..{si_hi} в {input_dir!r}')

    print(f'[load] Срезов: {len(selected)}  (si {selected[0][0]}..{selected[-1][0]}, шаг={step})')

    first_img = cv2.imread(os.path.join(input_dir, selected[0][1]))
    if first_img is None:
        raise RuntimeError(f'Не могу прочитать {selected[0][1]!r}')
    H, W = first_img.shape[:2]

    volume = np.zeros((len(selected), H, W), dtype=np.uint8)

    t0 = time.time()
    for z, (si, fname) in enumerate(selected):
        img = cv2.imread(os.path.join(input_dir, fname))
        if img is None:
            print(f'  [!] Пропускаю {fname}')
            continue

        labels = decode_labels(img)

        # Очищаем каждый класс на 2D срезе
        for cls in [LABEL_SHELL, LABEL_KERNEL, LABEL_PARTITIONS]:
            binary = (labels == cls).astype(np.uint8)
            cleaned = clean_slice_2d(binary, MIN_COMPONENT_AREA, EDGE_MARGIN)
            # Там где очистка убрала пиксели — ставим фон
            labels[(labels == cls) & (cleaned == 0)] = LABEL_BG

        volume[z] = labels

        if (z + 1) % 50 == 0 or z == len(selected) - 1:
            elapsed = time.time() - t0
            pct = (z + 1) / len(selected) * 100
            print(f'  [load] {z+1}/{len(selected)} ({pct:.0f}%)  {elapsed:.1f}s')

    print(f'[load] Объём: {volume.shape}')

    # 3D очистка — убираем плавающие фрагменты
    print('\n[3D clean] Удаление изолированных 3D-компонент...')
    for cls in [LABEL_SHELL, LABEL_KERNEL, LABEL_PARTITIONS]:
        if (volume == cls).sum() > 0:
            volume = remove_small_3d_components(volume, cls, MIN_3D_COMPONENT_VOXELS)

    return volume, [si for si, _ in selected]


def build_mesh_for_class(volume: np.ndarray, label: int,
                          spacing: tuple, smooth_iter: int = 0):
    """Строит trimesh.Trimesh для одного класса меток."""
    binary = (volume == label).astype(np.float32)

    # Морфологическое закрытие + заполнение дырок.
    # FloodFill только для shell/kernel (solid объекты).
    # Для partitions — НЕ применяем: перегородки образуют замкнутый
    # контур вокруг ореха, и заполнение создаёт огромный диск внутри,
    # внутренняя поверхность которого выглядит как паразитное кольцо.
    is_solid = label in (LABEL_SHELL, LABEL_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for z in range(binary.shape[0]):
        sl = binary[z].astype(np.uint8)
        sl = cv2.morphologyEx(sl, cv2.MORPH_CLOSE, kernel)
        if is_solid:
            flooded = sl.copy()
            h, w = flooded.shape
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flooded, flood_mask, (0, 0), 1)
            sl = (sl | (flooded == 0)).astype(np.uint8)
        binary[z] = sl.astype(np.float32)

    n_voxels = int(binary.sum())
    if n_voxels < 8:
        print(f'  [!] Класс {label}: слишком мало вокселей ({n_voxels}), пропускаю')
        return None

    # Gaussian blur отключён — нужен low-poly вид с видимыми полигонами.
    # Раскомментируйте если нужно вернуть сглаживание:
    # from scipy.ndimage import gaussian_filter
    # binary = gaussian_filter(binary, sigma=1.0)

    print(f'  [mc] Marching Cubes класс {label}  ({n_voxels:,} вокселей)...')
    t0 = time.time()

    verts, faces, normals, _ = marching_cubes(
        binary,
        level=0.5,
        spacing=spacing,
        allow_degenerate=False,
    )
    print(f'  [mc] вершин={len(verts):,}  треугольников={len(faces):,}  за {time.time()-t0:.1f}s')

    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_normals=normals, process=False)

    # Убираем вырожденные грани
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())

    # Сглаживание Лапласа
    if smooth_iter > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iter)

    # Убираем мусорные микро-компоненты, но сохраняем все значимые доли
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        max_faces = max(len(m.faces) for m in components)
        # Порог: компонента значима если > 1% граней от наибольшей
        threshold = max(100, max_faces * 0.01)
        kept    = [m for m in components if len(m.faces) >= threshold]
        removed = [m for m in components if len(m.faces) < threshold]
        removed_verts = sum(len(m.vertices) for m in removed)
        print(f'  [mesh clean] {len(components)} компонент → '
              f'оставляем {len(kept)} (порог {threshold:.0f} граней), '
              f'удалено {len(removed)} мусорных (~{removed_verts:,} вершин)')
        if kept:
            mesh = trimesh.util.concatenate(kept)

    return mesh


def save_mesh(mesh: trimesh.Trimesh, base_path: str):
    for ext in ('.obj', '.ply'):
        path = base_path + ext
        mesh.export(path)
        size_kb = os.path.getsize(path) / 1024
        print(f'  [save] {path}  ({size_kb:.0f} KB)')


def print_decode_stats(volume: np.ndarray):
    total = volume.size
    for label, name in [(0,'фон'), (1,'shell'), (2,'kernel'), (3,'partitions')]:
        n = int((volume == label).sum())
        print(f'  {name:12s}: {n:>12,}  ({n/total*100:.1f}%)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('si_lo', nargs='?', type=int, default=0)
    parser.add_argument('si_hi', nargs='?', type=int, default=999)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--no-partitions', action='store_true')
    parser.add_argument('--smooth', type=int, default=0)
    parser.add_argument('--input',  default=INPUT_DIR)
    parser.add_argument('--output', default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print('=' * 60)
    print(f'INPUT  : {args.input}')
    print(f'OUTPUT : {args.output}')
    print(f'Диапазон si: {args.si_lo}..{args.si_hi}  шаг={args.step}')
    print(f'MIN_COMPONENT_AREA      : {MIN_COMPONENT_AREA} пикс')
    print(f'MIN_3D_COMPONENT_VOXELS : {MIN_3D_COMPONENT_VOXELS} вокс')
    print(f'EDGE_MARGIN             : {EDGE_MARGIN} пикс')
    print('=' * 60)

    volume, si_list = load_slices(args.input, args.si_lo, args.si_hi, args.step)

    print('\n[decode] Статистика вокселей после очистки:')
    print_decode_stats(volume)

    spacing = (VOXEL_SPACING[2] * args.step,
               VOXEL_SPACING[1],
               VOXEL_SPACING[0])

    classes = [(LABEL_SHELL, 'shell'), (LABEL_KERNEL, 'kernel')]
    if not args.no_partitions:
        classes.append((LABEL_PARTITIONS, 'partitions'))

    for label, name in classes:
        print(f'\n[mesh] {name.upper()} ...')
        mesh = build_mesh_for_class(volume, label,
                                    spacing=spacing,
                                    smooth_iter=args.smooth)
        if mesh is not None:
            save_mesh(mesh, os.path.join(args.output, name))

    print('\n[done] Готово.')
    print(f'       Папка: {args.output}')


if __name__ == '__main__':
    main()
