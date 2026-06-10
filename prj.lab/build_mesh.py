"""
build_mesh.py — строит 3D mesh из overlay_si***.png срезов.

Использование:
    python build_mesh.py                        # все срезы в INPUT_DIR
    python build_mesh.py 100 300               # только si 100..300 (быстро для теста)
    python build_mesh.py --step 2              # брать каждый 2-й срез (в 2x быстрее)
    python build_mesh.py 100 300 --step 2

Выходные файлы (в OUTPUT_DIR):
    shell.obj / shell.ply
    kernel.obj / kernel.ply
    partitions.obj / partitions.ply

Формат OBJ открывается в MeshLab, Blender, онлайн-просмотрщиках.
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
INPUT_DIR  = r'C:\masks_test\out'   # папка с overlay_si***.png
OUTPUT_DIR = r'C:\masks_test\mesh'  # куда сохранить .obj / .ply

# ── Масштаб вокселя (мкм или пикселей — не важно для формы) ──────────────────
# Если знаешь реальный шаг сканера — подставь сюда.
# Формат: (X_мкм, Y_мкм, Z_мкм) — влияет только на пропорции модели.
VOXEL_SPACING = (1.0, 1.0, 1.0)    # изотропный (1:1:1)

# ── Классы ────────────────────────────────────────────────────────────────────
LABEL_BG         = 0
LABEL_SHELL      = 1
LABEL_KERNEL     = 2
LABEL_PARTITIONS = 3

# ── Декодирование цветов ───────────────────────────────────────────────────────
# make_overlay из segment_walnut.py делает так (всё в BGR):
#   gl[shell]      = (0, 180,   0)  → blend 50%/50% с серым
#   rl[kernel]     = (0,   0, 210)  → добавляет 45%
#   yl[partitions] = (0, 200, 200)  → добавляет 45%
#
# После blending на типичном сером (80..180):
#   SHELL:      G >> R, G >> B  (зелёный)
#   KERNEL:     R >> G, R >> B  (красный)
#   PARTITIONS: G ≈ R >> B, оба большие  (жёлтый/cyan)
#   BG:         R ≈ G ≈ B  (серый или чёрный)

def decode_labels(img_bgr: np.ndarray) -> np.ndarray:
    """
    BGR image → 2D uint8 массив меток.
    Метки: 0=фон, 1=shell, 2=kernel, 3=partitions.
    """
    b = img_bgr[:, :, 0].astype(np.int16)
    g = img_bgr[:, :, 1].astype(np.int16)
    r = img_bgr[:, :, 2].astype(np.int16)

    # Пиксель считается частью ореха если хоть один канал > порога
    ACTIVE = 40

    # SHELL: G явно доминирует над R и B, B ≈ R (симметрично)
    is_shell = (
        (g - r > 35) &
        (g - b > 35) &
        (np.abs(r - b) < 40) &
        (g > ACTIVE)
    )

    # KERNEL: R явно доминирует над G и B
    is_kernel = (
        (r - g > 30) &
        (r - b > 30) &
        (r > ACTIVE)
    )

    # PARTITIONS: G ≈ R, оба большие, B низкий
    is_partitions = (
        (g > ACTIVE) &
        (r > ACTIVE) &
        (np.abs(g - r) < 45) &
        (g - b > 30) &
        (~is_shell) &   # убираем пересечения
        (~is_kernel)
    )

    labels = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    labels[is_shell]      = LABEL_SHELL
    labels[is_kernel]     = LABEL_KERNEL
    labels[is_partitions] = LABEL_PARTITIONS

    return labels


def si_from_filename(fname: str):
    """Извлекает номер si из имени файла overlay_si***.png."""
    m = re.search(r'si(\d+)', fname, re.IGNORECASE)
    return int(m.group(1)) if m else None


def load_slices(input_dir: str, si_lo: int, si_hi: int, step: int):
    """
    Читает PNG срезы из папки, возвращает 3D массив меток.
    Ось 0 = Z (si), ось 1 = Y (строки), ось 2 = X (столбцы).
    """
    all_files = sorted(
        f for f in os.listdir(input_dir)
        if re.search(r'overlay_si\d+\.png$', f, re.IGNORECASE)
    )

    # Фильтруем по диапазону и шагу
    selected = []
    for f in all_files:
        si = si_from_filename(f)
        if si is None:
            continue
        if si_lo <= si <= si_hi and (si - si_lo) % step == 0:
            selected.append((si, f))

    if not selected:
        raise RuntimeError(f'Нет файлов overlay_si*.png в диапазоне {si_lo}..{si_hi} в {input_dir!r}')

    print(f'[load] Срезов для загрузки: {len(selected)}  (si {selected[0][0]}..{selected[-1][0]}, шаг={step})')

    # Определяем размер по первому изображению
    first_path = os.path.join(input_dir, selected[0][1])
    first_img  = cv2.imread(first_path)
    if first_img is None:
        raise RuntimeError(f'Не могу прочитать {first_path!r}')
    H, W = first_img.shape[:2]

    volume = np.zeros((len(selected), H, W), dtype=np.uint8)

    t0 = time.time()
    for z, (si, fname) in enumerate(selected):
        path = os.path.join(input_dir, fname)
        img  = cv2.imread(path)
        if img is None:
            print(f'  [!] Пропускаю {fname} — не читается')
            continue
        volume[z] = decode_labels(img)

        if (z + 1) % 50 == 0 or z == len(selected) - 1:
            elapsed = time.time() - t0
            pct = (z + 1) / len(selected) * 100
            print(f'  [load] {z+1}/{len(selected)} ({pct:.0f}%)  {elapsed:.1f}s')

    print(f'[load] Объём: {volume.shape}  dtype={volume.dtype}')
    return volume, [si for si, _ in selected]


def build_mesh_for_class(volume: np.ndarray, label: int,
                         spacing: tuple, smooth_iter: int = 3):
    """
    Строит trimesh.Trimesh для одного класса меток.
    spacing — размер вокселя (z, y, x).
    """
    binary = (volume == label).astype(np.float32)

    # Небольшое морфологическое закрытие убирает дырки в поверхности
    # (делаем по срезам — быстрее, чем 3D операция)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for z in range(binary.shape[0]):
        binary[z] = cv2.morphologyEx(binary[z], cv2.MORPH_CLOSE, kernel)

    n_voxels = int(binary.sum())
    if n_voxels < 8:
        print(f'  [!] Класс {label}: слишком мало вокселей ({n_voxels}), пропускаю')
        return None

    print(f'  [mc] Marching Cubes для класса {label}  ({n_voxels:,} вокселей)...')
    t0 = time.time()

    verts, faces, normals, _ = marching_cubes(
        binary,
        level=0.5,
        spacing=spacing,   # (z_step, y_step, x_step)
        allow_degenerate=False,
    )

    print(f'  [mc] вершин={len(verts):,}  треугольников={len(faces):,}  '
          f'за {time.time()-t0:.1f}s')

    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_normals=normals, process=False)

    # Сглаживание (Laplacian)
    if smooth_iter > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iter)

    return mesh


def save_mesh(mesh: trimesh.Trimesh, base_path: str):
    """Сохраняет mesh в .obj и .ply."""
    for ext in ('.obj', '.ply'):
        path = base_path + ext
        mesh.export(path)
        size_kb = os.path.getsize(path) / 1024
        print(f'  [save] {path}  ({size_kb:.0f} KB)')


def print_decode_stats(volume: np.ndarray):
    """Выводит сводку по классам для контроля декодирования."""
    total = volume.size
    for label, name in [(0,'фон'), (1,'shell'), (2,'kernel'), (3,'partitions')]:
        n = int((volume == label).sum())
        print(f'  {name:12s}: {n:>12,}  ({n/total*100:.1f}%)')


def main():
    parser = argparse.ArgumentParser(description='Строит 3D mesh из overlay PNG срезов.')
    parser.add_argument('si_lo', nargs='?', type=int, default=0,
                        help='Начало диапазона si (по умолчанию 0)')
    parser.add_argument('si_hi', nargs='?', type=int, default=999,
                        help='Конец диапазона si (по умолчанию 999)')
    parser.add_argument('--step', type=int, default=1,
                        help='Брать каждый N-й срез (ускоряет тест, default=1)')
    parser.add_argument('--no-partitions', action='store_true',
                        help='Не строить mesh перегородок (самый маленький класс)')
    parser.add_argument('--smooth', type=int, default=3,
                        help='Итераций сглаживания Лапласа (default=3, 0=выкл)')
    parser.add_argument('--input',  default=INPUT_DIR,  help='Папка с PNG')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Папка для .obj/.ply')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print('=' * 60)
    print(f'INPUT  : {args.input}')
    print(f'OUTPUT : {args.output}')
    print(f'Диапазон si: {args.si_lo}..{args.si_hi}  шаг={args.step}')
    print('=' * 60)

    # 1. Загрузка и декодирование
    volume, si_list = load_slices(args.input, args.si_lo, args.si_hi, args.step)

    print('\n[decode] Статистика вокселей:')
    print_decode_stats(volume)

    # Реальный шаг по Z с учётом step (если берём каждый 2-й срез — Z вдвое шире)
    spacing = (VOXEL_SPACING[2] * args.step,   # z
               VOXEL_SPACING[1],                # y
               VOXEL_SPACING[0])                # x

    # 2. Строим mesh для каждого класса
    classes = [
        (LABEL_SHELL,      'shell'),
        (LABEL_KERNEL,     'kernel'),
    ]
    if not args.no_partitions:
        classes.append((LABEL_PARTITIONS, 'partitions'))

    for label, name in classes:
        print(f'\n[mesh] {name.upper()} ...')
        mesh = build_mesh_for_class(volume, label,
                                    spacing=spacing,
                                    smooth_iter=args.smooth)
        if mesh is not None:
            save_mesh(mesh, os.path.join(args.output, name))

    print('\n[done] Готово. Открой .obj файлы в MeshLab или Blender.')
    print(f'       Папка: {args.output}')


if __name__ == '__main__':
    main()
