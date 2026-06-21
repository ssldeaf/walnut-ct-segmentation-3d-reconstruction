# Векторизация КТ-данных грецкого ореха

Сегментация КТ-срезов грецкого ореха (датасет Walnut2, 501 срез TIFF) и построение трёхмерных моделей анатомических классов: **скорлупа**, **ядро** и **перегородки**.

Пайплайн состоит из двух этапов:

1. **Сегментация** (`prj.lab/segm.py`) — для каждого среза строятся три бинарные маски и контрольный цветной overlay. Стадии: A — маска ореха, B — скорлупа и внутренность, C — ядро (region growing + фильтры формы), D — перегородки (Frangi + кольцо вокруг ядра + зона скорлупы).
2. **Векторизация** (`prj.lab/mesh_builder/build_mesh.py`) — стопка overlay-срезов собирается в воксельный объём, из которого методом Marching Cubes строятся три полигональные поверхности (OBJ + PLY).

Параметры, чувствительные к номеру среза, вынесены в словарь `PARAM_ANCHORS` и интерполируются по `si`; структурные параметры заданы константами. Подробности — в отчёте `notes/Отчет_Векторизация_КТ_v5_Леликов.pdf`.

## Структура репозитория

```
misis2026s-23-03-lelikov-s-s/
├── etalon_masks/
│   ├── original_images/             # эталонные КТ-срезы, TIFF (14 шт., каждый 20-й)
│   └── segmentation_masks/          # эталонные маски классов (TIF) для оценки Dice/IoU
├── notes/
│   ├── segmentation_metrics.csv     # метрики Dice/IoU по срезам
│   └── Отчет_Векторизация_КТ_v5_Леликов.pdf   # отчёт по работе
├── prj.lab/
│   ├── __init__.py
│   ├── segm.py                      # пайплайн сегментации срезов (стадии A–D)
│   ├── commands.md                  # список команд запуска segm.py
│   ├── mesh_builder/
│   │   ├── build_mesh.py            # построение 3D-мешей (Marching Cubes)
│   │   └── input.rar               # overlay-срезы — вход построения мешей
│   ├── utilities/
│   │   ├── area_check.py            # контроль площадей классов по объёму
│   │   ├── evaluate.py              # подсчёт Dice / IoU по эталонам
│   │   ├── recolor_mask.py          # ручная перекраска масок (перегородки/ядро)
│   │   └── segment_from_masks.py    # сборка бинарных масок в overlay
│   ├── volume_slicer/
│   │   ├── __init__.py
│   │   └── vs.py                    # нарезка/сборка объёма, xz/yz-проекции
│   └── results/
│       ├── segmentation_results/    # overlay_si###.png — результат сегментации (501 шт.)
│       ├── mesh_builder_results/    # shell/kernel/partitions — .obj + .ply
│       └── volume_slicer_results/   # ортогональные проекции (xz_y###.png, yz_x###.png)
└── .gitignore
```

> `segm.py --mode debug` дополнительно создаёт папки `debug_<имя>/` с пронумерованными промежуточными шагами каждого среза.

## Воспроизведение

```bash
# 1. Сегментация всех срезов в overlay (пути INPUT_DIR/OUTPUT_DIR заданы в __main__ segm.py)
python prj.lab/segm.py

# Один срез / диапазон / debug-шаги
python prj.lab/segm.py 183            # только si = 183
python prj.lab/segm.py 70 100         # диапазон si 70..100
python prj.lab/segm.py 183 debug      # один срез + промежуточные шаги

# 2. Контроль качества
python prj.lab/utilities/evaluate.py      # Dice / IoU по эталонам
python prj.lab/utilities/area_check.py    # площади классов по объёму

# 3. Ортогональные проекции (диагностика) и построение мешей
python prj.lab/volume_slicer/vs.py
python prj.lab/mesh_builder/build_mesh.py 45 476 --step 3
```

> Маски ранних (≈67–100) и поздних (≈400–433) срезов размечены вручную: на них ткань ореха выражена слабо, и автоматическая сегментация ненадёжна.

## Зависимости

Python 3.10+, `opencv-python`, `numpy`, `tifffile`, `scikit-image` (Frangi-фильтр); для построения и записи мешей — `scikit-image` (Marching Cubes) и `trimesh`.
