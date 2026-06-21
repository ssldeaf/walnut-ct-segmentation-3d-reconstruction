# segm.py — список команд

## CLI (через `__main__`)

Пути зашиты в начале блока `if __name__ == '__main__'`:
`INPUT_DIR = C:\masks_test`, `OUTPUT_DIR = C:\masks_test\out`.
Поменяй их под свои данные. Аргументы: числа задают диапазон `si`,
слово `debug` включает сохранение промежуточных шагов.

| Команда | Что делает |
|---|---|
| `python segm.py` | Обработать ВСЕ `.tiff/.tif` в `INPUT_DIR` (режим `all`) |
| `python segm.py 183` | Только срез `si = 183` |
| `python segm.py 70 100` | Диапазон `si` от 70 до 100 включительно |
| `python segm.py 400 432` | Диапазон `si` 400–432 |
| `python segm.py debug` | Все срезы + debug-шаги в `debug_<имя>/` |
| `python segm.py 183 debug` | Один срез `si=183` + debug-шаги |
| `python segm.py 70 100 debug` | Диапазон + debug-шаги |

Результат: `overlay_si###.png` в `OUTPUT_DIR`; при `debug` — папка
`debug_<имя файла>/` с пронумерованными шагами `NNN_<стадия>__<шаг>_si###.png`.

## Python API (произвольные пути, в т.ч. папка etalon)

```python
import segm, cv2

path = r'C:\Users\...\etalon_masks\original_images\full_AGD_50_000183.tiff'

# 1. Один срез без debug (продакшен; segm_debug не подгружается)
r = segm.process_slice(path)
#   r = {'img','bl','si','mask','shell','interior','kernel','partitions'}

# 2. Один срез с debug-шагами в конкретную папку
r = segm.process_slice(path, debug=True, output_dir=r'C:\out\dbg_183')

# 3. Цветной overlay (зелёный=скорлупа, красный=ядро, жёлтый=перегородки)
cv2.imwrite(r'C:\out\overlay_183.png', segm.make_overlay(r))

# 4. Сохранить отдельные маски
for k in ('mask','shell','interior','kernel','partitions'):
    cv2.imwrite(rf'C:\out\{k}_183.png', r[k])

# 5. Вся папка разом
segm.process_all(r'C:\masks_test', r'C:\masks_test\out', debug=False)
segm.process_all(r'C:\masks_test', r'C:\masks_test\out', debug=True)  # + debug
```

## Проверка/обслуживание

```bash
python -m py_compile segm.py segm_debug.py   # синтаксис обоих модулей
```

## Где теперь debug-код

- `segm.py` — только алгоритм; вызывает no-op `Reporter` (`NULL`).
- `segm_debug.py` — `DebugReporter` + вся `cv2`-отрисовка; импортируется
  лениво, лишь когда `process_slice(debug=True)`.
