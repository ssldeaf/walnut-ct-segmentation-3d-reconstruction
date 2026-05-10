import argparse
from volume_slicer import VolumeSlicer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Просмотр срезов 3D-объёма из TIFF 32-bit"
    )

    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Папка с TIFF-срезами"
    )

    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Префикс имени файла"
    )

    parser.add_argument(
        "--num-slices",
        type=int,
        required=True,
        help="Количество срезов (ось Z)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["x", "y"],
        required=True,
        help="Тип среза: x, y"
    )

    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Индекс среза"
    )

    return parser.parse_args() 

if __name__ == "__main__":
    args = parse_arguments()

    slicer = VolumeSlicer(folder=args.folder, file_name=args.prefix,num_slices=args.num_slices)

    if args.mode == "x":
        img = slicer.get_vertical_slice_x(args.index)
        title = f"X slice {args.index}"

    elif args.mode == "y":
        img = slicer.get_vertical_slice_y(args.index)
        title = f"Y slice {args.index}"

    slicer.show(img, title=title)
