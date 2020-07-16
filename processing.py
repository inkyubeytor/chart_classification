import os
import shutil
from PIL import Image


def _create_tmp() -> None:
    """
    If no temporary folder exists, create one.
    :return: None.
    """
    try:
        os.mkdir("tmp")
    except FileExistsError:
        pass


def copy_to_tmp(fp: str) -> str:
    """
    Copies an image to temporary folder.
    :param fp: The image to copy.
    :return: A path to the new image.
    """
    new_path = f"tmp/{os.path.basename(fp)}"
    try:
        shutil.copyfile(fp, new_path)
    except FileNotFoundError:
        _create_tmp()
        shutil.copyfile(fp, new_path)
    return new_path


def convert_to_png(fp: str) -> str:
    """
    Destructively converts an image to a PNG.
    :param fp: The path to the image to convert.
    :return: The path to the converted image.
    """
    new_fp = f"{'.'.join(fp.split('.')[:-1])}.png"

    img = Image.open(fp)
    img.load()  # Force loading of the image into memory
    img.save(new_fp, format="png")
    img.close()

    os.remove(fp)

    return new_fp
