import os
from PIL import Image


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
