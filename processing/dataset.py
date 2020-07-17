import numpy as np
from PIL import Image


def load_image_array(fp: str) -> np.ndarray:
    """
    Converts an image on disk to a numpy array.
    :param fp: The image to convert.
    :return: The image data as a numpy array.
    """
    img = Image.open(fp)
    arr = np.array(img)
    img.close()
    return arr
