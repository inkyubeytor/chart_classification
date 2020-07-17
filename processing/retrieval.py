import os
import shutil


def copy_to_store(fp: str) -> str:
    """
    Copies an image to data folder.
    :param fp: The image to copy.
    :return: A path to the new image.
    """
    new_path = f"data/images/{os.path.basename(fp)}"
    shutil.copyfile(fp, new_path)
    return new_path
