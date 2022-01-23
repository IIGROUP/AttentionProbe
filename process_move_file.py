import os
import shutil
from os import rmdir


def move_files(root):
    folders = os.listdir(root)
    folders.sort()
    for folder in folders:
        images = os.listdir(os.path.join(root, folder, "images"))
        for i in images:
            print(f"processing folder {folder} and image {i} ...")
            src_path = os.path.join(root, folder, "images", i)
            dst_path = os.path.join(root, folder, i)
            shutil.move(src_path, dst_path)

# move_files('/database/tiny-imagenet/val/')
# move_files('/database/tiny-imagenet/train/')

def remove_files(root):
    folders = os.listdir(root)
    folders.sort()
    for folder in folders:
        src_path = os.path.join(root, folder, "images")
        rmdir(src_path)

# remove_files('/database/tiny-imagenet/val/')
remove_files('/database/tiny-imagenet/train/')