import os
import shutil


def recreate_folder(path):
    if os.path.exists(path):
        print(f"Tree of path {path} will be deleted.")
        shutil.rmtree(path)

    os.mkdir(path)

def create_if_not_exists(path):
    os.makedirs(path, exist_ok=True)