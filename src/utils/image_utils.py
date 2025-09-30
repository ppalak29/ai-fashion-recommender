from pathlib import Path
import os

def get_image_files(folder_path, extensions=('.jpg', '.jpeg', '.png')):
    image_files = []
    for ext in extensions:
        image_files.extend(Path(folder_path).glob(f'*{ext}'))
        image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
    return [str(f) for f in image_files]

def validate_image_path(image_path):
    return os.path.exists(image_path) and Path(image_path).suffix.lower() in ['.jpg', '.jpeg', '.png']