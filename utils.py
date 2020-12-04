import os
import numpy as np
from pathlib import Path

def create_path(path):
    """
    check if dir exist, otherwise create new dir
    Args:
        path: the absolute path to be created. If already exists, don't do anything

    Returns:

    """
    Path(path).mkdir(parents=True, exist_ok=True)