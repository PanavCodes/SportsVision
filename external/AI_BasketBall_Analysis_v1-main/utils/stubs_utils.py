"""
stubs_utils.py

Utility functions for caching and loading intermediate results using pickle.

Useful for avoiding recomputation during object tracking or preprocessing.
"""

import os
import pickle
from typing import Any, Optional


def save_stub(stub_path: str, obj: Any) -> None:
    """
    Save a Python object to a pickle file.
    """
    if not stub_path:
        return
    dir_name = os.path.dirname(stub_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    with open(stub_path, 'wb') as f:
        pickle.dump(obj, f)


def read_stub(read_from_stub: bool, stub_path: Optional[str]) -> Optional[Any]:
    """
    Load a pickled object from a file, if allowed and file exists.
    """
    if read_from_stub and stub_path and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            return pickle.load(f)
    return None
