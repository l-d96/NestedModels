from pathlib import Path
from typing import Optional, Callable
import importlib.util
import os
import sys

from . import default_transformations as dt


def load_custom_transformations(custom_dir: Optional[Path | str] = None) -> dict[str, Callable]:
    if custom_dir is None:
        custom_dir = (Path(os.getenv('XDG_DATA_HOME')) /
                      'nestedmodels')

    custom_dir = custom_dir.expanduser().absolute()
    if not custom_dir.exists():
        custom_dir.mkdir(parents=True)

    if not custom_dir.is_dir():
        raise Exception(
            f"The path for custom transformations is not a directory: '{custom_dir}'")

    module_paths = sorted(custom_dir.iterdir())

    loaded_modules = [dt]
    for mp in module_paths:
        if not mp.is_file():
            continue
        module_name = mp.stem
        spec = importlib.util.spec_from_file_location(module_name, mp)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        loaded_modules.append(module)

    transformations = {name: getattr(
        module, name) for module in loaded_modules for name in module.__all__}

    return transformations
