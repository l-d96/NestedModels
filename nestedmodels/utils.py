from pathlib import Path
from typing import Optional, Callable
import importlib.util
import os
import sys
import inspect

from . import default_transformations as dt


# from https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
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
        module = import_module(mp)
        if module is None:
            continue
        loaded_modules.append(module)

    transformations = {name: getattr(
        module, name) for module in loaded_modules for name in module.__all__}

    # validation step
    for k, f in transformations.items():
        validate_transformation_args(f)

    return transformations


def import_module(mp: Path | str):
    mp = Path(mp)
    if not mp.is_file():
        return None
    module_name = mp.stem
    spec = importlib.util.spec_from_file_location(module_name, mp)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def validate_transformation_args(function: Callable) -> None:
    # it must contain 'input_'
    signature = inspect.signature(function)
    mandatory_args = [k for k, v in
                      signature.parameters.items()
                      if v.default is inspect.Parameter.empty]
    if 'input_' not in mandatory_args:
        raise Exception(
            f"transformation '{function.__name__}' doesn't contain mandatory paramater 'input_'")
    return True
