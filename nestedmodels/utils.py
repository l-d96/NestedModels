from .config import Config
from pathlib import Path
from tomlkit import load
import json


def read_toml(file: Path | str) -> Config:
    try:
        with open(file, "r") as f:
            toml = load(f)
            config = Config.from_toml(toml)
            return config
    except FileNotFoundError as e:
        print("error:", e)
        return


def read_json(file: Path | str) -> Config:
    try:
        with open(file, "r") as f:
            json_f = json.load(f)
            config = Config.from_json(json_f)
            return config
    except FileNotFoundError as e:
        print("error:", e)
        return
