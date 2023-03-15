from functools import singledispatch
from pathlib import Path, PosixPath, WindowsPath
from typing import Union

import yaml
import copy
from configs.tpg_config import TPGConfig

PathTypes = Union[str, Path, PosixPath, WindowsPath]


@singledispatch
def load_config(path: PathTypes) -> dict:
    raise NotImplementedError("Loading this path type not implemented.")


@load_config.register(Path)
@load_config.register(PosixPath)
@load_config.register(WindowsPath)
@load_config.register(str)
def local_yaml_loader(path: Union[Path, PosixPath, WindowsPath, str]) -> dict:
    """Loads a config YAML file from the local file system if the path is any of the following types: Path, PosixPath, WindowsPath, str.

    Args:
        path (Union[Path, PosixPath, WindowsPath, str]): Path to the local YAML file.

    Returns:
        dict: Loaded YAML file as a dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def config_from_yaml(path: PathTypes) -> TPGConfig:
    """Takes a yaml file and converts it to the pythonic TPGConfig object

    Args:
        path (PathTypes): Path to the config YAML file.

    Returns:
        TPGConfig: Loaded TPGConfig object
    """
    parsed_yaml = load_config(path)
    return TPGConfig(**parsed_yaml)
