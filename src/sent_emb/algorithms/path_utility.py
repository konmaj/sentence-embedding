import os
from pathlib import Path

env_var_str = 'RESOURCES_DIR'
assert env_var_str in os.environ
RESOURCES_DIR = Path(os.environ[env_var_str])

DATASETS_DIR = RESOURCES_DIR.joinpath('datasets')
EMBEDDINGS_DIR = RESOURCES_DIR.joinpath('embeddings')
OTHER_RESOURCES_DIR = RESOURCES_DIR.joinpath('other')
