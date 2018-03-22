from pathlib import Path

RESOURCES_DIR = Path('/', 'opt', 'resources') # TODO: read resources dir from config file
DATASETS_DIR = RESOURCES_DIR.joinpath('datasets')
EMBEDDINGS_DIR = RESOURCES_DIR.joinpath('embeddings')
OTHER_RESOURCES_DIR = RESOURCES_DIR.joinpath('other')
