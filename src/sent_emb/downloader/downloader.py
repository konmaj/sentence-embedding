from pathlib import Path
from urllib.request import urlretrieve
from shutil import unpack_archive


DOWNLOAD_DIR = Path('/', 'opt', 'resources')


def get_datasets():
    print('Ensure that datasets exist and download if not')


GLOVE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
GLOVE_SIZE = 2.2 #GB
GLOVE_DIR = DOWNLOAD_DIR.joinpath('embeddings', 'glove')
GLOVE_ZIP = GLOVE_DIR.joinpath(Path(GLOVE_URL).name)


def get_embeddings():
    print('Checking for embeddings:')
    
    if not GLOVE_DIR.exists():
        GLOVE_DIR.mkdir(parents=True, exist_ok=True)
        
        print('  GloVe embeddings not found. Downloading', GLOVE_URL, '(' + str(GLOVE_SIZE) + 'GB)')
        urlretrieve(GLOVE_URL, str(GLOVE_ZIP))
        
        print('  Unpacking', GLOVE_ZIP)
        unpack_archive(str(GLOVE_ZIP), extract_dir=str(GLOVE_DIR))
        
        GLOVE_ZIP.unlink()
    else:
        print('  Found GloVe embeddings')

