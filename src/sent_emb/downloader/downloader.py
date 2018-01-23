from pathlib import Path
from urllib.request import urlretrieve
from shutil import unpack_archive, move
from os import listdir, rmdir


DOWNLOAD_DIR = Path('/', 'opt', 'resources')

STS_TEST_URLS = {
    'STS12' : 'http://ixa2.si.ehu.es/stswiki/images/4/40/STS2012-en-test.zip',
    'STS13' : 'http://ixa2.si.ehu.es/stswiki/images/2/2f/STS2013-en-test.zip',
    'STS14' : 'http://ixa2.si.ehu.es/stswiki/images/8/8c/STS2014-en-test.zip',
    'STS15' : 'http://ixa2.si.ehu.es/stswiki/images/d/da/STS2015-en-test.zip',
    'STS16' : 'http://ixa2.si.ehu.es/stswiki/images/9/98/STS2016-en-test.zip',
}

# STS12 - the only year with prepared training data
# In subsequent years former STS test data (and STS12 training data) were used for training.
STS12_TRAIN_URL = 'http://ixa2.si.ehu.es/stswiki/images/e/e4/STS2012-en-train.zip'

STS_DIRS = { sts : DOWNLOAD_DIR.joinpath('datasets', sts) for sts in STS_TEST_URLS.keys() }


def mkdir_if_not_exist(dir_path):
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=True)
        return True
    else:
        return False


def zip_download_and_extract(url, dir_path):
    dir_pathname = str(dir_path.resolve())
    zip_path = dir_path.joinpath(Path(url).name)
    zip_pathname = str(zip_path.resolve())

    print('Downloading from', url)
    urlretrieve(url, zip_pathname)

    print('Extracting into', dir_pathname)
    unpack_archive(zip_pathname, extract_dir=dir_pathname)

    zip_path.unlink()


def normalize_sts16_prefix(data_path):
    '''
    Converts names of files as in example: STS2016.gs.headlines.txt -> STS.gs.headlines.txt
    '''
    for file_name in listdir(data_path):
        if file_name[:3] == 'STS':
            assert file_name[:7] == 'STS2016'
            old_name = data_path.joinpath(file_name)
            new_name = data_path.joinpath('STS' + file_name[7:])
            move(old_name, new_name)


def flatten_dir(dir_path):
    '''
    Removes the only child of directory 'dir_path' and move its content to 'dir_path'.

    Assumes, that directory 'dir_path' has exactly one child and this child
    is a directory.

    Example:
    dir_path == '/dir_1/dir_2/.../dir_k/'
    child_path == '/dir_1/dir_2/.../dir_k/dir_k+1/'

    After flatten_dir() all childs of directory 'child_path' will be moved
    to direcory 'dir_path'.
    '''
    dir_name_list = listdir(dir_path)
    assert len(dir_name_list) == 1

    dir_name = dir_name_list[0]
    src_path = dir_path.joinpath(dir_name)
    assert src_path.is_dir()

    for file_name in listdir(src_path):
        move(src_path.joinpath(file_name), dir_path.joinpath(file_name))
    rmdir(src_path)


def get_sts_dataset(sts):
    '''
    Gets proper STS dataset.

    1) Downloads and extracts proper STS dataset.
    2) Unifies names of files and directories if needed.
    '''

    out_path = STS_DIRS[sts].joinpath('out')
    out_path.mkdir()

    test_data_path = STS_DIRS[sts].joinpath('test-data')
    test_data_path.mkdir()

    # get test data
    zip_download_and_extract(STS_TEST_URLS[sts], test_data_path)

    # flatten redundant directory
    flatten_dir(test_data_path)

    # special cases

    if sts == 'STS12':
        train_data_path = STS_DIRS[sts].joinpath('train-data')
        train_data_path.mkdir()

        # get training data
        zip_download_and_extract(STS12_TRAIN_URL, train_data_path)

        # flatten redundant directory
        flatten_dir(train_data_path)

    # normalize file names
    if sts == 'STS16':
        normalize_sts16_prefix(test_data_path)


def get_datasets():
    print('Checking for datasets:')

    for sts in STS_TEST_URLS.keys():
        if mkdir_if_not_exist(STS_DIRS[sts]):
            print(sts, 'dataset not found')
            get_sts_dataset(sts)
        else:
            print('Found', sts, 'dataset')


def get_word_frequency():
    print('Checking for word frequency:')
    URL = 'http://www.kilgarriff.co.uk/BNClists/all.num.gz'

    path = DOWNLOAD_DIR.joinpath('other')
    mkdir_if_not_exist(path)
    word_frequency_path = path.joinpath('word_frequency')
    if mkdir_if_not_exist(word_frequency_path):
        print('Word frequency not found')
        urlretrieve(URL, word_frequency_path.joinpath(Path(URL).name))
    else:
        print('Found word frequency')