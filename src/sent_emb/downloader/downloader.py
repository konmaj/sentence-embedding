from pathlib import Path
from urllib.request import urlretrieve
from shutil import unpack_archive, move
from os import listdir, rmdir


DOWNLOAD_DIR = Path('/', 'opt', 'resources')

STS_URLS = {
    'STS12' : 'http://ixa2.si.ehu.es/stswiki/images/4/40/STS2012-en-test.zip',
    'STS13' : 'http://ixa2.si.ehu.es/stswiki/images/2/2f/STS2013-en-test.zip',
    'STS14' : 'http://ixa2.si.ehu.es/stswiki/images/8/8c/STS2014-en-test.zip',
    'STS15' : 'http://ixa2.si.ehu.es/stswiki/images/d/da/STS2015-en-test.zip',
    'STS16' : 'http://ixa2.si.ehu.es/stswiki/images/9/98/STS2016-en-test.zip',
}
STS_DIRS = { sts : DOWNLOAD_DIR.joinpath('datasets', sts) for sts in STS_URLS.keys() }


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


def get_sts_dataset(sts):
    '''
    Gets proper STS dataset.

    1) Downloads and extracts proper STS dataset.
    2) Unifies names of files and directories if needed.
    '''

    # prepare directory structure
    data_path = STS_DIRS[sts].joinpath('data')
    data_path.mkdir()

    out_path = STS_DIRS[sts].joinpath('out')
    out_path.mkdir()

    # get data
    zip_download_and_extract(STS_URLS[sts], data_path)

    # remove redundant directory
    dir_name_list = listdir(data_path)
    assert len(dir_name_list) == 1
    dir_name = dir_name_list[0]
    src_path = data_path.joinpath(dir_name)

    for file_name in listdir(src_path):
        move(src_path.joinpath(file_name), data_path.joinpath(file_name))
    rmdir(src_path)

    # normalize file names
    if sts == 'STS16':
        normalize_sts16_prefix(data_path)


def get_datasets():
    print('Checking for datasets:')
    
    for sts in STS_URLS.keys():
        if mkdir_if_not_exist(STS_DIRS[sts]):
            print(sts, 'dataset not found')
            get_sts_dataset(sts)
        else:
            print('Found', sts, 'dataset')
