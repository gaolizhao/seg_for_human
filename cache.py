"""
Copyright (C) 2017, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import configparser
import logging
import logging.config
import pickle
import random

import utils


def exists(paths):
    for path in paths:
        if not os.path.exists(path):
            return False
    return True


def cache(config, path):
    cachedir = os.path.dirname(path)
    phase = os.path.splitext(os.path.basename(path))[0]
    phasedir = os.path.join(cachedir, phase)
    os.makedirs(phasedir, exist_ok=True)
    root = os.path.expanduser('/cephfs/share/data/human_seg_data')
    data = []
    logging.info('loading ' + root)
    root = os.path.expanduser(os.path.expandvars(root))
    path = os.path.join(root, phase) + '.txt'
    if not os.path.exists(path):
        logging.warning(path + ' not exists')
        return
    with open(path, 'r') as f:
        filenames = [line.strip() for line in f]
    for filename in filenames:
        path = os.path.join(root, filename)
        #print (path)
        filename, ext = os.path.splitext(os.path.basename(filename))
        paths = [os.path.join(root, 'mask', filename + '_seg' + ext)]
        print(paths)
        if not exists(paths):
            continue
        data.append(dict(
            path=path, paths=paths,
        ))
    logging.info('%d of %d images are saved' % (len(data), len(filenames)))
    return data


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    cache_dir = utils.get_cache_dir(config)
    os.makedirs(cache_dir, exist_ok=True)
    for phase in ['train', 'val', 'test']:
        path = os.path.join(cache_dir, phase) + '.pkl'
        logging.info('save cache file: ' + path)
        data = cache(config, path)
        #if config.getboolean('cache', 'shuffle'):
        #    random.shuffle(data)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    logging.info('data are saved into ' + cache_dir)


if __name__ == '__main__':
    main()
