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

import pickle
import random
import copy

import numpy as np
import torch.utils.data
import torchvision.transforms
import cv2


def load_pickles(paths):
    data = []
    for path in paths:
        with open(path, 'rb') as f:
            data += pickle.load(f)
    return data


class Rotator(object):
    def __init__(self, y, x, height, width, angle):
        """
        A efficient tool to rotate multiple images in the same size.
        :author 申瑞珉 (Ruimin Shen)
        :param y: The y coordinate of rotation point.
        :param x: The x coordinate of rotation point.
        :param height: Image height.
        :param width: Image width.
        :param angle: Rotate angle.
        """
        self._mat = cv2.getRotationMatrix2D((x, y), angle, 1.0)
        r = np.abs(self._mat[0, :2])
        _height, _width = np.inner(r, [height, width]), np.inner(r, [width, height])
        fix_y, fix_x = _height / 2 - y, _width / 2 - x
        self._mat[:, 2] += [fix_x, fix_y]
        self._size = int(_width), int(_height)

    def __call__(self, image, flags=cv2.INTER_LINEAR, fill=None):
        if fill is None:
            fill = np.random.rand(3) * 256
        return cv2.warpAffine(image, self._mat, self._size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)

    def _rotate_points(self, points):
        _points = np.pad(points, [(0, 0), (0, 1)], 'constant')
        _points[:, 2] = 1
        _points = np.dot(self._mat, _points.T)
        return _points.T.astype(points.dtype)

    def rotate_points(self, points):
        return self._rotate_points(points[:, ::-1])[:, ::-1]


def random_rotate(image, masks):
    angle = random.uniform(-40, 40)
    height, width = image.shape[:2]
    rotator = Rotator(height / 2, width / 2, height, width, angle)
    image = rotator(image, fill=0)
    masks = [rotator(mask, fill=0) for mask in masks]
    return image, masks


def flip_horizontally(image, masks):
    assert len(image.shape) == 3
    image = cv2.flip(image, 1)
    masks = [cv2.flip(mask, 1) for mask in masks]
    return image, masks


def random_crop(image, masks, height, width, dtype=np.float32):
    margin = np.maximum(np.array(image.shape[:2], dtype) - (height, width), 0)
    y, x = map(int, np.random.rand(2).astype(dtype) * margin)
    _y, _x = y + height, x + width
    pad_y, pad_x = np.maximum(np.array([_y, _x]) - image.shape[:2], 0)
    image = np.pad(image, [(0, pad_y), (0, pad_x), (0, 0)], 'constant')
    image = image[y:_y, x:_x, :]
    pad = [(0, pad_y), (0, pad_x)]
    masks = [np.pad(mask, pad, 'constant')[y:_y, x:_x] for mask in masks]
    return image, masks


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = copy.deepcopy(self.data[index])
        image = cv2.imread(data['path'])
        data['image'] = image
        data['size'] = np.array(image.shape[:2])
        masks = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in data['paths']]
        for mask in masks:
            assert image.shape[:2] == mask.shape, [image.shape[:2], mask.shape]
        data['masks'] = masks
        data['image'], data['masks'] = random_rotate(data['image'], data['masks'])
        data['image'], data['masks'] = flip_horizontally(data['image'], data['masks'])
        return data


class Collate(object):
    def __init__(self, size, _size):
        self.size = size
        self._size = _size
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

    def __call__(self, batch):
        _batch = []
        for data in batch:
            data['image'], data['masks'] = random_crop(data['image'], data['masks'], *self.size)
            data['tensor'] = self.trans(data['image'])
            data['masks'] = np.array([(cv2.resize(mask, self._size) > 127).astype(np.uint8) for mask in data['masks']])
            _batch.append(data)
        return torch.utils.data.dataloader.default_collate(_batch)
