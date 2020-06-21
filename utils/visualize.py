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

import numpy as np
import cv2


class DrawMasks(object):
    def __init__(self, alpha=0.5, color=(255, 0, 0)):
        self.alpha = alpha
        self.color = color

    def __call__(self, image, masks, debug=False):
        color = np.reshape(self.color, [1, 1, len(self.color)]).astype(image.dtype)
        m = masks[0]
        mask = np.tile(np.reshape(m, m.shape + (1,)), [1, 1, 3]) * color
        canvas = (image * (1 - self.alpha) + mask * self.alpha).astype(np.uint8)
        if debug:
            cv2.imshow(canvas)
            cv2.waitKey(0)
        return canvas
