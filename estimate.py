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

import configparser
import os

import numpy as np
import torch.autograd
import torch.cuda
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms
import cv2

import model
import utils.visualize


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    model_dir = utils.get_model_dir(config)
    dnn = utils.parse_attr(config.get('model', 'dnn'))()
    inference = model.Inference(dnn)
    inference.eval()
    inference.cuda()
    state_dict = torch.load(os.path.join(model_dir, 'latest.pth'))
    dnn.load_state_dict(state_dict)
    draw_masks = utils.visualize.DrawMasks()
    height, width = tuple(map(int, config.get('image', 'size').split()))
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])
    cap = cv2.VideoCapture('2.mp4')
    env = {}
    try:
        while cap.isOpened():
            ret, image_bgr = cap.read()
            if not ret:
                break
            _height, _width = image_bgr.shape[:2]
            scale = min(height, width) / min(_height, _width)
            image_resized = cv2.resize(image_bgr, (int(_width * scale), int(_height * scale)))
            #print (scale)
            #print (image_resized.size)
            tensor = trans(image_resized)
            tensor = tensor.unsqueeze(0).cuda()
            print (tensor.size())
            feature = inference(torch.autograd.Variable(tensor, volatile=True)).contiguous()
            print (feature.size())
            feature = torch.autograd.Variable(torch.from_numpy(np.array([[cv2.resize(f.data.cpu().numpy(), (_width, _height)) for f in b] for b in feature])), volatile=True)
            prob, pred = torch.max(F.softmax(feature, -1), -1)
            print (pred.size())
            image_result = draw_masks(image_bgr, pred[0].data.numpy())
            #cv2.imshow('estimate', image_result)
            cv2.waitKey(1)
            if 'writer' in env:
                env['writer'].write(image_result)
            else:
                env['writer'] = cv2.VideoWriter('2_seg.mp4', int(cap.get(cv2.CAP_PROP_FOURCC)), int(cap.get(cv2.CAP_PROP_FPS)), (_width, _height))
                #env['writer'] = cv2.VideoWriter('2_seg.mp4', 0x00000021, int(cap.get(cv2.CAP_PROP_FPS)), (_width, _height))

    finally:
        cv2.destroyAllWindows()
        env['writer'].release()
        cap.release()


if __name__ == '__main__':
    main()
