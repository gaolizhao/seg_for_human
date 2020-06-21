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

import numpy as np
import torch.autograd
import torch.cuda
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms
import cv2
#from lib.nn import as_numpy
import model
import utils.data
import utils.visualize
import time
#from datatime import datatime
def calc_iou(data, pred):
    batch_size, num, rows, cols = pred.size()
    intersection = data * pred
    union = (data + pred) > 0
    return intersection.float().view(batch_size, num, -1).sum(-1) / union.float().view(batch_size, num, -1).sum(-1)


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    model_dir = utils.get_model_dir(config)
    cache_dir = utils.get_cache_dir(config)
    dnn = utils.parse_attr(config.get('model', 'dnn'))()
    inference = model.Inference(dnn)
    inference.eval()
    inference.cuda()
    path = os.path.join(model_dir, 'latest.pth')
    #print (path)
    dnn.load_state_dict(torch.load(path))
    height, width = tuple(map(int, config.get('image', 'size').split()))
    #print (str(height)+' '+str(width))
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])
    paths = [os.path.join(cache_dir, phase + '.pkl') for phase in ['test']]
    print (paths)
    draw_masks = utils.visualize.DrawMasks()
    results = []
    for data in utils.data.load_pickles(paths):
        image_bgr = cv2.imread(data['path'])
        #print (data['path'])
        _height, _width = image_bgr.shape[:2]
        #print (str(_height)+' '+str(_width))
        scale = min(height, width) / min(_height, _width)
        image_resized = cv2.resize(image_bgr, (int(_width * scale), int(_height * scale)))
        #print (str(scale))
        tensor = trans(image_resized)
        tensor = tensor.unsqueeze(0).cuda()
        feature = inference(torch.autograd.Variable(tensor, volatile=True)).contiguous()
        #print ('features:')
        #print (feature.size())
        feature = torch.autograd.Variable(torch.from_numpy(np.array([[cv2.resize(f.data.cpu().numpy(), (_width, _height)) for f in b] for b in feature])), volatile=True)
        #print (feature.size())

        prob, pred = torch.max(F.softmax(feature, -1), -1)
        image_result = draw_masks(image_bgr,pred[0].data.numpy())
        pred = pred.data.byte().cuda()

        _size = pred.size()[-2:]
        #print (pred.size())
        _size = _size[::-1]
        masks = torch.from_numpy(np.array([(cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), _size) > 127).astype(np.uint8) for path in data['paths']])).unsqueeze(0).cuda()
        #print (masks.size())
        iou = calc_iou(masks, pred)
        results.append(iou.cpu().numpy())
        
        name = data['path'].split('/')[-1].split('.')[-2]
        #print (name)
        cv2.imwrite(os.path.join('./result',name+'_seg.png'),image_result) 
    print(np.mean(results))
    results = np.mean(results)
    with open('./record/eval.txt','a') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
        f.write('\n')
        f.write('the mIOU of artifact is: '+str(results))
        f.write('\n')


if __name__ == '__main__':
    main()
