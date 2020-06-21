"""
Copyright (C) 2017, (Ruimin Shen)

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

import argparse
import configparser
import multiprocessing
import os
import shutil
import traceback

import torch.autograd
import torch.cuda
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

import model
import utils.data
import utils.visualize

os.environ["CUDA_VIBILE_DEVICES"] = '0,1,2'
def main():
    args = make_args()
    config = configparser.ConfigParser()
    config.read('config.ini')
    model_dir = utils.get_model_dir(config)
    cache_dir = utils.get_cache_dir(config)
    os.makedirs(model_dir, exist_ok=True)
    dnn = utils.parse_attr(config.get('model', 'dnn'))()
    inference = model.Inference(dnn)
    inference.train()
    inference.cuda()
    try:
        state_dict = torch.load(os.path.join(model_dir, 'latest.pth'))
        dnn.load_state_dict(state_dict)
        print('latest model loaded')
    except:
        pass
    optimizer = torch.optim.Adam(inference.parameters(), args.learning_rate)
    paths = [os.path.join(cache_dir, phase + '.pkl') for phase in ['train']]
    dataset = utils.data.Dataset(utils.data.load_pickles(paths))
    try:
        workers = config.getint('data', 'workers')
    except configparser.NoOptionError:
        workers = multiprocessing.cpu_count()
    size = tuple(map(int, config.get('data', 'size').split()))
    _size = tuple(map(int, config.get('data', '_size').split()))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=workers, collate_fn=utils.data.Collate(size, _size))
    writer = SummaryWriter(os.path.join(model_dir))
    step = 0
    for epoch in range(args.epoch):
        for data in loader:
            data['tensor'] = data['tensor'].cuda()
            data['masks'] = data['masks'].cuda()
            #print ('the masks is: {}').format(data['masks'])
            tensor = torch.autograd.Variable(data['tensor'])
            feature = inference(tensor).contiguous()
            loss = model.loss(data, feature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 30 == 0:
                writer.add_scalar('loss_total', loss, step)
            step += 1
        try:
            path = os.path.join(model_dir, str(step)) + '.pth'
            torch.save(dnn.state_dict(), path)
            print(path)
            shutil.copy(path, os.path.join(model_dir, 'latest.pth'))
        except:
            traceback.print_exc()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--steps', type=int, default=None, help='max number of steps')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('-lr', '--learning_rate', default=5e-6, type=float, help='learning rate')
    parser.add_argument('-e', '--epoch', type=int, default=300)
    return parser.parse_args()


if __name__ == '__main__':
    main()
