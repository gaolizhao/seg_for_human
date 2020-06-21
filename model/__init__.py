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

import logging
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


class Inference(nn.Module):
    def __init__(self, dnn, reshape=True):
        nn.Module.__init__(self)
        self.dnn = dnn
        self.reshape = reshape

    def forward(self, x):
        feature = self.dnn(x)
        if self.reshape:
            batch_size, channels, rows, cols = feature.size()
            feature = feature.view(batch_size, -1, 2, rows, cols).permute(0, 1, 3, 4, 2)
        return feature


def loss(data, feature):
    masks = torch.autograd.Variable(data['masks'].long())
    #print ('the masks is: {}'.format(masks) )
    #print (masks[0,0,:,:])
    #[16,1,21,21]
    #print (feature.view(-1,2).size())
    weights = torch.autograd.Variable(torch.cuda.FloatTensor([0.1,0.9]))
    res =  F.cross_entropy(feature.view(-1, 2), masks.view(-1),weights)
    #print (res.shape)
    return res
