# This code was written by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# it contains the implementation of the loss functions used to train the model
# please Modify the args on the args_****.txt files to control the coef. of each loss value
# Contact: medchaoukiziara@gmail.com || chaouki.ziara@univ-sba.dz

import torch
import warnings
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Tuple, Union

class HistLoss(nn.Module):
    def __init__(self, args):
        super(HistLoss, self).__init__()
        self.args = args
        self.min = args.min_depth
        self.max = args.max_depth
        self.t = args.t
        self.bins = args.bins 
        self.epsilon = 1e-3
        self.int_len = (self.max - self.min) / self.bins
        if args.kernel == 'laplacian' :
            self.estimator = self.laplacian
        elif args.kernel == 'cauchy' :
            self.estimator = self.cauchy
        else  :
            self.estimator = self.gaussian
    
    def __centers__ (self, gt):
        maxs = gt.view(gt.size(0), -1).max(dim=1, keepdim=True)[0]
        max_len = torch.tensor(self.bins).to(gt.device)
        linspace = [torch.linspace(0, max_val.item(), steps=int(max_val.item())+1).unsqueeze(0) for max_val in maxs ]
        centers = torch.stack([torch.nn.functional.pad(x, (0, int(max_len.item() - x.size(1))), mode='constant', value=0) for x in linspace])
        centers = centers.view(gt.size(0), -1, 1, 1)
        return centers

    def gaussian(self, img, centers):
        d = img - centers.to(img.device)
        s = torch.exp(-torch.square(d) / self.t**2)
        s = s / (torch.sum(s, axis=1, keepdims=True) + 1e-10)
        return s
    
    def laplacian(self, img, centers):
        d = torch.abs(img - centers.to(img.device))  # Absolute difference
        s = torch.exp(-d / (self.t + 1e-10))
        s = s / (torch.sum(s, axis=1, keepdims=True) + 1e-10)
        return s
    
    def cauchy(self, img, centers):
        d = img - centers.to(img.device)
        s = 1 / (1 + 100*torch.square(d / (self.t + 1e-10)))
        s = s / (torch.sum(s, axis=1, keepdims=True) + 1e-10)
        return s
    
    def function(self, s):
        #h = s / s.size(-1)*s.size(-2)
        h = s.sum(axis=[2, -1])#/ s.size(-1)*s.size(-2)
        h = torch.cumsum(h, axis=1) / h.sum(axis=[-1,-2], keepdims=True)
        return h

class Hist1D_loss(HistLoss):
    def __init__(self, args):
        super(Hist1D_loss, self).__init__(args=args)
        self.name = '1DLoss'
    def forward(self, gt, pred, mask = None, interpolate = False):
        if interpolate:
            pred = nn.functional.interpolate(pred, gt.shape[-2:], mode='bilinear', align_corners=True)
        if mask is not None : 
            pred = pred*mask
            gt   = gt*mask

        h_gt   = self.function(self.estimator(gt  , self.__centers__(gt)))
        h_pred = self.function(self.estimator(pred, self.__centers__(gt))) #self.function(pred) # self.function(self.estimator(pred, self.__centers__(pred))) #
        return self.emd(h_gt, h_pred)
    
    def emd(self, h_gt, h_pred):
        return torch.mean(torch.sum(torch.abs(h_gt - h_pred), dim=1))
    
class Hist2D_loss(HistLoss):
    def __init__(self, args):
        super(Hist2D_loss, self).__init__(args=args)
        self.name = '2DLoss'    
    def forward(self, gt, pred, mask = None, interpolate = False):
        if interpolate:
            pred = nn.functional.interpolate(pred, gt.shape[-2:], mode='bilinear', align_corners=True)
        if mask is not None : 
            #convert inputs
            pred = pred*mask
            gt   = gt*mask
        h_gt   = torch.cumsum(self.estimator(gt  , self.__centers__(gt))  , 1)   # self.estimator(gt,   self.__centers__(gt))
        h_pred = torch.cumsum(pred , 1) 
        #print(h_gt.shape, h_pred.shape)
        return self.loss(h_pred, h_gt).mean()
    
    def cosine_similarity(self, inp, inp2):   
        inp = inp.view(inp.size(0), inp.size(1), -1)
        inp2 = inp2.view(inp2.size(0),inp.size(1), -1)
        cos = torch.cosine_similarity(inp, inp2,  dim=-1) * 0.5
        return torch.relu(0.5 - ((cos + 0.5 )/2))
    
    def mean_absolute_percentage_error(self, y_pred, y_true):
        return torch.mean(torch.abs(y_true - y_pred) / (y_true+self.epsilon))
    
    def loss(self, pred, gt):
        grad_x_pred, grad_y_pred = self.gradients(pred)
        grad_x_gt  , grad_y_gt   = self.gradients(gt)
        mag_pred = torch.sqrt(grad_x_pred**2 + grad_y_pred**2 + self.epsilon)  
        mag_gt   = torch.sqrt(  grad_x_gt**2 + grad_y_gt**2 + self.epsilon)
        sim = self.cosine_similarity(grad_x_pred, grad_x_gt).mean() + self.cosine_similarity(grad_y_pred, grad_y_gt).mean()
        mag = self.mean_absolute_percentage_error(mag_pred, mag_gt) + self.mean_absolute_percentage_error(mag_gt, mag_pred)
        return sim + mag * 0.5
        
    def gradients(self, x):
        grad_y = -x[:, :, :, :-1] + x[:, :, :, 1:]
        grad_x = -x[:, :,:-1, : ] + x[:, :, 1:, :]
        grad_y = F.pad(grad_y, (0, 1, 0, 0))  # Pad last dimension (width)
        grad_x = F.pad(grad_x, (0, 0, 0, 1))  # Pad height dimension
        return grad_x, grad_y

class SILogLoss(nn.Module): 
    def __init__(self, args):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.args = args
    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)    
        if mask is not None:
            input  = input[mask]  + 1e-8
            target = target[mask] + 1e-8
        g = torch.log(input) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)
