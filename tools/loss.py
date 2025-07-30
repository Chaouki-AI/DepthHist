# This code was written by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# it contains the implementation of the loss functions used to train the model
# please Modify the args on the args_****.txt files to control the coef. of each loss value
# Contact: medchaoukiziara@gmail.com || chaouki.ziara@univ-sba.dz

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.loss import chamfer_distance

class HistLoss(nn.Module):
    def __init__(self, args):
        super(HistLoss, self).__init__()
        self.args = args
        self.min = args.min_depth
        self.max = args.max_depth
        self.t = None
        self.bins = args.bins #int(self.max - int(self.min))
        self.epsilon = 1e-5
        self.int_len = (self.max - self.min) / self.bins
        
        self.estimator = {
            'laplacian': self.laplacian,
            'cauchy': self.cauchy,
            'acts': self.acts}.get(args.kernel, self.gaussian)

    def gaussian(self, img, centers):
        d = img - centers.to(img.device)
        s = torch.exp(-torch.square(d) / (self.t**2 + self.epsilon))
        s = s / (torch.sum(s, axis=1, keepdims=True) + 1e-10)
        return s
    
    def laplacian(self, img, centers):
        d = torch.abs(img - centers.to(img.device))  # Absolute difference
        s = torch.exp(-d / (self.t + 1e-10))
        s = s / (torch.sum(s, axis=1, keepdims=True) + 1e-10)
        return s
    
    def cauchy(self, img, centers):
        d = img - centers.to(img.device)
        s = 1 / (1 + torch.square(d / (self.t + 1e-10)))
        s = s / (torch.sum(s, axis=1, keepdims=True) + 1e-10)
        return s
    
    def function(self, s):
        #h = s / s.size(-1)*s.size(-2)
        h = s.sum(axis=[2, -1])#/ s.size(-1)*s.size(-2)
        h = torch.cumsum(h, axis=1) / h.sum(axis=[-1,-2], keepdims=True)
        return h

    def acts(self, image, centers):
        img_minus_bins_av = torch.sub(image, centers)
        img_plus_bins_av  = torch.add(image, centers)
        self.kernel_width = self.interval_length / (self.t + self.epsilon)
    
        maps = torch.sigmoid((img_minus_bins_av + self.interval_length / 2) / self.kernel_width) \
                                 - torch.sigmoid((img_minus_bins_av - self.interval_length / 2) / self.kernel_width) \
                                 + torch.sigmoid((img_plus_bins_av - 2 * self.min + self.interval_length / 2) / self.kernel_width) \
                                 - torch.sigmoid((img_plus_bins_av - 2 * self.min - self.interval_length / 2) / self.kernel_width) \
                                 + torch.sigmoid((img_plus_bins_av - 2 * self.max + self.interval_length / 2) / self.kernel_width) \
                                 - torch.sigmoid((img_plus_bins_av - 2 * self.max - self.interval_length / 2) / self.kernel_width)
        maps = maps / (torch.sum(maps, axis=1, keepdims=True) + 1e-10)
        return maps

class Hist2D_loss(HistLoss):
    def __init__(self, args):
        super(Hist2D_loss, self).__init__(args=args)
        self.name = '2DLoss'  

    def forward(self, gt, pred, centers, scales, mask = None, interpolate = False):
        self.t = scales.unsqueeze(-1).unsqueeze(-1)
        if interpolate:
            pred = nn.functional.interpolate(pred, gt.shape[-2:], mode='bilinear', align_corners=True)
        if mask is not None : 
            #convert inputs
            pred = pred*mask
            gt   = gt*mask
        h_gt   = self.estimator(gt  , centers)  
        h_gt = torch.cumsum(h_gt * centers, 1)
        h_pred = torch.cumsum(pred * centers, 1)
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

class CenterLoss(nn.Module):
    def __init__(self, args):
        super(CenterLoss, self).__init__()
        self.name = 'MSELoss'
        self.args = args
        self.bins = args.bins
        self.distance = chamfer_distance    
    
    def forward(self, image, pred, centers, mask=None, **kwargs):
        #centers = nn.functional.interpolate(centers, image.shape[-2:], mode='bilinear', align_corners=True)   
        
        target_points = image.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(image.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1
        #print(centers.shape)
        loss, _ = chamfer_distance(x=centers.squeeze(-1), y=target_points, y_lengths=target_lengths)
        return loss.mean()
        
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


