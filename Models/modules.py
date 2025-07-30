# This script was written by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# here is the main modules used to build the model
# Contact: medchaoukiziara@gmail.com || chaouki.ziara@univ-sba.dz

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

class BinAttention(nn.Module):
    def __init__(self, args, num_heads=2):
        super(BinAttention, self).__init__()
        self.bins = args.bins
        self.num_heads = num_heads
        self.head_dim = self.bins // num_heads

        assert self.head_dim * num_heads == self.bins, "Bins must be divisible by num_heads"
        encoder_layers = nn.TransformerEncoderLayer(self.bins, self.num_heads, 1024)
        self.multihead_attn = nn.TransformerEncoder(encoder_layers, num_layers=4)

        # Linear transformations for query, key, and value
        self.query_proj = nn.Sequential(nn.Linear(self.bins, self.bins), nn.ReLU())
        self.key_proj = nn.Sequential(nn.Linear(self.bins, self.bins), nn.ReLU())
        self.value_proj = nn.Sequential(nn.Linear(self.bins, self.bins), nn.ReLU())



    def forward(self, query, key, value):

        # Linear projections
        query = self.query_proj(query)  # (B, Seq_len, Bins)
        key = self.key_proj(key)        # (B, Seq_len, Bins)
        value = self.value_proj(value)  # (B, Seq_len, Bins)
        input = torch.cat([query, key, value], 0) 

        #input = query+key+value
        attn_weights = torch.relu(self.multihead_attn(input).permute(1,0,2))

        output = torch.einsum("bst,qbk->bk", attn_weights, value) 
        output = output / output.sum(-1, keepdim = True)
        return output


class HistLayer(nn.Module):
    def __init__(self, args):
        super(HistLayer, self).__init__()
        self.bins = args.bins
        self.args = args
        kernel_size = 10

        # Set backbone-specific channels
        bn_channels = 1536 if args.backbone == 'DepthHistL' else 1024


        # Define convolutional layers for center projections
        self.query_conv_center = self._create_conv_layer_(3, args.bins, kernel_size)
        self.key_conv_center = self._create_conv_layer_(bn_channels, args.bins, kernel_size)
        self.value_conv_center = self._create_conv_layer_(128, args.bins, kernel_size)

        # Define convolutional layers for scale projections
        self.query_conv_scales = self._create_conv_layer_(3, args.bins, kernel_size)
        self.key_conv_scales = self._create_conv_layer_(bn_channels, args.bins, kernel_size)
        self.value_conv_scales = self._create_conv_layer_(128, args.bins, kernel_size)

        # Additional layers
        self.act = nn.ReLU()    
        self.side = nn.Sequential(nn.Conv2d(128, args.bins, kernel_size=3, stride=2, padding=1)) #Half 323, same 313
        #
        self.sharpness_factor = None
        self.centers = None
        self.scales = None

        # Attention mechanisms
        self.compute_att_centers = BinAttention(args, num_heads=4)
        self.compute_att_scales = BinAttention(args, num_heads=4)

        # Positional encodings
        self.pos_encodings = nn.Parameter(torch.zeros(25000, args.bins), requires_grad=True)
        #nn.init.trunc_normal_(self.pos_encodings, std=0.02)
        with torch.no_grad():
               self.pos_encodings.copy_(
                    torch.randint(low=0, high=25000, size=self.pos_encodings.shape).float()/24999.            
                )

        # Kernel selection
        self.estimator = {
            'laplacian': self.laplacian,
            'cauchy': self.cauchy,
            'acts': self.acts
        }.get(args.kernel, self.gaussian)

        if args.kernel == 'acts':
             self.interval_length = (self.args.max_depth - self.args.min_depth) / self.bins
             self.kernel_width = self.interval_length / 2.5

        # Small constant to avoid division by zero
        self.epsilon = 1e-8
        self.loss = None
    def _create_conv_layer_(self, in_channels, out_channels, kernel_size, stride=None, padding=None):
        """Separable version of the conv layer with same output shape."""
        return nn.Sequential(
            # Normalization and activation
            nn.InstanceNorm2d(in_channels, affine=True),
            
            # Depthwise convolution
            nn.Conv2d(in_channels, 128, kernel_size=kernel_size, stride=stride if stride is not None else kernel_size,
                    padding=padding if padding is not None else 0, dilation=3, groups=1, bias=False),
            nn.PReLU(128),
            
            # Pointwise convolution
            nn.Conv2d(128, out_channels, kernel_size=1, stride=1, bias=False),
            nn.PReLU(out_channels)
        )

    def forward(self, rgb, bn, decoded):
        """
            Forward pass of the model.

            Args:
            - rgb (Tensor): Input RGB image tensor.
            - bn (Tensor): Input batch-normalized tensor.
            - decoded (Tensor): Intermediate decoded feature map.

            Returns:
            - scores (Tensor): Estimated depth values.
        """

        # Extract batch size, channels, height, and width

        all = decoded#torch.cat([F.interpolate(rgb, decoded.shape[-2:]),
              #           decoded], 1)
        b, c, h, w = decoded.size()

        # ---------------------------- Process Centers ---------------------------- #
        
        # Compute value_c
        value_c = self.value_conv_center(decoded)  # (batch_size, bins, H', W')
        _, _, hk, wk = value_c.size()
        value_c = value_c.view(b, self.bins, -1).transpose(1, 2)  # (batch_size, N, bins)

        # Compute query_c
        query_c = self.query_conv_center(F.interpolate(rgb, decoded.shape[-2:]))
        query_c = query_c.view(b, self.bins, -1).transpose(1, 2)  # (batch_size, N, bins)

        # Compute key_c
        key_c = self.key_conv_center(F.interpolate(bn, decoded.shape[-2:]))
        key_c = key_c.view(b, self.bins, -1).transpose(1, 2)  # (batch_size, N, bins)

        # Add positional encodings
        seq_len_c = query_c.size(1)
        pos_encodings_c = self.pos_encodings[:seq_len_c, :].unsqueeze(0)  # (1, N, bins)

        query_c += pos_encodings_c
        key_c += pos_encodings_c
        value_c += pos_encodings_c


        # ---------------------------- Process Scales ---------------------------- #

        # Compute value_s
        value_s = self.value_conv_scales(decoded)  # (batch_size, bins, H', W')
        value_s = value_s.view(b, self.bins, -1).transpose(1, 2)  # (batch_size, N, bins)

        # Compute query_s
        query_s = self.query_conv_scales(F.interpolate(rgb, decoded.shape[-2:]))
        query_s = query_s.view(b, self.bins, -1).transpose(1, 2)  # (batch_size, N, bins)

        # Compute key_s
        key_s = self.key_conv_scales(F.interpolate(bn, decoded.shape[-2:]))
        key_s = key_s.view(b, self.bins, -1).transpose(1, 2)  # (batch_size, N, bins)

        # Add positional encodings
        seq_len_s = query_s.size(1)
        pos_encodings_s = self.pos_encodings[:seq_len_s, :].unsqueeze(0)  # (1, N, bins)

        query_s += pos_encodings_s
        key_s += pos_encodings_s
        value_s += pos_encodings_s

        # Compute attention for centers
        centers = self.compute_att_centers(
            query_c.permute(1, 0, 2), key_c.permute(1, 0, 2), value_s.permute(1, 0, 2) + value_c.permute(1, 0, 2)
        )  # (batch_size, N, bins)

        # Prepare weighted centers
        weighted_centers = centers.unsqueeze(-1).unsqueeze(-1)

        # Compute attention for scales
        scales = self.compute_att_scales(
            query_s.permute(1, 0, 2), key_s.permute(1, 0, 2), value_s.permute(1, 0, 2) + value_c.permute(1, 0, 2)
        )  # (batch_size, N, bins)

        # Prepare weighted scales
        weighted_scales = scales.unsqueeze(-1).unsqueeze(-1)

        # ---------------------------- Compute Final Scores ---------------------------- #

        # Convert RGB to grayscale (Y channel from YUV)
        #decoded = (
        #    0.2989 * rgb[:, 0, :, :]
        #    + 0.5870 * rgb[:, 1, :, :]
        #    + 0.1140 * rgb[:, 2, :, :]
        #)

        # Resize to match the original dimensions
        #decoded = F.interpolate(decoded.unsqueeze(1), (h, w))

        # Compute final scores
        scores = self.estimator(self.side(decoded), weighted_centers, weighted_scales)
        #scores = scores / scores.sum(1, keepdim = True)
        self.sharpness_factor = scores
        self.centers = weighted_centers
        self.scales = scales

        return scores, weighted_centers
    
    def gaussian(self, data, centers, scales):
        '''
            Compute the soft histogram using a gaussian kernel.
            Args:
                data (Tensor): Input data of shape (batch_size, bins, H, W).
                centers (Tensor): Centers of the bins.
                scales (Tensor): Scale parameters for the Laplacian distribution.
            Returns:
                Tensor: Soft histogram representation.
        '''
        # Compute the difference between the data and the bin centers
        diff  = data  - centers
        # Apply the gaussian kernel
        scores = torch.exp(-torch.square(diff) / (torch.square(scales) + self.epsilon))
        
        # Normalize the scores to sum to 1 along the bin dimension
        scores = scores / (torch.sum(scores, dim=1, keepdim=True) + self.epsilon)
        
        return scores
    
    def acts(self, data, centers, scales):
        '''
            Compute the soft histogram using a huennet kernel.
            Args:
                data (Tensor): Input data of shape (batch_size, bins, H, W).
                centers (Tensor): Centers of the bins.
                scales (Tensor): Scale parameters for the Laplacian distribution.
            Returns:
                Tensor: histogram representation.
        '''
        # Compute the difference between the data and the bin centers
        img_minus_bins_av = data  - centers
        img_plus_bins_av  = data  + centers
        self.kernel_width = self.interval_length / (scales + self.epsilon)
        
        # Apply the gaussian kernel
        maps = torch.sigmoid((img_minus_bins_av + self.interval_length / 2) / self.kernel_width) \
                                 - torch.sigmoid((img_minus_bins_av - self.interval_length / 2) / self.kernel_width) \
                                 + torch.sigmoid((img_plus_bins_av - 2 * self.args.min_depth + self.interval_length / 2) / self.kernel_width) \
                                 - torch.sigmoid((img_plus_bins_av - 2 * self.args.min_depth - self.interval_length / 2) / self.kernel_width) \
                                 + torch.sigmoid((img_plus_bins_av - 2 * self.args.max_depth + self.interval_length / 2) / self.kernel_width) \
                                 - torch.sigmoid((img_plus_bins_av - 2 * self.args.max_depth - self.interval_length / 2) / self.kernel_width)
        maps = maps / (torch.sum(maps, axis=1, keepdims=True) + 1e-10)        
        
        return maps
    
    def laplacian(self, data, centers, scales):
        '''
            Compute the soft histogram using a Laplacian kernel.
            Args:
                data (Tensor): Input data of shape (batch_size, bins, H, W).
                centers (Tensor): Centers of the bins.
                scales (Tensor): Scale parameters for the Laplacian distribution.
            Returns:
                Tensor: Soft histogram representation.
        '''
        # Compute the absolute difference between the data and the bin centers
        diff  = data  - centers
        diff = torch.abs(diff)  # (batch_size, bins, H, W)
        
        # Apply the Laplacian kernel
        scores = torch.exp(-diff / (torch.abs(scales) + self.epsilon))
        
        # Normalize the scores to sum to 1 along the bin dimension
        scores = scores / (torch.sum(scores, dim=1, keepdim=True) + self.epsilon)  
        
        return scores
    

    def cauchy(self, data, centers, scales):
        '''
            Compute the soft histogram using a Cauchy kernel.
            Args:
                data (Tensor): Input data of shape (batch_size, bins, H, W).
                centers (Tensor): Centers of the bins.
                scales (Tensor): Scale parameters for the Cauchy distribution.
            Returns:
                Tensor: Soft histogram representation.
        '''
        # Compute the difference between the data and the bin centers
        diff  = data  - centers
        # Apply the Cauchy kernel
        scores = 1 / (1 + torch.square(diff / (torch.abs(scales) + self.epsilon)))

        # Normalize the scores to sum to 1 along the bin dimension
        scores = scores / (torch.sum(scores, dim=1, keepdim=True) + self.epsilon)
        
        return scores
    
    
class UpSample_EffNet(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSample_EffNet, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)

class Decoder_EffNet(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(Decoder_EffNet, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSample_EffNet(skip_input=features // 1 + 112 + 64, output_features=features // 2)
        self.up2 = UpSample_EffNet(skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSample_EffNet(skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSample_EffNet(skip_input=features // 8 + 16 + 8, output_features=features // 16)

        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        #         x_d5 = self.up5(x_d4, features[0])
        #print(x_d4.shape)
        out = self.conv3(x_d4)

        return out

class Encoder_EffNet(nn.Module):
    def __init__(self, backend):
        super(Encoder_EffNet, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features
