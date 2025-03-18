import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks.swin_transformer import SwinTransformer

class DepthHist_encoder(nn.Module):
    def __init__(self, args,  **kwargs):
        super(DepthHist_encoder, self).__init__()        
        window_size = 7

        if args.backbone == 'DepthHistB': #if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif args.backbone == 'DepthHistL': #elif version[:-2] == 'large'
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        self.in_channels = in_channels
        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1)
        self.backbone = SwinTransformer(**backbone_cfg)
        self.backbone.init_weights(pretrained=args.path_pretrained)
    
    def forward(self, imgs):
        enc_feats = self.backbone(imgs)
        return enc_feats

class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpscaleBlock, self).__init__()
        
        # Step 1: Channel reduction using pointwise convolution
        self.channel_reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        # Step 2: Spatial upscaling using transposed convolution
        self.upscale = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
        # Step 3: Depthwise separable convolution for refinement
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        # Optional: BatchNorm and ReLU for non-linearity
        self.bn =nn.BatchNorm2d(out_channels) # nn.InstanceNorm2d(out_channels, affine=True) #
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # Channel reduction
        x = self.channel_reduction(x)
        
        # Spatial upscaling
        x = self.upscale(x)
        
        # Depthwise separable convolution
        x = self.depthwise(x)
       
        x = self.pointwise(x)     
        x = self.bn(x)
        x = self.relu(x)
        
        return x

class DepthHist_decoder(nn.Module):
    def __init__(self, args, **kwargs):
        super(DepthHist_decoder, self).__init__()
        self.in_channels = [192, 384, 768, 1536] if args.backbone == 'DepthHistL' else [128, 256, 512, 1024]
        self.block_1    = UpscaleBlock(self.in_channels[-1], self.in_channels[-2])
        self.block_2    = UpscaleBlock(self.in_channels[-2], self.in_channels[-3])
        self.block_3    = UpscaleBlock(self.in_channels[-3], self.in_channels[-4])
        self.block_4    = UpscaleBlock(self.in_channels[-4], 128)
        
    def forward (self, feats):
        out_1 = self.block_1(feats[-1])
        out_2 = self.block_2(out_1 + feats[-2])
        out_2 = F.interpolate(out_2, feats[-3].shape[-2:], mode='bilinear')
        out_3 = self.block_3(out_2 + feats[-3])
        out_3 = F.interpolate(out_3, feats[-4].shape[-2:], mode='bilinear')
        out_4 = self.block_4(out_3 + feats[-4])
        return out_4
