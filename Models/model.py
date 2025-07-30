# This code was written by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# Below is the code for the model
# Contact: medchaoukiziara@gmail.com || chaouki.ziara@univ-sba.dz


import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import HistLayer
from .loader import DepthHist_encoder, DepthHist_decoder


class DepthHist(nn.Module):

    """ 
    DepthHist Model: 
        This model supports two configurations: 
            1. Simple Mode: 
                This is a straightforward encoder-decoder approach. 
                Only the encoder and decoder are used.
                Outputs are passed through a convolutional layer and returned directly. 
            2. Non-Simple Mode: 
                This is an advanced model that uses encoder-decoder along with a histogram layer for depth computation.
                After passing through the encoder-decoder, the output is further processed using a histogram layer. 
                Depth is calculated by multiplying the output with the histogram and summing along a specific dimension. 
        Attributes: 
            - backend: The backbone model used for encoding. 
            - bins: The number of bins used in the histogram (only for non-simple mode). 
            - simple: A boolean flag indicating whether to use the simple mode. 
    """
    
    def __init__(self, args):
        super(DepthHist, self).__init__()
        # Initialize the encoder based on the backend provided
        self.args = args
        if args.path_pth_model == None :
            self.set_seed(22)
            print("\n\n\nThe weights will be initialized evenly\n\n\n")
        
        self.encoder = DepthHist_encoder(args)
        self.decoder = DepthHist_decoder(args)
    
        self.simple = args.simple
        

        # Number of bins for the output
        self.bins = args.bins

        # Initialize the output convolution layer and histogram if not in simple mode
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, 1 , kernel_size=1, stride=1, padding=1),
            nn.ReLU() 
        ) if self.simple else nn.Identity()

        # If not in simple mode, initialize the histogram layer
        if not self.simple:
            print(f"This model will use {self.bins} bins")
            self.Histogram = HistLayer(args)
        else : 
            print(f"This model will use a basic Encoder Decoder Arch without bining")


    def forward(self, x, **kwargs):
        if self.args.backbone == 'NewCRFB' or self.args.backbone == 'NewCRFL':
            decoded, bn = self.model(x)
            return self._compute_output(x, bn, decoded)
        else :
            # Pass through the encoder and decoder
            encoded = self.encoder(x)
            decoded = self.decoder(encoded, **kwargs)
            # Return the appropriate output based on whether the model is in simple mode or not
            return self._compute_output(x, encoded[-1], decoded)

    def _compute_output(self, rgb, bn, decoded):
        if self.simple:
            unet_out = self.conv_out(decoded)
            return unet_out
        else:
            # Compute histogram and depth for non-simple mode
            histogram, centers = self.Histogram(rgb, bn, decoded)
            centers = centers * (self.args.max_depth - self.args.min_depth)
            depth   = torch.cumsum(centers * histogram,  dim=1)[:,-1:,...]
            return depth, histogram, centers

    @classmethod
    def build(cls, args, **kwargs):
        # Method to build the model with the appropriate backbone
        print('Building Encoder --- Decoder model .....', end='\n')
        print(f'\n\nLoading base model {args.backbone} \n', end=' ')
        print('Done.')
        # Instantiate the class with the appropriate configuration
        return cls(args, **kwargs)

    def get_1x_lr_params(self):
        # Return parameters of the encoder for lower learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):
        # Return parameters of the decoder and other layers for higher learning rate
        modules = [self.decoder, self.conv_out]
        if not self.simple:
            modules.append(self.Histogram)
        for m in modules:
            yield from m.parameters()

    
    def set_seed(self, seed):
        """
            Set the random seed for reproducibility.

            Parameters:
            seed (int): The seed to use for all random number generators.

            Notes:
            - This is important for reproducibility, as it ensures that the model will
            always produce the same results given the same inputs.
            - Also useful for debugging, as it will allow you to isolate any issues
            that are due to randomness.
        """
        torch.manual_seed(seed)
        # If using CUDA
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

