# This code was written by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# Below is the code for the model
# Contact: medchaoukiziara@gmail.com || chaouki.ziara@univ-sba.dz


import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Encoder_EffNet, Decoder_EffNet, HistLayer
from .loader import DepthHist_encoder, DepthHist_decoder

import torch
import torch.nn as nn

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
    
    def __init__(self, backend, args):
        """
        Initializes the DepthHist model.

        Attributes:
            - backend: The backbone model used for encoding.
            - args: The arguments for the model, including the number of bins and whether to use the simple mode.
            - simple: A boolean flag indicating whether to use the simple mode.
        """
        super(DepthHist, self).__init__()
        # Initialize the encoder based on the backend provided
        self.args = args
        if args.path_pth_model == None :
            self.set_seed(22)
            print("\n\n\nThe weights will be initialized evenly\n\n\n")
        if backend is not None : 
            self.encoder = Encoder_EffNet(backend)
            self.decoder = Decoder_EffNet(num_classes=128)
        elif args.backbone == 'DepthHistB' or args.backbone == 'DepthHistL' :
            self.encoder = DepthHist_encoder(args)
            self.decoder = DepthHist_decoder(args)
        # Determine if the model is in simple mode or not
        self.simple = args.simple
        
        self.ind = 11 if  args.backbone == 'efficientnet' else -1
        # Number of bins for the output
        self.bins = args.bins

        # Initialize the output convolution layer and histogram if not in simple mode
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, 1 if self.simple else self.bins, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # If not in simple mode, initialize the histogram layer
        if not self.simple:
            print(f"This model will use {self.bins} bins")
            self.Histogram = HistLayer(args)
        else : 
            print(f"This model will use a basic Encoder Decoder Arch without bining")


    def forward(self, x, **kwargs):
            # Pass through the encoder and decoder
            encoded = self.encoder(x)
            decoded = self.decoder(encoded, **kwargs)
            # Return the appropriate output based on whether the model is in simple mode or not
            return self._compute_output(x, encoded[self.ind], decoded)

    def _compute_output(self, rgb, bn, decoded):
        if self.simple:
            unet_out = self.conv_out(decoded)
            return unet_out
        else:
            # Compute histogram and depth for non-simple mode
            histogram = self.Histogram(rgb, bn, decoded)
            decoded   = self.conv_out(decoded)
            depth     = torch.cumsum(decoded * histogram,  dim=1)[:,-1:,...] #keepdim=True,
            return depth, histogram #, decoded

    @classmethod
    def build(cls, args, **kwargs):
        # Method to build the model with the appropriate backbone
        print('Building Encoder --- Decoder model .....', end='\n')
        if args.backbone == "efficientnet":
            basemodel_name = 'tf_efficientnet_b5_ap'
            print(f'\n\nLoading base model {basemodel_name}\n', end=' ')
            basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True, verbose=False)
            print('Done.')

            # Remove global pooling and classifier layers
            basemodel.global_pool = nn.Identity()
            basemodel.classifier = nn.Identity()
        else :
            print(f'\n\nLoading base model {args.backbone} \n', end=' ')
            print('Done.')
            basemodel = None
            
        # Instantiate the class with the appropriate configuration
        return cls(basemodel, args, **kwargs)

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

