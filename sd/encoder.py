import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock   # from decoder.py file 


class VAE_Encoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            # (batch_size,channels,height,width) -> (batch_size,128,height,width) as we use padding =1
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            
            # (batch_size,128(i/p),height,width) -> (batch_size,128(o/p),height,width) residual block not change size
            VAE_ResidualBlock(128,128),
            # (batch_size,128(i/p),height,width) -> (batch_size,128(o/p),height,width) residual block not change size
            VAE_ResidualBlock(128,128),
            
            # skips 2 pxls as stride =2
            # (batch_size,128,height,width) -> (batch_size,128,height/2 ,width/2)
            # size of image is decreasing
            nn.Conv2d(128,128,kernel_size=3,padding=0,stride=2),
            
            # (batch_size,128(i/p),height/2 ,width/2) -> (batch_size,256(o/p),height/2 ,width/2) 
            # no of features increasing represented by a pxl
            VAE_ResidualBlock(128,256),
            # (batch_size,256(i/p),height/2 ,width/2) -> (batch_size,256(o/p),height/2 ,width/2) residual block not change size
            VAE_ResidualBlock(256,256),
            
            # (batch_size,256,height/2 ,width/2) -> (batch_size,256,height/4,width/4)
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),
            
            # (batch_size,256(i/p),height/4 ,width/4) -> (batch_size,512(o/p),height/4 ,width/4) 
            VAE_ResidualBlock(256,512),
            # (batch_size,512(i/p),height/4 ,width/4) -> (batch_size,512(o/p),height/4 ,width/4) 
            VAE_ResidualBlock(512,512),
            
            # (batch_size,512,height/4,width/4) -> (batch_size,512,height/8,width/8)
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),
            
            VAE_ResidualBlock(512,512),
            
            VAE_ResidualBlock(512,512),
            
            # (batch_size,512,height/8,width/8) ->(batch_size,512,height/8,width/8)
            VAE_ResidualBlock(512,512),
            
            VAE_AttentionBlock(512),
            
            # (batch_size,512,height/8,width/8) ->(batch_size,512,height/8,width/8)
            VAE_ResidualBlock(512,512),
            
            # (batch_size,512,height/8,width/8) ->(batch_size,512,height/8,width/8)
            nn.GroupNorm(32,512),
            # (batch_size,512,height/8,width/8) ->(batch_size,512,height/8,width/8)
            nn.SiLU(),
            # (batch_size,512,height/8,width/8) ->(batch_size,8,height/8,width/8)
            nn.Conv2d(512,8,kernel_size=3,padding=1),
            # (batch_size,8,height/8,width/8) ->(batch_size,8,height/8,width/8)
            nn.Conv2d(8,8,kernel_size=1,padding=0)
            
            
            )
    
    def forward(self,x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        #  x: (batch_size,channel,height,width)
        #  noise: (batch_size, output_channel ,height/8,width/8)
        
        for module in self:
            if getattr(module, 'stride',None) == (2,2):
                # (padding_left,padding_right,padding_top,padding_bottom)
                # asymetrical padding when stride
                x = F.pad(x, (0,1,0,1))
            
            x =module(x)    
            
        # variatonal autoencoder o/p is mean and log of variance
        # (batch_size,8,height/8,width/8) -> (chunk) two tensors of shape (batch_size,4,height/8,width/8)
        mean, log_variance =torch.chunk(x,2,dim=2)
        
        # (batch_size,4,height/8,width/8) define a range of log variance
        log_variance = torch.clamp(log_variance,-30,20)
        
        # (batch_size,4,height/8,width/8)  
        variance = log_variance.exp()
        
        # (batch_size,4,height/8,width/8)
        stdev = variance.sqrt()
        
        # z = N(0,1) -> N(mean,var) =x?  how do we sample from it from a given mean and var 
        # x = mean + stdev * z  formula for transformation 
        x = mean + stdev *noise
        
        # scale the o/p by a const
        x *= 0.18215
        
        return x
            
            
            
            
            
            