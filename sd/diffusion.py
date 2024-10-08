import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention,CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self,n_embd: int):
        super().__init__()
        # 320 -> 1280
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, 4*n_embd)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (1,320)
        
        x = self.linear_1(x)
        
        x = F.silu(x)
        
        x = self.linear_2(x)
        
        # (1,1280)
        return x
    
class UNET_ResidualBlock(nn.Module):
    def __init__(self,in_channels: int,out_channels: int,n_time =1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32,in_channels)
        self.conv_feature = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.linear_time = nn.Linear(n_time,out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(32,out_channels)
        self.conv_merged = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
        
    def forward(self,feature,time):
        # feature: (batch_size, in_channels, height, width)
        # time: (1,1280)
        
        residue = feature
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)
        
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)
        
        
        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residue)
    
class UNET_AttentionBlock(nn.Module):
    def __init__(self,n_head:int, n_embd:int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
    
        self.groupnorm = nn.GroupNorm(32,channels,eps=1e-6)
        self.conv_input = nn.Conv2d(channels,channels,kernel_size=1,padding=0)
        
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head,channels,in_proj_bias =False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head,channels,d_context,in_proj_bias =False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels,4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels,channels)
        
        self.conv_output = nn.Conv2d(channels,channels ,kernel_size = 1,padding=0)
        
    def forward(self,x,context):
        # x: (batch_size,features,height,width)
        # context: (batch_size,seq_len,Dim)
        
        residue_long = x
        
        x = self.groupnorm(x)
        
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (batch_size,features,height,width) -> (batch_size,features,height*width)
        x = x.view((n,c,h*w))
        
        # (batch_size,features,height*width) -> (batch_size,height*width,features,)
        x = x.transpose(-1,-2)
        # Normalization + self attention with skip connections
        residue_short = x
        
        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residue_short(x)
        
        residue_short = x
        
        # normalization + cross attention with skip connections
        x = self.layernorm_2(x)
        
        # cross attention
        self.attention_2(x,context)
        
        x += residue_short
        
        # normalization + feedforward with GeGLU and skip connections
        residue_short = x
        
        x = self.layernorm_3(x)
        
        x, gate = self.linear_geglu_1(x).chunk(2,dim=-1)
        x = x* F.gelu(gate)
        
        x = self.linear_geglu_2(x)
        
        x += residue_short
        
        # (batch_size,features,height*width) -> (batch_size,features,height*width)
        x = x.transpose(-1,-2)
        
        # (batch_size,features,height*width) -> (batch_size,features,height,width)
        x = x.view((n,c,h,w))
        
        return self.conv_output(x) + residue_long
        
            
    
class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    
    def forward(self, x):
        # (batch_size, Featues, height, width) -> (batch_size, Featues, height*2, width*2)
        x = F.interpolate(x,scale_factor=2,mode="nearest")
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x:torch.Tensor, contxt: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            # calculate cross attention b/w latent and prompt
            if isinstance(layer,UNET_AttentionBlock):
                x = layer(x,contxt)
            elif isinstance(layer,UNET_ResidualBlock):
                x = layer(x,time)   
            else:
                x = layer(x)
                
        return x 
            
        

class UNET(nn.Module): #encoding,decoding,seitch connection, bottleneck
    def __init__(self):
        super().__init__()
        
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])
        
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280,1280),
            
            UNET_AttentionBlock(8,160),
            
            UNET_ResidualBlock(1280,1280),
            
        )
        
        self.decoders = nn.ModuleList([
            # o/p from bottleneck = 1280
            # o/p from skip connection from encoder = 1280
            # so decoder i/p = 2560
            # (batch_size,2560,height/64,width/64) -> (batch_size,1280,height/64,width/64)
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560,1280), UpSample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560,1280),UNET_AttentionBlock(8,160)),
            
            SwitchSequential(UNET_ResidualBlock(2560,1280),UNET_AttentionBlock(8,160)),
            
            SwitchSequential(UNET_ResidualBlock(1920,1280),UNET_AttentionBlock(8,160),UpSample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(1920,640),UNET_AttentionBlock(8,80)),
            
            SwitchSequential(UNET_ResidualBlock(1280,640),UNET_AttentionBlock(8,80)),
            
            SwitchSequential(UNET_ResidualBlock(960,640),UNET_AttentionBlock(8,80),UpSample(640)),
            
            SwitchSequential(UNET_ResidualBlock(960,320),UNET_AttentionBlock(8,40)),
            
            SwitchSequential(UNET_ResidualBlock(640,320),UNET_AttentionBlock(8,40)),
            
            SwitchSequential(UNET_ResidualBlock(640,320),UNET_AttentionBlock(8,40)),
        ])
    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x        


class UNET_OutputLayer(nn.Module):
    def __init__(self,in_channels: int,out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)
        
    def forward(self,x):
        # x: (batch_size,320,height/8,width/8)
        x = self.groupnorm(x)
        
        x =F.silu(x)
        
        x = self.conv(x)
        
        # (batch_size,4,height/8,width/8)
        return x
        


# UNET model
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        # size of time embd = 320
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320,4)
        
    def forward(self,latent: torch.Tensor, content: torch.Tensor, time: torch.Tensor):
        #  4 =output of encoder
        # latent: (batch_size,4,height/2,width/2)
        # context: (batch_size,seq_len,Dim)
        # time: (1,320)
        
        # (1,320) -> (1,1280)
        time = self.time_embedding(time)
        
        # (batch,4,height/8,width/8) -> (batch,320,height/8,width/8)
        output = self.unet(latent, content, time)
        
        # (batch,320,height/8,width/8) -> (batch,4,height/8,width/8)
        output = self.final(output)
        
        # (batch,4,height/8,width/8)
        return output
        