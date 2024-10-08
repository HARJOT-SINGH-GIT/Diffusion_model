import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self,n_vocab: int,n_embd: int,n_tokens: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab,n_embd)
        # parameters learned by model during training
        # tells position of token to model
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens,n_embd))
        
    def forward(self,tokens):
        # (batch_size,seq_len) -> (batch_size,seq_len,Dim)
        x = self.token_embedding(tokens)
        
        x += self.position_embedding
        
        return x


class CLIPLayer(nn.Module):
    
    def __init__(self,n_head: int,n_embd: int):
        super().__init__()
        
        self.layernorm_1 =nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head,n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd,4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd,n_embd)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # (batch_size,seq_len,Dim)
        
        residue = x
        
        ## SELF ATTENTION
        x = self.layernorm_1(x)
        # causal_mask -> token can only see tokens on it left 
        # can not watch future token of it
        x = self.attention(x,causal_mask =True)
        x +=residue
        
        ## FEEDFORWARD LAYER
        residue = x
        
        x = self.layernorm_2(x)
        
        x = self.linear_1(x)
        
        # quick grow function
        x = x*torch.sigmoid(1.702*x) # QuickGELU activation Function
        
        x = self.linear_2(x)
        x += residue
        
        return x
        

class CLIP(nn.Module):
    def __init__(self):
        # vocabulary size , embedding_size, max seq_len
        self.embedding = CLIPEmbedding(49408,768,77)
        
        self.layers = nn.Module([
            # no of heads of multi head attention, embedding size(12 layers)
            CLIPLayer(12,768) for i in range(12)
        ])
        
        self.layernorms = nn.LayerNorm(768)
        
    def forward(self,tokens: torch.LongTensor) -> torch.FloatTensor:
        # id's inside vocabulary so LongTensor
        tokens = tokens.type(torch.long)
        
        # (batch_size,seq_len) -> (batch_size,seq_len,Dim)
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
            
        # (batch_size,seq_len,Dim)
        output = self.layernorms(state)
        
        return output