"""
This is transformers from scratch implementation with pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embeddings(nn.Module):
    def __init__(self, num_embed, dim):
        # super().__init__(*args, **kwargs)
        # self.num_embed = num_embed
        # self.seq_len = seq_len
        # self.dim = dim
        self.embedding = nn.Embedding(num_embed, dim)

    # def tokenization(self, num_embed: int, dim: int):
    #     """
    #     This function does general tokenization of words

    #     input:
    #         num_embed:  This signifies the number of words you want to have in your vocab. 
    #                     For example, if you set it to 100, You want to train this mini LLM with only 100 words.

    #         dim: This handles the dimension of each of these words. For example, if you set it as 512, then each word in your 
    #             sentence or paragraph gets converted into a vector with 512 dimension. 
        
    #     Returns:
    #         Embedding: A embedding matrix of size (number of words in input x dim)
    #     """
    #     input_embeddings = nn.Embedding(num_embeddings=num_embed, embedding_dim=dim)
    #     return input_embeddings

    def forward(self, x):
        return self.embedding(x)
    
    def pos_embedding(self, seq_len, d_model):
        """
        This function adds the positional information. There are plenty of methods to implement positional encoding. 
        But I have implemented sinusoidal based method. 

        Input:

        """

        # Create a position ID
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000)/d_model))

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        return pe
    
class SelfAttn(nn.Module):
    """
    This performs the self attention. This is implemented for single head only. 
    """
    def __init__(self, d_model, d_k):
        super().__init__()

        self.wq = nn.Linear(d_model, d_k)
        self.wk = nn.Linear(d_model, d_k)
        self.wv = nn.Linear(d_model, d_k)

    def forward(self, x):
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        scores = Q @ K.T
        scores /= math.sqrt(Q.size(-1))

        weights = F.softmax(scores, dim=-1)
        outputs = weights @ V
        return outputs

class MultiHeadAttn(nn.Module):
    """
    This class performs the multi head attention.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)


    def forward(self, x):
        seq_len = x.shape[0]

        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        Q = Q.view(seq_len, self.num_heads, self.d_k)
        K = K.view(seq_len, self.num_heads, self.d_k)
        V = V.view(seq_len, self.num_heads, self.d_k)

        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)

        # Calculate the attention scores

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_k)

        weights = F.softmax(scores, dim=-1)
        outputs = weights @ V

        # Combine the heads
        outputs = outputs.transpose(0, 1).continguous()
        outputs = outputs.view(seq_len, self.d_model)

        outputs = self.wo(outputs)

        return outputs

class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm(ffn_out)
        return x

# class Decoder

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attn = MultiHeadAttn(d_model=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FFN(d_model=d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x+attn_out)
        ffn_out = self.ffn.ffn(x)
        x = self.norm2(x+ffn_out)

        return x