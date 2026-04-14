import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embeddings(nn.Module):
    def __init__(self, num_embed, dim, seq_len):
        # super().__init__(*args, **kwargs)
        self.num_embed = num_embed
        self.seq_len = seq_len
        self.dim = dim

    def tokenization(self, num_embed: int, dim: int):
        """
        This function does general tokenization of words

        input:
            num_embed:  This signifies the number of words you want to have in your vocab. 
                        For example, if you set it to 100, You want to train this mini LLM with only 100 words.

            dim: This handles the dimension of each of these words. For example, if you set it as 512, then each word in your 
                sentence or paragraph gets converted into a vector with 512 dimension. 
        
        Returns:
            Embedding: A embedding matrix of size (number of words in input x dim)
        """
        input_embeddings = nn.Embedding(num_embeddings=num_embed, embedding_dim=dim)
        return input_embeddings
    
    def pos_embedding(self, seq_len, d_model):
        """
        This function adds the positional information. There are plenty of methods to implement positional encoding. 
        But I have implemented sinusoidal based method. 

        Input:

        """

        # Create a position ID
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, seq_len, 2)*(-math.log(10000)/d_model))

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        return pe
    
class SelfAttn(nn.Module):
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
        scores /= math.sqrt(Q.shape(-1))

        weights = F.softmax(scores, dim=-1)
        outputs = weights @ V
        return outputs

class MultiHeadAttn(nn.Module):
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