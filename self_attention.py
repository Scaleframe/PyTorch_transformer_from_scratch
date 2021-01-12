import torch 
from torch import nn
import torch.nn.functional as F 

import random, math

class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        """ Initializes parameters for input 
                sequence to transformer model

        Input is a sequence of t vectors 
                            of dimension k.

        @param k (int): number of dimensions 
                            for input vector.

        @param head (int): number of separate
                query, key, value tranforms per
                input vector.
        """
        super().__init__()
        
        self.k, self.heads = k, heads

    
        # More efficient to work on 3 matrices 
        # at once instead of query,key, value 
        # operations per head in sequence. 

        # Compute queries, keys, values for all 
        # heads (as single concatenated vector)
        self.tokeys = nn.Linear(k, k* heads, bias=False)
        self.toqueries = nn.Linear(k, k* heads, bias=False)
        self.tovalues = nn.Linear(k, k* heads, bias=False)


        self.unifyheads = nn.Linear(heads * k, k)


    def forward(self, x):
        """ Compute self attention 
            from an input vector. 

            @param x: matrix of size (b:batch, t: vector, k: t_dimensions)

            
        """
        b, t, k = x.size()
        h = self.heads
        assert k == self.k, f'Input vector dim [{k}] should match layer \
                                                    dimension [{self.k}]'


        keys = self.tokeys(x).view(b, t, h, k)
        queries = self.toqueries(x).view(b, t, h, k)
        values = self.tovalues(x).view(b, t, h, k)
        # reshape output of linear transform from 
            # (b,t,h*k) tp b,t,h,k 


        # dot product is next, to do that we need to
        # get the batch dimension next to the head so
        # that our operation works on (batch, head, vector, 
        # dimension). 
    
        keys = keys.transpose(1,2).contiguous().view(b* h, t, k)
        queries = queries.transpose(1,2).contiguous().view(b* h, t, k)
        values = values.transpose(1,2).contiguous().view(b* h, t, k)


        # scale back the keys and queries to save memory on the 
                                        # dot product computation. 
        keys = keys / (k ** (1/4))
        queries = queries / (k ** (1/4))
        
        # dot product operation:
        dot = torch.bmm(queries, keys.transpose(1, 2))


        # normalize weights with softmax
        dot = F.softmax(dot, dim=2)
        
        assert dot.size() == (b*h, t, t), f"weight matrix has size of {dot.size()}, expected {b*h, t, t}."

        # apply dot product to the values=
        out = torch.bmm(dot, values).view(b,h, t, k)
        
        # get the head and embedding dimension next to each other, 
                        # then transfrom for shape (b, t, h * k)

        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        

        # linear transform back down to h*k, k dimensions

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    
    def __init__(self, k, heads):
        super().__init__()

        # add self attention layer 

        self.attention = SelfAttention(k, heads=heads)

        # add layer normalization

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        
        # add feedforward layer

        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLu(),
            nn.Linear(4 * k, k))


        