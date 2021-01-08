import torch 
from torch import nn
import torch.nn.functional as F 


class SelfAttention(nn.Module):
    def __init__(self, k, head=8):
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

            @param x: matrix of size 
                    (b:batch, t: vector, k: t_dimensions)

            
        """
        b, t, k = x.size()
        h = self.heads
        assert k == self.k, f'Input vector dim [{k}] should match layer dimension [{self.k}]'


        keys = self.tokeys(x).view(b, t, h, k)
        queries = self.toqueries(x).view(b, t, h, k)
        values = self.tovalues(x).view(b, t, h, k)
        # reshape output of linear transform from 
            # (b,t,h*k) tp b,t,h,k 


        # bring back torch.bmm() to compute dot products


















