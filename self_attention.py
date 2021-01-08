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

