import torch 
import torch.nn.functional as F 

# assume we have some tensor x with size (b, t, k)
x = ...


raw_weights = torch.bmm(x, x.transpose(1,2))
    # torch.bmm is batched matmul. 

# our goal is to get a weighted sum over all our embedding vectors in the input sequence. 

weights = F.softmax(raw_weights, dim=2)
    # this gives us values that sum to one


# comput the output sequence
y = torch.bmm(weights, x)

# two matrix muliplications and one softmax gives us self attention. 


