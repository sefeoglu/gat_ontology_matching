import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticAttention(nn.Module):

    def __init__(self, in_size, hidden_size=768):

        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):

        w = self.project(z).mean(0)                 # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        result =  (beta * z)                     # (N, D * K)
        return result
