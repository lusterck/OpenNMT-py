import torch.nn as nn
import torch
import torch.cuda
import math
from torch.autograd import Variable

class MatrixTree(nn.Module):
    """Implementation of the matrix-tree theorem for computing marginals
    of non-projective dependency parsing. This attention layer is used
    in the paper "Learning Structured Text Representations."
    """
    def __init__(self, eps=1e-5):
        self.eps = eps
        super(MatrixTree, self).__init__()

    def forward(self, input):
        laplacian = input.exp() + self.eps
        output = input.clone()
        for b in range(input.size(0)):
            lap = laplacian[b].masked_fill(
                Variable(torch.eye(input.size(1)).cuda().ne(0)), 0)
            lap = -lap + torch.diag(lap.sum(0))
            # store roots on diagonal
            lap[0] = input[b].diag().exp()
            inv_laplacian = lap.inverse()

            factor = inv_laplacian.diag().unsqueeze(1)\
                                         .expand_as(input[b]).transpose(0, 1)
            term1 = input[b].exp().mul(factor).clone()
            term2 = input[b].exp().mul(inv_laplacian.transpose(0, 1)).clone()
            term1[:, 0] = 0
            term2[0] = 0
            output[b] = term1 - term2
            roots_output = input[b].diag().exp().mul(
                inv_laplacian.transpose(0, 1)[0])
            output[b] = output[b] + torch.diag(roots_output)
        return output

    def forward(self, input,lengths):
        laplacian = input.exp() + self.eps
        output = input.clone()
        output.data.fill_(0)
        for b in range(input.size(0)):
            lx = lengths[b] if lengths is not None else input.size(1)
            input_b = input[b, :lx, :lx]
            lap = laplacian[b, :lx, :lx].masked_fill(Variable(torch.eye(lx).cuda().ne(0)), 0)
            lap = -lap + torch.diag(lap.sum(0))
            # store roots on diagonal
            lap[0] = input_b.diag().exp()
            inv_laplacian = lap.inverse()

            factor = inv_laplacian.diag().unsqueeze(1)\
                                         .expand(lx, lx).transpose(0, 1)
            term1 = input_b.exp().mul(factor).clone()
            term2 = input_b.exp().mul(inv_laplacian.transpose(0, 1)).clone()
            term1[:, 0] = 0
            term2[0] = 0
            output_b = term1 - term2
            roots_output = input_b.diag().exp().mul(
                inv_laplacian.transpose(0, 1)[0])
            output[b, :lx, :lx] = output_b + torch.diag(roots_output)
        return output

class TreeAttention(nn.Module):
    """Structured attention class"""
    def __init__(self, dim, clamp_thres=15):
        super(TreeAttention, self).__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = math.sqrt(1 / dim)
        self.dtree = MatrixTree()
        self.clamp_thres = clamp_thres

    def forward(self, input, lengths=None):
        s_len, batch, dim = input.size()
        input = input.contiguous().transpose(0, 1) \
            .contiguous().view(-1, dim)
        print(self.q)
        print(input)
        q = self.q(input).view(batch, s_len, -1)
        k = self.k(input).view(batch, s_len, -1)
        v = self.v(input).view(batch, s_len, -1)
        score = torch.bmm(q, k.transpose(1, 2)) * self.scale
        score = score.clamp(-self.clamp_thres, self.clamp_thres)
        attn = self.dtree(score, lengths)
        return torch.bmm(attn.transpose(1, 2), v)

if __name__ == "__main__":
    dtree = MatrixTree()
    q = torch.rand(1, 5, 5).cuda()
    marg = dtree.forward(Variable(q))
    print(marg.sum(1))
