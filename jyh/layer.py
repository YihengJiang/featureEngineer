#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch as tc
import math

__all__ = ['L2Norm', 'SignedSqrt',
           'BilinearPool', 'BilinearWithNorm',
           'WCenters',
           'ConvNorm']


class L2Norm(nn.Module):
    """L2Norm layer."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class SignedSqrt(nn.Module):
    """Signed square root."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # It was said that torch.sign() couldn'tc perform properly.
        # x = torch.sign(x) * torch.sqrt(torch.abs(x))
        return torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))


class BilinearPool(nn.Module):
    """Bilinear pooling layer."""

    def __init__(self, reshape=True):
        super().__init__()

        self.reshape = reshape

    def forward(self, x, y):
        bs, ch_x, h, w = x.size()
        ch_y = y.size(1)

        out = torch.bmm(x.view(bs, ch_x, -1),
                        y.view(bs, ch_y, -1).permute(0, 2, 1)) / (h * w)

        return out.view(bs, -1) if self.reshape else out


class BilinearWithNorm(nn.Module):
    """Bilinear pooling layer with signed sqrt and l2-normalization."""

    def __init__(self):
        super().__init__()

        self.bilinear = BilinearPool(reshape=True)
        self.sqrt = SignedSqrt()
        self.l2norm = L2Norm()

    def forward(self, x, y):
        x = self.bilinear(x, y)
        x = self.sqrt(x)
        x = self.l2norm(x)

        return x


class WCenters(nn.Module):
    """Calculate assign * mu with dim expansion, propagate to assign only."""

    def __init__(self, centers, dmu=True):
        super().__init__()

        self.centers = centers
        self.dmu = dmu

    def forward(self, assign):
        bs, k, h, w = assign.size()

        centers = self.centers if self.dmu else Variable(self.centers.data)
        return assign.mean(3).mean(2).view(bs, 1, k) * centers.view(1, -1, k)


class ConvNorm(nn.Conv2d):
    """2D convolution layer with normalization over weight."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, dw=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)

        self.norm = L2Norm()
        self.dw = dw

    def forward(self, x):
        weight = self.weight
        if self.dw is False:
            weight = Variable(self.weight.data)

        weight = self.norm(weight)

        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


'''
Description:
The model here implements a generalized version of the TDNN based on the model descriptions given in [1],[2].
In the description given by Waibel et al. a TDNN uses the full context specified. For example: if the delay specified is N = 2, the model uses the current frame and frames at delays of 1 and 2.
In the description given by Peddinti et al. the TDNN only looks at the farthest frames from the current frame as specified by the context parameter. The description in section 3.1 of the paper discusses the differences between their implementation and Waibel et al.
The TDNN implemented here allows for the usage of an arbitrary context which shall be demonstrated in the usage code snippet.

Usage:
For the model specified in the Waibel et al. paper, the first layer is as follows:
context = [0,2]
input_dim = 16
output_dim = 8
net = TDNN(context, input_dim, output_dim, full_context=True)

# For the model specified in the Peddinti et al. paper, the second layer is as follows (taking arbitrary I/O dimensions since it's not specified):
context = [-1,2]
input_dim = 16
output_dim = 8
net = TDNN(context, input_dim, output_dim, full_context=False)

# You may also use any arbitrary context like this:
context = [-11,0,5,7,10]
nput_dim = 16
output_dim = 8
net = TDNN(context, input_dim, output_dim, full_context=False)
# The above will convole the kernel with the current frame, 11 frames in the past, 5, 7, and 10 frames in the future.
output = net(input) # this will run a forward pass
'''

"""Time Delay Neural Network as mentioned in the 1989 paper by Waibel et al. (Hinton) and the 2015 paper by Peddinti et al. (Povey)"""


class TDNN(nn.Module):
    def __init__(self, context, input_dim, output_dim, full_context=True):
        """
        Definition of context is the same as the way it's defined in the Peddinti paper. It's a list of integers, eg: [-2,2]
        By deault, full context is chosen, which means: [-2,2] will be expanded to [-2,-1,0,1,2] i.e. range(-2,3)
        """
        super(TDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.check_valid_context(context)
        self.kernel_width, context = self.get_kernel_width(context, full_context)
        self.register_buffer('context', tc.LongTensor(context))
        self.full_context = full_context
        stdv = 1. / math.sqrt(input_dim)
        self.kernel = nn.Parameter(tc.Tensor(output_dim, input_dim, self.kernel_width).normal_(0, stdv))
        self.bias = nn.Parameter(tc.Tensor(output_dim).normal_(0, stdv))
        # self.cuda_flag = False

    def forward(self, x):
        """
        x is one batch of data
        x.size(): [batch_size, sequence_length, input_dim]
        sequence length is the length of the input spectral data (number of frames) or if already passed through the convolutional network, it's the number of learned features
        output size: [batch_size, output_dim, len(valid_steps)]
        """
        # Check if parameters are cuda type and change context
        # if type(self.bias.data) == torch.cuda.FloatTensor and self.cuda_flag == False:
        #     self.context = self.context.cuda()
        #     self.cuda_flag = True
        conv_out = self.special_convolution(x, self.kernel, self.context, self.bias)
        return F.relu(conv_out)

    def special_convolution(self, x, kernel, context, bias):
        """
        This function performs the weight multiplication given an arbitrary context. Cannot directly use convolution because in case of only particular frames of context,
        one needs to select only those frames and perform a convolution across all batch items and all output dimensions of the kernel.
        """
        input_size = x.size()
        assert len(input_size) == 3 and input_size[
                                            2] == self.input_dim, 'Input tensor Should be a 3D tensor and the last dim==TDNN.input_dim'
        [batch_size, input_sequence_length, input_dim] = input_size
        x = x.transpose(1, 2).contiguous()

        # Allocate memory for output
        valid_steps = self.get_valid_steps(self.context, input_sequence_length)
        xs = Variable(self.bias.data.new(batch_size, kernel.size()[0], len(valid_steps)))

        # Perform the convolution with relevant input frames
        for c, i in enumerate(valid_steps):
            features = tc.index_select(x, 2, Variable(context + i))
            xs[:, :, c] = F.conv1d(features, kernel, bias=bias)[:, :, 0]
        return xs

    @staticmethod
    def check_valid_context(context):
        # here context is still a list
        assert context[0] <= context[-1], 'Input context context is incorrect. Should be ascending'

    @staticmethod
    def get_kernel_width(context, full_context):
        if full_context:
            context = range(context[0], context[-1] + 1)
        return len(context), context

    @staticmethod
    def get_valid_steps(context, input_sequence_length):
        start = 0 if context[0] >= 0 else -1 * context[0]
        end = input_sequence_length if context[-1] <= 0 else input_sequence_length - context[-1]
        # if end<start:
        #     raise Exception("error,in get_valid_steps")
        assert end > start, "error,in get_valid_steps"
        return range(start, end)


class Maxout_BaseLinear(nn.Module):
    def __init__(self, pool_size, d_in, d_out):  # pool_size is number of middle units
        super(Maxout_BaseLinear, self).__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        # shape = list(inputs.size())
        # shape[-1] = self.d_out
        # shape.append(self.pool_size)
        # last_dim = len(shape) - 1
        out = self.lin(inputs)
        out = out.view(out.size(0), self.d_out, self.pool_size)
        return out.max(-1)[0]


class Maxout_BaseConv2d(nn.Module):
    def __init__(self, pool_size, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        self.out_channels = out_channels
        self.pool_size = pool_size
        super(Maxout_BaseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * pool_size, kernel_size, stride,
                              padding, dilation, groups, bias)

    def forward(self, inputs):
        # shape = list(inputs.size())
        # shape[-1] = self.out_channels
        # shape.append(self.pool_size)
        # last_dim = len(shape) - 1
        out = self.conv(inputs)
        out = out.view(out.size(0), self.out_channels, self.pool_size, out.size(2), out.size(3))

        return out.max(2)[0]


class Maxout_BaseLinear2(nn.Module):
    def __init__(self, pool_size, d_in, d_out):  # pool_size is number of middle units
        super(Maxout_BaseLinear2, self).__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.fcList = nn.ModuleList([nn.Linear(d_in, d_out) for i in range(pool_size)])
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, x):
        max_output = self.fcList[0](x)
        for _, layer in enumerate(self.fcList, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output


class Maxout_BaseConv2d2(nn.Module):
    def __init__(self, pool_size, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        self.out_channels = out_channels
        self.pool_size = pool_size
        super(Maxout_BaseConv2d2, self).__init__()
        self.convList = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                                 padding, dilation, groups, bias) for i in range(pool_size)])

    def forward(self, x):
        max_output = self.fcList[0](x)
        for _, layer in enumerate(self.fcList, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output


if __name__ == '__main__':
    # m=Maxout_BaseConv2d(3,1,2,3)
    # input = tc.autograd.Variable(torch.randn(1,1,4,4))
    # output = m(input)

    m = Maxout_BaseLinear(3, 2, 2)
    input = tc.autograd.Variable(torch.randn(1, 2))
    output = m(input)

    # context = [0, 2]
    # input_dim = 3
    # output_dim = 3
    # net = TDNN(context, input_dim, output_dim, full_context=False)
    # x=Variable(tc.randn(1, 4, 3))
    # print(net(x).size())
