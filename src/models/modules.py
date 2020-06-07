import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import initialize_weights

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1,
                            self.conv2,
                            self.conv3,
                            self.conv4,
                            self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ResidualDenseBlock_3C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_3C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1,
                            self.conv2,
                            self.conv3], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        return x3 * 0.2 + x


class ConvReluConv(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ConvReluConv, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1,
                            self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class UpsamplingBlockx4(nn.Module):
    def __init__(self, in_nf, bias, custom_init=False):
        super(UpsamplingBlockx4, self).__init__()
        if in_nf < 3*16:
            raise Exception('in_nf < 48')

        nf = 48
        self.conv0 = nn.Conv2d(in_nf, nf, 3, 1, 1, bias=bias)
        self.conv1 = nn.Conv2d(nf//4, nf//4, 3, 1, 1, bias=bias)
        self.relu = nn.ReLU(inplace=True)

        self.up0 = nn.PixelShuffle(2)
        self.up1 = nn.PixelShuffle(2)

        # initialization
        initialize_weights([self.conv0, self.conv1], 0.1)

    def forward(self, x):

        x = self.conv0(x)
        x = self.relu(x)

        x = self.up0(x)
        x = self.conv1(x)

        x = self.up1(x)
        #no need for another conv
        return x

class Upscalex4V2(nn.Module):
    def __init__(self, nf):
        super(Upscalex4V2, self).__init__()

        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights([self.upconv1,
                            self.upconv2], 0.1)


    def forward(self, x):
        x = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
        x = self.lrelu(self.pixel_shuffle(self.upconv2(x)))
        return x

class NearestNeighbourx4(nn.Module):
    def __init__(self, nf, bias, custom_init=False):
        super(NearestNeighbourx4, self).__init__()
        self.conv0 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)
        self.relu = nn.ReLU(inplace=True)

        #initialization
        if custom_init:
            for conv in [self.conv0, self.conv1, self.conv2]:
                torch.nn.init.kaiming_normal_(conv.weight)

    def forward(self, x):
        x = self.relu(self.conv0(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.relu(self.conv1(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.relu(self.conv2(x))

        return x




#class MyModel_debug_0_1(SR_Model):
#    def __init__(self, in_c, out_c, nf, n_blocks, n_CRC, bias=False, custom_init=False):
#        super(MyModel_debug, self).__init__()
#        self.name = "MyNet_debug_{}_{}_{}".format(nf, n_blocks, n_CRC)
#
#        self.first_conv = nn.Conv2d(in_c, nf, 3, 1, 1, bias=bias)
#        self.block1 = ResidualDenseBlock_5C(nf=64, gc=32, bias=True, learnable_coef=False)
#        self.up = UpsamplingBlockx4(nf, bias=bias)
#
#        # initialization
#        if custom_init:
#            for conv in [self.first_conv]:
#                torch.nn.init.kaiming_normal_(conv.weight)
#
#    def forward(self, inputs):
#
#        x = self.first_conv(inputs) #no relu
#        x = self.block1(x)
#        x = self.up(x)
#
#        return x












