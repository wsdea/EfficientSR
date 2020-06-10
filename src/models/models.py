import os
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules  import RRDB, ConvReluConv, Upscalex4V2, ResidualDenseBlock_3C, UpsamplingBlockx4
from .utils    import make_layer, initialize_weights
from .SRResNet import MSRResNet

class CustomModule(torch.nn.Module):
    def __init__(self):
        super(CustomModule, self).__init__()

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def save_weights(self, folder, name):
        if not name.endswith('.pt'):
            name += '.pt'

        torch.save(self.state_dict(), os.path.join(folder, name))

    def trainable_params(self):
        return sum(map(lambda x: x.numel(), self.parameters()))


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        self.name = "RRDBNet_{}_{}_{}".format(nf, nb, gc)
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv_first,
                            self.trunk_conv,
                            self.upconv1,
                            self.upconv2,
                            self.HRconv,
                            self.conv_last], 0.1)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


class FasterMSRResNet(CustomModule):
    def __init__(self, nf=64, nb=16):
        super(FasterMSRResNet, self).__init__()
        self.name = "FasterMSRResNet"
        in_nc = 3
        self.YUV_first_conv = nn.Conv2d(in_nc, in_nc, 1, 1, 0, bias=True)

        #Y branch
        self.Y_first_conv  = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        CRC = functools.partial(ConvReluConv, nf=nf)
        self.Y_CRC_trunk = make_layer(CRC, nb)
        self.Y_up  = Upscalex4V2(nf)
        self.Y_HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.Y_conv_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)

        #UV branch
        self.UV_first_conv = nn.Conv2d(2, nf, 3, 1, 1, bias=True)
        self.UV_up  = Upscalex4V2(nf)
        self.UV_HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.UV_conv_last = nn.Conv2d(nf, 2, 3, 1, 1, bias=True)

        #final steps
        self.YUV_final_conv = nn.Conv2d(in_nc, in_nc, 1, 1, 0, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights([
                            self.YUV_first_conv,
                            self.Y_first_conv,
                            self.Y_HRconv,
                            self.Y_conv_last,
                            self.UV_first_conv,
                            self.UV_HRconv,
                            self.UV_conv_last,
                            self.YUV_final_conv,
                            ], 0.1)

#        pretrained_trunk = baseline_model(True).recon_trunk
#        self.Y_CRC_trunk.load_state_dict(pretrained_trunk.state_dict())

    def forward(self, inputs):
        x = self.YUV_first_conv(inputs)

        Y = x[:, [0]] #shape torch.Size([batch_size, 1, crop_size, crop_size])
        Y = self.lrelu(self.Y_first_conv(Y))
        Y = self.Y_CRC_trunk(Y)
        Y = self.Y_up(Y)
        Y = self.Y_conv_last(self.lrelu(self.Y_HRconv(Y)))

        UV = x[:, 1:]  #shape torch.Size([batch_size, 2, crop_size, crop_size])
        UV = self.lrelu(self.UV_first_conv(UV))
        UV = self.UV_up(UV)
        UV = self.UV_conv_last(self.lrelu(self.UV_HRconv(UV)))

        YUV = torch.cat((Y, UV), dim=1) #shape torch.Size([batch_size, 3, crop_size, crop_size])
        YUV = self.YUV_final_conv(YUV)

        base = F.interpolate(inputs, scale_factor=4, mode='bilinear', align_corners=False)
        YUV += base
        return YUV


class vanilla_ESRGAN(RRDBNet, CustomModule):
    def __init__(self):
        super(vanilla_ESRGAN, self).__init__(3, 3, 64, 23, gc=32)


class small_ESRGAN(RRDBNet, CustomModule):
    def __init__(self):
#        super(small_ESRGAN, self).__init__(3, 3, nf=32, nb=8, gc=16)
        super(small_ESRGAN, self).__init__(3, 3, nf=64, nb=2, gc=32)


class debug_ESRGAN(RRDBNet, CustomModule):
    def __init__(self):
        super(debug_ESRGAN, self).__init__(3, 3, 16, 1, gc=8)


class small_baseline_model(MSRResNet, CustomModule):
    def __init__(self):
        self.name = "SmallBaseline"
        super().__init__(in_nc=3, out_nc=3, nf=64, nb=6, upscale=4)

    def forward(self, x):
        if x.shape[1] != 3 or len(x.shape) != 4:
            print(x.shape)
            raise Exception('Shape is wrong...')
        x = x[:, [2, 1, 0], :, :] #BRG->RGB
        out = super().forward(x)
        return out[:, [2, 1, 0], :, :]

class baseline_model(MSRResNet, CustomModule):
    def __init__(self, pretrained = False):
        self.name = "Baseline"
        super(baseline_model, self).__init__(in_nc=3, out_nc=3, nf=64, nb=16, upscale=4)

        if pretrained:
            print('Loading pretrained weights')
            baseline_ckpt = os.path.join('Challenge files',
                                         'MSRResNet',
                                         'MSRResNetx4_model',
                                         'MSRResNetx4.pth')
            self.load_weights(baseline_ckpt)

    def forward(self, x):
        if x.shape[1] != 3 or len(x.shape) != 4:
            print(x.shape)
            raise Exception('Shape is wrong...')
        x = x[:, [2, 1, 0], :, :] #BRG->RGB
        out = super(baseline_model, self).forward(x)
        return out[:, [2, 1, 0], :, :]


class MyModel_debug(CustomModule):
    def __init__(self, in_c=3, out_c=3, nf=64, nb=6, bias=True):
        super(MyModel_debug, self).__init__()
        self.name = "MyNet_debug_{}_{}".format(nf, nb)

        self.bilin_layer = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.first_conv = nn.Conv2d(in_c, nf, 3, 1, 1, bias=bias)
        CRC = functools.partial(ConvReluConv, nf=nf)
        self.block1 = make_layer(CRC, nb)
        self.block2 = make_layer(CRC, nb)
        self.block3 = make_layer(CRC, nb)
        self.up1 = UpsamplingBlockx4(nf, bias=bias)
        self.up2 = UpsamplingBlockx4(nf, bias=bias)
        self.up3 = UpsamplingBlockx4(nf, bias=bias)

        # initialization
        initialize_weights([self.first_conv], 0.1)

    def forward(self, inputs):
        bilin = self.bilin_layer(inputs)

        x = self.first_conv(inputs) #no relu

        x = self.block1(x)
        out1 = self.up1(x) + bilin

        x = self.block2(x)
        out2 = self.up2(x) + out1

        x = self.block3(x)
        out3 = self.up3(x) + out2

        return out3

