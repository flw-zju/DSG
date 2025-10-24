import torch
import math
from torch import nn
import torch.nn.functional as F
from augument import DiffAugment


class ModStd(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, input, style):
        style = self.fc(style).unsqueeze(-1).unsqueeze(-1)
        return input * style


class Res(nn.Module):
    def __init__(self, in_c, out_c, up=False):
        super().__init__()
        if up:
            self.conv1 = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_c, in_c, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        z = self.conv2(x)
        z = self.shortcut(x) + z
        return z


class StyleConv(nn.Module):
    def __init__(self, in_c, out_c, style_dim, upsample=True):
        super().__init__()
        self.to_style = ModStd(style_dim, in_c)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(out_c)
        )
        self.upsample = upsample
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, style):
        x = self.to_style(x, style)
        x = self.conv(x)
        if self.upsample:
            x = self.up(x)
        return x


class ToRGB(nn.Module):
    def __init__(self, in_c, upsample=True):
        super().__init__()
        self.to_style = ModStd(512, in_c)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, 3, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, style, skip=None):
        mid = self.to_style(x, style)
        x = self.conv(mid)
        if skip is not None:
            x = x + self.up(skip)
        return x, mid


class StyleBlock(nn.Module):
    def __init__(self, in_c, out_c, style_dim):
        super().__init__()
        self.conv0 = StyleConv(in_c, in_c, style_dim, upsample=True)
        self.conv1 = StyleConv(in_c, out_c, style_dim, upsample=False)
        self.to_rgb = ToRGB(out_c, upsample=True)

    def forward(self, x, style, skip=None):
        x = self.conv0(x, style[0])
        x = self.conv1(x, style[1])
        skip, mid = self.to_rgb(x, style[2], skip)
        return x, skip, mid


class Mlp(nn.Module):
    def __init__(self, style_dim):
        super().__init__()
        self.low = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),

        )
        self.mid = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
        )
        self.high = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
        )
        self.rgb_mlp = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, style):
        low_style = self.low(style).unsqueeze(1).repeat(1, 3, 1)
        mid_style = self.mid(style).unsqueeze(1).repeat(1, 4, 1)
        high_style = self.high(style).unsqueeze(1).repeat(1, 2, 1)
        rgb_style = self.rgb_mlp(style)
        return [torch.cat((low_style, mid_style, high_style), dim=1), rgb_style]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            Res(64, 64),
            Res(64, 128),
            Res(128, 256),
            Res(256, 512),
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, style_dim):
        super().__init__()
        self.conv0 = StyleConv(512, 256, style_dim, False)
        self.to_rgb0 = ToRGB(256, style_dim)
        self.blocks = nn.ModuleList()
        channels = [256, 128, 128, 128, 64]
        for i in range(len(channels) - 1):
            self.blocks.append(StyleBlock(channels[i], channels[i + 1], style_dim))

    def forward(self, x, style, return_features=False):
        style_basic, style_rgb = style[0], style[1]

        z = self.conv0(x, style_basic[:, 0, ...])
        skip_rgb, mid = self.to_rgb0(z, style_rgb)
        hs = []
        if return_features:
            hs.append(mid)
        for i, module in enumerate(self.blocks):
            z, skip_rgb, mid = module(z, [style_basic[:, 2 * i + 1, ...], style_basic[:, 2 * i + 2, ...], style_rgb],
                                      skip_rgb)
            if return_features:
                hs.append(mid)
        if return_features:
            return skip_rgb, hs
        else:
            return skip_rgb


class SwappingNet(nn.Module):
    def __init__(self, style_dim):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(style_dim)
        self.mlp = Mlp(style_dim)

    def forward(self, x, style, return_features=False):
        x = self.encoder(x)
        w_plus = self.mlp(style)
        if return_features:
            x, hs = self.decoder(x, w_plus, return_features)
            return x, hs
        else:
            x = self.decoder(x, w_plus, return_features)
            return x


class FreqConv(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.real = nn.Sequential(
            nn.Conv2d(in_c * 2, in_c, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_c, affine=True),
            nn.LeakyReLU(0.2),
        )
        self.imag = nn.Sequential(
            nn.Conv2d(in_c * 2, in_c, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_c, affine=True),
            nn.LeakyReLU(0.2),
        )

    def fre(self, x):
        freq_x = torch.fft.fft2(x, norm='ortho')
        freq_x = torch.fft.fftshift(freq_x)
        real = torch.real(freq_x)
        imag = torch.imag(freq_x)
        ffted = torch.cat((real, imag), dim=1)
        return ffted

    def fre_(self, freq):
        freq_ = torch.split(freq.unsqueeze(-1), freq.shape[1] // 2, dim=1)
        freq = torch.view_as_complex(torch.cat(freq_, dim=-1))
        freq = torch.fft.ifftshift(freq)
        x = torch.fft.ifft2(freq, norm='ortho').real
        return x

    def forward(self, x):
        freq_x = self.fre(x)
        freq_x_ = torch.cat((self.real(freq_x), self.imag(freq_x)), dim=1)
        return self.fre_(freq_x_)


class V2tBlock(nn.Module):
    def __init__(self, in_c, out_c, upsample=True):
        super().__init__()

        self.freq_conv = FreqConv(in_c)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_c, 3, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip=None):
        x = self.freq_conv(x)
        x = self.conv1(x)
        if skip is not None:
            x = x + self.upsample(skip)
        return x


class V2tTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([])
        channels = [256, 128, 128, 128, 64]
        self.blocks.append(V2tBlock(channels[0], channels[0], upsample=False))
        for i in range(1, len(channels)):
            self.blocks.append(V2tBlock(channels[i], channels[i], upsample=True))

    def forward(self, hs):
        skip = None
        for i, module in enumerate(self.blocks):
            skip = module(hs[i], skip)
        return skip


class Net(nn.Module):
    def __init__(self, style_dim, train_mode=0):
        super().__init__()
        self.swapping_net = SwappingNet(style_dim)
        self.v2tt_net = V2tTNet()
        self.train_mode = train_mode
        self.set_train_mode(train_mode)

    def set_train_mode(self, train_mode):
        if train_mode == 0:
            self.swapping_net.requires_grad_(True)
            self.v2tt_net.requires_grad_(False)
        elif train_mode == 1:
            self.swapping_net.requires_grad_(False)
            self.v2tt_net.requires_grad_(True)
        else:
            self.swapping_net.requires_grad_(False)
            self.v2tt_net.requires_grad_(False)

    def forward(self, x, style):
        if self.train_mode == 0:
            return self.swapping_net(x, style)
        else:
            rgb_img, mid = self.swapping_net(x, style, return_features=True)
            inf_img = self.v2tt_net(mid)
            return rgb_img, inf_img


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(0.2, True),
        )

        self.res = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(0.2, True),
        )
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        residual = self.res(x)
        x = self.block1(x)
        x = self.block2(x)
        return (x + residual) * self.scale



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.from_rgb = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(64, 256),
            DiscriminatorBlock(256, 512),
            DiscriminatorBlock(512, 512),
            DiscriminatorBlock(512, 512),
            DiscriminatorBlock(512, 512),
        ])
        self.final_conv1 = nn.Sequential(
            nn.Conv2d(513, 513, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Sequential(
            nn.Linear(513 * 4 * 4, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, x_t, p=None):
        if p is not None:
            x_t = DiffAugment(x_t, p)
        x = self.from_rgb(x_t)

        for module in self.blocks:
            x = module(x)

        b, c, h, w = x.shape
        stddev = x.view(x.shape[0], -1, 1, c, h, w)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(x.shape[0], 1, h, w)
        x = torch.cat([x, stddev], 1)

        z1 = self.final_conv1(x)
        z1 = self.fc(z1.view(x.shape[0], -1))
        return z1




