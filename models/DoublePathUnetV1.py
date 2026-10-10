from models.ConvBlock import ConvBlock
from models.UDCUnet.arch_util import ResidualBlockNoBN
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets.dwt.transform2d import DWTForward, DWTInverse
import torch.fft as fft


class GRM(nn.Module):
    def __init__(self, ch: int, red: int = 4):
        super().__init__()
        red_ch = max(ch // red, 8)
        self.energy_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
        nn.Conv2d(3*ch, red_ch, 1), nn.ReLU(),
        nn.Conv2d(red_ch, 3*ch, 1)
        )
        self.merge = nn.Sequential(
        ConvBlock(ch, ch, 3, 1, 1),
        ConvBlock(ch, ch, 1, 1, 0)
        )

    def forward(self, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor) -> torch.Tensor:
        # Global energy stats as a cheap proxy; can be replaced by orientation histograms
        stat = torch.cat([self.energy_pool(LH), self.energy_pool(HL), self.energy_pool(HH)], dim=1)
        g = torch.sigmoid(self.mlp(stat)) # (B,3C,1,1)
        g1, g2, g3 = torch.chunk(g, 3, dim=1)
        Hsum = g1*LH + g2*HL + g3*HH
        F_token = self.merge(Hsum) # (B,C,H/2,W/2)
        return F_token


class AFSR(nn.Module):
    def __init__(self, kernel=16):
        super().__init__()
        self.proj = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        # 灰度化或通道平均
        xg = x.mean(dim=1, keepdim=True)
        B, C, H, W = xg.shape
        # 提取局部块
        kernel = H//8
        unfold = F.unfold(xg, kernel_size=kernel, stride=kernel)
        # print(unfold.shape, "unfold")
        patches = unfold.transpose(1, 2).reshape(-1, kernel, kernel)
        # print(patches.shape, "patches")
        # 计算频谱熵
        Fpatch = fft.fft2(patches)
        P = (Fpatch.abs() ** 2)
        P = P / (P.sum(dim=(-2,-1), keepdim=True) + 1e-8)
        Hlse = -(P * torch.log(P + 1e-8)).sum(dim=(-2,-1))
        Hlse = Hlse.reshape(B, 1, H//kernel, W//kernel)
        Hlse = F.interpolate(Hlse, size=(H, W), mode='bilinear')
        return torch.sigmoid(self.proj(Hlse))


class SBypass(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.reduce = ConvBlock(in_ch, out_ch, 3, 2, 1)
        self.body = ResidualBlockNoBN(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # spatial downsample via bilinear + conv
        x = self.reduce(x)
        return self.body(x)


class DownSample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_afsr: bool = True):
        super().__init__()
        self.dwt = DWTForward()
        self.grm = GRM(in_ch)
        self.align_freq = nn.Conv2d(2 * in_ch, out_ch, 1)
        self.s_bypass = SBypass(in_ch, out_ch)
        self.use_afsr = use_afsr
        self.afsr = AFSR(in_ch) if use_afsr else None

    def forward(self, x: torch.Tensor):
        LL, yh = self.dwt(x)
        LH, HL, HH = [yh[0][:, :, i, :, :] for i in range(3)]
        Ftok = self.grm(LH, HL, HH)
        freq = self.align_freq(torch.cat([LL, Ftok], dim=1))  # (B,out,H/2,W/2)
        spat = self.s_bypass(x)  # (B,out,H/2,W/2)
        if self.use_afsr:
        # Predict gate on pre-DWT feature (downsample to match resolution)
            gate_in = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            m = self.afsr(gate_in)  # (B,1,H/2,W/2)
            out = m * freq + (1.0 - m) * spat
        else:
            out = 0.5 * (freq + spat)
        # Return stage output and cached frequency summary (LL, Ftok)
        return out, (LL, Ftok)


class FiLM(nn.Module):
    def __init__(self, freq_ch: int, target_ch: int):
        super().__init__()
        hid = max(freq_ch // 2, 32)
        self.mlp = nn.Sequential(
        nn.Conv2d(freq_ch, hid, 1, 1, 0), nn.ReLU(),
        nn.Conv2d(hid, 2*target_ch, 1, 1, 0)
        )

    def forward(self, freq_cond: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        stats = F.adaptive_avg_pool2d(freq_cond, 1)
        gamma_beta = self.mlp(stats)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return y * (1 + gamma) + beta


class CrossDomainAttention(nn.Module):
    def __init__(self, ch: int, freq_ch: int, heads: int = 2):
        super().__init__()
        self.h = heads
        self.q = nn.Conv2d(ch, ch, 1, 1, 0)
        self.kv = nn.Conv2d(freq_ch, 2*ch, 1, 1, 0)
        self.proj = nn.Conv2d(ch, ch, 1, 1, 0)

    def forward(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        B, C, H, W = y.shape
        q = self.q(y).reshape(B, self.h, C//self.h, H*W)
        f = F.interpolate(f, size=y.shape[-2:], mode='bilinear', align_corners=False)
        kv = self.kv(f)
        k, v = torch.chunk(kv, 2, dim=1)
        k = k.reshape(B, self.h, C//self.h, H*W)
        v = v.reshape(B, self.h, C//self.h, H*W)
        attn = torch.softmax((q.transpose(-1, -2) @ k) / (C//self.h) ** 0.5, dim=-1)
        out = (attn @ v.transpose(-1, -2)).transpose(-1, -2).reshape(B, C, H, W)
        return self.proj(out) + y


class CDA(nn.Module):
    def __init__(self, y_ch: int, freq_ch: int, use_xattn: bool):
        super().__init__()
        self.use_x = use_xattn
        self.freq_proj = nn.Conv2d(freq_ch*2, y_ch, 1)
        self.film = FiLM(y_ch, y_ch)
        self.use_xattn = use_xattn
        self.xattn = CrossDomainAttention(y_ch, y_ch, heads=2)
        self.post = ResidualBlockNoBN(y_ch)

    def forward(self, y: torch.Tensor, freq_pair):
        LL, Ftok = freq_pair
        f = self.freq_proj(torch.cat([LL, Ftok], dim=1))
        y = self.film(f, y)
        if self.use_xattn:
            y = self.xattn(y, f)
        return self.post(y)


class UpSample(nn.Module):
    def __init__(self, ch: int, out_ch: int, use_xattn: bool = True):
        super().__init__()

        self.up = nn.Sequential(
            ConvBlock(ch, ch * 2, 3, 1, 1),
            nn.PixelShuffle(2),
        )
        self.pre = ConvBlock(ch//2 + out_ch, out_ch, 3, 1, 1)
        self.cda = CDA(out_ch, out_ch, use_xattn)

    def forward(self, y: torch.Tensor, skip_spatial: torch.Tensor, freq_pair):
        y = self.up(y)
        y = self.pre(torch.cat([y, skip_spatial], dim=1))
        y = self.cda(y, freq_pair)
        return y


class PositiveEvidence(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.avg = nn.AvgPool2d(k, 1, k//2)

    def forward(self, xt, x):
        mu1, mu2 = self.avg(x), self.avg(xt)
        sigma1 = self.avg((x - mu1)**2)
        sigma2 = self.avg((xt - mu2)**2)
        sigma12 = self.avg((x - mu1)*(xt - mu2))
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2))/((mu1**2 + mu2**2 + C1)*(sigma1 + sigma2 + C2) + 1e-6)
        ssim_map = ssim_map.mean(1, keepdim=True)
        mse = self.avg((xt - x)**2)
        psnr_proxy = 1/(mse + 1e-6)
        return torch.tanh(0.5*(ssim_map + psnr_proxy/psnr_proxy.mean((2,3), keepdim=True))).clamp(0,1)


class NegativeEvidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWTForward()

    def forward(self, residual):
        LLr, yh = self.dwt(residual)
        LHr, HLr, HHr = [yh[0][:, :, i, :, :] for i in range(3)]
        e_ll = LLr.pow(2).mean(1, keepdim=True)
        e_tot = e_ll + LHr.pow(2).mean(1, keepdim=True) + HLr.pow(2).mean(1, keepdim=True) + HHr.pow(2).mean(1, keepdim=True) + 1e-6
        spec_risk = 1.0 - e_ll/e_tot
        return torch.tanh(spec_risk).clamp(0, 1)


class ScoreBranch(nn.Module):
    def __init__(self, pos_ch: int, neg_ch: int, mid_ch: int = 64, alpha=0.5, beta=1.5):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.pos_net = nn.Sequential(
            ConvBlock(pos_ch, mid_ch, 3, 1, 1),
            ConvBlock(mid_ch, mid_ch, 3, 1, 1)
        )
        self.neg_net = nn.Sequential(
            ConvBlock(neg_ch, mid_ch, 3, 1, 1),
            ConvBlock(mid_ch, mid_ch, 3, 1, 1)
        )
        self.fp = nn.Conv2d(mid_ch, 1, 1, 1, 0)
        self.fn = nn.Conv2d(mid_ch, 1, 1, 1, 0)

    def forward(self, pos_feats, neg_feats, m0):
        neg_feats = F.interpolate(neg_feats, size=m0.shape[-2:], mode='bilinear', align_corners=False)
        p, n = self.pos_net(pos_feats), self.neg_net(neg_feats)
        fusion = self.alpha * self.fp(p) - self.beta * self.fn(n)
        logit_m0 = torch.log(m0.clamp(1e-6, 1 - 1e-6) / (1 - m0.clamp(1e-6, 1 - 1e-6)))
        m = torch.sigmoid(0.5*logit_m0 + fusion)
        return m


class Mask(nn.Module):
    def __init__(self, img_ch: int = 3, mid_ch: int = 64):
        super().__init__()
        self.conv = ConvBlock(img_ch, mid_ch, 3, 1, 1)
        self.res = ResidualBlockNoBN(mid_ch + img_ch)
        self.f = ConvBlock(mid_ch + img_ch, mid_ch // 2, 3, 1, 1)
        self.mask = nn.Sequential(nn.Conv2d(mid_ch // 2, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x = torch.cat((x, self.conv(x)), dim=1)
        res = self.res(x)
        mask = self.mask(self.f(res))
        return mask


class AdaptiveScoringMask(nn.Module):
    """Generate final mask M from host image + frequency hints and score branch."""
    def __init__(self, pos_ch: int, neg_ch: int, img_ch: int = 3, mid_ch: int = 64):
        super().__init__()
        self.score_branch = ScoreBranch(pos_ch, neg_ch, mid_ch)
        self.mask_gen = Mask(img_ch, mid_ch)

    def forward(self, host, residual, pos_feats=None, neg_feats=None):
        mask = self.mask_gen(host)
        if pos_feats is not None:
            m = self.score_branch(pos_feats, neg_feats, mask)
            watermarked = host + m * mask * residual
        else:
            watermarked = host + 1 * mask * residual
        return watermarked


class UNet(nn.Module):
    def __init__(self, inc=3, outc=3, nf=32):
        super(UNet, self).__init__()
        out_channels = [nf * (2 ** (i + 1)) for i in range(5)]  # [64, 128, 256, 512, 1024]
        # 下采样
        self.pre_layer = nn.Sequential(
            ConvBlock(inc, out_channels[0], 3, 1, 1),
            ConvBlock(out_channels[0], out_channels[0], 3, 1, 1)
        )
        self.d1 = DownSample(out_channels[0], out_channels[1])  # 3-64
        self.d2 = DownSample(out_channels[1], out_channels[2])  # 64-128
        self.d3 = DownSample(out_channels[2], out_channels[3])  # 128-256
        self.d4 = DownSample(out_channels[3], out_channels[4])  # 256-512

        self.mid = ConvBlock(out_channels[4], out_channels[4], 3, 1, 1)
        # 上采样
        self.u1 = UpSample(out_channels[4], out_channels[3], True)  # 512-1024-512
        self.u2 = UpSample(out_channels[3], out_channels[2], False)  # 1024-512-256
        self.u3 = UpSample(out_channels[2], out_channels[1], False)  # 512-256-128
        self.u4 = UpSample(out_channels[1], out_channels[0], False)  # 256-128-64
        # 输出
        self.out = nn.Sequential(
            ConvBlock(out_channels[0], 32, 3, 1, 1),
            ConvBlock(32, outc, 3, 1, 1),
        )

    def forward(self, x):
        x_pre = self.pre_layer(x)
        out1, f1 = self.d1(x_pre)
        out2, f2 = self.d2(out1)
        out3, f3 = self.d3(out2)
        out4, f4 = self.d4(out3)
        out5 = self.mid(out4)
        out6 = self.u1(out5, out3, f4)
        out7 = self.u2(out6, out2, f3)
        out8 = self.u3(out7, out1, f2)
        out9 = self.u4(out8, x_pre, f1)
        out = self.out(out9)

        return out

