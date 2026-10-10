from models.ConvBlock import ConvBlock
from models.UDCUnet.arch_util import ResidualBlockNoBN
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets.dwt.transform2d import DWTForward, DWTInverse

# dct = dct3(device="cuda:0")
# idct = idct3(device="cuda:0")


class LightACFT(nn.Module):
    """
    轻量级空-频融合模块 (可嵌入Unet下采样路径)
    """

    def __init__(self, in_ch, reduction=4):
        super(LightACFT, self).__init__()
        hidden_ch = in_ch // reduction
        self.q_proj = nn.Conv2d(in_ch, hidden_ch, 1)
        self.k_proj = nn.Conv2d(in_ch, hidden_ch, 1)
        self.v_proj = nn.Conv2d(in_ch, hidden_ch, 1)
        self.out_proj = nn.Conv2d(hidden_ch, in_ch, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, hidden_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, in_ch, 1),
            nn.PReLU()
        )
        self.norm = nn.BatchNorm2d(in_ch)

    def forward(self, Fs, Ff):
        # Gating vector controls frequency influence
        gate = self.gate(Ff)
        Fs_mod = Fs * gate  # spatially modulated by freq strength

        # Linear projection (dimension reduction)
        Q = self.q_proj(Fs_mod)
        K = self.k_proj(Ff)
        V = self.v_proj(Ff)

        # Attention (局部点积)
        attn = torch.sigmoid(Q * K)  # simple elementwise attention
        fused = Q + attn * V

        # Back to original dimension
        out = self.out_proj(fused)
        out = self.norm(Fs + out)  # residual connection
        return F.relu(out)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(ch, ch, 3, 1, 1),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class DWTLayer(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.dwt = DWTForward()
        # self.idwt = DWTInverse()

    def forward(self, x):
        ll, yh = self.dwt(x)
        lh = yh[0][:, :, 0, :, :]
        hl = yh[0][:, :, 1, :, :]
        hh = yh[0][:, :, 2, :, :]
        # ll, (lh, hl, hh) = pywt.dwt2(x, self.wave)
        return ll, lh, hl, hh


class FGC(nn.Module):
    """
    Frequency-Gated Compensation Module
    动态感知频谱能量，对低能频带补偿嵌入能力
    """
    def __init__(self, num_bands=4):
        super(FGC, self).__init__()
        self.energy_predictor = nn.Sequential(
            nn.Linear(num_bands, 32),
            nn.ReLU(),
            nn.Linear(32, num_bands),
            nn.PReLU()
        )
        self.alpha = nn.Parameter(torch.ones(num_bands))

    def forward(self, bands):
        # bands: list of DWT bands [LL, LH, HL, HH]
        energies = torch.stack([b.abs().mean() for b in bands])  # (4,)
        normE = energies / (energies.sum() + 1e-8)
        weights = self.energy_predictor(normE.unsqueeze(0))  # (1,4)
        compensated = [
            b + self.alpha[i] * (1 - normE[i]) * weights[0, i] * b
            for i, b in enumerate(bands)
        ]
        return compensated


class SubBandModulation(nn.Module):
    def __init__(self, in_ch, reduction=8):
        super().__init__()
        self.fgc = FGC()
        self.fc = nn.Sequential(
            ConvBlock(in_ch * 3, in_ch // reduction, 3, 1, 1),
            ConvBlock(in_ch // reduction, 4, 3, 1, 1),
            nn.Linear(in_ch * 3, in_ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, 4),
            nn.PReLU()
        )

    def forward(self, LL, LH, HL, HH):
        B, C, _, _ = LL.shape
        bands = [LL, LH, HL, HH]
        bands = self.fgc(bands)

        # 全局平均池化获取统计
        stats = torch.cat([
            bands[1].mean(dim=[2, 3]), bands[2].mean(dim=[2, 3]), bands[3].mean(dim=[2, 3])
        ], dim=1)
        weights = self.fc(stats).view(B, 4, 1, 1, 1)
        fused = weights[:, 0] * bands[0] + weights[:, 1] * bands[1] + weights[:, 2] * bands[2] + weights[:, 3] * bands[3]
        return fused


class LE_DWT_Path(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dwt1 = DWTLayer()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.smb = SubBandModulation(out_ch)

    def forward(self, x):
        LL, LH, HL, HH = self.dwt1(x)
        LL, LH, HL, HH = map(lambda f: self.conv(f), [LL, LH, HL, HH])
        Ff = self.smb(LL, LH, HL, HH)
        return Ff


class AAM(nn.Module):
    """
    Adaptive Attention Masking with Multi-Scale Mask Support
    - feat: 当前层特征 [B, C, H, W]
    - mask: 当前尺度掩码 [B, 1, H_mask, W_mask]
    """
    def __init__(self, out_ch, reduction=8):
        super(AAM, self).__init__()
        self.conv_g = ConvBlock(out_ch, out_ch // reduction, 3, 1, 1)
        self.conv_m = ConvBlock(3, out_ch // reduction, 3, 1, 1)
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_ch // reduction, out_ch, 1, 1, 0),
            nn.BatchNorm2d(out_ch)
        )
        self.PReLU = nn.PReLU()

    def forward(self, feat, mask):
        B, C, H, W = feat.shape
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        g_feat = self.conv_g(feat)      # [B, C//r, H, W]
        m_feat = self.conv_m(mask)      # [B, C//r, H, W]
        gate = self.PReLU(self.conv_out(g_feat + m_feat))  # [B, C, H, W]
        return feat * gate


class SpatialPath(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SpatialPath, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_ch, out_ch, 3, 2, 1),
            ResidualBlock(out_ch)
        )
        self.aam = AAM(out_ch)

    def forward(self, x, mask):
        xs = self.conv(x)
        out = self.aam(xs, mask)
        return out


class HoloDownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HoloDownSample, self).__init__()
        self.spatial = SpatialPath(in_ch, out_ch)
        self.frequency = LE_DWT_Path(in_ch, out_ch)
        self.fusion = LightACFT(out_ch)

    def forward(self, x, mask):
        Fs = self.spatial(x, mask)
        Ff = self.frequency(x)
        out = self.fusion(Fs, Ff)
        return out


class RM_MRC(nn.Module):
    """
    Residual-Modulated Meta-Robustness Controller
    非加权式元鲁棒性控制器
    """
    def __init__(self, in_ch, reduction=8):
        super().__init__()
        self.meta_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * in_ch, in_ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, 1, bias=False)
        )

        self.residual = nn.Conv2d(in_ch*2, in_ch, 1, bias=False)
        self.act = nn.ReLU()

    def forward(self, F_fused, F_skip):
        concat_feat = torch.cat([F_fused, F_skip], dim=1)
        delta = self.meta_mlp(concat_feat)
        F_mod = self.residual(concat_feat)
        out = self.act(delta * F_mod + F_mod)
        return out


class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSample, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            ConvBlock(in_ch, out_ch * 4, 3, 1, 1),
            ResidualBlockNoBN(out_ch * 4)
        )
        self.upsample = nn.Sequential(
            ConvBlock(out_ch * 4, out_ch * 4, 3, 1, 1),
            nn.PixelShuffle(2),
        )
        self.RC = RM_MRC(out_ch)

    def forward(self, x, pred):
        '''
        :param x: 上采样输入
        :param out:与上采样层进行cat
        :return:
        '''
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        # x_out = torch.cat((x_out, pred), dim=1)
        x_out = self.RC(x_out, pred)
        return x_out


class UNet(nn.Module):
    def __init__(self, inc=3, outc=3, nf=32):
        super(UNet, self).__init__()
        out_channels = [nf * (2 ** (i + 1)) for i in range(5)]  # [64, 128, 256, 512, 1024]
        # 下采样
        self.pre_layer = nn.Sequential(
            ConvBlock(inc, out_channels[0], 3, 1, 1),
            ConvBlock(out_channels[0], out_channels[0], 3, 1, 1)
        )
        self.d1 = HoloDownSample(out_channels[0], out_channels[1])  # 3-64
        self.d2 = HoloDownSample(out_channels[1], out_channels[2])  # 64-128
        self.d3 = HoloDownSample(out_channels[2], out_channels[3])  # 128-256
        self.d4 = HoloDownSample(out_channels[3], out_channels[4])  # 256-512

        self.mid = ConvBlock(out_channels[4], out_channels[4], 3, 1, 1)
        # 上采样
        self.u1 = UpSample(out_channels[4], out_channels[3])  # 512-1024-512
        self.u2 = UpSample(out_channels[3], out_channels[2])  # 1024-512-256
        self.u3 = UpSample(out_channels[2], out_channels[1])  # 512-256-128
        self.u4 = UpSample(out_channels[1], out_channels[0])  # 256-128-64
        # 输出
        self.out = nn.Sequential(
            ConvBlock(out_channels[0], 32, 3, 1, 1),
            ConvBlock(32, outc, 3, 1, 1),
        )

    def forward(self, x, mask):
        x_pre = self.pre_layer(x)
        out1 = self.d1(x_pre, mask)
        out2 = self.d2(out1, mask)
        out3 = self.d3(out2, mask)
        out4 = self.d4(out3, mask)
        out5 = self.mid(out4)
        out6 = self.u1(out5, out3)
        out7 = self.u2(out6, out2)
        out8 = self.u3(out7, out1)
        out9 = self.u4(out8, x_pre)
        out = self.out(out9)

        return out


if __name__ == '__main__':
    import time

    a = torch.rand(16, 32, 128, 128).cuda()
    model = UNet(32).cuda()
    start = time.time()
    model(a)
    print(time.time() - start)
