import torch.nn as nn
import math
import os
import torch
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat
import torch.optim as optim
from torch.nn.modules.utils import _pair


class TriangularPattern(nn.Module):
    def __init__(self, flip=False, sal_size=56):
        super(TriangularPattern,self).__init__()
        self.flip = flip
        target_size = int(sal_size * 3 / 4)
        self.downsample = nn.AdaptiveMaxPool2d(output_size=(target_size, target_size))

    def forward(self, x, s):
        if self.flip:
            x = torch.flip(x, dims=(3,))

        up_indices = torch.triu_indices(x.shape[2], x.shape[3],
                                        device=x.device,
                                        offset=1)
        up_feat    = x[:,:,up_indices[0], up_indices[1]].mean(dim=2)

        lw_indices = torch.tril_indices(x.shape[2], x.shape[3],
                                        device=x.device,
                                        offset=-1)
        lw_feat    = x[:,:,lw_indices[0], lw_indices[1]].mean(dim=2)
        fused = torch.stack([up_feat, lw_feat], dim=2).unsqueeze(3)
        if s is None:
            return fused

        if self.flip:
            s = torch.flip(s, dims=(3,))
        s = self.downsample(s)
        up_sal_indices = torch.triu_indices(s.shape[2],
                                            s.shape[3],
                                            device=s.device,
                                            offset=1)
        up_sal = s[:, :, up_sal_indices[0], up_sal_indices[1]].flatten(1)

        lw_sal_indices = torch.tril_indices(s.shape[2],
                                            s.shape[3],
                                            device=s.device,
                                            offset=-1)
        lw_sal = s[:, :, lw_sal_indices[0], lw_sal_indices[1]].flatten(1)


        fused_sal = torch.stack([up_sal, lw_sal], dim=2).unsqueeze(3)
        return fused, fused_sal

class CrossPattern(nn.Module):
    def __init__(self):
        super(CrossPattern, self).__init__()

    def forward(self, x, s):
        ones_vec = torch.ones(x.size(2), x.size(3), requires_grad=False)
        up_flip  = torch.triu(ones_vec, diagonal=0).flip(dims=(-1,))
        lw_flip  = torch.tril(ones_vec, diagonal=0).flip(dims=(-1,))
        up_inds  = torch.where(torch.triu(up_flip, diagonal=0) > 0)
        lf_inds  = torch.where(torch.tril(up_flip, diagonal=0) > 0)
        rt_inds  = torch.where(torch.triu(lw_flip, diagonal=0) > 0)
        lw_inds  = torch.where(torch.tril(lw_flip, diagonal=0) > 0)

        up_feat   = x[:, :, up_inds[0], up_inds[1]].mean(2)
        lf_feat   = x[:, :, lf_inds[0], lf_inds[1]].mean(2)
        rt_feat   = x[:, :, rt_inds[0], rt_inds[1]].mean(2)
        lw_feat   = x[:, :, lw_inds[0], lw_inds[1]].mean(2)

        fused1 = torch.stack([lf_feat, up_feat], dim=2)
        fused2 = torch.stack([lw_feat, rt_feat], dim=2)
        fused = torch.stack([fused1, fused2], dim=3)
        if s is None:
            return fused

        assert s.shape[2] == s.shape[3], \
            'saliency map should be square, but get a shape {}'.format(s.shape)
        ones_vec = torch.ones(s.size(2), s.size(3), requires_grad=False)
        up_flip = torch.triu(ones_vec, diagonal=0).flip(dims=(-1,))
        lw_flip = torch.tril(ones_vec, diagonal=0).flip(dims=(-1,))
        up_inds = torch.where(torch.triu(up_flip, diagonal=0) > 0)
        lf_inds = torch.where(torch.tril(up_flip, diagonal=0) > 0)
        rt_inds = torch.where(torch.triu(lw_flip, diagonal=0) > 0)
        lw_inds = torch.where(torch.tril(lw_flip, diagonal=0) > 0)

        up_sal = s[:, :, up_inds[0], up_inds[1]].flatten(1)
        lf_sal = s[:, :, lf_inds[0], lf_inds[1]].flatten(1)
        rt_sal = s[:, :, rt_inds[0], rt_inds[1]].flatten(1)
        lw_sal = s[:, :, lw_inds[0], lw_inds[1]].flatten(1)

        sal1 = torch.stack([lf_sal, up_sal], dim=2)
        sal2 = torch.stack([lw_sal, rt_sal], dim=2)
        fused_sal = torch.stack([sal1, sal2], dim=3)

        return fused, fused_sal

class SurroundPattern(nn.Module):
    def __init__(self, crop_size=1./2):
        super(SurroundPattern, self).__init__()
        self.crop_size = crop_size

    def forward(self, x, s):
        H,W         = x.shape[2:]
        crop_h      = (int(H / 2 - self.crop_size / 2 * H), int(H / 2 + self.crop_size / 2 * H))
        crop_w      = (int(W / 2 - self.crop_size / 2 * W), int(W / 2 + self.crop_size / 2 * W))
        x_mask      = torch.zeros(H,W,device=x.device, dtype=torch.bool)
        x_mask[crop_h[0] : crop_h[1], crop_w[0] : crop_w[1]] = True

        inside_indices  = torch.where(x_mask)
        inside_part = x[:, :, inside_indices[0], inside_indices[1]]
        inside_feat = inside_part.mean(2)

        outside_indices = torch.where(~x_mask)
        outside_part    = x[:, :, outside_indices[0], outside_indices[1]]
        outside_feat    = outside_part.mean(2)
        fused = torch.stack([inside_feat, outside_feat], dim=2).unsqueeze(3)
        if s is None:
            return fused

        SH,SW       = s.shape[2:]
        crop_sh     = (int(SH / 2 - self.crop_size / 2 * SH), int(SH / 2 + self.crop_size / 2 * SH))
        crop_sw     = (int(SW / 2 - self.crop_size / 2 * SW), int(SW / 2 + self.crop_size / 2 * SW))
        s_mask      = torch.zeros(SH, SW, device=s.device, dtype=torch.bool)
        s_mask[crop_sh[0] : crop_sh[1], crop_sw[0] : crop_sw[1]] = True

        s_inside_indices = torch.where(s_mask)
        inside_sal  = s[:, :, s_inside_indices[0], s_inside_indices[1]].flatten(1)

        s_outside_indices = torch.where(~s_mask)
        outside_sal = s[:, :, s_outside_indices[0], s_outside_indices[1]].flatten(1)
        if outside_sal.shape != inside_sal.shape:
            outside_sal = F.adaptive_max_pool1d(outside_sal.unsqueeze(1), output_size=inside_sal.shape[1])
            outside_sal = outside_sal.squeeze(1)
        fused_sal    = torch.stack([inside_sal, outside_sal], dim=2).unsqueeze(3)
        return fused, fused_sal

class HorizontalPattern(nn.Module):
    def __init__(self):
        super(HorizontalPattern, self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

    def forward(self, x, s):
        H = x.shape[2]
        up_part = x[:, :, : H // 2, :]
        up_feat = up_part.mean(3).mean(2)

        lw_part = x[:, :, H // 2 :, :]
        lw_feat = lw_part.mean(3).mean(2)
        fused = torch.stack([up_feat, lw_feat], dim=2).unsqueeze(3)
        if s is None:
            return fused
        SH = s.shape[2]
        s = self.downsample(s)
        up_sal  = s[:, :, : SH // 2, :].flatten(1)
        if SH % 2 == 0:
            lw_sal  = s[:, :, SH // 2 :, :].flatten(1)
        else:
            lw_sal  = s[:, :, SH // 2 + 1 :, :].flatten(1)
        fused_sal = torch.stack([up_sal, lw_sal], dim=2).unsqueeze(3)
        return fused, fused_sal

class VerticalPattern(nn.Module):
    def __init__(self):
        super(VerticalPattern, self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x, s):
        W = x.shape[3]
        left_part  = x[:, :, :, : W // 2]
        left_feat  = left_part.mean(3).mean(2)

        right_part = x[:, :, :, W // 2 :]
        right_feat = right_part.mean(3).mean(2)
        fused = torch.stack([left_feat, right_feat], dim=2).unsqueeze(2)
        if s is None:
            return fused

        SW = s.shape[3]
        s = self.downsample(s)
        left_sal   = s[:, :, :, : SW // 2].flatten(1)
        if SW % 2 == 0:
            right_sal  = s[:, :, :, SW // 2 :].flatten(1)
        else:
            right_sal  = s[:, :, :, SW // 2 + 1 :].flatten(1)
        fused_sal = torch.stack([left_sal, right_sal], dim=2).unsqueeze(2)
        return fused, fused_sal

class GlobalPattern(nn.Module):
    def __init__(self):
        super(GlobalPattern, self).__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(1)
        )
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x, s):
        gap_feat = self.gap(x)
        if s is None:
            return gap_feat
        sal_feat = self.downsample(s).unsqueeze(2).unsqueeze(3)
        return gap_feat, sal_feat

class QuarterPattern(nn.Module):
    def __init__(self):
        super(QuarterPattern, self).__init__()

    def forward(self, x, s):
        feat = F.adaptive_avg_pool2d(x, output_size=(2,2))
        if s is None:
            return feat

        s_chunks = torch.chunk(s, 2, dim=2)
        up_left, up_right = torch.chunk(s_chunks[0], 2, dim=3)
        lw_left, lw_right = torch.chunk(s_chunks[1], 2, dim=3)
        up_sal = torch.stack([up_left.flatten(1), up_right.flatten(1)],dim=2)
        lw_sal = torch.stack([lw_left.flatten(1), lw_right.flatten(1)],dim=2)
        feat_sal = torch.stack([up_sal, lw_sal], dim=2)
        return feat, feat_sal

class ThirdOfRulePattern(nn.Module):
    def __init__(self):
        super(ThirdOfRulePattern, self).__init__()

    def forward(self, x, s):
        feat = F.adaptive_avg_pool2d(x, output_size=(3,3))
        if s is None:
            return feat
        out_size = (s.shape[-1] // 3) * 3
        s = F.adaptive_max_pool2d(s, output_size=(out_size,out_size))
        hor_chunks = torch.chunk(s, 3, dim=3)
        feat_sal = []
        for h_chunk in hor_chunks:
            tmp = []
            for v_chunk in torch.chunk(h_chunk, 3, dim=2):
                tmp.append(v_chunk.flatten(1))
            tmp = torch.stack(tmp, dim=2)
            feat_sal.append(tmp)
        feat_sal = torch.stack(feat_sal, dim=3)
        return feat, feat_sal

class VerticalThirdPattern(nn.Module):
    def __init__(self):
        super(VerticalThirdPattern, self).__init__()

    def forward(self, x, s):
        feat = F.adaptive_avg_pool2d(x, output_size=(3,1))
        if s is None:
            return feat
        out_size = (s.shape[-2] // 3) * 3
        s = F.adaptive_max_pool2d(s, output_size=(out_size,s.shape[-1]))
        ver_chunks = torch.chunk(s, 3, dim=2)
        feat_sal = []
        for ver in ver_chunks:
            feat_sal.append(ver.flatten(1))
        feat_sal = torch.stack(feat_sal, dim=2).unsqueeze(3)
        return feat, feat_sal

class HorThirdPattern(nn.Module):
    def __init__(self):
        super(HorThirdPattern, self).__init__()

    def forward(self, x, s):
        feat = F.adaptive_avg_pool2d(x, output_size=(1,3))
        if s is None:
            return feat
        out_size = (s.shape[-1] // 3) * 3
        s = F.adaptive_max_pool2d(s, output_size=(s.shape[-2], out_size))
        hor_chunks = torch.chunk(s, 3, dim=3)
        feat_sal = []
        for hor in hor_chunks:
            feat_sal.append(hor.flatten(1))
        feat_sal = torch.stack(feat_sal, dim=2).unsqueeze(2)
        return feat, feat_sal

class MultiDirectionPattern(nn.Module):
    def __init__(self):
        super(MultiDirectionPattern, self).__init__()
        feat_mask = self.generate_multi_direction_mask(7, 7)
        sal_mask  = self.generate_multi_direction_mask(56, 56)
        self.register_buffer('feat_mask', feat_mask)
        self.register_buffer('sal_mask',  sal_mask)

    def generate_multi_direction_mask(self, w, h):
        mask = torch.zeros(8, h, w)
        degree_mask = torch.zeros(h, w)
        if h % 2 == 0:
            cx, cy = float(w-1)/2, float(h-1)/2
        else:
            cx, cy = w // 2, h // 2

        for i in range(w):
            for j in range(h):
                degree = math.degrees(math.atan2(cy - j, cx - i))
                degree_mask[j, i] = (degree + 180) % 360
        for i in range(8):
            if i == 7:
                degree_mask[degree_mask == 0] = 360
            mask[i, (degree_mask >= i * 45) & (degree_mask <= (i + 1) * 45)] = 1
            if h % 2 != 0:
                mask[i, cy, cx] = 1
        # mask = torch.count_nonzero(mask.flatten(1))
        return mask

    def forward(self, x, s):
        # B, C, H, W = x.shape
        mask = rearrange(self.feat_mask, 'p h w -> 1 1 p h w')
        count = torch.count_nonzero(mask.flatten(-2), dim=-1)
        x    = x.unsqueeze(2)
        part = (x * mask).sum(-2).sum(-1)
        feat = part / count
        feat = rearrange(feat, 'b c (h w) -> b c h w', h=2, w=4)
        if s is None:
            return feat
        sal_feat = []
        for i in range(self.sal_mask.shape[0]):
            sal_feat.append(s[:, :, self.sal_mask[i] > 0].flatten(1))
        sal_feat = torch.stack(sal_feat, dim=2)
        sal_feat = rearrange(sal_feat, 'b c (h w) -> b c h w', h=2, w=4)
        return feat, sal_feat

class MultiRectanglePattern(nn.Module):
    def __init__(self):
        super(MultiRectanglePattern, self).__init__()
        feat_mask = self.generate_multi_direction_mask(7, 7)
        sal_mask = self.generate_multi_direction_mask(56, 56)
        self.register_buffer('feat_mask', feat_mask)
        self.register_buffer('sal_mask', sal_mask)

    def generate_multi_direction_mask(self, w, h):
        square_part = torch.zeros(4, 4, h, w)
        index_y = torch.split(torch.arange(h), (h + 1) // 4)
        index_x = torch.split(torch.arange(w), (w + 1) // 4)
        for i in range(4):
            for j in range(4):
                for x in index_x[i]:
                    for y in index_y[j]:
                        square_part[i, j, y, x] = 1
        mask = torch.zeros(8, h, w)
        group_x = [[0, 0, 1], [1], [2, 3, 3], [2], [0, 0, 1], [1], [2, 3, 3], [2]]
        group_y = [[1, 0, 0], [1], [0, 0, 1], [1], [2, 3, 3], [2], [3, 3, 2], [2]]
        for i in range(len(group_x)):
            mask[i] = torch.sum(square_part[group_x[i], group_y[i]], dim=0)
        mask = torch.clip(mask, min=0, max=1)
        return mask

    def forward(self, x, s):
        # B, C, H, W = x.shape
        mask = rearrange(self.feat_mask, 'p h w -> 1 1 p h w')
        count = torch.count_nonzero(mask.flatten(-2), dim=-1)
        x = x.unsqueeze(2)
        part = (x * mask).sum(-2).sum(-1)
        feat = part / count
        feat = rearrange(feat, 'b c (h w) -> b c h w', h=2, w=4)
        if s is None:
            return feat
        sal_count = torch.count_nonzero(self.sal_mask.flatten(1), dim=-1)
        target_size = sal_count.min().item()
        sal_feat = []
        for i in range(self.sal_mask.shape[0]):
            sal = s[:, :, self.sal_mask[i] > 0].flatten(1)
            if sal.shape[1] > target_size:
                sal = F.adaptive_max_pool1d(sal.unsqueeze(1), output_size=target_size).squeeze(1)
            sal_feat.append(sal)
        sal_feat = torch.stack(sal_feat, dim=2)
        sal_feat = rearrange(sal_feat, 'b c (h w) -> b c h w', h=2, w=4)
        return feat, sal_feat

class MPPModule(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5,
                 pattern_list=[1,2,3,4,5,6,7,8],
                 fusion='sum'):
        super(MPPModule, self).__init__()
        self.pattern_list = pattern_list
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.fusion = fusion
        pool_list = []
        conv_list = []
        print('Multi-Pattern Pooling pattern: {}, fusion manner: {}, dropout: {}'.\
              format(pattern_list, fusion, dropout))
        for pattern in pattern_list:
            p_fn = getattr(self, 'pattern{}'.format(int(pattern)))
            p, c = p_fn()
            pool_list.append(p)
            conv_list.append(c)
        self.pool_list = nn.ModuleList(pool_list)
        self.conv_list = nn.ModuleList(conv_list)

    def forward(self, x, weights):
        outputs = []
        for pool,conv in zip(self.pool_list, self.conv_list):
            feat = self.dropout(pool(x,s=None))
            feat = self.dropout(conv(feat))
            outputs.append(feat)

        if len(outputs) == 1:
            return outputs[0]
        if self.fusion == 'sum':
            outputs = torch.stack(outputs, dim=2)
            if weights is None:
                outputs = torch.sum(outputs, dim=2)
            else:
                weights = F.softmax(weights, dim=1)
                outputs = torch.sum(outputs * weights.unsqueeze(1), dim=2)
        elif self.fusion == 'mean':
            outputs = torch.stack(outputs, dim=2)
            outputs = torch.mean(outputs, dim=2)
        elif self.fusion == 'concat':
            outputs = torch.cat(outputs, dim=1)
        else:
            raise ValueError('Unkown fusion type {}'.format(self.fusion))
        return outputs

    def pattern0(self):
        pool = GlobalPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (1,1), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern1(self):
        pool = HorizontalPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (2,1), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern2(self):
        pool = VerticalPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (1,2), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern3(self):
        pool = TriangularPattern(flip=False)
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (2,1), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern4(self):
        pool = TriangularPattern(flip=True)
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (2, 1), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern5(self):
        pool = SurroundPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (2, 1), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern6(self):
        pool = QuarterPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (2, 2), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern7(self):
        pool = CrossPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (2,2), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern8(self):
        pool = ThirdOfRulePattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (3,3), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern9(self):
        pool = HorThirdPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (1,3), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern10(self):
        pool = VerticalThirdPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (3,1), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern11(self):
        pool = MultiDirectionPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (2,4), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern12(self):
        pool = MultiRectanglePattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (2,4), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv


class SAMPPModule(nn.Module):
    def __init__(self, in_dim, out_dim,
                 saliency_size, dropout=0.5,
                 pattern_list=[1,2,3,4,5,6,7,8],
                 fusion='sum'):
        super(SAMPPModule, self).__init__()
        self.pattern_list = pattern_list
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.fusion  = fusion
        self.saliency_size = saliency_size
        pool_list = []
        conv_list = []
        print('Saliency-aware Multi-Pattern Pooling pattern: {}, fusion manner: {}, dropout: {}'.\
              format(pattern_list, fusion, dropout))
        for pattern in pattern_list:
            p_fn = getattr(self, 'pattern{}'.format(int(pattern)))
            p,c  = p_fn()
            pool_list.append(p)
            conv_list.append(c)
        self.pool_list = nn.ModuleList(pool_list)
        self.conv_list = nn.ModuleList(conv_list)

    def forward(self, x, s, weights):
        outputs = []
        idx = 0
        for pool,conv in zip(self.pool_list, self.conv_list):
            feat, sal = pool(x,s)
            feat = self.dropout(feat)
            # print('pattern{}, x {}, s {}, feat_dim {}, sal_dim {}'.format(
            #     self.pattern_list[idx], x.shape, s.shape, feat.shape, sal.shape))
            fused = torch.cat([feat, sal], dim=1)
            fused = self.dropout(conv(fused))
            outputs.append(fused)
            idx += 1

        if len(outputs) == 1:
            return outputs[0]
        if self.fusion == 'sum':
            outputs = torch.stack(outputs, dim=2)
            if weights is None:
                outputs = torch.sum(outputs, dim=2)
            else:
                weights = F.softmax(weights, dim=1)
                outputs = torch.sum(outputs * weights.unsqueeze(1), dim=2)
        elif self.fusion == 'mean':
            outputs = torch.stack(outputs, dim=2)
            outputs = torch.mean(outputs, dim=2)
        elif self.fusion == 'concat':
            outputs = torch.cat(outputs, dim=1)
        else:
            raise ValueError('Unkown fusion type {}'.format(self.fusion))
        return outputs

    def pattern0(self):
        sal_length = (self.saliency_size // 2) ** 2
        pool = GlobalPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim + sal_length, self.out_dim, (1, 1), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv


    def pattern1(self):
        sal_length = (self.saliency_size // 2) ** 2
        pool = HorizontalPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim + sal_length, self.out_dim, (2,1), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern2(self):
        sal_length = (self.saliency_size // 2) ** 2
        pool = VerticalPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim + sal_length, self.out_dim, (1,2), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern3(self):
        s_size = int(self.saliency_size * 3 / 4)
        sal_length = s_size * (s_size - 1) // 2
        pool = TriangularPattern(flip=False, sal_size=self.saliency_size)
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim + sal_length, self.out_dim, (2,1), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern4(self):
        s_size = int(self.saliency_size * 3 / 4)
        sal_length = s_size * (s_size - 1) // 2
        pool = TriangularPattern(flip=True, sal_size=self.saliency_size)
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim + sal_length, self.out_dim, (2, 1), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern5(self):
        crop_size = 1./2
        sal_length = int(self.saliency_size * crop_size) ** 2
        pool = SurroundPattern(crop_size)
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim + sal_length, self.out_dim, (2, 1), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern6(self):
        sal_length = (self.saliency_size // 2) ** 2
        pool = QuarterPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim + sal_length, self.out_dim, (2, 2), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern7(self):
        sal_length = 0
        row_len = self.saliency_size
        while row_len > 0:
            sal_length += row_len
            row_len    -= 2
        pool = CrossPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim + sal_length, self.out_dim, (2,2), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern8(self):
        sal_length = (self.saliency_size // 3)**2
        pool = ThirdOfRulePattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim + sal_length, self.out_dim, (3,3), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern9(self):
        sal_length = (self.saliency_size // 3) * self.saliency_size
        pool = HorThirdPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim + sal_length, self.out_dim, (1,3), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern10(self):
        sal_length = (self.saliency_size // 3) * self.saliency_size
        pool = VerticalThirdPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim + sal_length, self.out_dim, (3,1), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern11(self):
        sal_length = int((self.saliency_size // 2) * (self.saliency_size // 2 + 1) / 2)
        pool = MultiDirectionPattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim + sal_length, self.out_dim, (2,4), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv

    def pattern12(self):
        sal_length = (self.saliency_size // 4) ** 2
        pool = MultiRectanglePattern()
        conv = nn.Sequential(
            nn.Conv2d(self.in_dim + sal_length, self.out_dim, (2,4), bias=False),
            nn.ReLU(True),
            nn.Flatten(1)
        )
        return pool, conv



if __name__ == '__main__':
    pattern_list = list(range(1,13))
    x = torch.randn(2, 512, 7, 7)
    s = torch.randn(2, 1, 56, 56)
    w = torch.randn(2, len(pattern_list))
    mpp = MPPModule(512, 512, 0.5, pattern_list)
    sampp = SAMPPModule(512, 512, 56, 0.5, pattern_list)
    mpp_out = mpp(x, weights=w)
    sa_out  = sampp(x,s, weights=w)
    print('mpp_out', mpp_out.shape, 'sampp_out', sa_out.shape)

    # third_pattern = ThirdOfRulePattern()
    # print(third_pattern(x,s)[0].shape)

    # h = w = 56
    # square_part = torch.zeros(4, 4, h, w)
    # index_y = torch.split(torch.arange(h), (h+1)//4)
    # index_x = torch.split(torch.arange(w), (w+1)//4)
    # for i in range(4):
    #     for j in range(4):
    #         for x in index_x[i]:
    #             for y in index_y[j]:
    #                 square_part[i,j,y,x] = 1
    # mask = torch.zeros(8, h, w)
    # group_x = [[0, 0, 1], [1], [2, 3, 3], [2], [0, 0, 1], [1], [2, 3, 3], [2]]
    # group_y = [[1, 0, 0], [1], [0, 0, 1], [1], [2, 3, 3], [2], [3, 3, 2], [2]]
    # for i in range(len(group_x)):
    #     mask[i] = torch.sum(square_part[group_x[i], group_y[i]], dim=0)
    #     print(mask[i], torch.count_nonzero(mask[i]))






