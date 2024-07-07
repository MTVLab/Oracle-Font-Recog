'''
Description: the model of oracle-font-recog
Author: YuanJiang
Date: 2024-07-07 10:54:34
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from gcn_lib import Grapher, act_layer, GrapherTrans
from config import ModelConfig

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)

class Patch(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, height_img=64, in_dim=3, out_dim=512, act='relu'):
        super(Patch,self).__init__()
        self.path_local = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, height_img//4,stride=height_img//4),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 1, stride=1),
            nn.BatchNorm2d(out_dim)
        )
        self.path_global = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, height_img,stride=height_img),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 1, stride=1),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x, pos):
        pathched_local = self.path_local(x)
        pathched_global = self.path_global(x)
        B, C, _, _ = pathched_local.shape
        pathched_local = pathched_local.reshape(B, C, -1).transpose(2, 1)
        pathched_global = pathched_global.reshape(B, C, -1).transpose(2, 1)

        pos_local = pos.repeat(1, 4, 1).repeat(1, 4, 1)
        pathched_local = torch.cat((pathched_local, pos_local), dim=2)
        pathched_global = torch.cat((pathched_global, pos), dim=2)

        return pathched_local, pathched_global

class MutiHeadAttention(nn.Module):
    """
    muti-head Attention
    """
    def __init__(self, n_dim=512, n_head=8):
        super(MutiHeadAttention, self).__init__()

        self.n_dim = n_dim
        self.n_head = n_head
        self.w_q = nn.Linear(n_dim, n_dim)
        self.w_k = nn.Linear(n_dim, n_dim)
        self.w_v = nn.Linear(n_dim, n_dim)
        self.combine = nn.Linear(n_dim, n_dim)

    def attention(self, q, k, v, mask=None):
        d_k = self.n_dim // self.n_head
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        out = torch.matmul(scores, v)
        return out

    def forward(self, x, mask=None):
        batch, times, dimension = x.shape
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        q = q.view(batch, times, self.n_head, dimension // self.n_head).permute(0, 2, 1, 3)
        k = k.view(batch, times, self.n_head, dimension // self.n_head).permute(0, 2, 1, 3)
        v = v.view(batch, times, self.n_head, dimension // self.n_head).permute(0, 2, 1, 3)

        scores = self.attention(q, k, v, mask)
        concat = scores.transpose(1, 2).contiguous().view(batch, -1, dimension)
        out = self.combine(concat)
        return out

class TransGnnBlock(nn.Module):
    def __init__(self, in_channels=128,  kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
             bias=True,  stochastic=False, epsilon=0.0, r=1, drop_path=0.0):
        super(TransGnnBlock, self).__init__()
        self.branch_loacl = nn.Sequential(
            GrapherTrans(in_channels, kernel_size, dilation, conv, act, norm,
                    bias, stochastic, epsilon, r, drop_path),
            FFN(in_channels, in_channels * 4, act=act, drop_path=drop_path)
        )
        self.branch_global= nn.Sequential(
            GrapherTrans(in_channels, kernel_size, dilation, conv, act, norm,
                    bias, stochastic, epsilon, r, drop_path),
            FFN(in_channels, in_channels * 4, act=act, drop_path=drop_path)
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.attention_cross = MutiHeadAttention(in_channels, n_head=4)

    def forward(self, x_local, x_global):
        x_local = self.branch_loacl(x_local.transpose(2, 1).unsqueeze(-1)).squeeze(-1)
        x_global = self.branch_global(x_global.transpose(2,1).unsqueeze(-1)).squeeze(-1)
        x = torch.cat((x_local, x_global), dim=2)
        w = self.fc(self.avg_pool(x).squeeze(-1)).unsqueeze(-1)
        x = x * w.expand_as(x)
        out = self.attention_cross(x.transpose(2,1))
        return out

class Downsample(nn.Module):
    """ Convolution-based downsample
    """

    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class TransGnn(nn.Module):

    def __init__(self, options):
        super(TransGnn, self).__init__()

        self.patch_embed = nn.ModuleList()
        self.patch_embed.append(Patch(options.image_height, options.in_channel, options.channel_list[0], options.act))
        self.patch_embed.append(Patch(options.image_height//4, options.channel_list[1], options.channel_list[2], options.act))

        downsample = []
        downsample.append(Downsample(options.in_channel, options.channel_list[0]))
        downsample.append(Downsample(options.channel_list[0], options.channel_list[1]))
        self.downsample = nn.Sequential(*downsample)

        self.gnn_block = nn.ModuleList()
        self.gnn_block.append(TransGnnBlock(options.channel_list[0] + 4, 
                                            options.kernel_size, options.dilation, options.conv, options.act, options.norm, 
                                            options.bias,options.stochastic, options.epsilon, options.r, options.drop_path))
        self.gnn_block.append(TransGnnBlock(options.channel_list[2] + 4, 
                                            options.kernel_size, options.dilation, options.conv, options.act, options.norm, 
                                            options.bias, options.stochastic, options.epsilon, options.r, options.drop_path))

        self.gru = nn.GRUCell(options.channel_list[2] + 4, options.channel_list[2] + 4)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        out_dim = options.channel_list[2]*2 + options.channel_list[0] + 12
        self.classfier = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, options.num_classes)
        )
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x, pos):
        x_down = self.downsample(x)

        f_local, f_global = self.patch_embed[0](x, pos)
        feat_1 = self.gnn_block[0](f_local, f_global)

        f_local, f_global = self.patch_embed[1](x_down, pos)
        feat_2 = self.gnn_block[1](f_local, f_global)
        out_gnn = torch.cat((feat_1, feat_2), dim=2).transpose(2,1)

        out_gnn = self.avg_pool(out_gnn).squeeze(-1)
        out_gru = f_global[:,0,:]
        for i in range(1, f_global.shape[1]):
            out_gru = self.gru(f_global[:, i, :], out_gru)
        out = torch.cat((out_gnn, out_gru), dim=1)
        out = self.classfier(out)
        return out

def create_model():
    return TransGnn(options=ModelConfig())

if __name__ == '__main__':
    opt = ModelConfig()
    x = torch.randn(size=(2,3,64,4096))
    pos = torch.randn(size=(2,64,4))
    transGnn = TransGnn(opt)
    out = transGnn(x, pos)
    print(out)
    params = sum(p.numel() for p in transGnn.parameters() if p.requires_grad) / 1e6
    print('the total parameters is {}'.format(params))