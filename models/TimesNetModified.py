import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

#1D version of Inception_Block_V1: same multi-kernel idea but with Conv1d.
#Used by the use_2d=0 ablation.
class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(InceptionBlock1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

#MODIFY 1: use a fixed period of 24 hours instead of finding the best period,
#24 hours is the result of period_study.py on the train set

FIXED_PERIOD = 24   #found by period_study.py on the train set 

def fixed_period():

    return FIXED_PERIOD

class TimesBlock(nn.Module):

    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        #MODIFY 2: choose the conv type from the command line
        self.use_inception = configs.use_inception
        #MODIFY 3: choose whether to use 2D conv or 1D conv from the command line
        self.use_2d = configs.use_2d

        #choose the conv type based on the command line arguments: 2D conv (InceptionBlock_V1) or 1D conv (InceptionBlock1D)
        if self.use_2d == 1:
            InceptionBlock = Inception_Block_V1   # 2D, from layers
            Conv = nn.Conv2d
        else:
            InceptionBlock = InceptionBlock1D     # 1D, defined in our class
            Conv = nn.Conv1d

        if self.use_inception == 1:
            self.conv = nn.Sequential(
                InceptionBlock(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
                nn.GELU(),
                InceptionBlock(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
            )
        else:
            self.conv = nn.Sequential(
                Conv(configs.d_model, configs.d_ff, kernel_size=3, padding=1),
                nn.GELU(),
                Conv(configs.d_ff, configs.d_model, kernel_size=3, padding=1),
            )

    def forward(self, x):

        B, _, N = x.size() 

        #1D path (use_2d=0). No reshape, no period: just a 1D conv
        #along time. This ablation tests whether the 2D reshaping matters.
        if self.use_2d == 0:
            #Conv1d wants [B, channels, length] = [B, N, T]; our x is [B, T, N]
            out = x.permute(0, 2, 1)
            out = self.conv(out)
            out = out.permute(0, 2, 1)
            return out + x
        
        period = fixed_period()

        #no padding needed: all our sequence lengths (192, 288, 432, 816) are multiples of the period (24)
        T = self.seq_len + self.pred_len
        out = x

        #reshape 1D -> 2D: rows = days, columns = hours of the day
        out = out.reshape(B, T // period, period, N).permute(0, 3, 1, 2).contiguous()

        #2D convolution on the (days x hours) image
        out = self.conv(out)

        #reshape back 2D -> 1D for the residual connection
        out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
        out = out[:, :(self.seq_len + self.pred_len), :]

        #residual connection
        out = out + x
        return out

class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        # forecasting only: this project targets long-term forecasting, so we
        # keep just the forecast head (imputation/anomaly/classification removed)
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # forecasting only (x_dec/x_mark_dec unused: we forecast by extending
        # the encoder sequence, as in TimesNet)
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
