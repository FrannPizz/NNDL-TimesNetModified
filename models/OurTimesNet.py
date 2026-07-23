"""
OurTimesNet: our own re-implementation of the TimesNet idea (Wu et al., ICLR 2023),
written from scratch for the NNDL project and parameterized so that every
architectural choice of the paper can be switched on/off from the command line.

This is the CENTRAL CONTRIBUTION of the project: the TSLib TimesNet
(models/TimesNet.py) is kept untouched and used only as the reference
implementation to reproduce the paper numbers; all the experiments/ablations
run on THIS model.

The macro-structure (embedding -> predict_linear -> stack of blocks ->
projection, with per-window normalization) intentionally mirrors the reference
so that the comparison "ours vs reference" isolates the pieces we change,
not the surrounding plumbing.

Command-line switches (added in run.py, group "OurTimesNet"):

  --period_mode {fft,fixed}
        fft   : discover the top-k dominant periods with the FFT, per batch,
                exactly like the paper (k = --top_k).
        fixed : impose a hand-chosen list of periods (--fixed_periods).
                IMPORTANT (no test leakage): the fixed periods must be justified
                a priori, i.e. from domain knowledge (hourly electricity ->
                daily cycle 24, weekly cycle 168) and/or from the periodogram
                computed ON THE TRAINING SPLIT ONLY (see
                analysis/train_periodogram.py). They must never be tuned by
                looking at test metrics.

  --fixed_periods "24" or "24,168" ...
        comma-separated periods (in time steps) used when --period_mode fixed.

  --block_type {inception,simple}
        inception : our multi-kernel 2D conv block (re-implementation of the
                    paper's Inception-style block, kernel sizes 1,3,...,2k-1
                    with k = --num_kernels).
        simple    : a single 3x3 conv2d. Ablation: "how much of TimesNet's
                    performance comes from the multi-scale Inception design?"

  --use_2d {1,0}
        1 : reshape the sequence into a (n_periods x period) 2D tensor and use
            2D convolutions (the core idea of TimesNet).
        0 : keep the sequence 1D and use the same conv block with 1D kernels.
            Ablation: "does the 2D reshaping itself add anything over plain
            1D convolutions with the same structure?" With use_2d=0 the
            period selection is irrelevant and is skipped entirely.

Only the long-term forecasting task is implemented (per the professor's
indication that a single well-developed task is sufficient).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
class OurEmbedding(nn.Module):
    """Our re-implementation of the value+time embedding used by TimesNet.

    - values  : Conv1d over time with kernel 3 and circular padding
                (a "token embedding": each time step is mixed with its
                neighbours before entering the network).
    - time    : linear projection of the calendar features produced by the
                'timeF' encoding (for freq='h' these are 4 features:
                hour-of-day, day-of-week, day-of-month, day-of-year).
    The two are summed and passed through dropout, as in the reference.
    """

    # number of 'timeF' calendar features per sampling frequency (same table
    # used by the library's TimeFeatureEmbedding)
    _TIME_FEATS = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}

    def __init__(self, c_in, d_model, freq, dropout):
        super().__init__()
        self.value_proj = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular', bias=False)
        # kaiming init as in the reference token embedding
        nn.init.kaiming_normal_(self.value_proj.weight,
                                mode='fan_in', nonlinearity='leaky_relu')
        n_time = self._TIME_FEATS.get(freq[-1].lower(), 4)
        self.time_proj = nn.Linear(n_time, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        # x: [B, T, C] values, x_mark: [B, T, n_time] calendar features
        out = self.value_proj(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T, d_model]
        if x_mark is not None:
            out = out + self.time_proj(x_mark)
        return self.dropout(out)


# ---------------------------------------------------------------------------
# Convolutional blocks (2D and 1D variants)
# ---------------------------------------------------------------------------
class MultiKernelConv2d(nn.Module):
    """Our Inception-style block: parallel 2D convolutions with kernel sizes
    1x1, 3x3, ..., (2k-1)x(2k-1); the outputs are averaged.

    Averaging parallel multi-scale branches is what lets the block capture
    both very local and longer-range 2D variations at the same cost of a
    single wide conv.
    """

    def __init__(self, in_ch, out_ch, num_kernels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        for m in self.branches:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.stack([b(x) for b in self.branches], dim=-1).mean(-1)


class MultiKernelConv1d(nn.Module):
    """1D counterpart of MultiKernelConv2d (kernels 1,3,...,2k-1 over time).

    Used only by the use_2d=0 ablation: same multi-scale structure, same
    channel widths, but no 2D reshaping - so the comparison with the 2D
    version isolates the contribution of the (period x n_periods) layout.
    Note: parameter count differs (k vs k*k weights per kernel), which we
    report in the ablation table.
    """

    def __init__(self, in_ch, out_ch, num_kernels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        for m in self.branches:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.stack([b(x) for b in self.branches], dim=-1).mean(-1)


def build_conv_block(block_type, dim, d_model, d_ff, num_kernels):
    """conv "sandwich" d_model -> d_ff -> d_model with GELU in between,
    as in the paper's parameter-efficient design.

    block_type: 'inception' (multi-kernel) or 'simple' (single 3x3 conv).
    dim: 2 for the standard 2D path, 1 for the use_2d=0 ablation.
    """
    if block_type == 'inception':
        Block = MultiKernelConv2d if dim == 2 else MultiKernelConv1d
        return nn.Sequential(
            Block(d_model, d_ff, num_kernels),
            nn.GELU(),
            Block(d_ff, d_model, num_kernels),
        )
    elif block_type == 'simple':
        Conv = nn.Conv2d if dim == 2 else nn.Conv1d
        return nn.Sequential(
            Conv(d_model, d_ff, kernel_size=3, padding=1),
            nn.GELU(),
            Conv(d_ff, d_model, kernel_size=3, padding=1),
        )
    raise ValueError(f'unknown block_type: {block_type}')


# ---------------------------------------------------------------------------
# Period selection
# ---------------------------------------------------------------------------
def periods_from_fft(x, k):
    """Paper-style period discovery: pick the k frequencies with the largest
    mean FFT amplitude (mean over batch and channels) and convert them to
    periods. Returns (periods [k], per-sample weights [B, k]).

    The .float() is needed because cuFFT under AMP (float16) only supports
    power-of-two lengths, and seq_len+pred_len generally is not one.
    """
    T = x.shape[1]
    xf = torch.fft.rfft(x.float(), dim=1)          # [B, F, C] complex
    amp = xf.abs().mean(-1)                         # [B, F] per-sample spectrum
    mean_amp = amp.mean(0)                          # [F]   batch-level spectrum
    mean_amp[0] = 0                                 # remove DC component
    _, top_idx = torch.topk(mean_amp, k)            # dominant frequency bins
    periods = (T // top_idx).detach().cpu().numpy() # frequency -> period
    return periods, amp[:, top_idx]                 # weights: per-sample amplitude


def periods_fixed(x, fixed_periods):
    """Fixed-period mode: the periods are imposed a priori (see the module
    docstring for the no-leakage justification protocol). The FFT is still
    used, but ONLY to compute the per-sample aggregation weights, i.e. how
    much each imposed period is expressed in the current window - no
    discovery happens here.
    """
    T = x.shape[1]
    xf = torch.fft.rfft(x.float(), dim=1)
    amp = xf.abs().mean(-1)                         # [B, F]
    idx = [max(1, T // p) for p in fixed_periods]   # bin of each imposed period
    return fixed_periods, amp[:, idx]               # ([n_p], [B, n_p])


# ---------------------------------------------------------------------------
# Core block
# ---------------------------------------------------------------------------
class OurTimesBlock(nn.Module):
    """One TimesNet block, re-implemented.

    2D path (use_2d=1):  for each selected period p
        1D sequence [B, T, d] -> pad to multiple of p -> reshape to
        [B, d, T/p, p] -> conv block -> back to [B, T, d];
      the per-period outputs are combined with softmax(FFT-amplitude) weights
      (the paper's "adaptive aggregation") + residual connection.

    1D path (use_2d=0): the same conv sandwich applied directly along time;
      no periods, no reshape, no aggregation - just conv + residual.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.period_mode = configs.period_mode
        self.use_2d = bool(configs.use_2d)
        # parse "--fixed_periods 24,168" once here
        self.fixed_periods = [int(p) for p in str(configs.fixed_periods).split(',')]
        self.conv = build_conv_block(configs.block_type,
                                     2 if self.use_2d else 1,
                                     configs.d_model, configs.d_ff,
                                     configs.num_kernels)

    def forward(self, x):
        B, T, N = x.size()

        # ---- 1D ablation path: no periodicity modelling at all -------------
        if not self.use_2d:
            out = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
            return out + x  # residual

        # ---- 2D path -------------------------------------------------------
        if self.period_mode == 'fixed':
            periods, weight = periods_fixed(x, self.fixed_periods)
        else:
            periods, weight = periods_from_fft(x, self.k)

        branch_outs = []
        for p in periods:
            p = int(p)
            if p >= T:
                # a period longer than the window carries no repetition to
                # exploit; fall back to the identity for this branch
                branch_outs.append(x)
                continue
            # zero-pad so that the length is divisible by the period
            if T % p != 0:
                pad_len = ((T // p) + 1) * p - T
                inp = torch.cat(
                    [x, torch.zeros(B, pad_len, N, device=x.device, dtype=x.dtype)],
                    dim=1)
            else:
                inp = x
            # 1D -> 2D: rows = one full period, columns = successive periods
            L = inp.shape[1]
            inp = inp.reshape(B, L // p, p, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(inp)                    # 2D conv on [B, d, L/p, p]
            # 2D -> 1D, drop the padding
            out = out.permute(0, 2, 3, 1).reshape(B, L, N)[:, :T, :]
            branch_outs.append(out)

        # adaptive aggregation: softmax over the FFT amplitudes of each period
        stacked = torch.stack(branch_outs, dim=-1)             # [B, T, d, n_p]
        w = F.softmax(weight, dim=1)                            # [B, n_p]
        w = w.unsqueeze(1).unsqueeze(1)                         # [B, 1, 1, n_p]
        res = (stacked * w).sum(-1)
        return res + x  # residual


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class Model(nn.Module):
    """OurTimesNet - long-term forecasting only.

    Pipeline (mirrors the reference for a fair comparison):
      1. per-window standardization ("non-stationary" normalization): each
         input window is normalized by its own mean/std, and the prediction
         is de-normalized at the end; this removes the level/scale of each
         window and is crucial on Electricity, where the 321 customers have
         very different magnitudes;
      2. embedding (values + calendar features);
      3. linear extension of the time axis from seq_len to seq_len+pred_len
         (the model then refines the whole extended sequence);
      4. e_layers x OurTimesBlock with LayerNorm;
      5. linear projection d_model -> c_out and de-normalization.
    """

    def __init__(self, configs):
        super().__init__()
        assert configs.task_name in ('long_term_forecast', 'short_term_forecast'), \
            'OurTimesNet only implements forecasting (single-task project)'
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.blocks = nn.ModuleList(
            [OurTimesBlock(configs) for _ in range(configs.e_layers)])
        self.embedding = OurEmbedding(configs.enc_in, configs.d_model,
                                      configs.freq, configs.dropout)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        # 1. per-window standardization (stats detached: they are treated as
        #    constants, the gradient does not flow through them)
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # 2. embedding  [B, seq_len, C] -> [B, seq_len, d_model]
        out = self.embedding(x, x_mark_enc)
        # 3. extend the time axis to seq_len + pred_len
        out = self.predict_linear(out.permute(0, 2, 1)).permute(0, 2, 1)
        # 4. TimesNet blocks
        for block in self.blocks:
            out = self.layer_norm(block(out))
        # 5. back to channel space
        out = self.projection(out)                  # [B, seq+pred, c_out]

        # de-normalization with the input-window statistics
        out = out * stdev + means
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_dec/x_mark_dec are required by the framework interface but unused:
        # like TimesNet, we forecast by extending the encoder sequence.
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]       # [B, pred_len, c_out]
