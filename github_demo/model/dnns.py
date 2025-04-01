import torch
import torch.nn as nn


class Tokenizer:
    def __init__(self, fea_n, pad='zero', stride=1):
        self.fea_n = fea_n
        self.stride = stride
        if pad == 'zero':
            self.pad = nn.ConstantPad1d(fea_n, 0)
        elif pad == 'rep':
            self.pad = nn.ReplicationPad1d(fea_n)
        elif pad == 'ref':
            self.pad = nn.ReflectionPad1d(fea_n)
        else:
            self.pad = None
        assert self.pad is not None, "padding type error"

    def transform(self, x):
        x = x[:, ::self.stride]
        n_batch, n_spec = x.size()[0], x.size()[1]
        enc_res = torch.zeros([int(2 * self.fea_n + 1), n_batch, n_spec]).to(x.device)
        pad_x = self.pad(x)
        for i in range(int(2 * self.fea_n + 1)):
            enc_res[i] = pad_x[:, i:i + n_spec]
        enc_res = enc_res.permute(1, 2, 0)
        return enc_res


class SingleTokenizer:
    def __init__(self, fea_n, pad='zero', stride=1):
        self.fea_n = fea_n
        self.stride = stride

    def transform(self, x):
        x = x[:, ::self.stride]
        n_batch, n_spec = x.size()[0], x.size()[1]
        enc_res = torch.zeros([int(2 * self.fea_n + 1), n_batch, n_spec]).to(x.device)
        enc_res = enc_res + x
        enc_res = enc_res.permute(1, 2, 0)
        return enc_res


class SinCosPosEncoding:
    def __init__(self, b=10000):
        super(SinCosPosEncoding, self).__init__()
        self.b = b

    def get_pe(self, x):
        seq_len, d_model = x.size()[-2], x.size()[-1]
        pe = torch.zeros([seq_len, d_model]).to(x.device)
        i_list = torch.arange(d_model, dtype=torch.float).to(x.device)
        odd_i = 1 / (self.b ** ((i_list[1::2] - 1) / d_model))
        even_i = 1 / (self.b ** (i_list[0::2] / d_model))
        pos = torch.arange(seq_len, dtype=torch.float).reshape(-1, 1).to(x.device)

        pe[:, 0::2] = torch.sin(torch.matmul(pos, even_i.reshape(1, -1)))
        pe[:, 1::2] = torch.cos(torch.matmul(pos, odd_i.reshape(1, -1)))
        return pe
