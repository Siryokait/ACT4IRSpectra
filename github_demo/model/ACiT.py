import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.ACT_modules import SpectrA
from model.dnns import Tokenizer, SinCosPosEncoding
from sklearn.model_selection import StratifiedShuffleSplit


class ACiNet(nn.Module):
    def __init__(self, configs):
        super(ACiNet, self).__init__()
        self.token = Tokenizer(configs.aci_token, stride=configs.aci_stride)
        self.pe = SinCosPosEncoding()
        dp_rate = configs.aci_dp

        token_len = configs.aci_token * 2 + 1
        self.token_l = token_len
        band_num = int(np.ceil(configs.band_num / configs.aci_stride))
        self.stride = configs.aci_stride

        self.encoder1 = SpectrA(d_model=token_len, nhead=configs.aci_nhead, dropout=dp_rate,
                                dim_feedforward=configs.aci_ff, band_num=band_num)
        self.fc1 = nn.Linear(band_num, configs.aci_fc1)
        self.dp_out1 = nn.Dropout(p=dp_rate)
        self.fc2 = nn.Linear(configs.aci_fc1, configs.aci_fc2)
        self.dp_out2 = nn.Dropout(p=dp_rate)
        self.fc3 = nn.Linear(configs.aci_fc2, configs.y_dim)
        self.bn1 = nn.BatchNorm1d(configs.aci_fc1)
        self.bn2 = nn.BatchNorm1d(configs.aci_fc2)
        self.configs = configs

        self.alpha1 = nn.Parameter(torch.Tensor([0.8]))
        self.alpha2 = nn.Parameter(torch.Tensor([0.2]))
        self.beta1 = nn.Parameter(torch.ones([token_len, token_len]))
        self.token_v = nn.Linear(token_len, token_len)

        self.x_mean = None
        self.x_scale = None

    def load_corr_map(self, corr_map):
        corr_map = corr_map[::self.stride, ::self.stride]
        self.encoder1.get_corr_map(corr_map)

    def load_data_scale(self, x, whole=True):
        self.x_mean = torch.min(x[:, 20:-20])
        self.x_scale = torch.abs(torch.max(x[:, 20:-20] - self.x_mean))

    def forward(self, x, basel, cali_spec=None):
        x = (x - self.x_mean) / self.x_scale
        basel = basel[:, ::self.stride]

        tx = self.token.transform(x)

        pe = self.pe.get_pe(tx)
        tx = tx + pe
        if cali_spec is not None:
            cali_spec = (cali_spec - self.x_mean) / self.x_scale

            tx1 = self.token.transform(cali_spec)

            pe1 = self.pe.get_pe(tx1)
            cali_spec = tx1 + pe1
            att_f = self.encoder1(tx, cali_spec)
        else:
            att_f = self.encoder1(tx)
        att_f = torch.mul(self.token_v(att_f), F.softmax(torch.matmul(att_f, self.beta1), dim=-1))
        att_f = torch.sum(att_f.squeeze(), dim=-1)

        fea = self.alpha1 * att_f + self.alpha2 * basel

        fc1 = self.dp_out1(F.relu(self.fc1(fea)))

        fc1 = self.bn1(fc1)

        fc2 = self.dp_out2(F.relu(self.fc2(fc1)))

        res = self.fc3(self.bn2(fc2))

        if self.configs.task == 'classification':
            res = F.softmax(res, dim=1)
        else:
            res = F.tanh(res)

        return res


class CalibrationSpectra:
    def __init__(self):
        self.cali_spec = None
        self.x = None
        self.y = None
        self.task = None
        self.spec_n = None

    def get_cali_spec(self):
        return torch.Tensor(self.cali_spec)

    def load_cali_spec(self, cali_spec):
        self.cali_spec = cali_spec

    def chang_seed(self, random_seed):
        x, y, task, spec_n = self.x, self.y, self.task, self.spec_n
        sample_n = np.shape(x)[0]
        seed = random_seed
        if task == 'regression':
            idxes = np.argsort(y, axis=0)
            step = int(sample_n / spec_n)
            idx_mat = idxes[0:step * spec_n].reshape((spec_n, step))
            for i, idx_arr in enumerate(idx_mat):
                np.random.seed(seed)
                np.random.shuffle(idx_arr)
                seed += 1
            select_idxes = idx_mat[:, 0]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=spec_n, random_state=random_seed)
            select_idxes = sss.split(x, y)[0][1]
        self.cali_spec = x[select_idxes]
        return torch.Tensor(self.cali_spec)

    def select_sepc(self, data: dict, spec_n, task, random_seed=0, special=False):
        x = data['x']
        y = data['y']
        self.x = x
        self.y = y
        self.task = task
        self.spec_n = spec_n
        sample_n = np.shape(x)[0]
        if task == 'regression':
            idxes = np.argsort(y, axis=0)
            step = int(sample_n / spec_n)
            idx_mat = idxes[0:step * spec_n].reshape((spec_n, step))
            for i, idx_arr in enumerate(idx_mat):
                np.random.shuffle(idx_arr)
            select_idxes = idx_mat[:, 0]
        elif special:
            this_label = 0
            select_idxes = [0]
            for i, lab in enumerate(y):
                lab_a = np.argmax(lab)
                if lab_a != this_label:
                    this_label = lab_a
                    select_idxes.append(i)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=spec_n, random_state=random_seed)
            select_idxes = None
            for tr_idxes, te_idxes in sss.split(x, y):
                select_idxes = te_idxes
                break
        self.cali_spec = x[select_idxes]
