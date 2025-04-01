import copy
import os

import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat


class Dataset_tablet(Dataset):
    def __init__(self, args, flag='train'):
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.train_d = args.train_d
        self.traget_d = args.target_d
        self.target_n = args.target_n
        self.spec_type = args.spec_type
        self.mean_x = 0

        self.root_path = args.root_path
        self.__read_data__()

    def __read_data__(self):
        index = ['calibrate_' + self.train_d, 'validate_' + self.train_d, 'test_' + self.traget_d][self.set_type]
        index2 = ['calibrate_Y', 'validate_Y', 'test_Y'][self.set_type]
        file_x = 'tablet.mat'
        x_path = os.path.join(self.root_path, file_x)
        mat_raw = loadmat(x_path)
        self.data_x = mat_raw[index]['data'][0][0]
        self.data_x = np.ascontiguousarray(self.data_x, dtype=np.float32)
        self.data_y = np.array(mat_raw[index2]['data'][0][0][:, self.target_n]).reshape((-1, 1))
        self.data_y = np.ascontiguousarray(self.data_y, dtype=np.float32)
        self.wavelength = mat_raw[index]['axisscale'][0][0][1][0][0]
        self.baselines = baseline_corr(self.data_x, mod=self.spec_type)

    def __onehot__(self, labels):
        label_max = np.max(labels)
        return np.eye(label_max)[labels - 1]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], self.baselines[index]

    def __len__(self):
        return len(self.data_x)


def baseline_corr(specs, threshold=0.01, mod='abs'):
    baseline = np.zeros(specs.shape)
    for i, spec in enumerate(specs):
        baseline[i] = _baseline(spec, threshold, mod)
    return baseline


def _baseline(spec, threshold, mod='abs'):
    x = np.arange(spec.shape[0])
    y = copy.deepcopy(spec)
    baseline = None
    max_time = 100
    for i in range(max_time):
        para_poly = np.polyfit(x, y, deg=1)
        formula = np.poly1d(para_poly)
        baseline = formula(x)
        if (np.max(baseline - spec) < threshold and mod == 'abs') or (
                np.max(spec - baseline) < threshold and mod == 'ref'):
            break
        if mod == 'abs':
            change_idxes = np.where(spec > baseline)
        else:
            change_idxes = np.where(spec < baseline)
        y[change_idxes] = baseline[change_idxes]
    return baseline - threshold
