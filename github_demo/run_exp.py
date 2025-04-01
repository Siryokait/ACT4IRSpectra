import random
import torch
import numpy as np

import argparse
from exp.exp_dnn import Exp_Main

#  Demo submitted to AAAI 2025. Please do not distribute.
#  We prepare Tablet dataset in this demo for AAAI 2025.
#  If you want to try other datasets, please refer to the cited references in our paper to download those datasets.
#  If you use the Tablet dataset provided in this demo for research, please cite the original paper.
if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # ENSURE REPRODUCIBILITY

    parser = argparse.ArgumentParser(description='ACT')

    parser.add_argument('--model', type=str, default='ACi',
                        help='model name, options: [ACi]')
    # data loader
    parser.add_argument('--data', type=str, default='tablet',
                        help='options:[apple_leaf, mango_dmc, melamine, strawberry_puree, tablet]')
    parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
    #  ---- Melamine[R562, R568], tablet[1, 2] dataset
    parser.add_argument('--train_d', type=str, default='1', help='input domain')
    parser.add_argument('--target_d', type=str, default='2', help='target domain')
    #  ---- tablet dataset
    parser.add_argument('--target_n', type=int, default=2, help='prediction target')
    parser.add_argument('--aci_stride', type=int, default=1, help='down sampling')

    # preprocess
    parser.add_argument('--baseline', type=bool, default=True, help='baseline correction')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=5, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function [mse ce]')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    # ----lr_scheduler.ExponentialLR
    parser.add_argument('--lr_sch_gamma', type=float, default=0.99, help='gamma')

    # Label Norm
    parser.add_argument('--label_norm', type=bool, default=True, help='label normalizer')
    parser.add_argument('--norm_lr', type=float, default=0.01, help='label normalizer')
    # ----SpectrAttn
    parser.add_argument('--aci_token', type=int, default=12, help='token size')
    parser.add_argument('--aci_dp', type=float, default=0.1, help='drop out rate')
    parser.add_argument('--aci_nhead', type=int, default=5, help='num of attention heads')
    parser.add_argument('--aci_ff', type=int, default=512, help='dim of feed forward')
    parser.add_argument('--aci_use_SpectrA', type=bool, default=True, help='use SpectrA attention layer')
    parser.add_argument('--aci_ref_n', type=int, default=36, help='the number of reference spectra')
    parser.add_argument('--aci_fc1', type=int, default=256, help='feed forward network')
    parser.add_argument('--aci_fc2', type=int, default=64, help='feed forward network')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    args = parser.parse_args()
    task_dict = {"apple_leaf": "classification",
                 "mango_dmc": "regression", "melamine": "regression",
                 "strawberry_puree": "classification", "tablet": "regression"}
    band_n_dict = {"apple_leaf": 1500,
                   "mango_dmc": 242, "melamine": 346,
                   "strawberry_puree": 235, "tablet": 650}
    y_dim_dict = {"apple_leaf": 20,
                  "mango_dmc": 1, "melamine": 1,
                  "strawberry_puree": 2, "tablet": 1}
    filedir_dict = {"tablet": "tablet"}
    spectype_dict = {"apple_leaf": "abs",
                     "mango_dmc": "abs", "melamine": "abs",
                     "strawberry_puree": "abs", "tablet": "abs"}
    args.task = task_dict[args.data]
    if args.task == 'classification':
        args.loss = 'ce'
    if args.data == 'apple_leaf':
        args.aci_stride = 5
    args.band_num = band_n_dict[args.data]
    args.spec_type = spectype_dict[args.data]
    args.y_dim = y_dim_dict[args.data]
    args.root_path = args.root_path + filedir_dict[args.data]

    if args.task == 'classification':
        final_res = {'acc': [], 'auc': [], 'weighted_f1': []}
    else:
        final_res = {'rmse': [], 'rmsep': [], 'mae': [], 'rse': [], 'corr': [], 'r2': []}
    final_means = {}
    final_stds = {}

    for ii in range(args.itr):
        setting = '{}_{}_{}'.format(args.model, args.data, ii)
        exp = Exp_Main(args)

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        this_res = exp.test(setting)

        for key, value in this_res.items():
            final_res[key].append(value)

        torch.cuda.empty_cache()

    for key in final_res:
        final_means[key] = np.mean(final_res[key])
        final_stds[key] = np.std(final_res[key])

    print('mean: ', final_means)
    print('std: ', final_stds)

    setting_overall = '{}_{}'.format(args.model, args.data)
    f = open("overall_result.txt", 'a')
    f.write(setting_overall + "  \n")
    for key in final_means:
        f.write('{}: {} +- {}'.format(key, final_means[key], final_stds[key]))
        f.write(', ')
    f.write('\n')
    f.write('\n')
    f.close()
