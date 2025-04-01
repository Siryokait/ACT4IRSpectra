from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, visual, spec_visual, corr_map_gen
from utils.metric import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from model.ACiT import ACiNet, CalibrationSpectra
from model.ACT_modules import RegLabelNorm

import os
import time

import warnings


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'ACi': ACiNet,
        }
        model = model_dict[self.args.model](self.args).float()
        return model

    def _get_data(self, flag='train'):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                      weight_decay=self.args.weight_decay)

    def _set_label_normalizer(self):
        self.label_norm = RegLabelNorm()
        self.norm_opt = optim.Adam(self.label_norm.parameters(), lr=self.args.norm_lr, weight_decay=0.01)
        self.norm_scheduler = lr_scheduler.ExponentialLR(self.norm_opt, gamma=0.99)

    def _select_lr_scheduler(self):
        self.model_scheduler = lr_scheduler.ExponentialLR(self.model_optim, gamma=self.args.lr_sch_gamma)
        # self.model_scheduler = lr_scheduler.CosineAnnealingLR(self.model_optim, T_max=self.ar gs.lr_sch_tmax)

    def _select_criterion(self):
        if self.args.loss == 'mse':
            self.criterion = nn.MSELoss()
        elif self.args.loss == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

    def _spectra_calibration(self):
        self.spec_calibrate = CalibrationSpectra()

    def vali(self, vali_data=None, vali_loader=None, criterion=None, epoch=None, select_cali=False, cali_seed=0):
        total_loss = []
        self.model.eval()
        if self.args.label_norm and self.args.task == 'regression':
            self.label_norm.eval()
        if self.args.model == 'ACi':
            if select_cali:
                spec_cali = self.spec_calibrate.chang_seed(cali_seed).to(self.device)
            else:
                spec_cali = self.spec_calibrate.get_cali_spec().to(self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, baselines) in enumerate(vali_loader):
                if self.args.baseline:
                    batch_x = batch_x - baselines
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                baselines = baselines.float().to(self.device)
                # print('-------------------')

                if self.args.model == 'ACi':
                    outputs = self.model(batch_x, baselines, spec_cali)
                else:
                    outputs = self.model(batch_x, baselines)
                if self.args.label_norm and self.args.task == 'regression':
                    outputs = self.label_norm.de_norm(outputs)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss.cpu().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting=None):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_count = 0
        ep_n = 0
        time_now = time.time()

        if self.args.model == 'ACi':
            corr_map = torch.Tensor(corr_map_gen(train_data.data_x)).to(self.device)
            self.model.load_corr_map(corr_map)
            self._spectra_calibration()
            special = False
            self.spec_calibrate.select_sepc({"x": train_data.data_x - train_data.baselines, "y": train_data.data_y},
                                            spec_n=self.args.aci_ref_n, task=self.args.task, special=special)
            spec_cali = self.spec_calibrate.get_cali_spec().to(self.device)
            self.model.load_data_scale(torch.Tensor(train_data.data_x - train_data.baselines))

        if self.args.label_norm and self.args.task == 'regression':
            self._set_label_normalizer()
            self.label_norm.fit(train_data.data_y)
            self.label_norm.to(self.device)

        if self.args.data == 'mango_dmc':
            self.args.baseline = False

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self._select_optimizer()
        self._select_lr_scheduler()
        self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            if self.args.label_norm and self.args.task == 'regression':
                self.label_norm.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, baselines) in enumerate(train_loader):
                iter_count += 1
                self.model_optim.zero_grad()
                if self.args.baseline:
                    batch_x = batch_x - baselines
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                baselines = baselines.float().to(self.device)

                if self.args.model == 'ACi':
                    outputs = self.model(batch_x, baselines, spec_cali)
                else:
                    outputs = self.model(batch_x, baselines)

                if self.args.label_norm and self.args.task == 'regression':
                    self.norm_opt.zero_grad()
                    outputs = self.label_norm.de_norm(outputs)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                            (self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.model_optim.step()
                self.model_optim.zero_grad()
                if self.args.label_norm and self.args.task == 'regression':
                    self.norm_opt.step()
                    self.norm_opt.zero_grad()
                train_loss.append(loss.item())

            t_ep_time = time.time() - epoch_time
            print("Epoch: {} cost time: {}".format(epoch + 1, t_ep_time))
            time_count += t_ep_time
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, self.criterion, epoch)
            test_loss = self.vali(test_data, test_loader, self.criterion, epoch)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            if self.args.label_norm and self.args.task == 'regression':
                early_stopping(vali_loss, self.model, path, self.label_norm)
            else:
                early_stopping(vali_loss, self.model, path)
            self.model_scheduler.step()
            if self.args.label_norm and self.args.task == 'regression':
                self.norm_scheduler.step()
            ep_n = epoch + 1
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if self.args.label_norm and self.args.task == 'regression':
            best_ln_path = path + '/' + 'lb_checkpoint.pth'
            self.label_norm.load_state_dict(torch.load(best_ln_path))
        print('avg epoch time(training): {}'.format(time_count/ep_n))
        return self.model

    def test(self, setting=None, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        tr_preds, tr_trues = [], []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        if self.args.label_norm and self.args.task == 'regression':
            self.label_norm.eval()
        if self.args.model == 'ACi':
            spec_cali = self.spec_calibrate.get_cali_spec().to(self.device)
        inference_time1 = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, baselines) in enumerate(test_loader):
                if self.args.baseline:
                    batch_x = batch_x - baselines
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                baselines = baselines.float().to(self.device)
                input_x = batch_x

                if self.args.model == 'ACi':
                    outputs = self.model(batch_x, baselines, spec_cali)
                else:
                    outputs = self.model(batch_x, baselines)
                if self.args.label_norm and self.args.task == 'regression':
                    outputs = self.label_norm.de_norm(outputs)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

            inference_time2 = time.time()

            for i, (batch_x, batch_y, baselines) in enumerate(train_loader):
                if self.args.baseline:
                    batch_x = batch_x - baselines
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                baselines = baselines.float().to(self.device)
                input_x = batch_x

                if self.args.model == 'ACi':
                    outputs = self.model(batch_x, baselines, spec_cali)
                else:
                    outputs = self.model(batch_x, baselines)
                if self.args.label_norm and self.args.task == 'regression':
                    outputs = self.label_norm.de_norm(outputs)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                tr_preds.append(pred)
                tr_trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        tr_trues = np.concatenate(tr_trues, axis=0)
        tr_preds = np.concatenate(tr_preds, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('inference time: {}'.format(inference_time2 - inference_time1))

        if self.args.task == 'regression':
            mae, mse, rmse, rmsep, mape, mspe, rse, corr, r2 = metric(preds, trues, self.args.task)
            file_name = setting
            if self.args.data in ['tablet', 'melamine']:
                file_name = file_name + self.args.train_d + self.args.target_d
            visual(trues, preds, file=file_name+'.pdf')
            visual(tr_trues, tr_preds, file=file_name + '_training.pdf')
            print('rmse:{}, rmse(%):{}, mae:{}, r2:{}'.format(rmse, rmsep, mae, r2))
            f = open(folder_path + "result.txt", 'a')
            f.write(setting + "  \n")
            f.write('rmse:{}, mae:{}, rse:{}, corr:{}, r2:{}'.format(rmse, mae, rse, corr, r2))
            f.write('\n')
            f.write('\n')
            f.close()
            np.save(folder_path + 'pred.npy', preds)
            return {'rmse': rmse, 'rmsep': rmsep, 'mae': mae, 'rse': rse, 'corr': corr, 'r2': r2}
        else:
            acc, auc, weighted_f1 = metric(preds, trues, self.args.task)
            acc1, auc1, weighted_f11 = metric(tr_preds, tr_trues, self.args.task)
            print('training acc:{}, auc:{}, weighted f1:{}'.format(acc1, auc1, weighted_f11))
            print('predicting acc:{}, auc:{}, weighted f1:{}'.format(acc, auc, weighted_f1))
            f = open(folder_path + "result.txt", 'a')
            f.write(setting + "  \n")
            f.write('acc:{}, auc:{}, weighted f1:{}'.format(acc, auc, weighted_f1))
            f.write('\n')
            f.write('\n')
            f.close()
            np.save(folder_path + 'pred.npy', preds)
            return {'acc': acc, 'auc': auc, 'weighted_f1': weighted_f1}
