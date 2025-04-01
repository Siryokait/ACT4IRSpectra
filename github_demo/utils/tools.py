import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, label_norm=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if label_norm is None:
                self.save_checkpoint(val_loss, model, path)
            else:
                self.save_checkpoint(val_loss, model, path, label_norm)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if label_norm is None:
                self.save_checkpoint(val_loss, model, path)
            else:
                self.save_checkpoint(val_loss, model, path, label_norm)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, label_norm=None):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        if label_norm is not None:
            torch.save(label_norm.state_dict(), path + '/' + 'lb_checkpoint.pth')
        self.val_loss_min = val_loss


def visual(true, preds=None, folder='./vis_results/', file='test.pdf'):
    """
    Results visualization
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    name = folder + file

    plt.figure()
    lower_bond = np.min([np.min(true), np.min(preds)]).astype(np.int32)
    higher_bond = np.max([np.max(true), np.max(preds)]).astype(np.int32)
    x = [max(lower_bond * 0.8, lower_bond - 5), min(higher_bond * 1.2, higher_bond + 5)]
    y = [max(lower_bond * 0.8, lower_bond - 5), min(higher_bond * 1.2, higher_bond + 5)]
    plt.plot(x, y, c='k')
    plt.scatter(true, preds, s=5, c='r')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')

    plt.savefig(name, bbox_inches='tight')


def spec_visual(spec, wavelength, basel=None, folder='./vis_results/', file='spec.pdf'):
    name = folder + file
    if spec is None:
        if basel is not None:
            plt.figure()
            plt.plot(wavelength, basel, linewidth=0.5)
            plt.savefig(name, bbox_inches='tight')
            plt.close()
            return
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure()
    plt.plot(wavelength, spec, linewidth=0.5)
    if basel is not None:
        plt.plot(wavelength, basel, c='r')
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def corr_map_gen(specs, folder='./vis_results/', file='corr.pdf'):
    specs_df = pd.DataFrame(specs)
    corr_map = specs_df.corr(method='spearman')
    plt.figure()
    plt.imshow(corr_map, cmap=plt.cm.jet, filterrad=8, filternorm=3, interpolation='spline36')
    plt.colorbar()
    name = folder + file
    plt.savefig(name, dpi=300)
    return corr_map.values
