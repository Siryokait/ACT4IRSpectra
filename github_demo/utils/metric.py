import numpy as np
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, f1_score


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def RMSEper(pred, true):
    return np.sqrt(MSE(pred, true)) / np.mean(true)


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def R2(pred, true):
    return 1 - MSE(pred, true) / np.mean((true - np.mean(true))**2)


def metric(pred, true, task):
    if task == 'regression':
        mae = MAE(pred, true)
        mse = MSE(pred, true)
        rmse = RMSE(pred, true)
        rmsep = RMSEper(pred, true)
        mape = MAPE(pred, true)
        mspe = MSPE(pred, true)
        rse = RSE(pred, true)
        corr = CORR(pred, true)
        r2score = r2_score(y_true=true, y_pred=pred)
        return mae, mse, rmse, rmsep, mape, mspe, rse, corr, r2score
    else:
        pred1 = np.argmax(pred, axis=1)
        true1 = np.argmax(true, axis=1)
        acc = accuracy_score(y_pred=pred1, y_true=true1)
        pred = pred * 0
        for i, idx in enumerate(pred1):
            pred[i, idx] = 1
        auc = roc_auc_score(y_score=pred, y_true=true, multi_class='ovo')
        weighted_f1 = f1_score(y_pred=pred1, y_true=true1, average='weighted')
        return acc, auc, weighted_f1
